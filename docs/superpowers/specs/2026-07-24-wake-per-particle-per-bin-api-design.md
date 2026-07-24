# Design: `get_wake_per_particle` / `get_wake_per_bin` — two exclusive wake entry points

Date: 2026-07-24
Status: Approved (pending spec review)

## Motivation

Time-domain wake sources currently expose the wake through an inconsistent,
duplicated set of methods:

- `get_wake` (point-sampled wake) and `get_wake_binned` (bin-averaged wake),
- parallel `get_wake_counter_rotation` / `get_wake_counter_rotation_binned`,
- near-identical `get_impedance_from_wake` / `get_impedance_from_wake_counter_rotation`
  bodies in every source (the code carries a literal `# Fixme all
  get_impedance_from_wake same`).

The names do not communicate the one distinction that actually matters: the
**beam model**. A point charge sees the wake Green's function `W(t)`; a
histogram bin (piecewise-constant charge) sees the wake **averaged over the
bin**. Point-sampling the wake for a histogram beam is the low-Q / broadband
resonator bug fixed earlier in this branch. This refactor makes that
distinction the explicit, primary API.

This is a **pure refactor**: no numerical behaviour changes. Every existing
assertion — including the 1e-9 pole-vs-convolution cross-check — must still
pass unchanged.

## Two exclusive wake entry points

Every wake-kernel source exposes exactly two wake entry points, each taking a
`counter_rotating` flag:

- **`get_wake_per_particle(time, counter_rotating=False)`** — the point-charge
  wake (Green's function) sampled at `time`. This is today's `get_wake`
  (`counter_rotating=False`) / `get_wake_counter_rotation` (`True`).
- **`get_wake_per_bin(time, counter_rotating=False)`** — the histogram-bin
  wake, i.e. the per-particle wake averaged over each centred bin. This is
  today's `get_wake_binned` / `get_wake_counter_rotation_binned`.

`get_impedance_from_wake` is **derived** (the rfft of the per-bin wake), not a
third wake concept.

## Interface (base `TimeDomain`)

```python
class TimeDomain(ABC):
    def get_wake_per_particle(self, time, counter_rotating=False):
        """Point-charge wake (Green's function) sampled at `time`, in [V].
        Kernel sources override. Sources that define their impedance another
        way (e.g. InductiveImpedance) do not implement this and instead
        override get_impedance_from_wake."""
        raise NotImplementedError

    def get_wake_per_bin(self, time, counter_rotating=False):
        """Histogram-bin wake = per-particle wake averaged over each centred
        bin. Default: exact bin-average of the piecewise-linear per-particle
        wake, i.e. the parameter-free stencil (w[n-1] + 6 w[n] + w[n+1]) / 8
        on the interior (edges extrapolate the boundary value). Exact for a
        tabulated wake; analytic sources override with a closed form."""
        w = self.get_wake_per_particle(time, counter_rotating)
        return _binned_stencil(w)

    def get_impedance_from_wake(self, time, simulation, beam, n_fft,
                                counter_rotating=False):
        """Derived: rfft(get_wake_per_bin(...)), cached per
        (hash(time), counter_rotating)."""
```

- `counter_rotating=True` on a source without a counter-rotating wake raises
  `RuntimeError` (as `get_wake_counter_rotation` does today).
- `TimeDomainCounterRotation` is **deleted**; its four abstract/concrete
  methods fold into the flag. (Verified: it has zero `isinstance` users.)
- The generic stencil default lives once, on the base.

## Per-source migration

Each source declares only its point-charge wake and, where a closed form
exists, its bin-average; impedance derivation is inherited.

| Source | `get_wake_per_particle` | `get_wake_per_bin` | `get_impedance_from_wake` |
|---|---|---|---|
| `Resonators` | closed-form `W(t)`; flag selects co-/counter-rotating shunt impedances | **override**: exact closed-form bin-average (`_wake_bin_average`); flag selects shunt | inherited default |
| `ImpedanceTableTime` | `interp(time, wake_x, wake_y)`; `counter_rotating=True` raises | inherited stencil (exact for a piecewise-linear table) | inherited default |
| `TravelingWaveCavity` | `wake_calc(time)`; `counter_rotating=True` raises | inherited stencil | inherited default |
| `InductiveImpedance` | not implemented (no wake kernel) | not implemented | keeps its own override |

Collapses:
- `get_wake` + `get_wake_counter_rotation` → one flagged `get_wake_per_particle`.
- `get_wake_binned` + `get_wake_counter_rotation_binned` → one flagged
  `get_wake_per_bin`.
- per-source `get_impedance_from_wake` + `..._counter_rotation` → the inherited
  base default.

`Resonators._wake_bin_average(time, shunt_impedances)` remains the private
closed-form helper behind its `get_wake_per_bin`.

### Caching / one loose end

The base `get_impedance_from_wake` default owns the impedance cache, keyed by
`(hash(time), counter_rotating)`. `Resonators.get_impedance_from_wake_freq`
currently reads `len(self._cache_impedance_from_wake)` to build its frequency
axis; it moves to reading the length from that shared cache (or recomputing it
from the last `time`/`n_fft`). No behaviour change.

## Solver call sites (no logic change)

- Convolution solvers (`SingleTurnResonatorConvolutionSolver`,
  `MultiPassResonatorSolver`, `ContinuousMultiTurnTimeDomainSolver`):
  `get_wake_binned` / `..._counter_rotation_binned` →
  `get_wake_per_bin(time, counter_rotating=…)`. `MultiPass`'s co/counter branch
  becomes the flag. The `ContinuousMultiTurn` duck-type check targets
  `get_wake_per_bin`.
- `TimeDomainFftSolver`: `source.get_impedance_from_wake(..., counter_rotating=False)`.
- `MultiPoleSparseSolve`: **untouched** — it consumes `get_vectorfit()`, not a
  wake kernel; its residue g-scaling and causal self-bin correction stay
  exactly as they are (out of scope, per decision).

## Testing

- Rename references to the renamed methods in the unit tests.
- Because the refactor is numerically inert, **all current assertions must
  still pass unchanged**, including:
  - `get_wake_per_bin` exact-bin-average tests,
  - the low-Q time-vs-freq convergence test,
  - the pole-vs-convolution CR cross-check at `rtol=1e-9`.
- Add/keep a test that `counter_rotating=True` raises on a non-CR source
  (table/TWC).

## Scope guard (YAGNI)

No new features. No pole-solver reframing. No changes to `InductiveImpedance`'s
math. No unrelated refactoring. Just the rename/merge and the derived-impedance
default described above.

## Non-goals / future

- Expressing the pole solver's per-bin correctness through the same abstraction
  (deferred by decision).
- A `counter_rotating` capability marker/property (the flag simply raises when
  unsupported; add a marker only if a future caller needs to query it).
