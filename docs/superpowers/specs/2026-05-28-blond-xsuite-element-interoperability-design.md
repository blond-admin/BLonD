# BLonD 3 ↔ xsuite element interoperability — design

Date: 2026-05-28
Status: Approved (ready for implementation planning)
Location: `blond/interfaces/xsuite/elements/`

## Goal

Make BLonD 3 and xsuite interoperable at the **element/tracking** level, in both
directions, with a clean architecture (no monkey-mocking of private attributes):

1. **`WrapBlond4Xsuite`** — run an individual BLonD trackable (e.g. a
   `SingleHarmonicRFStation`) as an element inside an xsuite `Line` (xsuite owns
   the main tracking loop).
2. **`WrapXsuite4Blond`** — run an xsuite element/`Line` as an element inside a
   BLonD `Ring` (BLonD owns the main tracking loop).

A working `XsuiteRFBucketMatcher` (beam *preparation*) already exists under
`blond/interfaces/xsuite/beam_preparation/` and is out of scope here.

The previously-removed `BLonD3Cavity` / `DriftXsuite` code is explicitly **not**
a reference — it relied on mocking private `_ring`/`_magnetic_cycle` attributes,
which this design replaces.

## Guiding principles

- **Host owns the reference frame; the wrapper adapts the guest to it.**
  - xsuite host → xsuite is the source of truth for reference energy / beta /
    timing (confirmed requirement).
  - BLonD host → BLonD's `Simulation`/reference clock is the source of truth.
- **Single trackable is the core unit; sequences are sugar** built by chaining
  single wrappers.
- **One conversion source of truth.** All coordinate/state math lives in
  `helpers.py`; wrappers never duplicate it.
- **Reuse the existing `headless()` factory** for building standalone BLonD
  elements rather than constructing a parallel `Simulation`.
- **Acceleration / energy ramps are in scope** — the reference energy can change
  turn-to-turn and the embedded BLonD element must follow it without any Mock
  poking.

## Module layout

```
blond/interfaces/xsuite/elements/
  helpers.py              # pure conversion + particle-state mapping (no framework objects)
  wrap_blond_elelemt.py   # WrapBlond4Xsuite   (BLonD trackable → xsuite element)
  wrap_xsuite_elelemt.py  # WrapXsuite4Blond   (xsuite element/Line → BLonD element)
```

Dependency direction: **wrappers → helpers**. `helpers` imports neither wrapper
and is agnostic to which framework is the host.

## Coordinate conventions

| Quantity | BLonD | xsuite |
| --- | --- | --- |
| longitudinal position | `dt` [s] (time vs. synchronous particle) | `zeta` [m] |
| energy/momentum dev. | `dE` [eV] | `ptau` |

Transforms (c = speed of light):

```
dt   = -zeta / (beta0 * c) + dt_shift
zeta = -(dt - dt_shift) * beta0 * c
dE   =  ptau * beta0 * energy0
ptau =  dE   / (beta0 * energy0)
```

where `dt_shift = phi_s / omega_rf` aligns `zeta = 0` with the synchronous
particle's `dt`.

## Component 1 — `helpers.py` (shared conversion core)

An immutable value object carries the per-turn frame; pure functions do the math.

```python
@dataclass(frozen=True)
class ReferenceFrame:
    beta0: float        # reference relativistic beta
    energy0: float      # reference total energy [eV]
    omega_rf: float     # main-harmonic design RF frequency [rad/s]
    dt_shift: float     # phi_s / omega_rf

def zeta_to_dt(zeta, frame) -> dt
def dt_to_zeta(dt,  frame) -> zeta
def ptau_to_dE(ptau, frame) -> dE
def dE_to_ptau(dE,  frame) -> ptau
```

Beam-level converters that also handle particle identity and losses:

- `particles_to_beam(particles, beam, frame)` — convert the active particles
  (`state > 0`) into the BLonD beam's `dt`/`dE` arrays and set BLonD loss flags
  for the rest. Stores the active mask for symmetric write-back.
- `beam_to_particles(beam, particles, frame)` — write updated `dt`/`dE` back into
  exactly the same active slots of the xsuite `Particles`.

The empty stub functions in the current file map onto this API:
`to_dt → zeta_to_dt`, `to_zeta → dt_to_zeta`, `to_dE → ptau_to_dE`,
`to_ptau → dE_to_ptau`, `beam_xsuite_to_blond → particles_to_beam`,
`beam_blond_to_xsuite → beam_to_particles`.

This layer is fully unit-testable with no xsuite/`Simulation` dependency.

## Component 2 — `WrapBlond4Xsuite` (BLonD element inside an xsuite Line)

Implements the xsuite element interface (`track(particles)`), delegates physics
to a **headless-built** BLonD trackable, delegates conversion to `helpers`.

```python
class WrapBlond4Xsuite:
    def __init__(self, element, *, momentum_compaction_factor=None):
        self._element = element     # headless-built BLonD trackable
        self._beam = ...            # reusable BLonD Beam buffer (sized to the line)
        self._cycle = ...           # ExternalReferenceCycle held by the wrapper

    def track(self, particles):                 # xsuite calls this each turn
        frame = self._frame_from(particles)      # xsuite = source of truth
        self._cycle.set_total_energy(frame.energy0)   # clean energy seam
        helpers.particles_to_beam(particles, self._beam, frame)
        self._element.track(self._beam)
        helpers.beam_to_particles(self._beam, particles, frame)
```

### Energy seam (the one core change)

The cavity's `headless()` currently hardwires a `Mock(ConstantMagneticCycle)`
with a constant `get_target_total_energy.return_value`. To support ramps without
poking a Mock, introduce a small **real** cycle in core:

```python
class ExternalReferenceCycle(MagneticCycleBase):
    """Reference total energy supplied externally each turn (here: by xsuite)."""
    def __init__(self, reference_particle, total_energy_init): ...
    def set_total_energy(self, e): self._total_energy = e
    def get_target_total_energy(self, turn_i, section_i, reference_time, particle_type):
        return self._total_energy
    @staticmethod
    def headless(...): ...
```

and add an **optional** `magnetic_cycle=` kwarg to the cavity's `headless(...)`
(default preserves today's constant behavior — nothing breaks). The wrapper owns
the `ExternalReferenceCycle` and calls `set_total_energy(particles.energy0)` each
turn. The above/below-transition (η-sign) follows from
`momentum_compaction_factor` (user-supplied, or from `line.twiss4d()` when there
is no energy program).

**Net core change:** one new small class + one optional kwarg on `headless`. No
edits to any element `_track`.

## Component 3 — `WrapXsuite4Blond` (xsuite element/Line inside a BLonD Ring)

A normal BLonD physics element (`UserDefinedElement` / `BeamPhysicsRelevant`).
BLonD's `Simulation` already owns the frame, so this is the simpler direction.

```python
class WrapXsuite4Blond(UserDefinedElement):
    def __init__(self, xsuite_element_or_line): ...
    def on_init_simulation(self, sim):    # cache circumference; prepare frame source
        ...
    def _track(self, beam):
        frame = self._frame_from(beam.reference)     # BLonD = source of truth
        particles = self._as_particles(beam, frame)  # reuse buffer
        self._xs.track(particles)                    # element or full line
        helpers.beam_to_particles(... beam from particles ...)
```

`particle_ref` is fed from `beam.reference` each turn.

## Component 4 — Sequence sugar

- `WrapBlond4Xsuite.from_sequence([...])` and `WrapXsuite4Blond.from_line(line)`
  build one wrapper per guest element and expose them as a list to insert into
  the host sequence. Convenience only; no new conversion logic.

## Cross-cutting concerns

- **Losses:** xsuite `state > 0` ↔ BLonD loss flags handled once inside
  `helpers.particles_to_beam` / `beam_to_particles`, using a stored active mask
  so write-back targets the same slots. Single tested code path.
- **Performance:** the BLonD `Beam` and any temporary `Particles` are allocated
  once and reused; conversions write in place.

## Testing strategy

1. `helpers` — round-trip `x → blond → x`, plus loss-mask handling. Pure unit
   tests, no xsuite.
2. `WrapBlond4Xsuite` — one RF cavity in a trivial xsuite line:
   - flat energy → matches a pure-BLonD one-turn kick;
   - ramp → embedded reference energy tracks `particles.energy0`.
3. `WrapXsuite4Blond` — an xsuite drift vs. `DriftSimple` over N turns.
4. `ExternalReferenceCycle` — unit test in isolation.

## Required core change (summary)

- New `ExternalReferenceCycle(MagneticCycleBase)` in `blond/cycles/magnetic_cycle.py`.
- Optional `magnetic_cycle=` kwarg on `SingleHarmonicRFStation.headless(...)`
  (default unchanged). This enables headless elements to follow a turn-by-turn
  reference energy with zero mocking on the energy path.

Everything else is contained within `blond/interfaces/xsuite/elements/`.

## Out of scope

- `XsuiteRFBucketMatcher` / beam preparation (already implemented).
- Transverse-coupled co-simulation beyond passing particles through xsuite
  transport elements.
- Collective/intensity effects bridging between the two frameworks.
