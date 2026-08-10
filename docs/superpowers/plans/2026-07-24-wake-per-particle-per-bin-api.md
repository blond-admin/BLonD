# `get_wake_per_particle` / `get_wake_per_bin` Wake API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad-hoc `get_wake` / `get_wake_binned` / `*_counter_rotation` method zoo on time-domain wake sources with two exclusive, beam-model-named entry points — `get_wake_per_particle` and `get_wake_per_bin` — plus a single derived `get_impedance_from_wake` default.

**Architecture:** A point charge sees the wake Green's function; a histogram bin sees the bin-averaged wake. Sources declare only their per-particle wake; `get_wake_per_bin` defaults to the exact stencil bin-average and is overridden by analytic sources; `get_impedance_from_wake` is derived (rfft of per-bin). Counter-rotation becomes a `counter_rotating` flag. `MultiPoleSparseSolve` (pole-residue) is out of scope.

**Tech Stack:** Python, NumPy/CuPy via the `blond` backend, pytest.

## Global Constraints

- Pure refactor: **no numerical behaviour change**. The existing impedance + physics suites must stay green throughout, including the pole-vs-convolution cross-check at `rtol=1e-9` and the low-Q time-vs-freq convergence test.
- Run tests with `.venv/bin/python -m pytest`.
- Backend in this environment is `Numpy64Bit` (float64); the `test_get_impedance_from_wake` float32 pin stays skipped.
- Commit messages end with the two trailer lines used elsewhere on this branch.
- Do **not** touch `MultiPoleSparseSolve`, `InductiveImpedance`'s math, or the pole self-bin correction.

---

### Task 1: Per-particle wake entry point (fold in the CR flag)

Introduce `get_wake_per_particle(time, counter_rotating=False)` and retire `get_wake` / `get_wake_counter_rotation`. This is the point-charge layer; its only in-source caller is the `get_wake_per_bin` stencil default (renamed in Task 2, so for now update the current `get_wake_binned` default's internal call).

**Files:**
- Modify: `blond/physics/impedances/base.py` — `TimeDomain.get_wake_binned` internal call (`base.py:158`); `TimeDomainCounterRotation.get_wake` abstract decl (`base.py:158` region under class at `:193`).
- Modify: `blond/physics/impedances/sources.py` — `Resonators.get_wake` (`:669`), `Resonators.get_wake_counter_rotation` (`:736`), `ImpedanceTableTime.get_wake` (`:1140`), `TravelingWaveCavity.get_wake` (`:1318`).
- Test: `tests/unittests/physics/impedances/test_sources.py`.

**Interfaces:**
- Produces: `get_wake_per_particle(self, time, counter_rotating: bool = False) -> ndarray` on `Resonators`, `ImpedanceTableTime`, `TravelingWaveCavity`; and a base `TimeDomain.get_wake_per_particle` that raises `NotImplementedError`. `counter_rotating=True` selects the counter-rotating shunt impedances on `Resonators` and raises `RuntimeError` on the table / TWC.

- [ ] **Step 1: Write the failing tests**

Add to `test_sources.py` (`TestResonators` and `TestImpedanceTableTime`):

```python
def test_get_wake_per_particle_counter_rotating_selects_cr_shunt(self):
    res = Resonators(
        shunt_impedances=np.array([1.0]),
        center_frequencies=np.array([1e9]),
        quality_factors=np.array([5.0]),
        shunt_impedances_counter_rotating=np.array([-1.0]),
    )
    time = backend.array(np.linspace(0, 5e-9, 64))
    co = copy_to_cpu(res.get_wake_per_particle(time, counter_rotating=False))
    cr = copy_to_cpu(res.get_wake_per_particle(time, counter_rotating=True))
    np.testing.assert_allclose(co, -cr)

def test_get_wake_per_particle_cr_raises_without_cr_shunt(self):
    res = Resonators(np.array([1.0]), np.array([1e9]), np.array([5.0]))
    with self.assertRaises(RuntimeError):
        res.get_wake_per_particle(backend.array(np.linspace(0, 1e-9, 8)),
                                  counter_rotating=True)
```

(In `TestImpedanceTableTime`, add the raise case for a table with `counter_rotating=True`.)

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/unittests/physics/impedances/test_sources.py -k "per_particle" -q`
Expected: FAIL (`AttributeError: ... has no attribute 'get_wake_per_particle'`).

- [ ] **Step 3: Add the base entry point**

In `base.py`, inside `class TimeDomain`, add (above `get_wake_binned`):

```python
def get_wake_per_particle(
    self, time: NumpyArray | CupyArray, counter_rotating: bool = False
) -> NumpyArray | CupyArray:
    """Point-charge wake (Green's function) sampled at ``time``, in [V].

    Kernel sources override this. Sources that define their impedance
    another way (e.g. InductiveImpedance) do not implement it and instead
    override get_impedance_from_wake.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not provide a point-charge wake."
    )
```

- [ ] **Step 4: Rename the source implementations**

In `sources.py`:
- `Resonators.get_wake` (`:669`) → `get_wake_per_particle(self, time, counter_rotating=False)`. Its body selects the shunt array: `shunt = self._shunt_impedances_counter_rotating if counter_rotating else self._shunt_impedances`; if `counter_rotating` and that is `None`, raise `RuntimeError` (reuse the message from the old `get_wake_counter_rotation`). Delete `Resonators.get_wake_counter_rotation` (`:736`) — its logic is now the `counter_rotating=True` branch.
- `ImpedanceTableTime.get_wake` (`:1140`) → `get_wake_per_particle(self, time, counter_rotating=False)`; add `if counter_rotating: raise RuntimeError("ImpedanceTableTime has no counter-rotating wake.")` before the interp.
- `TravelingWaveCavity.get_wake` (`:1318`) → `get_wake_per_particle(self, time, counter_rotating=False)`; add the same guard, then `return self.wake_calc(time=time)`.
- In `base.py` `get_wake_binned` default (`:158`), change `self.get_wake(time)` → `self.get_wake_per_particle(time, counter_rotating)` (the `counter_rotating` param is added in Task 2; for now keep it `self.get_wake_per_particle(time)`).

- [ ] **Step 5: Repoint the counter-rotating binned path (temporary)**

`Resonators.get_wake_counter_rotation_binned` (`:569`) and `_wake_bin_average` currently take a shunt array; leave them, but where `get_wake_counter_rotation` was called internally, call `get_wake_per_particle(time, counter_rotating=True)`. Grep `get_wake_counter_rotation(` and `.get_wake(` under `blond/` and fix each remaining caller (there is one: the base stencil).

- [ ] **Step 6: Run the focused + full source tests**

Run: `.venv/bin/python -m pytest tests/unittests/physics/impedances/test_sources.py -q`
Expected: the two new tests PASS; update any test still calling `.get_wake(`/`.get_wake_counter_rotation(` to `.get_wake_per_particle(..., counter_rotating=...)` until green.

- [ ] **Step 7: Run the whole impedance suite**

Run: `.venv/bin/python -m pytest tests/unittests/physics/impedances/ -q`
Expected: PASS (fix any remaining `get_wake`/`get_wake_counter_rotation` references in solvers/tests — solvers use the binned path, so few if any).

- [ ] **Step 8: Commit**

```bash
git add blond/physics/impedances/base.py blond/physics/impedances/sources.py tests/unittests/physics/impedances/test_sources.py
git commit -m "Introduce get_wake_per_particle with counter_rotating flag"
```

---

### Task 2: Per-bin wake entry point (fold in the CR flag)

Rename `get_wake_binned` → `get_wake_per_bin(time, counter_rotating=False)` and retire `get_wake_counter_rotation_binned`; thread the flag through the stencil default and the `Resonators` closed-form override; repoint the convolution solvers.

**Files:**
- Modify: `blond/physics/impedances/base.py` — `TimeDomain.get_wake_binned` (`:124`); `TimeDomainCounterRotation.get_wake_counter_rotation_binned` (`:231`).
- Modify: `blond/physics/impedances/sources.py` — `Resonators.get_wake_binned` (`:547`), `Resonators.get_wake_counter_rotation_binned` (`:569`).
- Modify: `blond/physics/impedances/solvers.py` — `SingleTurnResonatorConvolutionSolver` (`:667` region), `ContinuousMultiTurnTimeDomainSolver._update_wake_kernel` (`get_wake_binned` call) and `_check_source_ducktypes`, `MultiPassResonatorSolver` co/counter branch.
- Test: `tests/unittests/physics/impedances/test_sources.py`, `test_solvers.py`.

**Interfaces:**
- Consumes: `get_wake_per_particle(time, counter_rotating)` (Task 1).
- Produces: `get_wake_per_bin(self, time, counter_rotating: bool = False) -> ndarray`. Base default = stencil bin-average of `get_wake_per_particle(time, counter_rotating)`. `Resonators` overrides with the exact closed form (co/counter shunt by flag). Retired: `get_wake_binned`, `get_wake_counter_rotation_binned`.

- [ ] **Step 1: Write the failing test**

```python
def test_get_wake_per_bin_counter_rotating_matches_negated(self):
    res = Resonators(np.array([1.0]), np.array([1e9]), np.array([5.0]),
                     shunt_impedances_counter_rotating=np.array([-1.0]))
    time = backend.array(np.arange(64) * 0.05e-9)
    co = copy_to_cpu(res.get_wake_per_bin(time, counter_rotating=False))
    cr = copy_to_cpu(res.get_wake_per_bin(time, counter_rotating=True))
    np.testing.assert_allclose(co, -cr, atol=1e-12 * np.max(np.abs(co)))
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/unittests/physics/impedances/test_sources.py -k per_bin -q`
Expected: FAIL (`no attribute 'get_wake_per_bin'`).

- [ ] **Step 3: Rename the base default**

In `base.py`, rename `get_wake_binned` → `get_wake_per_bin(self, time, counter_rotating=False)`; body:

```python
w = self.get_wake_per_particle(time, counter_rotating)
wake_prev = backend.concatenate((w[:1], w[:-1]))
wake_next = backend.concatenate((w[1:], w[-1:]))
return (wake_prev + 6.0 * w + wake_next) / 8.0
```

Delete `TimeDomainCounterRotation.get_wake_counter_rotation_binned` (`:231`).

- [ ] **Step 4: Rename the `Resonators` override**

`Resonators.get_wake_binned` (`:547`) → `get_wake_per_bin(self, time, counter_rotating=False)`; body:

```python
shunt = (
    self._shunt_impedances_counter_rotating
    if counter_rotating
    else self._shunt_impedances
)
if counter_rotating and shunt is None:
    raise RuntimeError(
        "_shunt_impedances_counter_rotating needs to be set before"
        " calling this function."
    )
return self._wake_bin_average(time, shunt)
```

Delete `Resonators.get_wake_counter_rotation_binned` (`:569`).

- [ ] **Step 5: Repoint the solvers**

In `solvers.py`:
- `SingleTurnResonatorConvolutionSolver`: `source.get_wake_binned(...)` → `source.get_wake_per_bin(...)`.
- `ContinuousMultiTurnTimeDomainSolver._update_wake_kernel`: `source.get_wake_binned(time_axis)` → `source.get_wake_per_bin(time_axis)`; `_check_source_ducktypes`: `hasattr(source, "get_wake_binned")` → `"get_wake_per_bin"` and update the error message.
- `MultiPassResonatorSolver`: replace the `get_wake_counter_rotation_binned(...) if … else get_wake_binned(...)` branch with `source.get_wake_per_bin(time, counter_rotating=<the same condition>)`.

- [ ] **Step 6: Update the moved test and run**

Update `test_solvers.py::test_update_wake_kernel_fails` faulty-mock method name and regex from `get_wake_binned` to `get_wake_per_bin`. Run:
`.venv/bin/python -m pytest tests/unittests/physics/impedances/ -q`
Expected: PASS (fix any leftover `get_wake_binned`/`get_wake_counter_rotation_binned` references in tests until green).

- [ ] **Step 7: Commit**

```bash
git add blond/physics/impedances/ tests/unittests/physics/impedances/
git commit -m "Rename get_wake_binned to get_wake_per_bin with counter_rotating flag"
```

---

### Task 3: Derive `get_impedance_from_wake` on the base

Collapse the three duplicate `get_impedance_from_wake` bodies (and `Resonators.get_impedance_from_wake_counter_rotation`) into one cached base default = `rfft(get_wake_per_bin(...))`.

**Files:**
- Modify: `blond/physics/impedances/base.py` — `TimeDomain.get_impedance_from_wake` (`:164`, currently abstract → concrete default); `TimeDomainCounterRotation.get_impedance_from_wake_counter_rotation` (`:254`, delete).
- Modify: `blond/physics/impedances/sources.py` — delete `Resonators.get_impedance_from_wake` (`:436`) and `..._counter_rotation` (`:484`); delete `ImpedanceTableTime.get_impedance_from_wake` (`:1105`) and `TravelingWaveCavity.get_impedance_from_wake` (`:1288`); rework `Resonators.get_impedance_from_wake_freq` (`:544`) to not depend on the old single cache attribute.
- Modify: `blond/physics/impedances/solvers.py` — `TimeDomainFftSolver` call already uses `get_impedance_from_wake(...)`; add `counter_rotating=False` for clarity if desired (no behaviour change).
- Test: `test_sources.py` (existing `get_impedance_from_wake` tests must still pass; keep the Nyquist assertion).

**Interfaces:**
- Consumes: `get_wake_per_bin(time, counter_rotating)` (Task 2).
- Produces: `TimeDomain.get_impedance_from_wake(self, time, simulation, beam, n_fft, counter_rotating=False) -> ndarray`, cached by `(get_hash(time), counter_rotating)` in a lazily-created dict `self._impedance_from_wake_cache`.

- [ ] **Step 1: Confirm the current tests as the safety net**

Run: `.venv/bin/python -m pytest tests/unittests/physics/impedances/test_sources.py -k "impedance_from_wake" -q`
Expected: PASS now (these become the regression guard for this task).

- [ ] **Step 2: Implement the base default**

Replace the abstract `TimeDomain.get_impedance_from_wake` with a concrete method:

```python
def get_impedance_from_wake(
    self, time, simulation, beam, n_fft, counter_rotating=False
):
    """Impedance from the bin-averaged wake: ``rfft(get_wake_per_bin(...))``.

    Cached per (time, counter_rotating). Sources whose impedance is not a
    wake FFT (e.g. InductiveImpedance) override this.
    """
    cache = getattr(self, "_impedance_from_wake_cache", None)
    if cache is None:
        cache = self._impedance_from_wake_cache = {}
    key = (get_hash(time), bool(counter_rotating))
    if key in cache:
        return cache[key]
    self._assert_wake_time_resolves_resonances(time)  # Resonators only; see below
    wake = self.get_wake_per_bin(time, counter_rotating)
    impedance = backend.fft.rfft(wake, n=n_fft)
    cache[key] = impedance
    return impedance
```

Move the Nyquist assertion currently in `Resonators.get_impedance_from_wake` into a `Resonators` hook `_assert_wake_time_resolves_resonances(self, time)` that the base calls if present (default: no-op on the base). Keep the assertion text identical.

- [ ] **Step 3: Delete the per-source duplicates**

Delete `Resonators.get_impedance_from_wake` (`:436`), `Resonators.get_impedance_from_wake_counter_rotation` (`:484`), `ImpedanceTableTime.get_impedance_from_wake` (`:1105`), `TravelingWaveCavity.get_impedance_from_wake` (`:1288`), and `TimeDomainCounterRotation.get_impedance_from_wake_counter_rotation` (`:254`). Keep `InductiveImpedance.get_impedance_from_wake` (`:272`).

- [ ] **Step 4: Fix `get_impedance_from_wake_freq`**

Rewrite `Resonators.get_impedance_from_wake_freq` (`:529`) to derive length from the cached impedance for that `time`:

```python
def get_impedance_from_wake_freq(self, time):
    key = (get_hash(time), False)
    impedance = self._impedance_from_wake_cache[key]
    return backend.fft.rfftfreq(len(impedance), time[1] - time[0])
```

(It is only called after `get_impedance_from_wake`, so the cache entry exists — the existing `test_get_impedance_from_wake` calls them in that order.)

- [ ] **Step 5: Run the impedance suite**

Run: `.venv/bin/python -m pytest tests/unittests/physics/impedances/ -q`
Expected: PASS. If the counter-rotation impedance test referenced the deleted `get_impedance_from_wake_counter_rotation`, switch it to `get_impedance_from_wake(..., counter_rotating=True)`.

- [ ] **Step 6: Commit**

```bash
git add blond/physics/impedances/ tests/unittests/physics/impedances/
git commit -m "Derive get_impedance_from_wake from get_wake_per_bin on the base"
```

---

### Task 4: Delete `TimeDomainCounterRotation`

With its methods folded into the flag, the mixin is empty and unused.

**Files:**
- Modify: `blond/physics/impedances/base.py` — delete `class TimeDomainCounterRotation` (`:193`).
- Modify: `blond/physics/impedances/sources.py` — drop `TimeDomainCounterRotation` from the import (`:41`) and from `Resonators`'s bases (`:316`); update the two docstring references (`:575` etc.) to point at `TimeDomain`.
- Test: whole suites.

**Interfaces:** none new.

- [ ] **Step 1: Grep for remaining references**

Run: `grep -rn "TimeDomainCounterRotation\|get_wake_binned\|get_wake_counter_rotation\|get_impedance_from_wake_counter_rotation\|\.get_wake(" blond/ tests/ --include=*.py`
Expected: only the class definition, import, base list, and docstrings remain.

- [ ] **Step 2: Delete the class and references**

Remove the class, the import, the base-list entry, and fix the docstrings.

- [ ] **Step 3: Run the full impedance + physics suites**

Run: `.venv/bin/python -m pytest tests/unittests/physics/ -q`
Expected: `275 passed` (or current count), `impedances/` all green including the `rtol=1e-9` CR cross-check.

- [ ] **Step 4: Run the EX_05 integration test**

Run: `.venv/bin/python -m pytest tests/integration/examples/scripts/test_EX_05_Wake_impedance.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blond/physics/impedances/
git commit -m "Delete TimeDomainCounterRotation; counter-rotation is now a flag"
```

---

## Self-Review

- **Spec coverage:** two entry points (Tasks 1–2); CR flag (Tasks 1–2); derived impedance default + dedup (Task 3); `TimeDomainCounterRotation` deletion (Task 4); `InductiveImpedance` untouched (explicit in Task 3); pole solver untouched (Global Constraints + no task references it); `get_impedance_from_wake_freq` loose end (Task 3 Step 4). All covered.
- **Placeholder scan:** every code step shows the code; no TBD/TODO.
- **Type consistency:** `get_wake_per_particle(time, counter_rotating=False)` and `get_wake_per_bin(time, counter_rotating=False)` used identically across Tasks 1–3; `get_impedance_from_wake(..., counter_rotating=False)` consistent between Task 3 definition and Task 2/4 callers.
