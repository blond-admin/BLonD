# Session context — counter-rotating R_CR fail-fast parity (MultiPole vs MultiPass)

Date: 2026-07-15
Branch: `blonder_feature/mucol_feedbacks` (BLonD submodule)

## Task

The two counter-rotating-capable wake solvers disagreed on a missing
`shunt_impedances_counter_rotating` (`R_CR`): `MultiPassResonatorSolver`
**raised**, while `MultiPoleSparseSolve` **silently defaulted** to the wrong
cross-coupling sign. Goal: give MultiPole the same fail-fast behavior, add a
regression test, and document *why* MultiPass never needed the guard. (Surfaced
by the mucol feedback review; this is shared impedance-solver code, beyond mucol
scope.)

## The bug (silent-wrong cross-beam coupling)

A counter-rotating two-beam **MultiPole** run that forgot to set `R_CR` got
silently-wrong cross-beam coupling on a symmetric mode, whereas the identical
mistake hard-errored on **MultiPass**.

- MultiPass: `get_wake_counter_rotation` raises `RuntimeError` when
  `_shunt_impedances_counter_witness is None` —
  `blond/physics/impedances/sources.py:648` (inside the method defined at
  `sources.py:631`).
- MultiPole: `get_vectorfit` **substitutes** `cr_signs = np.ones` (i.e. `+1`)
  when `_shunt_impedances_counter_witness is None` — `sources.py:935` — instead
  of the real `-np.sign(R_CR)` (`sources.py:947`). After the CR-4 convention
  flip, `+1` equals the asymmetric-fundamental-mode cross-coupling and is the
  **sign-opposite** of the symmetric-mode (`+R`) case, so a symmetric mode is
  silently mis-signed.

## Why MultiPass never needed a separate guard (the crux)

MultiPass's raise is **intrinsic and precise** — it comes for free from the
architecture, at the exact point of use:

1. **Intrinsic.** MultiPass builds induced voltage by convolving a wake
   *function* it fetches per passage. On a genuine cross-direction passage it
   calls `source.get_wake_counter_rotation(...)`
   (`blond/physics/impedances/solvers.py:990`), and that method *itself* raises
   on unset `R_CR`. There is no place for a silent default to enter.
   MultiPole never calls it — it consumes a pre-baked pole-sign array from
   `get_vectorfit`, which defaults instead of raising.

2. **Precise.** MultiPass keeps a per-past-profile direction flag and XORs each
   stored profile against the current passage (index 0), calling
   `get_wake_counter_rotation` **only when the directions differ**
   (`solvers.py:985-1001`). A lone counter-rotating beam reading back its own
   past passages XORs to `False`, takes `get_wake`, and never touches `R_CR` —
   which is correct, since self-wake is `R_CR`-independent.

MultiPole *cannot* be that precise: it merges every deposit into one continuous
`states` vector and relies on the `flip²=1` cancellation to separate self- from
cross-wake numerically, so at raise-time it can't cheaply tell self- from
cross-wake. Its guard is therefore a coarser superset (see below).

## Fix (MultiPole fail-fast parity)

- `MultiPoleSparseSolve._finalize_solver` records whether any contributing
  source lacked `R_CR`: `_any_source_missing_shunt_cr` via
  `getattr(source, "_shunt_impedances_counter_witness", None)` —
  `solvers.py:1399`.
- The track/kernel dispatch raises `RuntimeError` when
  `beam.is_counter_rotating and self._any_source_missing_shunt_cr` —
  `solvers.py:1515-1516` (rationale comment at `solvers.py:1503`, "mirrors the
  RuntimeError").
- The guard keys on the **current beam's direction only**, so it is a touch
  stricter than MultiPass: it fires on the harmless, `R_CR`-independent lone
  counter-rotating self-wake too. Accepted trade-off — no false negatives on the
  silent-wrong bug, at the cost of requiring `R_CR` be set (any sign, since it's
  bit-identical there) for a lone counter-rotating MultiPole run.
- Does **not** raise for co-rotating MultiPole runs (the common case).

## Documentation

Added a numpydoc **Notes** section to the `MultiPoleSparseSolve` class docstring
(`solvers.py`, just after `See Also`) capturing the whole contrast: why this
solver needs the guard, why `MultiPassResonatorSolver` raises intrinsically, and
the deliberate lone-self-wake strictness trade-off.

## Regression tests (run from `BLonD/` with `.venv\Scripts\python.exe`, `MPLBACKEND=Agg`)

- `tests/unittests/physics/impedances/test_solvers.py:4049`
  `test_counter_rotating_without_shunt_cr_raises_both_solvers` — new; both
  solvers must raise on counter-rotating + Resonators without `R_CR`.
- `test_solvers.py:3723`
  `test_single_beam_never_consults_the_counter_rotating_shunt` — existing; a
  lone counter-rotating beam with `R_CR` unset must **not** raise on MultiPass
  (the precision property above).

## Verification

- `ruff format` / `ruff check` on `blond/physics/impedances/solvers.py`: pass.
- Docstring imports and parses cleanly.
