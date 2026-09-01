# Session context — counter-rotating R_CR fail-fast parity (MultiPole vs MultiPass)

Date: 2026-07-15
Branch: `blonder_feature/mucol_feedbacks` (BLonD submodule)

> **Status: superseded working note — kept deliberately as a session log.**
> Re-verified 2026-08-08.
>
> The *conclusions* of this note are published in two class docstrings in
> `blond/physics/impedances/solvers.py` and are the canonical statements:
> the `MultiPoleSparseSolve` `Notes` (eager fail-fast for any
> counter-rotating beam, and why the lone-beam strictness is accepted) and
> the `MultiPassResonatorSolver` `Notes` (the XOR direction test, and why a
> lone counter-rotating beam needs no `R_CR`). Read those first. What is
> **not** published, and is the reason this file survives, is the
> *mechanism* half of "Why MultiPass never needed a separate guard" below —
> that MultiPass's raise is intrinsic because
> `Resonators.get_wake_counter_rotation` raises at its own point of use, so
> no silent default can enter. That is an implementation note, not
> public-API text.
>
> Two rots to be aware of while reading:
>
> - **Every line number originally quoted here was a 2026-07-15 snapshot and
>   has since moved.** They have been replaced by symbol names throughout —
>   navigate by symbol, and do not reintroduce line numbers here.
> - The public kwarg was renamed `shunt_impedances_counter_rotating` →
>   `shunt_impedances_counter_witness` on 2026-07-22, **with a
>   sign-convention change** (the CR-4 flip). The old name is now a trapped
>   kwarg raising `TypeError` in `Resonators.__init__`. Passages below that
>   still name the old spelling are quoting the state of the tree in July.
>
> Attribution caveat: `MUCOL_FEEDBACK_CONTEXT.md` §3 records that the
> MultiPole guard actually landed via `origin/blonder` (commit `2235e519`,
> 2026-07-10, merged in `b047e972`) rather than being implemented in this
> session. The two accounts conflict; the git history is the arbiter.

## Task

The two counter-rotating-capable wake solvers disagreed on a missing
`R_CR` (then spelled `shunt_impedances_counter_rotating`):
`MultiPassResonatorSolver` **raised**, while `MultiPoleSparseSolve`
**silently defaulted** to the wrong cross-coupling sign. Goal: give
MultiPole the same fail-fast behavior, add a regression test, and document
*why* MultiPass never needed the guard. (Surfaced by the mucol feedback
review; this is shared impedance-solver code, beyond mucol scope.)

## The bug (silent-wrong cross-beam coupling) — HISTORICAL

*Historical record of the defect as it stood on 2026-07-15. The current
behaviour is the fixed one; see the two `Notes` sections named in the banner.*

A counter-rotating two-beam **MultiPole** run that forgot to set `R_CR` got
silently-wrong cross-beam coupling on a symmetric mode, whereas the identical
mistake hard-errored on **MultiPass**.

- MultiPass: `Resonators.get_wake_counter_rotation` raises `RuntimeError`
  when `_shunt_impedances_counter_witness is None`.
- MultiPole: `Resonators.get_vectorfit` **substitutes** `cr_signs = np.ones`
  (i.e. `+1`) when `_shunt_impedances_counter_witness is None`, instead of
  the real `-np.sign(R_CR)`. After the CR-4 convention flip, `+1` equals the
  asymmetric-fundamental-mode cross-coupling and is the **sign-opposite** of
  the symmetric-mode (`+R`) case, so a symmetric mode is silently mis-signed.

## Why MultiPass never needed a separate guard (the crux)

*This is the part of the note that is not published anywhere else. The
conclusion — "a lone counter-rotating beam needs no `R_CR`" — is in the
`MultiPassResonatorSolver` `Notes`; the mechanism below is not.*

MultiPass's raise is **intrinsic and precise** — it comes for free from the
architecture, at the exact point of use:

1. **Intrinsic.** MultiPass builds induced voltage by convolving a wake
   *function* it fetches per passage. On a genuine cross-direction passage
   `MultiPassResonatorSolver._update_past_profile_wake_functions` calls
   `source.get_wake_counter_rotation(...)` (and its `..._quadrature` twin),
   and that method *itself* raises on unset `R_CR`. The chain from the
   public entry point is `calc_induced_voltage` →
   `_update_potential_sources` → `_update_past_profile_wake_functions`,
   which is why the class `Notes` attribute the raise to
   `calc_induced_voltage`. There is no place for a silent default to enter.
   MultiPole never calls it — it consumes a pre-baked pole-sign array from
   `Resonators.get_vectorfit`, which defaults instead of raising.

2. **Precise.** MultiPass keeps a per-past-profile direction flag and XORs
   each stored profile against the current passage (index 0), calling
   `get_wake_counter_rotation` **only when the directions differ**. A lone
   counter-rotating beam reading back its own past passages XORs to `False`,
   takes `get_wake`, and never touches `R_CR` — which is correct, since
   self-wake is `R_CR`-independent.

MultiPole *cannot* be that precise: it merges every deposit into one
continuous `states` vector and relies on the `flip²=1` cancellation to
separate self- from cross-wake numerically, so at raise-time it can't
cheaply tell self- from cross-wake. Its guard is therefore a coarser
superset.

## Fix (MultiPole fail-fast parity) — SUPERSEDED

*Canonical description: the `MultiPoleSparseSolve` `Notes` in
`blond/physics/impedances/solvers.py`. In the code, the two sites are
`MultiPoleSparseSolve._finalize_solver` (which sets
`_any_source_missing_shunt_cr` from a `getattr(source,
"_shunt_impedances_counter_witness", None)` probe) and the track/kernel
dispatch in `calc_induced_voltage` (which raises `RuntimeError` on
`beam.is_counter_rotating and self._any_source_missing_shunt_cr`); both carry
their own rationale comments.*

The one design decision worth keeping in prose, because the comments state
it but the docstring only alludes to it: the guard keys on the **current
beam's direction only**, so it is a touch stricter than MultiPass — it fires
on the harmless, `R_CR`-independent lone counter-rotating self-wake too.
Accepted trade-off: no false negatives on the silent-wrong bug, at the cost
of requiring `R_CR` be set (any sign, since it is bit-identical there) for a
lone counter-rotating MultiPole run. Co-rotating MultiPole runs (the common
case) are unaffected.

## Documentation — SUPERSEDED

The `Notes` section this session added to `MultiPoleSparseSolve` exists, and
a matching one was since added to `MultiPassResonatorSolver`. Both are named
in the banner above; nothing further is owed here.

## Regression tests

Run from the BLonD checkout with `MPLBACKEND=Agg`. There is no
`BLonD/.venv`: BLonD is a submodule of the outer muon-collider-blonder
repo and shares that repo's venv -- `../.venv/Scripts/python.exe` on
Windows, `../.venv/bin/python` on Linux/macOS. (A standalone BLonD clone
has neither; create your own venv and install per CLAUDE.md.)
Both live in `tests/unittests/physics/impedances/test_solvers.py`:

- `test_counter_rotating_without_shunt_cr_raises_both_solvers` — added by
  this session; both solvers must raise on counter-rotating + `Resonators`
  without `R_CR`.
- `test_single_beam_never_consults_the_counter_rotating_shunt` — pre-existing;
  a lone counter-rotating beam with `R_CR` unset must **not** raise on
  MultiPass (the precision property above).

## Verification — HISTORICAL

One-time checks run on 2026-07-15, recorded for the log only; they say
nothing about the current tree:

- `ruff format` / `ruff check` on `blond/physics/impedances/solvers.py`: pass.
- Docstring imports and parses cleanly.
