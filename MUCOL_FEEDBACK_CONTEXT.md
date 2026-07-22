# Muon-Collider Cavity Feedback — Work Context & Handoff

Consolidated record of the mucol cavity-feedback work on the BLonD submodule
(branch `blonder_feature/mucol_feedbacks`), July 2026. Checkpoint-committed
2026-07-22 ("current state, tbr") — still to be reviewed before the MR.

This file is a handoff summary. The authoritative living docs are
`docs/feedbacks/mucol_cavity_feedback.rst` (design) and
`docs/tests/mucol_cavity_feedback_tests.rst` (test inventory).

---

## 0. Scope, constraints, and how to work in this repo

**Scope**: muon-collider (mucol) cavity feedbacks and their base classes only.
LHC / SPS / experimental / legacy feedback code is **out of scope** and must not
change behaviour. Shared code (impedance solvers, `rf_beam_current`) was touched
only where explicitly authorised (the counter-rotating work).

**Hard invariants (must always hold):**
- **LHC path frozen**: `rf_beam_current` (now in `beam_current.py`) must stay
  byte-identical for co-rotating beams. The LHC comparison suite must stay green
  at its original tolerances.
- **n = 1 / single-beam path bit-identical**: a single co-rotating beam must
  produce bit-identical results before/after any change; a single
  counter-rotating µ⁻ beam must reproduce the co-rotating µ⁺ run bit-for-bit.
- Feedback classes / traversals are deliberately separate (base vs timing class;
  `rf_beam_current` vs `rf_beam_current_partial`) — do not merge them.

**Environment / gotchas:**
- Run pytest from `BLonD/` with `.venv/Scripts/python.exe` and `MPLBACKEND=Agg`.
- The pre-commit `check copyright` (`custom-py-check`) hook is **broken on this
  machine** (always fails, `WinError 3`); ignore it, trust the other hooks
  (ruff, isort, numpydoc). Module-docstring summary must start on the line
  **after** the opening `"""` (numpydoc GL01 convention in this repo).
- All mucol test files gate debug plotting on `DEBUG_PLOT = False`; never leave
  it `True` (a guarded `plt.show()` would fire).
- 2 SPS `TestTravelingWaveCavity` failures (`test_vind`, `test_beam_fine_coarse`)
  were pre-existing and have since been fixed SPS-locally (90° IQ rotation);
  unrelated to mucol.

---

## 1. Physics conventions established (READ FIRST)

### 1.1 Direction-signed charge (counter-rotating beams)
In the symmetric collider ring the counter-rotating µ⁻ beam has opposite charge
*and* opposite direction, so its gap (beam) current has the **same sign** as the
co-rotating µ⁺ beam; both beams see the same beam loading and receive the same
kick.

The whole beam-loading chain uses `beam.signed_charge_with_direction()`
(`blond/core/beam/base.py`, returns `particle_type.charge * -1` for a
counter-rotating beam), on **every source-current site and every kick**:
- `rf_beam_current` / `rf_beam_current_partial` (`beam_current.py`)
- all four wake-solver source-charge sites (`impedances/base.py`,
  `impedances/solvers.py`: SingleTurnResonatorConvolution, MultiPassResonator,
  MultiPoleSparse)
- kicks: `cavities.py`, `impedances/base.py` `WakeField._track`,
  `experimental/physics/kick_pooling.py`

For a co-rotating beam the signed charge equals the plain charge, so the LHC
path is bit-unchanged.

Why deposits had to change (not just labels): the kick side was *already* signed
before this work (shared with the design RF kick, immovable). Raw deposits ×
signed kick meant a µ⁻ counter-rotating beam's self-wake **accelerated** it
(measured `dE = −(µ⁺co)`). Signing the deposits was the only fix.

### 1.2 `shunt_impedances_counter_witness` (R_CR) convention
Public kwarg renamed from `shunt_impedances_counter_rotating` (2026-07-22,
after the sign-convention flip); the old name is a trapped kwarg raising
`TypeError` with a migration message. The internal attribute was renamed in
a follow-up (user directive) to `_shunt_impedances_counter_witness` — the
MultiPole solver guard's `getattr` string, EX_28, and the direct test
accesses were swept together, so there is no old-name attribute anywhere.
`R_CR` is **the shunt the counter-rotating witness experiences**, its reversed
integration direction included. Effective cross-coupling = `−R_CR/R`. Therefore:

| | `R_CR = +R` | `R_CR = −R` |
|---|---|---|
| **same charge** (µ⁺/µ⁺CR) | build-up | cancellation |
| **opposite charge** (µ⁺/µ⁻CR, the collider pair) | cancellation | **build-up, same kick** |

- `R_CR = +R` ⇒ two counter-rotating beams of the **same charge** add up.
- `R_CR = −R` ⇒ an **asymmetric fundamental mode**: the collider pair (µ⁺/µ⁻)
  adds up and receives the same kick.

**IMPORTANT**: the −R↔FM mapping is a property of the mode's **field symmetry**,
NOT of fundamental modes in general — comments/docs everywhere say "asymmetric
fundamental mode". `|R_CR|` must equal `R` (constructor assert, "no energy
conservation"); only the sign is free. A single beam / self-wake / same-direction
interaction **never consults R_CR** (XOR wake selection), so single-turn / single-
beam behaviour is independent of it.

Closed form for two counter-rotating passages offset by Δ on the ringing tail:
`v₂ = (s₂ − F·g)·v₁`, `s₂` = signed charge of the CR beam, `F = sign(R_CR/R)`,
`g = exp(−ωΔ/2Q)`. Build when `s₂F = −1`, cancel when `s₂F = +1`.

---

## 2. Work completed (by theme)

### 2.1 Review-driven cleanup & bug fixes (earlier in the session)
- Documentation drift fixed; `PassiveCavity` deleted after porting its pre-fill
  capability into `IQCavityFeedbackTimingClass`; `"yorak"` placeholders removed.
- `±π/2` demodulation convention verified.
- Base-class `np.floor→int` crash fixed; `voltage_setpoint` read-only-property
  bug fixed; multi-station `delta_omega_rf` guard added.
- `delta_omega_rf` phase-slip reworked to elapsed-reference-time (must stay at
  the END of `_track`).
- Test-hardening campaign (sections × acceleration × substepping); five
  production bugs found by tests and fixed: substepped demod sign flip, LHC
  centering convention, multi-section frame drift, stale reverse re-pass,
  bistable demod residual.

### 2.2 P1 — exact exponential coarse propagator (option)
`exponential_coarse_solver: bool = False` on `IQCavityFeedbackTimingClass`.
`cavity_response` routes through `_advance_coarse_voltage`, which does either
forward-Euler (default, **bit-unchanged**) or the exact
`V_{n+1} = e^L V_n + src·(e^L−1)/L`. Under pure detuning the exponential step
preserves `|V|` (a rotation) where Euler grows it by `√(1+(δω·dt)²)`. Tests:
`TestExponentialCoarseSolver`.

**Review fix**: `_check_step_sizes` now early-returns when
`exponential_coarse_solver` is set — the forward-Euler stability cap and its
warnings no longer gate the exact solver (it was previously unreachable in the
low-Q_L / large-detuning regime it exists for).

### 2.2b P6 — numba coarse-envelope kernel (performance item #1, DONE)
The per-cell coarse recursion (`circuit_track`→`cavity_response`→
`_advance_coarse_voltage` + inline PI, ~10⁵ cells/turn, ~95% interpreter
overhead) is compiled to a numba host kernel `envelope_pi_scan`
(`envelope_kernel.py`). It is **host-only** (feedback is sequential signal
processing — no GPU parallel scan) and **on by default**
(`use_numba_envelope_kernel: bool = True`, class attribute; set `False` per
instance to force the reference). Measured ~79× on a 1000-cell RCS segment
(9.4 µs→0.12 µs per cell).

`circuit_track`'s cell loop was extracted into `_circuit_track_cells`
(dispatch) → `_circuit_track_cells_python` (byte-identical reference **and**
fallback) / `_circuit_track_cells_kernel` (host glue + kernel call). The glue
precomputes the *state-independent* per-cell voltage multiplier `B` (`1+L` Euler
/ `e^L` exponential) and drive weight `W` (`1` / `(e^L−1)/L`) — "the elementwise
glue on the python side" — so the kernel is **solver-agnostic** and byte-identical
to *both* solvers without numba ever evaluating `exp`/`expm1`. Step sizes
(`_coarse_step_sizes`), beam current (`_kernel_beam_current`) and PI state
(`_kernel_controller_params`/`_store_controller_state`, deque↔circular-buffer)
are marshalled per segment.

Two exact-fallback paths keep it **byte-identical** (proven: full feedback suite
+ ~17-digit pinned trajectories pass unchanged with the kernel on):
- **Coincident (zero) coarse step** → `_circuit_track_cells_python`
  (skip-and-warn can't vectorise).
- **Klystron-limit saturation** → the kernel flags any cell within a 1e-9 guard
  band of `max_output` and the segment reruns on the reference path, because
  numba's complex `abs` differs from numpy's *scalar* `np.abs` by 1 ULP (~40% of
  values), which the reference `clamp_magnitude` would otherwise not match. When
  no cell nears the limit the clamp is never applied → identical.

`_check_beam_kick_magnitude` runs as a vectorised post-pass (`_check_beam_kicks`)
that *delegates* to the per-cell checker for message fidelity + warn-then-raise
ordering. Tests: `test_envelope_kernel.py` (bit-identity across Euler/exponential
× constant/PI × delay/clamp/detuning × multi-section). `TestPIReverseSpanFrameConsistency`
(below) and the pinned trajectories force/exercise the reference path.

### 2.3 P2 — PI reverse-span frame fix
The controller is stepped only on the real forward passage
(`if self._controller_active and not no_beam`), never on the reverse
reconstruction segments (which carry a per-segment frame phase). Tests:
`TestPIReverseSpanFrameConsistency` (structural call-count on the pure-Python
reference path — the kernel inlines the PI, so these count-tests set
`use_numba_envelope_kernel=False`; mutation-verified — reverting the gate fails
them).

### 2.4 Coverage tests T3, T5–T10
- `test_wake_vs_feedback_dynamics.py` (T3): self-consistent multi-turn *dynamics*
  twin (wake vs feedback) on the fast ramp.
- `test_mtw_vs_nondriven_feedback.py` (T5–T10): δω_rf+beam, long-horizon secular
  drift, non-divisible harmonic (**xfail** — real gap: charge in first coarse
  cell → ValueError), detuned regression-lock, driven beam-part linearity,
  substepped+detuned; plus `test_multiturn_counter_rotating_mu_minus_matches_mu_plus`.
- `test_multibunch_beam_loading.py` (T8): multi-populated-coarse-cell loading.

### 2.5 Adversarial test-quality audit (fixed 4 confirmed defects)
- T5 gate was too loose to detect the δω_rf demod chain (offset cancels in the
  reference-subtraction) — documented / gate reasoned.
- `test_first_coarse_cell_precondition` was vacuous — rewritten to exercise the
  real coarse-cell guard via `rf_beam_current(forbid=True)` + negative control.
- Multibunch trailing/global gates tightened toward the measured floors
  (~0.2 %) with headroom, instead of a loose 2 %.

### 2.6 CR-1 … CR-4 (counter-rotating beams)
- **CR-1**: signed charge in `rf_beam_current` / `rf_beam_current_partial`;
  `TestRfBeamCurrentCounterRotating` (full sign matrix, mutation-verified);
  `StubBeam` gained `particle_type`/`is_counter_rotating` + the method. LHC
  battery bit-identical.
- **CR-2**: signed charge on all four wake-solver source sites. Fixture/test
  migration for the sign; `TestCounterRotatingBeamKickSymmetry` (4-corner
  self-kick equality + deceleration + coincident doubling/cancellation +
  cross-kick), production `WakeField._track` path, mutation-verified.
- **CR-3**: two simultaneous beams under `MainloopCounterRotatingBeams`.
  Empirically (not code-reading): **offset passages** (even section counts,
  stations off the meeting azimuths) already work at reference accuracy
  (0.04–0.13 % vs the two-beam convolution) — pinned by
  `TestTwoBeamOffsetPassages`. **Meeting-azimuth / simultaneous passages**
  (e.g. single mid-ring station) are serialized wrongly → the feedback now
  **raises `NotImplementedError`** (`_track`, coincidence guard); pinned by
  `TestSimultaneousPassageGuard`. The equal-time patch path is **deferred by
  user decision**. New file: `test_two_beam_counterrotating_feedback.py`. Also
  fixed the `kick_pooling.py` unsigned-charge kick.
- **CR-4**: flipped the R_CR convention (see §1.2). 3 sign negations in
  `sources.py` only (`get_wake_counter_rotation`, its quadrature,
  `get_vectorfit`) + the parameter docstring; `get_impedance` counter-rotating
  branch negated too (found in the review — it was inconsistent with the wake
  path). All consumers inherit; deposits/kicks/kernels/feedback untouched.
- **2×2 matrix**, both solvers: `TestCounterRotatingTwoBeamMatrix` verifies the
  charge-pair × shunt-sign table on MultiPassResonatorSolver AND
  MultiPoleSparseSolve, closed form on the ringing tail, cross-solver agreement
  ~1e-13. MultiPole cannot take two coincident same-time passes (sequential
  state machine) → matrix uses a 2·t_rf offset.
- **Single-beam independence** pinned: `test_single_beam_never_consults_the_
  counter_rotating_shunt` (µ⁻CR alone with `R_CR` **unset** — the source raises
  if the CR wake is ever consulted — runs and is bit-identical across shunt
  signs).

### 2.7 Full five-dimension review (Opus, adversarially verified)
Confirmed the core physics **sound** (ODE discretizations, PI law + anti-windup
+ power inverse, pre-fill fixed point on/off resonance, the full signed-charge
sweep, two-beam test rigor). Confirmed findings, all fixed or flagged:
- exponential-solver step-size guard (fixed, §2.2)
- `get_impedance` CR sign inconsistency (fixed, §2.6)
- `samples_per_rf` docstring was 2π-wrong (it is `ω·dt`, not `dt·f`) — fixed
- stale `TestPIReverseSpanFrameConsistency` class docstring — fixed
- `test_single_section_convolution_reference_needs_delta_t_zero` now actually
  exercises `allow_delta_t_zero=True` (was assert-fail-only)
- duplicate RST class block + dangling matrix-test cross-reference — fixed

### 2.8 Module partition (P1–P5, behaviour-preserving)
See §4. `cavity_feedback.py` 2462 → 1672 lines.

### 2.9 Misc
- `DEBUG_PLOT` in `test_mucol_cav_fdbk.py` was `True` (the last stray) → `False`;
  no `plt.show()` fires in a normal mucol run now.
- Public kwarg renamed `shunt_impedances_counter_rotating` →
  `shunt_impedances_counter_witness` (see §1.2): legacy kwarg trapped with a
  `TypeError` + migration message (sign convention changed with the rename, so
  no silent pass-through), internal attribute kept, both source-side
  `RuntimeError` messages now name the public kwarg, unit typo `\omega` →
  `\Omega` fixed on both shunt docstrings. Swept: `sources.py`, `solvers.py`
  (incl. the MultiPole guard message), 4 test files, feedback RST.

---

## 3. Open items / flagged (NOT done — need decisions)

- ~~MultiPole vs MultiPass on missing R_CR~~ **RESOLVED**: the follow-up
  session's guard landed via `origin/blonder` (commit `2235e519`, merged in
  `b047e972`) — MultiPoleSparseSolve now raises on a CR beam with any source
  missing R_CR. NOTE: that merge also produced one **semantic conflict** in
  `test_sources.py::test_get_impedance` (origin's `−R` construction ×
  our negating `get_impedance` kept by the merge ⇒ sign flip); fixed by
  restoring the `+R` construction matching the surviving convention.
- **`barrier_bucket.py` CR kick** (spawned task, still pending): line ~245 uses
  raw `particle_type.charge` instead of `signed_charge_with_direction()` —
  wrong-sign barrier kick for a CR beam. Out of feedback scope.
- ~~`print_one_turn_execution_order` crash on empty `rf_centers`~~ **RESOLVED**
  (committed `0936668f`, with regression tests in
  `tests/unittests/core/ring/test_beam_physics_relevant_elements.py`).
- ~~`phase_correction` vs `pi_setpoint` frame~~ **RESOLVED (user decision:
  error)**: the constructor now rejects a non-real / non-positive explicit
  `voltage_setpoint` with `ValueError` (rotate `phi_rf` on the station
  instead); `TestVoltageSetpointValidation` pins it.
- **CR-3 equal-time patch path** (deferred by user): integrating two coincident
  beam currents in the feedback (deposit-sum + envelope rewind). Design options
  recorded in the memory note. Kick-ordering fork: symmetric one-passage delay /
  pooled kick / asymmetric lag.
- **Per-beam live profiles** under two-beam tracking clobber each other (tests
  use frozen profiles) — core gap. Mitigation shipped (user decision): the CR
  mainloop now validates at start that every live profile consumed by an
  element (direct, via `_local_wakefield`, or via `cavity_feedback_list`) is
  tracked both before AND after that consumer — the (profile, consumer,
  profile) sandwich — or is frozen (`active = False`); raises `ValueError`
  otherwise. `_check_two_beam_profile_placement` in
  `blond/core/simulation/execution_models/conterrotating_beams.py`, tests in
  `test_simulation.py::TestTwoBeamProfilePlacementCheck`.
- **`delta_omega_rf` lab-frame demod slip** (characterized, not fixed): fix =
  anchor the demod carrier to accumulated ∫ω dt (reuse the kick-side slip
  bookkeeping), gated on `delta_omega_rf ≠ 0` so LHC is untouched. Effort
  ~0.5–1 day: the code change is small; validation dominates (T5 differential
  rework, larger-offset multi-turn tests, turn-0 application convention,
  substepping/multi-section interplay — the demod-frame area is historically
  delicate).
- `experimental/physics/feedbacks/helpers.py` still uses raw charge —
  **confirmed intended**: its `rf_beam_current` copy is consumed only inside
  `blond/experimental/physics/feedbacks/` (the old LHC/SPS feedbacks); nothing
  in the main tree or mucol imports it.
- **P6** (RF-parameter view mixin) skipped per user.
- **Full Sphinx doc build not yet run**: only docutils structure-lint was run on
  the two RSTs. CI builds with `sphinx-build -W` (warnings = errors, see
  CLAUDE.md); run `cd docs && bash create_docs.sh` (sequentially, never looped)
  before the MR. No new top-level exports were added, so
  `ASSIGNED_CATEGORIES` needs no update.
- The extracted mixins (`RFCenterGridMixin`, `GeneratorRegulationMixin`) are
  pure moves (methods still use `self`); promoting them to composed collaborators
  is the natural follow-up.

---

## 4. Module layout after the P1–P5 partition

`blond/physics/feedbacks/`:

| module | holds |
|---|---|
| `cavity_feedback.py` | `IQCavityFeedbackBase` + `IQCavityFeedbackTimingClass(IQCavityFeedbackBase, RFCenterGridMixin, GeneratorRegulationMixin)`; orchestration (`_track`, `circuit_track`, `_circuit_track_cells{,_python,_kernel}` + `_coarse_step_sizes`/`_kernel_*` glue, `cavity_response`, `_advance_coarse_voltage`, `cavity_response_fine`, `calculate_rf_beam_current_partial`, `on_run_simulation`, pre-fill, `_check_step_sizes`, `_check_beam_kick_magnitude`/`_check_beam_kicks`) |
| `envelope_kernel.py` | numba host kernel `envelope_pi_scan` — the sequential coarse-envelope + PI recursion (performance item #1); solver-agnostic, byte-identical to the Python reference |
| `rf_center_segment.py` | `RFCenterSegment` value class (re-exported from cavity_feedback for compat) |
| `rf_center_grid.py` | `RFCenterGridMixin` — coarse `rf_centers` grid construction (forward/reverse reference walks, segment generation, derived arrays) |
| `generator_regulation.py` | `GeneratorRegulationMixin` — `_controller_active`, `pi_setpoint`, `generator_power`, `_update_generator_current` |
| `cavity_solvers.py` | **mucol-only** `cavity_response_sparse_matrix_second_order` (Crank-Nicolson) + `pretrack_fill_voltage` |
| `helpers.py` | first-order `cavity_response_sparse_matrix` (**shared with LHC**) + re-export shims |
| `beam_current.py` | `low_pass_filter`, `rf_beam_current` (**LHC-frozen**), `rf_beam_current_partial` |
| `iq.py` | `cartesian_to_polar`, `polar_to_cartesian` |
| `generator_current_controller.py` | `GeneratorCurrentPIController` (unchanged) |
| `base.py` | `FeedbackBaseClass` / `LocalFeedback` / `GlobalFeedback` (unchanged) |

**Re-exports**: `helpers.py` re-exports the beam_current + iq symbols
(`# noqa: F401`) so experimental/LHC/SPS/legacy imports (`from ...helpers import
rf_beam_current`) keep working untouched. Mucol production + tests import from
the new canonical modules.

**Test split** (`tests/unittests/physics/feedbacks/`):
`test_rf_center_grid.py` (was `TestIQCavityFeedbackTimingClass`),
`test_rf_center_segment.py` (was `TestRFCenterSegment`); `test_cavity_feedback.py`
reduced to the empty `TestIQCavityFeedbackObservationClass` stub. The unused
debug method `plot_antenna_voltage` moved to
`unittests/physics/feedbacks/accelerators/mucol/plotting.py` as a function.

---

## 5. Verification status

- Full battery (mucol + LHC comparisons + impedances): **492 passed**, the only
  failures being the (now separately fixed) pre-existing SPS TravelingWaveCavity
  tests — zero regressions from any of this work.
- Every production sign/gate change is **mutation-verified** (reverting the fix
  fails the pinning test): P2 gate, exponential branch, T7 xfail reason, CR-1/CR-2
  sign matrices, CR-4 matrix, MultiPole deposit.
- The P1–P5 partition is **byte-identical** (pure moves), verified by the full
  battery + per-step ruff/numpydoc/import/MRO checks.
- Docs (`docs/feedbacks/…rst`, `docs/tests/…rst`) updated and structurally lint-
  clean; memory notes updated (`mucol-feedback-module-layout`,
  `cr-beam-loading-architecture`, and the P1/P2/open-limitations notes).

---

## 6. Nothing is committed
The entire body of work above is uncommitted in the working tree. Recommended
commit grouping: (a) review fixes + coverage tests, (b) CR-1…CR-4 +
convention/tests + docs, (c) the P1–P5 module partition (a large but pure move),
(d) the `DEBUG_PLOT` fix. Re-run the battery before/after any reshuffle.
