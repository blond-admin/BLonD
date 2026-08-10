# Muon-Collider Cavity Feedback — Work Context & Handoff

Consolidated record of the mucol cavity-feedback work on the BLonD submodule
(branch `blonder_feature/mucol_feedbacks`), July–August 2026. Checkpoint-
committed throughout; last structural pass 2026-08-07 (§2.11). Still to be
reviewed before the MR.

This file is a handoff summary, and it is the *derived* copy of everything it
describes. Where it disagrees with the code or with the two RSTs, the code
wins and this file is the bug. The authoritative living docs are
`docs/feedbacks/mucol_cavity_feedback.rst` (design) and
`docs/tests/mucol_cavity_feedback_tests.rst` (test inventory); the
design-clock / demodulation invariant in particular is canonically stated in
the `IQCavityFeedbackTimingClass` docstring and in the design RST's
"Interplay with the RF station" — §3's `delta_omega_rf` bullet below only
records *when* it was resolved and *how well* it measures.

---

## 0. Scope, constraints, and how to work in this repo

**Scope**: muon-collider (mucol) cavity feedbacks and their base classes only.
LHC / SPS / experimental / legacy feedback code is **out of scope** and must not
change behaviour. Shared code (impedance solvers, `rf_beam_current`) was touched
only where explicitly authorised (the counter-rotating work).

**Hard invariants (must always hold):**
- ~~LHC path frozen~~ **OBSOLETE (2026-07-25)**: the LHC/SPS cavity feedbacks
  and the blond2 comparison suite were REMOVED from the codebase (the phase
  loop survived — moved to `blond/physics/feedbacks/beam_feedback.py`). The
  byte-identical obligation and its bridge machinery (`dT_index_sign`,
  `coarse_center_offset`, the helpers re-export shims) were stripped in the
  same cleanup. `blond/legacy/blond2/` keeps its own self-contained copies.
- **n = 1 / single-beam path bit-identical**: a single co-rotating beam must
  produce bit-identical results before/after any change; a single
  counter-rotating µ⁻ beam must reproduce the co-rotating µ⁺ run bit-for-bit.
- ~~Feedback splits~~ **MERGED (2026-07-25, user-approved unification)**:
  `rf_beam_current_partial` was folded into the single `rf_beam_current`
  (keyword-only coarse args `sampling_time`/`n_points`; offset always
  `sampling_time/2`; `external_reference`/`downsample`/`T_rev` removed;
  byte-exact migration pin `TestUnifiedRfBeamCurrentMigrationPin`).
  `IQCavityFeedbackBase` was SLIMMED, not dissolved (name kept so
  `@requires(["IQCavityFeedbackBase"])` in observables keeps string-matching
  the MRO): dead members deleted (base on_run_simulation/_track/
  track_no_beam/calculate_rf_beam_current/set_point_from_rfstation/
  update_feedback_variables/omega_carrier/residual_time_shift/t_rf/
  HasPropertyCache machinery/n_samples_coarse/use_lowpass_filter); the
  timing override now carries its OWN `@requires` decorator (regression
  test `test_cavity_feedback_requires.py` — previously it inherited the
  constraint from the decorated dead base method); `n_cavities` legalized
  as `int | float` (fractional effective-voltage scale — do NOT
  int-coerce); `harmonic_index=1` hardcode preserved + flagged
  (suspicious, unreachable with SingleHarmonicRFStation).
  `helpers.py` was DELETED — `cavity_response_sparse_matrix` now lives in
  `cavity_solvers.py` beside its second-order twin.

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
- `rf_beam_current` (`beam_current.py`; unified fine+coarse, 2026-07-25)
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
`exponential_coarse_solver_enable: bool = False` on `IQCavityFeedbackTimingClass`.
`cavity_response` routes through `_advance_coarse_voltage`, which does either
forward-Euler (default, **bit-unchanged**) or the exact
`V_{n+1} = e^L V_n + src·(e^L−1)/L`. Under pure detuning the exponential step
preserves `|V|` (a rotation) where Euler grows it by `√(1+(δω·dt)²)`. Tests:
`TestExponentialCoarseSolver`.

**Review fix**: `_check_step_sizes` now early-returns when
`exponential_coarse_solver_enable` is set — the forward-Euler stability cap and its
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

**Adversarial review (5-dimension find → verify-with-probe) found & fixed 3
carried-state divergences the first tests missed** (they seeded
`last_val_beam_current=0` and `last_val_generator_current=bias` — the values
that hide the bugs), all in the carried index-0 cell of a `no_beam` (reverse)
segment starting at grid index 0, i.e. the **first reverse segment of a
multi-section ring on turn ≥ 1**:
1. **Generator-current drive** (HIGH): the kernel held `generator_current_init`
   for every cell; `cavity_response` uses `last_val_generator_current` only at
   cell 0 and the (static, reset-bias) `generator_current_coarse_grid[idx-1]`
   for cells ≥ 1. Fix: the kernel now reads each cell's drive from the
   pre-filled `generator_current_out[cell-1]` (PI output when active, static
   grid when inactive), mirroring the reference exactly.
2. **Beam current at cell 0** (HIGH): `_kernel_beam_current` zeroed cell 0 for
   `no_beam`; the reference `idx==0` branch uses `last_val_beam_current`
   unconditionally. Fix: set it before the `no_beam` early return.
3. **Warn/assert ordering on an invalid grid** (LOW): `_coarse_step_sizes` now
   defers *any* non-positive step to the reference loop (which warns-and-skips a
   zero then asserts on a negative in order) instead of a pre-emptive vectorised
   assert.
Regression tests seed off-bias / nonzero carried state (bit-exact
`np.array_equal`); `TestKernelMatchesReferenceEndToEnd` pins a full 2-section
4-turn run kernel-vs-python byte-for-byte. REFUTED by the review: float32-grid
NEP-50 (unreachable), hard-kick post-raise state (simulation aborts), the
test-coverage observations (gaps, since covered).

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

### 2.10 LHC blond2-comparison suite — speed refactor (pinned references)

`tests/unittests/physics/feedbacks/accelerators/lhc/comparison_with_blond2/`
(reviewed physics + structure + speed; suite was 806.5 s / 13.4 min):

- **All 5 test classes now load pinned blond2 references** via
  `support.blond2_reference(name, builder)`: the frozen-legacy half runs
  once, its outputs are stored losslessly in
  `resources/<name>_blond2_reference.npz` and loaded on later runs.
  Regenerate with `BLOND_REGEN_BLOND2_REFERENCE=1` (env var). Every
  compared value and tolerance is unchanged; blond3 sides byte-identical
  except the two approved physics fixes below.
- blond2 closures became module-level `_run_blond2()` builders; dead code
  removed (tqdm/matplotlib/DEBUG blocks, never-asserted captures, the
  full-machine `imped_calc` block, a double `generator_power()` call);
  `pytestmark = backend_mutation` added where the backend is switched.
- **Physics fixes (both strictly tighten):** phase_error blond3
  commissioning now passes `open_tuner=True` (matches blond2's frozen
  tuner — it silently diverged before); the transfer-function suite's
  three phase assertions were vacuous (`atol=7` > 2π, chosen to survive
  ±π branch cuts) → replaced by wrapped-difference
  `angle(H3·conj(H2))` checks calibrated at 1e-2 / 1e-5 / 7e-1 rad
  (measured 3.9e-3 / 1.5e-6 / 5.8e-1). Also: vacuous rf-power angle
  assert deleted (power is real-positive both sides), full-machine
  set-point imag → exact-zero invariant on both sides, full-machine
  circularity (blond3 injects blond2's histogram) documented in the
  class docstring.
- Deliberately **skipped**: shrinking full_machine's dummy
  `_dE/_flags/_ids` arrays (`n_macroparticles_partial` reads
  `_dE.local_size`; risk not worth ~2 GB RAM).

### 2.11 Structural pass, 2026-08-06/07 (committed: `46d9d989`, `db86a65b`, `d2ce9d19`, `9b870c1b`, `a30e8acc`)

Behaviour-preserving except where noted. This supersedes any earlier
description of where the guards and the per-turn orchestration live.

- **`_track` decomposed into nine named phases.** `_track` now does no work
  itself — it only names the phases in order, and where a phase depends on a
  value another produced, that value is *returned and passed* rather than
  left on `self`, so the argument lists are the dependency graph. The nine:
  `_guard_simultaneous_passage`, `_carrier_slip_gap_at_passage`,
  `_close_previous_turn_grid`, `_rebuild_per_turn_grid` (returns a
  `PerTurnGridSpan`), `_replay_reverse_span`, `_accumulate_registration_phase`,
  `_write_debug_readout`, `_track_forward_span`, `_write_station_readout`.
  Two orderings cannot be expressed as arguments and are stated in the
  respective docstrings (and the first is additionally asserted per turn):
  `reset_arrays` must size the coarse state before any `circuit_track`, and
  `_carrier_slip_gap` must be complete before
  `calculate_rf_beam_current_partial` reads it off the instance. The
  registration phase `Ψ` is *accumulated* by `_accumulate_registration_phase`;
  `_track` only folds it into `_carrier_slip_gap` — so "the `_track` frame
  correction" is no longer the right name for it.
- **Reverse-span generator prefill (zero-order hold).** `reset_arrays`
  gained `n_reverse_cells`: the generator grid is still seeded with the
  feedforward bias, *except* over the leading no-beam reverse-reconstruction
  cells, which are seeded with `_last_val_generator_current`. Those cells
  replay an already-elapsed interval during which the loop issued no new
  command, so the generator kept running at whatever it was last told — it
  did not snap back to the bias. `cavity_response` already drove the *first*
  reverse cell from the held value; this extends it over the rest of the
  span. Without a controller the held value *is* the bias, so the
  constant-current path is bit-unchanged.
- **Segment-boundary residual fix.** `RFCenterSegment.residual` is now
  actually **consumed**: `RFCenterGridMixin._preceding_segment_residual`
  reads it back to form the first coarse step of the *following* segment
  (`rf_centers` are segment-local, so that step is the following segment's
  first local centre plus the preceding segment's unfilled tail). The live
  host scalar `_residual_time_last_rf_centers_calculation` cannot serve
  there — the whole per-turn grid is generated before any of it is walked,
  so by consumption time it holds the last-generated (forward) segment's
  value. The first segment of a turn takes `_residual_time_carried_into_turn`
  (snapshotted before this turn's generation); hand-built grids with no
  segment list fall back to the live scalar, bit-for-bit as before.
  `__post_init__` bounds `residual` to `[0, duration]` for a non-empty
  segment (empty segments legitimately carry the previous one's).
  **Still write-only**: `RFCenterSegment.omega` and `.duration` are
  validated in `__post_init__` and never read back — only `.centers` and
  `.residual` are consumed. Keep them (they make the record
  self-describing), but do not assume they are load-bearing.
- **Unified profile-span guard `ProfileBaseClass.check_fits_in_span`**
  (`profiles.py`), plus the new `profile_duration` cached property (the
  outer-edge span `cut_right - cut_left`, i.e. `n_bins * hist_step` — one
  `hist_step` MORE than the first-to-last-bin-centre distance the ad-hoc
  checks used to compute, which understated the window by one bin). One
  guard now serves every consumer that must place the profile window inside
  a span it does not control, and in each case `span` is the same physical
  quantity — the interval between two consecutive passages of the consuming
  element. Two call sites: `rf_beam_current` (`beam_current.py`, span =
  `n_points * sampling_time`, the FORWARD segment only — 1/n_sections of a
  turn, not a full turn and **not periodic**) and
  `MultiPassResonatorSolver.calc_induced_voltage` (span = `delta_t`).
  Both mechanisms destroy charge (re-bin fold at exactly 50 % loss; past-
  deposit self-overlap), so both raise. **Correction to an earlier note:**
  there was never a symbol named `check_profile_span_within_passage_time` —
  nothing of that name was deleted; `check_fits_in_span` and
  `profile_duration` are both new, and what they replaced were inline
  ad-hoc width computations.
  Consequence in `rf_beam_current`: the `% n_points` **wrap-around was
  removed** from the coarse-charge writes (a wrap would overwrite an earlier
  cell rather than accumulate into it, because the coarse grid is not
  periodic), backed by the span guard plus a new explicit
  `ind_fine[-1] >= n_points` `ValueError`.
- **`ForwardEulerValidityGuard` extracted** to `cavity_solvers.py`, beside
  the solvers it certifies, because it is pure numerics: it reads no grid,
  no RF station and no beam, all cavity parameters are passed per call, and
  the only state it owns is the once-only beam-kick warning flag. It holds
  the three tripwires (`check_step_sizes`, `check_beam_kick_magnitude`,
  `check_beam_kicks`) and the four thresholds (`max_step_angle`,
  `max_step_angle_hard`, `max_relative_kick`, `max_relative_kick_hard`).
  The feedback owns one instance (`self._euler_guard`) constructed with
  `enabled=not exponential_coarse_solver_enable`, which is how the
  exponential-solver early-returns of §2.2 / §3-#4 are now expressed —
  a constructor flag instead of two hand-written early returns. The
  `_check_*` methods on `IQCavityFeedbackTimingClass` survive as thin
  delegating wrappers.
- **Controller separated from the feedback.** `generator_current_controller.py`
  gained the `GeneratorCurrentController` ABC above
  `GeneratorCurrentPIController`, so the feedback holds only an instance of
  the interface and need not know the control law. The compiled path is now
  an opt-in *controller* capability, not a feedback special case: a
  controller advertises `supports_envelope_scan` and then owns its scan
  kernel (`envelope_scan_kernel`), the marshalling of its own tuning/state
  (`envelope_scan_state`) and the write-back (`absorb_envelope_scan_state`).
  Controllers that do not advertise it are driven cell-by-cell through
  `update_generator_current`. `envelope_kernel.py` keeps `envelope_pi_scan`
  and gained `inactive_controller_scan_state`.
  §2.4/§2.6 physics and the byte-identity pins are unaffected.

---

## 3. Open items / flagged (NOT done — need decisions)

**Physics review (2026-07-23, 46-agent adversarial workflow on committed HEAD
post-phase-rework): 13 findings → 7 confirmed, 6 refuted, 4 coverage gaps.**
Three confirmed items actioned as a focused cleanup pass (TDD, mutation-verified,
zero regressions — full mucol battery 156 passed):
- **#1/#5 (low) FIXED** — dead `_forward_carrier_omega_rf` attribute removed
  (was computed in `rf_center_grid`, read nowhere); demod carrier is the
  *design* RF, and the inter-turn slip enters only as the constant
  `carrier_phase_offset=-(delta_phi_rf + carrier_slip_gap)`. Corrected the
  class docstring + grid comments (they claimed the actual carrier was used
  within the window); the only residual is the intra-window `δω·hist_x`, bounded
  ~1e-6 rad and non-accumulating (bunch-local `hist_x`). Dropped the stale
  `# TODO: this is wrong` on the validated anchor. **Answers the "which path is
  more correct / can the param go to 0.0 / can it be removed" question**:
  demod-at-actual-carrier is exactly correct, demod-at-design is correct to
  ~1e-6 rad and simpler — both fine; the dead attribute is gone.
- **#4 (medium) FIXED** — the forward-Euler beam-kick guards
  (`_check_beam_kick_magnitude`, `_check_beam_kicks`) now early-return when
  `_exponential_coarse_solver_enable` is set, mirroring `_check_step_sizes`. The
  exact exponential propagator integrates the piecewise-constant drive (beam
  included) exactly, so a large per-step beam kick is not a discretisation error
  there; the guards could spuriously abort a valid exact large-step run. Tests:
  `TestExponentialCoarseSolver::test_beam_kick_guard_skipped_for_exponential_solver`
  (+ kernel-path variant), mutation-verified (Euler still raises).
- **#7 (medium) doc-fixed** — the RST no longer *recommends* the
  `MultiPassResonatorSolver(allow_delta_t_zero=True)` meeting-azimuth workaround;
  a `.. warning::` now states it gives order-asymmetric coincident kicks (first
  beam W(0)/2, second W(0)). The underlying solver asymmetry (solvers.py:934) is
  **left for a decision** — the real fix is to deposit both coincident profiles
  before evaluating either kick (symmetrise the mutual W(0)/2).
- **#2/#3/#6 (2026-07-23) FIXED** via three parallel agents (TDD, all green):
  - **#3** — `calc_phi_s_main_harmonic` / `calc_synchrotron_tune_main_harmonic`
    (`cavities.py:962`, `:899`) now use `signed_charge_with_direction()`, so a
    CR µ⁻ beam's analytic `phi_s` matches the µ⁺ co beam. Co-rotating
    bit-identical (signed==raw). The **tune is sign-robust** (uses `|charge|`,
    `|cos φ_s|`) so only `phi_s` changes value. Tests
    `TestCounterRotatingSynchronousPhase` (`test_cavities.py`).
  - **#6** — `_check_step_sizes` comment + raise message (and the class-docstring
    Sub-stepping paragraph) corrected: the Euler decay factor turns negative at
    `decay_per_step > 1` (sign-flip, still contracting) and diverges at `> 2`.
    **DECIDED + SHIPPED** (superseding the "judgment call for the user" this
    bullet used to pose): the hard cap was tightened from `2.0` to `1.0`
    (`ForwardEulerValidityGuard.max_step_angle_hard`), so the sign-flipping
    `1 < d < 2` band is forbidden too, not just the divergent `d > 2`.
    Pinned by `test_decay_hard_cap_forbids_sign_flip` (0.9 warns, 1.1 raises).
  - **#2** — the sandwich guard was REPLACED, and a prior belief CORRECTED (see
    the per-beam-profiles bullet below).
  Coverage gaps — **all 4 now CLOSED by tests (2026-07-23, 4 parallel agents,
  all GREEN, ZERO production bugs — the feedback was sound, only tests missing):**
  - **δω_rf × CR** — `..._matches_mu_plus_with_delta_omega_rf` (test_mtw): lone CR
    µ⁻ + offset reproduces co µ⁺ bit-for-bit (single beam tracks forward; slip
    anchor is direction-agnostic, only signed charge differs = +1 for both).
  - **accel × two-beam CR** — `TestTwoBeamAcceleratingOffsetPassages`
    (test_two_beam): fast-ramp two-beam vs retuning convolution, 0.13%→0.025%
    (error shrinks = non-ramping = healthy composition of frame-slip × reverse
    traversal).
  - **δω_rf × two-beam CR (the reverse-traversal case, added 2026-07-24)** —
    `TestTwoBeamDeltaOmegaRfOffsetPassages` (test_two_beam): static cycle,
    δω_rf=2000, feedback vs retuning convolution (delta_f=δω_rf/2π),
    0.13%→0.02% (shrinks). Reverse-stream slip anchor is direction-correct;
    a crossed probe (offset dropped for the reverse stream) grows to 9%/turn,
    18× over gate — so the test is strongly discriminating. Closes the last
    residual (δω_rf × two-beam, which neither gap 3 lone-beam nor gap 2 accel
    covered).
  - **generator↔beam power** — `test_generator_power_conservation.py` (new):
    phasor balance closes to 1 WITH the SSB factor-2, 0.5 with raw I_beam (both
    pinned); current_limit_from_power ⇄ generator_power round-trips 1e-12.
    Doc-clarity note: beam_current.py:209 "factor 2 recovers the fundamental"
    should state the units bridge (physical fundamental = I_beam/2 in
    generator-current units) — not a bug.
  - **closed-loop Robinson** — `test_closed_loop_stability.py` (new, test-only,
    300 turns ≈30 Q_s periods): nominal net-DAMPS (−9e-4/turn, genuinely
    stable), perturbed (flipped detuning) grows (+1.65e-3/turn). 3 new
    characterizations: (a) loop gain/delay dynamically inert on the dipole
    (loop ≫ synchrotron rate → only detuning-sign destabilizes); (b) feedback
    fully suppresses *linear* Robinson (needs a filamenting bunch); (c) slow
    secular drift common to both signs beyond ~350 turns (matches the known
    bounded-secular-drift limitation).

- ~~MultiPole vs MultiPass on missing R_CR~~ **RESOLVED**: the follow-up
  session's guard landed via `origin/blonder` (commit `2235e519`, merged in
  `b047e972`) — MultiPoleSparseSolve now raises on a CR beam with any source
  missing R_CR. NOTE: that merge also produced one **semantic conflict** in
  `test_sources.py::test_get_impedance` (origin's `−R` construction ×
  our negating `get_impedance` kept by the merge ⇒ sign flip); fixed by
  restoring the `+R` construction matching the surviving convention.
- **`barrier_bucket.py` CR kick** — **user decision (2026-07-24): IGNORE.**
  (line ~245 raw charge; wrong-sign barrier kick for a CR beam; out of
  feedback scope, deliberately not pursued.)
- **DRIVEN MULTI-SECTION FAST-RAMP FRAME SLIP — FIXED (2026-07-24, root
  cause).** Was: `|V_ant|` drifted ~3% over 5 turns (2 sections, fast ramp,
  driven); superlinear, diverging. **Root cause (measured, NOT the assumed
  geometry bug — seed mis-registration is 1e-6 t_rf/seam, four orders too
  small):** a multi-section passage builds its grid piecewise, accumulating
  RF phase `Σ_k ω_k·T_k`, while the demodulation (`omega_c =
  _forward_tracking_omega_rf`) and the readout both reference the single
  carrier `ω_0` — a *carrier-phase bookkeeping* mismatch
  `Ψ = Σ_k (ω_k − ω_0)·T_k`, identically 0 for one section (which is why
  single-section never needed a correction). The old code applied `Ψ` as a
  ROTATION OF THE ANTENNA-VOLTAGE STATE (cavity_feedback.py:2161-2177),
  which also hit the generator-driven field; that field carries no
  registration error (re-injected on the current grid every cell), so the
  constant drive fought the rotating state and a phase error became an
  AMPLITUDE drift. **Fix:** `Ψ` accumulates into `_grid_carrier_phase`,
  folded into `_carrier_slip_gap` → subtracted at demodulation
  (`carrier_phase_offset`), added back at readout (`phase_correction`) —
  the same phase idiom the RF-frequency offset uses, and what the
  design-clock invariant prescribes. State (hence driven steady state) left
  untouched; the state rotation is DELETED. `rf_center_grid.py`,
  `envelope_kernel.py`, `rf_center_segment.py` untouched — no geometry or
  kernel change needed. **Proof it is a real fix, not a compensation:** the
  5 mtw tests that failed when the rotation was merely removed now pass
  WITHOUT it. Bit-identity (SHA-256 over full V_ant grids) for
  single-section and no-ramp. `TestPIFullTrackingMultiSectionSlowRamp` PIN
  regenerated (it encoded the old rotated state; behavioural assertions
  unchanged). **Stretch achieved:** the fast-ramp exclusion is LIFTED —
  new `TestPIFullTrackingMultiSectionFastRamp` (setpoint restored to 1e-16
  relative per turn). **Residual caveat:** `Ψ` still reaches the beam via
  `phase_correction`, so a driven multi-section fast ramp keeps a readout-
  PHASE offset (the beam-induced part needs `Ψ` at readout, the driven part
  does not; one readout phase cannot separate them). Amplitude is exact.
- **2026-07-24 follow-up fixes (3 parallel agents, all green — 73 passed
  combined):**
  - **Guard message accuracy (#2 review finding)** — both
    `_check_two_beam_profile_placement` raise messages corrected: no more
    false "never histogrammed" claim; they now state the conservative
    rationale truthfully (replay verifies over ring-element writes only;
    self-histogramming consumers write atomically but invisibly to the
    check; feedbacks additionally entangle cross-turn state). Logic
    untouched; `test_untracked_live_profile_raises` now pins the accuracy
    fix (`assertNotIn("never histogrammed")`).
  - **`|R_CR| == |R|` validation (review gap)** — converted from a bare
    `assert` (stripped under `python -O`) to `raise ValueError` in
    `Resonators.__init__`; new mismatch test
    `test___init__counter_witness_magnitude_mismatch` (TDD RED captured:
    AssertionError vs ValueError). NOTE: constructor now mixes
    assert/RuntimeError/ValueError styles — future cleanup, would break
    existing assertRaises expectations.
  - **Exponential solver end-to-end (review gap)** —
    `TestExponentialSolverEndToEnd` in test_mtw: (1) standard-Q_L
    composition pin vs convolution (2.9e-3, gate 0.02; exp-vs-Euler-fb
    ≤8.5e-7); (2) low-Q_L=32 absolute pin (1.8e-2, gate 0.03) — NOTE the
    empirical finding: at n=1 low-Q_L the propagators do NOT differ on this
    observable (both floored by the O(1/(2Q_L)) IQ-envelope truncation ≈
    1.6% — an inherent model limitation, not a solver defect); (3) the
    REAL discriminator: large detuning (δω=3.5e6 rad/s, θ=2.7e-3/step,
    below Euler's own warn threshold) — exp stays at 1.4e-3-3e-3 vs Euler
    compounding to 6.7e-2/1.3e-1 (38×/98×), mutation-verified (flag off →
    6.7× over gate). Harness kwargs kept OUT of the comparison cache key
    (counter_rotating precedent).
  - **Public kwarg is now `exponential_coarse_solver_enable`** (user's
    second rename); all docs/RSTs/context/test-harness names aligned — the
    documented call constructs.
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
  use frozen profiles) — core gap. **CORRECTION (2026-07-23, #2):** the item-7
  belief that a *(profile, consumer, profile) sandwich is sufficient* is WRONG.
  The exact-interleave replay proves even the minimal sandwich `[P, C, P]` is
  corrupt (the *forward* beam reads the counter's histogram), and `[P,C,P,Drift]`
  corrupts the counter beam. A padded layout (e.g. `PCPDD`, `DPCPDDD`) can be
  safe, but no simple positional rule characterises it. The guard was therefore
  REPLACED: `_check_two_beam_profile_placement` now **replays the exact mainloop
  interleave** (forward tracks `elements[k]`, then counter `elements[N-1-k]`;
  two turns for steady state) and raises if any consumer reads the other beam's
  histogram — 0 too-lax / 0 too-strict over 1792 layouts **under the
  pure-reader model** the replay assumes. CAVEAT (review 2026-07-24, #1): the
  real mucol consumers (WakeField with `track_profile=True`, and the feedback's
  `calculate_rf_beam_current_partial`) SELF-histogram their profile in place
  before consuming it, so the write+read is atomic per beam and no interleave
  corrupts it. The replay models consumers as pure readers and does NOT see
  those self-writes, so it is *conservative*: it rejects the natural
  attached-live-profile layout with the (inaccurate) message "never tracked as
  a ring element … never histogrammed". No supported config is wrong (the
  shipped two-beam path uses frozen profiles), but the guard message + the
  "provably correct" framing overstate; the real long-term fix remains per-beam
  profile instances. Frozen (`active=False`) profiles remain exempt. Tests in
  `test_simulation.py::TestTwoBeamProfilePlacementCheck` (the old
  `test_sandwiched_live_profile_passes` was corrected to
  `test_minimal_sandwich_rejected`). The real long-term fix remains per-beam
  profile instances; this guard turns silent corruption into a loud error.
- ~~`delta_omega_rf` lab-frame demod slip~~ **RESOLVED (2026-07-22, task
  9)**: redesigned — the coarse-grid geometry is fully on the *design* RF
  clock (`forward_tracking_omega_rf` design-only; no detuned spacing; the
  former `phase_offset_frwrd`/`_next` slip attributes are removed entirely,
  and BOTH the forward and reverse paths seed at the constant `t_rf / 2` —
  the `phi_rf` parameter of `_generate_rf_centers` and the
  `_get_time_to_next_rising_edge_zero` helper are deleted, resolving the
  reverse path's per-segment-phase TODO; the multi-section 2e3 rad/s
  offset run is measured IDENTICAL to the δω=0 baseline per turn), and
  the offset enters only as explicit phases: the demodulation carrier
  itself stays on the *design* clock
  (`omega_c=self._forward_tracking_omega_rf`, from
  `calc_omega_rf_design`) — the `forward_carrier_omega_rf` (+δω,
  window-local) attribute this bullet used to name was deleted, see the
  #1/#5 bullet above — and the offset rides only on the accumulated slip
  `−(parent.delta_phi_rf + live gap)` on the demodulation with the same
  total re-applied at readout (station clock via `phi_rf`, live gap added
  to `phase_correction`). The live gap `δω·(t_now −
  station._last_reference_time_phase_slip)` compensates the kick clock's
  end-of-track lag (blond2 convention, untouchable). Measured: net
  carrier-phase error vs the retuning convolution ≤ 2e-5 rad/turn at 8e2
  and 2e3 rad/s (was ~2 %/turn per 1e3 rad/s). Four tests
  (`test_multiturn_delta_omega_rf_{large_offset_consistency,differential,
  substepped,multisection}`), mutation-verified; grid geometry tests
  updated to design-clock expectations; docs (design RST + test inventory)
  updated. Diagnosis was empirical: a per-turn linear-response solve over
  free demod phases proved the residual error was a whole-envelope readout
  frame drift from the slipping grid, not a per-deposit demod error —
  hence the geometry redesign rather than a phase patch.
- ~~`experimental/physics/feedbacks/helpers.py` still uses raw charge~~
  **MOOT (verified 2026-08-08)**: `blond/experimental/physics/feedbacks/`
  no longer exists (the tree now holds only `kick_pooling.py`), so there is
  no experimental `rf_beam_current` copy left to worry about.
- **P6** (RF-parameter view mixin) skipped per user.
- **Full Sphinx doc build not yet run**: only docutils structure-lint was run on
  the two RSTs. CI builds with `sphinx-build -W` (warnings = errors, see
  CLAUDE.md); run `cd docs && bash create_docs.sh` (sequentially, never looped)
  before the MR. No new top-level exports were added, so
  `ASSIGNED_CATEGORIES` needs no update.
- The extracted mixins (`RFCenterGridMixin`, `GeneratorRegulationMixin`) are
  pure moves (methods still use `self`); promoting them to composed collaborators
  is the natural follow-up.
- **LHC-suite npz references vs the large-files hook** (user decision): the
  five pinned `resources/*_blond2_reference.npz` total ~61 MB and three
  exceed pre-commit's `check-added-large-files --maxkb=5000`
  (transfer_function 10.8, full_machine 15.1, phase_error 30.4 MB).
  Options: git-lfs, raise `maxkb`, or don't commit the large ones — the
  loader self-heals by regenerating on first run (at the old one-time
  cost per machine).
- **LHC suite CI visibility** (user decision): `pyproject.toml`
  `testpaths = ["tests/unittests", "tests/integration"]` means CI never
  collects the repo-root `unittests/` tree (LHC + mucol suites). With
  pinning the LHC suite is now cheap enough to consider moving under
  `tests/unittests/` (behind a marker) or adding a dedicated CI job.
- **pytest-xdist** for the comparison directory (follow-up, after the CI
  decision): with blond2 pinned, `-n auto --dist loadscope` would run the
  five classes' blond3 sides in parallel processes.

---

## 4. Module layout after the P1–P5 partition

`blond/physics/feedbacks/`:

| module | holds |
|---|---|
| `cavity_feedback.py` | `IQCavityFeedbackBase` + `IQCavityFeedbackTimingClass(IQCavityFeedbackBase, RFCenterGridMixin, GeneratorRegulationMixin)`; orchestration (`_track` + its nine phase methods — see §2.11 —, `circuit_track`, `_circuit_track_cells{,_python,_kernel}` + `_coarse_step_sizes`/`_kernel_*` glue, `cavity_response`, `_advance_coarse_voltage`, `cavity_response_fine`, `calculate_rf_beam_current_partial`, `reset_arrays`, `on_run_simulation`, pre-fill). `_check_step_sizes`, `_check_beam_kick_magnitude`, `_check_beam_kicks` remain here as thin wrappers delegating to `self._euler_guard` |
| `envelope_kernel.py` | numba host kernel `envelope_pi_scan` + `inactive_controller_scan_state` — the sequential coarse-envelope + PI recursion (performance item #1); solver-agnostic, byte-identical to the Python reference. Reached through the controller's `supports_envelope_scan` capability (2026-08-06), not called by the feedback directly |
| `rf_center_segment.py` | `RFCenterSegment` value class (re-exported from cavity_feedback for compat). `.centers` and `.residual` are consumed; `.omega`/`.duration` are validated only (see §2.11) |
| `rf_center_grid.py` | `RFCenterGridMixin` — coarse `rf_centers` grid construction (forward/reverse reference walks, segment generation, `_append_segment`/`_clear_segments`/`_rebuild_grid_arrays`, `_preceding_segment_residual`, derived arrays). `_segments` is the single source of truth; the flat arrays are derived |
| `generator_regulation.py` | `GeneratorRegulationMixin` — `_controller_active`, `pi_setpoint`, `generator_power`, `_update_generator_current` |
| `cavity_solvers.py` | **mucol-only** `cavity_response_sparse_matrix` (first-order/forward-Euler) + `cavity_response_sparse_matrix_second_order` (Crank-Nicolson) + `pretrack_fill_voltage` + `ForwardEulerValidityGuard` (moved here 2026-08-07 — the discretisation tripwires, beside the solvers they certify) |
| ~~`helpers.py`~~ | DELETED (2026-07-25): `cavity_response_sparse_matrix` moved into `cavity_solvers.py`; re-export shims already gone |
| `beam_current.py` | `low_pass_filter`, `rf_beam_current` (unified 2026-07-25: single function, keyword-only coarse args; 2026-08-07: coarse-write wrap-around removed, `ProfileBaseClass.check_fits_in_span` called instead) |
| `iq.py` | `cartesian_to_polar`, `polar_to_cartesian` |
| `generator_current_controller.py` | `GeneratorCurrentController` ABC (2026-08-06) + `GeneratorCurrentPIController`; `current_limit_from_power`, `clamp_magnitude` |
| `base.py` | `FeedbackBaseClass` / `LocalFeedback` / `GlobalFeedback` (unchanged) |

**Re-exports**: none — `helpers.py` was deleted 2026-07-25 together with its
`# noqa: F401` shims. The only other `rf_beam_current` in the tree is
`blond/legacy/blond2/llrf/signal_processing.py`, which is self-contained and
out of scope. Mucol production + tests import from the canonical modules
above.

**Test split** (`tests/unittests/physics/feedbacks/`):
`test_rf_center_grid.py` (was `TestIQCavityFeedbackTimingClass`),
`test_rf_center_segment.py` (was `TestRFCenterSegment`); `test_cavity_feedback.py`
reduced to the empty `TestIQCavityFeedbackObservationClass` stub. The unused
debug method `plot_antenna_voltage` moved to
`tests/unittests/physics/feedbacks/accelerators/mucol/plotting.py` as a function.

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

## 6. Commit status

**HISTORICAL — superseded 2026-08-07.** This section used to read "Nothing is
committed" and proposed a commit grouping ((a) review fixes + coverage tests,
(b) CR-1…CR-4 + convention/tests + docs, (c) the P1–P5 module partition,
(d) the `DEBUG_PLOT` fix). That grouping was never used: the work was
checkpoint-committed incrementally on `blonder_feature/mucol_feedbacks`
instead, and `blonder` has since been merged in (`52a03664`). Everything
described above is committed; check `git log` rather than this section for
the current state. Re-run the full battery before/after any reshuffle.
