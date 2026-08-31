# Muon-Collider Cavity Feedback — Work Context & Handoff

Working notes for the mucol cavity-feedback branch
(`blonder_feature/mucol_feedbacks`), July–August 2026. Checkpoint-committed
throughout; last full re-verification of this file against the source
2026-08-11; docs-consistency sweep (this file + both RSTs against the
source, scripted name/role/test-name checks) 2026-08-12; split-envelope
docs pass (this file + both RSTs + `observables.py` + the outer-repo
example, see §2.13) 2026-08-12. Still to be reviewed before the MR.

**What this file is.** A maintainer's handoff map: the invariants, the
decisions (including the rejected ones and the user directives behind them),
the module layout, and the open questions. It is kept deliberately and it is
deliberately *not* published documentation.

**What it is not.** It is not the design doc and it is not the test
inventory. Those are:

- `docs/feedbacks/mucol_cavity_feedback.rst` — design. Sections: *Concepts
  and notation*, *Classes at a glance*, *Signal path of one turn*, *Initial
  conditions and cavity pre-fill*, *Interplay with the RF station*,
  *Counter-rotating beams*, *Validation*, *Known limitations*.
- `docs/tests/mucol_cavity_feedback_tests.rst` — test inventory. Sections:
  *Common physics context*, *Test modules*, *Shared feedback-machinery
  tests*, *Guards tested outside the feedbacks tree*, *Support modules*,
  *Data and assets*, *Running the tests*.

Where this file would only restate them, it points instead.

**Precedence.** The code wins over all three. This file has asserted the
*opposite* of the code on a central invariant before (see the corrections
logged in §2.11 and §3), so read the source, not the note. Every claim below
was re-checked against the source on 2026-08-11 unless it is marked
**HISTORY** — history bullets record *what was decided and when*, not what
the code does today.

---

## 0. Scope, constraints, and how to work in this repo

**Scope**: muon-collider (mucol) cavity feedbacks and their base classes.
Shared code (impedance solvers, `rf_beam_current`, `profiles.py`,
`cavities.py`) was touched only where explicitly authorised (the
counter-rotating work, the unified span guard, the multi-harmonic slot
check).

**Hard invariants (must always hold):**

- **n = 1 / single-beam path bit-identical**: a single co-rotating beam must
  produce bit-identical results before/after any change; a single
  counter-rotating µ⁻ beam must reproduce the co-rotating µ⁺ run bit-for-bit.
- **Design-clock grid**: the coarse `rf_centers` geometry is a pure function
  of `calc_omega_rf_design` and the reference times. Frequency and phase
  offsets (`delta_omega_rf`, `phi_rf_design`, the multi-section registration
  phase Ψ) enter **only** as demodulation/readout phases, never as grid
  geometry, and never as a rotation of the antenna-voltage state. Canonical
  statement: the `IQCavityFeedbackTimingClass` class docstring (*RF-frequency
  offset* under Notes) and the design RST's *Interplay with the RF station*.
  Since the envelope split (§2.13) the propagated state is the two
  source-split components; the offsets reach the generator-sourced one only
  through the per-passage *composition* rotation of the demod-frame sum,
  still never through its propagated state.
- **`_segments` is the single source of truth** for the per-turn grid; the
  flat `_rf_centers` / `_rf_centers_lengths` arrays are derived from it
  (`_rebuild_grid_arrays`) and can therefore not desync.
- **Every segment holds ≥ 2 coarse centres** (`RFCenterSegment.__post_init__`).
  Three separate gates rely on it — see §2.12(b).
- **`harmonic_index` == list slot** on a `MultiHarmonicRFStation` — see
  §2.12(a).
- `IQCavityFeedbackBase` was SLIMMED, not dissolved: the name is kept because
  `@requires(["IQCavityFeedbackBase"])` in `handle_results/observables.py`
  string-matches the MRO. `n_cavities` is legally `int | float` (a fractional
  effective-voltage scale) and must NOT be int-coerced.

**HISTORY — invariants that were retired:**

- ~~LHC path frozen~~ **OBSOLETE (2026-07-25)**: the LHC/SPS cavity feedbacks
  and the blond2 comparison suite were REMOVED from the codebase (the phase
  loop survived — moved to `blond/physics/feedbacks/beam_feedback.py`). The
  byte-identical obligation and its bridge machinery (`dT_index_sign`,
  `coarse_center_offset`, the helpers re-export shims) were stripped in the
  same cleanup. `blond/legacy/blond2/` keeps its own self-contained copies.
  Verified 2026-08-11: `tests/.../accelerators/lhc/` holds only
  `test_beam_feedback.py`; no `comparison_with_blond2/` directory and no
  `*_blond2_reference.npz` exist anywhere in `tests/`.
- ~~Feedback splits~~ **MERGED (2026-07-25, user-approved unification)**:
  `rf_beam_current_partial` was folded into the single `rf_beam_current`
  (keyword-only coarse args `sampling_time`/`n_points`; offset always
  `sampling_time/2`; `external_reference`/`downsample`/`T_rev` removed).
  Dead base members deleted (base `on_run_simulation`/`_track`/
  `track_no_beam`/`calculate_rf_beam_current`/`set_point_from_rfstation`/
  `update_feedback_variables`/`omega_carrier`/`residual_time_shift`/`t_rf`/
  HasPropertyCache machinery/`n_samples_coarse`/`use_lowpass_filter`); the
  timing override now carries its OWN `@requires` decorator (regression test
  `test_cavity_feedback_requires.py`). `helpers.py` was DELETED —
  `cavity_response_sparse_matrix` lives in `cavity_solvers.py`.
- ~~`harmonic_index=1` hardcode preserved + flagged as suspicious~~
  **SUPERSEDED**: `harmonic_index` is now a real constructor parameter
  (default 0) with multi-harmonic support; see §2.12(a).

**Environment / gotchas:**

- Run pytest from `BLonD/` with `MPLBACKEND=Agg`. The venv is the **outer**
  repo's: `../.venv/Scripts/python.exe` (there is no `BLonD/.venv`).
- The pre-commit `check copyright` (`custom-py-check`) hook is **broken on
  this machine** (always fails, `WinError 3`); ignore it, trust the other
  hooks (ruff, isort, numpydoc). Module-docstring summary must start on the
  line **after** the opening `"""` (numpydoc GL01 convention in this repo).
- All mucol test files gate debug plotting on `DEBUG_PLOT = False`; never
  leave it `True` (a guarded `plt.show()` would fire). Verified 2026-08-11.
  Out of scope but worth knowing: the non-mucol helper
  `tests/unittests/physics/impedances/comparisons/mtw.py` hardcodes
  `DEBUG_PLOTTING = True` inside a helper function.
- Observation tests write `last_*.npy` / `last_*.json` into the CWD (now git-ignored at the repo root) — see the
  open item in §3.

---

## 1. Conventions and decisions the published docs do not own

The *physics* of §1.1/§1.2 is stated in the design RST's *Counter-rotating
beams*. What follows is the decision record behind it: why the conventions
are what they are, and what was rejected.

### 1.1 Direction-signed charge (counter-rotating beams)

`beam.signed_charge_with_direction()` (`blond/core/beam/base.py`, returns
`particle_type.charge * -1` for a counter-rotating beam) is used on **every
source-current site and every kick**. Verified sites (2026-08-11):

- source: `feedbacks/beam_current.py` (`rf_beam_current`),
  `impedances/base.py` `WakeField` deposit, and the three solver deposits in
  `impedances/solvers.py` (SingleTurnResonatorConvolution, MultiPassResonator,
  MultiPoleSparse);
- kicks: `physics/cavities.py` — five sites: `_track_interp` and both
  `_track_no_interp` overrides, plus the two analytic
  `calc_phi_s_main_harmonic` / `calc_synchrotron_tune_main_harmonic` (the
  **tune is sign-robust** — it uses `|charge|` and `|cos φ_s|` — so only
  `phi_s` changes value for a CR µ⁻ beam); `impedances/base.py`
  `WakeField._track`; `experimental/physics/kick_pooling.py`;
- test double: `blond/testing/mocks.py` patches it onto `beam_mock`.

**Why the deposits had to change (not just the labels).** The kick side was
*already* signed before this work (shared with the design RF kick,
immovable). Raw deposits × signed kick meant a µ⁻ counter-rotating beam's
self-wake **accelerated** it (measured `dE = −(µ⁺co)`). Signing the deposits
was the only fix. For a co-rotating beam signed == raw, so those paths are
bit-unchanged.

`blond/physics/barrier_bucket.py` still kicks with the raw
`beam.particle_type.charge` — **user decision (2026-07-24): IGNORE.** Out of
feedback scope; still true on 2026-08-11.

### 1.2 `shunt_impedances_counter_witness` (R_CR) convention

The sign table and the field-symmetry rationale are in the design RST
(*Counter-rotating beams*). Decision record only:

- Public kwarg renamed from `shunt_impedances_counter_rotating` (2026-07-22,
  **with** a sign-convention flip). The old name is a **trapped kwarg** that
  raises `TypeError` with a migration message — deliberately not a
  pass-through alias, because the meaning changed with the name.
- The internal attribute is `_shunt_impedances_counter_witness` (user
  directive); the MultiPole solver guard's `getattr` string, `EX_28` and the
  direct test accesses were swept together, so no old-name attribute
  survives.
- `|R_CR|` must equal `|R|` — enforced in `Resonators.__init__` as a
  `raise ValueError` (converted from a bare `assert`, which `python -O`
  strips). Only the sign is free.
- A single beam / self-wake / same-direction interaction **never consults
  R_CR** (XOR wake selection), so single-turn / single-beam behaviour is
  independent of it. Pinned by
  `test_single_beam_never_consults_the_counter_rotating_shunt`.
- Closed form for two counter-rotating passages offset by Δ on the ringing
  tail: `v₂ = (s₂ − F·g)·v₁`, with `s₂` the signed charge of the CR beam,
  `F = sign(R_CR/R)` and `g = exp(−ωΔ/2Q)`. Build when `s₂F = −1`, cancel
  when `s₂F = +1`.

### 1.3 "backfill" (time) vs "reverse" (space) — naming rule

The walk that reconstructs the stretch of grid **already elapsed since this
feedback's previous passage** is called **BACKFILL**. It is a *time*
direction, and **every multi-section ring needs it**, whichever way its beam
goes.

"**Reverse**" is reserved for the *space* direction: a counter-rotating beam
meets the ring's reference-altering elements in the reversed order.

The distinction is the whole point of the rename: a reviewer reading the old
names cold concluded the walk existed *for counter-rotating beams*. Canonical
statement in source: the "Two independent notions share the word 'backwards'"
paragraph of `rf_center_grid.py`'s module docstring.

Renames applied (no old name survives in `blond/` or `tests/`):

| old (time-sense "reverse") | new |
|---|---|
| `calculate_rf_centers_for_reverse_direction` | `calculate_rf_centers_for_backfill` |
| `get_time_omega_array_reverse_direction` | `get_time_omega_array_backfill` |
| `_replay_reverse_span` | `_replay_backfill_span` |
| `_reverse_tracking_time_array` | `_backfill_time_array` |
| `_reverse_tracking_omega_list` | `_backfill_segment_omega_design_list` |
| `PerTurnGridSpan.n_reverse_centers` | `.n_backfill_centers` |
| `residual_from_reverse_span` | `residual_from_backfill_span` |
| `_generate_reverse_segments_if_due` | `_generate_backfill_segments_if_due` |
| `_unify_same_frequency_time_points_reverse` | `..._backfill` |
| `_forward_tracking_omega_rf` | `_forward_segment_omega_design` |
| `_last_forward_tracking_freq` | `_last_segment_omega_design` |
| `_generate_rf_centers(omega_rf=)` | `(omega_design=)` |
| `reset_arrays(n_reverse_cells=)` | `(n_backfill_cells=)` |

**Space-sense "reverse" deliberately KEPT** (all in `cavity_feedback.py`
`__init__` / `on_run_simulation` and `rf_center_grid.py`):
`_reference_altering_elements_reverse`, `_own_index_in_reference_list_reverse`,
`reference_index_until_tracked_reverse`, and the two selector helpers that
dispatch on them, `_reference_list_for_direction` /
`_own_index_for_direction`.

**Stragglers — RESOLVED (verified by scripted scan 2026-08-12):** the two
known stragglers are gone. `envelope_kernel.py` carries no time-sense
"reverse" anymore (`inactive_controller_scan_state`'s docstring now says
"no-beam **backfill** segment"), and neither RST contains any pre-rename
production name — both use the backfill vocabulary throughout, and every
remaining "reverse" in them is the space sense (counter-rotating element
traversal) or a genuine `test_get_slice_..._reverse` test name.

### 1.4 One quantity, one name: `omega_times_dt`

The RF phase advanced in one step [rad] is spelled `omega_times_dt`
**everywhere** in the mucol feedback. Two earlier spellings were retired:
`samples_per_rf` (which asserted the *reciprocal* of what it held — callers
pass ω·dt ≈ 0.06 rad, not a sample count) and `omega_times_T_s` (which
collided with `T_s` = *synchrotron* period). Rationale is recorded in the
*On naming* paragraph of `cavity_solvers.py`'s module docstring; do not
reintroduce a synonym.

---

## 2. Work completed (by theme)

**HISTORY** unless noted. For *what the code does*, read the design RST; for
*what is tested*, read the test-inventory RST. These bullets exist to record
decisions, root causes and rejected alternatives.

### 2.1 Review-driven cleanup & bug fixes (earliest pass)

`PassiveCavity` deleted after porting its pre-fill capability into
`IQCavityFeedbackTimingClass`; `"yorak"` placeholders removed; `±π/2`
demodulation convention verified; base-class `np.floor→int` crash fixed;
`voltage_setpoint` read-only-property bug fixed; multi-station
`delta_omega_rf` guard added; `delta_omega_rf` phase-slip reworked to
elapsed-reference-time (**must stay at the END of `_track`** — still true).
Five production bugs were found *by* the test-hardening campaign: substepped
demod sign flip, LHC centering convention, multi-section frame drift, stale
backfill re-pass, bistable demod residual.

### 2.2 P1 — exact exponential coarse propagator (option)

`exponential_coarse_solver_enable: bool = False`. `cavity_response` routes
through `_advance_coarse_voltage`, which does either forward-Euler (default,
**bit-unchanged**) or the exact `V_{n+1} = e^L V_n + src·(e^L−1)/L`. Under
pure detuning the exponential step preserves `|V|` (a rotation) where Euler
grows it by `√(1+(δω·dt)²)`.

The step-size / beam-kick guards must not gate the exact solver (they were
previously unreachable in the low-`Q_L` / large-detuning regime the option
exists for). Since 2026-08-07 that is expressed once, as a constructor flag:
`self._euler_guard = ForwardEulerValidityGuard(enabled=not
exponential_coarse_solver_enable)` — not as hand-written early returns.

Public kwarg was renamed to `exponential_coarse_solver_enable` (user's second
rename); all docs/tests aligned.

### 2.2b P6 — numba coarse-envelope kernel (performance item #1, DONE)

The per-cell coarse recursion (~10⁵ cells/turn, ~95 % interpreter overhead)
is compiled to a numba **host** kernel `envelope_pi_scan`
(`envelope_kernel.py`). Host-only by design (sequential signal processing —
no GPU parallel scan) and **on by default**
(`use_numba_envelope_kernel: bool = True`, a class attribute; set `False` per
instance to force the reference). Measured ~79× on a 1000-cell RCS segment.

`_circuit_track_cells` dispatches to `_circuit_track_cells_python`
(byte-identical reference **and** fallback) or `_circuit_track_cells_kernel`.
The glue precomputes the *state-independent* per-cell voltage multiplier `B`
(`1+L` Euler / `e^L` exponential) and drive weight `W` (`1` / `(e^L−1)/L`),
so the kernel is **solver-agnostic** and byte-identical to *both* solvers
without numba ever evaluating `exp`/`expm1`.

**CORRECTION (2026-08-11).** The controller marshalling this section used to
name (`_kernel_controller_params` / `_store_controller_state` on the
feedback) no longer exists. Since the controller separation (§2.11) the
kernel glue calls the *controller's* own
`envelope_scan_kernel()` / `envelope_scan_state()` /
`absorb_envelope_scan_state()`. Step sizes (`_coarse_step_sizes`), the
per-cell multipliers (`_kernel_step_multipliers`) and the beam current
(`_kernel_beam_current`) are still marshalled per segment by the feedback.

Two exact-fallback paths keep it byte-identical:

- **Coincident (zero) coarse step** → the Python reference (skip-and-warn
  can't vectorise).
- **Klystron-limit saturation** → the kernel flags any cell within a 1e-9
  guard band of `max_output` and the segment reruns on the reference path,
  because numba's complex `abs` differs from numpy's *scalar* `np.abs` by
  1 ULP (~40 % of values). When no cell nears the limit the clamp is never
  applied → identical.

`_check_beam_kick_magnitude` runs as a vectorised post-pass
(`_check_beam_kicks`) that *delegates* to the per-cell checker for message
fidelity + warn-then-raise ordering.

**Adversarial review found & fixed 3 carried-state divergences the first
tests missed** (they seeded `last_val_beam_current=0` and
`last_val_generator_current=bias` — the values that *hide* the bugs), all in
the carried index-0 cell of a `no_beam` (backfill) segment starting at grid
index 0, i.e. the **first backfill segment of a multi-section ring on turn
≥ 1**:

1. **Generator-current drive** (HIGH): the kernel held
   `generator_current_init` for every cell; the reference uses
   `last_val_generator_current` only at cell 0 and the static grid at
   `idx-1` for cells ≥ 1. Fix: the kernel reads each cell's drive from the
   pre-filled `generator_current_out[cell-1]`.
2. **Beam current at cell 0** (HIGH): `_kernel_beam_current` zeroed cell 0
   for `no_beam`; the reference `idx==0` branch uses
   `last_val_beam_current` unconditionally. Fix: set it before the `no_beam`
   early return.
3. **Warn/assert ordering on an invalid grid** (LOW): `_coarse_step_sizes`
   defers *any* non-positive step to the reference loop instead of a
   pre-emptive vectorised assert.

REFUTED by that review: float32-grid NEP-50 (unreachable), hard-kick
post-raise state (simulation aborts), the test-coverage observations (since
covered).

### 2.3 P2 — PI on the forward passage only

The controller is stepped only on the real forward passage
(`if self._controller_active and not no_beam`), never on the backfill
reconstruction segments (which carry a per-segment frame phase). Still true
2026-08-11, in `cavity_response`. The structural call-count tests must set
`use_numba_envelope_kernel=False` — the kernel inlines the control law, so
there is no call to count.

### 2.4–2.7 Coverage, audits, counter-rotating work, five-dimension review

Superseded as an inventory: see the test RST (*Test modules*) for what each
file covers, and the design RST (*Validation*) for what the model is
certified against. What survives as a decision record:

- **CR-3 two-beam.** Offset passages work at reference accuracy; the
  **meeting-azimuth / simultaneous** case is refused with
  `NotImplementedError` in `_guard_simultaneous_passage`, and the equal-time
  patch path is **deferred by user decision**.
- **CR-4 sign flip.** Three negations in `sources.py`
  (`get_wake_counter_rotation`, its quadrature, `get_vectorfit`) plus the
  `get_impedance` counter-rotating branch (found in review — it was
  inconsistent with the wake path). All consumers inherit; deposits, kicks,
  kernels and the feedback were untouched.
- **MultiPole cannot take two coincident same-time passes** (sequential state
  machine), so the 2×2 cross-solver matrix uses a 2·t_rf offset.
- **`samples_per_rf` docstring was 2π-wrong** (it is ω·dt, not dt·f) — fixed,
  and the name has since been retired entirely (§1.4).
- Every production sign/gate change is **mutation-verified** (reverting the
  fix fails the pinning test): P2 gate, exponential branch, T7 xfail reason,
  CR-1/CR-2 sign matrices, CR-4 matrix, MultiPole deposit.

### 2.8 Module partition (P1–P5, behaviour-preserving)

See §4.

### 2.9 Misc

`DEBUG_PLOT` in `test_mucol_cav_fdbk.py` was `True` (the last stray) →
`False`. The `shunt_impedances_counter_rotating` → `..._counter_witness`
sweep (see §1.2) also fixed a unit typo `\omega` → `\Omega` on both shunt
docstrings.

### 2.10 LHC blond2-comparison suite — speed refactor

**HISTORY, and now moot.** The suite (pinned `.npz` references via
`support.blond2_reference`, `BLOND_REGEN_BLOND2_REFERENCE=1`, the two
approved physics tightenings: `open_tuner=True` on the phase-error
commissioning and wrapped-difference `angle(H3·conj(H2))` phase assertions
replacing three vacuous `atol=7` ones) **no longer exists in this tree** —
verified 2026-08-11. It went with the LHC/SPS feedback purge (§0). The three
follow-up items it generated (npz vs the large-files hook, CI visibility of
the repo-root `unittests/` tree, pytest-xdist over the comparison directory)
are therefore **MOOT** and have been dropped from §3.

### 2.11 Structural pass, 2026-08-06/07

Committed: `46d9d989`, `db86a65b`, `d2ce9d19`, `9b870c1b`, `a30e8acc`.
Behaviour-preserving except where noted.

- **`_track` decomposed into named per-turn phases.** `_track` does no work
  itself — it only names the phases in order, and where a phase depends on a
  value another produced, that value is *returned and passed* rather than
  left on `self`, so the argument lists are the dependency graph.
  **CORRECTED 2026-08-11:** this section used to list *nine* phases including
  `_close_previous_turn_grid` and `_write_debug_readout`.
  **CORRECTED AGAIN 2026-08-12 (split envelope, §2.13):** `_track` now calls
  **nine**, in this order: `_guard_simultaneous_passage`,
  `_carrier_slip_gap_at_passage` (assigned to `_kick_clock_slip_gap` — reset
  per passage), `_rebuild_per_turn_grid` (returns a
  `PerTurnGridSpan`; it is the one that calls `_close_previous_turn_grid`,
  and `reset_arrays` last), `_accumulate_registration_phase` (its total plus
  the kick-clock gap forms `_carrier_slip_gap`), `_update_frame_rotations`
  (the per-passage generator/kick frame rotations — must precede every
  `circuit_track` of the passage), `_replay_backfill_span`, then either
  `_write_no_correction_readout`
  (early return, `grid_only_no_correction` only) or `_track_forward_span` +
  `_write_station_readout`. Note the registration phase and the rotations
  moved BEFORE the backfill replay: the replay's cell updates already
  compose the demod-frame sum with this passage's rotations.
  Two orderings cannot be expressed as arguments and are stated in the
  respective docstrings (the first is additionally asserted per turn):
  `reset_arrays` must size the coarse state before any `circuit_track`, and
  `_carrier_slip_gap` must be complete before
  `calculate_rf_beam_current_partial` reads it off the instance. The
  registration phase Ψ is *accumulated* by `_accumulate_registration_phase`;
  `_track` only folds it into `_carrier_slip_gap` — so "the `_track` frame
  correction" is no longer the right name for it.
- **Backfill-span generator prefill (zero-order hold).** `reset_arrays`
  gained `n_backfill_cells`: the generator grid is seeded with the
  feedforward bias, *except* over the leading no-beam backfill cells, which
  are seeded with `_last_val_generator_current`. Those cells replay an
  already-elapsed interval during which the loop issued no new command, so
  the generator kept running at whatever it was last told — it did not snap
  back to the bias. `cavity_response` already drove the *first* backfill cell
  from the held value; this extends it over the rest of the span. Without a
  controller the held value *is* the bias, so the constant-current path is
  bit-unchanged. This fixed a real detuned-cavity defect
  (3.1e-2 / 4.6e-2 relative setpoint error at 2 / 4 sections).
- **Segment-boundary residual fix.** `RFCenterSegment.residual` is now
  actually **consumed**: `RFCenterGridMixin._preceding_segment_residual`
  reads it back to form the first coarse step of the *following* segment
  (`rf_centers` are segment-local, so that step is the following segment's
  first local centre plus the preceding segment's unfilled tail). The live
  host scalar `_residual_time_last_rf_centers_calculation` cannot serve there
  — the whole per-turn grid is generated before any of it is walked, so by
  consumption time it holds the last-generated (forward) segment's value. The
  first segment of a turn takes `_residual_time_carried_into_turn`
  (snapshotted before this turn's generation); hand-built grids with no
  segment list fall back to the live scalar, bit-for-bit as before.
  `__post_init__` bounds `residual` to `[0, duration]` (± 1e-9 s of float
  slack).
  **CORRECTED 2026-08-11:** this bullet used to end "**Still write-only**:
  `RFCenterSegment.omega` and `.duration` are validated in `__post_init__`
  and never read back". That is **no longer true** — all four fields are
  load-bearing. `_replay_backfill_span` walks `_segments[:-1]` and drives
  each segment's `circuit_track` from `segment.omega`, and
  `_accumulate_registration_phase` computes
  `Ψ = Σ_k (segment.omega − ω_0) · segment.duration` over the same records.
  Both replaced earlier slices of loose parallel arrays, which is exactly why
  the value class exists.
- **Unified profile-span guard `ProfileBaseClass.check_fits_in_span`**
  (`profiles.py`), plus the new `profile_duration` cached property (the
  outer-edge span `cut_right - cut_left`, i.e. `n_bins * hist_step` — one
  `hist_step` MORE than the first-to-last-bin-centre distance the ad-hoc
  checks used to compute, which understated the window by one bin). One guard
  now serves every consumer that must place the profile window inside a span
  it does not control, and in each case `span` is the same physical quantity:
  the interval between two consecutive passages of the consuming element.
  Exactly two call sites (verified 2026-08-11): `rf_beam_current`
  (`beam_current.py`, span = `n_points * sampling_time`, the FORWARD segment
  only — 1/n_sections of a turn, not a full turn and **not periodic**) and
  `MultiPassResonatorSolver._update_past_profile_times_wake_times`
  (`solvers.py`, span = the depositing beam's own previous passage interval).
  Both mechanisms destroy charge (re-bin fold at exactly 50 % loss;
  past-deposit self-overlap), so both raise.
  **Correction to an earlier note, still standing:** there was never a symbol
  named `check_profile_span_within_passage_time` — nothing of that name was
  deleted (verified again 2026-08-11 with `git log -S`: the string appears
  only in this file's own history). `check_fits_in_span` and
  `profile_duration` are both new, and what they replaced were inline ad-hoc
  width computations.
  Consequence in `rf_beam_current`: the `% n_points` **wrap-around was
  removed** from the coarse-charge writes (a wrap would overwrite an earlier
  cell rather than accumulate into it, because the coarse grid is not
  periodic), backed by the span guard plus the explicit index bounds of
  §2.12(h).
  On the solver side the span judged is deliberately **not** `delta_t` (the
  inter-deposit gap): that is shorter than a passage interval whenever the
  element is first reached mid-turn, and interleaved two-beam deposits
  legitimately overlap because they carry different beams' charge. A beam
  with no previous passage skips the check.
- **`ForwardEulerValidityGuard` extracted** to `cavity_solvers.py`, beside
  the solvers it certifies, because it is pure numerics: it reads no grid, no
  RF station and no beam, all cavity parameters are passed per call, and the
  only state it owns is the once-only beam-kick warning flag. It holds the
  three tripwires (`check_step_sizes`, `check_beam_kick_magnitude`,
  `check_beam_kicks`) and the four thresholds (`max_step_angle`,
  `max_step_angle_hard`, `max_relative_kick`, `max_relative_kick_hard`). The
  feedback owns one instance (`self._euler_guard`); the `_check_*` methods on
  `IQCavityFeedbackTimingClass` survive as thin delegating wrappers.
- **Controller separated from the feedback.** `generator_current_controller.py`
  gained the `GeneratorCurrentController` ABC above
  `GeneratorCurrentPIController`, so the feedback holds only an instance of
  the interface and need not know the control law. The compiled path is now
  an opt-in *controller* capability, not a feedback special case: a
  controller advertises `supports_envelope_scan` (class attribute, `False` on
  the ABC, `True` on the PI controller) and then owns its scan kernel
  (`envelope_scan_kernel`), the marshalling of its own tuning/state
  (`envelope_scan_state`) and the write-back (`absorb_envelope_scan_state`).
  Controllers that do not advertise it are driven cell-by-cell through
  `update_generator_current`. `envelope_kernel.py` keeps `envelope_pi_scan`
  and gained `inactive_controller_scan_state`.

### 2.12 Capability + hardening pass, 2026-08-08…11

Commits `51f3f604`, `6086b0d4`, `531ff49b`, `4566b5cd`, `a678e72f`,
`d93eaf3c` (plus uncommitted working-tree changes at the time of writing —
`git status` showed ~24 modified files).

**(a) Multi-harmonic support.** The feedback takes `harmonic_index`
(default 0) and can regulate one harmonic of a `MultiHarmonicRFStation`. All
RF parameters go through `IQCavityFeedbackBase._resolve_main_harmonic`
(scalar pass-through for a `SingleHarmonicRFStation`, `value[harmonic_index]`
otherwise), and the coarse grid is built from **that harmonic's** design
frequency.

**INVARIANT**: the feedback's `harmonic_index` must equal its slot in the
station's `cavity_feedback_list`, because
`calc_gap_voltage_with_feedbacks` applies each feedback's corrections at the
LIST index (`enumerate(...)`) while the feedback computes them from its own
index. A mismatch is silent wrong physics, not a crash. Guarded at two
layers (prevention at attach, validation at run start):

- at attach time — **SUPERSEDED (2026-08-12, maintainer ruling — see
  §3.3)**: the original mismatch `ValueError`
  (`RFStationBaseClass._check_feedback_harmonic_index_matches_slot`,
  called from both `attach_cavity_feedback` branches) was replaced by
  SET-from-slot: attaching now *overwrites* the feedback's
  `harmonic_index` with the slot, so a mismatch cannot arise through the
  attach at all;
- at run start — `IQCavityFeedbackTimingClass._validate_multi_harmonic_slot`,
  from `on_run_simulation` (the first hook that both knows the parent and
  still precedes every grid build). It also rejects a feedback that is
  *missing* from the list, and one occupying several slots. Reached only
  by mutating `cavity_feedback_list` directly after the attach.

`attach_cavity_feedback` additionally rejects out-of-range slots — both
`harmonic_index > n_rf - 1` and `harmonic_index < 0` (a negative index passed
the old upper-bound check and silently addressed a slot from the END).

**(b) ≥ 2 coarse centres per segment**, enforced in
`RFCenterSegment.__post_init__` with an actionable message (it reports the
segment duration in RF periods and names the two remedies). This retired
three defects at once, and the latter two gates now **RELY** on the
invariant — both say so at the gate:

1. the meeting-azimuth coincidence tolerance
   `rf_centers[-1] - rf_centers[-2]` (in `_rebuild_per_turn_grid`) only
   measures a genuine forward cell width when both entries lie inside the
   forward segment; a single-centre forward segment made it cross a segment
   boundary, go negative, and silently **disarm the guard**;
2. an empty segment carried the preceding residual through without adding its
   own duration to the bridging coarse step;
3. the registration-phase gate `n_backfill_centers > 0` is only equivalent to
   "no backfill segments at all" when no segment can be empty — otherwise an
   all-empty backfill span **permanently drops its Ψ** from the running
   total.

**(c) Fine-grid initial-condition causality.**
`_check_fine_grid_initial_condition_is_causal` runs **every turn** (the first
forward centre moves with the design frequency and the carried residual;
`cut_left` is itself settable) and requires
`rf_centers_forward[0] <= profile.cut_left` **when the window carries
charge** — gated on `beam_current_fine_grid`, not on `profile.hist`, because
direct-drive tests hand the fine grid a current without ever slicing.

Record the counter-intuitive, **MEASURED** fact: the seed is deliberately
coarse index **[0]** and is **NOT** interpolated to `cut_left`. Coarse cell 0
is charge-free by construction (`forbid_charge_in_first_coarse_cell`), but
cell 1 typically already holds ~50 % of the bunch, so interpolating from 0
toward 1 drags up to ~10 % of the beam-induced voltage **backwards in time**,
into an initial condition that predates the charge that produced it — and the
fine solve then integrates the same current twice. Interpolating broke 57
tests. The comment at the seed in `_resolve_fine_grid_voltage` says "must
stay"; believe it. This guard sits **beside**
`forbid_charge_in_first_coarse_cell`; neither subsumes the other.

**(d) The `debug` flag was SPLIT into three.** Previously `debug=True`
silently disabled the feedback (unit gain, zero phase) as a side effect of
asking for diagnostics. Now: `debug` records the inspection-only snapshots
only; `validate_grid_each_turn` runs the per-turn grid-vs-`_segments`
integrity check (and asserts that the forward segment's boundary residual
equals its demodulation frame — the two used to be derived independently and
silently disagree); `grid_only_no_correction` — **and nothing else** — stops
the turn before any correction and writes the neutral readout. All three
default `False`, which is bit-for-bit the old `debug=False` path.

**(e) Coincident coarse point DUPLICATES the previous cell** (with a warning)
instead of skipping it. A zero-length step carries zero elapsed time, so
`V(t+0) = V(t)`; leaving the zeros prefill would advance the *next* cell from
`v_prev = 0`, destroying the coherent cavity voltage and refilling it only
over `τ = 2 Q_L / ω` (hundreds of turns at `Q_L ~ 1e6`). Duplication also
keeps the two downstream readers honest — `reset_arrays` carries the LAST
cell into the next turn, and the fine solve seeds from the FIRST forward
cell.

**(f) The pre-fill seed is evaluated on the DESIGN clock.**
`on_run_simulation` passes `omega=self.omega_rf_design` and the matching
`t_rev` to `pretrack_fill_voltage`, because the seed initialises the coarse
recursion, which is driven at `_forward_segment_omega_design`. Evaluating at
the actual (offset) frequency would miss the no-beam fixed point by
`O(delta_omega_rf / omega)` — an injection transient the PI would then have
to burn off.

**(g) `rf_beam_current` guards** (all in `beam_current.py`):

- profile-window-vs-span → `profile.check_fits_in_span` (raises);
- `hist_step > sampling_time` → **raises**: the downsampling loop derives each
  coarse cell from *consecutive-index* steps in `ind_fine`, so a fine grid
  coarser than the coarse grid lets the index jump by ≥ 2, the running
  counter falls behind and charge is placed at the **wrong time** while the
  total stays conserved (silent corruption). Reachable from a
  legitimate-looking setup, since sub-stepping shrinks `sampling_time`;
- index bounds, `_check_coarse_index_bounds`: mapping **past** the last cell
  always raises; mapping **before** the first cell (which NumPy's negative
  indexing would deposit into the *last* cells, ~one forward span late and
  out of reach of the first-cell guard) raises when those bins carry
  > 1e-9 of the total demodulated charge and otherwise still only **warns** —
  the same relative-threshold idiom the first-coarse-cell guard uses, because
  far Gaussian tails are non-zero in float arithmetic (~1e-100) without being
  physically populated. So "raises instead of warning-then-wrapping" is true
  for charge-carrying underflow, not for a numerically-noisy tail.

**(h) `cavity_sum_phase` raises `NotImplementedError` whenever the station
carries ANY cavity feedback.** It used to consult
`get_main_harmonic_cavity_feedback` (main-harmonic slot only), so a feedback
regulating a non-main harmonic of a `MultiHarmonicRFStation` slipped through
and the phase loop ran as if the cavity were unregulated. It now uses
`any_feedback_not_none` — the same predicate the beam controls use at run
start, so both guards fire for exactly the same configurations. With no such
feedback it stays a silent no-op, which is a normal supported configuration
(the beam controls call it unconditionally every turn).
The neighbouring `check_main_rf_stations_with_cavity_feedback` had a
tautological condition and warned on every run; it now warns only on a
genuine **mixture** (`any(...) and not all(...)`).

**(i) `_track` decomposition, `circuit_track` split, and the module moves.**
`circuit_track` now does two things: advance the coarse cells
(`_circuit_track_cells`) and — only when the segment carries beam — resolve
the envelope onto the fine grid (`_resolve_fine_grid_voltage`). The
coarse-step arithmetic (`coarse_step_exponent`, `euler_voltage_multiplier`,
`exponential_voltage_multiplier`, `exponential_drive_weight`) lives in
`cavity_solvers.py` beside `ForwardEulerValidityGuard`, so the per-cell
reference and the vectorised kernel spell it once. `PerTurnGridSpan` lives in
`rf_center_segment.py` next to `RFCenterSegment`.
`_validate_voltage_setpoint` (the explicit-setpoint constructor policy) and
`_limit_fine_grid_generator_current` (the fine-grid klystron clamp) moved
from the timing class onto `GeneratorRegulationMixin`.

### 2.13 Split coarse envelope — driven readout-phase residual FIXED (2026-08-12)

The residual caveat of the Ψ fix (§3.3) is CLOSED. The coarse envelope is
now TWO components propagated by the same recursion (the ODE is linear —
superposition is exact): `antenna_voltage_gen_coarse_grid`
(generator-sourced) and `antenna_voltage_beam_coarse_grid` (beam-sourced).
The public `antenna_voltage_coarse_grid` remains the DEMOD-FRAME SUM,
recomposed per passage as
`V_beam + V_gen · exp(−i(delta_phi_rf + gap + Ψ))`
(`_compose_coarse_sum` / the kernel's per-cell composition) — exactly
`1+0j` rotation without an offset and without multi-section acceleration,
hence undriven runs bit-identical (additionally enforced by the
`_generator_active` gate, which skips the gen-component update and every
composition multiply for undriven feedbacks; refreshed by `reset_arrays`).

- **Anchoring decision (maintainer-settled): the klystron drive follows the
  DESIGN frequency.** The gen component is natively design-clock-anchored —
  a constant injection per segment at each segment's own design frequency
  is exactly right (those *are* samples of the design program), and it
  carries no registration phase Ψ at readout. Under a station
  `delta_omega_rf` the design-locked drive physically walks off at MINUS
  the kick-clock slip relative to the actual RF — modelled physics, pinned
  by `TestDesignLockedDriveWalkOffUnderRFOffset`
  (`phase_correction == −delta_phi_rf`, atol 1e-9). The beam component
  keeps the pre-existing demod/readout closure byte-for-byte.
- **`_carrier_slip_gap` contract update:** its two constituents are now
  held separately — `_kick_clock_slip_gap` (live kick-clock tail, reset per
  passage) and `_grid_carrier_phase` (running Ψ total) — because the
  generator frame rotation needs the kick-clock part with the station
  clock on top (`_update_frame_rotations`:
  `exp(−i(delta_phi_rf + gap + Ψ))` for composition,
  `exp(+i(gap + Ψ))` for the kick frame). The demod/readout sides still
  subtract/add the identical folded total, unchanged.
- **PI regulates the KICK-frame sum**
  (`error = V_set − V·exp(+i(gap + Ψ))`), python
  (`_update_generator_current`) and kernel identically; the
  `envelope_pi_scan` signature grew (the two component in/out arrays, the
  `generator_active` gate, the two rotation scalars).
- **`initial_voltage` / the pre-fill seed the GEN component** (flagged
  corner, decided): the seed models a generator-established,
  design-anchored field, so `reset_arrays` puts it on
  `_last_val_ant_voltage_gen`; the beam component starts empty.
- **Result:** at intensity 0 the feedback is exactly phase-neutral — the
  synchronous-phase bug is gone (RCS1 example probe: −4.65 deg growing over
  8 turns before, ~1.5e-6 deg after; measured in the fix session). Pinned
  in-repo by `TestDrivenFeedbackIsPhaseNeutralWithoutBeam` (both drive
  variants, 1e-12 rad gate) and in the outer repo by
  `test_feedback_is_a_no_op_without_beam`, whose strict `xfail` became a
  plain pass.
- **Re-pins:** the multi-section FAST-ramp PI pins were regenerated (the
  old set encoded the artefact — Ψ ~0.14 rad/turn/station fought by the
  PI; `i_max_dev` dropped ~9 %, `|V_ant|` moved ≤ 1.8e-2 relative). The
  multi-section SLOW-ramp pins were regenerated too: the kick-frame PI
  error shift is `V_beam·(1 − e^{iΨ})` with Ψ ~7e-6 rad/turn there, moving
  `|V_ant|` ≤ 2.4e-6 and the current response ≤ 1.7e-6 relative —
  marginally past the 1e-6 pin tolerance, declared in the pin comments.
  **ACCEPTED by the maintainer (2026-08-13).** The shift is a real
  modelling improvement -- the loop now regulates the frame the beam
  actually sees -- and its magnitude was reproduced independently by
  the verifier before acceptance. The rejected alternative was to
  restore the pre-fix values and widen that test's `rtol` to ~5e-6,
  which would hide future drift of exactly this size.

---

## 3. Open items / flagged (NOT done — need decisions)

### 3.1 Counter-rotating / two-beam

- **CR-3 equal-time patch path** (deferred by user): integrating two
  coincident beam currents in the feedback (deposit-sum + envelope rewind).
  Design options recorded in the memory note. Kick-ordering fork: symmetric
  one-passage delay / pooled kick / asymmetric lag.
- **Coincident-kick asymmetry in `MultiPassResonatorSolver`.** With
  `allow_delta_t_zero=True` each beam is kicked inside its own track call, so
  the first-tracked beam sees `W(0)/2` where the second sees `W(0)`, and
  swapping the track order swaps which beam is under-kicked. For equal
  coincident charges the kicks are `0.5` and `1.5` times the correct
  `W(0)·Q` — the sum survives, the split does not — so it shows up as a
  spurious beam-to-beam differential.
  **DECIDED (2026-08-31): no fix.** The maintainer ruled that the case is
  unreachable without explicitly choosing `allow_delta_t_zero=True`, so
  symmetrising it (deposit both coincident profiles, then kick) is a
  **non-goal**. Instead the wrongness is stated where it happens: the
  parameter docstring, the construction warning and the feedback's
  `NotImplementedError` all now say the results are *wrong* rather than
  merely order-dependent (the error message no longer offers the solver as
  a substitute), and a genuinely coincident deposit emits a runtime
  `UserWarning` — one-shot per solver, guarded on `delta_t <= 0` *and* a
  stored profile, so a first deposit and ordinary passages stay silent.
  Pinned by four tests in `TestMultiPassResonatorSolver`
  (`test_coincident_passage_warns_that_the_result_is_wrong` and the three
  negatives/one-shot). Verified on a real meeting-azimuth two-beam run:
  exactly one warning, at the first coincident deposit.
- **Per-beam live profiles** under two-beam tracking clobber each other
  (tests use frozen profiles) — core gap.
  **CORRECTION (2026-07-23):** the earlier belief that a
  *(profile, consumer, profile)* sandwich is sufficient is WRONG. The
  exact-interleave replay proves even the minimal sandwich `[P, C, P]` is
  corrupt (the *forward* beam reads the counter's histogram), and
  `[P,C,P,Drift]` corrupts the counter beam. A padded layout (e.g. `PCPDD`,
  `DPCPDDD`) can be safe, but no simple positional rule characterises it. The
  guard was therefore REPLACED:
  `MainloopCounterRotatingBeams._check_two_beam_profile_placement`
  (`blond/core/simulation/execution_models/conterrotating_beams.py`) now
  **replays the exact mainloop interleave** (forward tracks `elements[k]`,
  then counter `elements[N-1-k]`; two turns for steady state) and raises if
  any consumer reads the other beam's histogram — 0 too-lax / 0 too-strict
  over 1792 layouts **under the pure-reader model** the replay assumes.
  **CAVEAT (review 2026-07-24):** the real mucol consumers (`WakeField` with
  `track_profile=True`, and the feedback's
  `calculate_rf_beam_current_partial`) SELF-histogram their profile in place
  before consuming it, so the write+read is atomic per beam and no interleave
  corrupts it. The replay models consumers as pure readers and does not see
  those self-writes, so it is *conservative*: it rejects the natural
  attached-live-profile layout. No supported config is wrong (the shipped
  two-beam path uses frozen profiles), but the "provably correct" framing
  overstates. Frozen (`active=False`) profiles are exempt. Tests in
  `test_simulation.py::TestTwoBeamProfilePlacementCheck` (the old
  `test_sandwiched_live_profile_passes` became
  `test_minimal_sandwich_rejected`). The real long-term fix remains per-beam
  profile instances; this guard turns silent corruption into a loud error.

### 3.2 Housekeeping

- ~~Observation tests write `last_*.npy` / `last_*.json` into the CWD~~
  **RESOLVED (2026-08-13)**: the `.gitignore` already ignored
  `tests/last_*`, but observables default to `folder=""` and therefore
  write relative to the CWD -- and the documented workflow runs pytest
  from the repo *root*, so the pattern never matched. Repo-root
  `last_*.json` / `last_*.npy` rules added alongside the `tests/` ones
  (with the reason recorded in `.gitignore`), and the ten stray files
  removed. Verified: a full suite run regenerates them and `git status`
  stays clean. The deeper fix (a `tmp_path` fixture per emitting test)
  is still open but no longer leaks into the working tree.
- ~~`_phase_offset_frwrd` / `_phase_offset_frwrd_next` are vestigial~~
  **RESOLVED (2026-08-13)**: both were always exactly `0.0` (initialised
  in `__init__`, re-zeroed in `on_run_simulation`, never written
  elsewhere). Deleted, together with the one test term that added
  `_phase_offset_frwrd` to a `np.sin` argument where it contributed
  zero. Grep now returns no occurrence anywhere in `blond/` or
  `tests/`; suite unchanged.
- **The extracted mixins** (`RFCenterGridMixin`, `GeneratorRegulationMixin`)
  are still pure moves (methods take `self: IQCavityFeedbackTimingClass`);
  promoting them to composed collaborators is the natural follow-up.
- **P6** (RF-parameter view mixin) skipped per user.
- ~~Full Sphinx doc build not yet run~~ **RESOLVED (2026-08-13)**: built
  green (`build succeeded`, exit 0, zero warnings) under `-W` +
  `nitpicky = True` for the current state of both RSTs and all docstrings.
  One `-W` failure was found and fixed en route: a `:meth:` role on the
  private `_compose_coarse_sum` in `cavity_response`'s docstring (a role on
  an underscore-leading member never resolves -- use ``literal`` markup).
  Run it **sequentially, never looped** (a looped/concurrent build wipes the
  shared `examples/`/`_build/` dirs and produces spurious warnings); from a
  Bash tool use the ABSOLUTE path, `cmd //c "C:\...\BLonD\docs\create_docs.bat"`,
  because the shell cwd drifts between calls. No new top-level exports were
  added, so `ASSIGNED_CATEGORIES` needs no update.
- ~~RST/source name drift~~ **RESOLVED (2026-08-12)**: both RSTs now use
  the backfill vocabulary throughout and `envelope_kernel.py` carries no
  time-sense "reverse" — see the resolved-stragglers note in §1.3.

### 3.3 Resolved / decided (kept as records, not as open work)

- ~~Beam phase loop ↔ cavity feedback coupling~~ **DECIDED (2026-08-12,
  maintainer ruling): deliberate NON-GOAL — the phase loop must not couple
  to the cavity feedback at all.** The `NotImplementedError` that
  `cavity_sum_phase` raises whenever a station carries ANY cavity feedback
  (§2.12(h)) is the intended *permanent contract*, not a stub awaiting an
  implementation. Background (still accurate): the original body read the
  deleted blond2 coarse-array API (`I_BEAM_COARSE`, `V_ANT_COARSE`, a
  fixed `n_coarse` per turn), which exists nowhere in the live tree; with
  no such feedback the method stays a silent no-op — a normal, supported
  configuration the beam controls exercise unconditionally every turn.
  Known wording drift (code frozen during the 2026-08-12 docs sweep): the
  method's two `TODO`s and its message's "open design task" phrasing
  predate the ruling — and `TestCavitySumPhaseGuard` pins that phrase
  verbatim, so message and test must move together in the next code pass.
  Related: `get_main_harmonic_voltage` warns that it returns the
  UNPERTURBED voltage while local feedbacks are active (triggered from
  `BeamFeedbackBase.beam_phase()` when it forms the main-harmonic vector
  sum); with the coupling ruled out, that warning documents the permanent
  state, not an interim one.
- ~~`harmonic_index` API questions~~ **RESOLVED (2026-08-12, maintainer
  ruling)**, in three parts:
  - **SET, not validate**: `attach_cavity_feedback` now SETS the feedback's
    `harmonic_index` from the slot (single-feedback path: the
    `harmonic_index` argument; list path: each non-None entry's position),
    via `RFStationBaseClass._set_feedback_harmonic_index_from_slot`
    (`hasattr` guard, so duck-typed feedbacks without the attribute attach
    untouched). The slot is authoritative; the constructor value is only
    the unattached default. Set silently, documented loudly. This
    supersedes the attach-time mismatch `ValueError`
    (`_check_feedback_harmonic_index_matches_slot`) described in §2.12(a);
    the run-start guard `_validate_multi_harmonic_slot` is unchanged and
    still catches direct `cavity_feedback_list` mutation.
  - **Fractional `harmonic_index` hard-rejects** (`ValueError`) at both
    entry points (`IQCavityFeedbackBase.__init__` and
    `attach_cavity_feedback`), shared via
    `blond.physics.cavities._coerce_harmonic_index`. A harmonic index is a
    list slot, not a physical quantity, so the lenient
    `int_from_float_with_warning` idiom (untouched — other quantities keep
    it) no longer applies here. `int`, `np.integer` and integral `float`
    are accepted silently.
  - **`IQCavityFeedbackBase(harmonic_index=np.int64(1))` FIXED**: the
    shared strict coercion accepts `np.integer` at both entry points, so
    the constructor no longer raises on `np.int64`.
- ~~`delta_omega_rf` lab-frame demod slip~~ **RESOLVED (2026-07-22)**. The
  coarse-grid geometry went fully onto the design clock; the offset enters
  only as explicit phases (demod `carrier_phase_offset = −(delta_phi_rf +
  live gap)`, readout `phi_rf` + `phase_correction`). The live gap
  `δω·(t_now − station._last_reference_time_phase_slip)` compensates the kick
  clock's end-of-track lag (blond2 convention, untouchable). Measured net
  carrier-phase error vs the retuning convolution **≤ 2e-5 rad/turn** at 8e2
  and 2e3 rad/s (was ~2 %/turn per 1e3 rad/s). Diagnosis was empirical: a
  per-turn linear-response solve over free demod phases proved the residual
  was a whole-envelope *readout frame* drift from the slipping grid, not a
  per-deposit demod error — hence a geometry redesign rather than a phase
  patch. Also deleted in that pass: the `phi_rf` parameter of
  `_generate_rf_centers`, the `_get_time_to_next_rising_edge_zero` helper,
  and (in the follow-up) the dead `_forward_carrier_omega_rf` attribute.
- ~~Driven multi-section fast-ramp frame slip~~ **FIXED (2026-07-24)**. Root
  cause was *not* the assumed geometry bug (seed mis-registration is 1e-6
  t_rf/seam, four orders too small) but a carrier-phase bookkeeping mismatch
  `Ψ = Σ_k (ω_k − ω_0)·T_k`, identically 0 for one section. The old code
  applied Ψ as a **rotation of the antenna-voltage state**, which also hit
  the generator-driven field — a field that carries no registration error at
  all (re-injected on the current grid every cell) — so the constant drive
  fought the rotating state and a *phase* error became an *amplitude* drift.
  Fix: Ψ accumulates into `_grid_carrier_phase`, folds into
  `_carrier_slip_gap`, is subtracted at demodulation and added back at
  readout; the state rotation is DELETED. Proof it is a real fix and not a
  compensation: the 5 mtw tests that failed when the rotation was merely
  removed now pass **without** it. **Residual caveat — RESOLVED
  (2026-08-12, split envelope, §2.13)**: a driven multi-section fast ramp
  used to keep a readout-**phase** offset, because the beam-induced part
  needs Ψ at readout and the driven part does not, and ONE readout phase
  could not separate them (amplitude was already exact). The source-split
  envelope separates them per component, so the former design-RST *Known
  limitations* bullet for it is retired; the RCS1-example measurements of
  the old residual are kept in §2.13 as record.
- ~~Forward-Euler hard cap~~ **DECIDED + SHIPPED**: tightened from `2.0` to
  `1.0` (`ForwardEulerValidityGuard.max_step_angle_hard`), so the
  sign-flipping `1 < d < 2` band is forbidden too, not just the divergent
  `d > 2`.
- ~~`phase_correction` vs `pi_setpoint` frame~~ **RESOLVED (user decision:
  error)** — the constructor rejects a non-real / non-positive explicit
  `voltage_setpoint` with `ValueError` (rotate `phi_rf` on the station
  instead).
- ~~MultiPole vs MultiPass on missing R_CR~~ **RESOLVED** via
  `origin/blonder` (`2235e519`, merged in `b047e972`). That merge produced
  one **semantic conflict** in `test_sources.py::test_get_impedance`
  (origin's `−R` construction × our negating `get_impedance`); fixed by
  restoring the `+R` construction matching the surviving convention.
- ~~`print_one_turn_execution_order` crash on empty `rf_centers`~~
  **RESOLVED** (`0936668f`; regression tests in
  `tests/unittests/core/ring/test_beam_physics_relevant_elements.py`).
- ~~`experimental/physics/feedbacks/helpers.py` still uses raw charge~~
  **MOOT** — `blond/experimental/physics/` now holds only `kick_pooling.py`
  (re-verified 2026-08-11), so there is no experimental `rf_beam_current`
  copy left.
- ~~LHC-suite npz vs the large-files hook / LHC suite CI visibility /
  pytest-xdist for the comparison directory~~ **MOOT** — the suite no longer
  exists (§2.10).

---

## 4. Module layout

`blond/physics/feedbacks/` (line counts as of 2026-08-12):

| module | holds |
|---|---|
| `cavity_feedback.py` (3043) | `IQCavityFeedbackBase` + `IQCavityFeedbackTimingClass(IQCavityFeedbackBase, RFCenterGridMixin, GeneratorRegulationMixin)`. Per-turn orchestration: `_track` + its **nine** phase methods (§2.11, incl. `_update_frame_rotations`), `circuit_track` → `_circuit_track_cells{,_python,_kernel}` + `_resolve_fine_grid_voltage`, the kernel glue (`_coarse_step_sizes`, `_kernel_step_multipliers`, `_kernel_beam_current`), `cavity_response` (advances the two source-split envelope components, §2.13), `_compose_coarse_sum`, `_advance_coarse_voltage`, `cavity_response_fine`, `calculate_rf_beam_current_partial`, `reset_arrays` (incl. `_generator_active` refresh and the gen-component seeding), `on_run_simulation`, `_validate_multi_harmonic_slot`, `_check_fine_grid_initial_condition_is_causal`, the pre-fill call. `_check_step_sizes`, `_check_beam_kick_magnitude`, `_check_beam_kicks` are thin wrappers delegating to `self._euler_guard` |
| `rf_center_grid.py` (884) | `RFCenterGridMixin` — coarse `rf_centers` construction: the forward and **backfill** reference walks, `_generate_rf_centers`, segment generation (`_append_segment` / `_clear_segments` / `_rebuild_grid_arrays` / `_close_previous_turn_grid`), `_preceding_segment_residual`, `_validate_grid`, and the two direction selectors (`_reference_list_for_direction`, `_own_index_for_direction` — the *space*-sense reverse, §1.3). `_segments` is the single source of truth; the flat arrays are derived. Its module docstring is the canonical statement of the backfill-vs-reverse rule and of the design-clock-only geometry |
| `rf_center_segment.py` (165) | The two value classes: `RFCenterSegment` (all four fields load-bearing — see the correction in §2.11 — with the ≥ 2-centres and `residual ∈ [0, duration]` validation) and `PerTurnGridSpan` (`n_backfill_centers`, `n_forward_centers`, `residual_from_backfill_span`). Both are imported by `cavity_feedback.py` |
| `cavity_solvers.py` (768) | **mucol-only.** Fine-grid solvers `cavity_response_sparse_matrix` (forward-Euler) and `..._second_order` (Crank-Nicolson); the coarse-step arithmetic `coarse_step_exponent`, `euler_voltage_multiplier`, `exponential_voltage_multiplier`, `exponential_drive_weight` (spelled once for both the reference and the kernel path); `pretrack_fill_voltage`; and `ForwardEulerValidityGuard` — the discretisation tripwires, beside the solvers they certify. Its module docstring owns the `omega_times_dt` naming rule (§1.4) |
| `envelope_kernel.py` (304) | numba host kernel `envelope_pi_scan` + `inactive_controller_scan_state` — the sequential coarse-envelope + PI recursion; solver-agnostic and byte-identical to the Python reference. Since §2.13 it advances the two source-split components, composes the demod-frame sum per cell and forms the PI error in the kick frame; the signature carries the component in/out arrays, the `generator_active` gate and the two per-passage rotation scalars. Reached through the **controller's** `supports_envelope_scan` capability, not called by the feedback directly |
| `generator_regulation.py` (244) | `GeneratorRegulationMixin` — `_controller_active`, `pi_setpoint`, `_validate_voltage_setpoint`, `generator_power`, `_update_generator_current` (forms the PI error in the KICK frame via `_kick_frame_rotation`, §2.13), `_limit_fine_grid_generator_current`. **What it does NOT own** (and its module docstring says so): the compiled envelope scan and the per-cell stepping decision stay on the timing class, because they need both coarse grids and the three values carried across the turn boundary, and because the scan depends on `pi_setpoint` staying *unevaluated* on a span the controller sits out (that property may reach through to the parent station, which a no-beam backfill span must not require) |
| `generator_current_controller.py` (446) | `GeneratorCurrentController` ABC + `GeneratorCurrentPIController`; the envelope-scan capability hooks (`supports_envelope_scan`, `envelope_scan_kernel`, `envelope_scan_state`, `absorb_envelope_scan_state`); `current_limit_from_power`, `clamp_magnitude` |
| `beam_current.py` (435) | `low_pass_filter`, `rf_beam_current` (unified; keyword-only coarse args; no wrap-around; `check_fits_in_span` + `hist_step`/`sampling_time` + `_check_coarse_index_bounds` guards) |
| `beam_feedback.py` (481) | the surviving phase loop (`BeamFeedbackBase`), incl. `cavity_sum_phase`, whose `NotImplementedError` guard is the permanent contract — coupling is a deliberate non-goal (§3.3) |
| `iq.py` (66) | `cartesian_to_polar`, `polar_to_cartesian` |
| `base.py` (244) | `FeedbackBaseClass` / `LocalFeedback` / `GlobalFeedback` (unchanged) |
| ~~`helpers.py`~~ | DELETED (2026-07-25); its contents moved into `cavity_solvers.py`, re-export shims gone |

**Re-exports**: none. The only other `rf_beam_current` in the tree is
`blond/legacy/blond2/llrf/signal_processing.py`, which is self-contained and
out of scope. Mucol production + tests import from the canonical modules
above.

**Tests** — inventory lives in the test RST (*Test modules*). Layout only:

- `tests/unittests/physics/feedbacks/` — `test_base.py`,
  `test_beam_feedback.py`, `test_cavity_feedback.py`,
  `test_cavity_feedback_requires.py`, `test_helpers.py`,
  `test_rf_center_grid.py`, `test_rf_center_segment.py`.
- `tests/unittests/physics/feedbacks/accelerators/mucol/` — the 14 mucol test
  modules plus the shared harness (`mucol_cav_fdbk.py`, `support.py`,
  `stubs.py`, `conftest.py`, `plotting.py`, `fdbk_testing/`). The unused
  debug method `plot_antenna_voltage` lives in `plotting.py` as a function.
- **CORRECTED 2026-08-11:** `test_cavity_feedback.py` is no longer "reduced
  to the empty `TestIQCavityFeedbackObservationClass` stub". It now carries
  the diagnostic-flag-split tests, the multi-harmonic resolution /
  degenerate-multi-harmonic / non-main-harmonic-attachment /
  slot-agreement suites and the coarse-cell step-sizing suite.
- Guards owned elsewhere: `TestTwoBeamProfilePlacementCheck` in
  `tests/unittests/core/simulation/test_simulation.py`; the counter-rotating
  2×2 shunt matrix in `tests/unittests/physics/impedances/test_solvers.py`;
  the span guard's own unit tests in `tests/unittests/physics/test_profiles.py`.

---

## 5. Verification status

- `tests/unittests/physics/feedbacks` **collects 525 tests** (2026-08-12,
  collection only — this pass did not re-run the battery; up from 520
  earlier the same day with the five split-envelope tests of §2.13:
  `TestDrivenFeedbackIsPhaseNeutralWithoutBeam` (2),
  `TestDesignLockedDriveWalkOffUnderRFOffset` (1) and the two
  `test_split_components_*` kernel-identity configs; 513 on
  2026-08-11 before the seven tests of `TestConstructorHarmonicIndexValidation`
  and `TestAttachSetsHarmonicIndexFromSlot`).
- **HISTORY**: the last full battery run recorded here (mucol + LHC
  comparisons + impedances) was **492 passed**, the only failures being the
  pre-existing SPS `TestTravelingWaveCavity` ones (`test_vind`,
  `test_beam_fine_coarse`), since fixed SPS-locally (90° IQ rotation) and
  unrelated to mucol. That number predates the LHC-suite removal (§2.10) and
  the 2026-08-08…11 pass; treat it as a historical marker, not a target.
- Every production sign/gate change is **mutation-verified** (see §2.4–2.7).
- The P1–P5 partition was **byte-identical** (pure moves), verified by the
  full battery + per-step ruff/numpydoc/import/MRO checks.
- Docs: both RSTs are maintained; the full `-W` Sphinx build has **not** been
  run for the current state (§3.2).

---

## 6. Commit status

**HISTORY — superseded 2026-08-07.** This section used to read "Nothing is
committed" and proposed a four-way commit grouping. That grouping was never
used: the work was checkpoint-committed incrementally on
`blonder_feature/mucol_feedbacks`, and `blonder` has since been merged in
(`52a03664`). Check `git log` rather than this section for the current state;
as of 2026-08-11 the tip is `d93eaf3c` with a substantial uncommitted working
tree. Re-run the full battery before/after any reshuffle.
