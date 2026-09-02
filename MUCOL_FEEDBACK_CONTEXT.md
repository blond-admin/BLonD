# Muon-Collider Cavity Feedback — Work Context & Handoff

Working notes for the mucol cavity-feedback branch
(`blonder_feature/mucol_feedbacks`), July–September 2026. Checkpoint-committed
throughout; last full re-verification of this file against the source
2026-08-11; docs-consistency sweep (this file + both RSTs against the
source, scripted name/role/test-name checks) 2026-08-12; split-envelope
docs pass (this file + both RSTs + `observables.py` + the outer-repo
example, see §2.13) 2026-08-12; design-gain-ledger work plus the
absence-claim and portability audit 2026-09-01 (§2.14–§2.16, §3.1, §3.2,
§5, §6); **re-sync against the `ec159a87` tip 2026-09-02** — every test
count in §5 re-run rather than carried over, and the claims that
`cdca671b` / `ec159a87` had invalidated corrected in §0, §1, §2.13, §3.2,
§3.3, §4, §5 and §6. Still to be reviewed before the MR.

**One thing that re-sync could NOT verify: the Sphinx doc build.** It has
not been run since `064305e3`, four commits and ~2200 rewritten RST lines
ago, because this checkout's venv has no sphinx installed. §5 says what to
install and why it is the likeliest pre-MR failure.

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
logged in §2.11 and §3), so read the source, not the note.

**Claims here are dated, bullet by bullet. A date means the claim was checked
*then* — it is NOT a promise that it still holds.** Read the §3.2
`_phase_offset_frwrd` entry before you trust anything else in this file: it is
a "RESOLVED … deleted … grep returns nothing" bullet that was already false on
the day it was written, and the blanket assurance that used to stand here
("every claim below was re-checked") is exactly what let it survive
unchallenged for three weeks. **Grep before you rely on any absence claim.**
**HISTORY** bullets record *what was decided and when*, not what the code does
today.

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
  (`_rebuild_grid_arrays`) and can therefore not desync. The sanctioned READ
  surface is the three PUBLIC read-only properties on
  `IQCavityFeedbackTimingClass` — `rf_centers`, `rf_centers_lengths` and
  `forward_offset` (`cavity_feedback.py`, 1011/1042/1063 as of 2026-09-02,
  having been ~1006/1037/1058 the day before; grep the `def` lines, not the
  line numbers) — pinned by
  `TestCoarseGridAccessorsAreStatedPublic`. Read through those; the private
  `_rf_centers` / `_rf_centers_lengths` are the mixin's to mutate and are not
  a public API.
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
  and the blond2 comparison suite were REMOVED from the codebase. **The phase
  loop survived**: `BeamFeedbackBase` in
  `blond/physics/feedbacks/beam_feedback.py`, with the machine-specific
  subclasses in `blond/physics/feedbacks/accelerators/{lhc,ps,psb,sps}/beam_feedback.py`
  (four modules, each with a matching test module — see §4). `lhc` and `sps`
  call `cavity_sum_phase`, i.e. they reach the `NotImplementedError` contract
  of §3.3; do not read "LHC feedback removed" as "no LHC feedback code left".
  The byte-identical obligation and its bridge machinery (`dT_index_sign`, the
  `coarse_center_offset` **kwarg**, the helpers re-export shims) were stripped
  in the same cleanup — note the *name* `coarse_center_offset` survives as a
  local variable holding the now-unconditional `sampling_time/2` offset
  (`beam_current.py` ~405), so grepping it is not a contradiction.
  `blond/legacy/blond2/` keeps its own self-contained copies.
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
  HasPropertyCache machinery/`n_samples_coarse`/the base-class
  `use_lowpass_filter` attribute — **NOT** the `rf_beam_current` keyword of
  the same name, which is live at `beam_current.py:146` and used by four mucol
  test modules); the
  timing override now carries its OWN `@requires` decorator (regression test
  `test_cavity_feedback_requires.py`). `helpers.py` was DELETED —
  `cavity_response_sparse_matrix` lives in `cavity_solvers.py`.
- ~~`harmonic_index=1` hardcode preserved + flagged as suspicious~~
  **SUPERSEDED**: `harmonic_index` is now a real constructor parameter
  (default 0) with multi-harmonic support; see §2.12(a).

**Environment / gotchas:**

- **Repo layout this file assumes.** This `BLonD/` checkout is a git
  **submodule** of the outer `muon-collider-blonder` repo: that repo's
  `.gitmodules` declares `BLonD` → `https://gitlab.cern.ch/blond/BLonD` (and
  `MuColAccelerationParameters` alongside it), and owns
  `muon_collider_blonder/` + `test_muon_collider_blonder/`. Everywhere below,
  "the outer repo" means that parent checkout, i.e. the directory containing
  `BLonD/`. In a fresh clone start with
  `git clone --recurse-submodules <outer-url>` (or
  `git submodule update --init --recursive`), otherwise `BLonD/` is empty.
  BLonD can also legitimately be cloned standalone — then there is no outer
  repo at all, and the cross-repo pins below do not exist.
- **Running the tests.** Run pytest from the `BLonD/` checkout root:
  `python -m pytest tests/unittests/`. On the machine this file was written on
  the interpreter is the OUTER repo's shared venv, one level up — there is no
  `BLonD/.venv` here. **The venv's NAME is not stable and this file has been
  wrong about it before**: it is `.venv_312` (Python 3.12.8) as of
  2026-09-02, and every earlier revision of this bullet said `.venv`, which
  does not exist. Resolve it, do not type it from memory —
  `ls -d ../.venv*` from `BLonD/`:
  - Windows: `..\.venv_312\Scripts\python.exe -m pytest tests\unittests\`
  - Linux/macOS: `../.venv_312/bin/python -m pytest tests/unittests/`

  Two other things depend on that same name and are therefore also wrong
  wherever `.venv` is hardcoded: `docs/create_docs.bat`'s auto-activation
  line (see §3.2) and the outer repo's own `CLAUDE.md` / `README.md`, which
  document a `.venv` + `.venv314` pair that is not what is on disk. The
  outer repo is not this file's to fix.

  That shared-venv layout is this developer's choice, **not** a requirement.
  If you cloned BLonD standalone (or your outer checkout has no venv at
  all), create your own per `CONTRIBUTING.md` §2
  *Create a Virtual Environment* (`python -m venv .venv` inside `BLonD/`) and
  `pip install -e ".[dev]"` per `CLAUDE.md` *Install*. Test with
  `ls -d ../.venv* .venv` — whichever exists is yours.
  **The doc extras are a separate install and are NOT in `.venv_312`**
  (verified 2026-09-02: `python -m sphinx` reports *No module named
  sphinx*, and neither `sphinx-build` nor `sphinx-apidoc` is on PATH). So
  `.venv_312` runs the tests but cannot run `docs/create_docs.sh`; that
  needs `pip install -e ".[doc]"` in whichever venv you point it at. See
  §5 for what this leaves unverified.
  `MPLBACKEND=Agg` is no longer needed — `BLonD/conftest.py` calls
  `matplotlib.use("Agg", force=True)` when `MPLBACKEND` is unset. (The bare
  `VAR=value cmd` prefix older notes use is bash-only; PowerShell needs
  `$env:MPLBACKEND = "Agg"` on its own line.)
- **Cross-repo references.** `rcs_two_beam_example`, `RCS1`,
  `test_feedback_is_a_no_op_without_beam` and `test_rcs_two_beam_example.py`
  all live in the OUTER repo, at `<outer>/muon_collider_blonder/` and
  `<outer>/test_muon_collider_blonder/` — never inside `BLonD/`. In a
  standalone BLonD clone they are simply unavailable: treat those pins as
  unverifiable, not as failing.
- **`check copyright` (`custom-py-check`) — MACHINE-SPECIFIC, test before you
  believe it.** The hook is a `language: system` hook whose entry is bare
  `python dev_tools/precommit_check_copyright.py`
  (`.pre-commit-config.yaml:43-46`), so it inherits whatever `python` resolves
  to on PATH. On the original author's Windows box that is the Microsoft Store
  App Execution Alias stub rather than the venv interpreter, so it fails on
  *every* file regardless of content (reported variously as exit 9009
  "Python was not found" and `WinError 3` — same root cause, different
  surface). **This is a local PATH problem, not a broken hook**: in CI it
  works and it blocks the MR (see `CLAUDE.md` *CI gates*).
  **Your test:** run `pre-commit run custom-py-check --all-files`. If it
  passes, treat it as a normal blocking gate. If it fails with a
  `python`-not-found error, fix PATH first (activate the venv; check
  `where python` / `which python`) — only if it still fails spuriously should
  you judge the commit by the other hooks (ruff, isort, pyupgrade, numpydoc,
  taplo) and re-run
  `python dev_tools/precommit_check_copyright.py` by hand with the venv
  interpreter. A genuine missing copyright header must never be ignored.

  **Two corrections, both measured 2026-09-02.**

  (a) **`sync-agent-docs` is the OTHER `language: system` hook and fails
  identically** — `.pre-commit-config.yaml:53` is bare
  `entry: python dev_tools/sync_agent_docs.py`, same as line 45 — so it too
  exits 9009 here. Do not list it among the hooks you can still trust, as
  the paragraph above used to. This one matters more than the copyright
  hook: when it dies, `CLAUDE.md`, `AGENTS.md` and the whole
  `.claude/skills/` mirror are **not** regenerated, and an edit to
  `.agents/skills/` commits with its mirrors silently stale. Run it by hand
  with the venv interpreter after touching any skill —
  `<venv>/python.exe dev_tools/sync_agent_docs.py` — and re-stage what it
  rewrites. Exit 1 from it means "I rewrote files", which is success.

  (b) **The self-test above no longer returns a clean 0 in this working
  tree, and the reason is not your change.** Run by hand with the venv
  interpreter, `precommit_check_copyright.py` now exits **1** on
  `blond/examples/notebooks/getting_started.py:1` — an *untracked* file
  (§6) with no copyright header. The script sweeps the tree rather than the
  staged set, so any stray `.py` anywhere under `blond/` trips it. Before
  concluding you broke something: check whether the file it names is one of
  yours, or one of the untracked strays.
- True everywhere: module-docstring summaries must start on the line **after**
  the opening `"""` (numpydoc GL01 convention in this repo).
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
`_reference_index_until_tracked_reverse`, and the two selector helpers that
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
`harmonic_index > self._n_rf - 1` and `harmonic_index < 0` (`cavities.py`
~1038 / ~1042; the attribute is the private `_n_rf` — there is no public
`n_rf` on `RFStationBaseClass` to grep for). A negative index passed
the old upper-bound check and silently addressed a slot from the END.

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
- **`_carrier_slip_gap` contract update:** its two constituents are
  held separately — `_kick_clock_slip_gap` (live kick-clock tail, reset per
  passage) and `_grid_carrier_phase` (running Ψ total). The demod/readout
  sides subtract/add the identical folded total, unchanged.
  **CORRECTED 2026-09-02:** this bullet used to justify the split by
  claiming the generator frame rotation "needs the kick-clock part with
  the station clock on top". It does not, and did not by the time the
  claim was written: `_update_frame_rotations` computes
  `total_generator_slip = delta_phi_rf + _carrier_slip_gap`, i.e. the
  **full** gap including Ψ. `_kick_clock_slip_gap` survives only as the
  named intermediate of that sum, so the split is presentational — which
  is what the design RST (*Signal path of one turn*, step 2) has said all
  along. The formulas quoted here were right
  (`exp(−i(delta_phi_rf + gap + Ψ))` for composition, `exp(+i(gap + Ψ))`
  for the kick frame); only the reason attached to them was wrong.
- **PI regulates the KICK-frame sum**
  (`error = V_set − V·exp(+i(gap + Ψ))`), python
  (`_update_generator_current`) and kernel identically; the
  `envelope_pi_scan` signature grew (the two component in/out arrays, the
  `generator_active` gate, and **three** rotation scalars — this bullet
  said two until 2026-09-02, missing `_pi_error_frame_rotation`, which
  `afd5d96a` had already added alongside `_generator_frame_rotation` and
  `_kick_frame_rotation`; see the design RST's step-4 list).
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

### 2.14 Design energy gain vs reference move — ledger (2026-09-01)

**The defect.** `DriftSubstepped` (and `ReferenceEnergyChange`) move the
reference energy themselves, so by the time the RF station runs,
`target - reference.total_energy` is identically 0. The station therefore
reported a NON-ACCELERATING machine on a genuinely ramping one: measured on a
20 MeV/turn cycle, `phi_s = 3.141592654` (exactly pi, the stationary-bucket
value) and the symbolic Hamiltonian's `dt` tilt = 0, while `DriftSimple` on
the same cycle gave `phi_s = 2.730076` and a tilt of 2.0e7 eV. The *tracking*
was never wrong — only the analytic layer (`calc_phi_s_main_harmonic`,
`calc_synchrotron_tune_main_harmonic`, both `get_hamilton_symbolic`, and hence
`SymbolicSeparatrixHelper`).

**Two quantities, deliberately separated.** `_last_reference_energy_change` is
how much THIS element moved the reference — it drives the `acceleration_kick`
and must stay 0 here, or absolute energy is double-counted.
`design_energy_gain` (new public property on `RFStationBaseClass`) is the ramp
the RF must SUPPLY this turn — it drives phi_s, Q_s and the Hamiltonians. They
are equal only in the classic `DriftSimple` + station layout.

**The bridge** is `reference.pending_rf_energy_gain`. A naive `+=` / clear on
shared mutable state was WRONG IN THREE WAYS, all found by an adversarial audit
of the first implementation, all reproduced before fixing, all now
mutation-verified in `tests/unittests/physics/test_drifts.py`:

1. **Unbounded growth.** With a reframing element running but no station
   consuming (`active=False`, `each_turn_i>1`, or no station at all) the sum
   piled up: after 6 idle turns the next station reported `1.199988e8` eV
   against a correct per-turn `2.0e7` — 6x too large, growing linearly with no
   bound. FIX: the ledger is TURN-SCOPED. Use
   `reference.add_pending_rf_energy_gain(delta, turn_i)`, which DROPS a total
   tagged with an earlier turn. Never `+=` the attribute directly.
2. **Silent destruction.** `RFManipulationBaseClass.track_reference` (the
   barrier-bucket path) cleared the ledger. It reports no phi_s and no
   Hamiltonian, so clearing there destroyed a drift's entire design gain and
   drove the next station back to 0.0 eV — exactly the failure this work
   exists to prevent. FIX: it does NOT touch the ledger; boundedness comes
   from the per-turn scoping above, not from destruction. Only a real station
   consumes, via `reference.take_pending_rf_energy_gain()`.
3. **Read-order dependence.** `calc_phi_s_main_harmonic` derived from the live
   reference returned exactly pi once the station had tracked — contradicting
   the station's own `design_energy_gain` by 30 deg in the measured case. FIX:
   it reuses `_last_design_energy_gain` when that was computed for this turn.

**A fourth defect was introduced by fix 3 and caught before it landed.** The
reuse was keyed by turn alone, but every beam in a ring shares one station
object, so a second beam was handed the first one's gain — an untracked beam
returned `2.730076471` instead of pi. The cache is now keyed by turn AND a
`weakref` to the beam's own `ReferenceCoordinates`
(`_last_design_energy_gain_reference`); the weak ref is so a station never
keeps a beam's clock alive.

- **Invariant to preserve:** over one turn, the sum of every station's
  `design_energy_gain` equals the reference's total energy change, up to the
  ring-tail carry into the NEXT turn's first station. Measured exact
  (`0.00e+00` relative closure) on a 10-section EX_16-shaped ring from turn 1
  onward; turn 0 is short by the tail (6.897 %), which is the transient, not a
  defect.
- **EX_16 deliberately moved.** Its stations previously saw only ~1/3 of a
  section's ramp (one `DriftSimple` third between two `ReferenceEnergyChange`
  elements). They now see the full section share, which is the physically
  correct number — with 10 stations the kicks must sum to the turn's ramp.
  phi_s and the matched distribution shift accordingly; the example's test has
  no numeric pin and still passes.
- **Mock trap:** `Mock(spec=ReferenceCoordinates)` provides neither
  `pending_rf_energy_gain` nor a usable `take_pending_rf_energy_gain`. Any mock
  reaching `track_reference` needs the attribute set to `0.0` AND the method's
  `return_value` set to `0.0`. Six such sites in `test_cavities.py`.
- **Look-ahead is safe, verified:** every `track_reference` call site that is a
  prediction rather than real tracking uses a copy — `rf_center_grid.py` and
  `solvers.py` take `deepcopy(beam.reference)`, `simulation.py:1636` uses
  `copy()`, and `simulation._exec_track_reference` / `magnetic_cycle.py` build
  their own fresh `ReferenceCoordinates`. None can consume the real ledger.
- **Known limitation (inert):** the accumulate side is direction-blind —
  `DriftSubstepped` / `ReferenceEnergyChange` do not flip `section_i` for a
  counter-rotating beam the way the station does. Harmless today because
  `MagneticCycleByTime.get_target_total_energy` ignores `section_i` entirely;
  commented at both call sites. See §3.1.

### 2.15 Smaller fixes shipped alongside (2026-09-01)

- **Headless `WakeField` fed the profile window as the revolution period.**
  `get_t_rev_init()` returned `profile.profile_duration`, which
  `PeriodicFreqSolver` then adopted as `t_periodicity` — a 5 ns bunch in a
  1 us ring wrapped the wake 200x too often, silently. `WakeField.headless`
  now takes `t_rev=`, and a wrap-at-one-turn solver with no periodicity RAISES
  rather than guessing. `ContinuousMultiTurnTimeDomainSolver` keeps the
  profile-duration fallback — its window IS the turn and it asserts so.
- **`DriftSubstepped.headless()` silently returned a plain `DriftSimple`** (the
  inherited `staticmethod` hard-codes the base class): no sub-stepping, no
  ramp. Overridden; `n_substeps`, alpha and `section_index` now round-trip.
  NOTE `headless` takes no `magnetic_cycle`, so the caller must `configure()`
  one — and must RE-PASS `turn_counter` in that same call, because
  `DriftSimple.configure` rebinds it to its `None` default.
- **`relative_voltage_correction` had no zero-voltage guard**, giving a NaN
  gap voltage for a harmonic driven at `V = 0`; the correction is now zeroed,
  which is what `calc_gap_voltage_with_feedbacks` reproduces anyway.
- **Several feedbacks on one station** share the FIRST one's profile grid.
  Not an error — the usual multi-harmonic setup shares one profile — but now
  a one-shot warning when they differ.
- **`InducedVoltageObservationCR.total_voltage` now INCLUDES the feedback.**
  It previously raised `NotImplementedError` for a station carrying both a
  local wakefield and a cavity feedback; the maintainer confirmed that setup is
  valid, and recording the uncorrected RF drive would drop the whole generator
  contribution. Interpolates onto the wakefield grid when the two differ, with
  a one-shot warning.
- **numba-vs-CUDA whole-feedback equivalence** now has an end-to-end test
  (`test_backend_equivalence_numba_vs_cuda.py`, ~8 s). Agreement 1e-15..1e-17
  against a 1e-11 tolerance; verified non-vacuous (no compared array is
  bit-identical between backends) and shown to catch faults down to 1 ppb.
  See its module docstring for what it does NOT reach.

### 2.16 Two review findings answered, no code change (2026-09-01)

- **`check_fits_in_span` one-bin tolerance is CORRECT**, including for
  `MultiPassResonatorSolver` (which does use it, unconditionally). One bin is
  exact geometry, not float slop: `profile_duration` is the OUTER-EDGE span,
  one `hist_step` wider than the first-to-last-bin-centre extent the stored
  deposits carry charge over, so two consecutive deposits separated by `span`
  first coincide at `profile_duration == span + hist_step`. The worst accepted
  case touches exactly one bin, and only if the edge bins are non-empty — a
  condition the solver already warns about. Reasoning added to the docstring
  so it is not re-raised.
- **`n_substeps` changes the beam map, and that is physics.** The reference
  does NOT move identically across `n_substeps` — it CONVERGES (that is the
  element's purpose; `n=1` is 0.045 t_rf/turn off the converged value). The
  map converges with it, by exactly `eta_0 * gamma**2` times the clock
  correction: measured ratio 0.0066 at transition (where the factor is 0) and
  -0.995 at `alpha_0 = 0` (where it is -1, i.e. the map shift exactly cancels
  the clock shift and the ABSOLUTE arrival time is n-independent — only the
  frame moved). "The map should be independent of `n_substeps`" is the wrong
  invariant. Documented in the class docstring and pinned by a test.

---

## 3. Open items / flagged (NOT done — need decisions)

### 3.1 Counter-rotating / two-beam

- ~~**Two-beam offset passages at `N >= 4` are not accuracy-validated.**~~
  **CLOSED (2026-08-31).** All three two-beam comparisons ran at
  `n_sections = 2` only, which is special twice over: every station sees the
  beams exactly `T_rev / 2` apart, AND the backfill interval is empty at
  every station so the backfill reference walk is never entered — while the
  shipped `rcs_two_beam_example` runs 16 sections and enters it at 14 of
  them every turn. `TestTwoBeamOffsetPassagesManySections` now repeats the
  comparison at four and six sections (static) and four (fast ramp,
  `delta_omega_rf`), against the two-section class's **pre-registered** 0.5 %
  gate rather than a fitted one. Result: 0.128 % on turn 0 falling to
  0.039 %, within 0.001 pp of the two-section numbers — more sections cost
  essentially nothing. Teeth verified by mutation: dropping the last
  backfilled element in `rf_center_grid.py` leaves the ENTIRE two-section
  two-beam class green and fails the new four-section accelerating test.
  (A direction-flip mutation is inert by construction — the symmetric
  half-drift/station/half-drift ring makes element order unobservable in the
  output, as `TestBackfillWalkDirectionConsistency` already documents.)
- **CR-3 equal-time patch path** (deferred by user, 2026-07-08 — "guard
  suffices"; real RCS layouts keep stations off the meeting azimuths, so
  revisit only if a layout needs a meeting-point station): integrating two
  coincident beam currents in the feedback (deposit-sum into the same forward
  segment + envelope rewind/re-advance from a snapshot;
  `calculate_rf_centers_for_backfill` already returns zero cells on exact time
  equality). The design options used to live *only* in an out-of-repo agent
  memory note, which does not travel with a clone — inlined here so they
  survive. **Kick-ordering fork:**
  - *(b1) symmetric one-passage-delayed corrections* — both beams' kicks lag
    by one passage. Preserves the µ⁺/µ⁻ symmetry exactly; was the review
    agent's recommendation.
  - *(b2) pooled kick* via the `PooledInterpolationKick` pattern — exact, but
    structurally riskier: it breaks `element.track` atomicity.
  - *(b3) asymmetric lag* — cheapest, but the first-tracked beam misses the
    other's fresh deposit, ≈ `(R/Q)·ω·q_bunch` ≈ 4.7 % of `V_design` per
    passage.

  Estimated 250–400 LOC in `cavity_feedback.py`, medium risk; single-beam
  bit-identity is preservable because the patch would be gated on a
  second passage at the frontier.
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

### 3.1b Numerics

- ~~**Bounded secular drift, ~0.03 pp/turn.**~~ **FIXED (2026-09-01) — it was
  a sign-and-reference error in the registration phase, not a tolerance.**
  `_accumulate_registration_phase` computed the increment as
  `sum_k (w_k - w_0^(N)) * T_k`, referencing the carrier of the passage that
  ENDS the interval, with the wrong sign. The carried envelope it must
  correct was demodulated against the carrier of the passage that STARTED
  it, so the exact increment is `sum_k (w_prev - w_k) * T_k`. The fix holds
  the previous passage's `_forward_segment_omega_design` in
  `_previous_forward_segment_omega_design` (snapshotted UNCONDITIONALLY,
  outside the gate — inside it, the held carrier goes stale on any passage
  with no backfill centres).
  **Why it hid for weeks:** the old and correct forms differ by a *second
  difference* of the design-frequency programme, which vanishes identically
  for a linear ramp. Only the curvature term survived, so the first-order
  compensation looked right and the residual scaled as `Psi^2.2` — it read
  as an inherent second-order artefact rather than a bug. The derivation
  predicted the residual as `3*(dE_kick/E)*|dPsi|` and matched the running
  code to 1 % (7.437e-3 measured vs 7.5e-3 predicted).
  **Measured (fast ramp, 20 turns, feedback beam-induced vs the multi-pass
  convolution):**

  | n | slope before | slope after | endpoint before | endpoint after |
  |---|---|---|---|---|
  | 2 | +0.03184 | **-0.00255** | 0.668 % | **0.021 %** |
  | 4 | +0.04275 | **-0.00219** | 0.869 % | **0.028 %** |
  | 8 | +0.04690 | **-0.00175** | 0.933 % | **0.036 %** |
  | 16 | +0.04853 | **-0.00145** | 0.951 % | **0.042 %** |

  Before, the slope saturated UPWARD with section count; after, it is
  negative at every `n` and its magnitude SHRINKS with `n`. The n=2
  residual (0.0215 %) now sits BELOW the single-section control (0.0262 %),
  which is the clean statement that no registration artefact is left.
  **Invariants bit-identical**, verified at full float64 repr: n=1 fast
  ramp and n=2 static reproduce their pre-fix values character-for-
  character, because Psi is exactly 0.0 in both.
  **Exactly one pin moved** in the whole feedbacks+impedances suite:
  `TestPIFullTrackingMultiSectionFastRamp::test_pinned_trajectories`
  (driven, two-section, accelerating — the path the fix acts on), by
  4913 V on ~2.98e7 V = 1.65e-4 relative. Regenerated, with the magnitude
  and cause in the test docstring; the reviewer independently re-ran the
  pin generator and confirmed the committed arrays match digit-for-digit.
  **Regression guard:** `test_multiturn_secular_drift_long_horizon` gates
  tightened from `slope < 0.05` / `endpoint < 1 %` to `slope < 0.005` /
  `endpoint < 0.05 %`, plus a non-degeneracy guard (both gates are
  one-sided, so a collapsed comparison would otherwise pass
  spectacularly). Proven to fire: re-introducing the old expression makes
  it fail at 0.03184 against the 0.005 gate.
  **Still open (small):** the counter-rotating two-beam path shares one
  feedback instance between two beams, so `_previous_forward_segment_
  omega_design` is snapshotted by one beam's passage and read by the
  other's. That is believed correct — the cavity does not care which beam
  last passed it — and the two-beam accelerating comparison passes
  unchanged, but the semantics are untested and undocumented.

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
  **RESOLVED (2026-09-01) — and the previous entry here was FALSE.**
  Both were always exactly `0.0` (declared in `__init__`, re-zeroed in
  `on_run_simulation`, never read anywhere). This bullet previously read
  "RESOLVED (2026-08-13) ... Deleted ... Grep now returns no occurrence
  anywhere in `blond/` or `tests/`". **That claim was untrue when it was
  written.** Git history (`git log -S"_phase_offset_frwrd" --all`) shows
  the occurrence count in `cavity_feedback.py` was 4 at *every* commit
  from `d3beab88` (2026-07-16, which introduced them) through `ed2cddbe`
  — the source lines were never removed. What actually happened on
  2026-08-13 was that the entry was written; the only real deletion was
  the single test term (`np.sin` argument), and that landed on
  2026-08-31 in `afd5d96a`. The four source lines were deleted for real
  on 2026-09-01, verified by a zero-reader sweep over `blond/`,
  `tests/`, both muon-collider packages and `docs/`.
  A **merge was NOT the vector** — an explicit audit cleared `e8d978ce`
  (does not touch the file at all) and `e411428b` (its only change under
  `blond/physics/feedbacks/` is a `cupy` → `cupy_` import rename), and
  neither appears in `git log -G"_phase_offset_frwrd"`.
  **The failure mode to learn from:** a resolution was recorded in this
  file without the deletion having been executed, and nothing detected
  it for three weeks. The deletion is therefore now pinned by
  `TestVestigialPhaseOffsetsStayDeleted` in `test_cavity_feedback.py`
  (two `hasattr` assertions plus a scan of the module source), so a
  re-introduction fails loudly instead of quietly landing. **When you
  record something here as deleted, verify it with a grep in the same
  breath — this entry is the proof that the note alone is not evidence.**
  The same audit checked the other 55 absence claims in this file
  against the tree: all of them hold.
  **Read the grep output before you panic (2026-09-02):**
  `grep -rn _phase_offset_frwrd --include=*.py .` returns **six** hits, and
  all six are `TestVestigialPhaseOffsetsStayDeleted` itself in
  `tests/unittests/physics/feedbacks/test_cavity_feedback.py` — the two
  `hasattr` assertions, the source scan, and their names. Production
  `blond/` is clean. An absence pinned by a test necessarily names the
  thing it forbids; that is the guard working, not a regression.
- **`_reference_turn_offset` was write-only — DELETED (2026-09-01).**
  Set to `-1` / `0` in `RFCenterGridMixin.get_passed_time_forward_direction`
  and defaulted in `__init__`, read nowhere. The backfill walk that would
  logically consume it instead computes a *local* variable of the same
  name (`rf_center_grid.py`, `get_time_omega_array_backfill`) and reads
  only that — the local is live and was deliberately left untouched, so
  do not "restore" the attribute if you meet the local. Removed from the
  module docstring's state list too.
- **`MultiPassResonatorSolver._simulation` was unused — DELETED
  (2026-09-01).** Assigned in `__init__` and in the wakefield-init hook,
  never read by that class. NOTE the trap: `_simulation` is genuinely
  live on the *sibling* solver classes in the same file
  (`InductiveImpedanceSolver`, `PeriodicFreqSolver`,
  `TimeDomainFftSolver`, `ContinuousMultiTurnTimeDomainSolver`), so any
  future sweep must attribute hits by receiver before concluding
  anything. ~~**Still open:** the same audit found
  `SingleTurnResonatorConvolutionSolver._simulation` is dead in the same
  way (assigned, never read); it was out of scope and is untouched.~~
  **CLOSED (`cdca671b`, 2026-09-01):** that second dead attribute was
  deleted too — both its `__init__` declaration and the assignment in
  `on_wakefield_init_simulation`. Verified 2026-09-02 by scanning the
  class body: no `_simulation` in
  `SingleTurnResonatorConvolutionSolver`. The receiver trap above still
  stands, and a bare `grep self._simulation solvers.py` still returns ten
  live hits on the four sibling classes.
- ~~**The extracted mixins** are still pure moves; promoting them to
  composed collaborators is the natural follow-up.~~
  **DECIDED (2026-08-31): stay mixins, state the host instead.** Composition
  was investigated and rejected on measurement, not taste:
  `RFCenterGridMixin` touches 25+ distinct host attributes (14x
  `_reference_state_until_tracked`, 11x `_parent_rf_station`, 10x
  `_segments`) and *mutates* host state (`_segments`, `_rf_centers`,
  `_last_tracked_*`), so a collaborator needs either a back-reference (the
  same coupling with an extra hop) or a 25-argument call — it would
  relocate the coupling, not remove it. That is the same argument
  `generator_regulation.py`'s module docstring already makes for the kernel
  marshalling. What WAS wrong was an inconsistency: the self-typing design
  (`docs/superpowers/specs/2026-07-23-rf-center-grid-mixin-self-typing-design.md`
  — **UNTRACKED**: `git ls-files docs/superpowers` returns nothing and it is
  not gitignored either, so this path exists only in the original working
  tree and is absent from any fresh clone. The decision is fully restated in
  this bullet, so nothing is lost if you do not have the file; see §6)
  was applied to `RFCenterGridMixin` (16 annotations) and never to
  `GeneratorRegulationMixin` (0), and nothing pinned it. Both mixins now
  annotate every method's `self` as `IQCavityFeedbackTimingClass`, with the
  host imported only under `TYPE_CHECKING` (the host inherits from the
  mixins, so a runtime import is a cycle), pinned by
  `TestMixinsDeclareTheirHost` in `test_cavity_feedback.py` — parametrised
  over both mixins so they cannot drift apart again.
- **P6** (RF-parameter view mixin) skipped per user.
- ~~Full Sphinx doc build not yet run~~ **RESOLVED (first green 2026-08-13;
  re-run green 2026-09-01)**: built green (`build succeeded`, exit 0, zero
  warnings) under `-W` + `nitpicky = True`. The 2026-09-01 re-run covers all
  the §2.14–§2.16 work, so the RSTs and docstrings as they stand today are
  green — see §5. One `-W` failure was found and fixed en route on
  2026-08-13: a `:meth:` role on the private `_compose_coarse_sum` in
  `cavity_response`'s docstring (a role on an underscore-leading member never
  resolves -- use ``literal`` markup).
  **How to run it.** The canonical command (`CLAUDE.md` *CI gates*,
  `CONTRIBUTING.md` *Documentation*) is `cd docs && bash create_docs.sh`;
  that is what CI uses, and `docs/create_docs.sh` is the tracked POSIX entry
  point.
  Requires graphviz `dot` on PATH. Run it **ONCE, sequentially, never looped
  or concurrently** — a second build racing the first wipes the shared
  `docs/examples/` and `docs/_build/` dirs mid-flight and produces spurious
  warnings. That rule is OS-independent and is the real lesson here.
  If your tool's shell cwd does not persist between calls, invoke it as one
  command: `bash -c 'cd docs && bash create_docs.sh'`.
  *Windows note:* `docs/create_docs.bat` is a convenience port of the same
  script (it `cd`s to its own directory, so any absolute invocation works,
  e.g. `cmd //c "<abs-path-to>\BLonD\docs\create_docs.bat"` from Git Bash). It
  additionally auto-activates a venv in the OUTER repo — one developer's
  layout, not a requirement.
  **CORRECTED 2026-09-02:** earlier revisions of this bullet and of §6 said
  the working-tree copy was locally edited to point at `.venv314` and must
  not be committed. That edit is **gone** — `git status docs/create_docs.bat`
  is clean and its line 23 activates `..\..\.venv\Scripts\activate.bat`,
  literally, with no glob. Which now means the guard simply no-ops: the
  outer venv is `.venv_312` (§0), so nothing matches and the script falls
  through to whatever `python` PATH resolves to — harmless if you activated
  the venv yourself, a Microsoft Store stub if you did not (the same trap as
  `check copyright`, above).
  No new top-level exports were added, so `ASSIGNED_CATEGORIES` needs no
  update.
- ~~RST/source name drift~~ **RESOLVED (2026-08-12)**: both RSTs now use
  the backfill vocabulary throughout and `envelope_kernel.py` carries no
  time-sense "reverse" — see the resolved-stragglers note in §1.3.

### 3.3 Resolved / decided (kept as records, not as open work)

- **`MultiPassResonatorSolver` fixed-frequency (`retune_to_rf=False`)
  precision — NOT a solver defect (2026-08-31).** With no retuning the carried-wake phase is just
  `omega_0 x (arrival gap)` and the phase-clock rotation is identically
  zero, so the accuracy is set entirely by how precisely the *reference
  clock* reports arrival times. `DriftSimple.track_reference` advances it
  with a single `beta` across the whole arc (`beta` steps only at the
  cavity): ~0.22 `t_rf` off over five turns at 4 GeV single-section, i.e.
  radians of wake phase. The lever already ships — `DriftSubstepped`
  (`blond/physics/drifts.py`, committed with `TestDriftSubstepped` and
  `TestFixedFrequencyWakeWithSubsteppedFrame`; an out-of-repo agent memory
  note — not part of this checkout, so you cannot open it — said "staged, not
  committed", which was stale) — and matches the analytic fixed-frequency
  reference to machine precision. The gap was **discoverability**: nothing
  in `solvers.py` mentioned it. The `delta_f` parameter doc and a
  *Frame-time fidelity* note in the class docstring now do, including why
  the fundamental mode needs the opposite treatment (`retune_to_rf=True`
  plus the phase clock, not a finer frame).
  **API note (2026-09-01, re-checked against the code 2026-09-02):**
  retuning is now stated by the explicit
  `retune_to_rf` boolean, defaulting to `False`; `delta_f` is a pure
  frequency offset in [Hz] that applies in BOTH modes (on the design
  frequency of every pass when retuning, on the constructed centre
  frequency once when not, via `_apply_fixed_frequency_offset`, which
  snapshots `_constructed_center_frequency` so re-running late init cannot
  stack the offset twice). Neither is inferred from the other, and the old
  `delta_f=None` / `delta_f=0.0` mode encoding is gone with no
  compatibility path.
  **Two corrections to how that landed.** (a) `cdca671b` shipped an
  intermediate API — `retune_to_rf=None` inferring the mode from `delta_f`,
  plus a `ValueError` on `retune_to_rf=False` with a nonzero `delta_f`.
  `ec159a87` deleted both: there is now **no validation at all** on this
  pair, and a fixed-frequency resonator carrying a `delta_f` is an ordinary
  supported configuration, not an error. Any note promising that
  `ValueError` describes a tree that existed for one commit.
  (b) "`None` is not accepted for either" overstates it, and in the
  direction that matters: `delta_f=None` does fail, but only incidentally,
  as a `TypeError` out of `float(None)`; `retune_to_rf=None` is **silently
  coerced to `False`** by `bool(None)` and runs a fixed-frequency
  resonator. So the dangerous migration case — a ported call whose mode
  used to be inferred — fails quietly, not loudly. Verified by
  construction, 2026-09-02.

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
  method's two `TODO`s (`beam_feedback.py` ~334/337) and its message's
  "open design task" phrasing predate the ruling, so message and tests
  must move together in the next code pass. **Re-checked 2026-09-02: that
  phrase is pinned verbatim in TWO places, not the one named here** —
  `TestCavitySumPhaseGuard` in
  `tests/unittests/physics/feedbacks/test_beam_feedback.py` *and* an
  `assertIn("open design task", message)` in
  `tests/unittests/physics/feedbacks/accelerators/lhc/test_beam_feedback.py`.
  Rewording the message breaks both.
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
  `d > 2`. Pinned by `test_decay_hard_cap_forbids_sign_flip`.
  **NOTE (2026-08-31):** an out-of-repo agent memory note (per-machine, not
  part of this checkout — you will not be able to open it) still described
  this as an open judgment call long after it shipped, and it was re-reported
  as open on that basis. That memory was corrected; THIS file was right all
  along — check it before trusting any memory note about open work.
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

`blond/physics/feedbacks/` (plus `__init__.py` and the `accelerators/`
subpackage, both listed below). Line counts used to be quoted here and rotted
within weeks; run `wc -l blond/physics/feedbacks/*.py` if you want current
sizes.

| module | holds |
|---|---|
| `cavity_feedback.py` | `IQCavityFeedbackBase` + `IQCavityFeedbackTimingClass(IQCavityFeedbackBase, RFCenterGridMixin, GeneratorRegulationMixin)`. Per-turn orchestration: `_track` + its **nine** phase methods (§2.11, incl. `_update_frame_rotations`), `circuit_track` → `_circuit_track_cells{,_python,_kernel}` + `_resolve_fine_grid_voltage`, the kernel glue (`_coarse_step_sizes`, `_kernel_step_multipliers`, `_kernel_beam_current`), `cavity_response` (advances the two source-split envelope components, §2.13), `_compose_coarse_sum`, `_advance_coarse_voltage`, `cavity_response_fine`, `calculate_rf_beam_current_partial`, `reset_arrays` (incl. `_generator_active` refresh and the gen-component seeding), `on_run_simulation`, `_validate_multi_harmonic_slot`, `_check_fine_grid_initial_condition_is_causal`, the pre-fill call. `_check_step_sizes`, `_check_beam_kick_magnitude`, `_check_beam_kicks` are thin wrappers delegating to `self._euler_guard` |
| `rf_center_grid.py` | `RFCenterGridMixin` — coarse `rf_centers` construction: the forward and **backfill** reference walks, `_generate_rf_centers`, segment generation (`_append_segment` / `_clear_segments` / `_rebuild_grid_arrays` / `_close_previous_turn_grid`), `_preceding_segment_residual`, `_validate_grid`, and the two direction selectors (`_reference_list_for_direction`, `_own_index_for_direction` — the *space*-sense reverse, §1.3). `_segments` is the single source of truth; the flat arrays are derived. Its module docstring is the canonical statement of the backfill-vs-reverse rule and of the design-clock-only geometry |
| `rf_center_segment.py` | The two value classes: `RFCenterSegment` (all four fields load-bearing — see the correction in §2.11 — with the ≥ 2-centres and `residual ∈ [0, duration]` validation) and `PerTurnGridSpan` (`n_backfill_centers`, `n_forward_centers`, `residual_from_backfill_span`). Both are imported by `cavity_feedback.py` |
| `cavity_solvers.py` | **mucol-only.** Fine-grid solvers `cavity_response_sparse_matrix` (forward-Euler) and `..._second_order` (Crank-Nicolson); the coarse-step arithmetic `coarse_step_exponent`, `euler_voltage_multiplier`, `exponential_voltage_multiplier`, `exponential_drive_weight` (spelled once for both the reference and the kernel path); `pretrack_fill_voltage`; and `ForwardEulerValidityGuard` — the discretisation tripwires, beside the solvers they certify. Its module docstring owns the `omega_times_dt` naming rule (§1.4) |
| `envelope_kernel.py` | numba host kernel `envelope_pi_scan` + `inactive_controller_scan_state` — the sequential coarse-envelope + PI recursion; solver-agnostic and byte-identical to the Python reference. Since §2.13 it advances the two source-split components, composes the demod-frame sum per cell and forms the PI error in the kick frame; the signature carries the component in/out arrays, the `generator_active` gate and the **three** per-passage rotation scalars (`_generator_frame_rotation`, `_kick_frame_rotation`, `_pi_error_frame_rotation` — this row said two until 2026-09-02). Reached through the **controller's** `supports_envelope_scan` capability, not called by the feedback directly |
| `generator_regulation.py` | `GeneratorRegulationMixin` — `_controller_active`, `pi_setpoint`, `_validate_voltage_setpoint`, `generator_power`, `_update_generator_current` (forms the PI error in the KICK frame via `_kick_frame_rotation`, §2.13), `_limit_fine_grid_generator_current`. **What it does NOT own** (and its module docstring says so): the compiled envelope scan and the per-cell stepping decision stay on the timing class, because they need **every** coarse grid (the summed, generator- and beam-sourced antenna voltages plus the generator current) and **all five** values carried across the turn boundary (`_last_val_ant_voltage`, `_last_val_ant_voltage_gen`, `_last_val_ant_voltage_beam`, `_last_val_generator_current`, `_last_val_beam_current` — this row said "both coarse grids and the three values" until 2026-09-02, a pre-envelope-split count the module docstring had already outgrown), and because the scan depends on `pi_setpoint` staying *unevaluated* on a span the controller sits out (that property may reach through to the parent station, which a no-beam backfill span must not require) |
| `generator_current_controller.py` | `GeneratorCurrentController` ABC + `GeneratorCurrentPIController`; the envelope-scan capability hooks (`supports_envelope_scan`, `envelope_scan_kernel`, `envelope_scan_state`, `absorb_envelope_scan_state`); `current_limit_from_power`, `clamp_magnitude` |
| `beam_current.py` | `low_pass_filter`, `rf_beam_current` (unified; keyword-only coarse args; no wrap-around; `check_fits_in_span` + `hist_step`/`sampling_time` + `_check_coarse_index_bounds` guards) |
| `beam_feedback.py` | the surviving phase loop (`BeamFeedbackBase`), incl. `cavity_sum_phase`, whose `NotImplementedError` guard is the permanent contract — coupling is a deliberate non-goal (§3.3) |
| `iq.py` | `cartesian_to_polar`, `polar_to_cartesian` |
| `base.py` | `FeedbackBaseClass` / `LocalFeedback` / `GlobalFeedback` (unchanged) |
| `accelerators/{lhc,ps,psb,sps}/beam_feedback.py` | the machine-specific `BeamFeedbackBase` subclasses (four modules + `accelerators/__init__.py`), each with a matching test module. `lhc` and `sps` call `cavity_sum_phase`, i.e. they hit the §3.3 `NotImplementedError` contract when the station carries a cavity feedback. **This subpackage survived the 2026-07-25 LHC/SPS purge** — that purge removed the LHC/SPS *cavity* feedbacks, not the phase loops |
| `__init__.py` | package init (9 lines) |
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
- `tests/unittests/physics/feedbacks/accelerators/mucol/` — the mucol test
  modules (17 as of 2026-09-01; `ls .../mucol/test_*.py` is authoritative,
  and the maintained inventory is the test RST's *Test modules* section)
  plus the shared harness (`mucol_cav_fdbk.py`, `support.py`,
  `stubs.py`, `conftest.py`, `plotting.py`, `fdbk_testing/`). The unused
  debug method `plot_antenna_voltage` lives in `plotting.py` as a function.
- `tests/unittests/physics/feedbacks/accelerators/{lhc,ps,psb,sps}/` — one
  `test_beam_feedback.py` each, mirroring the four phase-loop modules above.
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

- **Re-measured 2026-09-02 at tip `ec159a87`**, `.venv_312` (Python
  3.12.8), numba backend, no GPU present. Every number below was run for
  this entry, not carried over:

  | scope | result | previously recorded |
  |---|---|---|
  | `tests/unittests/` (full) | **1621 passed / 88 skipped / 176 subtests / 0 failed**, 206.8 s | 1612 / 63 / 175 (2026-09-01) |
  | `physics/feedbacks/accelerators/mucol` | 261 passed / 6 skipped / 137 subtests, 41.8 s | 259 / 5 / 136 |
  | `physics/feedbacks` | 572 passed / 8 skipped / 142 subtests, 38.7 s | 550 collected (2026-08-31) |
  | `physics/{feedbacks,impedances}` | 737 passed / 15 skipped / 159 subtests, 71.1 s | 692 / 16 / 147 |
  | outer `test_rcs_two_beam_example.py` | 14 passed, 38.2 s | 14 passed |

  The `+25` on the full run's skip count is **environment, not
  regression**: this machine has no CuPy, so every `cupy`/`cuda`-marked
  test skips. Read the skip counts as a property of the box, not of the
  tree. The outer-repo row is the cross-repo pin of §0 and is unavailable
  in a standalone BLonD clone; it was run from the outer root with
  `MPLBACKEND=Agg` and left that working tree byte-identical
  (`git status --porcelain` diffed before and after).
- **Pre-commit was NOT re-run over `--all-files` for this entry, on
  purpose.** Four of the hooks (`ruff-format`, `ruff-check --fix`, isort,
  `trailing-whitespace`) rewrite what they touch, and this checkout carries
  untracked work — `blond/examples/notebooks/getting_started.py`, two
  `blond/legacy/blond2/beam/*_old.py`, and a stray untracked `unittests/`
  tree at the repo root — that an `--all-files` sweep would silently
  reformat. Run it on your own changed files
  (`pre-commit run --files <paths>`) instead.

  Run that way over the nine files this 2026-09-02 pass changed, every
  content hook passed (`trim trailing whitespace`, `fix end of files`,
  `check for merge conflicts`, `check for added large files`,
  `don't commit to branch`; the python-only hooks had no files to check).
  The two `language: system` hooks both failed with the bare-`python` 9009
  — `check copyright` **and `sync-agent-docs`**, which §0 previously listed
  as still trustworthy. And re-running `precommit_check_copyright.py` by
  hand with the venv interpreter no longer exits 0 as §0 promised: it exits
  1 on the untracked `blond/examples/notebooks/getting_started.py`. Both
  corrections, with what to do about them, are in §0.
- HISTORY: full run 1612 / 63 / 175 on 2026-09-01, up from 1590 at the
  start of that session (+22). `tests/unittests/physics/feedbacks`
  collected 550 tests on 2026-08-31 (was 525 on 2026-08-12, 513 on
  2026-08-11).
- **HISTORY**: the last full battery run recorded here (mucol + LHC
  comparisons + impedances) was **492 passed**, the only failures being the
  pre-existing SPS `TestTravelingWaveCavity` ones (`test_vind`,
  `test_beam_fine_coarse`), since fixed SPS-locally (90° IQ rotation) and
  unrelated to mucol. That number predates the LHC-suite removal (§2.10) and
  the 2026-08-08…11 pass; treat it as a historical marker, not a target.
- Every production sign/gate change is **mutation-verified** (see §2.4–2.7).
- The P1–P5 partition was **byte-identical** (pure moves), verified by the
  full battery + per-step ruff/numpydoc/import/MRO checks.
- Docs: both RSTs are maintained, and the full `-W` + nitpicky Sphinx build
  is **green (last run 2026-09-01, `build succeeded`, no warnings)**. That
  run came AFTER all the 2026-09-01 work — §2.14–§2.16, the new public
  `RFStationBaseClass.design_energy_gain` property, the
  `WakeField.headless(t_rev=)` kwarg, the `DriftSubstepped.headless` override
  and the docstring rewrites in `cavities.py` / `drifts.py` / `solvers.py` /
  `reference_clock.py` — so it covers the tree as it stands at the
  `064305e3` tip. Anything committed after that needs a re-run before
  the MR: under `-W` + nitpicky a single new unresolved cross-reference fails
  the whole build. Run it ONCE, sequentially — `cd docs && bash
  create_docs.sh`; see §3.2 for why and for the Windows variant.
- **The doc build is therefore STALE as of 2026-09-02, and could not be
  re-run here.** Four commits have landed since `064305e3` — `534eb7c2`,
  `550df2f8`, `cdca671b`, `ec159a87` — and between them they rewrote
  `docs/feedbacks/mucol_cavity_feedback.rst` (+447) and
  `docs/tests/mucol_cavity_feedback_tests.rst` (+1786) wholesale and
  rewrote docstrings across `cavity_feedback.py`, `generator_regulation.py`,
  `observables.py` and `solvers.py`. None of that has been through `-W` +
  nitpicky. The blocker is the environment, not the tree: `.venv_312` has
  **no sphinx and no numpydoc** installed (§0), so `create_docs.sh` cannot
  run in it at all — `graphviz`/`dot` *is* on PATH, so that prerequisite is
  satisfied. **Do not treat the green above as covering the current tip.**
  Someone must `pip install -e ".[doc]"` and run the build once before the
  MR; on this much new RST it is the single most likely thing to fail.

---

## 6. Commit status

**HISTORY — superseded 2026-08-07.** This section used to read "Nothing is
committed" and proposed a four-way commit grouping. That grouping was never
used: the work was checkpoint-committed incrementally on
`blonder_feature/mucol_feedbacks`, and `blonder` has since been merged in
(`52a03664`). Check `git log` rather than this section for the current state;
as of 2026-08-11 the tip was `d93eaf3c`. Re-run the full battery
before/after any reshuffle.

**As of 2026-09-02** the tip of `blonder_feature/mucol_feedbacks` is
**`ec159a87`**, four commits past the `064305e3` this section used to name:

| commit | what it is |
|---|---|
| `534eb7c2` | *context update* — this file only (+445) |
| `550df2f8` | *docm* — both mucol RSTs rewritten (+447 / +1786) and `blond-migration`'s `api_mapping.md`; no production code |
| `cdca671b` | **code.** The registration-phase sign-and-reference fix (§3.1b) *and* the first `retune_to_rf` API (§3.3). Touched no doc file at all |
| `ec159a87` | **code.** Simplified that API to two independent kwargs, deleting the inference path and the `ValueError` `cdca671b` had just added (§3.3) |

Note the ordering trap for anyone reading the history: the two doc commits
land BEFORE the code they describe, so `534eb7c2`/`550df2f8` document a tree
that did not exist yet. §3.1b and the RSTs are nevertheless current, because
the code landed as written. What was NOT updated in the same breath is
everything `cdca671b` invalidated elsewhere in this file — corrected
2026-09-02 in §2.13, §3.2, §3.3, §4 and §5.

**As of 2026-09-01** the tip was
`064305e3`, and the §2.14/§2.15 work is IN it — verified by reading the blobs
at that commit, not by assuming: `take_pending_rf_energy_gain` present in
`reference_clock.py`, `_last_design_energy_gain_reference` present 5x in
`cavities.py`, and all three named ledger regression tests present in
`test_drifts.py`. `test_station_readout_edge_cases.py` is tracked.

**Important for anyone bisecting:** `15315271` committed the FIRST version of
the design-gain ledger — the one carrying all three defects described in
§2.14. Only `064305e3` carries the fixes. Do not ship or benchmark anything in
the range `15315271..ed2cddbe`; phi_s and the symbolic Hamiltonian are wrong
there in the reframing-element layouts.

**Working tree, re-checked 2026-09-02.** The old list here (this file
uncommitted, plus a machine-local `docs/create_docs.bat` edit, plus
`docs/superpowers/`) is **entirely superseded** — this file and
`create_docs.bat` are both committed and clean, and `docs/superpowers/`
does not exist in this checkout at all. What `git status --short` actually
shows is four untracked entries, none of them mucol work:
`blond/examples/notebooks/getting_started.py`,
`blond/legacy/blond2/beam/beam_old.py`,
`blond/legacy/blond2/beam/distributions_old.py`, and a stray `unittests/`
directory at the repo root — a duplicate test tree, untracked and *not*
gitignored, most likely dropped there by a pytest run from the wrong cwd.
Worth deleting after you have confirmed it holds nothing of yours; it is
the kind of thing that quietly gets committed. The outer repo carries its
own unrelated in-flight edits.

**Portability note:** almost everything an agent needs is in-repo (this file,
`CLAUDE.md` / `AGENTS.md`, and `.agents/skills/` + `.claude/skills/`, still
14 tracked files — re-counted 2026-09-02 with `git ls-files .agents .claude`,
now covering **three** skills: `blond-dev`, `blond-assistant` and
`blond-migration`). `CLAUDE.md` and `AGENTS.md` are byte-identical **to each
other** (md5 `ca6e4125…`, 20207 bytes) and identical to the **body** of their
source `.agents/skills/blond-dev/SKILL.md` (md5 `a6299993…`, 20591 bytes):
the `sync-agent-docs` generator replaces SKILL.md's 4-line YAML frontmatter
with a 6-line AUTO-GENERATED banner. Edit the skill, never the copies.

**Do not verify that with a plain `diff` on Windows.** The generator writes
its targets with `newline="\n"`, while `.agents/skills/blond-dev/SKILL.md`
comes out of a `core.autocrlf` checkout as CRLF (measured: 315 CR bytes in
SKILL.md, **0** in `CLAUDE.md`). Every line therefore differs and
`diff SKILL.md CLAUDE.md` reports one whole-file hunk — `1,315c1,317` here —
not the single header hunk earlier revisions of this note promised. Use
`diff --strip-trailing-cr`, or `diff <(tr -d '\r' < SKILL.md) CLAUDE.md`,
which does show exactly the frontmatter-for-banner hunk and nothing else
(verified 2026-09-02). The same asymmetry makes the byte counts above
platform-dependent: trust the CR-stripped diff, not the sizes.

Two things do **not** travel and are named here so you do not go hunting:
per-machine assistant memory under `~/.claude/projects/.../memory/` (outside
the repo by design — hence this file is the handoff surface), and the
untracked `docs/superpowers/` planning docs cited in §3.2 — **absent from
this checkout too** (`ls docs/superpowers` → no such directory, 2026-09-02),
so treat every reference to them as unresolvable rather than as something you
have failed to find. The outer repo has a `docs/superpowers/specs/` of its
own holding three *different* specs; the mixin self-typing design cited in
§3.2 is in neither. Cross-repo pins (`rcs_two_beam_example`,
`test_rcs_two_beam_example.py`) live in the OUTER repo, not in `BLonD/` —
see §0 *Environment*.
