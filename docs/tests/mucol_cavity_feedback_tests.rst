.. _mucol_cavity_feedback_tests:

Muon Collider Cavity-Feedback Test Suite
========================================

This page documents the test suite for the muon-collider RF cavity feedbacks.
Most of the files live in the source tree under::

    tests/unittests/physics/feedbacks/accelerators/mucol/

and exercise the longitudinal-beam-loading models used for the muon-collider
Rapid-Cycling Synchrotrons (RCS):

* the I/Q cavity-feedback timing model
  (``blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass``),
* the standalone PI generator-current controller
  (``blond.physics.feedbacks.generator_current_controller.GeneratorCurrentPIController``)
  and the feedback's controller-driven mode
  (``IQCavityFeedbackTimingClass(controller=...)``),
* the cavity-response solvers (``blond.physics.feedbacks.cavity_solvers``)
  and the beam-current demodulation
  (``blond.physics.feedbacks.beam_current``), and
* cross-checks of the feedback against the multi-turn resonator wake
  (``blond.physics.impedances.solvers.MultiPassResonatorSolver``).

The tests are written for the ``unittest`` framework but are collected and
run with ``pytest``. They share a small set of mock objects and numeric helpers
(see `Support modules`_), so the ``mucol`` directory is a package (it and every
directory above it carry an ``__init__.py``) and the test modules use
**package-relative imports** (``from .stubs import StubBeam``) for those shared
helpers.

The same production code is also unit-tested one directory higher, in
``tests/unittests/physics/feedbacks/`` itself. Those modules are documented
here too, under `Shared feedback-machinery tests`_, because they pin
invariants the mucol suites rely on and nothing else does. Three further
classes that the feedback depends on live with the subsystems they belong to
(profiles, resonator solvers, RF stations); they are documented under
`Guards tested outside the feedbacks tree`_ with their home paths.

.. contents:: Contents
   :local:
   :depth: 2


Common physics context
-----------------------

Most tests are parametrised with RCS1-like single-cavity numbers: a shunt
impedance ratio ``R_over_Q = 518``, a loaded quality factor ``Q_L`` of order
``1e4`` (single-pass tests) or ``1.29e6`` (multi-turn tests), harmonic number
``25900``, ring circumference ``5990`` m, a ``mu_plus`` beam of intensity
``2.7e12`` and a design voltage ``V_design = 30`` MV near the ``63`` GeV
operating point.

Several recurring techniques are worth knowing when reading the tests:

Non-driven / operating-point cavity
    With the generator current set to zero (and ``initial_voltage = 0``) the
    feedback's antenna voltage is *only* the beam-induced voltage, so it can be
    compared directly with a resonator wake. When a full ``Simulation`` is
    tracked, a cold cavity instead trips the coarse-grid beam-kick magnitude
    check, so the cavity is held at its operating point
    (``initial_voltage = V_design`` with the matched generator current
    ``I_g = V / (2 (R/Q) Q_L)``) and the beam-induced part is isolated by
    subtracting a **zero-intensity reference run** (exact, by linearity of the
    cavity equation).

Lab-frame projection
    The feedback works with the complex I/Q envelope of the antenna voltage,
    whereas the resonator solvers return a real, lab-frame induced voltage.
    The envelope is projected back with
    ``-Im[V_ant * exp(i omega_rf t)]`` (or ``-Re[...]``); see
    ``support.lab_frame_voltage``.

Accumulated phase under acceleration
    When the beam accelerates, the RF frequency slips turn to turn, so the
    carried cavity wake winds up the *accumulated* phase
    :math:`\theta(t) = \int \omega(t)\,dt` rather than a fixed
    :math:`\omega_0 t`. The phase-under-acceleration tests certify exactly this.


Test modules
------------

``test_mucol_cav_fdbk.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unit tests for the I/Q cavity-feedback timing class
(``IQCavityFeedbackTimingClass``): the discrete step-size sanity checks, a
single-turn benchmark of the beam-loading response, the cavity pre-fill /
injection matching, the exponential coarse solver, the shared coarse-step
arithmetic behind both propagator paths, the constructor validation of an
explicit ``voltage_setpoint`` and the causality of the fine-grid initial
condition.

**Class** ``TestCavityFeedback`` -- step-size sanity checks. ``setUp`` builds a
feedback instance with RCS1 four-station parameters on a mocked
``StaticProfile``.

``test_circuit_track_applies_delta_omega_phase_shift``
    Drives ``circuit_track`` with zero generator/beam current and a constant
    step grid, and checks the antenna voltage evolves purely by the per-step
    complex multiplier ``1 - 0.5 omega dt / Q_L + 1j delta_omega dt``.
``test_step_size_check_warns_for_large_decay_per_step``
    ``_check_step_sizes`` warns when the per-step decay sits between the soft
    (0.1) and hard (1.0) limits.
``test_step_size_check_warns_for_large_detuning_phase_per_step``
    Warns when the per-step detuning phase ``delta_omega * dt`` exceeds the
    soft limit.
``test_step_size_check_no_warning_for_small_step_parameters``
    No warning when both per-step parameters are well below the limit.
``test_cavity_response_warns_for_large_beam_kick``
    ``cavity_response`` warns when the relative beam kick is between the soft
    (0.1) and hard (1.0) limits.
``test_cavity_response_no_warning_for_small_beam_kick``
    No warning when the relative beam kick is negligible.
``test_step_size_check_raises_for_unphysical_decay_per_step``
    Raises ``ValueError`` when the per-step decay exceeds the hard limit of
    1.0 (the Euler decay factor ``1 - decay_per_step`` then turns negative,
    so the discretised voltage inverts every step -- unphysical, since the
    exact factor ``exp(-omega dt / (2 Q_L))`` is always positive). Checked
    just above the cap and far beyond it.
``test_step_size_check_raises_for_unphysical_detuning_phase_per_step``
    Raises when the per-step detuning phase exceeds the hard limit.
``test_step_size_check_fires_on_run_simulation``
    End-to-end companion: an unphysical detuning aborts the run-start
    initialisation inside ``on_run_simulation`` with a real RF station and a
    stubbed beam/simulation.
``test_cavity_response_raises_for_unphysical_beam_kick``
    Raises when the beam-induced kick exceeds the previous antenna voltage.
``test_decay_hard_cap_forbids_sign_flip``
    Pins the hard cap at the sign-flip boundary: a per-step decay of 0.9
    (Euler factor still positive) only warns, while 1.1 -- negative factor,
    yet ``|factor| < 1`` and hence still contracting -- raises, with a message
    naming the sign inversion and ``exponential_coarse_solver_enable=True``
    as the sanctioned option for such steps.

**Class** ``TestFineGridResonatorBenchmark`` -- benchmarks the single-turn
(fine-grid) beam-loading response, with the generator current zeroed, against
an independent ``Resonators`` induced-voltage model
(``SingleTurnResonatorConvolutionSolver``) on a real Gaussian-plus-noise
``mu_plus`` beam. Agreement is checked on shape (correlation > 0.999),
amplitude scale (~1) and waveform (NRMSE < 1 %).

``test_fine_grid_matches_resonator_on_resonance``
    On resonance (``delta_omega = 0``).
``test_fine_grid_matches_resonator_positive_detuning``
    Positive detuning (``delta_omega = 5e6``); the phase shift matches a
    detuned resonator.
``test_fine_grid_matches_resonator_negative_detuning``
    Negative detuning (``delta_omega = -2e7``).

**Class** ``TestCavityPrefill`` -- the feedforward cavity pre-fill / injection
matching. The no-beam, constant-current cavity fills from cold as
``V(t) = V_ss (1 - exp(lambda t))``; the helper
``blond.physics.feedbacks.cavity_solvers.pretrack_fill_voltage`` returns the complex
seed antenna voltage, and ``n_pretrack`` / ``injection_voltage`` on
``IQCavityFeedbackTimingClass`` route it through ``on_run_simulation``. The PI
controller (if any) does not act during the fill.

``test_steady_state_fill_on_resonance_matches_two_r_q_ql_ig``
    Without ``injection_voltage`` the fill converges to
    ``V_ss = 2 (R/Q) Q_L I_g`` on resonance.
``test_injection_voltage_seeds_at_the_requested_magnitude``
    With ``injection_voltage`` the seed magnitude equals that target.
``test_unreachable_injection_voltage_raises``
    A target above the steady-state fill cannot be reached, so it raises.
``test_fill_seed_is_an_equilibrium_of_the_coarse_step``
    A no-beam cavity started at the fill seed does not drift (the seed is the
    exact fixed point of the coarse Euler step).
``test_n_pretrack_seeds_init_voltage_with_the_fill``
    ``on_run_simulation`` replaces ``_init_voltage`` with the pre-fill seed.
``test_injection_voltage_without_n_pretrack_raises``
    ``injection_voltage`` without a ``n_pretrack`` budget raises.
``test_fill_seed_uses_the_design_clock_under_an_rf_offset``
    The seed is the **design-clock** fixed point
    ``V* = -(R/Q) omega_design I_gen / lambda(omega_design)``, not the
    one at the actual RF frequency: the coarse recursion drives every
    step at ``calc_omega_rf_design``, so evaluating the fill at
    ``omega_design + delta_omega_rf`` misses the fixed point by
    ``O(delta_omega_rf / omega)`` and reintroduces the injection
    transient the pre-fill exists to remove. A detuning
    (``delta_omega = 3e5``) is what makes the seed frequency-dependent at
    all -- on resonance ``V_ss = 2 (R/Q) Q_L I_g`` carries no ``omega``.
    With a one-permille offset programmed on the station (and a guard
    that the two clocks really disagree), 300 no-beam coarse steps must
    leave the voltage at the seed to ``1e-9`` relative.

**Class** ``TestExponentialCoarseSolver`` -- the optional exact exponential
coarse-grid propagator. ``_advance_coarse_voltage`` integrates one coarse step
of the cavity-envelope ODE with either forward-Euler (the default,
bit-unchanged) or, with ``exponential_coarse_solver_enable=True``, the exact
``V_{n+1} = e^L V_n + src (e^L - 1)/L`` (``L = -omega dt / (2 Q_L) +
1j delta_omega dt``). The exponential form is the accurate alternative to
sub-stepping when the per-step decay/detuning phase is not small.

``test_euler_branch_matches_the_forward_euler_formula``
    The default branch reproduces the forward-Euler update exactly.
``test_exponential_branch_matches_the_closed_form``
    The exponential branch matches ``e^L V + src (e^L - 1)/L``.
``test_pure_detuning_preserves_magnitude``
    Under pure detuning the exact step is a rotation (``|V|`` preserved),
    whereas forward-Euler grows the magnitude unphysically -- the ``O((delta_omega
    dt)^2)`` truncation error the exponential solver removes.
``test_small_step_reduces_to_euler``
    As the step shrinks the two solvers converge at ``O(step^2)``.
``test_beam_kick_guard_skipped_for_exponential_solver``
    The forward-Euler beam-kick guard (``_check_beam_kick_magnitude``)
    measures the *Euler* per-step beam increment, which the exact propagator
    does not take; a kick past the hard cap raises in Euler mode but must
    return without raising in exponential mode.
``test_beam_kick_guard_silent_at_zero_previous_voltage``
    The guard measures the kick *relative* to the antenna voltage it is
    added to, so with ``|V_prev| = 0`` there is no reference to compare
    against. Even a ``1e9`` A kick must then neither raise nor warn
    (checked with ``warnings.simplefilter("error")``), and the
    once-only warning budget (``_euler_guard._beam_kick_warning_issued``)
    must stay unconsumed.
``test_beam_kicks_kernel_guard_skipped_for_exponential_solver``
    Same skip for the kernel-path per-cell guard (``_check_beam_kicks``).

**Class** ``TestSharedCoarseStepArithmetic`` -- the per-cell and the
vectorised coarse step must be *one* spelling. The recursion exists twice --
``_advance_coarse_voltage`` (per cell, the reference) and
``_kernel_step_multipliers`` (vectorised, feeding the numba kernel) -- and
both must be built from the module-level helpers in
``blond.physics.feedbacks.cavity_solvers`` (``coarse_step_exponent``,
``euler_voltage_multiplier``, ``exponential_voltage_multiplier``,
``exponential_drive_weight``), beside the ``ForwardEulerValidityGuard`` that
caps them. Two independent spellings are exactly how the two paths drifted
apart before (the vectorised one lacked the scalar zero-step guard), so
these tests pin them to the shared functions **bit-for-bit**, not to within
a tolerance.

``test_step_exponent_is_shape_agnostic``
    A scalar step and a one-element array give the identical exponent
    ``L = -omega dt / (2 Q_L) + 1j relative_detuning omega dt``.
``test_euler_update_is_the_shared_multiplier``
    The per-cell Euler update equals ``v * euler_voltage_multiplier(L) +
    drive`` exactly (``assertEqual``, no tolerance).
``test_exponential_update_is_the_shared_propagator``
    The per-cell exact update equals ``v * exponential_voltage_multiplier(L)
    + drive * exponential_drive_weight(L)`` exactly.
``test_kernel_multipliers_match_the_per_cell_step``
    The invariant the numba-vs-Python bit-identity pin rests on: over three
    step sizes (``2 pi``, ``0.5 pi``, ``1e-6``) and both branches as
    subtests, ``_kernel_step_multipliers`` returns exactly the per-cell
    multiplier and drive weight (``1 + 0j`` in the Euler branch).
``test_drive_weight_guards_the_scalar_zero_step_only``
    The removable singularity of ``W = (e^L - 1) / L`` is guarded only where
    it is reachable: a scalar zero step takes the limit ``1``, while an
    *array* zero deliberately still yields ``nan`` -- the vectorised path
    never sees one (a segment with a coincident coarse point is deferred to
    the per-cell loop), so an elementwise guard would only cost the hot
    recursion an extra pass.
``test_zero_step_leaves_the_voltage_untouched``
    Both branches advance a zero-length step to exactly the input voltage.

**Class** ``TestVoltageSetpointValidation`` -- constructor validation of the
explicit ``voltage_setpoint``. The RF station's phase correction is formed
against the parent-derived ``station_voltage_coarse_grid``, whose phase is 0 by
construction, so an explicit setpoint with a non-zero phase would be regulated
by the PI controller but never reflected in the applied kick. The constructor
therefore accepts only real, positive setpoints (phase 0; rotate ``phi_rf``
on the station instead) rather than silently splitting the two frames.

``test_real_positive_setpoint_accepted``
    A real, positive setpoint is stored unchanged, both as a plain float
    (``30e6``) and as a zero-imaginary complex (``30e6 + 0.0j``).
``test_none_setpoint_accepted``
    ``None`` (the parent-derived setpoint) stays supported and is stored
    as ``None``.
``test_complex_setpoint_raises``
    A setpoint with a non-zero imaginary part raises ``ValueError``, with
    a message naming the required phase 0.
``test_negative_setpoint_raises``
    A negative (phase pi) setpoint raises ``ValueError``.

**Class** ``TestFineGridInitialConditionCausality`` -- causality of the
fine-grid initial condition in ``circuit_track``. The fine solve is seeded
with the coarse envelope at the **first forward coarse centre** ``c0`` and
then integrates the beam current over
``[profile.cut_left, profile.cut_right]``. Both times live in the same
segment-local frame, so the seed is causal only when ``c0 <= cut_left``:
otherwise the coarse cell that produced the seed already sits *after* the
start of the fine window and any charge there would be integrated twice.
A charge-free window has nothing to be causal about, so the guard is gated
on the beam current the fine solve actually consumes. Driven on a
hand-built 8-cell constant-step grid whose centres are ``(k + 0.5) t_rf``.

``test_charge_before_first_coarse_centre_raises``
    A window starting at ``0.5 pi`` (left of ``c0``) *with* charge raises
    ``ValueError``, with a message naming ``cut_left``, the
    ``first forward coarse centre`` and ``sampling_time_coarse``.
``test_charge_free_window_before_first_centre_is_allowed``
    The same acausal geometry with a zero fine-grid beam current completes
    and produces a fine-grid antenna voltage.
``test_charge_right_of_first_coarse_centre_is_allowed``
    The physical geometry (``cut_left = 1.5 pi >= c0``) with charge stays
    accepted.


``test_generator_current_controller.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unit tests for the standalone generator-current controller
(``blond.physics.feedbacks.generator_current_controller``): the
phase-preserving magnitude clamp, the klystron-power-to-current conversion, the
abstract controller interface and the saturating PI control law. The controller
is pure signal processing, so these tests drive it with plain numbers -- no
cavity, profile or ``Simulation``.

**Class** ``TestClampMagnitude`` -- the phase-preserving magnitude clamp
``clamp_magnitude``.

``test_preserves_phase_and_clamps_magnitude``
    Clamping sets the magnitude to the limit and keeps the phase.
``test_leaves_small_values_unchanged``
    A value below the limit passes through unchanged.
``test_handles_arrays_including_zero``
    Array input is clamped element-wise; zero entries stay zero.
``test_none_limit_is_a_no_op``
    With no limit the input is returned unchanged.

**Class** ``TestCurrentLimitFromPower`` -- the klystron-power-to-current-limit
conversion ``current_limit_from_power``.

``test_matched_generator_relation``
    ``I_max = sqrt(2 P / ((R/Q) Q_L))``.

**Class** ``TestAbstractController`` -- the ``GeneratorCurrentController``
interface.

``test_cannot_instantiate_abstract_base``
    The interface has an abstract ``update_generator_current`` and cannot be
    instantiated.
``test_default_limit_is_a_no_op``
    The base ``limit()`` applies no actuator limit.
``test_compiled_scan_interface_names_the_controller``
    A controller that does not advertise ``supports_envelope_scan`` must
    reject all three compiled-scan entry points -- ``envelope_scan_kernel``,
    ``envelope_scan_state`` and ``absorb_envelope_scan_state`` (one subtest
    each) -- with a ``NotImplementedError`` naming the offending class
    (``"_Trivial supplies no compiled envelope scan"``), so a feedback wired
    to the wrong controller fails loudly instead of running a
    half-implemented scan.

**Class** ``TestGeneratorCurrentPIController`` -- the saturating PI control law
mapping a voltage error to a generator current.

``test_constant_current_passthrough_with_zero_gains``
    With zero gains the output is the constant current bias, for any error.
``test_proportional_only``
    Pure P control returns ``I_bias + K_p * error``.
``test_loop_delay_holds_output_until_error_propagates``
    With ``n_delay`` samples the error acts only after ``n_delay`` updates.
``test_integral_accumulates_linearly``
    Pure I control integrates a constant error linearly.
``test_anti_windup_freezes_integral_while_saturated``
    The integrator does not wind up while the output is clamped.
``test_integral_resumes_after_desaturation``
    Once the output is back in range the integrator accumulates again.
``test_output_magnitude_is_clamped_with_phase_preserved``
    The returned command never exceeds the configured limit.
``test_limit_clamps_an_array_to_max_output``
    ``limit()`` enforces the klystron limit on an external current array.
``test_limit_is_a_no_op_without_a_limit``
    Without ``max_output``, ``limit()`` returns the input unchanged.
``test_negative_delay_is_rejected``
    A negative loop delay raises.


``test_generator_current_pi_feedback.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integration tests for the controller-driven cavity feedback: an
``IQCavityFeedbackTimingClass`` with a ``GeneratorCurrentPIController`` attached
(``controller=``). Module helpers ``build_controller``, ``build_feedback`` and
``run_coarse_transient`` construct an RCS1-like loop at its no-beam steady state
and drive ``circuit_track`` on a hand-built constant-step grid with a beam
current switching on mid-turn. The diagnostic plots are gated by the module
switch ``PLOT_DIAGNOSTICS`` (``None`` disables them, the CI default; ``"save"``
writes a PNG next to the file; ``"show"`` also opens a window), so the
``test_plot_*`` methods below skip unless it is set.

**Class** ``TestGeneratorPower`` -- klystron power from the generator current.

``test_generator_power_formula``
    Checks ``P = 0.5 (R/Q) Q_L |I|^2``.
``test_generator_power_defaults_to_the_coarse_grid``
    Called without an argument, ``generator_power()`` reads the stored
    ``generator_current_coarse_grid`` -- equal element-wise to passing that
    array explicitly, and carrying real watts (non-vacuous).

**Class** ``TestFeedbackControllerDelegation`` -- the feedback delegates the
error-to-current conversion to its controller.

``test_update_delegates_error_and_step_to_controller``
    The controller update receives the correct antenna-voltage error and
    per-step time.
``test_update_before_circuit_track_is_rejected``
    The controller recovers its sampling time from
    ``omega_times_dt / omega_input``, and ``omega_input`` is set only by
    ``circuit_track``; reaching the update path first raises
    ``RuntimeError`` (``"called before circuit_track"``) and the controller
    is never consulted with an undefined step.
``test_no_controller_keeps_constant_current``
    Without a controller the generator current stays at the constant bias.

**Class** ``TestHighIntensityBunchTransient`` -- one transient is run in
``setUpClass`` and shared. A constant RF beam current switches on mid-turn; the
beam loading sags the voltage, then after the loop delay the PI controller
restores it.

``test_steady_state_before_the_bunch``
    Voltage and current are stationary before the bunch arrives.
``test_generator_current_reacts_only_after_the_loop_delay``
    The generator current stays at the constant bias for ``n_delay`` samples,
    then moves.
``test_beam_loading_pulls_the_voltage_down_first``
    The antenna voltage sags within the delay window.
``test_voltage_vector_returns_to_setpoint``
    The integrator restores the complex (I and Q) voltage to its setpoint.
``test_generator_current_settles_at_compensation_value``
    The generator current ends at the beam-loading compensation
    ``I_ff + I_beam / 2``.
``test_generator_power_goes_up``
    The klystron power rises while the voltage is restored.
``test_plot_coarse_power_and_voltage``
    Opt-in diagnostic plot of the coarse power/voltage transient.

**Class** ``TestKlystronPowerLimit`` -- with the current clamped below the
required compensation value, the loop saturates.

``test_generator_current_never_exceeds_the_limit``
    ``|I_gen|`` stays at or below the configured maximum.
``test_generator_current_saturates_at_the_limit``
    The controller drives the klystron into saturation.
``test_voltage_does_not_return_to_setpoint``
    With saturated power the voltage cannot be restored.
``test_plot_coarse_power_and_voltage``
    Opt-in diagnostic plot of the power-limited (saturated) transient.

**Class** ``TestSinglePulseBunchTransient`` -- a single bunch passage (the
beam current switched on and then off again) contrasted with sustained loading,
both with a clamped generator current.

``test_clamp_is_active_during_the_passage``
    While the bunch is present the generator current saturates.
``test_voltage_returns_to_setpoint_after_the_bunch``
    Once the bunch has passed, the voltage vector recovers to ``V0``.
``test_generator_power_returns_to_baseline``
    The klystron power rises during the bunch and then drops back.
``test_sustained_loading_does_not_recover``
    The contrast case keeps sagging instead of returning to ``V0``.
``test_plot_single_pulse_vs_sustained``
    Opt-in diagnostic plot contrasting a bunch passage with sustained load.

**Class** ``TestLoopDelaySampleSemantics`` -- ``n_delay`` counts coarse-grid
*samples*, not time: with a sub-stepped grid the physical loop delay is
``n_delay * n_rf_periods_per_coarse_grid * t_rf`` and shrinks with the
sub-step.

``test_delay_is_counted_in_samples_not_time``
    The generator current first reacts at the same *sample* offset after
    beam-on for the standard (``n = 1``) and sub-stepped (``n = 0.5``) grid.

**Class** ``TestResponseMatrixClamping``

``test_fine_grid_solve_uses_clamped_generator_current``
    The fine-grid response-matrix solve (``cavity_response_sparse_matrix``)
    sees the clamped generator current, and the stored fine-grid current is
    clamped in place.

**Class** ``TestPerProfileVoltageOverTurns`` -- a multi-turn transient run once
in ``setUpClass`` and shared; checks the per-profile (fine-grid) gap voltage
turn over turn.

``test_no_distortion_before_the_bunch``
    Before the bunch, the per-profile voltage sits at the setpoint.
``test_beam_loading_distorts_the_profile_voltage``
    The bunch produces a clear transient distortion of the voltage.
``test_distortion_peaks_right_after_the_bunch_arrives``
    The largest distortion is the beam-arrival transient, not a slow drift.
``test_per_profile_voltage_recovers_toward_setpoint``
    The final-turn distortion is well below the transient peak.
``test_plot_per_profile_voltage_over_turns``
    Opt-in diagnostic plot of the per-profile voltage over turns.


``test_pi_feedback_full_tracking.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PI-controlled feedback inside a *real tracked* ``Simulation`` -- the
complement to the hand-built coarse grids above. A matched ``BiGaussian``
``mu_plus`` bunch with strong beam loading is tracked through a real ring
(half-drift / station / half-drift per section) with the backfill/forward
reference tracking, under acceleration, with a
``GeneratorCurrentPIController`` regulating every station. Each PI tracking
configuration asserts physical behaviour and then *pins* the end-of-turn
antenna-voltage and generator-current trajectories against hardcoded
reference values (characterization: any change of the tracked feedback
numerics shows up here first); the driven open-loop steady state, the
zero-intensity phase neutrality, the design-locked drive walk-off under
an RF-frequency offset and the numba-kernel bit identity are guarded
end to end here as well.
Setting the ``PI_TRACKING_PRINT_PINS`` environment variable prints the
recorded trajectories instead (used to regenerate the pins); while the pins
are unrecorded (``None``) the pin tests skip.

**Class** ``TestDrivenSteadyStateFastRamp`` -- a driven, beam-free cavity
holds its steady state on the fast ramp. With the matched generator bias
and no beam the coarse recursion has the exact fixed point
``V_ss = 2 (R/Q) Q_L I_gen = V_design``, independent of the RF frequency
and of the step size, so an on-resonance constant-drive cavity
(``use_controller=False``, zero intensity) must sit at ``V_ss`` however
fast the ramp moves and however many stations the ring has. Multi-section
used to rotate the carried antenna-voltage *state* by the per-turn
grid-vs-carrier registration phase ``sum_k (omega_k - omega_0) T_seg,k``,
which also hit the generator-driven field and dragged ``|V_ant|`` off
``V_ss`` by ~0.6 % per turn; the phase is now carried on the
demodulation/readout carrier, leaving the driven steady state exact. Each
test gates the end-of-turn ``|V_ant|`` at ``1e-8`` relative over five
fast-ramp turns -- far above the ~2e-12 single-section floor and far
below the ~3e-2 the state rotation produced.

``test_single_section_holds_steady_state``
    Control: one station holds ``V_ss`` on the fast ramp (the ~2e-12
    floor, asserted at the shared ``1e-8`` gate).
``test_multi_section_holds_steady_state``
    Two stations must hold it too -- the regression under test.
``test_four_sections_hold_steady_state``
    Four stations: three backfill segments per passage.

**Class** ``TestDrivenFeedbackIsPhaseNeutralWithoutBeam`` -- a driven,
beam-free cavity on its setpoint must hand the station NO phase: the
zero-intensity phase-neutrality guarantee of the split coarse envelope,
and the in-repo counterpart of the RCS example's
``test_feedback_is_a_no_op_without_beam``. Two-section fast ramp
(4 GeV + 20 MeV/turn), zero intensity, matched generator bias, six
turns; each test gates ``max |phase_correction|`` over all turns at
``1e-12`` rad -- FP dust of the fine-grid solve, against the ~0.3
rad/turn the fixed bug produced on this ring (the registration phase
``Psi`` handed to the design-locked generator drive at readout).

``test_matched_bias_applies_no_phase``
    Constant matched drive: the headline zero-intensity no-op.
``test_pi_loop_applies_no_phase``
    A PI loop holding the same setpoint must be phase-neutral too.

**Class** ``TestDesignLockedDriveWalkOffUnderRFOffset`` -- under a station
RF-frequency offset the design-locked drive walks off the actual RF.
The klystron drive follows the DESIGN frequency, so with
``delta_omega_rf`` set the actual RF accumulates the kick-clock slip
relative to the design clock and the driven (generator) field must
appear at MINUS that slip relative to the actual RF -- real physics,
not a bookkeeping artefact. Single section, constant 63 GeV, beam-free,
constant matched bias, offset ``1e-7 * omega_rf`` (~0.016 rad of slip
per turn -- far above the readout's FP floor, far below a wrap), six
turns.

``test_driven_field_appears_at_minus_the_kick_clock_slip``
    Per turn, ``phase_correction == -delta_phi_rf`` to ``atol=1e-9``
    rad, after asserting the premise has teeth
    (``|delta_phi_rf|`` really accumulates past 0.05 rad by the last
    turn).

**Class** ``TestDetunedLoopHoldsSetpointAcrossBackfillSpan`` -- a detuned,
PI-regulated cavity must hold its setpoint for the *whole* turn, backfill
span included. With ``delta_omega != 0`` the matched no-beam drive is
no longer the feedforward bias but ``I_0 (1 - i tan psi)``,
``tan psi = 2 Q_L delta_omega / omega_rf``: cancelling the detuning
precession needs a reactive standing current, which the PI finds on the
forward span. A multi-section ring then replays the remaining
``(N - 1) / N`` of the turn as no-beam backfill segments, and
``reset_arrays`` seeds that span with the **last commanded** generator
current rather than the constant feedforward bias -- a zero-order hold
over an interval in which the loop issued no new command. Replaying it
with the bias instead lets the antenna voltage precess by the analytic
excursion ``|dV| / V_set ~ delta_omega * T`` (independent of ``Q_L`` and
``R/Q``) -- ``3.2e-2`` per turn for the two-section case -- on the very
sample that seeds the fine grid the bunch is solved on, so it is not
self-correcting. No beam is tracked on purpose: without beam loading the
correct answer is exactly the setpoint on every coarse sample, so the
assertion has no tolerance budget to hide in. Constant energy (63 GeV, no
ramp and no frame slip), five turns, turn 1 skipped while the loop
converges from ``initial_voltage``, gate ``1e-6`` relative.

``test_detuned_loop_holds_setpoint_over_the_whole_turn``
    Two sections, one half-bandwidth of detuning: the half-turn backfill
    span must not drive the cavity off its setpoint.
``test_four_sections_hold_setpoint_over_the_whole_turn``
    Four sections, so the backfill span is 3/4 of the turn rather than
    1/2. The excursion scales with the span duration, which makes this
    the direct fingerprint of the backfill reconstruction rather than of
    any forward-pass effect.
``test_matched_bias_control_case_still_exact``
    Control: on resonance the bias *is* the held current. Same ring, same
    loop, same assertion, only ``delta_omega = 0`` -- so a failure of the
    detuned cases comes from the detuning, not from a broken fixture.
``test_undriven_detuned_cavity_is_left_free_running``
    Control: with no controller attached the detuned cavity must still
    precess away from the setpoint (> 20 % by the last turn). Guards
    against "fixing" the above by writing a matched current into the grid
    unconditionally.

**Class** ``TestPIFullTrackingSingleSectionFastRamp`` -- one section on the
fast (transition-adjacent, 4 GeV + 20 MeV/turn) ramp.

``test_reference_follows_energy_program``
    The reference energy gains exactly ``DELTA_E_TURN`` per turn.
``test_loop_acts_on_the_generator_current``
    The PI moves the generator current away from the pure feedforward bias.
``test_beam_loading_sags_the_voltage``
    The bunch passage visibly sags ``|V_ant|`` below the setpoint.
``test_voltage_recovers_by_turn_end``
    The loop restores ``|V_ant|`` to the setpoint by the end of a turn.
``test_bunch_stays_bounded``
    The bunch length stays finite and bounded (no blow-up).
``test_pinned_trajectories``
    Characterization pin of the exact recorded trajectories.

**Class** ``TestPIFullTrackingMultiSectionSlowRamp`` -- two sections on the
operating-point (63 GeV, slow) ramp, so the pinned trajectories
characterise a representative production regime; the transition-adjacent
fast ramp is covered by ``TestPIFullTrackingMultiSectionFastRamp``. The
pins were last regenerated when the coarse envelope was split into its
generator- and beam-sourced components and the PI error moved to the
KICK-frame sum: the loop now regulates the applied kick, whose
difference from the former raw state is ``V_beam (1 - e^{i Psi})`` with
this slow ramp's registration phase ``Psi ~ 7e-6`` rad/turn. That moved
``|V_ant|`` by <= 2.4e-6 relative and the current response by
<= 1.7e-6 -- marginally beyond the 1e-6 pin tolerance, a real
(declared) modelling shift, not FP noise; the behavioural tests below
independently assert that both stations still hold the setpoint and
respond to the loading. (An earlier regeneration moved the
registration phase from a rotation of the antenna-voltage state onto
the demodulation/readout carrier; see ``TestDrivenSteadyStateFastRamp``.)

``test_reference_follows_energy_program``
    The reference energy gains exactly ``DELTA_E_TURN`` per turn.
``test_loop_acts_on_both_stations``
    Both stations' PI loops move their generator currents.
``test_beam_loading_sags_both_stations``
    The bunch passage sags ``|V_ant|`` at both stations.
``test_voltage_recovers_on_both_stations``
    The loops restore ``|V_ant|`` to the setpoint by the turn end.
``test_pinned_trajectories``
    Characterization pin of the exact recorded trajectories.

**Class** ``TestPIBackfillSpanFrameConsistency`` -- the PI loop must act only
on the forward (real-beam) coarse cells, never on the ``no_beam`` backfill
reconstruction segments that rebuild the previous turn. Stepping the
controller on the backfill cells would double-advance its delay line and
integrator on frame-rotated errors; the fix gates the controller update on
``not no_beam``. The tests instrument the controller call count against the
recorded per-turn forward and total cell counts.

``test_controller_stepped_only_on_forward_cells``
    Two-section fast ramp: the controller is stepped on exactly the forward
    cells and never on the (larger) backfill reconstruction segments.
``test_single_section_controller_skips_turn0_backfill``
    Control: a single-section ring still reconstructs its very first turn by
    backfill (``n_total > n_forward`` on turn 0), and the gate skips those
    backfill cells too.

**Class** ``TestPIFullTrackingMultiSectionFastRamp`` -- two sections on the
transition-adjacent fast (4 GeV + 20 MeV/turn) ramp -- 5x steeper at 1/16
the energy of the slow-ramp pins. Previously excluded: the grid-vs-carrier
registration phase, applied as a rotation of the antenna-voltage state,
dragged the driven field off its steady state (see
``TestDrivenSteadyStateFastRamp``), so a pinned PI trajectory would have
characterised that drift rather than the loop; with the phase carried on
the demodulation/readout carrier the fast ramp behaves like the slow one.
The pins were then regenerated once more with the split coarse envelope:
the previous set still encoded the driven readout-phase artefact this
configuration exists to expose (``Psi ~ 0.14`` rad/turn/station handed
to the generator-driven field too, with the PI partially fighting that
bookkeeping rotation). With the generator component design-anchored and
the PI regulating the kick-frame sum, ``|V_ant|`` moved by up to 1.8e-2
relative and the current response by up to ~9 % here;
``TestDrivenFeedbackIsPhaseNeutralWithoutBeam`` pins the fixed
zero-intensity behaviour these numbers now build on.

``test_reference_follows_energy_program``
    The reference energy gains exactly ``DELTA_E_TURN`` per turn.
``test_beam_loading_sags_both_stations``
    The bunch passage sags ``|V_ant|`` at both stations.
``test_loop_acts_on_both_stations``
    Both stations' PI loops move their generator currents.
``test_voltage_recovers_on_both_stations``
    The loops restore ``|V_ant|`` to the setpoint by the end of every turn.
``test_bunch_stays_bounded``
    The bunch length stays finite and bounded (no blow-up).
``test_pinned_trajectories``
    Characterization pin of the exact recorded fast-ramp trajectories.

**Class** ``TestKernelMatchesReferenceEndToEnd`` -- the numba coarse-envelope
kernel (the default path) must reproduce the pure-Python reference
recursion bit-for-bit through a full tracked ``Simulation``, not just on
the isolated hand-built grids of ``test_envelope_kernel.py``.

``test_multi_section_kernel_vs_python_bit_identical``
    A two-section, four-turn fast-ramp PI run on the default numba kernel
    and again on the pure-Python reference records byte-identical
    ``v_min``, ``v_last`` and ``i_max_dev`` trajectories -- covering the
    multi-section, turn >= 1 carried-state backfill segments inside the
    real turn loop.


``test_helpers.py``
^^^^^^^^^^^^^^^^^^^

Tests for the cavity-response solvers (first- and second-order, both in
``blond.physics.feedbacks.cavity_solvers``) and the
beam-current demodulation in ``blond.physics.feedbacks.beam_current``. The
solvers are driven directly on a static profile -- no ``Beam`` tracking and
no full ``Simulation``.

**Class** ``TestCavityResponseSolverConvergence`` -- first-order forward Euler
(``cavity_response_sparse_matrix``) versus second-order Crank-Nicolson
(``cavity_response_sparse_matrix_second_order``), both integrating the same
cavity-envelope ODE and converging to the multi-turn resonator convolution.

``test_second_order_more_accurate_at_low_binning``
    At coarse binning the Crank-Nicolson solver beats Euler by orders of
    magnitude.
``test_solver_convergence_orders``
    Halving the bin size shrinks the error by ~2 for Euler (``O(dt)``) and
    ~4 for Crank-Nicolson (``O(dt^2)``).
``test_solvers_agree_as_predicted_by_convergence_rate``
    The two solvers differ by the (first-order) Euler truncation error, which
    halves as the bin count doubles.
``test_second_order_flag_routes_through_the_class``
    ``IQCavityFeedbackTimingClass(second_order_fine_grid_solver_enable=...)``
    reproduces the matching standalone solver bit-for-bit and lands far
    closer to the convolution.

An opt-in debug plot (``DEBUG_PLOT``, ``_plot_convergence``) shows the
convergence slopes and the residual against the convolution solver.

**Class** ``TestRfBeamCurrentDownsampling`` -- charge conservation of the
coarse-grid downsampling in ``rf_beam_current``, plus the argument and
geometry guards around it. Regression test for a dropped remainder that used
to silently discard demodulated charge past the last coarse-cell boundary (up
to the whole bunch, depending on its phase), and for the span/index guards
that replaced the ``% n_points`` wrap: every write index must now land inside
the coarse grid, or the call raises. The sweep positions the bunch at 0.08,
0.2, 0.5 and 0.9 of a 1.5-``t_rf`` window on an RCS1-like 25900-cell grid.

``test_downsampling_conserves_demodulated_charge``
    Re-binning the fine-grid demodulated charge onto the coarse grid conserves
    the complex sum, for bunches swept across the cell boundaries.
``test_lowpass_filter_attenuates_the_fine_current``
    The optional ``use_lowpass_filter`` (20 MHz cutoff against a 1 GHz
    carrier) genuinely acts: the filtered fine-grid current keeps the shape
    and stays finite, differs from the unfiltered one, and carries a smaller
    L2 norm.
``test_sampling_time_without_n_points_is_rejected``
    ``sampling_time`` with ``n_points=None`` cannot size the coarse grid and
    raises ``TypeError`` (``"n_points is required when sampling_time"``).
``test_raises_when_charged_bins_map_before_turn_zero``
    An underflow that *carries charge* raises rather than warns. With
    ``dT = -2 t_rf`` the whole bunch maps to ``ind_fine < 0``; NumPy negative
    indexing would deposit it in the *last* coarse cells (measured: 100 % of
    the fine-grid charge in negative-mapping bins, peak at index 25899 of
    25900) -- roughly a forward-segment span late and out of reach of the
    first-coarse-cell guard, which only inspects cell 0. The message names
    ``before the start of the coarse grid``. This method previously pinned
    the warn-only behaviour; it became a raise-test once the charge fraction
    in the negative-mapping bins was measured to be 1.0.
``test_warns_only_when_negative_bins_carry_no_charge``
    Anti-false-positive pin for that underflow guard: a small negative ``dT``
    pushes only the *leading, charge-free* bins below the grid start, so the
    long-standing ``before turn time 0`` warning must survive without a raise
    -- and the charge must still be conserved. The raise is reserved for bins
    that actually carry charge, using the same relative-threshold idiom as
    the first-coarse-cell guard.
``test_error_when_first_coarse_cell_populated``
    With ``forbid_charge_in_first_coarse_cell=True`` (used by the feedback to
    avoid double-counting), charge in the first cell raises.
``test_no_error_when_first_coarse_cell_empty``
    A mid-window bunch leaves the first cell numerically empty (the guard uses
    a relative threshold, not ``!= 0``).
``test_warns_on_particle_loss``
    Warns when the profile does not capture the whole beam (modelled by a
    density factor putting only half the macroparticles in the window).
``test_no_warning_when_profile_captures_full_beam``
    No warning when the window captures everything.
``test_raises_when_profile_longer_than_coarse_grid``
    A window longer than the coarse grid raises ``ValueError`` instead of
    wrapping. The ``% n_points`` wrap is gone: two bunches 3 ``t_rf``
    apart in a 5 ``t_rf`` window folded onto a 3-cell grid used to put the
    trailing bunch's index onto the leading bunch's cell and *overwrite*
    it, losing 50 % of the demodulated charge silently. The guard is
    ``ProfileBaseClass.check_fits_in_span``.
``test_particle_loss_warning_is_not_shadowed_by_the_span_guard``
    The two guards answer different questions -- "is the beam inside the
    profile?" versus "does the profile fit the span it is re-binned
    onto?" -- and the fold destroys charge even when the whole beam is
    captured, so the loss warning is emitted *before* the span check and
    must still be seen when that check raises.
``test_raises_when_profile_starts_after_coarse_grid``
    A window entirely past the grid raises a ``ValueError`` naming the
    coarse-grid index, not a bare ``IndexError``. This one is raised by
    ``rf_beam_current`` itself, not by ``check_fits_in_span``.
``test_raises_when_profile_binning_is_coarser_than_the_grid``
    ``hist_step > sampling_time`` desyncs the downsampling loop: it walks
    ``ind_fine`` assuming it advances by at most 1 per fine bin, and the
    running group counter *is* the coarse index, so a rounded jump of 2
    leaves the counter behind and charge lands at the wrong **time** while
    the total stays conserved -- silent corruption. Measured on the fixture
    (3 ``t_rf`` window, 16 bins, ``sampling_time = t_rf / 8``, ratio 1.5):
    all charge went into coarse cells 0-7 instead of the true 0-23, with the
    complex total conserved to 1.8e-16 relative. Raises ``ValueError``, the
    message naming ``coarser than the coarse grid`` and
    ``n_rf_periods_per_coarse_grid``.
``test_binning_just_finer_than_the_grid_is_accepted``
    Anti-false-positive pin for that binning guard: sub-stepping
    (``n_rf_periods_per_coarse_grid < 1``) shrinks ``sampling_time``, so a
    legitimate profile can approach the bound from below. The worst ratio
    measured across ``tests/unittests/physics/`` is 0.12 (the ``n = 0.25``
    sub-stepped grid-geometry cases in ``test_rf_center_grid.py``), so a
    ratio of ~0.94 sits far beyond any real configuration and must still be
    accepted -- and still conserve charge.
``test_long_window_that_fits_still_conserves_charge``
    Anti-false-positive pin: the same 5 ``t_rf`` two-bunch window the
    3-cell grid rejects is perfectly valid on an 8-cell grid and must keep
    conserving charge. This is the geometry
    ``test_multibunch_beam_loading.py`` relies on (~8 ``t_rf`` window on a
    13-cell grid, window/span ratio
    0.62 -- the widest legitimate one in the suite), so the threshold may
    not creep below it.

**Class** ``TestRfBeamCurrentCounterRotating`` -- direction-signed charge in
the RF beam current. In the symmetric muon-collider ring the counter-rotating
mu-minus beam has opposite charge *and* opposite direction, so its gap current
has the **same sign** as the co-rotating mu-plus beam. The source side of
``rf_beam_current`` uses
``beam.signed_charge_with_direction()`` (charge negated for a counter-rotating
beam), matching the RF-kick and wake-kick conventions; for co-rotating beams
it equals the plain particle charge, so co-rotating behaviour is unchanged.

``test_counter_rotating_mu_minus_matches_co_rotating_mu_plus``
    CR mu-minus current is bit-identical to the mu-plus current on both the
    fine-grid and coarse (downsampled) paths of ``rf_beam_current`` (was
    exactly sign-flipped before the fix).
``test_co_rotating_mu_minus_flips_the_sign``
    Charge alone (same direction) still flips the current -- ordinary
    opposite-charge physics untouched.
``test_counter_rotating_mu_plus_flips_the_sign``
    Direction alone (same charge) flips the current -- the complementary
    corner of the sign matrix.
``test_co_rotating_signed_charge_is_the_plain_charge``
    For any co-rotating beam the signed charge reduces to the plain particle
    charge, so the direction handling leaves co-rotating results bit-unchanged.

**Class** ``TestUnifiedRfBeamCurrentMigrationPin`` -- migration pin of the
unified ``rf_beam_current`` coarse path. The recorded values were produced by
driving the pre-merge ``rf_beam_current_partial`` (the timing-class
forward-pass variant that was folded into ``rf_beam_current``) on this exact
fixture -- mid-window Gaussian, 1024 bins -- with ``dT`` and
``carrier_phase_offset`` both nonzero and
``forbid_charge_in_first_coarse_cell=True``. The unified function must
reproduce them byte-exactly, pinning the merged coarse path (bin-centre
offset ``sampling_time / 2``, carrier-phase rotation, downsampling loop,
remainder handling) to the pre-merge behaviour.

``test_unified_coarse_path_matches_recorded_partial_output``
    Exact equality (no tolerance) against the recorded outputs: the fine-grid
    complex sum, absolute sum and centre sample, plus the full recorded
    8-cell coarse grid.


``test_mtw_vs_nondriven_feedback.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compares the same single cavity modelled as a ``MultiPassResonatorSolver``
(multi-turn resonator convolution) and as a non-driven
``IQCavityFeedbackTimingClass`` whose antenna voltage, with the beam as the
only excitation, is the beam-induced voltage. ``make_noisy_profile`` builds the
shared noisy-Gaussian static profile (edge bins zeroed).

**Class** ``TestSinglePassInducedVoltage`` -- single pass, mock-driven, no
``Simulation``; the lab-frame induced voltages agree to < 1 %.

``test_induced_voltage_matches_non_driven_feedback``
    Pointwise (within 1 % of peak), shape (relative L2 < 1 %) and peak
    amplitude (within 1 %) agreement.
``test_zeroed_profile_edges_remain_zero``
    Guards the precondition that the edge bins carry no charge.
``test_feedback_without_beam_or_generator_is_silent``
    A non-driven feedback with zero initial voltage induces nothing.

**Class** ``TestMultiTurnFeedbackVsConvolution`` -- full ``Simulation`` with a
dummy particle-less beam driving static profiles over several turns. The
feedback's coarse grid is propagated turn over turn through the
backfill/forward reference tracking, and its beam-induced gap voltage (minus
a no-beam reference run) is compared per turn and per section against the
accumulating convolution voltage. Uses a high ``Q_L = 1.29e6`` so the
previous-pass wake survives (~88 % per turn). Results are cached per the
full config tuple
(``n_sections``, ``acceleration``, ``n_rf_periods``, ``fast_ramp``,
``delta_omega``, ``delta_omega_rf`` and the turn/harmonic overrides).

``test_multiturn_wake_accumulates_over_turns``
    The multi-pass wake genuinely builds up turn over turn (peaks ~1, 1.9, 2.8)
    and the first turn matches the feedback to single-pass accuracy.
``test_multiturn_feedback_propagation_matches_convolution``
    Coarse-grid propagation matches the convolution on every turn (regression
    for the dropped downsample remainder, single section, static cycle).
``test_multiturn_multiple_sections``
    Holds for multi-section rings (2, 3, 10 RF stations per turn), exercising
    the backfill/forward reference tracking across stations.
``test_multiturn_with_acceleration``
    Holds under acceleration (``MagneticCyclePerTurnAllRFStations``), where
    ``t_rev``, the carrier frequency and the backfill-tracking frame slip vary
    turn over turn.
``test_multiturn_substepped_matches_convolution``
    Beam loading computed on a sub-stepped coarse grid
    (``n_rf_periods_per_coarse_grid < 1``) stays correct on a static cycle.
``test_multiturn_fast_ramp``
    Single section on the fast (transition-adjacent) ramp still matches the
    retuning convolution in the fast frame-slip regime.
``test_multiturn_fast_ramp_multisection``
    Multi-section (2 and 4 stations) on the fast ramp matches the retuning
    convolution: the grid-vs-carrier registration phase
    ``Psi = sum_k (omega_k - omega_0) T_seg,k``, which the other stations'
    mid-turn grid re-seeding accumulates in
    ``_accumulate_registration_phase``, is carried on the
    demodulation/readout carrier -- *not* applied as a rotation of the
    antenna-voltage state (see ``TestDrivenSteadyStateFastRamp`` in
    ``test_pi_feedback_full_tracking.py``). Uncorrected, the arrival time
    drifted ~0.023 ``t_rf`` per turn; corrected, the error stays ~0.2 %.
``test_multiturn_fast_ramp_substepped``
    Sub-stepped (n = 0.5) carried wake holds on the fast ramp: the stale
    backfill-segment re-pass is removed (it corrupted the demodulation frame
    by ``-(turn+1) * 2 pi S`` per turn for single-section rings) and the
    sub-stepped demodulation frame is the tiling boundary gap (a pure time,
    immune to the float-bistable residual landing flip). ~0.1 %, was ~40 %.
``test_multiturn_fast_ramp_multisection_substepped``
    The full combination (2 sections, fast ramp, n = 0.5) passes: the
    tiling-gap demodulation frame also covers the multi-section
    backfill-to-forward handover.
``test_multiturn_delta_omega_rf_with_beam``
    A beam-driven RF-frequency offset ``delta_omega_rf`` is *exercised* and
    stays consistent. Two checks: (1) a **non-triviality guard** that the offset
    genuinely moves the feedback's beam-induced voltage above the discretization
    floor (last-turn ``|fb(offset) - fb(no offset)|/|fb| ~ 3.4 %`` vs a ~0.1 %
    floor) -- a regression that dropped ``delta_omega_rf`` on the beam path
    collapses this to ~0 and fails, which a plain per-turn gate cannot catch
    (with the offset ignored the reference-subtracted voltage still sits at the
    baseline, ~88 % of the 2 % gate, and passes); (2) the small offset still
    tracks the retuning convolution to the 2 % gate.
``test_multiturn_delta_omega_rf_large_offset_consistency``
    A 2e3 rad/s offset (past half the cavity half-bandwidth) tracks the
    retuning convolution to the 2 % per-turn gate. Before the demodulation
    carrier was anchored to the accumulated actual RF phase this failed
    within two turns (the former lab-frame slip grew by
    ``delta_omega_rf * t_rev``, ~4 % vector error, per turn); anchored, the
    residual sits at the discretization floor (measured net carrier-phase
    error <= 2e-5 rad per turn, offset-independent).
``test_multiturn_delta_omega_rf_differential``
    Difference-of-differences at the small offset: the offset-induced move
    ``fb(offset) - fb(no offset)`` matches the convolution's move to
    < 0.5 % of ``|V|`` per turn (the baseline discretization error cancels
    in each difference, isolating the offset chain). Unanchored, the
    spurious move was 0.9-1.7 % of ``|V|``.
``test_multiturn_delta_omega_rf_substepped``
    The large offset also holds on the sub-stepped grid (n = 0.5): tiling
    residual carry-over and the tiling-gap demodulation frame compose with
    the carrier anchoring.
``test_multiturn_delta_omega_rf_multisection``
    The large offset also holds with two RF stations: backfill-tracked
    segments, per-station kick clocks and the multi-section frame
    correction stay consistent with the carrier anchoring. All four
    offset tests are mutation-verified (flipping the anchor sign fails
    every one).
``test_multiturn_secular_drift_long_horizon``
    Long-horizon guard for the shorter consistency tests: the most drift-prone
    case (2 sections, fast undriven) run for 20 turns has a bounded per-turn
    relative-error slope (~0.03 pp/turn) and an endpoint within 1 %.
``test_multiturn_nondivisible_harmonic_is_rejected``
    KNOWN LIMITATION, pinned as a contract. A harmonic not divisible by
    ``2 * n_sections`` de-aligns the coarse-grid tiling from the profile's
    zeroed leading edge, so beam charge is downsampled into the first coarse
    cell and ``rf_beam_current`` raises before any voltage is produced -- a
    genuine gap versus the geometry-agnostic solver. The test asserts that
    ``ValueError``, and that its message both names the cause and stays
    actionable (it must mention ``cut_left``), for the static and the fast
    two-section config. It is deliberately *not* an ``expectedFailure``: an
    xfail passes on any error, so it would survive the limitation being
    replaced by an unrelated crash, and it would fail as an unexpected pass
    once the geometry is generalised. The other multi-section tests reduce
    the harmonic to a multiple of ``2 * n_sections`` to avoid this.
``test_multiturn_detuned_regression_lock``
    Regression-locks the proven-good static detuned-cavity regimes
    (``delta_omega`` of a few to ~10 half-bandwidths across static/fast x 1/2
    sections); the feedback tracks the detuned convolution to < 0.3 %.
``test_multiturn_driven_generator_beam_part_linearity``
    With a matched generator bias (``I_g = V / (2 (R/Q) Q_L)``) driving the
    cavity, the isolated beam-induced part (beam run minus no-beam reference)
    is independent of the drive to ~1e-6, and single-section ``|V_ant|`` holds
    at ``V_ss`` to ~1e-9 -- the linearity the reference-subtraction relies on.
``test_multiturn_substepped_detuned``
    Sub-stepping (n = 0.5) combined with a static detuning of +/- two
    half-bandwidths holds against the convolution on both the static and fast
    cycles.
``test_multiturn_counter_rotating_mu_minus_matches_mu_plus``
    The symmetric-ring counter-rotating requirement applied to the feedback:
    a counter-rotating mu-minus beam (opposite charge x opposite direction =
    identical direction-signed gap current) reproduces the co-rotating
    mu-plus run **bit-for-bit** through the full multi-turn Simulation --
    feedback gap voltage, no-beam reference and convolution induced voltage
    alike. Single section, static cycle, one beam per run (the
    two-simultaneous-beam mainloop is a separate open problem).
``test_multiturn_counter_rotating_mu_minus_matches_mu_plus_with_delta_omega_rf``
    Extends the bit-for-bit counter-rotating invariant to a nonzero
    ``delta_omega_rf`` (2e3 and 8e2 rad/s, swept over the feedback beam
    run, the no-beam reference and the retuning convolution): the
    demodulation-anchoring slip chain (design-clock grid plus accumulated
    ``-(delta_phi_rf + live gap)``) was validated on the co-rotating
    stream only, so a direction-dependent sign or value in the slip
    anchor would surface here while the zero-offset invariant stayed
    green. Single section, static cycle.

**Class** ``TestExponentialSolverEndToEnd`` -- end-to-end validation of the
exact exponential coarse-grid propagator
(``exponential_coarse_solver_enable=True``; the unit-level closed-form
checks live in ``TestExponentialCoarseSolver`` under
``test_mucol_cav_fdbk.py``). Reuses the full-``Simulation`` harness of
``TestMultiTurnFeedbackVsConvolution`` -- convolution reference, beam
feedback and no-beam feedback reference per config -- with the feedback
switched from forward Euler to the exponential coarse propagator; like the
counter-rotating runs, the extra feedback runs call ``_run_multiturn_case``
directly (the flag and the ``q_l_override`` are deliberately not part of
the ``_feedback_vs_convolution`` cache key).

``test_exponential_solver_matches_convolution_standard_q_l``
    At the standard operating point (``Q_L = 1.29e6``, per-step decay
    ``pi / Q_L ~ 2.4e-6``, where Euler and exponential are numerically
    near-identical) the exponential run holds the established 2 %
    convolution gate and reproduces the cached Euler beam-induced voltage
    to < 1e-5 per turn (measured <= 8.5e-7) -- pinning that the
    exponential branch composes with the full tracking machinery (grids,
    demodulation, carried deposits) without touching anything else.
``test_exponential_solver_low_q_l_agreement``
    Low-``Q_L`` absolute accuracy pin: harmonic 20 and ``Q_L = 32`` give
    the largest per-step Euler decay any end-to-end test exercises
    (``pi / Q_L ~ 0.098``) while ~14 % of the wake survives each turn.
    Gates: < 3 % vs the convolution per turn, < 5 % on the carried-wake
    increment ``v(k) - v(0)``, plus a non-triviality guard that the two
    propagators genuinely diverge here (> 5e-3 on the last turn), so a
    regression that ignored the flag fails. Honest caveat: both
    propagators share common ``O(1/Q_L)`` floors (IQ-envelope
    truncation, within-cell charge placement), so this observable does
    not rank their accuracy -- the discriminating test is the detuning
    one below.
``test_exponential_solver_large_detuning_beats_euler``
    The discriminating regime: a static detuning of 3.5e6 rad/s (~1100
    half-bandwidths) at the standard operating point. Euler's per-step
    magnitude factor ``sqrt(1 + theta^2)`` silently compounds to ~10 %
    per turn on the carried wake (far below its own step-size warning)
    while the exponential propagator rotates at magnitude 1 (exact).
    Gates: exponential < 1 % vs the detuned convolution on every turn,
    Euler > 5x the exponential error on carried turns (measured
    38x / 98x) and > 2 % on the last turn -- the mutation-sensitivity
    anchor of the exponential end-to-end suite (flipping the flag fails
    the 1 % gate by ~13x).


``test_energy_gain_ind_voltage_vs_nondriven_feedback.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tracks an actual ``BiGaussian`` ``mu_plus`` bunch through a real one-turn
``Simulation`` (one ``DriftSimple`` + one ``SingleHarmonicRFStation``) and
checks that the induced-voltage *energy gain* applied to the particles is the
same whether it comes from the multi-turn resonator solver (a separate wake
kick) or from the operating-point feedback (beam-induced part isolated by the
zero-intensity reference run). Runs for a stationary ``ConstantMagneticCycle``
and an accelerating ``MagneticCyclePerTurn``.

**Class** ``TestEnergyGainMTWvsNonDrivenFeedback``

``test_feedback_runs_in_full_simulation``
    The feedback tracks through a full ``Simulation`` (regression for the
    renamed ``_turn_counter`` attribute); the beam-induced kick is a few % of
    the design voltage.
``test_mtw_applies_charge_times_induced_voltage``
    The multi-turn wake applies exactly ``charge * V_induced(dt)`` per particle.
``test_applied_energy_gain_consistent``
    The wake and the feedback apply the same beam-induced energy gain
    (pointwise within ~3 % of peak, relative error < 2 %).
``test_applied_energy_gain_consistent_with_acceleration``
    Same under acceleration, after removing the common ``-delta_E_turn``
    acceleration offset; both paths reproduce the programmed reference-energy
    gain.

An opt-in debug plot (``_plot_energy_kick``) shows the applied energy kick
against arrival time; its file output is commented out (see
`Data and assets`_).


``test_feedback_phase_under_acceleration.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Validates the feedback's multi-turn wake **phase** under acceleration against
an independent analytic reference. A real *matched* ``BiGaussian`` beam is
accelerated just above transition (``gamma_t ~ 31`` at ~4 GeV) so the RF frame
slips ~0.09 ``t_rf`` per turn -- far more than at the operating point -- which
exposes any phase-handling error.

The module function ``analytic_multipass_induced_voltage`` rebuilds the
induced voltage from first principles as a double sum of the resonator wake
``exp(-Phi / (2 Q_L)) * cos(Phi)`` over every past bunch passage, with the
accumulated phase ``Phi = theta(t_o) - theta(t_e)``. The ``fixed_freq`` flag
selects the *wrong* fixed-frequency phase clock and is used only to show the
test is sensitive to the accumulated-phase handling.

**Class** ``TestFeedbackPhaseUnderAcceleration``

``test_feedback_matches_analytic_multipass_reference``
    The feedback reproduces the integrated-phase reference to < 5 % on every
    carried turn (after one overall scale fixed on the first, single-pass turn).
``test_integrated_phase_is_required``
    A fixed-frequency reference diverges (> 30 %), proving the comparison is
    sensitive to the accumulated-phase handling.
``test_setup_is_in_the_frame_slipping_regime``
    Guard: the frame slips meaningfully (> 0.2 ``t_rf``) and the bunch stays
    in-window.

**Class** ``TestSolverPhaseUnderAcceleration``

``test_solver_matches_analytic_multipass_reference``
    Reusing the same matched-beam setup, the retuning ``MultiPassResonatorSolver``
    (``delta_f=0.0``) also reproduces the integrated-phase reference, i.e. it
    accumulates the carried-wake phase as ``integral of omega dt``.

**Class** ``TestFixedFrequencyWakeWithSubsteppedFrame`` -- a fixed-frequency
(higher-order-mode) wake does not retune with the RF, so its carried-wake
phase-clock rotation is identically zero and the only acceleration error left
is the frame (arrival) time. The solver is driven on a fixed profile while the
reference frame is advanced by ``DriftSubstepped`` (no beam tracking).

``test_substepped_frame_makes_fixed_frequency_wake_exact``
    A single-beta frame (``n_substeps = 1``) diverges from the analytic
    fixed-frequency reference (> 30 %), while a sub-stepped frame reproduces it
    on every carried turn (relative error < 1 %) -- so the residual is
    frame-time granularity, fixable in the tracking rather than the resonator.

.. note::

   Like the other modules, this one defaults to ``DEBUG_PLOT = False``; set it
   to ``True`` to open the Matplotlib diagnostic plots (blocking), and leave it
   at ``False`` for a headless/CI run.


``test_wake_vs_feedback_dynamics.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A self-consistent multi-turn *dynamics* twin: the same matched ``BiGaussian``
``mu_plus`` bunch is tracked through two full ``Simulation`` rings that differ
**only** in the beam-induced-voltage model -- one uses the multi-pass resonator
wake (``MultiPassResonatorSolver``, ``delta_f = 0``), the other a matched-bias
non-driven ``IQCavityFeedbackTimingClass`` (``delta_omega = 0``). Where the
neighbouring modules pin one slice of the equivalence (one turn of applied
``dE``; the induced voltage against an analytic reference), this closes the loop
and compares the *self-consistent bunch evolution* many turns deep on the
transition-adjacent fast ramp (~4 GeV, ~0.09 ``t_rf``/turn frame slip, strong
beam loading). It borrows the matched-beam constants and ``_t_rf`` /
``_accelerating_cycle`` / ``_matched_template`` classmethods from
``test_feedback_phase_under_acceleration.py`` via an uncollected ``_MatchedBeam``
shim.

**Class** ``TestWakeVsFeedbackDynamics``

``test_centroid_tracks_between_wake_and_feedback``
    The coherent centroid of the two twin bunches stays far inside ``sigma_dt``
    on every turn.
``test_bunch_shape_tracks_between_wake_and_feedback``
    Per-turn ``sigma_dt`` and ``sigma_dE`` agree to well below 1 %.
``test_emittance_difference_stays_small_and_barely_grows``
    The rms-emittance difference stays tiny and its per-turn *slope* is small --
    a per-turn induced-voltage mismatch would pump the trajectories apart, so a
    bounded slope is the real content.
``test_design_kicks_are_identical_without_beam``
    With the beam removed the two rings coincide to machine precision (both
    apply the identical design kick), so any beam-loaded difference is purely
    the induced-voltage model.
``test_beam_loading_is_strong_and_bunch_stays_captured``
    Guard: the induced voltage reaches a sizable fraction of ``V_DESIGN``, the
    bunch executes real synchrotron dynamics (``sigma_dt`` breathes) and stays
    in-window.
``test_beam_loading_actually_drives_the_dynamics``
    Sensitivity: beam loading moves the centroid ~0.9 ``sigma`` off the
    design-only (no-beam) trajectory -- hundreds of times the wake-vs-feedback
    twin difference -- so the tight agreement is a genuine beam-loading
    cross-check, not two runs that both ignore it.
``test_debug_plot_opt_in``
    Opt-in diagnostic overlay (skipped unless ``DEBUG_PLOT``).


``test_multibunch_beam_loading.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multi-bunch / multiple-populated-coarse-cell transient beam loading. Every
other feedback test holds a single bunch in a ~1.5 ``t_rf`` window, so only 1-2
of the ~25900 coarse cells are ever populated and the intra-turn wake
propagation *between* well-separated populated cells is never exercised. This
module widens the profile to several ``t_rf`` with two or three unevenly-spaced
Gaussian bunches and checks the trailing bunch, which rides the leading
bunch(es)' carried wake, against the ``MultiPassResonatorSolver``.

The local gates are anchored a few x above the measured feedback-vs-solver
discretization floor (trailing/leading/global ~0.19 %/0.46 %/0.11 %), not a
loose 2 %, so a sub-percent error in the carried inter-cell wake fails.

**Class** ``TestSinglePassMultiBunch`` -- solver vs non-driven feedback on one
static multi-bunch profile (no ``Beam`` tracking, no ``Simulation``).

``test_two_bunch_trailing_matches_solver``
    Two bunches (2 and 6 ``t_rf``): feedback vs solver agree at the trailing
    bunch (gate 0.6 %), leading bunch (1.0 %) and whole train (0.3 %).
``test_three_bunch_trailing_matches_solver``
    Three unevenly-spaced bunches (2, 4, 7 ``t_rf``): the last bunch integrates
    two upstream wakes at different lags and still matches locally (gate 0.6 %).
``test_first_coarse_cell_precondition``
    Drives the *real* mucol coarse downsampling (``rf_beam_current`` with
    ``forbid_charge_in_first_coarse_cell=True``, exactly as the forward pass
    calls it) and asserts it returns without raising and the first coarse
    cell carries negligible charge -- the actual invariant, not a re-read of
    the builder's zeroed fine bins.

**Class** ``TestMultiBunchMultiTurn`` -- full ``Simulation`` (a macroparticle-less
dummy beam holds the static multi-bunch profile); the coarse grid propagates
turn over turn and the beam-induced part (minus a no-beam reference) is compared
per turn against the convolution. Per-turn gates are parametrised and default to
a few x above the measured floors (trailing 0.6 %, leading 1.0 %, global 0.3 %).

``test_multibunch_static``
    Static single section: the coarse grid carries the leading-bunch wake
    across the empty gap to the trailing cell.
``test_multibunch_fast_ramp``
    Transition-adjacent fast ramp: the two populated cells shift as the grid is
    rebuilt each turn, and the carried multi-bunch wake still holds.


``test_two_beam_counterrotating_feedback.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two *simultaneous* counter-rotating beams (mu+ co-rotating, mu-
counter-rotating) through the cavity feedback, under
``MainloopCounterRotatingBeams`` (each station tracked once per beam per
turn; ``beams[1]`` traverses the elements in reverse order). Two regimes,
split by the station azimuth -- with the offset-passage regime covered on
the static cycle, under acceleration and with an RF-frequency offset:

**Class** ``TestTwoBeamOffsetPassages`` -- stations away from the beams'
meeting points (two sections: arrivals ``T_rev / 2`` apart, the true pattern
of counter-rotating beams in the symmetric ring). The per-passage grid
machinery handles the alternating arrivals natively: each ``_track`` spans
the half turn to that beam's next passage, so the envelope paces at the
physical rate and carries each beam's loading into the other's passage.

``test_feedback_matches_two_beam_convolution``
    The two-beam beam-induced part (two-beam gap voltage minus the two-beam
    zero-intensity reference) matches the two-beam multi-pass convolution at
    every station and turn (gate 0.5 %; measured floors 0.13 % -> 0.04 %).
``test_two_beam_loading_exceeds_single_beam``
    Non-triviality guard: the two-beam convolution differs from the
    single-beam run by well more than the comparison gate, so the equality
    cannot hold with the counter-rotating beam silently ignored.
``test_both_stations_carry_comparable_loading``
    Symmetric ring: both stations see the full two-beam loading (peaks agree
    to a few percent; profiles differ only by their noise seed).

**Class** ``TestTwoBeamAcceleratingOffsetPassages`` -- offset two-beam
passages under acceleration (two sections on the transition-adjacent fast
ramp, ~4 GeV, ``gamma_t ~ 31``: the RF frame slips ~0.09 ``t_rf`` per
turn). Covers the composition the static class never exercises: the
single-beam accelerating multi-section frame-slip correction composed with
the per-beam **reverse** traversal of the two-beam mainloop. The
beam-induced part (two-beam gap voltage minus the two-beam zero-intensity
reference, which by linearity also cancels the common acceleration kick)
is compared over five turns against the two-beam multi-pass **retuning**
convolution (``delta_f = 0``).

``test_accel_feedback_matches_two_beam_convolution``
    After a non-degeneracy guard (the last-turn convolution carries real
    voltage), the beam-induced gap voltage matches the retuning
    convolution at every station and turn (gate 0.5 %; measured 0.13 % on
    turn 0 falling to 0.025 % on turn 4 -- the error *shrinks* as the
    carried wake builds up).
``test_accel_error_does_not_grow_per_turn``
    Bounded and non-ramping: the worst-section error stays under the 0.5 %
    gate on every turn, the fitted slope over the post-transient turns
    1..4 stays below 0.05 pp/turn (measured ~-0.011 pp/turn) and the
    last-turn error is no larger than turn 0. A mis-composed frame-slip x
    reverse-traversal correction would instead ramp several pp/turn.

**Class** ``TestTwoBeamDeltaOmegaRfOffsetPassages`` -- offset two-beam
passages with a static RF-frequency offset (``delta_omega_rf = 2000``
rad/s, two sections, static cycle -- no acceleration, isolating the
offset's frame slip from the ramp slip covered above). The last untested
corner of the ``delta_omega_rf`` demodulation slip anchor: a lone beam is
tracked *forward* by ``MainloopSingleBeam`` regardless of direction, so
only the two-beam mainloop's **reverse** element order exercises the
anchor's sign/value for the reverse stream. The beam-induced part (both
runs carry the offset, so its rotation of the empty-cavity voltage cancels
by linearity) is compared over five turns against the retuning convolution
(``delta_f = delta_omega_rf / (2 pi)``).

``test_delta_omega_rf_feedback_matches_two_beam_convolution``
    After a non-degeneracy guard (the last-turn convolution carries real
    voltage), the beam-induced gap voltage matches the retuning
    convolution at every station and turn (gate 0.5 %).
``test_delta_omega_rf_error_does_not_grow_per_turn``
    Bounded and non-ramping: the worst-section error stays under the 0.5 %
    gate on every turn, the fitted slope over the post-transient turns
    1..4 stays below 0.05 pp/turn and the last-turn error is no larger
    than turn 0 -- a reverse-stream slip-anchor sign or value error would
    ramp as the accumulated offset slip builds, which the lone-beam and
    zero-offset tests structurally cannot catch.

**Class** ``TestTwoBeamOffsetPassagesManySections`` -- the same two-beam
comparison beyond two sections. Two sections is a special layout twice over:
every station sees the beams exactly ``T_rev / 2`` apart, and the backfill
interval is empty at every station, so the backfill reference walk is never
entered. A 16-section RCS -- what ``rcs_two_beam_example`` actually runs --
has neither property, so these tests carry the validation into the regime the
shipped example uses. The 0.5 % gate is taken from the two-section class
rather than fitted to the measurement.

``test_arrival_spacing_is_never_half_a_turn``
    The premise: at four sections no station sees the beams half a turn apart
    (nor coincident), so narrowing the section counts back to the covered
    two-section layout fails here instead of silently shrinking the coverage.
``test_feedback_matches_two_beam_convolution``
    Static, at four AND six sections: the beam-induced part matches the
    two-beam convolution at every station and turn. Measured 0.128 % falling
    to 0.039 %, essentially identical to two sections.
``test_accelerating_feedback_matches_two_beam_convolution``
    Fast ramp at four sections: more sections mean more mid-turn grid
    re-seedings per turn, each at its own past-station RF frequency, so a
    mis-composed frame-slip correction has more chances to accumulate.
    Bounded and non-growing. This is the test that catches a backfill-walk
    defect the whole two-section class misses -- verified by mutation
    (dropping the last backfilled element leaves every two-section
    comparison green and fails this one).
``test_delta_omega_rf_feedback_matches_two_beam_convolution``
    RF-frequency offset at four sections: the demodulation slip anchor is
    accumulated across the stations of a turn, so a non-empty backfill
    interval exercises it differently. Bounded and non-growing.
``test_two_beam_loading_exceeds_single_beam``
    Non-vacuity: the four-section two-beam convolution differs from the
    single-beam run of the same ring by well over the gate, so the agreement
    above cannot hold with the counter-rotating beam silently dropped.


**Class** ``TestSimultaneousPassageGuard`` -- a station at a meeting azimuth
(e.g. the single mid-ring station of a one-section layout) sees both beams
at the *same* reference time. The per-passage machinery would silently
serialize the coincident arrivals one full projection window apart (envelope
at twice the physical rate; measured ~47 % L2 waveform error on the first
turn), so the feedback detects the coincident opposite-direction passage
(within half a coarse cell) and refuses it.

``test_single_section_two_beam_raises``
    The coincident second passage raises ``NotImplementedError``. The message
    points to the only supported fix -- move the station off the meeting
    azimuth -- and names the ``MultiPassResonatorSolver`` wakefield
    (``allow_delta_t_zero=True``) explicitly as *not* a substitute: that path
    deposits each coincident beam's kick before the other beam's profile is
    registered, so its kicks are wrong (0.5 and 1.5 times the correct mutual
    term, depending on track order). See the design doc's *Counter-rotating
    beams* warning.
``test_single_section_convolution_reference_needs_delta_t_zero``
    Pins that the solver *can* be made to accept a coincident passage: its
    monotonic-clock assertion rejects ``delta_t = 0`` unless
    ``allow_delta_t_zero=True``. This is used to build the single-beam
    convolution reference; it does not make the two-beam coincident kick
    correct (see the warning above).

.. note::

   Integrating two *coincident* beam currents in the feedback (deposit-sum
   into the same forward segment plus an envelope rewind/re-advance) is a
   known open extension; the offset-passage regime above is the physically
   relevant one for even section counts.

**Class** ``TestBackfillWalkDirectionConsistency`` -- a structural invariant
the physics comparisons above cannot see. Note the two directions this
module mixes: *backfill* is the time direction (reconstructing the already
elapsed stretch of grid), *reverse* is the space direction (the
counter-rotating beam's element traversal). The test is about their
interaction. ``get_time_omega_array_backfill`` takes its element *order*
from the previously tracked beam (``_last_tracked_beam_state_frwrd``) but
hands ``beam.is_counter_rotating`` -- the *current* beam -- to
``track_reference``. In a single-beam run the two always agree, so only a
two-beam run can make them differ. They stay safe because the interval to
backfill is empty: the forward projection stops at the next RF station in
the tracked beam's traversal order, which under
``MainloopCounterRotatingBeams`` is exactly where the *other* beam next
reaches this feedback, so the reference times match to the bit and the
early return in ``calculate_rf_centers_for_backfill`` fires
before the walk is entered. Both tests share one instrumented run per
regime: they patch ``calculate_rf_centers_for_backfill`` and
``get_time_omega_array_backfill`` to record every call, and cover
all three two-beam regimes of this module as subtests (static,
accelerating fast ramp, ``delta_omega_rf``), each run once and cached,
because each moves the reference clock differently. Measured in the static
case: 10 of 12 backfill calls carry a direction mismatch and none
reaches the walk; with the early return removed all 10 do. The *outcome*
cannot be compared instead -- the symmetric half-drift / station /
half-drift layout yields identical arrays for the matched and the
mismatched element order -- so the "never entered" property and its gate
are what is asserted.

``test_backfill_walk_never_entered_with_mismatched_direction``
    Both halves matter: mismatched-direction calls must actually occur
    (otherwise the beams stopped alternating and the test no longer
    exercises the configuration it guards), and none of them may reach the
    element walk.
``test_mismatched_calls_are_gated_by_exact_time_equality``
    Pins the mechanism behind it: every mismatched call carries a
    **bit-exact** zero backfill gap. The assertion is ``gap == 0.0``, not
    ``assertAlmostEqual`` -- the production early return uses ``==``, so a
    merely-approximate equality would not be safe. A companion assertion
    shows the measurement is live: at least one first passage (nothing
    projected yet) has a genuinely nonzero gap and legitimately walks.


``test_envelope_kernel.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class** ``TestEnvelopeKernelBitIdentity`` -- the numba coarse-envelope
kernel (``envelope_pi_scan``) must reproduce the pure-Python coarse
recursion bit-for-bit. Each test drives both paths with identical inputs
and asserts exact equality across the regimes the kernel must cover.
The compared snapshot covers the whole coarse state: the two
source-split component grids (``antenna_voltage_gen_coarse_grid`` and
``antenna_voltage_beam_coarse_grid``) alongside the composed
demodulation-frame sum, the generator-current grid and, when a
controller is attached, its integral and delay line.

``test_no_beam_constant_current``
    Backfill-style segment: no beam, no controller.
``test_forward_constant_current``
    Forward segment with beam but a constant generator current.
``test_forward_pi_no_delay``
    Forward segment driving a PI controller with no loop delay.
``test_forward_pi_with_delay``
    PI controller with a two-sample loop delay (the delay line).
``test_forward_pi_saturating``
    PI controller hitting the klystron clamp (the anti-windup path).
``test_exponential_solver_pi``
    The exponential propagator with an active PI controller.
``test_detuned_pi``
    Non-zero detuning with an active PI controller.
``test_no_beam_carried_generator_current_off_bias``
    Backfill segment whose carried generator current is off the bias. The
    reference drives carried cell 0 from ``last_val_generator_current``
    but every later cell from the reset-bias grid; a kernel that held the
    carried value for all cells would diverge.
``test_no_beam_carried_beam_current_nonzero``
    Backfill segment whose carried index-0 beam current is nonzero.
    ``cavity_response`` uses ``last_val_beam_current`` at the carried cell
    even for a no-beam segment; a kernel that zeroed it would diverge.
``test_forward_pi_carried_beam_current_nonzero``
    Forward PI segment with a nonzero carried index-0 beam current.
``test_split_components_with_frame_rotations``
    Both carried components plus non-unit frame rotations -- the live
    multi-section / RF-offset condition: the generator and beam
    components carry distinct nonzero state and the per-passage
    rotations are away from unity (``exp(-0.37j)`` / ``exp(0.21j)``),
    so the kernel's composition
    ``V_beam + V_gen * generator_frame_rotation`` must reproduce the
    reference multiply bit-for-bit.
``test_split_components_pi_with_frame_rotations``
    The same configuration with an active PI controller: the regulation
    of the kick-frame sum under non-unit rotations is bit-identical.
``test_multi_section_backfill_then_forward``
    A two-segment (backfill + forward) run is bit-identical.
``test_multi_section_carried_state_off_trivial``
    Two segments with off-bias / nonzero carried state, reproducing the
    live multi-section turn >= 1 condition end to end.

**Class** ``TestDegenerateCoarseSteps`` -- first-cell seeding, coincident
points and empty segments. The per-cell reference loop
(``_circuit_track_cells_python``) and its vectorised twin
(``_coarse_step_sizes``) share the first-cell special cases, and a degenerate
(coincident / zero-step) grid must make the kernel path defer to the
reference loop -- the only one that duplicates the previous cell into the
coincident one. The shared fixture is a four-centre no-beam grid whose third
centre repeats the second (optionally offset by a few ULPs).

``test_first_turn_single_cell_segment_uses_own_period_step``
    A one-cell first-ever segment has no next centre to take the step proxy
    from, so the loop falls back to this segment's own coarse step
    ``n * t_rf`` at ``omega_input``. The single centre is parked *off* the
    coarse period (``0.3 t_rf``) so the wrong choice -- its local time -- is
    distinguishable, and both the expected and the wrong voltage are
    computed and compared.
``test_single_cell_segment_vectorised_step_matches_the_reference``
    ``_coarse_step_sizes`` uses the same one-cell period proxy.
``test_coincident_points_warn_and_duplicate_the_cell``
    Two identical consecutive centres carry zero elapsed time, so the
    correct voltage there is the previous cell's, ``V(t + 0) = V(t)``. The
    cell must hold that value (not the zeros prefill), the generator current
    likewise, and the following cell must advance from the real carried
    voltage; a ``double taking of rf_centers value, duplicating`` warning is
    required and the whole grid must stay finite.
``test_few_ulp_negative_step_is_clamped_not_asserted``
    A step a few ULPs below zero (offset ``-1e-10 t_rf``) is floating-point
    noise, not an ordering violation: it is clamped to zero and handled as a
    coincident point instead of tripping the hard ordering assertion.
``test_degenerate_segment_defers_the_kernel_to_the_reference``
    ``_coarse_step_sizes`` reports the degenerate segment with ``None``, and
    ``_circuit_track_cells_kernel`` then runs the pure-Python loop -- whose
    duplicate-and-warn behaviour *and* result must appear bit-for-bit.
``test_empty_segment_is_a_no_op_on_the_kernel_path``
    ``start_index == end_index`` leaves both coarse grids untouched (checked
    against a sentinel-filled voltage grid).

**Class** ``TestControllerAbstractionContract`` -- the compiled path must
honour the ``GeneratorCurrentController`` *interface*, not a concrete PI
class. Driven by ``_ProportionalOnlyController``, a minimal non-PI
implementation with no compiled scan, over one 32-cell forward segment.

``test_non_pi_controller_runs_on_the_default_path``
    The default (kernel) path steps the controller once per cell and
    produces finite voltage and current. Reaching for PI-only attributes
    on the compiled path used to break every other implementation of the
    interface.
``test_non_pi_controller_matches_the_python_path``
    A controller without a compiled scan still reproduces the pure-Python
    reference exactly (``np.array_equal`` on both coarse grids).


``test_closed_loop_stability.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class** ``TestClosedLoopRobinsonStability`` -- closed-loop (Robinson-style)
certification that the *driven* feedback is stable over many synchrotron
periods, not merely that two induced-voltage models agree.

``test_setup_spans_many_synchrotron_periods``
    Guards that the run is long enough for a dipole oscillation to develop.
``test_bunch_stays_captured_and_loop_is_driven``
    The bunch stays captured and the PI loop actually acts on the voltage.
``test_initial_dipole_is_excited`` / ``test_nominal_dipole_stays_bounded``
    An initial dipole is excited and, under nominal gains, stays bounded.
``test_perturbed_dipole_grows`` / ``test_perturbed_grows_measurably_more_than_nominal``
    A destabilising perturbation makes the dipole grow measurably more than
    the nominal case -- the certification is sensitive to loop stability, not
    vacuous.


``test_generator_power_conservation.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Energy/power self-consistency of the generator drive and the beam-loading
compensation.

**Class** ``TestBeamLoadingCompensationSustainsSetpoint`` -- the
compensation ``I_gen = I_ff + I_beam / 2`` is a fixed point of the coarse
step, in the real cavity solver.

``test_compensation_is_a_fixed_point_for_all_detunings``
    With the compensation applied the voltage never leaves ``V_SET``, for
    every detuning swept and every beam phasor.
``test_forgetting_the_factor_two_breaks_the_fixed_point``
    In-test mutation proving the ``/ 2`` is load-bearing: the correct
    compensation holds ``V_SET`` (residual at floating noise) while the raw
    over-compensation drifts by many volts.

**Class** ``TestGeneratorPowerBeamPowerBalance`` -- the incremental klystron
forward power against the power the beam extracts. The expected beam power
is derived analytically from the phasors, not mirrored from the
implementation.

``test_balance_closes_to_one_on_resonance``
    Beam power uses the *physical* fundamental current ``I_beam / 2``, so
    ``P_beam = 0.5 Re[V conj(I_beam / 2)]`` and the ratio is 1 to machine
    precision for every beam phasor.
``test_raw_beam_current_gives_the_factor_two_shortfall``
    The discriminator: computing the beam power from the *raw* ``I_beam``
    halves the balance, so a mis-accounted single-sideband factor 2 would
    read 0.5 rather than 1. Both values are pinned.
``test_balance_with_detuning_carries_only_the_reactive_term``
    With detuning the delivered power gains a purely reactive correction
    ``delta Q_L Im[V conj(I_beam / 2)]``; the real-power normalization is
    unchanged, so detuning smuggles in no spurious factor.

**Class** ``TestPowerCurrentRoundTrip`` -- ``current_limit_from_power``
inverts ``generator_power`` in watts.

``test_power_to_current_to_power_is_identity``
    Power to current back to power returns the same watts.
``test_current_to_power_to_current_is_identity``
    Current to power back to current returns the same amps.


Shared feedback-machinery tests
-------------------------------

The modules below live one directory *above* the accelerator packages::

    tests/unittests/physics/feedbacks/

They carry no accelerator in their name, but they unit-test exactly the
production modules this page is about -- ``cavity_feedback.py``,
``rf_center_grid.py`` and ``rf_center_segment.py`` under
``blond/physics/feedbacks/``. The mucol package holds the physics and
integration half of that feature; these hold the unit half, and several
invariants the mucol suites lean on (the ">= 2 centres per segment" rule,
the per-segment residual, the coarse-cell step sizing) are pinned *only*
here. Documenting them on this page, rather than only pointing at them, is
the honest split: a reader chasing a mucol failure needs both halves.

Two things differ from the mucol package: there is no ``conftest.py`` at
this level, so these tests are **not** marked ``backend_mutation`` (the ones
that need a specific backend set it themselves), and they use ordinary
absolute imports instead of the package-relative ``stubs`` / ``support``
helpers.


``test_cavity_feedback.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

What is left of the original cavity-feedback test module after the grid
builder and the segment value class moved out (to ``test_rf_center_grid.py``
and ``test_rf_center_segment.py``): the diagnostic flags, the
multi-harmonic-station support and the coarse-cell step sizing. The
full-tracking tests use a tiny ring (harmonic 5, circumference 5 m, one
station plus one ``DriftSimple``) at 63 GeV/c.
``TestIQCavityFeedbackObservationClass`` is an empty placeholder and
collects nothing.

**Class** ``TestDiagnosticsDoNotDisableTheFeedback`` -- the debug flags must
not switch the physics off. Three flags used to look alike; only one of them
disables the correction, and it now says so in its name. The helper
``_run_one_turn`` tracks one turn of an undriven, beam-loading-free cavity
(``R_over_Q = 0``, zero generator bias) that simply decays from
``initial_voltage``, which is enough to tell a real readout (relative voltage
correction ~6) from the neutral one (exactly 1 with zero phase).

``test_default_applies_a_real_correction``
    The baseline readout is not the neutral one.
``test_diagnostics_still_apply_a_real_correction``
    ``debug=True`` used to short-circuit ``_track`` and write the neutral
    readout -- turning diagnostics on silently turned the feedback off.
``test_grid_validation_still_applies_a_real_correction``
    Same for ``validate_grid_each_turn=True``.
``test_diagnostic_flags_leave_the_readout_bit_identical``
    Both flags are observation-only, so a run with both enabled reproduces
    the default run bit-for-bit in the relative voltage correction, the
    phase correction and the coarse antenna voltage.
``test_grid_only_mode_applies_no_correction``
    ``grid_only_no_correction=True`` is the one mode that *does* switch the
    physics off: the readout is neutral, yet the grid is still built
    (``_rf_centers`` non-empty) -- which is the point of the mode.

**Class** ``TestConstructorHarmonicIndexValidation`` -- ``harmonic_index``
handling at feedback construction, mirroring the attach-time rules of
``TestAttachCavityFeedbackIndexValidation`` (both entry points share
``_coerce_harmonic_index``): plain ``int``, ``np.integer`` and integral
floats are accepted silently; a fractional value is a hard error, because
a harmonic index is a list slot, not a physical quantity to be rounded.

``test_fractional_harmonic_index_raises``
    ``harmonic_index=1.5`` raises ``ValueError`` naming the value.
``test_integral_float_harmonic_index_is_accepted_silently``
    ``1.0`` is stored as the plain ``int`` ``1`` with no warning (checked
    under ``warnings.simplefilter("error")``).
``test_numpy_integer_harmonic_index_is_accepted``
    ``np.int64(1)`` indexes per-harmonic arrays fine but is not an
    ``int``; the coercion must not reject it.
``test_non_numeric_harmonic_index_raises``
    A string index raises ``TypeError``.

**Class** ``TestMultiHarmonicParentResolution`` -- the RF-parameter accessors
when the parent is a multi-harmonic station. A stub parent carrying
per-harmonic arrays is assigned directly (``set_parent_rf_station`` accepts
only the real station classes, and only the ``isinstance`` dispatch of
``_resolve_main_harmonic`` is under test), with ``harmonic_index = 1``.

``test_harmonic_indexes_the_per_harmonic_array``
    ``feedback.harmonic`` picks slot 1 out of ``[3.0, 7.0]``.
``test_resolve_main_harmonic_indexes_the_value``
    ``omega_rf`` goes through ``_resolve_main_harmonic`` and must pick the
    tracked harmonic out of the per-harmonic array.
``test_delta_phi_rf_is_zero_before_any_slip``
    The parent's kick clock is ``None`` before the first passage, so the
    accessor must report "no accumulated slip", not crash.
``test_delta_phi_rf_indexes_the_per_harmonic_array``
    Once the parent carries a slip array, slot 1 is returned.

**Class** ``TestDegenerateMultiHarmonicMatchesSingleHarmonic`` -- the physics
anchor of the multi-harmonic support. A two-harmonic station whose second
harmonic has zero voltage *is* a single-harmonic station, so a feedback
regulating slot 0 must reproduce the equivalent ``SingleHarmonicRFStation``
run. Both three-turn runs are built once in ``setup_class`` and compared at
``rtol = 1e-12``.

``test_antenna_voltage_matches``
    The coarse antenna voltage agrees.
``test_corrections_match``
    Relative voltage correction and phase correction agree.
``test_applied_kick_matches``
    The kick the beam actually received agrees, in both ``dE`` and ``dt``.
``test_correction_is_real_not_neutral``
    Guards the anchor against passing vacuously with a switched-off
    feedback.

**Class** ``TestNonMainHarmonicAttachment`` -- a feedback attached *only* at a
non-zero slot must run end to end. Slot 0 stays empty, so every former
``cavity_feedback_list[0]`` hardcode in ``MultiHarmonicRFStation`` would
crash with ``'NoneType' object has no attribute 'profile'``.

``test_runs_and_applies_a_real_correction``
    Two tracked turns produce a non-neutral readout.
``test_grid_frequency_is_harmonic_1_design_frequency``
    The grid must run on *this* feedback's harmonic: the stored
    ``_forward_segment_omega_design`` is a scalar (not the station's
    per-harmonic array, whose slot 0 is the main harmonic) and equals
    ``calc_omega_rf_design(...)[1]``.

**Class** ``TestAttachSetsHarmonicIndexFromSlot`` -- the blessed convenience
case of slot-authoritative attachment: a feedback constructed with the
DEFAULT ``harmonic_index`` (0) and attached at slot 1 must have its index
overwritten from the slot, run end to end, and read harmonic 1's RF
parameters -- no construct-time index bookkeeping is required of the user.

``test_attach_overwrites_the_constructor_index``
    After the attach the feedback's ``harmonic_index`` is 1.
``test_runs_and_applies_a_real_correction``
    Two tracked turns produce a non-neutral readout.
``test_reads_harmonic_1_rf_parameters``
    The stored ``_forward_segment_omega_design`` equals harmonic 1's
    design frequency, not the constructor default's (slot 0, the main
    harmonic).

**Class** ``TestHarmonicSlotAgreementIsEnforcedAtRunStart`` -- a feedback's
``harmonic_index`` must equal its list slot.
``calc_gap_voltage_with_feedbacks``
applies each feedback's corrections at its LIST index while the feedback
computes them from the RF parameters at its OWN ``harmonic_index``, so a
disagreement silently applies harmonic A's corrections to harmonic B.
``attach_cavity_feedback`` SETS the feedback's index from the slot (see
``TestAttachCavityFeedbackIndexValidation`` below), so a mismatch cannot
arise through the attach; the run-start guard is reached only by tampering
with ``cavity_feedback_list`` *after* the attach -- which is what these
tests set up, and what the attach path cannot see.

``test_feedback_harmonic_1_in_slot_0_raises``
    ``on_run_simulation`` raises ``ValueError`` naming ``harmonic_index=1``
    and ``slot 0``. The remedy must be followable: re-attaching an
    already-owned feedback trips the ownership assert, so the message points
    at ``Construct the feedback with harmonic_index=0`` and must *not*
    mention ``attach_cavity_feedback``.
``test_feedback_harmonic_0_in_slot_1_raises``
    The mirror case, naming ``harmonic_index=0`` and ``slot 1``.
``test_matching_slot_passes_the_guard``
    A correctly placed feedback passes ``_validate_multi_harmonic_slot``.
``test_feedback_missing_from_parent_list_raises``
    A parent whose ``cavity_feedback_list`` does not contain this feedback at
    all (an identity check) raises, naming ``cavity_feedback_list``.

**Class** ``TestCoarseCellStepSizing`` -- per-cell step sizing of the coarse
recursion, driven on hand-built grids with ``omega = 2 pi`` (so an RF period
is 1 s and the centre times read directly).

``test_single_cell_first_turn_uses_own_coarse_step``
    The first centre ever tracked *and* the only centre of the segment: there
    is no next centre to diff against, so the step proxy falls back to the
    segment's own coarse step ``n * t_rf``. Checked against a two-centre
    reference grid whose spacing *is* one coarse step, and guarded as
    non-vacuous (the cell decayed away from the initial voltage).
``test_ulp_negative_first_step_is_clamped_and_duplicated``
    A first-cell step a few ULPs below zero is floating-point noise, not an
    ordering violation: it is clamped to zero and handled as a coincident
    point (with the ``double taking of rf_centers value`` warning) instead of
    tripping the hard assertion. A coincident *first* cell has no predecessor
    in the grid, so the state it duplicates is the value carried over the
    turn boundary -- and tracking still completes for the next cell.
``test_coincident_centers_warn_and_duplicate_previous``
    A duplicated ``rf_centers`` value carries zero elapsed time, so the
    correct voltage there is exactly the previous cell's. The cell is
    **duplicated** -- it is no longer left at the zeros prefill, which would
    restart the envelope from ``V = 0`` -- so the following cell advances
    from the carried voltage; pinned against ``_advance_coarse_voltage`` and
    against the pure drive term the old propagate-from-zero behaviour gave.
``test_coincident_last_cell_does_not_poison_the_next_turn``
    ``reset_arrays`` carries the last coarse cell into the next turn, so a
    coincident *last* cell must hold the real voltage and current, not the
    prefill, or the whole next turn starts from a dead cavity.
``test_vectorised_first_turn_step_matches_reference_path``
    ``_coarse_step_sizes``, the vectorised twin, reproduces the first-turn
    single-cell fallback step.
``test_degenerate_segment_defers_to_the_reference_path``
    A coincident (zero) step makes the vectorised sizing return ``None``, and
    the kernel path must fall back to the reference loop -- reproducing its
    warning and its result exactly (``assert_array_equal``), with a
    non-vacuity check that something was tracked.
``test_kernel_empty_span_is_a_no_op``
    ``start_index == end_index``: the kernel must return before sizing the
    empty span (proceeding would index a zero-length step array) and leave
    the grids untouched.


``test_rf_center_grid.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Coarse-grid (``rf_centers``) construction for the timing class, moved here
alongside the ``RFCenterGridMixin`` extraction. By collected count this is
the largest module of the whole feedback tree (192 tests), almost all of them
parametrisations of the geometry sweeps in the first class.

``test_rf_center_grid_mixin_self_is_typed_as_timing_class`` (module level)
    Every function in ``RFCenterGridMixin.__dict__`` must annotate ``self``
    as ``"IQCavityFeedbackTimingClass"``, so the mixin keeps exposing its
    concrete host type.

**Class** ``TestIQCavityFeedbackTimingClass`` -- grid geometry, end to end
through a real ``Simulation``. Thirteen methods expand to 185 collected tests
through two shared sweeps: ``test_data_discontinuity`` (phase shift x
``delta_omega_factor`` in ``{0, +0.13, -0.13}`` x six
``(n_rf_periods_per_coarse_grid, Q_L)`` settings, covering integer ``n`` of
1/2/3 and the sub-stepping cases 0.25/0.4/0.6, whose low ``Q_L`` puts the
Euler decay past the hard cap and therefore switches the exponential coarse
solver on), and a section-count sweep. The coarse-grid *geometry* stays on
the design RF clock under an RF-frequency offset -- the offset enters only as
explicit carrier/kick-clock phases -- so the distance checks compare against
the design period regardless of ``delta_omega_factor``.

``test_fractional_n_below_one_is_supported_without_warning``
    ``n`` in ``(0, 1)`` is the deliberate sub-stepping mode, so construction
    must not emit the ``coupling between loops`` warning.
``test_non_integer_n_above_one_still_warns``
    A non-integer number of *whole* RF periods (``n = 1.5``) de-aligns the
    coarse grid from the RF buckets and must still warn.
``test_non_positive_n_is_rejected``
    ``n = 0`` raises ``ValueError`` (``must be > 0``).
``test_stability_check_reflects_actual_step_decay``
    The step-size check must measure the *actual* Euler decay
    ``n * pi / Q_L``, not the ``n``-independent carrier product
    ``(omega_rf / n) * sampling_time_coarse == 2 pi``. With ``Q_L = 2`` and
    ``n = 2`` the decay is ``pi``, far past the hard cap, and running the
    simulation must raise ``ValueError`` naming ``decay_per_step``.
``test_for_discontinuity_distances_single_section_no_acceleration``
    The full sweep on a static cycle: consecutive coarse centres are spaced
    by exactly one coarse step, and the turn-to-turn seam
    (``rf_centers[turn][0] - rf_centers[turn - 1][-1]``) is that same step to
    ``1e-7`` relative -- i.e. the grid has no discontinuity at the turn
    boundary. The harmonic is raised to ``max(5, 2 ceil(n))`` so the
    single-segment turn holds at least two coarse centres.
``test_matched_generator_envelope_invariant_acceleration``
    Physics extension of the same sweep: on resonance with the generator
    matched to the setpoint, ``V_ss = 2 (R/Q) Q_L I_g`` is the *exact* fixed
    point of the coarse forward-Euler step. The design-anchored
    generator-sourced component must hold ``V_ss`` exactly (phase
    included) under acceleration, while the composed demodulation-frame
    sum holds the same magnitude but, under the sweep's RF-frequency
    offsets, appears rotated by minus the accumulated kick-clock slip
    (the physical walk-off of a design-locked drive) -- so only its
    ``abs`` is asserted invariant.
``test_for_discontinuity_distances_single_section_acceleration``
    The discontinuity sweep again on an accelerating cycle, where the coarse
    step itself changes turn over turn.
``test_fine_sectioning_below_two_coarse_cells_raises``
    A walked interval shorter than two coarse cells must surface the
    ">= 2 centres" ``ValueError`` from segment construction rather than
    silently build a degenerate grid (a single-centre forward segment used to
    disarm the counter-rotating coincidence guard). Ten sections at harmonic
    20 give two RF periods per section, so the turn-0 backfill before station
    0 spans one coarse centre; the message must name both remedies,
    ``reduce n_rf_periods_per_coarse_grid`` and ``fewer/longer sections``.
``test_get_slice_of_elements_this_section_cnst_cycle_fwrd``
    Multi-section static cycle, forward stream (1/4/20 sections): each
    feedback resolves the correct slice of ring elements for its own section.
``test_get_slice_of_elements_this_section_cnst_cycle_reverse``
    The same for the reverse (counter-rotating) element order. Every section
    is kept at ``>= 5`` RF buckets because the turn-0 backfill spans only
    half a section and every segment must hold at least two centres.
``test_get_slice_of_elements_this_section_accelerating_cycle_cycle_reverse``
    The reverse-order slice under acceleration (1/2/4/10 sections), where the
    per-section reference energies advance mid-turn.
``test_get_slice_of_elements_this_section_accelerating_cycle_cycle_reverse_rf_centers``
    The same configuration (1/2/4/10/20 sections) checked on the produced
    ``rf_centers`` themselves rather than on the element slice.
``test_rf_centers_full_counterrotation_equality``
    Symmetric-ring invariant (2/4/10 sections): the grid a station builds for
    the co-rotating stream equals the one its mirror station builds for the
    counter-rotating stream, turn by turn.

**Class** ``TestBackfillWalkGuards`` -- error and warning guards of the
backfill reference walk. Driven directly on the mixin with stub
station/reference/element/beam objects, so no ``Simulation`` is needed.

``test_skipped_turn_is_rejected``
    The station's turn counter must never be *behind* the turn the last
    forward projection was made in; a lower value means a turn was skipped
    and the walk cannot reconstruct it, so
    ``get_time_omega_array_backfill`` raises ``RuntimeError``
    (``was a turn skipped``).
``test_reference_overshoot_warns_about_inconsistency``
    When the walked reference lands *above* the beam's reference time the two
    clocks disagree (e.g. a ``delta_omega_rf`` applied directly to the
    stations), which must be flagged with an ``Inconsistency with references``
    warning rather than silently accepted -- while the walk still completes
    and records the overshot interval in ``_backfill_time_array`` and
    ``_backfill_segment_omega_design_list``.

**Class** ``TestBackfillWalkRestoresForeignTurnCounter``

``test_turn_counter_restored_when_track_reference_raises``
    The walk temporarily decrements the turn counter of a **foreign** RF
    station (one the feedback does not own) so that station applies the
    previous turn's schedule, then restores it. If ``track_reference`` raises
    in between, an unprotected restore would leave that station -- and every
    element tracked after it -- on a corrupted turn counter, turning a clean
    error into cascading mis-tracking. The test injects a raising
    ``track_reference`` and asserts the counter is back at its original value.

**Class** ``TestPrecedingSegmentResidualFallback`` -- the live-scalar
fall-through of ``_preceding_segment_residual`` is legal only when there are
no segments at all.

``test_hand_built_grid_without_segments_uses_live_scalar``
    Documented fall-back: hand-built grids (tests, direct ``circuit_track``
    callers) carry no segment list and must keep reproducing the historical
    live-scalar value bit-for-bit.
``test_mid_segment_start_index_on_real_grid_trips``
    A ``start_index`` landing *inside* a segment means the caller sliced the
    grid at a non-segment boundary. Returning the live scalar there would
    hand back this turn's forward tail -- a plausible-but-wrong coarse step
    -- so it raises ``AssertionError`` naming ``start_index``.

**Class** ``TestGenerateRfCentersDegenerateSegment``

``test_zero_centre_interval_warns_and_returns_empty``
    The first centre sits half an RF period into a segment, so a segment
    shorter than that contains no centre at all. ``_generate_rf_centers``
    warns with the turn/section context the ">= 2 centres" ``ValueError`` of
    the ``RFCenterSegment`` built right afterwards cannot know
    (``no rf centers in turn 4``), returns an empty array, and leaves the
    tiling carry-over residual untouched so the next segment continues from
    it.


``test_rf_center_segment.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RFCenterSegment`` value class, the flat arrays derived from it, and the
two per-segment quantities the walks read out of it. ``rf_centers`` are
segment-**local** times, which is what makes all of this delicate.

**Class** ``TestRFCenterSegment`` -- field guards and derived arrays. The
flat ``rf_centers`` / ``rf_centers_lengths`` are now *derived* from the
segment list, so length consistency holds by construction and only needs a
focused check here (it used to be reconstructed inside the timing-class
integration tests).

``test_rejects_non_positive_omega``
    ``omega <= 0`` raises (``omega must be > 0``).
``test_rejects_negative_duration``
    ``duration < 0`` raises (``duration must be >= 0``).
``test_rejects_multidimensional_centers``
    ``centers`` must be 1-D.
``test_rejects_residual_outside_duration``
    The residual is the leftover after the segment's last centre, so it must
    lie in ``[0, duration]``.
``test_rejects_empty_centers``
    An empty segment used to carry the previous segment's residual through
    without adding its own duration to the bridging coarse step; it is now
    unconstructible. The message names the offending duration (``1e-06``) and
    both remedies (``reduce n_rf_periods_per_coarse_grid``,
    ``fewer/longer sections``).
``test_rejects_single_center``
    A single-centre segment is as degenerate as an empty one: the coincidence
    guard's tolerance ``rf_centers[-1] - rf_centers[-2]`` would cross the
    segment boundary, so it raises too.
``test_len_matches_centers``
    ``len(segment)`` is the centre count.
``test_flat_arrays_derived_from_segments``
    ``rf_centers`` is the concatenation and ``rf_centers_lengths`` the
    per-segment lengths -- purely derived, so they cannot desync -- and
    ``_validate_grid`` passes.
``test_clear_segments_empties_derived_arrays``
    ``_clear_segments`` empties both derived arrays and leaves the grid valid.
``test_validate_grid_detects_direct_mutation``
    Corrupting ``rf_centers_lengths`` directly (bypassing ``_append_segment``)
    must trip ``_validate_grid`` with ``out of sync``.
``test_segments_no_overlap_in_absolute_time``
    Offsetting each segment's local centres by the cumulative durations of the
    segments before it, the absolute centre times are strictly increasing --
    even when the raw local values repeat across segments.

**Class** ``TestSegmentBoundaryStep`` -- the coarse step across a segment
boundary is a **per-segment** quantity. Because centres are segment-local,
the step into the first cell of segment ``j`` is that cell's local time plus
the *preceding* segment's residual. ``_track`` generates the whole per-turn
grid before walking any of it, so the live
``_residual_time_last_rf_centers_calculation`` scalar holds the
last-generated (forward) segment's residual by the time the loop reads it --
a value from the *future* of the walk. The fixture builds a
backfill + forward grid whose three residuals (previous turn 0.30, backfill
0.40, forward 0.50 ``t_rf``) deliberately all differ, and the comparisons are
made in the loop's own ``omega * delta_t`` units so no ULP is lost.

``test_segment_boundary_step_vectorised``
    Into the forward segment, ``_coarse_step_sizes`` uses the **backfill**
    segment's tail, not the forward segment's own live-scalar one.
``test_segment_boundary_step_reference_loop``
    The scalar reference loop does the same.
``test_turn_boundary_step_vectorised``
    Into the first segment of the turn the step crosses the turn boundary, so
    it must use the residual the *previous turn* ended on.
``test_turn_boundary_step_reference_loop``
    The reference loop agrees there too.
``test_scalar_and_vectorised_paths_agree``
    Both start indices, exact equality -- the kernel-vs-Python byte-identity
    pin depends on both paths deriving the boundary step the same way.
``test_per_turn_grid_span_lives_with_the_value_types``
    ``PerTurnGridSpan`` is the per-turn coarse-grid value type, so its
    ``__module__`` must be ``blond.physics.feedbacks.rf_center_segment`` and
    the (much larger) feedback module must merely re-export it.
``test_hand_built_grid_without_segments_keeps_live_scalar``
    Byte-compat guard for the direct-call tests: a grid built by hand, with
    no segments at all, must keep reading the live host scalar.

**Class** ``TestGuardCellWidthInvariant`` -- the coincidence-guard tolerance
``rf_centers[-1] - rf_centers[-2]``. The simultaneous-passage guard compares
arrival times against half a forward coarse-cell width taken from the last
two flat entries, so both must lie inside the forward segment for that
difference to be a genuine cell width -- which the ">= 2 centres" invariant
guarantees.

``test_single_center_forward_segment_is_unconstructible``
    Appending a single-centre forward segment after a 3-centre backfill one
    used to give a tolerance of ``-2.3 t_rf``, so ``abs(arrival difference) <
    0.5 * width`` could never fire and the guard was silently disarmed. It
    now raises (``at least two``).
``test_forward_cell_width_is_forward_segment_spacing``
    With the invariant holding, the tolerance is strictly positive and equals
    the forward segment's own cell spacing -- even when that differs from the
    backfill segment's.

**Class** ``TestBackfillSpanWalksSegments`` -- the per-passage walks read the
segment records rather than parallel arrays. ``RFCenterSegment`` carries the
frequency and the time span its centres were generated over, so the backfill
replay and the multi-section registration phase take ``omega_k`` and
``T_seg,k`` from the segments themselves. The backfill segments of a passage
are ``_segments[:-1]``: the grid is cleared at the start of every passage,
the backfill generation appends exactly one segment per elapsed frequency
span, and the forward generation then appends exactly one more. The fixture
holds three backfill segments at different frequencies (the middle one at the
two-centre minimum) plus the forward one, with ``circuit_track`` recorded
instead of executed.

``test_replay_walks_backfill_segments_only``
    ``_replay_backfill_span`` makes exactly one no-beam pass per backfill
    segment, at that segment's own ``omega`` and over exactly its own
    centres; the forward segment is left to the real forward pass. Pinned as
    an exact list of ``(omega, start, end, no_beam)`` tuples.
``test_replay_is_a_no_op_without_backfill_centers``
    The gate: a passage that generated no backfill centre must not walk
    anything (a stale frequency list used to re-run the whole grid).
``test_registration_phase_sums_segment_omega_times_duration``
    ``_accumulate_registration_phase`` returns
    ``Psi = sum_k (omega_k - omega_0) T_seg,k`` over the backfill segments,
    both factors taken from the segment records -- and the result is checked
    to be non-zero, so the segments really do differ from ``omega_0``.


``test_beam_feedback.py``
^^^^^^^^^^^^^^^^^^^^^^^^^

Two guards on the shared ``BeamFeedbackBase``. Both answer the same question
-- "does this RF station carry a cavity feedback?" -- and neither may answer
it with the main-harmonic-only accessor ``get_main_harmonic_cavity_feedback``:
a cavity feedback may be attached to a **non-main** harmonic of a
``MultiHarmonicRFStation``, in which case the main-harmonic slot is ``None``
while a feedback is regulating. The module was previously a stub. A minimal
concrete subclass supplies the two abstract per-turn hooks as no-ops and the
guards are called directly, so nothing is tracked.

**Class** ``TestCavitySumPhaseGuard`` -- ``cavity_sum_phase`` must not skip
its own ``NotImplementedError``. Coupling the phase loop to the cavity
feedback is a deliberate non-goal -- the two must not couple at all -- so
the raise is the permanent contract whenever a station carries any cavity
feedback, not a stub awaiting an implementation. Every raise is checked to
still name both APIs involved (``I_BEAM_COARSE``,
``antenna_voltage_coarse_grid``); the message's ``open design task``
phrasing, also pinned verbatim, predates that ruling.

``test_raises_for_feedback_on_a_non_main_harmonic``
    The silent skip that motivated the guard: with the feedback on slot 1 the
    main-harmonic accessor returns ``None`` and the phase loop used to run as
    if no cavity feedback existed at all.
``test_raises_for_feedback_on_the_main_harmonic``
    Slot 0 raises as well.
``test_raises_for_feedback_on_a_single_harmonic_station``
    So does a ``SingleHarmonicRFStation`` carrying a feedback.
``test_raises_when_only_a_later_station_carries_a_feedback``
    The scan must not stop at the first feedback-free station.
``test_silent_without_any_cavity_feedback``
    Anti-regression, as subtests over a multi-harmonic and a single-harmonic
    station: the LHC and SPS beam controls call ``cavity_sum_phase``
    unconditionally every turn, and a station without a cavity feedback is a
    normal supported setup, so the call must return ``None`` quietly.

**Class** ``TestMixedCavityFeedbackWarning`` --
``check_main_rf_stations_with_cavity_feedback`` warns only on genuine
mixtures. The helper filters the caught warnings down to the
``do not have a cavity feedback model`` fragment.

``test_no_warning_when_no_station_has_a_feedback``
    All-bare is a uniform setup, not a mixture.
``test_no_warning_when_every_station_has_a_feedback``
    The regression: the second station's feedback sits on a *non-main*
    harmonic, which the main-harmonic-only accessor reported as "no
    feedback".
``test_no_warning_for_a_single_station_with_a_feedback``
    A single covered station is not a mixture either.
``test_warns_when_only_some_stations_have_a_feedback``
    A genuine mixture emits exactly one warning.


Neighbouring modules in the same directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These sit beside the four above but test different code, so they are named
here rather than documented:

``test_base.py``
    ``TestLocalFeedbackBase`` and ``TestGlobalFeedbackBase`` -- parent-station
    conformance and the cavity list of ``blond.physics.feedbacks.base``.
``test_helpers.py``
    ``TestLowPass``, ``TestIQ`` and ``TestACSSparseModel`` -- the LHC-side
    first-order solver and I/Q helpers. The module it was named after,
    ``blond.physics.feedbacks.helpers``, no longer exists: its re-export
    shims were dissolved, so these tests now import ``low_pass_filter``
    from ``blond.physics.feedbacks.beam_current``,
    ``cavity_response_sparse_matrix`` from
    ``blond.physics.feedbacks.cavity_solvers`` and the I/Q conversions from
    ``blond.physics.feedbacks.iq``. (The mucol suite's own
    ``test_helpers.py`` under ``accelerators/mucol/`` is a different module.)
``test_cavity_feedback_requires.py``
    One module-level test,
    ``test_timing_on_run_simulation_carries_own_requires``: the timing class
    overrides ``on_run_simulation`` without calling ``super()``, so the
    decorated base method is dead code and the override that actually runs
    must carry its own ``requires`` metadata
    (``["RFStationBaseClass", "BeamBaseClass"]``) or the init-ordering
    constraint would be silently dropped.

The remaining packages under ``accelerators/`` -- ``lhc/``, ``ps/``, ``psb/``
and ``sps/``, each a single ``test_beam_feedback.py`` holding
``TestLHCBeamFeedback``, ``TestPSBeamFeedback``, ``TestPSBBeamFeedback`` and
``TestSPSBeamFeedback`` -- are the *beam*-feedback (phase/radial loop) suites
of the other machines. They are collected by a run of the whole feedback tree
but are not part of the muon-collider cavity feedback and are not documented
here.


Guards tested outside the feedbacks tree
-----------------------------------------

Three modules outside ``tests/unittests/physics/feedbacks/`` carry tests the
cavity feedback depends on. Unlike the shared machinery above, these modules
are *not* about the feedback: they belong to the profile, resonator-solver
and RF-station suites and are dominated by tests with nothing to do with this
page. Only the feedback-driven classes and method groups are documented here,
each with its home path; the rest of those modules is deliberately out of
scope.


The profile-window-versus-span guard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shared infrastructure, not mucol code: ``ProfileBaseClass.profile_duration``
(the outer-edge span ``cut_right - cut_left``, exactly
``n_bins * hist_step``) and ``ProfileBaseClass.check_fits_in_span``, which
**raises** ``ValueError`` with a one-``hist_step`` tolerance. It replaced two
separate guards that used to disagree by one bin. The mucol-side consumer --
the coarse-grid re-binning in ``rf_beam_current`` -- is covered on this page
under ``TestRfBeamCurrentDownsampling``.

**Class** ``TestProfileWindowFitsInSpan``
(``tests/unittests/physics/test_profiles.py``) -- one check for every
consumer that has to place a profile window inside a time span it does not
control, and the span means the same thing for both consumers: the interval
between two consecutive passages of the consuming element. A re-binning
consumer (the feedback's coarse grid) folds the window onto a fixed grid
covering that interval, so a window longer than the span puts two parts of
the beam on the same cell and one replaces the other; a per-passage consumer
(``MultiPassResonatorSolver``) shifts its stored deposits by that interval,
so an over-long window overlaps the previous deposit and the same charge is
deposited twice. Both destroy charge, so the guard raises for both.

``test_window_duration``
    The window duration is ``cut_right - cut_left``.
``test_window_duration_is_n_bins_times_hist_step``
    The window is the **outer-edge** span: ``cut_left``/``cut_right`` sit
    half a bin outside the first/last centre, so the duration is
    ``n_bins * hist_step`` -- exactly one ``hist_step`` more than the
    first-to-last-centre distance the deleted module-level guard used.
``test_raises_when_window_longer_than_span``
    A window longer than the span raises, naming the span.
``test_accepts_window_shorter_than_span``
    The ordinary case is silent.
``test_accepts_window_equal_to_span``
    A window matching the span exactly is legal, not an overlap --
    ``MultiTurnWake`` builds exactly that geometry.
``test_tolerance_defaults_to_one_bin``
    A sub-bin overshoot is discretisation noise (the window is derived from
    bin centres, so an equality case can miss by a fraction of a bin through
    float arithmetic) and stays silent.
``test_raises_when_overshoot_exceeds_the_tolerance``
    Beyond the one-bin slack the guard still fires.
``test_message_names_both_durations_and_the_consumer``
    The message carries both numbers and the consumer's name -- a
    per-passage consumer has no ``span_description`` meaningful to the user,
    so the name is what identifies it.
``test_zero_span_is_not_judged``
    ``span <= 0`` means the consumer has coincident passages (the two-beam
    meeting-azimuth case), which its own guard already reports; this check
    must not pile a second failure on top.
``test_sentinel_span_below_one_bin_is_not_judged``
    Callers that must satisfy a strictly-positive clock assertion on a first
    deposit advance the reference by ``eps``. That is orders of magnitude
    below one bin, resolves no passage at all, and must not be read as a span
    the window overshoots.
``test_span_just_above_the_tolerance_is_judged_again``
    The boundary of the previous escape: a span above one bin is a real span,
    so an over-long window must still be rejected. The escape hatch is not a
    blanket bypass.


The solver call site
^^^^^^^^^^^^^^^^^^^^^

Seven methods on **Class** ``TestMultiPassResonatorSolver``
(``tests/unittests/physics/impedances/test_solvers.py``) cover the
per-passage consumer of the same guard; the class's other tests are the
ordinary solver suite and are out of scope here. The shared helpers build the
solver against a *real* ``StaticProfile`` (21 bins of 1e-10 s, so
``profile_duration = 2.1e-9`` s) -- on the mocked profile the shared
``setUp`` installs, the guard would itself be a mock and never run -- and
seed a past deposit exactly the way ``_update_potential_sources`` records
one, including the depositing beam's rotation tag.

``test_profile_wider_than_the_passage_interval_raises``
    Stored past profiles are shifted by ``delta_t`` every passage, so a
    profile covering more time than that overlaps the *same beam's* previous
    window and the same charge is summed twice into the induced voltage.
    Nothing detected this before (the only clock check was ``delta_t > 0``).
    It raises rather than warns: the false-positive sweep over
    ``tests/unittests/physics/`` found the widest genuine window/span ratio
    to be 0.075, with every larger ratio belonging to the deliberate boundary
    pins in this class.
``test_first_deposit_mid_ring_is_not_judged``
    The very first passage cannot overlap anything, so it is exempt: a
    once-per-turn wakefield placed mid-ring first sees the beam a fraction of
    a revolution in, while the window it is given legitimately spans a full
    turn.
``test_interleaved_two_beam_deposits_are_not_judged``
    A second beam's deposit is measured against *its own* passages. Two
    counter-rotating beams deposit alternately, so the gap between
    consecutive deposits is the arrival offset of the two beams, not either
    beam's passage interval; overlapping deposits there carry different
    beams' charge, which is the cross-wake the solver exists to model.
``test_previous_passage_time_picks_the_same_beam_deposit``
    The direction tag, not recency, selects the passage compared against:
    with both beams' deposits stored each direction resolves to its own most
    recent one, and a beam that has not deposited yet resolves to ``None`` so
    the guard can skip it.
``test_profile_shorter_than_the_passage_interval_stays_silent``
    The ordinary case -- a bunch far shorter than a passage.
``test_passage_equal_to_the_window_is_accepted``
    Pins the threshold against the one-bin ambiguity the unified guard
    removed: with the window defined as the outer-edge span, a ``delta_t``
    equal to it places the new deposit exactly after the previous one.
``test_coincident_passage_does_not_add_a_span_failure``
    ``allow_delta_t_zero=True`` already warns at construction that the kicks
    become order-dependent; a ``delta_t`` of zero must not additionally
    produce a span failure and break a case the user explicitly opted into.
``test_coincident_passage_warns_that_the_result_is_wrong``
    The construction-time warning fires for *every* use of the flag,
    including the legitimate single-beam ones, so it can only say that
    something may go wrong. A deposit that actually lands on top of an
    earlier one at the same reference time is the broken case, and it is
    broken unconditionally -- the earlier passage's kick was returned
    before this profile existed, and this passage takes the earlier
    deposit's full ``W(0)`` where the beam-loading theorem gives
    ``W(0) / 2``. A second, runtime ``UserWarning`` therefore says the
    result is wrong at the point that becomes true, and must name the
    defect (``W(0)``) rather than merely assert badness.
``test_ordinary_passage_does_not_warn_about_wrong_results``
    The flag alone does not make every passage suspect: with a positive
    ``delta_t`` the run stays silent.
``test_first_passage_does_not_warn_about_wrong_results``
    A degenerate clock with nothing deposited yet is not the broken case --
    there is no earlier kick to have missed anything -- so it must not raise
    a false alarm.
``test_coincident_passage_warns_only_once_per_solver``
    One-shot, not one per passage: a meeting-azimuth station is coincident
    on every turn for every beam, so a per-passage warning would bury the
    message and pay the warnings-filter machinery inside the tracking loop.


Attach-time harmonic-slot validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class** ``TestAttachCavityFeedbackIndexValidation``
(``tests/unittests/physics/test_cavities.py``) -- the attach-time half of the
slot agreement whose run-start half is
``TestHarmonicSlotAgreementIsEnforcedAtRunStart`` above. The station applies
each feedback's corrections at its LIST slot while a feedback computes them
from the RF parameters at its own ``harmonic_index``, so a disagreement is
silently wrong physics rather than a crash. The slot is authoritative:
attaching SETS the feedback's ``harmonic_index`` from the slot it is placed
at, silently overriding any constructor value, so the two cannot disagree
through this path. A slot that does not exist is rejected, and a fractional
slot is a hard error -- a harmonic index is a list slot, not a physical
quantity to be rounded. The fixture is a three-harmonic station and a
mocked ``LocalFeedback`` that optionally declares a ``harmonic_index``;
every rejection additionally asserts that nothing was attached
(``any_feedback_not_none`` is false).

``test_negative_harmonic_index_raises``
    ``-1`` passes the ``> n_rf - 1`` bound check and would write to the LAST
    slot, so it raises ``ValueError`` naming ``harmonic_index``.
``test_integral_float_harmonic_index_is_coerced``
    A float ``2.0`` used to reach the list assignment and die there with
    ``list indices must be integers or slices, not float``; it is coerced
    to the plain ``int`` slot 2.
``test_fractional_harmonic_index_raises``
    ``1.5`` raises ``ValueError`` naming the value: there is nothing
    meaningful to round a fractional list slot to.
``test_numpy_integer_harmonic_index_is_accepted``
    ``np.int64`` is a perfectly good list index but not an ``int``, so the
    coercion must not reject it.
``test_non_numeric_harmonic_index_raises``
    A string index raises ``TypeError``.
``test_mismatched_feedback_harmonic_index_is_set_from_slot``
    A feedback declaring ``harmonic_index=1`` attached at slot 0 lands in
    slot 0 with its ``harmonic_index`` overwritten to 0 -- a mismatch
    cannot survive the attach.
``test_feedback_without_harmonic_index_still_attaches``
    Duck-typed feedbacks that never declare a harmonic keep working exactly
    as before -- and do not have one grafted on.
``test_matching_feedback_harmonic_index_attaches``
    A matching pair attaches (and stays matched).
``test_list_path_sets_feedback_harmonic_index_from_position``
    The whole-list form of ``attach_cavity_feedback`` sets indexes too: a
    feedback declaring slot 2 placed at list position 1 is overwritten to
    slot 1.
``test_list_path_accepts_matching_feedback_harmonic_index``
    The matching whole-list form attaches.


Support modules
---------------

These are imported by the test modules and are not collected as tests
themselves.

``stubs.py``
    Lightweight, deepcopy-able mock objects that expose only the few
    attributes the solvers read, so the cavity-response and multi-turn
    resonator solvers can run without a full ``Beam`` or ``Simulation``:
    ``StubReference`` (reference time/beta), ``StubBeam`` (intensity,
    particle type, rotation direction and the direction-signed charge
    ``signed_charge_with_direction()``) and ``StubRFStation`` (fixed design
    RF frequency and a no-op reference tracking).
``support.py``
    Numeric helpers shared across the test modules: ``rel_err`` (relative L2
    error ``||a - b|| / ||b||``) and ``lab_frame_voltage`` (projects the
    complex antenna-voltage envelope back to the real lab frame, with both
    demodulation-sign conventions).
``mucol_cav_fdbk.py``
    Not a ``pytest`` module: a standalone driver for the full muon-collider
    RCS cavity-feedback simulation. ``setup_and_run`` builds an RCS
    (``RCS1``/``RCS2``/``RCS4``) ring with per-station profiles and either the
    convolution wake (``MTW=True``) or the I/Q feedback, derives the cavity
    parameters (``Q_L``, generator current ``I_g``, detuning ``delta_omega``)
    from the working point, optionally matches the beam with the
    ``SemiEmpiricMatcher`` (``match_beam``) and runs the cycle. Run directly,
    it compares the wake and feedback runs interactively (plots from
    ``plotting.py``). The single-turn fine-grid versus resonator benchmark it
    used to carry is covered, with assertions, by
    ``TestFineGridResonatorBenchmark``.
``plotting.py``
    Interactive/diagnostic plotting helpers for the driver's observations:
    ``plot_results`` (bunch statistics of the MTW vs feedback runs),
    ``plot_ind_volt_cav_fdbk_voltage`` (induced voltage against the
    cavity-feedback voltage per station) and
    ``plot_generator_power_and_voltage`` (klystron power and antenna-voltage
    swing of a PI-feedback run) and ``plot_antenna_voltage`` (coarse-grid
    antenna-voltage evolution of a feedback instance; moved here from the
    timing class, where it was an unused debug method). Not a test module.
``conftest.py``
    Loaded automatically by ``pytest``; it pins the backend for the whole
    package. The mucol suites are written against the import-time default
    (``Numpy64Bit`` with the ``python`` specials) -- the feedback signal
    processing is host-only by design and the tests build profile arrays
    with plain NumPy -- but in a full-suite session the global backend can
    arrive in any state, so every test re-pins it first. The pin is
    applied from ``pytest_runtest_setup`` because it has to land *before*
    ``setUpClass``, where several suites build their profiles, beams and
    simulations; the autouse fixture only re-applies it per test (a
    fixture alone would run too late). Setting ``BLOND_BACKEND_MODE=cuda``
    pins to the GPU backend instead, which is how the package's
    CuPy-safety is validated. Pinning without restoring is itself a
    backend mutation, so the same file marks every test in the package
    ``backend_mutation`` (see the warning under `Running the tests`_).
``__init__.py``
    Marks the directory as a package so the test modules can use the
    package-relative imports of ``stubs`` and ``support``.


Data and assets
---------------

``fdbk_testing/init_distr_convol_RCS1_n_stations_1.npz``
    Cached matched initial beam distribution (``dt``/``dE`` arrays) for the
    single-station RCS1 setup, loaded by ``mucol_cav_fdbk.setup_and_run`` (via
    ``load_beam_coordinates_from_file``) to skip the expensive beam matching.
    The ``fdbk_testing/`` directory is not tracked by git; on a fresh checkout
    run ``setup_and_run(..., MTW=True, force_rematch=True)`` once to (re)create
    the cache before using the default ``force_rematch=False``.
``energy_kick_over_time.png``
    Not written by default. The opt-in debug plot in
    ``test_energy_gain_ind_voltage_vs_nondriven_feedback.py``
    (``_plot_energy_kick``, the applied energy kick versus arrival time for
    the wake and feedback paths) only calls ``plt.show()``; its
    ``plt.savefig`` under this name is commented out next to a note on how to
    re-enable it. Nothing in the suite writes files to the source tree.


Running the tests
-----------------

From the ``BLonD`` project root, run the whole suite with ``pytest``:

.. code-block:: bash

   pytest tests/unittests/physics/feedbacks/accelerators/mucol/

or a single module / test, for example:

.. code-block:: bash

   pytest tests/unittests/physics/feedbacks/accelerators/mucol/test_helpers.py
   pytest "tests/unittests/physics/feedbacks/accelerators/mucol/test_mucol_cav_fdbk.py::TestFineGridResonatorBenchmark"

To include the shared machinery of `Shared feedback-machinery tests`_, run
the whole feedback tree instead -- which also picks up the other
accelerators' beam-feedback tests:

.. code-block:: bash

   pytest tests/unittests/physics/feedbacks/

.. warning::

   The mucol package's ``conftest.py`` marks **every** test *of that
   package* ``backend_mutation`` (it re-pins the global backend without
   restoring it). A run filtered with the repo's standard
   ``-m "not backend_mutation"`` therefore deselects the whole mucol
   suite and reports zero tests -- silently, with no error. Drop it, or
   select the suite explicitly with ``-m "backend_mutation"``, when you
   mean to run these tests. The marker is applied per collected item and
   only for items under the mucol directory, so the shared modules one
   level up are *not* marked and survive that filter; a few of them carry
   the marker individually, on their own decorator.

By default the pin is the import-time host backend (``Numpy64Bit`` with
the ``python`` specials), which is what these suites are written and
validated against. Setting ``BLOND_BACKEND_MODE=cuda`` pins to the GPU
backend instead -- that is how the package's CuPy safety is validated;
any other value keeps the host pin.

The debug plots are opt-in via the ``DEBUG_PLOT`` module constant (and
``PLOT_DIAGNOSTICS`` in ``test_generator_current_pi_feedback.py``); both default
to off in every module, so nothing opens in a headless/CI run.


Mixin host contract
^^^^^^^^^^^^^^^^^^^

**Class** ``TestMixinsDeclareTheirHost``
(``tests/unittests/physics/feedbacks/test_cavity_feedback.py``) -- both
feedback mixins are pure moves out of ``IQCavityFeedbackTimingClass``: their
methods run on a host instance and read host state they do not define. The
dependency is real either way; these tests pin that it is *stated* rather
than left for the reader to reconstruct from the attribute accesses.

``test_every_method_annotates_self_as_the_host``
    Every method of ``RFCenterGridMixin`` and ``GeneratorRegulationMixin``
    annotates its ``self`` as ``IQCavityFeedbackTimingClass``. Parametrised
    over both mixins, so neither can drift from the other -- which is exactly
    what had happened: the grid mixin carried the annotation and the
    regulation mixin did not.
``test_host_is_not_imported_at_runtime``
    The annotation stays type-checking-only. The host inherits from these
    mixins, so a runtime import of it would be a cycle; the test asserts the
    module exposes no such name at runtime.


Attribute visibility contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An audit of every attribute of the feedback classes (171 attributes, ten
modules) found six whose declared visibility no longer matched their real
consumers. The moves are pinned so they cannot silently regress.

``TestCoarseGridAccessorsAreStatedPublic``
    (``tests/unittests/physics/feedbacks/test_cavity_feedback.py``) The
    coarse grid was read across a class boundary through private names:
    ``IQCavityFeedbackObservation`` computed the forward offset from the
    feedback's ``_rf_centers`` and ``_rf_centers_lengths``. That made them
    public API in everything but name, so the class now exposes read-only
    ``rf_centers``, ``rf_centers_lengths`` and ``forward_offset``
    properties. The storage stays private and unrenamed deliberately: the
    flat arrays are derived from ``_segments`` (the grid's source of
    truth), so a writable accessor could desync them. The tests pin that
    each property returns the stored value, that ``forward_offset`` equals
    the expression it replaced, and that all three reject assignment.
``TestForwardWalkReverseIndexIsPrivate``
    (``tests/unittests/physics/feedbacks/test_rf_center_grid.py``)
    ``reference_index_until_tracked_reverse`` was a missed underscore --
    its sibling ``_reference_index_until_tracked`` is assigned two lines
    earlier and was already private, and the only reader is the backfill
    walk's own start-index selection. Writing it between the forward
    projection and the backfill would start the counter-rotating walk at
    the wrong element. The test pins the private name and that the old
    public one is gone.
``test_len_coarse_max_is_readable_but_not_assignable``
    (``tests/unittests/handle_results/test_observables.py``) The recorder
    width is derived once in ``on_run_simulation`` and the buffers are
    allocated from it, so a later write would disagree with the arrays
    already in memory. It is now private storage behind a read-only
    property.
``test_on_wakefield_init_stores_circumference_privately``
    (``tests/unittests/physics/impedances/test_solvers.py``)
    ``MultiPassResonatorSolver.circumference`` is derived late-init state
    with a single in-class reader, now ``_circumference``. It has no
    ``__init__`` default on purpose: the attribute exists only once the
    wakefield-init hook has run, so a hand-wired solver that never got
    that hook fails loudly instead of retuning against a silent ``None``.
``test_calc_induced_voltage_past_deques_stay_parallel``
    Guards a fixed memory leak. ``_past_charge_per_macroparticle`` is one
    of eight parallel deques but was the only one never popped when
    decayed profiles were evicted, so it grew without bound over a long
    run. Results were never wrong -- ``appendleft`` keeps the front
    indices aligned with the surviving profiles, so only unreachable tail
    entries accumulated. Note the eight are parallel *between* passages,
    not during one: ``calc_induced_voltage`` appends the charge entry
    after ``_update_potential_sources`` has already appended the other
    seven and run the eviction, so at pop time the charge deque is
    legitimately one shorter.

``GeneratorCurrentPIController.n_delay`` was demoted the same way, in the
mucol controller suite: it is read exactly once, inside ``__init__``, to
size the delay-line deque, so a post-construction write was silently
ignored. The read-only property makes that mistake raise.
