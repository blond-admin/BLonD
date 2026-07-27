.. _mucol_cavity_feedback_tests:

Muon Collider Cavity-Feedback Test Suite
========================================

This page documents the test suite for the muon-collider RF cavity feedbacks.
The files live in the source tree under::

    tests/unittests/physics/feedbacks/accelerators/mucol/

and exercise the longitudinal-beam-loading models used for the muon-collider
Rapid-Cycling Synchrotrons (RCS):

* the I/Q cavity-feedback timing model
  (``blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass``),
* the standalone PI generator-current controller
  (``blond.physics.feedbacks.generator_current_controller.GeneratorCurrentPIController``)
  and the feedback's controller-driven mode
  (``IQCavityFeedbackTimingClass(controller=...)``),
* the cavity-response solvers (``blond.physics.feedbacks.helpers`` /
  ``blond.physics.feedbacks.cavity_solvers``) and the beam-current
  demodulation (``blond.physics.feedbacks.beam_current``), and
* cross-checks of the feedback against the multi-turn resonator wake
  (``blond.physics.impedances.solvers.MultiPassResonatorSolver``).

The tests are written for the ``unittest`` framework but are collected and
run with ``pytest``. They share a small set of mock objects and numeric helpers
(see `Support modules`_), so the ``mucol`` directory is a package (it and every
directory above it carry an ``__init__.py``) and the test modules use
**package-relative imports** (``from .stubs import StubBeam``) for those shared
helpers.

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
(``IQCavityFeedbackTimingClass``): the discrete step-size sanity checks and a
single-turn benchmark of the beam-loading response.

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
    ``on_run_simulation`` replaces ``init_voltage`` with the pre-fill seed.
``test_injection_voltage_without_n_pretrack_raises``
    ``injection_voltage`` without a ``n_pretrack`` budget raises.

**Class** ``TestBaseCoarseGridSizing`` -- the shared ``IQCavityFeedback`` base
(the timing class overrides ``on_run_simulation``, so this covers the base path
used by the other IQ cavity feedbacks).

``test_on_run_simulation_sizes_coarse_grid_as_int``
    ``on_run_simulation`` sets ``n_samples_coarse`` to a Python ``int`` (floor
    of turns-per-cell) and allocates the coarse arrays with it, so ``np.zeros``
    accepts the length on numpy >= 2.

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

**Class** ``TestFeedbackControllerDelegation`` -- the feedback delegates the
error-to-current conversion to its controller.

``test_update_delegates_error_and_step_to_controller``
    The controller update receives the correct antenna-voltage error and
    per-step time.
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
(half-drift / station / half-drift per section) with the reverse/forward
reference tracking, under acceleration, with a
``GeneratorCurrentPIController`` regulating every station. Each configuration
asserts physical behaviour and then *pins* the end-of-turn antenna-voltage and
generator-current trajectories against hardcoded reference values
(characterization: any change of the tracked feedback numerics shows up here
first). Setting the ``PI_TRACKING_PRINT_PINS`` environment variable prints the
recorded trajectories instead (used to regenerate the pins); while the pins
are unrecorded (``None``) the pin tests skip.

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
operating-point (63 GeV, slow) ramp. The fast ramp is excluded on purpose:
a driven multi-section cavity on the fast ramp carries a constant-bias
reference-frame slip between the forward station and the mid-turn re-seeded
segments (the induced part still cancels in the linear reference subtraction the
non-driven comparisons use, but the *driven* voltage does not), so a pinned
fast-ramp PI trajectory would characterise that slip rather than the loop.

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

**Class** ``TestPIReverseSpanFrameConsistency`` -- the PI loop must act only on
the forward (real-beam) coarse cells, never on the ``no_beam`` reverse
reconstruction segments that rebuild the previous turn. Stepping the controller
on the reverse cells would double-advance its delay line and integrator on
frame-rotated errors; the fix gates the controller update on ``not no_beam``.
The tests instrument the controller call count against the recorded per-turn
forward and total cell counts.

``test_controller_stepped_only_on_forward_cells``
    Two-section fast ramp: the controller is stepped on exactly the forward
    cells and never on the (larger) reverse reconstruction segments.
``test_single_section_controller_skips_turn0_reverse``
    Control: a single-section ring still reconstructs its very first turn in
    reverse (``n_total > n_forward`` on turn 0), and the gate skips those
    reverse cells too.


``test_helpers.py``
^^^^^^^^^^^^^^^^^^^

Tests for the cavity-response solvers (first-order in
``blond.physics.feedbacks.helpers``, shared with the LHC feedback;
second-order in ``blond.physics.feedbacks.cavity_solvers``) and the
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
    ``IQCavityFeedbackTimingClass(second_order=...)`` reproduces the matching
    standalone solver bit-for-bit and lands far closer to the convolution.

An opt-in debug plot (``DEBUG_PLOT``, ``_plot_convergence``) shows the
convergence slopes and the residual against the convolution solver.

**Class** ``TestRfBeamCurrentDownsampling`` -- charge conservation of the
coarse-grid downsampling in ``rf_beam_current``. Regression test for a dropped
remainder that used to silently discard demodulated charge past the last
coarse-cell boundary (up to the whole bunch, depending on its phase).

``test_downsampling_conserves_demodulated_charge``
    Re-binning the fine-grid demodulated charge onto the coarse grid conserves
    the complex sum, for bunches swept across the cell boundaries.
``test_warns_when_beam_maps_before_turn_zero``
    A large negative time shift maps a coarse index below zero, which warns
    that part of the beam sits before turn time 0.
``test_error_when_first_coarse_cell_populated``
    With ``forbid_charge_in_first_coarse_cell=True`` (used by the feedback to
    avoid double-counting), charge in the first cell raises.
``test_no_error_when_first_coarse_cell_empty``
    A mid-window bunch leaves the first cell numerically empty (the guard uses
    a relative threshold, not ``!= 0``).
``test_warns_on_particle_loss``
    Warns when the profile does not capture the whole beam.
``test_no_warning_when_profile_captures_full_beam``
    No warning when the window captures everything.

**Class** ``TestRfBeamCurrentCounterRotating`` -- direction-signed charge in
the RF beam current. In the symmetric muon-collider ring the counter-rotating
mu-minus beam has opposite charge *and* opposite direction, so its gap current
has the **same sign** as the co-rotating mu-plus beam. The source side of
``rf_beam_current`` / ``rf_beam_current_partial`` uses
``beam.signed_charge_with_direction()`` (charge negated for a counter-rotating
beam), matching the RF-kick and wake-kick conventions; for co-rotating beams it
equals the plain particle charge, so the shared (LHC) path is bit-unchanged.

``test_counter_rotating_mu_minus_matches_co_rotating_mu_plus``
    CR mu-minus current is bit-identical to the mu-plus current on both the
    shared and the mucol downsampling paths (was exactly sign-flipped before
    the fix).
``test_co_rotating_mu_minus_flips_the_sign``
    Charge alone (same direction) still flips the current -- ordinary
    opposite-charge physics untouched.
``test_counter_rotating_mu_plus_flips_the_sign``
    Direction alone (same charge) flips the current -- the complementary
    corner of the sign matrix.
``test_co_rotating_signed_charge_is_the_plain_charge``
    The bit-identity guarantee for the shared LHC path.


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
feedback's coarse grid is propagated turn over turn through the reverse/forward
reference tracking, and its beam-induced gap voltage (minus a no-beam reference
run) is compared per turn and per section against the accumulating convolution
voltage. Uses a high ``Q_L = 1.29e6`` so the previous-pass wake survives
(~88 % per turn). Results are cached per ``(n_sections, acceleration)`` config.

``test_multiturn_wake_accumulates_over_turns``
    The multi-pass wake genuinely builds up turn over turn (peaks ~1, 1.9, 2.8)
    and the first turn matches the feedback to single-pass accuracy.
``test_multiturn_feedback_propagation_matches_convolution``
    Coarse-grid propagation matches the convolution on every turn (regression
    for the dropped downsample remainder, single section, static cycle).
``test_multiturn_multiple_sections``
    Holds for multi-section rings (2, 3, 10 RF stations per turn), exercising
    the reverse/forward reference tracking across stations.
``test_multiturn_with_acceleration``
    Holds under acceleration (``MagneticCyclePerTurnAllRFStations``), where
    ``t_rev``, the carrier frequency and the reverse-tracking frame slip vary
    turn over turn.
``test_multiturn_substepped_matches_convolution``
    Beam loading computed on a sub-stepped coarse grid
    (``n_rf_periods_per_coarse_grid < 1``) stays correct on a static cycle.
``test_multiturn_fast_ramp``
    Single section on the fast (transition-adjacent) ramp still matches the
    retuning convolution in the fast frame-slip regime.
``test_multiturn_fast_ramp_multisection``
    Multi-section (2 and 4 stations) on the fast ramp matches the retuning
    convolution: the ``_track`` frame correction removes the carried-envelope
    phase error ``sum_k (omega_k - omega_0) T_seg,k`` from the other stations'
    mid-turn grid re-seeding (drift ~0.023 t_rf/turn -> ~0.2 %).
``test_multiturn_fast_ramp_substepped``
    Sub-stepped (n = 0.5) carried wake holds on the fast ramp: the stale
    reverse-segment re-pass is removed (it corrupted the demodulation frame
    by ``-(turn+1) * 2 pi S`` per turn for single-section rings) and the
    sub-stepped demodulation frame is the tiling boundary gap (a pure time,
    immune to the float-bistable residual landing flip). ~0.1 %, was ~40 %.
``test_multiturn_fast_ramp_multisection_substepped``
    The full combination (2 sections, fast ramp, n = 0.5) passes: the
    tiling-gap demodulation frame also covers the multi-section
    reverse-to-forward handover.
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
    The large offset also holds with two RF stations: reverse-tracked
    segments, per-station kick clocks and the multi-section frame
    correction stay consistent with the carrier anchoring. All four
    offset tests are mutation-verified (flipping the anchor sign fails
    every one).
``test_multiturn_secular_drift_long_horizon``
    Long-horizon guard for the shorter consistency tests: the most drift-prone
    case (2 sections, fast undriven) run for 20 turns has a bounded per-turn
    relative-error slope (~0.03 pp/turn) and an endpoint within 1 %.
``test_multiturn_nondivisible_harmonic`` (``@expectedFailure``)
    KNOWN LIMITATION. A harmonic not divisible by ``2 * n_sections``
    de-aligns the coarse-grid tiling from the profile's zeroed leading edge, so
    beam charge is downsampled into the first coarse cell and ``rf_beam_current``
    raises before any voltage is produced -- a genuine gap versus the
    geometry-agnostic solver. The other multi-section tests reduce the harmonic
    to a multiple of ``2 * n_sections`` to avoid this.
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

An opt-in debug plot (``_plot_energy_kick``) writes ``energy_kick_over_time.png``
(see `Data and assets`_).


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
    Drives the *real* mucol coarse downsampling
    (``rf_beam_current_partial``, the function the forward pass calls and which
    hard-enforces ``forbid_charge_in_first_coarse_cell``) and asserts it returns
    without raising and the first coarse cell carries negligible charge -- the
    actual invariant, not a re-read of the builder's zeroed fine bins.

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
split by the station azimuth:

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

**Class** ``TestSimultaneousPassageGuard`` -- a station at a meeting azimuth
(e.g. the single mid-ring station of a one-section layout) sees both beams
at the *same* reference time. The per-passage machinery would silently
serialize the coincident arrivals one full projection window apart (envelope
at twice the physical rate; measured ~47 % L2 waveform error on the first
turn), so the feedback detects the coincident opposite-direction passage
(within half a coarse cell) and refuses it.

``test_single_section_two_beam_raises``
    The coincident second passage raises ``NotImplementedError``. The message
    points to the supported fix -- move the station off the meeting azimuth.
    It also mentions the ``MultiPassResonatorSolver`` wakefield
    (``allow_delta_t_zero=True``), but note the caveat in the design doc's
    *Counter-rotating beams* warning: that path deposits each coincident
    beam's kick before the other beam's profile is registered, so it gives an
    order-asymmetric kick and is *not* a correct model for a meeting-azimuth
    station.
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


``test_envelope_kernel.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class** ``TestEnvelopeKernelBitIdentity`` -- the numba coarse-envelope
kernel (``envelope_pi_scan``) must reproduce the pure-Python coarse recursion
bit-for-bit. Each test drives both paths with identical inputs and asserts
equality across the regimes the kernel must cover: no beam / constant
current, a forward pass, the inline PI controller (no delay, with delay,
saturating), the exponential-propagator branch, detuning, and the
carried-state / multi-section reverse-then-forward cases.


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

**Class** ``TestBeamLoadingCompensationSustainsSetpoint``
    The compensation ``I_gen = I_ff + I_beam / 2`` is a fixed point of the
    coarse step for every detuning; dropping the factor 2 breaks the fixed
    point (mutation guard).
**Class** ``TestGeneratorPowerBeamPowerBalance``
    Generator power versus beam power balances to one on resonance; the raw
    (missing-half) beam current gives the factor-2 shortfall; with detuning
    only the reactive term is carried.
**Class** ``TestPowerCurrentRoundTrip``
    ``current_limit_from_power`` and its inverse round-trip
    (power -> current -> power and current -> power -> current are
    identities).


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
    Saved output of the opt-in debug plot in
    ``test_energy_gain_ind_voltage_vs_nondriven_feedback.py`` -- the applied
    energy kick versus arrival time for the wake and feedback paths.


Running the tests
-----------------

From the ``BLonD`` project root, run the whole suite with ``pytest``:

.. code-block:: bash

   pytest tests/unittests/physics/feedbacks/accelerators/mucol/

or a single module / test, for example:

.. code-block:: bash

   pytest tests/unittests/physics/feedbacks/accelerators/mucol/test_helpers.py
   pytest "tests/unittests/physics/feedbacks/accelerators/mucol/test_mucol_cav_fdbk.py::TestFineGridResonatorBenchmark"

The debug plots are opt-in via the ``DEBUG_PLOT`` module constant (and
``PLOT_DIAGNOSTICS`` in ``test_generator_current_pi_feedback.py``); both default
to off in every module, so nothing opens in a headless/CI run.
