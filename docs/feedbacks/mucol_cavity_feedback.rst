.. _mucol_cavity_feedback_overview:

Muon-Collider Cavity Feedback -- Overview
=========================================

This page gives an architectural overview of the RF cavity-feedback model in
``blond.physics.feedbacks``, developed for the muon-collider Rapid-Cycling
Synchrotron (RCS) studies. It describes what the classes do and how the pieces
fit together; the API reference is generated from the docstrings (see
:mod:`blond.physics.feedbacks.cavity_feedback`), and the behaviour is certified
by the test suite documented in :ref:`mucol_cavity_feedback_tests`.

.. contents:: Contents
   :local:
   :depth: 2


Classes at a glance
-------------------

:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackBase`
    Abstract base for IQ-envelope cavity feedbacks. Owns the profile, the
    coarse/fine grid arrays, the beam-current demodulation and the parent-RF-
    station accessors (``omega_rf``, ``phi_rf``, ``delta_omega_rf``, ...). The
    LHC/SPS loops under ``blond.experimental`` subclass it as well.

:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
    The muon-collider cavity model. Tracks the antenna voltage of one RF
    station's cavities on a coarse time grid that follows the *actual* RF
    clock turn by turn (including acceleration and multiple stations per
    ring), and resolves the voltage seen by the bunch on the fine (profile)
    grid.

:class:`~blond.physics.feedbacks.generator_current_controller.GeneratorCurrentPIController`
    Standalone, saturating PI controller mapping an antenna-voltage error to a
    generator-current command: per-sample delay line (loop delay), conditional
    anti-windup integration, and a phase-preserving magnitude clamp (klystron
    current limit, convertible from a power limit via
    :func:`~blond.physics.feedbacks.generator_current_controller.current_limit_from_power`).
    It is pure signal processing -- no cavity, profile or station -- and is
    attached to the feedback via the ``controller`` argument.

:mod:`blond.physics.feedbacks.rf_center_grid`
    The coarse-grid construction: ``RFCenterGridMixin``, the forward/reverse
    reference walks and per-turn segment generation of the timing class.

:mod:`blond.physics.feedbacks.rf_center_segment`
    The ``RFCenterSegment`` value class the coarse grid is built from.

:mod:`blond.physics.feedbacks.generator_regulation`
    ``GeneratorRegulationMixin``: the controller-facing pieces of the timing
    class (per-cavity IQ setpoint, klystron power, per-step generator-current
    update).

:mod:`blond.physics.feedbacks.beam_current`
    The beam-current demodulation
    (:func:`~blond.physics.feedbacks.beam_current.rf_beam_current` -- shared
    with the LHC comparison path and kept byte-identical for co-rotating
    beams -- and
    :func:`~blond.physics.feedbacks.beam_current.rf_beam_current_partial`,
    the forward-pass variant the timing class uses).

:mod:`blond.physics.feedbacks.cavity_solvers`
    The muon-collider-only numerics:
    :func:`~blond.physics.feedbacks.cavity_solvers.cavity_response_sparse_matrix_second_order`
    (trapezoidal / Crank-Nicolson) and the feedforward fill seed
    :func:`~blond.physics.feedbacks.cavity_solvers.pretrack_fill_voltage`.

:mod:`blond.physics.feedbacks.helpers`
    The first-order (forward-Euler) fine-grid solver
    :func:`~blond.physics.feedbacks.helpers.cavity_response_sparse_matrix`,
    shared with the (experimental) LHC feedback, plus backward-compatible
    re-exports of the beam-current and IQ helpers.

:mod:`blond.physics.feedbacks.iq`
    IQ / polar conversions (``cartesian_to_polar``, ``polar_to_cartesian``).


Signal path of one turn
-----------------------

Each turn the timing class performs, in order:

1. **Coarse-grid construction** (``rf_centers``). The feedback tracks a copy
   of the beam reference forward to the next RF station and, on later turns,
   re-derives the segments that elapsed since its last update (reverse
   tracking). Each segment carries the *design* RF frequency it was tracked
   with (at the local reference energy), so the coarse-step spacing follows
   the design RF period even under acceleration and with several stations
   per ring. A station RF-frequency offset ``delta_omega_rf`` never moves
   the grid: it enters only as explicit phases (the demodulation carrier
   and the accumulated kick-clock slip, see below).

2. **Beam-current demodulation.**
   :func:`~blond.physics.feedbacks.beam_current.rf_beam_current` converts the beam
   profile into the complex IQ beam-current envelope at the carrier frequency
   (factor-2 single-sideband demodulation), applies the reference-frame phase
   correction, and re-bins the fine-grid charge onto the coarse cells
   charge-conservingly. A guard forbids charge in the first coarse cell,
   whose value seeds the fine-grid initial condition (double counting), and a
   warning fires if the profile window does not capture the whole beam.

3. **Coarse-grid cavity update.** The antenna voltage is advanced cell by
   cell with the forward-Euler discretisation of the cavity-envelope ODE:
   generator drive ``I_gen (R/Q) omega dt``, decay/detuning multiplier
   ``1 - 0.5 omega dt / Q_L + i delta_omega dt`` and beam loading
   ``-0.5 I_beam (R/Q) omega dt``. Discretisation validity is enforced:
   ``_check_step_sizes`` warns above a per-step decay/rotation of 0.1 and
   raises above 2.0, and an analogous check warns/raises when the per-step
   beam kick is large relative to the antenna voltage. With
   ``exponential_coarse_solver=True`` the exact exponential propagator
   ``V[n+1] = e^L V[n] + src (e^L - 1)/L`` replaces the Euler step: it is
   exact in decay and detuning rotation (a pure detuning becomes a pure
   rotation instead of growing ``|V|`` by ``sqrt(1 + (delta_omega dt)^2)``
   per step) and is the accurate alternative to sub-stepping at low ``Q_L``
   or large detuning.

4. **Optional generator-current control.** With a ``controller`` attached,
   each coarse step forms the error ``V_set - V_ant[n]`` and lets the
   controller produce ``I_gen[n]``, which drives the next step; without one,
   the generator current stays at the constant feedforward value
   ``generator_current_bias``. The controller is stepped only on the real
   forward passage, never on the reverse reconstruction segments (those
   carry a per-segment frame phase, so stepping there would integrate
   frame-rotated errors and double-advance the delay line and integrator).
   The klystron limit is enforced on the fine grid as well before the
   response solve.

5. **Fine-grid solve.** The generator current is interpolated onto the
   profile grid and the cavity response is solved as a sparse bidiagonal
   system -- first order by default, or the second-order (Crank-Nicolson)
   solver with ``second_order=True``, whose truncation error scales with the
   bin size squared. The result, scaled by ``n_cavities``, yields the
   voltage correction and phase correction the parent RF station applies to
   its kick.


Initial conditions and cavity pre-fill
--------------------------------------

By default the coarse grid starts from the scalar ``initial_voltage``. With
``n_pretrack`` set, the initial antenna voltage is instead seeded from the
closed-form feedforward fill of the cavity,
``V(t) = V_ss (1 - exp(lambda t))`` with
``lambda = -omega/(2 Q_L) + i delta_omega`` -- evaluated after ``n_pretrack``
turns, or, with ``injection_voltage`` given, at the moment ``|V(t)|`` first
reaches that target (beam injected part-way through the fill). The fill is
feedforward-only by design: a controller, if attached, regulates from the
first tracked turn after injection. On resonance the steady state reduces to
``V_ss = 2 (R/Q) Q_L I_gen``, which is also the exact fixed point of the
coarse-grid Euler step.


Interplay with the RF station
-----------------------------

Two distinct frequency knobs exist and must not be confused:

``delta_omega`` (feedback constructor)
    The *cavity resonance* detuning [rad/s]. Enters the cavity response as a
    per-step phase rotation; it does not move the coarse grid.

``delta_omega_rf`` (RF station attribute)
    The station's *RF frequency* offset added on top of the design frequency.
    The station integrates the resulting RF phase slip exactly from the
    elapsed reference time (``delta_omega_rf * dt``, accumulated at the end
    of each station track). The timing class demodulates the beam current
    onto the actual RF carrier -- the offset carrier within the profile
    window plus the accumulated slip (the station's kick clock
    ``delta_phi_rf`` and its live end-of-track tail) -- and the readout
    applies the identical total (the clock via ``phi_rf``, the tail via
    ``phase_correction``), so the demodulation/readout chain closes exactly
    for every carried deposit; the coarse grid itself stays on the design
    clock. Validated against the retuning convolution at the discretization
    floor (``test_multiturn_delta_omega_rf_*``: large offset, differential,
    sub-stepped, multi-section).
    Guards on the station enforce the supported use: in a ring with more than
    one RF station the offset cannot be changed during the run, and the
    slip bookkeeping only runs when a beam feedback (phase loop) exists in
    the simulation or the offset is nonzero.

For low loaded quality factors the per-RF-period Euler step can violate the
step-size limits; the sub-stepping mode
(``n_rf_periods_per_coarse_grid < 1``) subdivides the RF period, with the
coarse centres tiling continuously across turn boundaries.


Counter-rotating beams
----------------------

The collider ring accelerates a co-rotating mu+ and a counter-rotating mu-
beam through the same cavities. The whole beam-loading chain (RF beam
current, wake-solver sources, and every kick) uses the *direction-signed
charge* ``beam.signed_charge_with_direction()`` (charge negated for a
counter-rotating beam): the collider pair then carries same-sign gap
currents, so for an asymmetric fundamental mode the loading of both beams
adds constructively and both receive the same kick. A counter-rotating mu- beam alone reproduces the
co-rotating mu+ run bit-for-bit, through the feedback and through the
convolution reference alike.

With two simultaneous beams (``MainloopCounterRotatingBeams``: each station
is tracked once per beam per turn, the counter-rotating beam traversing the
elements in reverse order), the supported regime is *offset passages* --
stations away from the beams' meeting azimuths, e.g. any even section count
with the half-drift / station / half-drift layout, where the two arrivals
are ``T_rev / 2`` apart. There the per-passage grid machinery handles the
alternating arrivals natively and matches the two-beam convolution at
reference accuracy. A station *at* a meeting azimuth (both beams at the
same reference time, e.g. the single mid-ring station of a one-section
layout) is refused with ``NotImplementedError``: the machinery would
silently serialize the coincident arrivals one projection window apart.
Model such a station's loading with the ``MultiPassResonatorSolver``
wakefield (``allow_delta_t_zero=True``) instead.

For the wake-solver references, ``shunt_impedances_counter_witness`` is the
shunt a counter-rotating witness *experiences* (its reversed integration
direction included). The sign is a property of the mode's field symmetry,
not of fundamental modes in general: an *asymmetric* fundamental mode has
``R_CR = -R`` (opposite charges add up and receive the same kick), while
``R_CR = +R`` makes same-charge counter-rotating beams add up.


Validation
----------

The model is certified against independent references rather than against
itself (see :ref:`mucol_cavity_feedback_tests` for the full inventory):

* single-turn beam loading against a ``Resonators`` convolution (< 1 % NRMSE,
  on and off resonance);
* multi-turn wake build-up against the ``MultiPassResonatorSolver``, per turn
  and per section, for multi-station rings and under acceleration;
* the carried-wake phase under acceleration against an analytic multipass sum
  with the accumulated phase :math:`\int \omega \, dt`;
* the applied particle energy gain against the wake-kick path in a full
  simulation;
* the self-consistent multi-turn bunch *dynamics* (centroid, bunch length,
  emittance) against a twin simulation whose only difference is the
  induced-voltage model (wake vs feedback), under strong beam loading on the
  fast ramp;
* a counter-rotating mu- beam against the co-rotating mu+ run (bit-for-bit)
  and the two-beam offset-passage operation against the two-beam multi-pass
  convolution, per station and turn;
* the charge-pair x counter-rotating-shunt matrix (build-up vs cancellation,
  closed form on the ringing tail) on both the convolution and the
  pole-residue solver, which agree cell by cell to ~1e-13 -- this one lives
  in the impedance-solver suite
  (``tests/unittests/physics/impedances/test_solvers.py``), not the mucol
  inventory below;
* the shared helpers against the blond2 reference implementations (LHC
  comparison suite).


Known limitations
-----------------

* A harmonic number that is not divisible by ``2 * n_sections`` de-aligns
  the coarse-grid tiling from the profile's charge-free leading edge, so
  beam charge lands in the first coarse cell and ``rf_beam_current`` raises
  before any voltage is produced (marked as an expected failure in the
  multi-turn comparison suite).
* In a ring with more than one RF station the ``delta_omega_rf`` offset
  cannot be changed during the run (the station raises). The former
  lab-frame demodulation slip under an offset (an error growing with the
  absolute reference time) is fixed: the demodulation carrier is anchored
  to the accumulated actual RF phase and validated at the discretization
  floor for offsets beyond the cavity half-bandwidth
  (``test_multiturn_delta_omega_rf_*``).
* Driven (generator-bias) multi-section fast-ramp operation carries a
  bounded frame slip between the constant-phase bias and the slipping
  segment frame (percent-level ``|V_ant|`` drift over a few turns); it
  cancels in the linear beam-induced part but is visible in absolute
  antenna-voltage trajectories.
* The undriven two-section fast-ramp carried wake shows a slow bounded
  secular drift (~0.03 percentage points per turn over 20 turns) against
  the convolution.
* Two counter-rotating beams passing a station *simultaneously* (station at
  a meeting azimuth) are refused rather than integrated; see
  *Counter-rotating beams* above for the guard and the workaround.
* The coarse re-binning of the beam current assumes the analytic uniform
  grid; configurations far from the tested ones (unusual profile placement)
  should be validated against the wake solvers. Sub-stepped beam loading
  itself is validated against the convolution, including with detuning and
  on the fast ramp.
* The fine-grid initial antenna voltage is taken from the first coarse cell
  of the forward segment (guarded by the first-cell charge check) rather
  than interpolated to the profile edge.
