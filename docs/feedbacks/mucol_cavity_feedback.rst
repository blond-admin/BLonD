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

:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedback`
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

:mod:`blond.physics.feedbacks.helpers`
    The numeric building blocks: the cavity-envelope solvers
    (:func:`~blond.physics.feedbacks.helpers.cavity_response_sparse_matrix`,
    first-order forward Euler, and
    :func:`~blond.physics.feedbacks.helpers.cavity_response_sparse_matrix_second_order`,
    trapezoidal / Crank-Nicolson), the beam-current demodulation
    :func:`~blond.physics.feedbacks.helpers.rf_beam_current` and the
    feedforward fill seed
    :func:`~blond.physics.feedbacks.helpers.pretrack_fill_voltage`.


Signal path of one turn
-----------------------

Each turn the timing class performs, in order:

1. **Coarse-grid construction** (``rf_centers``). The feedback tracks a copy
   of the beam reference forward to the next RF station and, on later turns,
   re-derives the segments that elapsed since its last update (reverse
   tracking). Each segment carries the RF frequency it was tracked with
   (design frequency at the local reference energy plus the station's
   frequency offset ``delta_omega_rf``), so the coarse-step spacing follows
   the detuned RF period even under acceleration and with several stations
   per ring.

2. **Beam-current demodulation.**
   :func:`~blond.physics.feedbacks.helpers.rf_beam_current` converts the beam
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
   beam kick is large relative to the antenna voltage.

4. **Optional generator-current control.** With a ``controller`` attached,
   each coarse step forms the error ``V_set - V_ant[n]`` and lets the
   controller produce ``I_gen[n]``, which drives the next step; without one,
   the generator current stays at the constant feedforward value
   ``generator_current_bias``. The klystron limit is enforced on the fine
   grid as well before the response solve.

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
    The coarse grid follows it, and the station integrates the resulting RF
    phase slip exactly from the elapsed reference time
    (``delta_omega_rf * dt``, accumulated at the end of each station track).
    Guards on the station enforce the supported use: in a ring with more than
    one RF station the offset cannot be changed during the run, and the
    slip bookkeeping only runs when a beam feedback (phase loop) exists in
    the simulation or the offset is nonzero.

For low loaded quality factors the per-RF-period Euler step can violate the
step-size limits; the sub-stepping mode
(``n_rf_periods_per_coarse_grid < 1``) subdivides the RF period, with the
coarse centres tiling continuously across turn boundaries.


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
* the shared helpers against the blond2 reference implementations (LHC
  comparison suite).


Known limitations
-----------------

* ``delta_omega_rf != 0`` combined with multi-section rings is guarded
  against (the station raises on changes) but per-segment application of a
  static offset in the reverse tracking is not implemented; the reverse
  segments currently use the phase at the current passage.
* The coarse re-binning of the beam current assumes the analytic uniform
  grid; configurations far from the tested ones (unusual profile placement)
  should be validated against the wake solvers. Sub-stepped beam loading
  itself is validated against the convolution on a static cycle.
* On the fast (transition-adjacent) ramp, the *multi-section* and the
  *sub-stepped* carried wake drift in arrival time; these combinations are
  marked as expected failures in the multi-turn comparison suite and should
  not be relied on yet (single-section standard-grid fast ramp is validated).
* The fine-grid initial antenna voltage is taken from the first coarse cell
  of the forward segment (guarded by the first-cell charge check) rather
  than interpolated to the profile edge.
