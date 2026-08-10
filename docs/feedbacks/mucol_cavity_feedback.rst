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


Concepts and notation
---------------------

This is a *low-level RF* (LLRF) feedback: it keeps a cavity's accelerating
voltage on target while the beam itself perturbs that voltage. The terms
below recur throughout the page -- a reader new to cavity feedback should
skim them first; the later sections assume them.

**Physical quantities**

RF cavity
    A resonant metal structure whose oscillating electromagnetic field gives
    each passing bunch an energy kick.
antenna voltage (``V_ant``)
    The complex envelope of the cavity's accelerating voltage -- the quantity
    the feedback tracks and the controller regulates. It is *distinct* from
    the voltage an individual bunch sees (resolved separately on the fine
    grid): the bunch samples the field only during its short passage.
generator current (``I_gen``)
    The RF drive the amplifier (klystron) feeds into the cavity -- the
    actuator the feedback commands.
beam current (``I_beam``) / gap current
    The RF-frequency component of the beam's charge as it crosses the cavity
    gap. A passing bunch acts as a current source that *removes* energy from
    the cavity field.
beam loading
    The change in cavity voltage caused by that beam current. Left
    uncompensated it shifts the voltage every bunch sees; cancelling it is
    the feedback's main job.
kick
    The energy (and phase) change the cavity imparts to the beam on one
    passage -- the model's ultimate output, applied by the parent RF station.
``R/Q``, ``Q_L``
    Cavity figures of merit. ``R/Q`` (shunt-impedance-over-Q, [Ohm]) sets how
    strongly a current drives the voltage; ``Q_L`` is the *loaded* quality
    factor, setting how slowly the field decays (time constant
    ``~ 2 Q_L / omega``).
detuning (``delta_omega``)
    An offset of the cavity's *resonant* frequency from the RF frequency;
    it makes the complex voltage *rotate* as it decays.
(beam) profile
    The histogram of a bunch's charge versus time -- the feedback's input,
    and the grid on which the bunch-seen voltage is resolved.
wake / convolution reference
    An independent way to compute beam-induced voltage: convolve the beam
    profile with the cavity's impulse response (its *wake*). The tests check
    the feedback against this and other independent models.

**The complex-envelope (IQ) picture.** Everything oscillates at the RF
carrier ``omega_rf``. Instead of tracking the fast oscillation, the model
*demodulates* every signal down to its slowly-varying complex amplitude --
the **IQ envelope** (in-phase ``+ i`` quadrature). "Demodulating the beam
current onto the carrier" means projecting the profile onto
``cos(omega_rf t)`` and ``sin(omega_rf t)`` to recover that complex
amplitude. Every voltage and current on this page is such an envelope.

**Two grids.** The cavity field evolves over a whole turn, so it is stepped
on a sparse **coarse grid** (one point per RF period, or a fraction of one)
that spans the turn cheaply -- this is where the feedback loop lives. The
bunch, by contrast, samples the field over picoseconds, so the voltage it
actually receives is resolved on the dense **fine grid** (the profile grid),
onto which the coarse-grid result is interpolated.

**The generator control loop**

PI controller
    Proportional-Integral controller: commands ``I_gen`` from the voltage
    error ``V_set - V_ant`` (a term proportional to the error plus a term
    integrating it).
anti-windup
    Freezes the integrator while the actuator is saturated, so the integral
    does not "wind up" to an unrecoverable value.
klystron limit
    The largest generator current (or power) the amplifier can deliver; the
    command is clamped to it, keeping its phase.
feedforward fill (pre-fill)
    Charging the cavity to its operating voltage *before* the beam arrives,
    with a fixed generator current and no feedback.

**Reference frames / clocks.** Several distinct time-and-phase references
appear:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name
     - What it is
   * - design clock
     - RF phase at the *design* frequency; the coarse-grid geometry is built
       on it.
   * - actual RF clock
     - Design frequency *plus* any station offset ``delta_omega_rf``.
   * - kick clock (``delta_phi_rf``)
     - The station's accumulated RF phase slip from ``delta_omega_rf``,
       applied to the kick.
   * - segment frame
     - The phase reference of one reconstructed coarse-grid segment (below).

**Coarse-grid construction primitives**

beam reference
    A synchronous-particle clock (time + energy); a copy of the beam's
    reference coordinates the feedback advances to place its grid points.
forward / reverse tracking
    To build the grid the feedback advances the reference *forward* to the
    next RF station; on later turns it re-derives the stretch of grid that
    has *elapsed* since its previous update by walking it in *reverse*.
segment
    One contiguous piece of coarse grid produced by one such walk, at a
    single tracked frequency. Every segment holds at least two coarse
    centres -- ``RFCenterSegment`` rejects a shorter (degenerate) one at
    construction, because the coincidence-guard cell width and the
    residual bookkeeping below are only well-defined with two.
residual
    The unfilled tail of a segment: the time between its last coarse
    centre and the segment's end. Coarse centre times are
    segment-*local*, so the coarse step into the first cell of a segment
    is that cell's local time plus the *preceding* segment's residual.
    It is read back from the segment list
    (``_preceding_segment_residual``), not from the live accumulator,
    which by the time the grid is walked already holds the last-generated
    (forward) segment's value; the first segment of a turn steps across
    the turn boundary and takes the residual the previous turn ended on.
    The same quantity is the demodulation frame of the forward segment --
    under ``validate_grid_each_turn`` an assertion ties the two together
    so they cannot silently drift apart.
carried deposit
    Beam-induced voltage laid onto the grid on one turn that must then be
    propagated ("carried") consistently across later turns and segments.


Classes at a glance
-------------------

:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackBase`
    Slim abstract base for IQ-envelope cavity feedbacks: constructor
    validation, the coarse/fine grid arrays, and the parent-RF-station
    accessors (``omega_rf``, ``phi_rf``, ``delta_omega_rf``, ...). Its
    concrete subclass is the muon-collider timing class below, which owns
    the beam-current demodulation and all tracking.

:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
    The muon-collider cavity model. Tracks the antenna voltage of one RF
    station's cavities on a coarse time grid whose geometry follows the
    *design* RF clock turn by turn (including acceleration and multiple
    stations per ring; a station RF-frequency offset ``delta_omega_rf``
    enters only as a phase, not the grid), and resolves the voltage seen by
    the bunch on the fine (profile) grid.

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
    The two coarse-grid value classes. ``RFCenterSegment`` is what the
    grid is built from: the segment's frequency, duration, centre times
    and its ``residual`` -- the unfilled tail between its last centre and
    its end. The ``residual`` is read back by
    ``_preceding_segment_residual`` to form the coarse step into the
    *following* segment's first cell, and is the demodulation frame of the
    forward segment; the ``omega`` and ``duration`` are what the
    reverse-span replay and the registration phase walk.
    ``PerTurnGridSpan`` (below) is the per-turn span built out of those
    segments.

:class:`~blond.physics.feedbacks.rf_center_segment.PerTurnGridSpan`
    Frozen value class returned by one grid rebuild: this passage's
    reverse and forward centre counts plus ``residual_from_reverse_span``,
    the residual snapshotted *between* the reverse and the forward
    generation. Returning it rather than leaving it on the feedback is
    what makes the per-turn phase ordering enforceable by the data flow --
    the demodulation frame can only be read from a span object, and a span
    is only produced by a rebuild that snapshotted it in time.

:mod:`blond.physics.feedbacks.generator_regulation`
    ``GeneratorRegulationMixin``: the controller-facing pieces of the timing
    class (per-cavity IQ setpoint, klystron power, per-step generator-current
    update).

:mod:`blond.physics.feedbacks.beam_current`
    The beam-current demodulation: the single function
    :func:`~blond.physics.feedbacks.beam_current.rf_beam_current` (fine-grid
    demodulation, optionally re-binned onto the coarse grid when
    ``sampling_time``/``n_points`` are given).

:mod:`blond.physics.feedbacks.cavity_solvers`
    The muon-collider-only numerics: the first-order (forward-Euler)
    fine-grid solver
    :func:`~blond.physics.feedbacks.cavity_solvers.cavity_response_sparse_matrix`,
    its second-order twin
    :func:`~blond.physics.feedbacks.cavity_solvers.cavity_response_sparse_matrix_second_order`
    (trapezoidal / Crank-Nicolson) and the feedforward fill seed
    :func:`~blond.physics.feedbacks.cavity_solvers.pretrack_fill_voltage`.
    It also holds the coarse-grid step arithmetic shared by the per-cell and
    vectorised recursions --
    :func:`~blond.physics.feedbacks.cavity_solvers.coarse_step_exponent`,
    :func:`~blond.physics.feedbacks.cavity_solvers.euler_voltage_multiplier`,
    :func:`~blond.physics.feedbacks.cavity_solvers.exponential_voltage_multiplier`
    and
    :func:`~blond.physics.feedbacks.cavity_solvers.exponential_drive_weight`
    -- and
    :class:`~blond.physics.feedbacks.cavity_solvers.ForwardEulerValidityGuard`,
    the tripwires that decide whether the forward-Euler discretisation is
    admissible at all (per-step decay, detuning phase and beam kick) -- pure
    numerics kept beside the solvers they certify. The feedback owns one
    instance and passes the cavity parameters per call; it is constructed
    disabled for the exact exponential propagator, which is subject to none
    of these caps.

:mod:`blond.physics.feedbacks.envelope_kernel`
    The compiled numba host kernel (``envelope_pi_scan``) the coarse
    per-cell recursion runs on by default
    (``use_numba_envelope_kernel``). It is byte-identical to the
    pure-Python per-cell reference, which is kept both as that reference
    and as the exact fallback: a segment is re-run there when any cell
    reaches the klystron limit (whose numpy magnitude clamp the kernel
    cannot reproduce bit-for-bit) or when two coarse points coincide
    (zero step), and a controller that supplies no compiled form of
    itself (``supports_envelope_scan``) is driven cell by cell instead.
    Set the flag ``False`` on an instance to force the reference path.

:mod:`blond.physics.feedbacks.iq`
    IQ / polar conversions (``cartesian_to_polar``, ``polar_to_cartesian``).


Signal path of one turn
-----------------------

``_track`` is a pure call-order declaration: it does no work itself, it
only names the phases below in order and hands what one phase produced to
the phase that needs it, so the argument lists *are* the dependency graph.
Each turn the timing class runs:

1. ``_guard_simultaneous_passage`` -- refuses a coincident
   counter-rotating passage with ``NotImplementedError`` (station at a
   meeting azimuth; the tolerance is half the last forward coarse-cell
   width, a guaranteed-positive genuine width because every segment
   holds at least two centres), then records this passage's arrival time
   and direction as the
   record the next passage compares itself against. It runs first so that
   a refused passage cannot leave a half-rebuilt grid behind.

2. ``_carrier_slip_gap_at_passage`` -- returns the *live tail* of the
   RF-frequency-offset phase slip,
   ``delta_omega_rf * (t_passage - last kick-clock tick)``. The station
   accumulates its kick clock ``delta_phi_rf`` only at the end of each
   track, so this gap completes it to the exact accumulated slip at this
   passage. Exactly ``0.0`` without an offset. It is *returned* and then
   assigned to ``_carrier_slip_gap``, which makes visible that the value
   is reset at every passage rather than accumulated.

3. ``_rebuild_per_turn_grid`` -- rebuilds this passage's coarse grid
   (``rf_centers``), sizes the coarse state and returns a frozen
   :class:`~blond.physics.feedbacks.rf_center_segment.PerTurnGridSpan`. It
   first calls ``_close_previous_turn_grid``, which captures the previous
   turn's last centre and its end-of-turn residual
   (``_residual_time_carried_into_turn``) *before* clearing the segment
   list, and then generates this passage's segments: the feedback tracks a
   copy of the beam reference forward to the next RF station and, on later
   turns, re-derives the segments that elapsed since its last update
   (reverse tracking). Each segment carries the *design* RF frequency it
   was tracked with (at the local reference energy), so the coarse-step
   spacing follows the design RF period even under acceleration and with
   several stations per ring. A station RF-frequency offset
   ``delta_omega_rf`` never moves the grid, and does not shift the
   demodulation carrier either (which stays on the design clock): it
   enters only as the explicit constant phase of step 2. ``reset_arrays``
   is the last statement of this phase -- it can neither precede the grid
   generation it takes its size from, nor follow any ``circuit_track``.

4. ``_replay_reverse_span`` -- re-walks this passage's reverse segments
   with ``no_beam=True``, one ``circuit_track`` per reverse segment at
   that segment's own ``omega``, so that the envelope carries the
   already-elapsed interval forward. A passage that generated no reverse
   segments skips the replay entirely.

5. ``_accumulate_registration_phase`` -- accumulates the multi-section
   grid-vs-carrier registration phase
   ``Psi = sum_k (omega_k - omega_0) T_seg,k`` (explained under *Interplay
   with the RF station* below) and returns the running total, which is
   added to ``_carrier_slip_gap``. Exactly ``+0.0`` for a single section
   and for an unaccelerated ring, so both stay bit-identical.

6. ``_write_no_correction_readout`` -- only with
   ``grid_only_no_correction=True``: writes the neutral readout (unit
   relative voltage, zero phase, i.e. **no correction at all**) and ends
   the turn there, so neither the demodulation nor the forward pass
   runs. The three diagnostic switches are independent: ``debug`` only
   records the inspection-only grid snapshots,
   ``validate_grid_each_turn`` only runs the per-turn grid integrity
   check, and only this one stops the physics.

7. ``_track_forward_span`` -- the real work of the turn, in two steps.

   *Demodulation*: ``calculate_rf_beam_current_partial`` calls
   :func:`~blond.physics.feedbacks.beam_current.rf_beam_current` to
   convert the beam profile into the complex IQ beam-current envelope at
   the *design* carrier (factor-2 single-sideband demodulation), rotate it
   by the reference-frame phase and by the constant
   ``-(delta_phi_rf + _carrier_slip_gap)``, and re-bin the fine-grid
   charge onto the coarse cells charge-conservingly. The demodulation
   frame is the span's ``residual_from_reverse_span``, snapshotted before
   the forward generation overwrote the host scalar; re-reading that
   scalar here would silently shift the frame. Several guards protect this
   path: charge in the first coarse cell raises (that cell seeds the
   fine-grid initial condition, so its kick would be double-counted), a
   profile window longer than the coarse grid it is re-binned onto raises
   through ``ProfileBaseClass.check_fits_in_span``, a window mapping past
   the last coarse cell raises, and a warning fires if the profile window
   does not capture the whole beam.

   *Forward pass*: one ``circuit_track`` over the forward segment, which
   performs the coarse-grid cavity update, the optional generator control
   and the fine-grid solve described below.

8. ``_write_station_readout`` -- converts the fine-grid antenna voltage
   into ``relative_voltage_correction`` (divided by the station voltage)
   and ``phase_correction`` (referenced to the mean phase of
   ``station_voltage_coarse_grid``, plus the very same
   ``_carrier_slip_gap`` the demodulation subtracted, so that the
   demodulation/readout chain closes). These two are what the parent RF
   station applies to its kick.

**Coarse-grid cavity update** (inside ``circuit_track``). The antenna
voltage is advanced cell by cell with the forward-Euler discretisation of
the cavity-envelope ODE: generator drive ``I_gen (R/Q) omega dt``,
decay/detuning multiplier ``1 - 0.5 omega dt / Q_L + i delta_omega dt``
and beam loading ``-0.5 I_beam (R/Q) omega dt``. Discretisation validity
is enforced by
:class:`~blond.physics.feedbacks.cavity_solvers.ForwardEulerValidityGuard`
(the timing class's ``_check_step_sizes``, ``_check_beam_kicks`` and
``_check_beam_kick_magnitude`` only supply it the cavity's current
parameters): it warns above a per-step decay/rotation of 0.1 and raises
above 1.0 -- there the Euler decay factor ``1 - 0.5 omega dt / Q_L`` turns
negative and the discretised voltage flips sign every step, which the
exact (always positive) decay never does; use
``exponential_coarse_solver_enable=True`` for larger steps. An analogous
check warns/raises when the per-step beam kick is large relative to the
antenna voltage. With ``exponential_coarse_solver_enable=True`` the exact
exponential propagator ``V[n+1] = e^L V[n] + src (e^L - 1)/L`` replaces
the Euler step: it is exact in decay and detuning rotation (a pure
detuning becomes a pure rotation instead of growing ``|V|`` by
``sqrt(1 + (delta_omega dt)^2)`` per step) and is the accurate alternative
to sub-stepping at low ``Q_L`` or large detuning.

**Optional generator-current control.** With a ``controller`` attached,
each coarse step forms the error ``V_set - V_ant[n]`` and lets the
controller produce ``I_gen[n]``, which drives the next step; without one,
the generator current stays at the constant feedforward value
``generator_current_bias``. The controller is stepped only on the real
forward passage, never on the reverse reconstruction segments (those
carry a per-segment frame phase, so stepping there would integrate
frame-rotated errors and double-advance the delay line and integrator).
Over that reverse span ``reset_arrays`` therefore seeds the generator grid
with the *last commanded* current instead of the feedforward bias (a
zero-order hold): those cells replay an interval that has already elapsed
and during which the loop issued no new command, so the generator kept
running at whatever it was last told rather than snapping back to the
bias. Without a controller the held value *is* the bias, so the
constant-current path is bit-unchanged. The klystron limit is enforced on
the fine grid as well before the response solve.

**Fine-grid solve.** The generator current is interpolated onto the
profile grid and the cavity response is solved as a sparse bidiagonal
system -- first order by default, or the second-order (Crank-Nicolson)
solver with ``second_order_fine_grid_solver_enable=True``, whose
truncation error scales with the bin size squared. The result is scaled by
``n_cavities`` before the readout phase converts it into the voltage
correction and phase correction the parent RF station applies to its kick.


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
    The station's *RF frequency* offset, added on top of the design
    frequency. Only its *phase* enters the feedback -- both the coarse-grid
    geometry and the demodulation carrier stay on the design clock.
    Concretely:

    * the station accumulates the RF phase slip exactly from the elapsed
      reference time (``delta_omega_rf * dt``, summed at the end of each
      station track) into its kick clock ``delta_phi_rf``;
    * the beam current is demodulated at the *design* carrier and then
      rotated by that accumulated slip (the kick clock plus its live
      end-of-track tail), carried as one constant phase;
    * the readout applies the identical total (the clock via ``phi_rf``, the
      tail via ``phase_correction``), so the slip cancels and the
      demodulation/readout chain closes for every carried deposit.

    The only approximation is the intra-window mismatch
    ``delta_omega_rf * hist_x`` between the design carrier and the actual RF;
    because ``hist_x`` is the bunch-local profile time (about one RF period,
    reset every turn) this term is bounded to ~1e-6 rad and does not
    accumulate. Validated against the retuning convolution at the
    discretization floor (``test_multiturn_delta_omega_rf_*``: large offset,
    differential, sub-stepped, multi-section).
    Guards on the station enforce the supported use: in a ring with more than
    one RF station the offset cannot be changed during the run, and the
    slip bookkeeping only runs when a beam feedback (phase loop) exists in
    the simulation or the offset is nonzero.

For low loaded quality factors the per-RF-period Euler step can violate the
step-size limits; the sub-stepping mode
(``n_rf_periods_per_coarse_grid < 1``) subdivides the RF period, with the
coarse centres tiling continuously across turn boundaries.

**Multi-section registration phase.** A ring with several RF stations
builds each passage's grid piecewise: every reverse segment ``k`` spans
``T_seg,k`` at the past station's design frequency ``omega_k``, while the
forward segment and *both* the demodulation and the readout reference the
single carrier ``omega_0``. The grid therefore accumulates
``sum_k omega_k T_seg,k`` where the carrier accumulates
``omega_0 T_total``, and the difference

   ``Psi = sum_k (omega_k - omega_0) T_seg,k``

is a pure bookkeeping mismatch -- identically zero for a single section,
which is why single-section rings need no correction at all. It is
*separate* from the cavity resonance detuning ``delta_omega``, whose
physical precession the coarse recursion already applies on every step.
``Psi`` is carried as an explicit *carrier* phase, exactly the idiom the
RF-frequency offset above uses: subtracted at demodulation
(``carrier_phase_offset``) and added back at readout
(``phase_correction``). It is deliberately *not* applied as a rotation of
the antenna-voltage state -- that would also rotate the generator-driven
field, which carries no registration error, turning a phase error into an
amplitude drift. See ``_accumulate_registration_phase`` for the
implementation.


Counter-rotating beams
----------------------

The collider ring accelerates a co-rotating mu+ and a counter-rotating mu-
beam through the same cavities. The whole beam-loading chain (RF beam
current, wake-solver sources, and every kick) uses the *direction-signed
charge* ``beam.signed_charge_with_direction()`` -- the particle charge with
its sign flipped for a counter-rotating beam. The collider pair has
*opposite* charges but travels in *opposite* directions, and the two sign
flips cancel: both beams present the **same-sign gap current** to the cavity.
(This is why the two statements below are consistent -- opposite *charges*,
same-sign *currents*.) For an asymmetric fundamental mode their loading then
adds constructively and both receive the same kick. A counter-rotating mu-
beam alone reproduces the co-rotating mu+ run bit-for-bit, through the
feedback and through the convolution reference alike.

With two simultaneous beams (``MainloopCounterRotatingBeams``: each station
is tracked once per beam per turn, the counter-rotating beam traversing the
elements in reverse order), the supported regime is *offset passages* --
stations away from the beams' meeting azimuths. The validated case is the
**two-section** half-drift / station / half-drift layout, where the two
arrivals at a station are ``T_rev / 2`` apart; there the per-passage grid
machinery handles the alternating arrivals natively and matches the two-beam
convolution at reference accuracy. Layouts with more sections (``N >= 4``)
can also keep stations off the meeting azimuths, but the arrival spacing is
then not ``T_rev / 2`` and this regime is currently untested. A station *at* a meeting azimuth (both beams at the
same reference time, e.g. the single mid-ring station of a one-section
layout) is refused with ``NotImplementedError``: the machinery would
silently serialize the coincident arrivals one projection window apart.

.. warning::

   There is currently **no correct model for a station at a meeting
   azimuth** with simultaneous coincident passages. The
   ``MultiPassResonatorSolver`` wakefield with ``allow_delta_t_zero=True``
   permits the coincident (``delta_t = 0``) deposit but applies each beam's
   kick *inside its own track call*, before the other beam's coincident
   profile has been deposited. The beam tracked first therefore sees only
   its own self-loading ``W(0)/2`` while the beam tracked second sees
   ``W(0)`` (self + the first beam's cross-wake): one beam is under-kicked
   by the entire mutual beam-loading term, and swapping the track order
   swaps which beam is affected. Do **not** rely on this path for a
   meeting-azimuth station until the coincident cross-wake is symmetrised
   (deposit both beams' coincident profiles before evaluating either kick).
   Keep stations off the meeting azimuths (offset passages) instead.

For the wake-solver references, ``shunt_impedances_counter_witness``
(``R_CR``) is the shunt impedance a counter-rotating *witness* -- a test
charge integrating the wake in the reverse direction -- actually
*experiences* (its reversed integration direction is baked into the value).
Its sign is a property of the mode's field symmetry, not of fundamental
modes in general:

* ``R_CR = -R`` -- an *asymmetric* fundamental mode: two beams of *opposite*
  charge (the collider pair) add up and receive the same kick;
* ``R_CR = +R`` -- two *same-charge* counter-rotating beams add up.


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
  inventory below.


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
* Driven (generator-bias) multi-section fast-ramp operation keeps a
  readout-*phase* offset: the registration phase ``Psi`` (see *Interplay
  with the RF station* above) reaches the beam through
  ``phase_correction``, but the beam-induced part needs ``Psi`` at readout
  while the generator-driven part does not, and a single readout phase
  cannot separate the two. The amplitude drift this bullet used to
  describe -- percent-level ``|V_ant|`` growth per turn, from applying
  ``Psi`` as a rotation of the antenna-voltage state -- is gone. Pinned by
  ``TestDrivenSteadyStateFastRamp`` and
  ``TestPIFullTrackingMultiSectionFastRamp``.
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
* A configuration whose walked intervals are shorter than two coarse
  steps -- an RF-station section (or the partial first-turn stretch
  before a station, half a section in the symmetric layout) spanning
  fewer than two coarse cells -- is rejected at grid construction:
  ``RFCenterSegment`` requires at least two centres per segment. Reduce
  ``n_rf_periods_per_coarse_grid`` or use fewer/longer sections. (This
  replaced the former empty-segment behaviour, where such a segment
  carried the preceding residual through without adding its own duration
  to the bridging coarse step and a single-centre forward segment could
  silently disarm the counter-rotating coincidence guard.)
