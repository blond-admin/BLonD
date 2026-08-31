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

The coarse grid has two properties a reader will not guess from the array:
its entries are segment-*local* times (so the flat array is *not* globally
monotonic and differencing across a segment boundary is meaningless), and
its step is that segment's *design* RF period (so it is phase-consistent
but *not* uniformly spaced in time). Both are documented once, in the
"The coarse grid" part of the Notes of
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
-- read that before indexing or differencing the array; this page does not
repeat it.

**The generator control loop**

PI controller
    Proportional-Integral controller: commands ``I_gen`` from the voltage
    error ``V_set - V_ant`` (a term proportional to the error plus a term
    integrating it). Here the error is formed in the *kick frame*; see
    *Signal path of one turn*.
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
forward / backfill tracking
    To build the grid the feedback advances the reference *forward* to the
    next RF station; on later turns it *backfills* the stretch of grid that
    has already *elapsed* since its previous update, by re-deriving it from
    the elements it was carried through. Both are directions in **time**,
    and every multi-section ring needs both -- the backfill has nothing to
    do with counter-rotating beams. Do not confuse it with the **space**
    sense of "reverse" used further down, where a counter-rotating beam
    traverses the ring's elements in the reversed order; that is a
    different axis entirely (see the module docstring of
    :mod:`blond.physics.feedbacks.rf_center_grid`, which keeps the two
    apart by name).
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
    That live scalar survives only as the fall-back for a segment-less
    hand-built grid (tests, direct ``circuit_track`` callers): on a real
    per-turn grid a start index that is not a segment boundary trips an
    assertion instead of silently returning this turn's forward tail.
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
    The coarse-grid construction: ``RFCenterGridMixin``, the
    forward/backfill reference walks and per-turn segment generation of the
    timing class. Its module docstring is also where the two meanings of
    "direction" are held apart -- backfill (time) versus the reversed
    element order of a counter-rotating beam (space).

:mod:`blond.physics.feedbacks.rf_center_segment`
    The two coarse-grid value classes. ``RFCenterSegment`` is what the
    grid is built from: the segment's frequency, duration, centre times
    and its ``residual`` -- the unfilled tail between its last centre and
    its end. The ``residual`` is read back by
    ``_preceding_segment_residual`` to form the coarse step into the
    *following* segment's first cell, and is the demodulation frame of the
    forward segment; the ``omega`` and ``duration`` are what the
    backfill-span replay and the registration phase walk.
    ``PerTurnGridSpan`` (below) is the per-turn span built out of those
    segments.

:class:`~blond.physics.feedbacks.rf_center_segment.PerTurnGridSpan`
    Frozen value class returned by one grid rebuild: this passage's
    backfill and forward centre counts plus
    ``residual_from_backfill_span``, the residual snapshotted *between* the
    backfill and the forward generation. Returning it rather than leaving
    it on the feedback is
    what makes the per-turn phase ordering enforceable by the data flow --
    the demodulation frame can only be read from a span object, and a span
    is only produced by a rebuild that snapshotted it in time.

:mod:`blond.physics.feedbacks.generator_regulation`
    ``GeneratorRegulationMixin``: the parts of the timing class that need
    only the controller and the setpoint -- the setpoint policy
    (constructor validation and the per-cavity IQ ``pi_setpoint``), the
    klystron power readout, the per-step generator-current update and the
    fine-grid actuator clamp. The compiled envelope scan and the per-cell
    stepping decision stay on the timing class in ``cavity_feedback.py``:
    they need the coarse grids and the state carried across the turn
    boundary.

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
    (``use_numba_envelope_kernel``); it advances the two source-split
    envelope components, composes their demodulation-frame sum and runs
    the kick-frame PI per cell. It is byte-identical to the
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
   assigned to ``_kick_clock_slip_gap``, which makes visible that the
   value is reset at every passage rather than accumulated. It is one of
   the two constituents of ``_carrier_slip_gap`` -- the other is the
   multi-section registration phase of step 4 -- and the two are held
   separately because the generator-component frame rotation needs the
   kick-clock part with the station clock on top (step 4).

3. ``_rebuild_per_turn_grid`` -- rebuilds this passage's coarse grid
   (``rf_centers``), sizes the coarse state and returns a frozen
   :class:`~blond.physics.feedbacks.rf_center_segment.PerTurnGridSpan`. It
   first calls ``_close_previous_turn_grid``, which captures the previous
   turn's last centre and its end-of-turn residual
   (``_residual_time_carried_into_turn``) *before* clearing the segment
   list, and then generates this passage's segments: the feedback tracks a
   copy of the beam reference forward to the next RF station and, on later
   turns, re-derives the segments that elapsed since its last update
   (the backfill). Each segment carries the *design* RF frequency it
   was tracked with (at the local reference energy), so the coarse-step
   spacing follows the design RF period even under acceleration and with
   several stations per ring. A station RF-frequency offset
   ``delta_omega_rf`` never moves the grid, and does not shift the
   demodulation carrier either (which stays on the design clock): it
   enters only as the explicit constant phase assembled in steps 2 and
   4. ``reset_arrays``
   is the last statement of this phase -- it can neither precede the grid
   generation it takes its size from, nor follow any ``circuit_track``.

4. ``_accumulate_registration_phase`` -- accumulates the multi-section
   grid-vs-carrier registration phase
   ``Psi = sum_k (omega_k - omega_0) T_seg,k`` (explained under *Interplay
   with the RF station* below) and returns the running total; exactly
   ``+0.0`` for a single section and for an unaccelerated ring.
   ``_carrier_slip_gap`` is then formed as the kick-clock gap of step 2
   plus this total, and ``_update_frame_rotations`` derives the two
   per-passage frame rotations every later cell update reads: the
   *generator* frame rotation ``exp(-i (delta_phi_rf + gap + Psi))``,
   which rotates the design-anchored generator component into the
   demodulation frame when the sum is composed, and the *kick* frame
   rotation ``exp(+i (gap + Psi))``, in which the PI error is formed.
   Both are exactly ``1 + 0j`` without an RF-frequency offset and
   without multi-section acceleration, so those paths stay
   bit-identical.

5. ``_replay_backfill_span`` -- re-walks this passage's backfill segments
   with ``no_beam=True``, one ``circuit_track`` per backfill segment at
   that segment's own ``omega``, so that the envelope carries the
   already-elapsed interval forward. A passage that generated no backfill
   segments skips the replay entirely. It runs after step 4 because its
   cell updates already compose the sum with this passage's rotations.

6. ``_write_no_correction_readout`` -- only with
   ``grid_only_no_correction=True``: writes the neutral readout (unit
   relative voltage, zero phase, i.e. **no correction at all**) and ends
   the turn there, so neither the demodulation nor the forward pass
   runs. The three diagnostic switches are independent: ``debug`` only
   records the inspection-only grid snapshots,
   ``validate_grid_each_turn`` only runs the per-turn grid integrity
   check (including the residual-versus-demodulation-frame assertion),
   and only this one -- ``grid_only_no_correction`` -- stops the physics.
   They were once a single ``debug`` flag doing all three at once, so
   asking for diagnostics silently switched the feedback off entirely
   (unit gain, zero phase). With all three at their ``False`` default the
   tracked result is bit-for-bit what the old ``debug=False`` produced.

7. ``_track_forward_span`` -- the real work of the turn, in two steps.

   *Demodulation*: ``calculate_rf_beam_current_partial`` calls
   :func:`~blond.physics.feedbacks.beam_current.rf_beam_current` to
   convert the beam profile into the complex IQ beam-current envelope at
   the *design* carrier (factor-2 single-sideband demodulation), rotate it
   by the reference-frame phase and by the constant
   ``-(delta_phi_rf + _carrier_slip_gap)``, and re-bin the fine-grid
   charge onto the coarse cells charge-conservingly. The demodulation
   frame is the span's ``residual_from_backfill_span``, snapshotted before
   the forward generation overwrote the host scalar; re-reading that
   scalar here would silently shift the frame. Several guards protect this
   path, all of them raising rather than correcting:

   * charge in the *first* coarse cell -- that cell seeds the fine-grid
     initial condition, so its kick would be double-counted;
   * a profile window longer than the coarse grid it is re-binned onto,
     rejected by ``ProfileBaseClass.check_fits_in_span`` (the forward
     span is not periodic, so a wrapped group would overwrite an earlier
     cell instead of accumulating into it);
   * a profile binning coarser than the coarse cell
     (``hist_step > sampling_time``): the downsampling counts consecutive
     index steps, so a jumping index places charge at the wrong time
     while conserving the total -- reachable from a legitimate-looking
     sub-stepped setup, which shrinks ``sampling_time``;
   * a window mapping past the last coarse cell, and a window mapping
     *before* the first one. The latter used to warn and let NumPy's
     negative indexing deposit the charge onto the *last* coarse cells,
     about a forward span too late; it now raises as soon as the
     underflowing bins carry non-negligible charge (a charge-free
     Gaussian tail sticking out below the grid start still only warns).

   A warning -- not an error -- fires if the profile window does not
   capture the whole beam.

   *Forward pass*: one ``circuit_track`` over the forward segment, which
   performs the coarse-grid cavity update and the optional generator
   control, then hands the fine-grid half to
   ``_resolve_fine_grid_voltage`` (initial condition, generator-current
   interpolation and the fine solve described below).

8. ``_write_station_readout`` -- converts the fine-grid antenna voltage
   into ``relative_voltage_correction`` (divided by the station voltage)
   and ``phase_correction`` (referenced to the mean phase of
   ``station_voltage_coarse_grid``, plus the very same
   ``_carrier_slip_gap`` the demodulation subtracted). Per component
   that closes two different chains: the beam component gets back
   exactly the total its demodulation subtracted (the
   demodulation/readout closure, byte-for-byte as before the envelope
   split), while the generator component -- composed into the sum with
   ``exp(-i (delta_phi_rf + gap + Psi))`` -- nets to its design-clock
   phase. A driven, beam-free cavity on its setpoint therefore reads
   out ``phase_correction == 0`` exactly: at zero intensity the
   feedback is *guaranteed* phase-neutral, whatever the ramp and the
   section count (pinned at 1e-12 rad by
   ``TestDrivenFeedbackIsPhaseNeutralWithoutBeam``). These two arrays
   are what the parent RF station applies to its kick.

**Coarse-grid cavity update** (inside ``circuit_track``). The antenna
voltage is advanced cell by cell with the forward-Euler discretisation of
the cavity-envelope ODE: generator drive ``I_gen (R/Q) omega dt``,
decay/detuning multiplier ``1 - 0.5 omega dt / Q_L + i delta_omega dt``
and beam loading ``-0.5 I_beam (R/Q) omega dt``. The ODE is linear, so
the state is *source-split* and the same recursion runs once per source
-- exact superposition. The beam-sourced component
``antenna_voltage_beam_coarse_grid`` is driven by ``-I_beam / 2`` alone
and is anchored to the demodulation frame (for an undriven feedback it
*is* the former single state, bit-for-bit). The generator-sourced
component ``antenna_voltage_gen_coarse_grid`` is driven by ``I_gen``
alone and is natively anchored to the piecewise *design* clock: the
klystron drive follows the design frequency, whose per-segment values
the coarse grid already samples, so injecting a constant current per
segment is exactly right and the component carries neither the
kick-clock slip nor the registration phase (``initial_voltage`` and the
pre-fill seed this component -- they model a generator-established
field). The public ``antenna_voltage_coarse_grid`` remains the
DEMODULATION-FRAME SUM, (re)composed per passage as
``V_beam + V_gen * exp(-i (delta_phi_rf + gap + Psi))`` -- a rotation
that is exactly ``1 + 0j`` without an RF-frequency offset and without
multi-section acceleration, which is why undriven runs stay
byte-identical to the former single-state recursion.

.. note::

   The forward-Euler description is exact for the GENERATOR term only.
   The coarse recursion takes ``I_gen`` from cell ``c-1`` (left
   endpoint, i.e. forward Euler) but ``I_beam`` from cell ``c`` itself,
   so the bunch's own slice enters with weight 1. The three
   discretisations in the code therefore disagree on that self-slice
   weight: coarse = 1, fine first-order = 0, fine second-order = 1/2 --
   and 1/2 is the value the fundamental theorem of beam loading calls
   for. The difference does not move any published number (the coarse
   voltage never kicks the beam; the kicks come from the fine grid,
   whose second-order solver the example enables), but it is a real
   asymmetry and the numbers are pinned bit-for-bit, so changing the
   weights is a deliberate decision rather than a cleanup.

Discretisation validity is enforced by
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

A *coincident* coarse point -- two centres a step of ``delta_t == 0``
apart, which a segment or turn boundary can produce (and which float noise
of a few ULPs is clamped to) -- carries no elapsed time, so
``V(t + 0) = V(t)``: the cell duplicates the previous cell's
antenna-voltage components and generator current (and recomposes the
sum), taking them across the turn boundary when it
is the very first cell. It used to be skipped, which left the cell at the
zeros prefill so the *next* cell propagated from ``V = 0``, destroying the
coherent voltage and refilling it only over ``2 Q_L / omega`` -- hundreds
of turns. Duplication also keeps the two downstream readers honest, since
``reset_arrays`` carries the *last* cell into the next turn and the fine
solve seeds from the *first* forward cell. The controller is still not
stepped there: no time elapsed, so there is no new sample to regulate on.

**Optional generator-current control.** With a ``controller`` attached,
each coarse step forms the error in the KICK frame,
``V_set - V_ant[n] * exp(+i (gap + Psi))`` -- the envelope of the kick
the station actually applies against ``phi_rf``, so the loop regulates
the applied voltage rather than a bookkeeping frame; the rotation is
exactly unity without an RF-frequency offset and without multi-section
acceleration, and the pure-Python path and the numba kernel form it
identically -- and lets the
controller produce ``I_gen[n]``, which drives the next step; without one,
the generator current stays at the constant feedforward value
``generator_current_bias``. The controller is stepped only on the real
forward passage, never on the backfill reconstruction segments (those
carry a per-segment frame phase, so stepping there would integrate
frame-rotated errors and double-advance the delay line and integrator).
Over that backfill span ``reset_arrays`` therefore seeds the generator grid
with the *last commanded* current instead of the feedforward bias (a
zero-order hold): those cells replay an interval that has already elapsed
and during which the loop issued no new command, so the generator kept
running at whatever it was last told rather than snapping back to the
bias. Resetting them to the bias was a real defect, not a cosmetic one:
with a detuned cavity the PI holds a reactive standing current, which the
old reset discarded once per turn (measured setpoint errors of 3.1e-2 and
4.6e-2 relative at 2 and 4 sections). Without a controller the held value
*is* the bias, so the constant-current path is bit-unchanged. The klystron
limit is enforced on the fine grid as well before the response solve.

**Fine-grid solve** (``_resolve_fine_grid_voltage``). The generator current
is interpolated onto the profile grid and the cavity response is solved as
a sparse bidiagonal system -- first order by default, or the second-order
(Crank-Nicolson) solver with ``second_order_fine_grid_solver_enable=True``,
whose truncation error scales with the bin size squared. The result is
scaled by ``n_cavities`` before the readout phase converts it into the
voltage correction and phase correction the parent RF station applies to
its kick. The initial condition it starts from is described next.


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
coarse-grid Euler step. The fill is evaluated on the **design** clock
(``omega_rf_design``), the same clock the coarse recursion it seeds is
driven at, and ``t_rev`` is read on that clock too. It previously mixed
clocks: evaluating the fill at the actual (offset) RF frequency misses the
recursion's own no-beam fixed point by ``O(delta_omega_rf / omega)``,
leaving an injection transient the PI then has to burn off. Either seed
-- the scalar ``initial_voltage`` or the pre-fill -- models a
generator-established field, so it seeds the *generator* component of
the source-split coarse state (the beam component starts empty).

**The fine-grid initial condition.** The fine solve is seeded with the
coarse antenna voltage at index ``[0]`` of the forward segment -- the
*first* forward coarse centre -- and then integrates the beam current over
``[cut_left, cut_right]``. Two halves of one invariant keep that causal.

The first is a per-turn guard,
``_check_fine_grid_initial_condition_is_causal``: the centre the seed comes
from must not be later than the start of the window it initialises,

   ``first forward centre <= profile.cut_left``,

checked whenever the window carries charge, and raising otherwise -- the
seed would then be taken from later in the turn than the interval it
initialises, and the beam current would be integrated twice. It is checked
every turn rather than once at setup, because the first forward centre
moves with the design frequency and with the residual carried from the
previous passage (both turn-dependent under acceleration and sub-stepping)
and ``cut_left`` is itself settable. The remedy is to move the profile
window right, to ``cut_left >= max(t_rf / 2, sampling_time_coarse)``.

The second is that the seed is deliberately the coarse value *at index*
``[0]``, and deliberately *not* interpolated onto ``cut_left``. This looks
like an easy accuracy win and is not: coarse cell 0 is charge-free by
construction (``forbid_charge_in_first_coarse_cell``), but cell 1
typically already holds about half the bunch and therefore its beam-induced
voltage step, so interpolating from cell 0 towards cell 1 drags up to ~10 %
of the beam-induced voltage *backwards* in time, into an initial condition
that predates the charge which produced it -- and the fine grid then
re-integrates that same current. Trying it broke 57 tests, including the
independent comparisons against the multi-pass wake solver. Do not
"improve" it.


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
      demodulation/readout chain closes for every carried deposit;
    * the klystron drive, by contrast, follows the *design* frequency
      (the generator component is design-anchored), so under an offset
      the driven field physically walks off the actual RF at MINUS the
      accumulated kick-clock slip: a beam-free, matched-bias cavity
      reads out ``phase_correction == -delta_phi_rf``. This is modelled
      physics, not an artefact
      (``TestDesignLockedDriveWalkOffUnderRFOffset`` pins it per turn at
      1e-9 rad).

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
builds each passage's grid piecewise: every backfill segment ``k`` spans
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

The source-split coarse state (see *Signal path of one turn*) is what
lets ``Psi`` reach exactly the signal that needs it: the beam-sourced
component's demodulation/readout closure carries ``Psi``, while the
design-anchored generator component sees no registration phase at
readout at all, and the PI regulates the kick-frame sum. This closed
the former driven multi-section readout-phase offset -- one shared
readout phase used to hand ``Psi`` to the generator-driven field too,
walking the RF bucket off the design synchronous phase with no beam at
all -- and it is why the zero-intensity phase neutrality of the readout
is exact rather than approximate (the amplitude-drift half of the same
history, percent-level ``|V_ant|`` growth per turn from rotating the
state, had already been fixed by carrying ``Psi`` on the carrier; both
are pinned by ``TestDrivenSteadyStateFastRamp``,
``TestDrivenFeedbackIsPhaseNeutralWithoutBeam`` and
``TestPIFullTrackingMultiSectionFastRamp``).

One reading rule follows for driven runs: ``antenna_voltage_coarse_grid``
is the demodulation-frame sum, so under an accumulated slip its complex
value appears rotated by minus that slip while ``|V|`` is invariant -- a
naive complex comparison against the setpoint is the wrong check;
compare in the kick frame (as the PI does), or compare magnitudes.

**Multi-harmonic stations.** The feedback is not restricted to the main
harmonic: it can be attached to a
:class:`~blond.physics.cavities.MultiHarmonicRFStation`, and the
constructor argument ``harmonic_index`` (default ``0``) selects which
harmonic it regulates. Every RF parameter it reads -- ``omega_rf``,
``phi_rf``, ``delta_omega_rf``, the harmonic number, the station voltage --
and the design frequency the coarse grid is built from are taken at that
index. One feedback instance regulates one harmonic; build a separate
instance per harmonic.

Because the two sides address the harmonic differently, they must agree:

    **Slot agreement rule.** The station applies each feedback's
    ``relative_voltage_correction`` / ``phase_correction`` at that
    feedback's *position* in ``cavity_feedback_list``
    (``enumerate`` in ``calc_gap_voltage_with_feedbacks``), while the
    feedback *computes* them from the RF parameters at its own
    ``harmonic_index``. A disagreement applies corrections derived from
    harmonic A to harmonic B: no crash, wrong physics. The *slot* is
    authoritative: ``attach_cavity_feedback`` SETS the feedback's
    ``harmonic_index`` to the slot it is placed at (the
    ``harmonic_index`` argument, or its position in a provided list),
    silently overriding any value given at construction. The
    constructor value is only the default the feedback carries while it
    is unattached.

A mismatch therefore cannot arise through the attach path -- neither
through ``attach_cavity_feedback`` nor through the station constructor,
which routes through it. What the attach cannot see is a
``cavity_feedback_list`` mutated directly afterwards, so
``_validate_multi_harmonic_slot`` still checks the agreement at run
start (``on_run_simulation``). That run-start check also catches a
feedback that never made it into the list at all, and one instance
occupying several slots. Run start is the earliest it can run: the
parent station is attached *after* the feedback is constructed, so
``__init__`` cannot see it.

``attach_cavity_feedback`` also rejects an out-of-range slot at *both*
ends. The upper bound was always there; the lower one was missing, so a
negative ``harmonic_index`` indexed the list from its end and silently
regulated the last harmonic. A fractional slot is likewise a hard error
at both entry points (the attach and the feedback constructor): a
harmonic index is a list slot, not a physical quantity to be rounded.
Plain ``int``, ``np.integer`` and integral floats are accepted silently.


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
keep stations off the meeting azimuths at a different spacing -- station
``i`` sees the two beams ``|N - 2 i - 1| / N * T_rev`` apart, never half a
turn at ``N = 4`` -- and they are validated too: four and six sections match
the two-beam convolution to 0.128 % on the first turn, falling to 0.039 %,
against the same 0.5 % gate and within 0.001 percentage points of the
two-section numbers, in all three regimes (static, accelerating fast ramp,
``delta_omega_rf``). That matters because two sections is also the only count
at which the backfill interval is empty at every station, so the backfill
reference walk is never entered; a 16-section RCS enters it at 14 stations
every turn. A station
*at* a meeting azimuth (both beams at the
same reference time, e.g. the single mid-ring station of a one-section
layout) is refused with ``NotImplementedError``: the machinery would
silently serialize the coincident arrivals one projection window apart.

.. warning::

   There is **no correct model for a station at a meeting azimuth** with
   simultaneous coincident passages, and none is planned. The
   ``MultiPassResonatorSolver`` wakefield with ``allow_delta_t_zero=True``
   permits the coincident (``delta_t = 0``) deposit but applies each beam's
   kick *inside its own track call*, before the other beam's coincident
   profile has been deposited. The beam tracked first therefore sees only
   its own self-loading ``W(0)/2`` while the beam tracked second sees
   ``W(0)`` (self + the first beam's cross-wake). For two equal coincident
   charges the kicks come out as ``0.5`` and ``1.5`` times the correct
   ``W(0) Q``: the *sum* survives, the *split* does not, so the artefact
   appears as a spurious differential between the two beams -- exactly the
   quantity a two-beam study measures -- and swapping the track order swaps
   which beam is under-kicked.

   Results from this path are therefore **wrong**, not merely
   order-dependent. Symmetrising the coincident cross-wake (deposit both
   beams' profiles before evaluating either kick) is a deliberate
   **non-goal**: the case is unreachable unless ``allow_delta_t_zero=True``
   is chosen explicitly, and the feedback refuses a meeting-azimuth station
   outright. Instead of a fix, the situation announces itself -- the solver
   warns at construction, and a deposit that really is coincident emits a
   second ``UserWarning`` at that moment (once per solver) stating that the
   induced voltage from there on is wrong. Keep stations off the meeting
   azimuths (offset passages) instead.

For the wake-solver references, ``shunt_impedances_counter_witness``
(``R_CR``) is the shunt impedance a counter-rotating *witness* -- a test
charge integrating the wake in the reverse direction -- actually
*experiences* (its reversed integration direction is baked into the value).
Its sign is a property of the mode's field symmetry, not of fundamental
modes in general:

* ``R_CR = -R`` -- two beams of *opposite* charge (the collider pair) add
  up and receive the same kick, while same-charge beams cancel;
* ``R_CR = +R`` -- two *same-charge* counter-rotating beams add up,
  while opposite-charge beams cancel.

Choose by that behaviour. The sign does follow from the parity of the
mode's field under reversal of the direction of travel, but the parity
convention is not written down anywhere in this code base -- see the
warning under the ``shunt_impedances_counter_witness`` parameter of
:class:`~blond.physics.impedances.sources.Resonators`, and note that
nothing validates the sign you pass.


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
  the coarse-grid tiling from the RF bucket. **Only some of those cases are
  refused, and the rest are silently wrong** -- do not rely on this being
  caught.

  The grid seeds every segment half an RF period in, so a segment spanning
  a fractional number of RF periods leaves a residual different from
  ``t_rf / 2``; that residual is the demodulation frame ``dT``, and the
  fundamental theorem of beam loading needs ``omega * dT = pi``
  (mod ``2 pi``). Which fraction it is decides what happens:

  - ``1/4`` and ``3/4`` of a period push beam charge into the first coarse
    cell, so ``rf_beam_current`` raises before any voltage is produced.
    That refusal -- and only that one -- is pinned as a contract by
    ``test_multiturn_nondivisible_harmonic_is_rejected`` in the multi-turn
    comparison suite, which asserts the ``ValueError`` and that its message
    stays actionable.
  - ``1/2`` of a period (``harmonic % (2 * n_sections) == n_sections``,
    which includes every odd harmonic on a one-station ring) gives
    ``omega * dT = 2 pi``, i.e. the demodulation factor is ``+1`` where it
    must be ``-1``. Nothing complains: the run completes and **the
    beam-induced voltage has the wrong sign**, so the bunch is accelerated
    by its own wake. Measured against the ``MultiPassResonatorSolver``:
    199.9 % relative error on the first turn.

  There is currently no guard for the ``1/2`` case. Choose
  ``harmonic % (2 * n_sections) == 0`` for the symmetric half-drift /
  station / half-drift layout, and more generally make every stretch
  between the ring start and an RF station, and between two consecutive RF
  stations, span a whole number of RF periods.
  ``muon_collider_blonder.rcs_two_beam_example`` does this itself, reducing
  the JSON harmonic to a multiple of ``2 * n_sections``.

  Sub-stepped grids (``n_rf_periods_per_coarse_grid < 1``) are exempt: they
  tile continuously across segment boundaries instead of re-seeding at the
  bucket phase, so ``dT`` is one previous coarse step by construction.
* **The demodulation frame carries a stale-frequency lag under a ramp.**
  The tail ``dT`` that sets the beam-current demodulation frame is left by
  the *preceding* coarse segment, but is consumed against the *current*
  segment's design carrier. Under acceleration the two frequencies differ,
  so the frame is short (or long) by ``(omega_fwd - omega_prod) * dT``.
  Because ``dT ~ t_rf / 2 = pi / omega``, that error expressed in ``pi`` is
  simply the fractional per-segment frequency change::

      frame lag [pi]  ~  (omega_fwd - omega_prod) / omega

  This is an accepted approximation, not a defect to work around: measured
  over the shipped programmes it is ``7.9e-8 pi`` on RCS1 -- the fastest
  ramp, ~23 % energy gain per turn -- and ``9.4e-10 pi`` on RCS2, against
  the ``1e-3 pi`` tolerance of the demodulation-frame guard. The margin is
  ~1.3e4.

  A substantially more violent ramp would erode it. The failure is loud
  rather than silent -- the guard raises as soon as the lag reaches
  ``1e-3 pi`` -- and the fix is local: ``RFCenterSegment`` already stores
  ``omega`` beside ``residual``, so the frame can be rebuilt from the
  carrier that actually produced the residual.

* In a ring with more than one RF station the ``delta_omega_rf`` offset
  cannot be changed during the run (the station raises). The former
  lab-frame demodulation slip under an offset (an error growing with the
  absolute reference time) is fixed: the demodulation carrier is anchored
  to the accumulated actual RF phase and validated at the discretization
  floor for offsets beyond the cavity half-bandwidth
  (``test_multiturn_delta_omega_rf_*``).
* The undriven two-section fast-ramp carried wake shows a slow bounded
  secular drift (~0.03 percentage points per turn over 20 turns) against
  the convolution.
* Two counter-rotating beams passing a station *simultaneously* (station at
  a meeting azimuth) are refused rather than integrated; see
  *Counter-rotating beams* above for the guard and the workaround.
* The coarse re-binning of the beam current assumes the analytic uniform
  grid; configurations far from the tested ones (unusual profile placement)
  should be validated against the wake solvers. The two gross violations
  now raise instead of corrupting silently -- a window longer than the
  coarse span, and a profile binned more coarsely than a coarse cell (see
  the demodulation guards under *Signal path of one turn*) -- but the
  guards bound the input, they do not extend the assumption. Sub-stepped
  beam loading itself is validated against the convolution, including with
  detuning and on the fast ramp.
* The profile window must lie inside the forward coarse grid, with its
  first coarse cell charge-free and its left edge not earlier than the
  first forward coarse centre -- in practice
  ``cut_left >= max(t_rf / 2, sampling_time_coarse)``. All three are now
  enforced; see *the fine-grid initial condition* under *Initial
  conditions and cavity pre-fill*. Seeding from coarse index ``[0]``
  rather than interpolating to the profile edge is a deliberate, measured
  choice there, not an approximation waiting to be improved.
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
