.. _mucol_cavity_feedback_overview:

Muon-Collider Cavity Feedback -- Overview
=========================================

This page gives an architectural overview of the RF cavity-feedback model in
``blond.physics.feedbacks``, developed for the muon-collider Rapid-Cycling
Synchrotron (RCS) studies. It describes what the classes do and how the pieces
fit together; the API reference is generated from the docstrings (see
:mod:`blond.physics.feedbacks.cavity_feedback`). The tested configurations,
reference models and tolerances are documented in
:ref:`mucol_cavity_feedback_tests`; they do not establish correctness for
every possible configuration.

.. contents:: Contents
   :local:
   :depth: 2


Setup contract
--------------

Check these conditions before building a run. They describe the current
implementation; the detailed numerical limits appear under *Known
limitations* below.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Item
     - Required setup
   * - Backend
     - Use a supported 64-bit BLonD backend. Feedback signal processing
       uses host NumPy/SciPy arrays even when particle tracking uses a GPU;
       a CPU run does not validate GPU execution.
   * - RF geometry
     - For the symmetric half-drift/station/half-drift layout, choose
       ``harmonic % (2 * n_sections) == 0``. Each walked segment needs at
       least two coarse centres. This is a grid/demodulation constraint,
       not a restriction on which physical cavities can exist.
   * - Sampling
     - Use a static profile with ``hist_step <= sampling_time_coarse``.
       The standard coarse step is one RF period; ``0.5`` is the supported
       sub-step. The coarse step is the exact exponential propagator for
       any step length; resolve the fine dynamics sufficiently finely.
   * - Profile placement
     - Keep the entire bunch inside the profile and the seeding coarse
       cell charge-free. A charged fine window must begin at or after the
       first forward coarse centre. Recheck this under a ramp.
   * - Ownership and initialization
     - Give each station its own profile, feedback and controller. Attach
       feedback before ``Simulation.run_simulation``; late initialization
       supplies ring geometry and station parameters. Let the RF station
       invoke its feedback and the simulation establish passage order.
   * - Two beams
     - Pass both beams to one simulation, with opposite rotation flags.
       They share each physical station's feedback state. Use stations
       away from the beams' meeting azimuths; coincident passages are
       unsupported. Do not run two independent feedback loops for one cavity.
       Live profiles must have a placement the counter-rotating mainloop
       can verify; a shared frozen line density is suitable for circuit
       comparisons but does not follow evolving bunch shapes.

**Units and timestamps at the handoff**

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Quantity
     - Meaning
   * - ``initial_voltage``, coarse antenna voltage
     - Complex per-cavity envelopes [V]. Each coarse value belongs to its
       own segment-local centre timestamp; the full concatenated time
       array is not globally monotonic.
   * - Generator current, controller output
     - Per-cavity current [A]. Command at coarse centre ``i`` drives the
       interval to centre ``i + 1``. The seed-to-window reconstruction
       uses this history without updating controller state again.
   * - ``rf_beam_current`` return values
     - Demodulated charge [C] per bin. Divide by the corresponding bin
       duration to obtain the currents [A] supplied to the response solver.
   * - Fine initial voltage
     - Propagated per-cavity envelope at ``profile.cut_left``. The first
       output is half a fine bin later, at ``profile.hist_x[0]``.
   * - ``antenna_voltage_fine_grid``
     - Station-total complex envelope [V], already multiplied by
       ``n_cavities``. Do not multiply it again. The station projects it
       with the carrier phase to obtain the real voltage applied to particles.
   * - Station voltage and controller setpoint
     - Station voltage is total [V]; the feedback derives the per-cavity
       controller target by dividing by ``n_cavities``.


Minimal executable setup
------------------------

The downloadable :download:`example <examples/minimal_feedback.py>` runs
one turn on a constant-energy ring without writing output files. Run it
with a Python environment in which BLonD is installed:

.. code-block:: console

   python docs/feedbacks/examples/minimal_feedback.py

It runs both ``run_example(two_beams=False)`` and
``run_example(two_beams=True)``. The latter uses two offset stations and
shares each station's feedback between a co-rotating mu+ beam and a
counter-rotating mu- beam. Their macroparticles have zero total intensity
for this initialization check: the expected fine envelope at each station
is ``1e6 + 0j`` V. Each station represents two cavities initialized at
``0.5e6`` V apiece, with the resonant equilibrium current
``I_gen = V_per_cavity / (2 * (R/Q) * Q_L)``.

The two-beam example freezes the initial histogram shared by the equal
bunches. It demonstrates the circuit and passage ordering, not
self-consistent evolution of both bunch shapes.
Call ``run_example(two_beams=True, intensity=1e9)`` to include beam loading.
The return value contains each station's envelope after its last passage;
record per-passage observations when both beams' individual kicks matter.
The demonstration controller gains are not an RCS tuning prescription.
For an accelerating machine study, use the outer project's
``muon_collider_blonder.rcs_two_beam_example`` and its machine parameters.

.. literalinclude:: examples/minimal_feedback.py
   :language: python
   :start-at: import numpy
   :end-before: if __name__


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
gap voltage
    The accelerating voltage the bunch actually *sees* at the cavity gap:
    the real carrier projection of the fine-grid antenna voltage, whose
    array already includes ``n_cavities``. Distinct from
    the coarse ``antenna_voltage_coarse_grid``, which is the envelope the
    loop regulates and which never kicks the beam.
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
actually receives is resolved on the dense **fine grid** (the profile grid).
That solve evolves a timestamped, charge-free coarse seed; it does not
interpolate beam-loaded coarse voltages onto the profile.

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
    integrating it). Here the error is formed in the *kick frame* (see the
    frames table below) and then rotated into the *actuator frame* before
    it reaches the controller; both are defined there.
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
   * - kick frame
     - The frame the station's applied kick lives in: the demodulation
       frame rotated by ``exp(+i (gap + phi_acc))`` (the live kick-clock
       gap plus the accumulated phase: the forward segment's on the
       forward span, the phase accumulated up to the cell on a backfill
       cell). The PI error is formed here, so the loop regulates the
       applied voltage rather than a bookkeeping frame. Exactly the
       demodulation frame without an RF-frequency offset and without
       multi-section acceleration. Not the same thing as the *kick clock*
       above.
   * - actuator frame
     - The frame the commanded generator current acts in: the kick frame
       rotated by ``exp(+i delta_phi_rf)``. The PI error is taken here
       because the controller drives the design-anchored generator
       component, so ``d(V_kick)/d(I_gen)`` carries ``exp(-i
       delta_phi_rf)`` and rotating the error back keeps the open-loop
       gain real. Exactly the kick frame whenever ``delta_phi_rf`` is
       zero.

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
    The *same* residual, snapshotted between the backfill and the forward
    generation, is the demodulation frame of the forward segment (next
    entry). Which reader takes it from where, and why the snapshot has to
    exist, is in step 3 of *Signal path of one turn*.
demodulation frame (``dT``)
    The time offset the beam current is demodulated against: the residual
    left by the coarse segment preceding the forward one, carried on the
    span as ``residual_from_backfill_span``. This implementation's mixing
    and kick-phase convention requires ``omega * dT = pi`` (mod ``2 pi``)
    to align the beam-loading sign and phase. It seeds each segment half
    an RF period into the bucket. This is a convention-specific alignment
    condition, not a universal statement of the beam-loading theorem.
    Everything that can
    perturb ``dT`` (a harmonic not divisible by ``2 * n_sections``, a
    sub-step other than ``0.5``, a stale segment frequency under a
    violent ramp) rotates the beam-induced voltage, and past a quarter
    period inverts it; any frame more than ``1e-3 pi`` off an odd
    multiple of ``pi`` is refused outright (see *Known limitations*).
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
    backfill-span replay walks, and ``accumulated_phase`` is the
    grid-vs-carrier phase accumulated up to the segment's end (see
    *Multi-section registration phase*). Two pure helpers compute that
    phase: ``accumulated_phases`` at each backfill segment's end (what the
    records store) and ``accumulated_phases_at_centers`` at each backfill
    centre (from the records).
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
    klystron power and the reflected current and power readouts (see
    *Multi-section registration phase* for their frame), the per-step
    generator-current update and the fine-grid actuator clamp. The
    compiled envelope scan and the per-cell stepping decision stay on the
    timing class in ``cavity_feedback.py``: they need the coarse grids and
    the state carried across the turn boundary.

:mod:`blond.physics.feedbacks.beam_current`
    The beam-current demodulation:
    :func:`~blond.physics.feedbacks.beam_current.rf_beam_current` (fine-grid
    demodulation, optionally re-binned onto the coarse grid when
    ``sampling_time``/``n_points`` are given) and the ``low_pass_filter``
    it applies under ``use_lowpass_filter``.

:mod:`blond.physics.feedbacks.cavity_solvers`
    The muon-collider-only numerics: the first-order (forward-Euler)
    fine-grid solver
    :func:`~blond.physics.feedbacks.cavity_solvers.cavity_response_sparse_matrix`,
    its second-order twin
    :func:`~blond.physics.feedbacks.cavity_solvers.cavity_response_sparse_matrix_second_order`
    (trapezoidal / Crank-Nicolson) and the feedforward fill seed
    :func:`~blond.physics.feedbacks.cavity_solvers.pretrack_fill_voltage`.
    It also holds the arithmetic of the exact exponential coarse-grid step,
    shared by the per-cell and vectorised recursions --
    :func:`~blond.physics.feedbacks.cavity_solvers.coarse_step_exponent`,
    :func:`~blond.physics.feedbacks.cavity_solvers.exponential_voltage_multiplier`
    and
    :func:`~blond.physics.feedbacks.cavity_solvers.exponential_drive_weight`
    -- and the beam-free seed propagation ``propagate_beam_free_voltage``.
    The forward-Euler coarse step and its validity guard were removed on
    2026-09-11 (see *Coarse-grid cavity update*).

:mod:`blond.physics.feedbacks.envelope_kernel`
    The compiled numba host kernel (``envelope_pi_scan``) the coarse
    per-cell recursion runs on by default
    (``use_numba_envelope_kernel``); it advances the two source-split
    envelope components, composes their demodulation-frame sum and runs
    the kick-frame PI per cell, taking the generator and kick frame
    rotations as per-cell arrays. It is byte-identical to the
    pure-Python per-cell reference wherever the klystron clamp does not
    fire, and agrees with it to ``SATURATED_RTOL`` (1e-12) where it does:
    numba's complex ``abs`` and numpy's differ by one or two ULP, which
    moves the clamped current by ~3e-16 relative and leaves the antenna
    voltages and the PI integral exactly equal, so a saturated segment is
    committed from the kernel rather than re-run. The reference is kept
    as that reference and as the one remaining exact fallback: a segment
    is re-run there when two coarse points coincide (zero step), and a
    controller that supplies no compiled form of itself
    (``supports_envelope_scan``) is driven cell by cell instead.
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
   accumulated phase of the forward segment built in step 3, and step 4
   sums them. The split is presentational: ``_kick_clock_slip_gap`` has
   exactly that one consumer, and exists so that the reset-per-passage
   semantics of the gap stay visible against the accumulated phase it is
   added to.

3. ``_rebuild_per_turn_grid`` -- rebuilds this passage's coarse grid
   (``rf_centers``), sizes the coarse state and returns a frozen
   :class:`~blond.physics.feedbacks.rf_center_segment.PerTurnGridSpan`. It
   first calls ``_close_previous_turn_grid``, which captures the previous
   turn's last centre, its end-of-turn residual
   (``_residual_time_carried_into_turn``) and its forward segment
   (``_forward_segment_carried_into_turn``) *before* clearing the segment
   list, and then generates this passage's segments: the feedback tracks a
   copy of the beam reference forward to the next RF station and, on later
   turns, re-derives the segments that elapsed since its last update
   (the backfill). Each segment carries the *design* RF frequency it
   was tracked with (at the local reference energy), so the coarse-step
   spacing follows the design RF period even under acceleration and with
   several stations per ring, and stores the grid-vs-carrier phase
   accumulated up to its end (``accumulated_phase``, continued from the
   carried forward segment; the forward segment inherits the last
   backfill value). A station RF-frequency offset
   ``delta_omega_rf`` never moves the grid, and does not shift the
   demodulation carrier either (which stays on the design clock): it
   enters only as the explicit constant phase assembled in steps 2 and
   4. ``reset_arrays``
   is the last statement of this phase -- it can neither precede the grid
   generation it takes its size from, nor follow any ``circuit_track``.

   Two readers want a residual out of this phase, and they must not be
   confused. The coarse step into a segment's first cell reads it back
   from the segment list (``_preceding_segment_residual``), not from the
   live accumulator, which by the time the grid is walked already holds
   the last-generated (forward) segment's value; the first segment of a
   turn steps across the turn boundary and takes the residual the
   previous turn ended on. That live scalar survives only as the
   fall-back for a segment-less hand-built grid (tests, direct
   ``circuit_track`` callers): on a real per-turn grid a start index that
   is not a segment boundary trips an assertion instead of silently
   returning this turn's forward tail. The *demodulation frame* is the
   same tail, but it has to be snapshotted onto the span between the
   backfill and the forward generation, because the forward generation
   overwrites the live scalar; under ``validate_grid_each_turn`` an
   assertion ties the snapshot and the segment-list lookup together so
   they cannot silently drift apart.

4. ``_carrier_slip_gap`` is formed as the kick-clock gap of step 2 plus
   the forward segment's ``accumulated_phase`` ``phi_acc`` -- the
   multi-section grid-vs-carrier phase the segment records of step 3
   carry. Why that phase is referred to this station's *previous*
   passage's forward-segment design frequency, and why it is exactly
   ``+0.0`` for a single section, for an unaccelerated ring and on a
   station's very first passage, is under *Multi-section registration
   phase* below. ``_update_frame_rotations`` then derives the three
   frame rotations every later cell update reads:

   * the *generator* frame rotation ``exp(-i (delta_phi_rf + gap +
     phi_acc))``, which rotates the design-anchored generator component
     into the demodulation frame when the sum is composed;
   * the *kick* frame rotation ``exp(+i (gap + phi_acc))``, in which the
     PI error is formed;
   * the *actuator* (PI-error) rotation ``exp(+i delta_phi_rf)``, which
     takes that error into the frame the commanded generator current
     acts in. The ``gap`` and ``phi_acc`` halves cancel between the first
     two rotations, which is why this third one carries the station
     clock alone.

   The actuator rotation is one scalar per passage. The first two exist
   twice: as passage scalars with the forward segment's ``phi_acc``, which
   the forward span and the fine grid use, and as one value per backfill
   centre (``_backfill_generator_frame_rotations``,
   ``_backfill_kick_frame_rotations``) with the phase accumulated up to
   that centre in place of ``phi_acc`` (see *Multi-section registration
   phase*); ``_frame_rotations_of_cell`` / ``_frame_rotations_of_cells``
   hand each cell its own. The first two are exactly ``1 + 0j`` without
   an RF-frequency offset and without multi-section acceleration, on
   every cell; the third is exactly ``1 + 0j`` whenever ``delta_phi_rf``
   is zero, independently of ``gap`` and ``phi_acc``. Those paths
   therefore stay bit-identical.

5. ``_replay_backfill_span`` -- re-walks this passage's backfill segments
   with ``no_beam=True``, one ``circuit_track`` per backfill segment at
   that segment's own ``omega``, so that the envelope carries the
   already-elapsed interval forward. A passage that generated no backfill
   segments skips the replay entirely. It runs after step 4 because its
   cell updates compose the sum, and form the PI error, with the
   per-cell backfill rotations of step 4.

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
   ``-(phi_rf + _carrier_slip_gap)``, and re-bin the fine-grid charge
   onto the coarse cells charge-conservingly. That constant is the
   *full* station phase ``phi_rf = phi_rf_design + delta_phi_rf`` plus
   the kick-clock gap and the accumulated phase, not the slip alone:
   dropping ``phi_rf_design`` would rotate the beam-induced voltage by
   ``-phi_rf_design``, and at the ordinary above-transition
   ``phi_rf_design = pi`` that inverts the beam loading outright. The
   demodulation frame is the span's ``residual_from_backfill_span``
   (step 3), and ``_assert_demodulation_frame_aligned`` refuses the
   passage unless ``omega_c * dT`` is an odd multiple of ``pi`` (see
   *Known limitations*). Several further guards protect this path, all
   of them raising rather than correcting:

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

   Whether the window captures the whole beam is *not* checked here. It
   is a property of the profile's window rather than of this consumer,
   and ``ProfileBaseClass._warn_if_beam_not_captured`` warns about it
   once, at fill time -- before this step is reached, and for every
   consumer of a profile rather than for the feedback alone.

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
   ``exp(-i (delta_phi_rf + gap + phi_acc))`` -- nets to its design-clock
   phase. A driven, beam-free cavity on its setpoint therefore reads
   out ``phase_correction == 0`` exactly: at zero intensity the
   feedback is *guaranteed* phase-neutral, whatever the ramp and the
   section count (pinned at 1e-12 rad by
   ``TestDrivenFeedbackIsPhaseNeutralWithoutBeam``). These two arrays
   are what the parent RF station applies to its kick.

Coarse-grid cavity update
~~~~~~~~~~~~~~~~~~~~~~~~~

Inside ``circuit_track``. The antenna voltage is advanced cell by cell
with the exact exponential propagator of the cavity-envelope ODE
``dV/dt = lambda V + s``, ``lambda = -omega / (2 Q_L) + i delta_omega``,
for a source ``s = (R/Q) omega (I_gen - I_beam / 2)`` held constant over
the step: ``V[n+1] = e^L V[n] + s dt (e^L - 1) / L`` with
``L = lambda dt``, i.e. the decay/detuning multiplier ``e^L`` and the drive
weight ``(e^L - 1) / L`` on the per-step generator drive
``I_gen (R/Q) omega dt`` and beam loading ``-0.5 I_beam (R/Q) omega dt``.
The ODE is linear, so
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
kick-clock slip nor the accumulated phase (``initial_voltage`` and the
pre-fill seed this component -- they model a generator-established
field). The public ``antenna_voltage_coarse_grid`` remains the
DEMODULATION-FRAME SUM, composed per cell as
``V_beam + V_gen * exp(-i (delta_phi_rf + gap + phi_acc))``, with
``phi_acc`` the forward segment's on the forward span and the phase
accumulated up to the cell on a backfill cell (step 4 of *Signal path of
one turn*) -- a rotation that is exactly ``1 + 0j`` without an
RF-frequency offset and without multi-section acceleration, which is why
undriven runs stay byte-identical to the former single-state recursion.

.. note::

   The zero-order hold is taken from different cells for the two
   sources. The coarse recursion holds ``I_gen`` from cell ``c-1`` (the
   command issued one step earlier) but ``I_beam`` from cell ``c``
   itself, so the bunch's own slice enters with the drive weight
   ``(e^L - 1) / L = 1 + O(L)``, i.e. weight 1 to a few ``1e-6``. The
   three discretisations in the code therefore disagree on that
   self-slice weight: coarse ~ 1, fine first-order = 0, fine
   second-order = 1/2 --
   and 1/2 is the value the fundamental theorem of beam loading calls
   for. The difference does not move any published number (the coarse
   voltage never kicks the beam; the kicks come from the fine grid,
   whose second-order solver the shipped example enables), but it is a
   real asymmetry. Note the scope of that dismissal: the fine grid is
   *first* order -- weight 0 -- by default, and the 1/2 weight requires
   ``second_order_fine_grid_solver_enable=True``. The numbers are pinned
   bit-for-bit, so changing the weights, or the default, is a deliberate
   decision rather than a cleanup.

The derivation of this step -- and why the forward-Euler update
``V[n+1] = (1 + L) V[n] + s dt`` that BLonD 2's ``LHCCavityLoop`` used, and
that this class inherited as its default, is only its first-order truncation
-- is in the Notes of ``IQCavityFeedbackTimingClass._advance_coarse_voltage``.
In short: the exponential step is exact for a piecewise-constant source at
any step length, ``|e^L| <= 1`` for every step, and a pure detuning is a pure
rotation. The Euler step has an ``O(L^2)`` local error, grows ``|V|`` by
``sqrt(1 + (delta_omega dt)^2)`` per step under pure detuning, flips the sign
of its decay factor once ``omega dt / (2 Q_L) > 1`` and diverges once
``|1 + L| > 1``. The Euler step, the coarse-solver switch that selected the
exact one, and the forward-Euler validity guard that policed the Euler step
(per-step decay, detuning phase, multiplier magnitude and beam kick) were
removed on 2026-09-11. On the shipped parameters (``|L|`` of a few ``1e-6``
per step) the two steps differed by ~1e-6 relative in the tracked
beam-induced voltage, and the exact step costs the same, because its
multipliers are precomputed per cell.

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

Optional generator-current control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With a ``controller`` attached, each coarse step forms the error in the
KICK frame,
``V_set - V_ant[n] * exp(+i (gap + phi_acc[n]))`` -- the envelope of the kick
the station actually applies against ``phi_rf``, so the loop regulates
the applied voltage rather than a bookkeeping frame -- and then rotates
that error into the ACTUATOR frame by ``exp(+i delta_phi_rf)`` before
handing it to the controller: the controller drives the design-anchored
generator component, so ``d(V_kick)/d(I_gen)`` carries
``exp(-i delta_phi_rf)``, and rotating the error back cancels it and
keeps the open-loop gain real instead of turning it with the station
clock. All three rotations are exactly unity without an RF-frequency
offset and without multi-section acceleration, and the pure-Python path
and the numba kernel form them identically. The controller then produces
``I_gen[n]``, which drives the next step; without one, the generator
current stays at the constant feedforward value
``generator_current_bias``. The controller is stepped on **every** tracked
cell, the backfill reconstruction segments included: a real LLRF regulates
continuously, and a loop confined to the forward passage would be
open-loop for ``(N - 1) / N`` of every turn on an ``N``-section ring,
merely holding the current the forward pass last commanded (a 6 % duty
cycle on 16-section RCS1). The error on a backfill cell is formed with that
cell's own rotations, set before the backfill replay: ``phi_acc[n]`` is the
phase accumulated up to the cell (step 4 of *Signal path of one turn*), not
the passage's final phase. With the final phase there, as until 2026-09-11,
the beam-induced part of the carried voltage is rotated by the phase still
to accumulate, and the regulated kick-frame voltage jumps by one passage's
increment where the previous passage's forward span hands over to this
backfill span (0.14 and 0.21 rad at two and four sections on the fast test
ramp, a one-off measurement; the continuity is pinned by
``TestKickFrameVoltageIsContinuousAcrossBackfill``). ``reset_arrays``
seeds the backfill span of the generator grid with the *last commanded*
current
rather than the feedforward bias: that is the loop's initial condition for
the span it then regulates over, since those cells replay an interval that
began with the generator running at whatever it was last told. Resetting
them to the bias was a real defect, not a cosmetic one:
with a detuned cavity the PI holds a reactive standing current, which the
old reset discarded once per turn (setpoint errors of 3.1e-2 and 4.6e-2
relative at 2 and 4 sections -- a one-off measurement taken when the
defect was found, not a regression-guarded number). Without a controller
the held value *is* the bias, so the constant-current path is
bit-unchanged. The klystron
limit is enforced on the fine grid as well before the response solve.

Fine-grid solve
~~~~~~~~~~~~~~~

In ``_resolve_fine_grid_voltage`` the generator current is interpolated
onto the profile grid and the cavity response is solved as
a sparse bidiagonal system -- first order by default, or the second-order
(Crank-Nicolson) solver with ``second_order_fine_grid_solver_enable=True``,
whose truncation error scales with the bin size squared. The result is
scaled by ``n_cavities`` before the readout phase converts it into the
voltage correction and phase correction the parent RF station applies to
its kick. The initial condition it starts from is described next.

The passage handoff uses ``initial_at_bin_edge=True``: the initial voltage
is at ``profile.cut_left`` and the first output is at
``profile.hist_x[0] = cut_left + hist_step / 2``. The first step therefore
spans half a bin. Beam current is a bin density, so this first sample
includes half the first bin's charge; each later centre-to-centre step
includes half of each adjacent bin. The first-order fine-grid solver
retains forward-Euler stepping for voltage decay and generator drive.
Direct calls to the sparse solvers or ``cavity_response_fine`` retain the
uniform-step convention unless this keyword is explicitly enabled.


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
coarse-grid step. The fill is evaluated on the **design** clock
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

The second is that only the coarse voltage *at index* ``[0]`` enters the
initial condition. That cell is charge-free by construction
(``forbid_charge_in_first_coarse_cell``). Later coarse cells can already
contain this passage's beam loading: interpolating their voltages would
introduce charge before its arrival and then count it again in the fine
solve.

The seed's timestamp must also be respected. Before integrating the
profile, ``propagate_beam_free_voltage`` advances the seed from the first
forward centre to ``cut_left`` with zero new beam current. It includes
decay, detuning and the recorded generator commands, held from each coarse
centre to the next as in the coarse recursion. For a constant command
over an interval :math:`h`, it evaluates

.. math::

   V(t+h) = e^{\lambda h} V(t)
     + (R/Q)\,\omega I_{\mathrm{gen}}
       \frac{e^{\lambda h}-1}{\lambda},
   \qquad \lambda = -\frac{\omega}{2Q_L} + i\Delta\omega.

The drive weight uses ``expm1`` and its finite zero-exponent limit. Current
is actuator-limited and rotated into the seed's IQ frame before this
propagation. The controller is not stepped again, and cavity-count scaling
is applied only to the final fine-grid voltage. Thus a later profile
window includes the elapsed empty interval instead of restarting the
cavity clock at the old voltage. Within the fine window, generator current
retains the interpolated representation described above.

An empty diagnostic window may precede the first centre; the same
beam-free equation then evolves backward with the first available command
held constant. A charged window in that position remains rejected.


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

The sub-stepping mode (``n_rf_periods_per_coarse_grid < 1``) subdivides the
RF period, with the coarse centres tiling continuously across turn
boundaries. It is not a stability device -- the exact coarse step holds for
any step length and ``Q_L`` -- but a finer sampling of the held generator
command, of the controller and of the coarse beam current. ``n = 0.5`` is
the only usable sub-step: that tiling makes the demodulation frame one
previous coarse step, ``omega_c * dT = 2 pi n``, which is an odd multiple
of ``pi`` only there, so ``n = 0.25`` or ``0.9`` is rejected by
``_assert_demodulation_frame_aligned`` (see *Known limitations*).

Multi-section registration phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ring with several RF stations builds each passage's grid piecewise:
the backfill segments reconstruct the interval since *this station's
previous passage*, every segment ``k``
spanning ``T_seg,k`` at the past station's design frequency ``omega_k``,
while the envelope carried across that interval was demodulated against
a single carrier -- the one in force when the interval STARTED, i.e. the
forward-segment design frequency ``omega_prev`` of the previous passage.
The grid therefore accumulates ``sum_k omega_k T_seg,k`` where the
carried envelope's frame accumulated ``omega_prev T_total``, and over
backfill segment ``k`` the two part by

   ``(omega_prev - omega_k) T_seg,k``.

Every segment record stores the running sum of those differences up to
its end, continued from the forward segment the previous passage ended
on, as ``RFCenterSegment.accumulated_phase`` (``phi_acc``); the forward
segment of a passage inherits the value of its last backfill segment.
That forward value is the phase of the demodulation, the readout, the
forward span and the fine grid. The backfill span replays an interval
over which the phase is still accumulating, so a backfill cell takes the
phase accumulated up to its own centre instead: a centre at
segment-local time ``c`` of backfill segment ``k`` gets
``phi_start,k + (omega_prev - omega_k) c``, with ``phi_start,0`` the
carried forward segment's phase and ``phi_start,k`` the stored phase of
segment ``k - 1`` (``accumulated_phases_at_centers`` in
``rf_center_segment.py``, read off the grid by
``RFCenterGridMixin._backfill_center_phases``). The phase is a pure
bookkeeping mismatch -- identically zero for a single section, which is why
single-section rings need no correction at all, and zero on a station's
first passage, which has no previous carrier. The reference is the
previous passage's carrier and not the current one because the quantity
being corrected is an envelope that already exists: it was demodulated
before the interval, and nothing that happened during the interval can
retroactively change the frame it was written in. It is
*separate* from the cavity resonance detuning ``delta_omega``, whose
physical precession the coarse recursion already applies on every step.
The phase is carried as an explicit *carrier* phase, exactly the idiom
the RF-frequency offset above uses: subtracted at demodulation
(``carrier_phase_offset``) and added back at readout
(``phase_correction``). It is deliberately *not* applied as a rotation of
the antenna-voltage state -- that would also rotate the generator-driven
field, which carries no registration error, turning a phase error into an
amplitude drift. See ``accumulated_phases`` in ``rf_center_segment.py``
and ``RFCenterGridMixin._backfill_accumulated_phases`` for the
implementation. Until 2026-09-11 the feedback kept this phase as a
separate running total with its own copy of the previous carrier;
storing it on the segment records removed that parallel bookkeeping
without changing any result. The per-centre phases of the backfill cells
date from the same day and did change results: before, every backfill
cell was composed and regulated with the forward segment's phase (see
*Optional generator-current control*).

The reference-choice regression uses a curved frequency programme; see
``test_phase_refers_to_the_previous_carrier``. The historical diagnostics
and retired approaches are recorded in the development log
``MUCOL_FEEDBACK_CONTEXT.md`` in the BLonD repository root.

The source-split coarse state lets the accumulated phase affect only the
beam-sourced
component. The generator component remains design-anchored, and the PI
regulates their kick-frame sum. The equilibrium checks are
``TestDrivenSteadyStateFastRamp`` and
``TestDrivenFeedbackIsPhaseNeutralWithoutBeam``.

One reading rule follows for driven runs: ``antenna_voltage_coarse_grid``
is the demodulation-frame sum, so under an accumulated slip its
*generator-sourced* part appears rotated by minus that slip. A
beam-free driven run is therefore a pure rotation at constant ``|V|``,
but a run with beam loading is not -- the beam component carries no such
rotation, so the magnitude moves too. Either way a naive complex
comparison against the setpoint is the wrong check; compare in the kick
frame, as the PI does.

A related frame rule applies to the reflected current
``V_ant / ((R/Q) Q_L) - r_gen I_gen``: the design-frame generator current
has to be rotated into the frame each cell was composed in.
``GeneratorRegulationMixin.reflected_current`` and ``reflected_power`` do
that by default -- per cell for this passage's coarse-grid antenna voltage
(the default argument, or that very array), with the passage's rotation
for any other antenna voltage -- and
``FullTurnCavityObservation.reflected_current_coarse`` records that
default. With the passage's rotation on a backfill cell a driven,
beam-free cavity would appear to reflect
``|2 exp(i delta) - 1|^2 ~ 1 + 2 delta^2`` of its forward power, ``delta``
being the phase still to accumulate there.

The long-horizon carried-wake comparison is covered by
``test_multiturn_secular_drift_long_horizon``. Its assertions state the
allowed endpoint error and slope; past measurements are retained in the
development history rather than presented as current performance claims.

Multi-harmonic stations
~~~~~~~~~~~~~~~~~~~~~~~

The feedback is not restricted to the main harmonic: it can be attached
to a :class:`~blond.physics.cavities.MultiHarmonicRFStation`, and the
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
same-sign *currents*.) For a mode with ``R_CR = -R`` (see below) their
loading then adds constructively and both receive the same kick. A
counter-rotating mu- beam alone reproduces the co-rotating mu+ run
bit-for-bit, through the feedback and through the convolution reference
alike.

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
turn at ``N = 4`` -- and they are validated too, against the same 0.5 %
gate: on the static cycle four and six sections match the two-beam
convolution to 0.128 % on the first turn, falling to 0.039 %, within
0.001 percentage points of the two-section numbers. The accelerating
fast ramp and the ``delta_omega_rf`` regimes are carried to four
sections only, where the tests bound the error and its accumulated growth.
That matters because two sections is also the only count
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

The following comparisons constrain the error for their tested parameters
(see :ref:`mucol_cavity_feedback_tests` for configurations and tolerances).
Agreement between backends checks implementation parity; it cannot detect
an assumption shared by both. Analytic decay, driven equilibrium, localized
charge and physical-time handoff checks provide complementary references.
An error bound belongs to its tested ramp, geometry, binning and solver
options, not to every use of the model:

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

* Coarse charge re-binning currently splits at the last fine index of the
  preceding cell rather than after it. This shifts one boundary slice
  into the next cell while preserving total charge. Per-cell timing needs
  a separate correction and boundary-bin regression checks; global charge
  conservation alone does not establish correct coarse beam loading.
* ``MultiPassResonatorSolver(retune_to_rf=True)`` warns during initialization
  if there is more than one resonance frequency across its sources. Only
  the first resonance is retuned, but its carried-wake phase correction
  acts on the sum of all modes. The warning is advisory: execution
  continues with that limitation. Use separate solvers for the retuned
  fundamental and fixed-frequency modes when their phases must be correct.

* A harmonic number that is not divisible by ``2 * n_sections`` de-aligns
  the coarse-grid tiling from the RF bucket. **Every such case is
  refused**: ``_assert_demodulation_frame_aligned``
  (``cavity_feedback.py``, called unconditionally before every coarse
  demodulation) raises a ``ValueError`` whenever ``omega_c * dT`` is more
  than ``1e-3 pi`` from an odd multiple of ``pi``. Its only gate is
  whether the demodulation is observable at all -- ``R_over_Q != 0`` and
  a populated profile histogram -- so that the inert frames of pure
  grid-geometry fixtures are not rejected. This is a *configuration*
  limitation, not an uncaught defect: the run stops, it does not produce
  a wrong number.

  Why it has to be refused. The grid seeds every segment half an RF
  period in, so a segment spanning a fractional number of RF periods
  leaves a residual different from ``t_rf / 2``; that residual is the
  demodulation frame ``dT``. The implemented mixing and kick convention
  needs ``omega * dT = pi`` (mod ``2 pi``) to reproduce the physical
  beam-loading phase. Other geometries require a different frame treatment;
  the physical theorem itself does not impose this grid alignment.
  Which fraction it is decides
  what the run *would* have done:

  - ``1/4`` and ``3/4`` of a period leave ``omega * dT`` a quarter turn
    off ``pi``, which rotates the beam-induced voltage by that angle.
    Such a geometry also pushes beam charge into the first coarse cell,
    which ``rf_beam_current`` refuses on its own -- but the
    demodulation-frame guard catches it one step earlier, and by root
    cause rather than by symptom. That refusal is pinned as a contract
    by ``test_multiturn_nondivisible_harmonic_is_rejected`` in the
    multi-turn comparison suite, which asserts the ``ValueError`` and
    that its message still names ``omega_c * dT`` and the divisibility
    cause.
  - ``1/2`` of a period (``harmonic % (2 * n_sections) == n_sections``,
    which includes every odd harmonic on a one-station ring) gives
    ``omega * dT = 2 pi``, i.e. the demodulation factor would be ``+1``
    where it must be ``-1``: the beam-induced voltage would be
    sign-inverted, the bunch would be **accelerated by its own wake**,
    and the wrongly signed deposit would then decay only over
    ``2 Q_L / omega``, i.e. over many turns. The guard raises before any
    voltage is produced. (Historically this case was unguarded and did
    complete, at 199.9 % relative error against the
    ``MultiPassResonatorSolver`` on the first turn. That is what the
    guard now prevents, not what it currently does.)

  To satisfy the guard, choose ``harmonic % (2 * n_sections) == 0`` for
  the symmetric half-drift / station / half-drift layout, and more
  generally make every stretch between the ring start and an RF station,
  and between two consecutive RF stations, span a whole number of RF
  periods. ``muon_collider_blonder.rcs_two_beam_example`` does this
  itself, reducing the JSON harmonic to a multiple of
  ``2 * n_sections``.

  Sub-stepped grids (``n_rf_periods_per_coarse_grid < 1``) do not re-seed
  at the bucket phase: they tile continuously across segment boundaries,
  so ``dT`` is one previous coarse step by construction. That step is
  ``omega_c * dT = 2 pi n``, an odd multiple of ``pi`` only at
  ``n = 0.5`` -- so ``0.5`` is the only usable sub-step, and ``n = 0.25``
  or ``0.9`` is rejected by the same guard
  (``TestDemodulationFrameGuard.test_misaligned_sub_step_is_rejected``
  pins that raise).
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
* A bounded multi-turn residual remains on the undriven multi-section
  fast ramp. The per-turn error against the convolution is *negative*
  (converging toward flat, not accumulating) at every section count, and
  its endpoint after 20 turns grows only weakly with that count:
  ``0.02149``, ``0.02816``, ``0.03621`` and ``0.04169 %`` at 2, 4, 8 and
  16 sections -- one-off measurements, of which only the two-section and
  single-section figures are regression-guarded (by
  ``test_multiturn_secular_drift_long_horizon``, gates slope
  ``< 0.005`` pp/turn and endpoint ``< 0.05 %``). At two sections the
  endpoint already sits *below* the single-section control, so what is
  left is the generic multi-turn discretisation residual, with no
  registration artefact to attribute it to. The secular drift this
  replaced, and the reference fix that removed it, are described under
  *Multi-section registration phase*.
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
  rather than interpolating later beam-loaded voltages avoids double
  counting. The seed is then evolved to the profile edge through the
  beam-free interval; retaining its old value at a later time is incorrect.
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
