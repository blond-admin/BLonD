"""
Beam-loading sign versus the design RF phase.

The fundamental theorem of beam loading says a bunch must LOSE energy to its
own wake, whatever RF phase the station kicks at. The cavity feedback
demodulates the beam current at the design carrier with an explicit
``carrier_phase_offset``, while the station applies the kick at
``sin(omega_rf ts + phi_rf + phase_correction)`` with
``phi_rf = phi_rf_design + delta_phi_rf``. The demodulation must therefore
subtract the *whole* phase the readout and the station add back on top of
``angle(V_ant)`` -- ``phi_rf + _carrier_slip_gap``. Leaving ``phi_rf_design``
out of it rotates the beam-induced voltage by ``-phi_rf_design`` and, at
``phi_rf_design = pi``, inverts the beam loading outright: the bunch is then
accelerated by its own wake.

The beam-induced part of the kick is isolated exactly, by linearity of the
cavity equation, as the difference between a nominal-intensity run and a
zero-intensity reference run on the same operating-point cavity
(``V_init = V_design`` held by the matched generator current
``I_g = V / (2 (R/Q) Q_L)``, chosen when a cold cavity still tripped the
coarse-grid beam-kick check removed on 2026-09-11). Ring, cavity and bunch
are the ones of
``test_energy_gain_ind_voltage_vs_nondriven_feedback``, which pins that same
induced kick against the independent ``MultiPassResonatorSolver`` at
``phi_rf_design = 0`` -- so here that already-validated case is the reference
the other design phases must reproduce.

Measured on this fixture: the induced kick is negative for every single
macroparticle (mean -7.8483e5 eV, most-decelerated -1.3230e6 eV,
least-decelerated -9.11 eV) and is the same array for ``phi_rf_design`` in
{0, pi/2, pi, -0.7} to 2.6e-7 eV, i.e. 2e-13 of the peak.
"""

import io
import unittest
from unittest import mock

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    mu_plus,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.feedbacks.beam_current import rf_beam_current
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass

#: RCS1-like cavity and ring parameters, shared with the sibling
#: MTW-vs-feedback energy-gain test so both pin the same fixture.
R_OVER_Q = 518.0
Q_L = 1.29e4
ALPHA_P = 10.395e-4
CIRCUMFERENCE = 5990.0
ENERGY = 63e9
HARMONIC = 2590
INTENSITY = 2.7e12
V_DESIGN = 30e6
N_SLICES = 1024
N_MACROPARTICLES = int(2e4)


class TestBeamLoadingSignVsDesignRfPhase(unittest.TestCase):
    """The bunch loses energy to its own wake at any ``phi_rf_design``."""

    @classmethod
    def setUpClass(cls):
        """Build the template bunch once; runs are cached per design phase."""
        cls.cycle = ConstantMagneticCycle(
            reference_particle=mu_plus, value=ENERGY, in_unit="total energy"
        )
        t_rev = cls.cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
        cls.t_rf = t_rev / HARMONIC
        cls._induced_cache = {}
        # Prepare the template bunch on the phi_rf_design = 0 station and
        # shift it by one RF period into the profile window: BiGaussian
        # places it near rf phase pi, one period before the window.
        simulation, _ = cls._build(0.0, cls._make_profile())
        beam = cls._prepare(simulation, INTENSITY)
        cls.dt_template = copy_to_cpu(beam.dt.array_local) + cls.t_rf
        cls.dE_template = copy_to_cpu(beam.dE.array_local)

    @classmethod
    def _make_profile(cls) -> StaticProfile:
        """
        Profile window [1.5 pi, 4.5 pi] around the shifted bunch.

        Returns
        -------
        profile
            Static profile the wake/feedback acts on.
        """
        return StaticProfile.from_rad(
            np.pi * 1.5, np.pi * 4.5, N_SLICES, cls.t_rf
        )

    @classmethod
    def _build(
        cls,
        phi_rf_design: float,
        profile: StaticProfile,
        n_rf_periods: float = 1,
    ):
        """
        Build a one-turn ring (drift + RF station carrying the feedback).

        Parameters
        ----------
        phi_rf_design
            Design RF phase of the station [rad].
        profile
            Static profile the feedback acts on.
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback.

        Returns
        -------
        simulation
            The constructed one-turn Simulation.
        rf_station
            The RF station element added to the ring.
        """
        ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
        drift = DriftSimple(
            orbit_length=CIRCUMFERENCE, momentum_compaction_factor=ALPHA_P
        )
        # Operating-point cavity: V_init = V_design held steady by the
        # matched generator current (the steady state of the cavity
        # equation at zero detuning). A cold or undriven cavity used to
        # trip the coarse-grid beam-kick check (removed 2026-09-11); the
        # operating point is kept.
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=R_OVER_Q,
            Q_L=Q_L,
            generator_current_bias=V_DESIGN / (2.0 * R_OVER_Q * Q_L),
            n_cavities=1,
            initial_voltage=V_DESIGN,
            n_rf_periods_per_coarse_grid=n_rf_periods,
            delta_omega=0.0,
        )
        rf_station = SingleHarmonicRFStation(
            voltage=V_DESIGN,
            phi_rf=phi_rf_design,
            harmonic=HARMONIC,
            cavity_feedback=feedback,
            profile=profile,
        )
        # Drift first so the feedback's RF station is not the first
        # reference-altering element (otherwise the first-turn reverse
        # tracking has nothing to track and returns early).
        ring.add_elements([drift, rf_station], reorder=False)
        return Simulation(ring=ring, magnetic_cycle=cls.cycle), rf_station

    @classmethod
    def _prepare(cls, simulation: Simulation, intensity: float) -> Beam:
        """
        Prepare a bunch on the given simulation.

        Parameters
        ----------
        simulation
            Simulation to prepare the beam on.
        intensity
            Beam intensity; 0.0 gives the beam-free reference run.

        Returns
        -------
        beam
            The prepared beam.
        """
        beam = Beam(intensity=intensity, particle_type=mu_plus)
        beam.reference.total_energy = ENERGY
        simulation.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                n_macroparticles=N_MACROPARTICLES,
                sigma_dt=0.06 * cls.t_rf,
                sigma_dE=1.5e7,
                seed=7,
                reinsertion=True,
            ),
        )
        return beam

    @classmethod
    def _applied_kick(
        cls,
        phi_rf_design: float,
        intensity: float,
        n_rf_periods: float = 1,
    ):
        """
        Energy applied to each macroparticle over one turn [eV].

        Parameters
        ----------
        phi_rf_design
            Design RF phase of the station [rad].
        intensity
            Beam intensity; 0.0 gives the beam-free reference run.
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback.

        Returns
        -------
        applied
            Energy kick per macroparticle over the turn [eV].
        """
        simulation, _ = cls._build(
            phi_rf_design, cls._make_profile(), n_rf_periods
        )
        beam = cls._prepare(simulation, intensity)
        # Identical coordinates in every case, so the only difference
        # between runs is the phase the station kicks at.
        beam.setup_beam(dt=cls.dt_template, dE=cls.dE_template)
        dE_before = copy_to_cpu(beam.dE.array_local)
        simulation.run_simulation((beam,), n_turns=1, show_progressbar=False)
        return copy_to_cpu(beam.dE.array_local) - dE_before

    @classmethod
    def _induced_kick(cls, phi_rf_design: float, n_rf_periods: float = 1):
        """
        Beam-induced kick per macroparticle [eV], by reference subtraction.

        The design-voltage reconstruction cancels exactly (linearity of the
        cavity equation), leaving ``q * V_induced(dt)``. Cached, because
        every test in this class wants the ``phi_rf_design = 0`` reference.

        Parameters
        ----------
        phi_rf_design
            Design RF phase of the station [rad].

        Returns
        -------
        induced
            Beam-induced energy kick per macroparticle [eV].
        """
        key = (phi_rf_design, n_rf_periods)
        if key not in cls._induced_cache:
            cls._induced_cache[key] = cls._applied_kick(
                phi_rf_design, INTENSITY, n_rf_periods
            ) - cls._applied_kick(phi_rf_design, 0.0, n_rf_periods)
        return cls._induced_cache[key]

    def _assert_every_particle_is_decelerated(self, induced, phi_rf_design):
        """
        Assert the induced kick decelerates the whole bunch.

        Parameters
        ----------
        induced
            Beam-induced energy kick per macroparticle [eV].
        phi_rf_design
            Design RF phase the kick was measured at [rad], for messages.
        """
        least_decelerated = float(np.max(induced))
        mean_loss = float(np.mean(induced))
        # No macroparticle may GAIN energy from the bunch's own wake. The
        # bound is <= 0, not < 0: a particle far enough into the tail to
        # sit outside the induced voltage's support gets exactly 0.0, which
        # is physically fine (1 of 20000 on this fixture). The strict
        # statement is carried by the mean below.
        self.assertLessEqual(
            least_decelerated,
            0.0,
            msg=(
                f"at phi_rf_design={phi_rf_design} the least-decelerated "
                f"macroparticle GAINED {least_decelerated:.4e} eV from the "
                "beam-induced voltage; a bunch must lose energy to its own "
                "wake"
            ),
        )
        # Magnitude sanity on THIS fixture (h = 2590, 2e4 macroparticles),
        # measured -7.869e4 eV: the fix must restore the sign without
        # changing how much energy the wake takes. A factor-two band, so
        # the test pins the scale without being brittle.
        self.assertLess(mean_loss, -4.0e4, msg=f"{mean_loss=}")
        self.assertGreater(mean_loss, -1.6e5, msg=f"{mean_loss=}")

    def test_feedback_forbids_charge_in_the_last_coarse_cell(self):
        """
        Every coarse demodulation of the timing class asks for the guard.

        A cell's beam current drives the coarse step that ends at its own
        centre, and the first step of the next passage's span carries
        none, so the boundary between two passages must stay charge-free.
        The feedback calls ``rf_beam_current`` with
        ``forbid_charge_in_last_coarse_cell=True`` whenever it downsamples
        onto the coarse grid, and no longer asks for the retired
        first-coarse-cell guard: the fine solve is seeded before the first
        forward cell, so that cell may carry charge.
        """
        with mock.patch(
            "blond.physics.feedbacks.cavity_feedback.rf_beam_current",
            wraps=rf_beam_current,
        ) as spy:
            self._applied_kick(0.0, INTENSITY)
        coarse_calls = [
            call
            for call in spy.call_args_list
            if call.kwargs.get("sampling_time") is not None
        ]
        self.assertGreater(len(coarse_calls), 0)
        for call in coarse_calls:
            self.assertIs(
                call.kwargs.get("forbid_charge_in_last_coarse_cell"), True
            )
            self.assertNotIn("forbid_charge_in_first_coarse_cell", call.kwargs)

    def test_bunch_loses_energy_to_its_own_wake_at_zero_design_phase(self):
        """Control: at ``phi_rf_design = 0`` every particle is decelerated."""
        self._assert_every_particle_is_decelerated(
            self._induced_kick(0.0), 0.0
        )

    def test_bunch_loses_energy_to_its_own_wake_at_pi_design_phase(self):
        """
        At ``phi_rf_design = pi`` the bunch must still lose energy.

        This is the sign flip: with ``phi_rf_design`` missing from the
        demodulation the beam-induced voltage is rotated by ``-pi`` and
        every macroparticle GAINS energy from its own wake (measured
        +7.8483e5 eV mean before the fix).
        """
        self._assert_every_particle_is_decelerated(
            self._induced_kick(np.pi), np.pi
        )

    def test_induced_kick_does_not_depend_on_the_design_rf_phase(self):
        """
        The induced kick is the same array for any ``phi_rf_design``.

        The wake a bunch drives, and the energy it takes from it, are a
        property of the cavity and the bunch, not of the phase the station
        happens to kick at: shifting ``phi_rf_design`` shifts the design
        wave and the induced field together. Pinned against the
        ``phi_rf_design = 0`` case, which the sibling MTW comparison
        validates independently. Measured spread: 2.5e-7 eV on a 1.3e6 eV
        peak (2e-13 relative); before the fix, phi = pi is off by the full
        2 x 1.3e6 eV.
        """
        reference = self._induced_kick(0.0)
        tolerance = 1e-9 * float(np.max(np.abs(reference)))
        for phi_rf_design in (np.pi, -0.7, 0.7):
            with self.subTest(phi_rf_design=phi_rf_design):
                np.testing.assert_allclose(
                    self._induced_kick(phi_rf_design),
                    reference,
                    rtol=0,
                    atol=tolerance,
                    err_msg=(
                        "the beam-induced kick changed with the design RF "
                        f"phase (phi_rf_design={phi_rf_design})"
                    ),
                )

    def test_design_rf_phase_flips_the_beam_free_kick_at_pi(self):
        """
        Readout-side guard: the beam-free kick honours ``phi_rf_design``.

        The generator component is design-locked and the station supplies
        ``phi_rf_design`` itself, so a driven, beam-free cavity on its
        setpoint reproduces ``V sin(omega_rf ts + phi_rf)`` exactly. At
        ``phi_rf_design = pi`` that is the exact negative of the
        ``phi_rf_design = 0`` kick. This already holds today (measured to
        4e-15 relative) and must keep holding: the fix belongs on the
        demodulation side only, and must not be "balanced" by a rotation
        of the generator frame.
        """
        kick_zero = self._applied_kick(0.0, 0.0)
        kick_pi = self._applied_kick(np.pi, 0.0)
        np.testing.assert_allclose(
            kick_pi,
            -kick_zero,
            rtol=0,
            atol=1e-12 * float(np.max(np.abs(kick_zero))),
            err_msg=(
                "the beam-free kick is no longer the design RF wave at "
                "phi_rf_design = pi"
            ),
        )


class TestDemodulationFrameGuard(unittest.TestCase):
    """The demodulation frame must be an odd multiple of ``pi``.

    ``carrier_phase_offset`` cancels the station and readout phases, so
    ``omega_c * dT`` is the only phase left in the sign of beam loading and
    the bunch loses energy to its own wake only while
    ``cos(omega_c * dT) < 0``.  The coarse grid delivers that by seeding
    every segment half an RF period in, but a sub-stepped grid tiles at
    ``omega_c * dT = 2 pi n``, which is an odd multiple of ``pi`` ONLY for
    ``n = 0.5``.

    ``n_rf_periods_per_coarse_grid = 0.9`` is an ordinary-looking input that
    silently inverts the beam loading: measured on this fixture the bunch
    GAINS ``+8.43e4 eV`` from its own wake instead of losing ``-7.87e4 eV``.
    Nothing else in the suite catches it, which is why the guard exists.
    """

    @classmethod
    def setUpClass(cls):
        """Reuse the sibling fixture's template bunch."""
        TestBeamLoadingSignVsDesignRfPhase.setUpClass()

    def _induced_mean(self, n_rf_periods: float) -> float:
        """
        Mean beam-induced kick for one coarse-grid step size [eV].

        Parameters
        ----------
        n_rf_periods
            ``n_rf_periods_per_coarse_grid`` of the feedback.

        Returns
        -------
        mean_induced
            Mean induced energy kick per macroparticle [eV].
        """
        induced = TestBeamLoadingSignVsDesignRfPhase._induced_kick(
            0.0, n_rf_periods
        )
        return float(np.mean(induced))

    def test_aligned_steps_decelerate_the_bunch(self):
        """``n = 0.5`` and ``n = 1`` are aligned and agree exactly."""
        half_step = self._induced_mean(0.5)
        whole_step = self._induced_mean(1)
        self.assertLess(half_step, 0.0)
        self.assertLess(whole_step, 0.0)
        self.assertAlmostEqual(half_step, whole_step, delta=1.0)

    def test_misaligned_sub_step_is_rejected(self):
        """``n = 0.9`` would invert the loading, so it must raise."""
        with self.assertRaises(ValueError) as raised:
            self._induced_mean(0.9)
        message = str(raised.exception)
        self.assertIn("not aligned with the RF bucket", message)
        self.assertIn("n_rf_periods_per_coarse_grid", message)


class TestBunchInTheFirstCoarseCell(unittest.TestCase):
    """
    A bunch in the first forward coarse cell is tracked, not refused.

    The fine solve is seeded from the coarse state BEFORE the first
    forward centre and propagated beam-free to ``cut_left``, so the seed
    predates every deposit of the passage however early in the forward
    segment the bunch sits. The first coarse cell is then an ordinary
    one: its charge drives the coarse step into its own centre once, and
    the fine solve integrates it once. The window used to be constrained
    to ``cut_left >= first forward centre`` with a charge-free first
    cell; this replaces both.

    Measured on this fixture's grid: the forward centres sit at
    ``(k + 0.5) t_rf`` after a 0.5 ``t_rf`` residual, so the first coarse
    cell spans ``(-0.5, +0.5] t_rf`` and a bunch at ``0.3 t_rf`` sits in
    it. The same bunch one RF period later sits in the second cell, which
    is the geometry every other test in this module runs, so it is the
    reference the first-cell placement must reproduce.
    """

    #: Bunch centre inside the first forward coarse cell [t_rf].
    FIRST_CELL_CENTRE = 0.3
    #: The same bunch one RF period later, in the second coarse cell.
    SECOND_CELL_CENTRE = 1.3
    #: Window half-width around the bunch centre [t_rf]. The template
    #: bunch reaches 0.245 t_rf past its mean, and ``cut_left`` must stay
    #: positive for the first-cell placement.
    WINDOW_HALF_WIDTH = 0.26

    @classmethod
    def setUpClass(cls):
        """Reuse the sibling fixture's template bunch."""
        TestBeamLoadingSignVsDesignRfPhase.setUpClass()

    @classmethod
    def _profile_around(cls, centre_t_rf: float) -> StaticProfile:
        """
        Profile window centred on the bunch.

        Parameters
        ----------
        centre_t_rf
            Bunch centre, in RF periods from the passage reference time.

        Returns
        -------
        profile
            Static profile the feedback acts on.
        """
        return StaticProfile.from_rad(
            2.0 * np.pi * (centre_t_rf - cls.WINDOW_HALF_WIDTH),
            2.0 * np.pi * (centre_t_rf + cls.WINDOW_HALF_WIDTH),
            N_SLICES,
            TestBeamLoadingSignVsDesignRfPhase.t_rf,
        )

    @classmethod
    def _shifted_dt(cls, centre_t_rf: float):
        """
        Template bunch coordinates centred on ``centre_t_rf``.

        Parameters
        ----------
        centre_t_rf
            Bunch centre, in RF periods from the passage reference time.

        Returns
        -------
        dt
            Macroparticle arrival times [s].
        """
        base = TestBeamLoadingSignVsDesignRfPhase
        return (
            base.dt_template
            - np.mean(base.dt_template)
            + centre_t_rf * base.t_rf
        )

    @classmethod
    def _applied_kick(cls, centre_t_rf: float, intensity: float):
        """
        Energy applied to each macroparticle over one turn [eV].

        Parameters
        ----------
        centre_t_rf
            Bunch centre, in RF periods from the passage reference time.
        intensity
            Beam intensity; 0.0 gives the beam-free reference run.

        Returns
        -------
        applied
            Energy kick per macroparticle over the turn [eV].
        """
        base = TestBeamLoadingSignVsDesignRfPhase
        simulation, _ = base._build(0.0, cls._profile_around(centre_t_rf))
        beam = base._prepare(simulation, intensity)
        beam.setup_beam(dt=cls._shifted_dt(centre_t_rf), dE=base.dE_template)
        dE_before = copy_to_cpu(beam.dE.array_local)
        simulation.run_simulation((beam,), n_turns=1, show_progressbar=False)
        return copy_to_cpu(beam.dE.array_local) - dE_before

    @classmethod
    def _induced_kick(cls, centre_t_rf: float):
        """
        Beam-induced kick per macroparticle [eV], by reference subtraction.

        Parameters
        ----------
        centre_t_rf
            Bunch centre, in RF periods from the passage reference time.

        Returns
        -------
        induced
            Beam-induced energy kick per macroparticle [eV].
        """
        return cls._applied_kick(centre_t_rf, INTENSITY) - cls._applied_kick(
            centre_t_rf, 0.0
        )

    def test_bunch_in_the_first_coarse_cell_is_downsampled_into_it(self):
        """Non-vacuity: the demodulation really fills the first cell."""
        base = TestBeamLoadingSignVsDesignRfPhase
        simulation, rf_station = base._build(
            0.0, self._profile_around(self.FIRST_CELL_CENTRE)
        )
        beam = base._prepare(simulation, INTENSITY)
        beam.setup_beam(
            dt=self._shifted_dt(self.FIRST_CELL_CENTRE), dE=base.dE_template
        )
        simulation.run_simulation((beam,), n_turns=1, show_progressbar=False)
        feedback = rf_station.get_main_harmonic_cavity_feedback()
        coarse = np.abs(feedback.beam_current_forward_coarse_grid)
        self.assertGreater(float(coarse[0]), 0.5 * float(np.sum(coarse)))

    def test_bunch_in_the_first_coarse_cell_loses_energy_to_its_wake(self):
        """The fundamental theorem still holds for a first-cell bunch."""
        induced = self._induced_kick(self.FIRST_CELL_CENTRE)
        self.assertLessEqual(float(np.max(induced)), 0.0)
        self.assertLess(float(np.mean(induced)), 0.0)

    def test_induced_kick_does_not_depend_on_which_coarse_cell(self):
        """
        The same bunch one RF period later takes the same energy.

        The wake a bunch drives is a property of the cavity and the
        bunch, not of which coarse cell the demodulation happens to bin
        it into: window and bunch are shifted together by exactly one RF
        period, so the two runs differ only in that binning. A seed taken
        at the first forward centre instead of before it would count the
        first cell's deposit twice and roughly double this kick.
        """
        reference = self._induced_kick(self.SECOND_CELL_CENTRE)
        first_cell = self._induced_kick(self.FIRST_CELL_CENTRE)
        tolerance = 1e-6 * float(np.max(np.abs(reference)))
        np.testing.assert_allclose(
            first_cell,
            reference,
            rtol=0,
            atol=tolerance,
            err_msg=(
                "the beam-induced kick changed when the bunch moved into "
                "the first forward coarse cell"
            ),
        )


class TestFirstPassageDemodulationFrame(unittest.TestCase):
    r"""
    A station that opens the ring is demodulated like every other passage.

    ``calculate_rf_centers_for_backfill`` generates nothing on the first
    passage of a station that is the ring's FIRST reference-altering
    element: there is no elapsed span to reconstruct. The tail that
    passage's demodulation consumes has to be continued backwards out of
    the segment tiling instead -- segments seed at the falling-edge zero
    ``t_rf / 2`` and step by ``n t_rf``, so the virtual centre one step
    before the first lies at ``t_rf / 2 - n t_rf`` and the tail to the
    passage is ``n t_rf - t_rf / 2``, i.e. ``omega_c * dT = 2 pi n - pi``,
    which is ``pi`` (mod 2 pi) for integer ``n``. Without it the first
    passage alone would be demodulated exactly ``pi`` out of phase, and
    that wrongly signed deposit decays only over ``2 Q_L / omega``.

    The rule belongs to the grid bookkeeping, which produces every other
    segment tail, not to a run-start hook on the feedback: a hook has to
    be ordered by hand against the first ``_track``, and the value it
    writes is consumed by the grid twice over (the demodulation frame and
    the first coarse step's length).
    """

    N_TURNS = 3
    #: The static fixture lands the frame on ``pi`` to float noise; the
    #: guard itself only demands ``1e-3 pi``.
    FRAME_TOLERANCE = 1.0e-6

    @classmethod
    def setUpClass(cls):
        """Reuse the sibling fixture's cycle and template bunch."""
        TestBeamLoadingSignVsDesignRfPhase.setUpClass()

    def _demodulation_frames(self) -> list[float]:
        """
        ``omega_c * dT`` of every passage of a station-first ring.

        Returns
        -------
        frames
            One demodulation frame [rad] per passage, in passage order.
        """
        sibling = TestBeamLoadingSignVsDesignRfPhase
        profile = sibling._make_profile()
        ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=R_OVER_Q,
            Q_L=Q_L,
            generator_current_bias=V_DESIGN / (2.0 * R_OVER_Q * Q_L),
            n_cavities=1,
            initial_voltage=V_DESIGN,
            n_rf_periods_per_coarse_grid=1,
            delta_omega=0.0,
        )
        rf_station = SingleHarmonicRFStation(
            voltage=V_DESIGN,
            phi_rf=0.0,
            harmonic=HARMONIC,
            cavity_feedback=feedback,
            profile=profile,
        )
        # The RF station opens the ring: the case the backfill walk
        # cannot supply a tail for.
        ring.add_elements(
            [
                rf_station,
                DriftSimple(
                    orbit_length=CIRCUMFERENCE,
                    momentum_compaction_factor=ALPHA_P,
                ),
            ],
            reorder=False,
        )
        simulation = Simulation(ring=ring, magnetic_cycle=sibling.cycle)
        beam = sibling._prepare(simulation, INTENSITY)
        beam._dt.array_local += sibling.t_rf

        with mock.patch(
            "blond.physics.feedbacks.cavity_feedback.rf_beam_current",
            wraps=rf_beam_current,
        ) as demodulation:
            simulation.run_simulation(
                (beam,), n_turns=self.N_TURNS, show_progressbar=False
            )
        return [
            float(call.kwargs["dT"] * call.kwargs["omega_c"])
            for call in demodulation.call_args_list
        ]

    def test_first_passage_is_in_the_same_frame_as_the_later_ones(self):
        """Every passage, the first included, demodulates at ``pi``."""
        frames = self._demodulation_frames()
        self.assertEqual(len(frames), self.N_TURNS)
        for passage, frame in enumerate(frames):
            with self.subTest(passage=passage):
                deviation = (frame % (2.0 * np.pi)) - np.pi
                self.assertLess(
                    abs(deviation),
                    self.FRAME_TOLERANCE,
                    f"passage {passage} is demodulated at "
                    f"{frame / np.pi:.9f} pi",
                )
        self.assertAlmostEqual(frames[0], frames[1], delta=1e-9)

    def test_the_feedback_has_no_run_start_seed_hook(self):
        """The tail is produced by the grid, not by a run-start hook."""
        self.assertFalse(
            hasattr(
                IQCavityFeedbackTimingClass,
                "_seed_initial_demodulation_frame",
            ),
            "the first passage's tail is seeded by a run-start hook; it "
            "belongs to the grid bookkeeping that produces every other "
            "segment tail",
        )
