import unittest

import numpy as np

DEBUG_PLOTTING = False

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 1.6e11
n_turns = 2_000
voltage = 5e6
h = 35640
gamma_t = 53.8
alpha = 1 / gamma_t / gamma_t

injection_offset_phase = 40
n_macroparticles = 1_000_000  # Number of macroparticles per bunch [-]
tau_bunch = 1.2e-9
number_of_bunches = 1  # Length of the batch [number of bunches]
bunch_spacing = 10  # Bunch spacing [number of rf buckets]


class TestSingleBunchInjectionWithPhaseLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Initialize the simulations for the integration tests.
        """
        cls.pl_error_b2, cls.sl_error_b2, cls.omega_rf_b2, cls.phi_rf_b2 = (
            cls.setup_blond2()
        )
        cls.pl_error_b3, cls.sl_error_b3, cls.omega_rf_b3, cls.phi_rf_b3 = (
            cls.setup_blond3()
        )

    @staticmethod
    def setup_blond2():
        """
        Running the simulation in BLonD2.

        Returns
        -------
        beam_loop_error
            Turn-by-turn error signal from the beam-phase loop.
        synchro_loop_error
            Turn-by-turn error signal from the synchro loop.
        omega_rf
            Turn-by-turn value of the RF frequency.
        phi_rf
            Turn-by-turn value of the RF phase.
        """
        from blond.legacy.blond2.beam.beam import Beam, Proton
        from blond.legacy.blond2.beam.distributions import bigaussian
        from blond.legacy.blond2.beam.profile import CutOptions, Profile
        from blond.legacy.blond2.input_parameters.rf_parameters import (
            RFStation,
        )
        from blond.legacy.blond2.input_parameters.ring import Ring
        from blond.legacy.blond2.llrf.beam_feedback import BeamFeedback
        from blond.legacy.blond2.trackers.tracker import RingAndRFTracker

        # Initialize the accelerator
        ring = Ring(
            circumference, alpha, momentum, Proton(), n_turns=n_turns + 1
        )

        # The RF station
        rfstation = RFStation(ring, [h], [voltage], [0], n_rf=1)

        # The beam
        # First generate a single gaussian bunch
        beam = Beam(ring, n_macroparticles, intensity)
        bigaussian(ring, rfstation, beam, sigma_dt=tau_bunch / 4, seed=1234)

        # Add final corrections to the bunch positions
        beam.dt += injection_offset_phase * rfstation.t_rf[0, 0] / 360

        # The beam profile
        cut_options = CutOptions(
            cut_left=(-5.5) * rfstation.t_rf[0, 0],
            cut_right=(6.5 + number_of_bunches * bunch_spacing)
            * rfstation.t_rf[
                0,
                0,
            ],
            n_slices=(bunch_spacing * number_of_bunches + 12) * 2**6,
        )
        profile = Profile(beam, cut_options)

        PL_gain = 1 / (5 * ring.t_rev[0])
        SL_gain = PL_gain / 10
        bl_config = {
            "machine": "LHC",
            "PL_gain": PL_gain,
            "SL_gain": SL_gain,
        }

        beam_loop = BeamFeedback(ring, rfstation, profile, bl_config)

        # The RF tracker
        rftracker = RingAndRFTracker(
            rfstation,
            beam,
            Profile=profile,
            interpolation=True,
            BeamFeedback=beam_loop,
        )

        profile.track()

        # Initialize data arrays
        beam_loop_error = np.zeros(n_turns)
        synchro_loop_error = np.zeros(n_turns)

        omega_rf = np.zeros(n_turns)
        phi_rf = np.zeros(n_turns)

        for i in range(n_turns):
            profile.track()
            rftracker.track()

            beam_loop_error[i] = beam_loop.dphi * 180 / np.pi
            synchro_loop_error[i] = rfstation.dphi_rf[0] * 180 / np.pi
            omega_rf[i] = rfstation.omega_rf[0, i]
            phi_rf[i] = rfstation.phi_rf[0, i]

        return beam_loop_error, synchro_loop_error, omega_rf, phi_rf

    @staticmethod
    def setup_blond3():
        """
        Running the simulation in BLonD3.

        Returns
        -------
        beam_loop_error
            Turn-by-turn error signal from the beam-phase loop.
        synchro_loop_error
            Turn-by-turn error signal from the synchro loop.
        omega_rf
            Turn-by-turn value of the RF frequency.
        phi_rf
            Turn-by-turn value of the RF phase.
        """
        from blond import (
            Beam,
            BiGaussian,
            ConstantMagneticCycle,
            DriftSimple,
            MultiHarmonicRFStation,
            Ring,
            Simulation,
            StaticProfile,
            backend,
            proton,
        )
        from blond.core.backends.backend import Numpy64Bit
        from blond.physics.feedbacks.accelerators.lhc import (
            LHCBeamControl,
        )

        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        lattice = DriftSimple(
            orbit_length=circumference, momentum_compaction_factor=alpha
        )

        cavity = MultiHarmonicRFStation(
            voltage=np.array([voltage]),
            phi_rf=np.array([0.0]),
            harmonic=np.array([h]),
            n_harmonics=1,
            main_harmonic_idx=0,
        )

        f_rf = cavity.calc_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rf = 1 / f_rf
        t_rev = 1 / f_rev

        profile = StaticProfile(
            cut_left=-5.5 * t_rf,
            cut_right=(6.5 + number_of_bunches * bunch_spacing) * t_rf,
            n_bins=(bunch_spacing * number_of_bunches + 12) * 2**6,
        )

        bigaussian = BiGaussian(
            n_macroparticles, sigma_dt=tau_bunch / 4, seed=1234
        )
        beam_control = LHCBeamControl(
            pl_gain=1 / (5 * t_rev) * 1,
            sl_gain=1 / (5 * t_rev) / 10,
            profile=profile,
        )

        cavity.attach_beam_feedback(beam_control)

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [profile, cavity, beam_control, lattice],
        )

        simulation = Simulation(
            ring,
            cycle,
        )

        simulation.prepare_beam(beam, bigaussian)

        beam._dt.array_local += injection_offset_phase * t_rf / 360

        profile.track(beam)

        beam_loop_error = np.zeros(n_turns)
        synchro_loop_error = np.zeros(n_turns)
        omega_rf = np.zeros(n_turns)
        phi_rf = np.zeros(n_turns)

        simulation.finalize(
            (beam,),
            n_turns,
        )

        for i in range(n_turns):
            simulation.turn_counter.value = i

            omega_rf[i] = cavity.omega_rf[0]

            for element in ring.elements.elements:
                element.track(beam)

            phi_rf[i] = cavity.phi_rf[0]
            beam_loop_error[i] = beam_control.dphi * 180 / np.pi
            synchro_loop_error[i] = cavity._dphi_rf_next[0] * 180 / np.pi

        return beam_loop_error, synchro_loop_error, omega_rf, phi_rf

    def test_phase_loop_error(self):
        """Test of the phase loop error in BLonD2 and BLonD3."""
        np.testing.assert_allclose(
            self.pl_error_b3,
            self.pl_error_b2,
            atol=1e-1,
            err_msg="Error in phase loop error signal",
        )

    def test_synchronization_loop_error(self):
        """Test of the synchro loop error in BLonD2 and BLonD3."""
        np.testing.assert_allclose(
            self.sl_error_b3,
            self.sl_error_b2,
            atol=1e-1,
            err_msg="Error in synchronization loop error signal",
        )

    def test_rf_frequency_swing(self):
        """Test of the turn-by-turn rf frequency in BLonD2 and BLonD3."""
        np.testing.assert_allclose(
            self.omega_rf_b3,
            self.omega_rf_b2,
            atol=1e-2,
            err_msg="Error in rf frequency swing",
        )

    def test_rf_phase_swing(self):
        """Test of the turn-by-turn rf phase in BLonD2 and BLonD3."""
        np.testing.assert_allclose(
            self.phi_rf_b3,
            self.phi_rf_b2,
            atol=1e-2,
            err_msg="Error in rf phase swing",
        )
