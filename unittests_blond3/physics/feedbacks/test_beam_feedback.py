# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:27:10 2018

@author: schwarz
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond3 import (
    Ring,
    proton,
    MultiHarmonicCavity,
    Beam,
    ConstantMagneticCycle,
    Simulation,
    BiGaussian,
    StaticProfile,
    BunchObservation,
    CavityPhaseObservation,
)
from blond3.physics.feedbacks.accelerators.lhc.beam_feedback import SpsRlBeamFeedback


class TestBeamFeedback(unittest.TestCase):
    def setUpBlond2(self):
        from blond.llrf.beam_feedback import BeamFeedback
        from blond.beam.beam import Beam, Proton
        from blond.beam.distributions import bigaussian
        from blond.beam.profile import CutOptions, Profile
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring
        from blond.trackers.tracker import RingAndRFTracker

        n_turns = 200
        intensity_pb = 1.2e6  # protons per bunch
        n_macroparticles = int(1e6)  # macropartilces per bunch
        sigma = 0.05e-9  # sigma for gaussian bunch [s]
        self.time_offset = 0.1e-9  # time by which to offset the bunch

        # Ring parameters SPS
        C = 6911.5038  # Machine circumference [m]
        sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]
        gamma_transition = 17.95142852  # Q20 Transition gamma
        momentum_compaction = 1.0 / gamma_transition**2  # Momentum compaction array

        self.ring_blond2 = Ring(
            C, momentum_compaction, sync_momentum, Proton(), n_turns=n_turns
        )

        # RF parameters SPS
        harmonic = 4620  # Harmonic numbers
        voltage = 4.5e6  # [V]
        phi_offsets = 0

        self.rf_station_blond2 = RFStation(
            self.ring_blond2, harmonic, voltage, phi_offsets
        )
        t_rf = self.rf_station_blond2.t_rf[0, 0]

        # Beam setup
        self.beam_blond2 = Beam(self.ring_blond2, n_macroparticles, intensity_pb)

        bigaussian(
            self.ring_blond2,
            self.rf_station_blond2,
            self.beam_blond2,
            sigma,
            seed=1234,
            reinsertion=True,
        )

        # displace beam to see effect of phase error and phase loop
        self.beam_blond2.dt += self.time_offset

        # Profile setup

        self.profile_blond2 = Profile(
            self.beam_blond2,
            cut_options=CutOptions(cut_left=0, cut_right=t_rf, n_slices=1024),
        )
        PL_gain = 1000  # gain of phase loop
        self.phase_loop_blond2 = BeamFeedback(
            self.ring_blond2,
            self.rf_station_blond2,
            self.profile_blond2,
            {
                "machine": "SPS_RL",
                "PL_gain": PL_gain,
            },
        )

        # Tracker setup
        self.section_tracker_blond2 = RingAndRFTracker(
            self.rf_station_blond2,
            self.beam_blond2,
            profile=self.profile_blond2,
            beam_feedback=self.phase_loop_blond2,
            interpolation=False,
        )

    def setUpBlond3(self):
        intensity_pb = 1.2e6  # protons per bunch
        n_macroparticles = int(1e6)  # macropartilces per bunch
        sigma = 0.05e-9  # sigma for gaussian bunch [s]
        self.time_offset = 0.1e-9  # time by which to offset the bunch
        gamma_transition = 17.95142852  # Q20 Transition gamma

        C = 6911.5038  # Machine circumference [m]
        sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]

        self.ring = Ring(circumference=C)

        self.magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        t_rf = (
            self.magnetic_cycle.get_t_rev_init(
                circumference=self.ring.circumference,
                turn_i_init=0,
                t_init=0,
                particle_type=self.magnetic_cycle._reference_particle,
            )
            / 4620
        )

        self.profile = StaticProfile(
            cut_left=0,
            cut_right=t_rf,
            n_bins=1024,
        )

        self.sps_beam_feedback = SpsRlBeamFeedback(
            profile=self.profile,
            PL_gain=1000,  # gain of phase loop
        )
        self.cavity = MultiHarmonicCavity(
            n_harmonics=1,
            main_harmonic_idx=0,
            beam_feedback=self.sps_beam_feedback,
        )

        # RF parameters SPS
        self.cavity.harmonic = np.array([4620])  # Harmonic numbers
        self.cavity.voltage = np.array([4.5e6])  # [V]
        self.cavity.phi_rf = np.array([0.0])
        self.ring.add_element(self.cavity)
        self.ring.add_drifts(
            n_sections=1,
            n_drifts_per_section=1,
            transition_gamma=gamma_transition,
        )
        self.ring.add_element(self.profile)

        # Beam setup
        self.beam = Beam(n_particles=intensity_pb, particle_type=proton)
        self.simulation = Simulation(
            ring=self.ring,
            magnetic_cycle=self.magnetic_cycle,
        )
        self.simulation.prepare_beam(
            beam=self.beam,
            preparation_routine=BiGaussian(
                n_macroparticles=n_macroparticles,
                sigma_dt=sigma,
                seed=1234,
                reinsertion=True,
            ),
        )
        self.simulation.print_one_turn_execution_order()

        # displace beam to see effect of phase error and phase loop
        self.beam._dt += self.time_offset
        self.profile.track(self.beam)

    def setUp(self):
        self.setUpBlond2()
        self.setUpBlond3()

    def test_setup(self):
        obs_bunch = BunchObservation(each_turn_i=1)
        cav_obs = CavityPhaseObservation(each_turn_i=1, cavity=self.cavity)

        def callback(simulation: Simulation):
            plt.figure(3)
            plt.subplot(4, 1, 1)
            plt.title("phi_beam")
            plt.plot(
                simulation.turn_i.value, self.sps_beam_feedback.phi_beam, "o", c="C0"
            )
            plt.subplot(4, 1, 2)
            plt.title("phi_s")

            plt.plot(
                simulation.turn_i.value,
                self.sps_beam_feedback._parent_cavity.phi_s,
                "o",
                c="C0",
            )

        self.simulation.run_simulation(
            (self.beam,),
            n_turns=200,
            observe=(obs_bunch, cav_obs),
            # callback=callback,
        )
        for i in range(200):
            # animation if helpful for debugging
            # plt.cla()
            # plt.scatter(self.beam_blond2.dt[:], self.beam_blond2.dE[:])
            # plt.scatter(obs_bunch.dts[i,:], obs_bunch.dEs[i,:],marker="x")
            # plt.draw()
            # plt.pause(.1)
            self.section_tracker_blond2.track()
            self.profile_blond2.track()
            """plt.figure(3)

            plt.subplot(4, 1, 1)

            plt.plot(i, self.phase_loop_blond2.phi_beam, "x", c="C1")"""
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.figure(3)
            plt.subplot(4, 1, 2)

            plt.plot(self.rf_station_blond2.phi_s, "^", c="C1", label="phi_s")
            plt.legend()
            plt.figure(2)
            plt.subplot(3, 1, 1)
            plt.title("phases")
            plt.plot(cav_obs.phases[1:])
            plt.plot(self.rf_station_blond2.phi_rf[0, :-1], "--")
            plt.subplot(3, 1, 2)
            plt.title("omegas")
            plt.plot(cav_obs.omegas[1:])
            plt.plot(self.rf_station_blond2.omega_rf[0, :], "--")
            plt.subplot(3, 1, 3)
            plt.title("voltages")
            plt.plot(cav_obs.voltages[1:])
            plt.plot(self.rf_station_blond2.voltage[0, :], "--")
            plt.show()

        np.testing.assert_allclose(
            cav_obs.phases[1:, 0] + 1,
            self.rf_station_blond2.phi_rf[0, :-1] + 1,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            cav_obs.omegas[1:, 0] + 1,
            self.rf_station_blond2.omega_rf[0, :-1] + 1,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            cav_obs.voltages[1:, 0] + 1,
            self.rf_station_blond2.voltage[0, :-1] + 1,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
