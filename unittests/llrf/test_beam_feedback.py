# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:27:10 2018

@author: schwarz
"""

import cmath
import unittest

import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker


class TestBeamFeedback(unittest.TestCase):

    def setUp(self):
        n_turns = 200
        intensity_pb = 1e9  # protons per bunch
        n_macroparticles = int(1e6)  # macropartilces per bunch
        sigma = 0.05e-9  # sigma for gaussian bunch [s]
        self.time_offset = 0.05e-9  # time by which to offset the bunch

        # Ring parameters SPS
        C = 6911.5038  # Machine circumference [m]
        sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]
        gamma_transition = 17.95142852  # Q20 Transition gamma
        momentum_compaction = 1. / gamma_transition**2  # Momentum compaction array

        self.ring = Ring(C, momentum_compaction, sync_momentum, Proton(),
                         n_turns=n_turns)

        # RF parameters SPS
        harmonic = 4620  # Harmonic numbers
        voltage = 4.5e6  # [V]
        phi_offsets = 0

        self.rf_station = RFStation(self.ring, harmonic, voltage, phi_offsets)
        t_rf = self.rf_station.t_rf[0, 0]

        # Beam setup
        self.beam = Beam(self.ring, n_macroparticles, intensity_pb)

        bigaussian(self.ring, self.rf_station, self.beam, sigma, seed=1234,
                   reinsertion=True)

        # displace beam to see effect of phase error and phase loop
        self.beam.dt += self.time_offset

        # Profile setup

        self.profile = Profile(self.beam, CutOptions=CutOptions(cut_left=0,
                                                                cut_right=t_rf,
                                                                n_slices=1024))


    def test_SPS_RL(self):

        PL_gain = 1000      # gain of phase loop
        rtol = 0        # relative tolerance
        atol = 1.5             # absolute tolerance, in %
        # Phase loop setup

        phase_loop = BeamFeedback(self.ring, self.rf_station, self.profile,
                                  {'machine': 'SPS_RL', 'PL_gain': PL_gain})

        # Tracker setup
        section_tracker = RingAndRFTracker(
            self.rf_station, self.beam, Profile=self.profile,
            BeamFeedback=phase_loop, interpolation=False)
        tracker = FullRingAndRF([section_tracker])

        # average beam position
        beamAvgPos = np.zeros(self.ring.n_turns)

        n_turns = self.ring.n_turns
        for turn in range(n_turns):
            beamAvgPos[turn] = np.mean(self.beam.dt)
            self.profile.track()
            tracker.track()

        # difference between beam position and synchronuous position
        # (assuming no beam loading)
        delta_tau = beamAvgPos - (self.rf_station.phi_s[:-1]- self.rf_station.phi_rf[0, :-1])\
            / self.rf_station.omega_rf[0, :-1]

        # initial position for analytic solution
        init_pos = self.time_offset

        omega_eff = cmath.sqrt(-PL_gain**2 + 4 * self.rf_station.omega_s0[0]**2)
        time = np.arange(n_turns) * self.ring.t_rev[0]
        # initial derivative for analytic solution;
        # defined such that analytical solution at turn 1 agrees with numerical
        # solution
        init_slope = 0.5 * (delta_tau[1] * omega_eff * np.exp(0.5 * PL_gain * time[1])
                            / np.sin(0.5 * omega_eff * time[1]) - delta_tau[0]
                            * (PL_gain + omega_eff / np.tan(0.5 * omega_eff * time[1]))).real

        delta_tau_analytic = init_pos * np.exp(-0.5 * PL_gain * time)
        delta_tau_analytic *= np.cos(0.5 * time * omega_eff).real\
            + (PL_gain + 2 * init_slope / init_pos)\
            * (np.sin(0.5 * time * omega_eff) / omega_eff).real

        difference = np.abs(delta_tau - delta_tau_analytic)
        # normalize result
        difference = difference / np.max(delta_tau)
        # Percentage relative error
        difference = difference * 100
        # expected difference


        np.testing.assert_allclose(np.zeros(len(difference)), difference,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestBeamFeedback test_SPS_RL: difference between simulated and analytic result different than expected')


if __name__ == '__main__':

    unittest.main()
