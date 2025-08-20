import unittest

import numpy as np
from scipy.constants import c

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.impedances.impedance import InducedVoltageTime, TotalInducedVoltage
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.llrf.cavity_feedback import (SPSCavityLoopCommissioning,
                                        SPSCavityFeedback)
from blond.trackers.tracker import RingAndRFTracker


class TestSPSCavityFeedback(unittest.TestCase):
    def test_something(self):
        C = 2 * np.pi * 1100.009  # Ring circumference [m]
        gamma_t = 18.0  # Gamma at transition
        alpha = 1 / gamma_t ** 2  # Momentum compaction factor
        p_s = 25.92e9  # Synchronous momentum at injection [eV]
        h = 4620  # 200 MHz system harmonic
        phi = 0.  # 200 MHz RF phase

        # With this setting, amplitude in the two four-section, five-section
        # cavities must converge, respectively, to
        # 2.0 MV = 4.5 MV * 4/18 * 2
        # 2.5 MV = 4.5 MV * 5/18 * 2
        V = 4.5e6  # 200 MHz RF voltage

        N_t = 1  # Number of turns to track

        self.ring = Ring(C, alpha, p_s, particle=Proton(), n_turns=N_t)
        self.rf = RFStation(self.ring, h, V, phi)

        N_m = int(1e6)  # Number of macro-particles for tracking
        N_b = 288 * 2.3e11  # Bunch intensity [ppb]

        # Gaussian beam profile
        self.beam = Beam(self.ring, N_m, N_b)
        sigma = 1.0e-9
        bigaussian(self.ring, self.rf, self.beam, sigma, seed=1234,
                   reinsertion=False)

        n_shift = 1550  # how many rf-buckets to shift beam
        self.beam.dt += n_shift * self.rf.t_rf[0, 0]

        self.profile = Profile(
            self.beam, cut_options=CutOptions(
                cut_left=(n_shift - 1.5) * self.rf.t_rf[0, 0],
                cut_right=(n_shift + 2.5) * self.rf.t_rf[0, 0],
                n_slices=4 * 64))
        self.profile.track()

        # Cavities
        l_cav = 32 * 0.374
        v_g = 0.0946
        tau = l_cav / (v_g * c) * (1 + v_g)
        f_cav = 200.222e6
        n_cav = 4  # factor 2 because of two four/five-sections cavities
        short_cavity = TravelingWaveCavity(l_cav ** 2 * n_cav * 27.1e3 / 8,
                                           f_cav, 2 * np.pi * tau)
        shortInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                 [short_cavity])
        l_cav = 43 * 0.374
        tau = l_cav / (v_g * c) * (1 + v_g)
        n_cav = 2
        long_cavity = TravelingWaveCavity(l_cav ** 2 * n_cav * 27.1e3 / 8,
                                          f_cav, 2 * np.pi * tau)
        longInducedVoltage = InducedVoltageTime(self.beam, self.profile,
                                                [long_cavity])
        self.induced_voltage = TotalInducedVoltage(
            self.beam, self.profile, [shortInducedVoltage, longInducedVoltage])
        self.induced_voltage.induced_voltage_sum()

        self.cavity_tracker = RingAndRFTracker(
            self.rf, self.beam, profile=self.profile, interpolation=True,
            total_induced_voltage=self.induced_voltage)

        self.OTFB = SPSCavityFeedback(
            self.rf, self.profile, G_llrf=20,
            G_tx=[1.0355739238973907, 1.078403005653143],
            a_comb=63 / 64, turns=1000, post_LS2=True, df=[0.18433333e6, 0.2275e6],
            commissioning=SPSCavityLoopCommissioning(open_ff=True, rot_iq=-1))

        self.OTFB_tracker = RingAndRFTracker(self.rf, self.beam,
                                             profile=self.profile,
                                             total_induced_voltage=None,
                                             cavity_feedback=self.OTFB,
                                             interpolation=True)