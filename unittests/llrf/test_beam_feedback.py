# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:27:10 2018

@author: schwarz
"""

import unittest
import numpy as np
import cmath

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF


class TestBeamFeedback(unittest.TestCase):

    def setUp(self):
        n_turns = 200
        intensity_pb = 1.2e6  # protons per bunch
        n_macroparticles = int(1e6)  # macropartilces per bunch
        sigma = 0.05e-9  # sigma for gaussian bunch [s]
        self.time_offset = 0.1e-9  # time by which to offset the bunch

        # Ring parameters SPS
        C = 6911.5038  # Machine circumference [m]
        sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]
        gamma_transition = 17.95142852  # Q20 Transition gamma
        momentum_compaction = 1./gamma_transition**2  # Momentum compaction array

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

    @unittest.skip("FIXME")
    def test_SPS_RL(self):

        PL_gain = 1000      # gain of phase loop
        rtol = 1e-4         # relative tolerance
        atol = 0              # absolute tolerance
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
        delta_tau = beamAvgPos - (np.pi - self.rf_station.phi_rf[0, :-1])\
            / self.rf_station.omega_rf[0, :-1]

        # initial position for analytic solution
        init_pos = self.time_offset

        omega_eff = cmath.sqrt(-PL_gain**2 + 4*self.rf_station.omega_s0[0]**2)
        time = np.arange(n_turns) * self.ring.t_rev[0]
        # initial derivative for analytic solution;
        # defined such that analytical solution at turn 1 agrees with numerical
        # solution
        init_slope = 0.5 * (delta_tau[1] * omega_eff * np.exp(0.5*PL_gain*time[1])
                            / np.sin(0.5*omega_eff*time[1]) - delta_tau[0]
                            * (PL_gain+omega_eff/np.tan(0.5*omega_eff*time[1]))).real

        delta_tau_analytic = init_pos * np.exp(-0.5*PL_gain*time)
        delta_tau_analytic *= np.cos(0.5*time*omega_eff).real\
            + (PL_gain+2*init_slope/init_pos)\
            * (np.sin(0.5*time*omega_eff)/omega_eff).real

        difference = delta_tau - delta_tau_analytic
        # normalize result
        difference = difference / np.max(difference)
        # expected difference
        difference_exp = np.array([
            -1.56306635e-05, -1.55605315e-05,  -2.10224435e-05, -3.18525050e-05,
            -4.74014489e-05, -6.70584402e-05, -9.01307422e-05, -1.15823959e-04,
            -1.43290487e-04, -1.71572162e-04,  -1.99820151e-04,  -2.27071730e-04,
            -2.52331681e-04,  -2.74668126e-04,  -2.93165304e-04, -3.06972913e-04,
            -3.15442474e-04,  -3.17857324e-04, -3.13794970e-04,  -3.02786089e-04,
            -2.84680298e-04, -2.59322215e-04,  -2.26874004e-04,  -1.87452375e-04,
            -1.41293604e-04,  -8.89863575e-05,  -3.08865701e-05, 3.22411495e-05,
            9.97408029e-05,   1.70914181e-04, 2.44766912e-04, 3.20596833e-04,
            3.97403451e-04, 4.74233283e-04,   5.50189125e-04, 6.24368453e-04,
            6.95836553e-04,   7.63737143e-04,   8.27069057e-04, 8.84995559e-04,
            9.36770723e-04,   9.81561780e-04, 1.01869959e-03,   1.04738842e-03,
            1.06711062e-03, 1.07736961e-03,   1.07778386e-03,   1.06805613e-03,
            1.04797776e-03,   1.01747638e-03,   9.76519221e-04, 9.25420191e-04,
            8.64415092e-04,   7.93844624e-04, 7.14396030e-04, 6.26549187e-04,
            5.31154439e-04, 4.28985322e-04,   3.21198916e-04,   2.08550190e-04,
            9.21607082e-05,  -2.68249728e-05,  -1.47278123e-04, -2.67890543e-04,
            -3.87642210e-04,  -5.05244473e-04, -6.19660328e-04,  -7.29670300e-04,
            -8.34272846e-04, -9.32388033e-04,  -1.02301036e-03,  -1.10520861e-03,
            -1.17824066e-03,  -1.24119243e-03,  -1.29350096e-03, -1.33458128e-03,
            -1.36388379e-03,  -1.38105465e-03, -1.38595634e-03,  -1.37832214e-03,
            -1.35829791e-03, -1.32588558e-03,  -1.28146000e-03,  -1.22518721e-03,
            -1.15769141e-03,  -1.07943574e-03,  -9.91143310e-04, -8.93671637e-04,
            -7.87961546e-04,  -6.74866999e-04, -5.55444011e-04,  -4.30919368e-04,
            -3.02270469e-04, -1.70824836e-04,  -3.77396109e-05,   9.56816273e-05,
            2.28299979e-04,   3.58842001e-04,   4.86074690e-04, 6.08875045e-04,
            7.26090501e-04,   8.36677390e-04, 9.39639556e-04,   1.03407702e-03,
            1.11906014e-03, 1.19386315e-03,   1.25779004e-03,   1.31037519e-03,
            1.35108872e-03,   1.37958003e-03,   1.39570542e-03, 1.39927441e-03,
            1.39033118e-03,   1.36892681e-03, 1.33533475e-03,   1.28987173e-03,
            1.23311389e-03, 1.16551418e-03,   1.08773037e-03,   1.00059786e-03,
            9.04879918e-04,   8.01551710e-04,   6.91575582e-04, 5.75952750e-04,
            4.55756793e-04,   3.32302985e-04, 2.06487043e-04,   7.95882588e-05,
            -4.72208138e-05, -1.72823958e-04,  -2.96101535e-04,  -4.15925168e-04,
            -5.31250383e-04,  -6.41017819e-04,  -7.44349685e-04, -8.40276057e-04,
            -9.28032591e-04,  -1.00688055e-03, -1.07610640e-03,  -1.13518206e-03,
            -1.18370702e-03, -1.22129557e-03,  -1.24764964e-03,  -1.26264035e-03,
            -1.26627364e-03,  -1.25857717e-03,  -1.23964021e-03,
            -1.20980891e-03,  -1.16944284e-03,  -1.11887385e-03,
            -1.05870668e-03,  -9.89617769e-04,  -9.12311681e-04,
            -8.27560752e-04,  -7.36170045e-04,  -6.39042153e-04,
            -5.37114997e-04,  -4.31247869e-04,  -3.22637131e-04,
            -2.12194968e-04,  -1.00869243e-04,   1.03136916e-05,
            1.20241207e-04,   2.27979265e-04,   3.32675130e-04,
            4.33323979e-04,   5.29105666e-04,   6.19142445e-04,
            7.02728753e-04,   7.79114404e-04,   8.47787258e-04,
            9.08216047e-04,   9.59821724e-04,   1.00228122e-03,
            1.03538392e-03,   1.05880009e-03,   1.07260841e-03,
            1.07662550e-03,   1.07093155e-03,   1.05577003e-03,
            1.03129797e-03,   9.97904596e-04,   9.55975595e-04,
            9.05955028e-04,   8.48396342e-04,   7.83925297e-04,
            7.13242537e-04,   6.36896396e-04,   5.55809454e-04,
            4.70697276e-04,   3.82464668e-04,   2.91766220e-04,
            1.99564879e-04,   1.06707654e-04,   1.40463177e-05,
            -7.76333806e-05,  -1.67470574e-04,  -2.54708122e-04,
            -3.38623857e-04,  -4.18484684e-04])
        difference_exp = difference_exp/np.max(difference_exp)
        np.testing.assert_allclose(difference_exp, difference,
                                   rtol=rtol, atol=atol,
                                   err_msg='In TestBeamFeedback test_SPS_RL: difference between simulated and analytic result different than expected')


if __name__ == '__main__':

    unittest.main()
