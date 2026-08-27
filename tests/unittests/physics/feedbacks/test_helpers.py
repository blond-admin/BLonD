import unittest

import numpy as np
import pytest
from scipy.constants import elementary_charge as e

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    StaticProfile,
    momentum_compaction_factor,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, backend
from blond.physics.feedbacks.helpers import (
    cartesian_to_polar,
    low_pass_filter,
    polar_to_cartesian,
    rf_beam_current,
)


class TestLowPass(unittest.TestCase):
    def test_1(self):
        # Example based on SciPy.org filtfilt
        t = np.linspace(0, 1.0, 2001)
        xlow = np.sin(2 * np.pi * 5 * t)
        xhigh = np.sin(2 * np.pi * 250 * t)
        x = xlow + xhigh

        y = low_pass_filter(x, cutoff_frequency=1 / 8)

        # Test for difference between filtered signal and xlow;
        # using signal.butter(8, 0.125) and filtfilt(b, a, x, padlen=15)
        # from the SciPy documentation of filtfilt gives the stated
        # value 9.10862958....e-6
        self.assertAlmostEqual(
            np.abs(y - xlow).max(),
            0.0230316365,
            places=10,
        )


class TestRFBeamCurrent(unittest.TestCase):
    # @unittest.skip("feedbacks not working")
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        C = 2 * np.pi * 1100.009  # Ring circumference [m]
        gamma_t = 18.0  # Gamma at transition
        alpha = 1 / gamma_t**2  # Momentum compaction factor
        p_s = 25.92e9  # Synchronous momentum at injection [eV]

        N_m = int(1e5)  # Number of macro-particles for tracking
        N_b = 1.0e11  # Bunch intensity [ppb]
        self.n_macroparticles = N_m

        # Set up machine parameters
        # self.ring = Ring(C, alpha, p_s, Proton(), n_turns=1)
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=p_s,
            in_unit="momentum",
        )
        self.ring = Ring(circumference=C)
        # self.rf = RFStation(self.ring, 4620, 4.5e6, 0)
        self.rf = MultiHarmonicRFStation(
            harmonic=np.array([4620], dtype=backend.float),
            voltage=np.array([4.5e6], dtype=backend.float),
            phi_rf=np.array([0], dtype=backend.float),
            n_harmonics=1,
            main_harmonic_idx=0,
        )
        self.drift = DriftSimple(
            orbit_length=C,
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=gamma_t
            ),
        )
        self.ring.add_elements((self.rf, self.drift))

        # RF-frequency at which to compute beam current
        self.omega = 2 * np.pi * 200.222e6

        # Create Gaussian beam
        self.beam = Beam(
            intensity=N_b,
            particle_type=proton,
        )

        self.profile = StaticProfile(
            cut_left=-2e-9,
            cut_right=8e-9,
            n_bins=100,
        )
        simulation = Simulation(ring=self.ring, magnetic_cycle=magnetic_cycle)
        self.simulation = simulation
        self.beam.reference.total_energy = (
            magnetic_cycle.get_total_energy_init(
                particle_type=proton,
            )
        )
        self.omega_rf = float(
            self.rf.calc_omega_rf_design(
                self.beam.reference.beta, self.ring.circumference
            )[self.rf.main_harmonic_idx]
        )
        self.beam.setup_beam(dt=np.zeros(N_m), dE=np.zeros(N_m))

    @pytest.mark.backend_mutation
    def test_setup(self):
        pass  # see if setup works

    # Test charge distribution with analytic functions
    # Compare with theoretical value
    @pytest.mark.backend_mutation
    def test_1(self):
        t = self.profile.hist_x
        self.profile._hist_y = 2600 * np.exp(
            -((t - 2.5e-9) ** 2) / (2 * 0.5e-9) ** 2
        )

        t_rev = float(
            (2 * np.pi * self.rf.get_main_harmonic()) / self.omega_rf
        )

        rf_current = rf_beam_current(
            self.beam,
            self.profile,
            self.omega,
            use_lowpass_filter=False,
            external_reference=True,
            dT=0,
        )

        rf_current_real = np.around(rf_current.real, 12)
        rf_current_imag = np.around(rf_current.imag, 12)

        rf_theo_real = (
            2
            * self.beam.ratio
            * self.beam.particle_type.charge
            * e
            * 2600
            * np.exp(-((t - 2.5e-9) ** 2) / (2 * 0.5 * 1e-9) ** 2)
            * np.cos(self.omega * t)
        )
        rf_theo_real = np.around(rf_theo_real, 12)

        rf_theo_imag = (
            -2
            * self.beam.ratio
            * self.beam.particle_type.charge
            * e
            * 2600
            * np.exp(-((t - 2.5e-9) ** 2) / (2 * 0.5 * 1e-9) ** 2)
            * np.sin(self.omega * t)
        )
        rf_theo_imag = np.around(rf_theo_imag, 12)

        np.testing.assert_allclose(
            rf_current_real,
            rf_theo_real,
        )
        np.testing.assert_allclose(
            rf_current_imag,
            rf_theo_imag,
        )

    # Test charge distribution of a bigaussian profile, without LPF
    # Compare to simulation data
    @pytest.mark.backend_mutation
    def test_2(self):
        self.simulation.prepare_beam(
            beam=self.beam,
            preparation_routine=BiGaussian(
                n_macroparticles=self.n_macroparticles,
                sigma_dt=3.2e-9 / 4,
                sigma_dE=None,
                reinsertion=True,
                seed=1234,
            ),
        )
        self.profile.track(beam=self.beam)
        t_rev = float(
            (2 * np.pi * self.rf.get_main_harmonic())
            / self.rf.calc_omega_rf_design(
                self.beam.reference.beta, self.ring.circumference
            )[self.rf.main_harmonic_idx]
        )
        rf_current = rf_beam_current(
            self.beam,
            self.profile,
            self.omega,
            use_lowpass_filter=False,
            external_reference=True,
        )

        Iref_real = np.array(
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                7.92343713e-14,
                1.17565209e-13,
                1.54037861e-13,
                3.76151710e-13,
                4.38282401e-13,
                1.48045735e-12,
                1.35222334e-12,
                5.79743820e-13,
                9.14152671e-13,
                9.44240899e-13,
                3.19801617e-13,
                0.00000000e00,
                3.46221663e-12,
                6.39906870e-12,
                1.56530831e-11,
                2.35286862e-11,
                3.47907477e-11,
                4.58005109e-11,
                5.77392874e-11,
                6.62362803e-11,
                7.82984295e-11,
                6.79830906e-11,
                4.32594460e-11,
                -7.71572435e-13,
                -6.98622431e-11,
                -1.72921639e-10,
                -2.93663454e-10,
                -4.53113976e-10,
                -6.44608738e-10,
                -8.27691108e-10,
                -1.00045780e-09,
                -1.20532535e-09,
                -1.37275783e-09,
                -1.50700319e-09,
                -1.62579598e-09,
                -1.66460303e-09,
                -1.68753194e-09,
                -1.55757951e-09,
                -1.50906664e-09,
                -1.38532852e-09,
                -1.18191538e-09,
                -9.77471102e-10,
                -8.18280669e-10,
                -6.11484460e-10,
                -4.58559052e-10,
                -2.80009598e-10,
                -1.62320000e-10,
                -6.47893125e-11,
                2.23593075e-12,
                4.60546387e-11,
                6.78858076e-11,
                7.31008392e-11,
                7.22251131e-11,
                6.24707337e-11,
                4.98931974e-11,
                4.11950132e-11,
                1.73847516e-11,
                1.80306776e-11,
                8.24583997e-12,
                6.30317063e-12,
                9.59802958e-13,
                3.19653359e-13,
                9.42960274e-13,
                6.08037625e-13,
                8.66737017e-13,
                2.69239438e-13,
                0.00000000e00,
                4.35010763e-13,
                0.00000000e00,
                1.52074465e-13,
                0.00000000e00,
                7.70670378e-14,
                0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )

        I_real = rf_current.real
        Iref_real = Iref_real

        np.testing.assert_allclose(I_real, Iref_real)

        Iref_imag = np.array(
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                3.10484642e-13,
                2.98089282e-13,
                2.80982448e-13,
                5.18869045e-13,
                4.67572168e-13,
                1.22665512e-12,
                8.59338117e-13,
                2.73152518e-13,
                2.97378683e-13,
                1.80328346e-13,
                2.01426047e-14,
                0.00000000e00,
                -6.61203935e-13,
                -2.08165078e-12,
                -7.37511800e-12,
                -1.49524832e-11,
                -2.88263953e-11,
                -4.88612915e-11,
                -7.96463984e-11,
                -1.20822453e-10,
                -1.98527462e-10,
                -2.66395823e-10,
                -3.46908120e-10,
                -4.42520514e-10,
                -5.44764748e-10,
                -6.67874490e-10,
                -7.37049493e-10,
                -8.19736598e-10,
                -8.82688669e-10,
                -8.76856073e-10,
                -8.23077356e-10,
                -7.60096780e-10,
                -6.40949423e-10,
                -4.84431348e-10,
                -3.04617414e-10,
                -9.90179787e-11,
                1.12198045e-10,
                3.03095363e-10,
                4.96733215e-10,
                6.58625421e-10,
                7.56906322e-10,
                8.15663690e-10,
                8.79089063e-10,
                8.49710096e-10,
                8.43427611e-10,
                7.17289345e-10,
                6.45446562e-10,
                5.34741143e-10,
                4.27454878e-10,
                3.49457174e-10,
                2.58477023e-10,
                1.81627143e-10,
                1.29594307e-10,
                8.49203349e-11,
                5.24889624e-11,
                3.36509712e-11,
                1.08785464e-11,
                8.34217998e-12,
                2.61896305e-12,
                1.15825707e-12,
                5.37351873e-14,
                -2.23725006e-14,
                -1.86909360e-13,
                -2.02498004e-13,
                -4.15783756e-13,
                -1.73749600e-13,
                0.00000000e00,
                -4.70617500e-13,
                0.00000000e00,
                -2.82049917e-13,
                0.00000000e00,
                -3.11029694e-13,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )

        I_imag = rf_current.imag  # round
        Iref_imag = Iref_imag
        np.testing.assert_allclose(I_imag, Iref_imag)

    # Test charge distribution of a bigaussian profile, with LPF
    # Compare to simulation data
    @pytest.mark.backend_mutation
    def test_3(self):
        self.simulation.prepare_beam(
            beam=self.beam,
            preparation_routine=BiGaussian(
                n_macroparticles=self.n_macroparticles,
                sigma_dt=3.2e-9 / 4,
                sigma_dE=None,
                reinsertion=True,
                seed=1234,
            ),
        )
        t_rev = float(
            (2 * np.pi * self.rf.get_main_harmonic())
            / self.rf.calc_omega_rf_design(
                self.beam.reference.beta, self.ring.circumference
            )[self.rf.main_harmonic_idx]
        )
        self.profile.track(self.beam)
        self.assertEqual(
            len(self.beam.read_partial_dt()),
            np.sum(self.profile.hist_y),
            "In"
            + " TestBeamCurrent: particle number mismatch in Beam vs Profile",
        )

        # RF current calculation with low-pass filter
        rf_current = rf_beam_current(
            self.beam, self.profile, self.omega, use_lowpass_filter=True
        )

        Iref_real = np.array(
            [
                -4.8900223267e-12,
                -4.8923104036e-12,
                -4.8945321340e-12,
                -4.8966886226e-12,
                -4.8987809767e-12,
                -4.9008103057e-12,
                -4.9027777204e-12,
                -4.9046843330e-12,
                -4.9065312562e-12,
                -4.9083196031e-12,
                -4.9100504867e-12,
                -4.9117250191e-12,
                -4.9133443118e-12,
                -4.9149094745e-12,
                -4.9164216151e-12,
                -4.9178818391e-12,
                -4.9192912492e-12,
                -4.9206509448e-12,
                -4.9219620218e-12,
                -4.9232255716e-12,
                -4.9244426815e-12,
                -4.9256144334e-12,
                -4.9267419040e-12,
                -4.9278261641e-12,
                -4.9288682779e-12,
                -4.9298693033e-12,
                -4.9308302909e-12,
                -4.9317522835e-12,
                -4.9326363162e-12,
                -4.9334834154e-12,
                -4.9342945991e-12,
                -4.9350708757e-12,
                -4.9358132440e-12,
                -4.9365226930e-12,
                -4.9372002011e-12,
                -4.9378467360e-12,
                -4.9384632542e-12,
                -4.9390507006e-12,
                -4.9396100084e-12,
                -4.9401420982e-12,
                -4.9406478784e-12,
                -4.9411282441e-12,
                -4.9415840773e-12,
                -4.9420162464e-12,
                -4.9424256058e-12,
                -4.9428129959e-12,
                -4.9431792423e-12,
                -4.9435251561e-12,
                -4.9438515331e-12,
                -4.9441591540e-12,
                -4.9444487839e-12,
                -4.9447211719e-12,
                -4.9449770515e-12,
                -4.9452171396e-12,
                -4.9454421370e-12,
                -4.9456527277e-12,
                -4.9458495793e-12,
                -4.9460333420e-12,
                -4.9462046495e-12,
                -4.9463641181e-12,
                -4.9465123469e-12,
                -4.9466499179e-12,
                -4.9467773955e-12,
                -4.9468953268e-12,
                -4.9470042416e-12,
                -4.9471046520e-12,
                -4.9471970530e-12,
                -4.9472819219e-12,
                -4.9473597189e-12,
                -4.9474308869e-12,
                -4.9474958514e-12,
                -4.9475550211e-12,
                -4.9476087877e-12,
                -4.9476575261e-12,
                -4.9477015945e-12,
                -4.9477413348e-12,
                -4.9477770725e-12,
                -4.9478091173e-12,
                -4.9478377631e-12,
                -4.9478632882e-12,
                -4.9478859560e-12,
                -4.9479060147e-12,
                -4.9479236980e-12,
                -4.9479392256e-12,
                -4.9479528031e-12,
                -4.9479646227e-12,
                -4.9479748636e-12,
                -4.9479836920e-12,
                -4.9479912622e-12,
                -4.9479977164e-12,
                -4.9480031857e-12,
                -4.9480077898e-12,
                -4.9480116385e-12,
                -4.9480148312e-12,
                -4.9480174579e-12,
                -4.9480195998e-12,
                -4.9480213293e-12,
                -4.9480227111e-12,
                -4.9480238022e-12,
                -4.9480246527e-12,
            ]
        )

        np.testing.assert_allclose(
            rf_current.real,
            Iref_real,
            rtol=1e-6,
            atol=0,
            err_msg="In TestRFCurrent test_3, mismatch in real part of RF current",
        )

        Iref_imag = np.array(
            [
                -1.4943332367e-12,
                -1.4949861694e-12,
                -1.4956191965e-12,
                -1.4962326814e-12,
                -1.4968269874e-12,
                -1.4974024776e-12,
                -1.4979595147e-12,
                -1.4984984611e-12,
                -1.4990196784e-12,
                -1.4995235278e-12,
                -1.5000103691e-12,
                -1.5004805616e-12,
                -1.5009344630e-12,
                -1.5013724300e-12,
                -1.5017948176e-12,
                -1.5022019793e-12,
                -1.5025942669e-12,
                -1.5029720303e-12,
                -1.5033356174e-12,
                -1.5036853737e-12,
                -1.5040216428e-12,
                -1.5043447657e-12,
                -1.5046550808e-12,
                -1.5049529237e-12,
                -1.5052386275e-12,
                -1.5055125220e-12,
                -1.5057749341e-12,
                -1.5060261876e-12,
                -1.5062666027e-12,
                -1.5064964962e-12,
                -1.5067161815e-12,
                -1.5069259682e-12,
                -1.5071261621e-12,
                -1.5073170650e-12,
                -1.5074989748e-12,
                -1.5076721852e-12,
                -1.5078369857e-12,
                -1.5079936613e-12,
                -1.5081424928e-12,
                -1.5082837562e-12,
                -1.5084177232e-12,
                -1.5085446606e-12,
                -1.5086648303e-12,
                -1.5087784894e-12,
                -1.5088858903e-12,
                -1.5089872800e-12,
                -1.5090829007e-12,
                -1.5091729893e-12,
                -1.5092577775e-12,
                -1.5093374918e-12,
                -1.5094123534e-12,
                -1.5094825781e-12,
                -1.5095483764e-12,
                -1.5096099533e-12,
                -1.5096675084e-12,
                -1.5097212358e-12,
                -1.5097713242e-12,
                -1.5098179566e-12,
                -1.5098613109e-12,
                -1.5099015591e-12,
                -1.5099388680e-12,
                -1.5099733988e-12,
                -1.5100053072e-12,
                -1.5100347436e-12,
                -1.5100618529e-12,
                -1.5100867747e-12,
                -1.5101096432e-12,
                -1.5101305874e-12,
                -1.5101497310e-12,
                -1.5101671927e-12,
                -1.5101830859e-12,
                -1.5101975190e-12,
                -1.5102105955e-12,
                -1.5102224140e-12,
                -1.5102330683e-12,
                -1.5102426475e-12,
                -1.5102512361e-12,
                -1.5102589139e-12,
                -1.5102657566e-12,
                -1.5102718353e-12,
                -1.5102772170e-12,
                -1.5102819647e-12,
                -1.5102861373e-12,
                -1.5102897898e-12,
                -1.5102929738e-12,
                -1.5102957369e-12,
                -1.5102981235e-12,
                -1.5103001744e-12,
                -1.5103019276e-12,
                -1.5103034176e-12,
                -1.5103046762e-12,
                -1.5103057323e-12,
                -1.5103066124e-12,
                -1.5103073401e-12,
                -1.5103079370e-12,
                -1.5103084221e-12,
                -1.5103088126e-12,
                -1.5103091235e-12,
                -1.5103093683e-12,
                -1.5103095585e-12,
            ]
        )

        np.testing.assert_allclose(
            rf_current.imag,
            Iref_imag,
            rtol=1e-6,
            atol=0,
            err_msg="In TestRFCurrent test_3, mismatch in imaginary part of RF current",
        )

    # Test RF beam current on coarse grid integrated from fine grid
    # Compare to simulation data for peak RF current
    @pytest.mark.backend_mutation
    def test_4(self):
        t_rev = float(
            (2 * np.pi * self.rf.get_main_harmonic())
            / self.rf.calc_omega_rf_design(
                self.beam.reference.beta, self.ring.circumference
            )[self.rf.main_harmonic_idx]
        )
        t_rf = float(t_rev / self.rf.get_main_harmonic())
        # Create a batch of 100 equal, short bunches
        bunches = 100
        T_s = float(5 * t_rev / self.rf.get_main_harmonic())
        N_m = int(1e5)
        N_b = 2.3e11
        self.simulation.prepare_beam(
            beam=self.beam,
            preparation_routine=BiGaussian(
                n_macroparticles=self.n_macroparticles,
                sigma_dt=0.1e-9,
                sigma_dE=None,
                reinsertion=True,
                seed=1234,
            ),
        )
        beam2 = Beam(bunches * N_b, particle_type=self.beam.particle_type)
        beam2.setup_beam(
            dt=np.zeros(bunches * N_m),
            dE=np.zeros(bunches * N_m),
            reference_total_energy=self.beam.reference.total_energy,
        )
        bunch_spacing = 5 * t_rf
        buckets = 5 * bunches
        for i in range(bunches):
            beam2_dt = beam2.write_partial_dt()
            beam2_dt[i * N_m : (i + 1) * N_m] = (
                self.beam.read_partial_dt() + i * bunch_spacing
            )
            beam2_dE = beam2.write_partial_dE()
            beam2_dE[i * N_m : (i + 1) * N_m] = self.beam.read_partial_dE()
        profile2 = StaticProfile(
            cut_left=0,
            cut_right=bunches * bunch_spacing,
            n_bins=1000 * buckets,
        )
        profile2.track(beam2)

        tot_charges = (
            np.sum(profile2.hist_y) / beam2.common_array_size * beam2.intensity
        )
        self.assertAlmostEqual(tot_charges, 2.3000000000e13, 9)

        # Calculate fine- and coarse-grid RF current
        rf_current_fine, rf_current_coarse = rf_beam_current(
            beam2,
            profile2,
            float(self.omega_rf),
            use_lowpass_filter=False,
            downsample={"Ts": T_s, "points": self.rf.get_main_harmonic() / 5},
        )
        rf_current_coarse /= T_s

        # Peak RF current on coarse grid
        peak_rf_current = np.max(np.absolute(rf_current_coarse))
        self.assertAlmostEqual(peak_rf_current, 2.9284593979, 7)

    @pytest.mark.backend_mutation
    def test_5(self):
        self.simulation.prepare_beam(
            beam=self.beam,
            preparation_routine=BiGaussian(
                n_macroparticles=self.n_macroparticles,
                sigma_dt=3.2e-9 / 4,
                sigma_dE=None,
                reinsertion=True,
                seed=1234,
            ),
        )
        t_rev = float(
            (2 * np.pi * self.rf.get_main_harmonic())
            / self.rf.calc_omega_rf_design(
                self.beam.reference.beta, self.ring.circumference
            )[self.rf.main_harmonic_idx]
        )
        self.profile.track(self.beam)
        self.assertEqual(
            len(self.beam.read_partial_dt()),
            np.sum(self.profile.hist_y),
            "In"
            + " TestBeamCurrent: particle number mismatch in Beam vs Profile",
        )

        # RF current calculation with incorrect downsampling
        with self.assertRaises(RuntimeError):
            rf_current = rf_beam_current(
                self.beam,
                self.profile,
                self.omega,
                use_lowpass_filter=False,
                external_reference=False,
                downsample={"Ts": 25e-9},
            )


class TestIQ(unittest.TestCase):
    # Run before every test
    def setUp(self, f_rf=200.1e6, T_s=5e-10, n=1000):
        self.f_rf = f_rf  # initial frequency in Hz
        self.T_s = T_s  # sampling time
        self.n = n  # number of points

    # Run after every test
    def tearDown(self):
        del self.f_rf
        del self.T_s
        del self.n

    def test_1(self):
        # Define signal in range (-pi, pi)
        phases = np.pi * (
            np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1
        )
        signal = np.cos(phases) + 1j * np.sin(phases)
        # From IQ to polar
        amplitude, phase = cartesian_to_polar(signal)

        # Drop some digits to avoid rounding errors
        amplitude = np.around(amplitude, 12)
        phase = np.around(phase, 12)
        phases = np.around(phases, 12)
        self.assertSequenceEqual(
            amplitude.tolist(),
            np.ones(self.n).tolist(),
            msg="In TestIQ test_1, amplitude is not correct",
        )
        self.assertSequenceEqual(
            phase.tolist(),
            phases.tolist(),
            msg="In TestIQ test_1, phase is not correct",
        )

    def test_2(self):
        # Define signal in range (-pi, pi)
        phase = np.pi * (
            np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1
        )
        amplitude = np.ones(self.n)
        # From polar to IQ
        signal = polar_to_cartesian(amplitude, phase)

        # Drop some digits to avoid rounding errors
        signal_real = np.around(signal.real, 12)
        signal_imag = np.around(signal.imag, 12)
        theor_real = np.around(np.cos(phase), 12)  # what it should be
        theor_imag = np.around(np.sin(phase), 12)  # what it should be
        self.assertSequenceEqual(
            signal_real.tolist(),
            theor_real.tolist(),
            msg="In TestIQ test_2, real part is not correct",
        )
        self.assertSequenceEqual(
            signal_imag.tolist(),
            theor_imag.tolist(),
            msg="In TestIQ test_2, imaginary part is not correct",
        )

    def test_3(self):
        # Define signal in range (-pi, pi)
        phase = np.pi * (
            np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1
        )
        amplitude = np.ones(self.n)
        # Forwards and backwards transform
        signal = polar_to_cartesian(amplitude, phase)
        amplitude_new, phase_new = cartesian_to_polar(signal)

        # Drop some digits to avoid rounding errors
        phase = np.around(phase, 11)
        amplitude = np.around(amplitude, 11)
        amplitude_new = np.around(amplitude_new, 11)
        phase_new = np.around(phase_new, 11)
        self.assertSequenceEqual(
            phase.tolist(),
            phase_new.tolist(),
            msg="In TestIQ test_3, phase is not correct",
        )
        self.assertSequenceEqual(
            amplitude.tolist(),
            amplitude_new.tolist(),
            msg="In TestIQ test_3, amplitude is not correct",
        )

    def test_4(self):
        # Define signal in range (-pi, pi)
        phase = np.pi * (
            np.fmod(2 * np.arange(self.n) * self.f_rf * self.T_s, 2) - 1
        )
        signal = np.cos(phase) + 1j * np.sin(phase)
        # Forwards and backwards transform
        amplitude, phase = cartesian_to_polar(signal)
        signal_new = polar_to_cartesian(amplitude, phase)

        # Drop some digits to avoid rounding errors
        signal_real = np.around(signal.real, 11)
        signal_imag = np.around(signal.imag, 11)
        signal_real_2 = np.around(np.real(signal_new), 11)
        signal_imag_2 = np.around(np.imag(signal_new), 11)
        self.assertSequenceEqual(
            signal_real.tolist(),
            signal_real_2.tolist(),
            msg="In TestIQ test_4, real part is not correct",
        )
        self.assertSequenceEqual(
            signal_imag.tolist(),
            signal_imag_2.tolist(),
            msg="In TestIQ test_4, imaginary part is not correct",
        )


if __name__ == "__main__":
    unittest.main()
