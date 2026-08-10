import unittest

import numpy as np

from blond.physics.feedbacks.beam_current import low_pass_filter
from blond.physics.feedbacks.cavity_solvers import (
    cavity_response_sparse_matrix,
)
from blond.physics.feedbacks.iq import (
    cartesian_to_polar,
    polar_to_cartesian,
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
        # Define signal in range (-pi, pi)1
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


class TestACSSparseModel(unittest.TestCase):
    def test_acs_model_vs_euler_forward_no_beam(self):
        def ACS_model_euler_forward(
            n_samples,
            V_init,
            I_beam,
            I_gen,
            R_over_Q_,
            Q_L_,
            detuning_,
            omega_times_dt_,
        ):
            def cavity_response(
                omega_times_dt: float,
                R_over_Q,
                i_gen,
                i_beam,
                v_ant,
                detuning,
                Q_L,
            ):
                r"""ACS cavity response model"""

                return (
                    i_gen * R_over_Q * omega_times_dt
                    + v_ant
                    * (
                        1
                        - 0.5 * omega_times_dt / Q_L
                        + 1j * detuning * omega_times_dt
                    )
                    - i_beam * 0.5 * R_over_Q * omega_times_dt
                )

            voltage = np.zeros(len(I_gen) + 1, dtype=complex)
            voltage[0] = V_init
            for _i in range(1, n_samples + 1):
                voltage[_i] = cavity_response(
                    omega_times_dt_,
                    R_over_Q_,
                    I_gen[_i - 1],
                    I_beam[_i - 1],
                    voltage[_i - 1],
                    detuning_,
                    Q_L_,
                )
            return voltage[1:]

        n_samples = 1000
        omega_times_dt = 2 * np.pi / 1
        I_beam = np.zeros(n_samples)
        I_gen = (0.2565950699764863 + 0.004372312359083769j) * np.ones(
            n_samples
        )
        R_over_Q_in = 518
        Q_L_in = 1e3
        V_ant_init_in = 15e4
        rel_detuning_in = -3.52881428058616e-07
        res_sparse_matrix = cavity_response_sparse_matrix(
            R_over_Q=R_over_Q_in,
            Q_L=Q_L_in,
            V_ant_init=V_ant_init_in,
            omega_times_dt=omega_times_dt,
            I_gen_init=I_gen[0],
            I_beam=I_beam,  # shortening due to internal extension
            I_gen=I_gen,
            relative_detuning=rel_detuning_in,
        )

        res_euler_forward = ACS_model_euler_forward(
            n_samples,
            V_ant_init_in,
            I_beam,
            I_gen,
            R_over_Q_in,
            Q_L_in,
            rel_detuning_in,
            omega_times_dt,
        )

        DEBUG_PLOT = False
        if DEBUG_PLOT:
            import matplotlib.pyplot as plt

            plt.clf()
            plt.plot(res_sparse_matrix)
            plt.show()

        np.testing.assert_allclose(res_euler_forward, res_sparse_matrix)

        with self.assertRaisesRegex(AssertionError, "length of "):
            _ = cavity_response_sparse_matrix(
                R_over_Q=R_over_Q_in,
                Q_L=Q_L_in,
                V_ant_init=V_ant_init_in,
                omega_times_dt=omega_times_dt,
                I_beam=I_beam[1:],
                I_gen=I_gen,
                relative_detuning=rel_detuning_in,
                I_gen_init=I_gen[0],
            )
        with self.assertRaisesRegex(AssertionError, "length of "):
            _ = cavity_response_sparse_matrix(
                R_over_Q=R_over_Q_in,
                Q_L=Q_L_in,
                V_ant_init=V_ant_init_in,
                omega_times_dt=omega_times_dt,
                I_beam=I_beam,
                I_gen=I_gen[1:],
                I_gen_init=I_gen[0],
                relative_detuning=rel_detuning_in,
            )


if __name__ == "__main__":
    unittest.main()
