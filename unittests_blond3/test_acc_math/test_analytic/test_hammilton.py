import unittest

import numpy as np
from matplotlib import pyplot as plt

from blond3.acc_math.analytic.hammilton import (
    calc_phi_s_single_harmonic,
    phase_modulo_above_transition,
    phase_modulo_below_transition,
    single_rf_sin_hamiltonian,
    is_in_separatrix,
)


class TestPhiS(unittest.TestCase):
    def test_phi_s_1(self):
        xs = np.linspace(-1, 1, 200)
        charge = 5
        voltage = 1.5
        omega = 5
        phi = 1.5
        ys = charge * voltage * np.sin(omega * xs + phi)
        plt.plot(xs, ys)
        energy_gain = 4
        plt.axhline(energy_gain)
        for above_transition in (0, 1):
            phi_s = calc_phi_s_single_harmonic(
                charge, voltage, phi, energy_gain, above_transition=above_transition
            )
            t_s = phi_s / (omega)
            plt.axvline(t_s)
            # plt.show()
            self.assertAlmostEqual(
                charge * voltage * np.sin(omega * t_s + phi), energy_gain
            )
        # todo add assertion


class TestFunctions(unittest.TestCase):
    def test_phase_modulo_above_transition(self):
        upper_limit = +3 * np.pi / 2.0
        lower_limit = -np.pi / 2
        phis = np.linspace(-100, 100, 200)
        phis_corrected = phase_modulo_above_transition(phis)
        plt.title("phase_modulo_above_transition")
        plt.plot(phis, "o")
        plt.plot(phis_corrected, "o")
        plt.axhline(upper_limit)
        plt.axhline(lower_limit)
        # plt.show()
        self.assertTrue(
            np.all(phis_corrected <= upper_limit),
            msg=f"{phis_corrected.max()=}",
        )
        self.assertTrue(
            np.all(phis_corrected >= lower_limit),
            msg=f"{phis_corrected.min()=}",
        )

    def test_phase_modulo_below_transition(self):
        upper_limit = +3 * np.pi / 2.0
        lower_limit = -np.pi / 2
        phis = np.linspace(-100, 100, 200)
        phis_corrected = phase_modulo_below_transition(phis)
        plt.title("phase_modulo_below_transition")
        plt.plot(phis, "o")
        plt.plot(phis_corrected, "o")
        plt.axhline(upper_limit)
        plt.axhline(lower_limit)
        # plt.show()
        self.assertTrue(
            np.all(phis_corrected <= upper_limit),
            msg=f"{phis_corrected.max()=}",
        )
        self.assertTrue(
            np.all(phis_corrected >= lower_limit),
            msg=f"{phis_corrected.min()=}",
        )


class TestSingleRFSinHamiltonian(unittest.TestCase):
    def setUp(self):
        self.charge = 1.0  # elementary charge units
        self.harmonic = 10
        self.voltage = 1e6  # V
        self.omega_rf = 2 * np.pi * 1e6  # rad/s
        self.phi_rf_d = 0.0  # rad
        self.phi_s = np.pi / 6  # stable phase, rad
        self.etas = [-0.01]  # below transition
        self.beta = 0.9
        self.total_energy = 1e9  # eV
        self.ring_circumference = 100.0  # m

    def test_hamiltonian_at_separatrix_max(self):
        # Max point of separatrix in phase: phi_b = π - phi_s
        dt_sep_max = (np.pi - self.phi_s - self.phi_rf_d) / self.omega_rf
        dE_sep_max = 0.0  # maximum in phase, energy = 0

        H = single_rf_sin_hamiltonian(
            charge=self.charge,
            harmonic=self.harmonic,
            voltage=self.voltage,
            omega_rf=self.omega_rf,
            phi_rf_d=self.phi_rf_d,
            phi_s=self.phi_s,
            etas=self.etas,
            beta=self.beta,
            total_energy=self.total_energy,
            ring_circumference=self.ring_circumference,
            dt=dt_sep_max,
            dE=dE_sep_max,
        )
        self.fail()  # TODO

    def test_is_in_separatrix(self):
        self.fail()  # TODO
        is_in_separatrix(
            charge=self.charge,
            harmonic=self.harmonic,
            voltage=self.voltage,
            omega_rf=self.omega_rf,
            phi_rf_d=self.phi_rf_d,
            phi_s=self.phi_s,
            etas=self.etas,
            beta=self.beta,
            total_energy=self.total_energy,
            ring_circumference=self.ring_circumference,
            dt=dt_sep_max,
            dE=dE_sep_max,
        )


class TestIsInSeparatrix(unittest.TestCase):
    def test1(self):
        is_in_separatrix
        self.fail()  # TODO


if __name__ == "__main__":
    unittest.main()
