import unittest

import numpy as np
from matplotlib import pyplot as plt

from blond3.acc_math.analytic.hammilton import calc_phi_s_single_harmonic


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
        for above_transition in (0,1):
            phi_s = calc_phi_s_single_harmonic(
                charge, voltage, phi, energy_gain, above_transition=above_transition
            )
            t_s = phi_s / (omega)
            plt.axvline(t_s)
        # plt.show()
            self.assertAlmostEqual(charge * voltage * np.sin(omega * t_s + phi),energy_gain)
        # todo add assertion


if __name__ == "__main__":
    unittest.main()
