import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond.acc_math.analytic.hamilton import phase_modulo_below_transition


class TestPhaseModuloBelowTransition(unittest.TestCase):
    def test_scalar_values(self):
        self.assertAlmostEqual(phase_modulo_below_transition(0.5), 0.5)
        self.assertAlmostEqual(
            phase_modulo_below_transition(-np.pi / 2), -np.pi / 2
        )
        self.assertAlmostEqual(
            phase_modulo_below_transition(3 * np.pi), -np.pi
        )
        self.assertAlmostEqual(
            phase_modulo_below_transition(-3 * np.pi / 2), np.pi / 2
        )

    def test_array_values(self):
        phi = np.linspace(-10, 10)
        result = phase_modulo_below_transition(phi)
        DEV_PLOT = False
        if DEV_PLOT:
            plt.plot(phi)
            plt.plot(result)
            plt.show()
        self.assertTrue(np.all(result < np.pi))
        self.assertTrue(np.all(result >= -np.pi))


if __name__ == "__main__":
    unittest.main()
