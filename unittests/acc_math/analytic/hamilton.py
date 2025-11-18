import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond.acc_math.analytic.hamilton import (
    phase_modulo_above_transition,
    phase_modulo_below_transition,
)


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



class TestPhaseModuloAboveTransition(unittest.TestCase):
    def test_scalar_values(self):
        # 0 stays 0
        self.assertAlmostEqual(phase_modulo_above_transition(0.0), 0.0)

        # Positive values below 2π remain unchanged
        self.assertAlmostEqual(phase_modulo_above_transition(np.pi / 2), np.pi / 2)

        # Values above 2π wrap around
        self.assertAlmostEqual(phase_modulo_above_transition(3 * np.pi), np.pi)

        # Negative values wrap into the positive range
        self.assertAlmostEqual(
            phase_modulo_above_transition(-np.pi / 2),
            3 * np.pi / 2
        )

    def test_array_values(self):
        phi = np.linspace(-10, 10)
        result = phase_modulo_above_transition(phi)

        # All results should be within [0, 2π)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < 2 * np.pi))

    def test_periodicity(self):
        # Check that adding 2π doesn't change the result
        vals = np.linspace(-5, 5, 10)
        self.assertTrue(
            np.allclose(
                phase_modulo_above_transition(vals),
                phase_modulo_above_transition(vals + 2 * np.pi)
            )
        )

if __name__ == "__main__":
    unittest.main()
