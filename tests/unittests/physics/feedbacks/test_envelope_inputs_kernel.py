"""
Bit-identity tests for the compiled per-cell frame rotations.

:mod:`~blond.physics.feedbacks.envelope_inputs_kernel` replaces a NumPy
expression that runs over every coarse cell of every passage (the frame
rotation phasors). The compiled scan is pinned byte-for-byte to the
per-cell Python reference, so the kernel must reproduce the NumPy
expression it replaces exactly -- ``np.array_equal`` on complex128, not a
tolerance.
"""

import unittest

import numpy as np

from blond.physics.feedbacks.envelope_inputs_kernel import unit_phasors


class TestUnitPhasors(unittest.TestCase):
    """``exp(sign * i * phase)``, exactly ``1 + 0j`` at a zero phase."""

    @staticmethod
    def _phases() -> np.ndarray:
        rng = np.random.default_rng(7)
        return np.concatenate(
            [
                rng.uniform(-np.pi, np.pi, 10_000),
                10.0 ** rng.uniform(-12.0, 4.0, 5_000),
                -(10.0 ** rng.uniform(-12.0, 4.0, 5_000)),
                np.zeros(50),
                np.full(50, -0.0),
            ]
        )

    def test_bit_identical_to_the_numpy_expression(self):
        phases = self._phases()
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                expected = np.where(
                    phases == 0.0, 1.0 + 0.0j, np.exp(sign * 1j * phases)
                )
                phasors = unit_phasors(phases, sign)
                self.assertTrue(
                    np.array_equal(phasors, expected),
                    msg=f"{np.sum(phasors != expected)} phasors differ",
                )

    def test_a_zero_phase_is_exactly_one_with_a_positive_zero(self):
        phasors = unit_phasors(np.array([0.0, -0.0]), -1.0)
        np.testing.assert_array_equal(phasors.real, [1.0, 1.0])
        np.testing.assert_array_equal(np.signbit(phasors.imag), [False, False])

    def test_outputs_are_complex128_shaped_like_the_phases(self):
        phasors = unit_phasors(np.zeros(5), 1.0)
        self.assertEqual(phasors.dtype, np.complex128)
        self.assertEqual(phasors.shape, (5,))


if __name__ == "__main__":
    unittest.main()
