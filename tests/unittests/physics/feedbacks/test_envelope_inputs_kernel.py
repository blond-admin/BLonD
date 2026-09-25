"""
Bit-identity tests for the compiled per-cell inputs of the envelope scan.

:mod:`~blond.physics.feedbacks.envelope_inputs_kernel` replaces NumPy
expressions that run over every coarse cell of every passage (the
propagator's ``B = e^L`` and ``W = (e^L - 1) / L``, and the frame-rotation
phasors). The compiled scan is pinned byte-for-byte to the per-cell Python
reference, so these kernels must reproduce the NumPy expressions they
replace exactly -- ``np.array_equal`` on complex128, not a tolerance.
"""

import unittest

import numpy as np

from blond.physics.feedbacks.cavity_solvers import (
    coarse_step_exponent,
    exponential_drive_weight,
    exponential_voltage_multiplier,
)
from blond.physics.feedbacks.envelope_inputs_kernel import (
    step_multipliers,
    unit_phasors,
)

#: Loaded quality factors spanning a heavily loaded to a nearly bare cavity.
Q_LOADED_VALUES = (1.0e2, 1.29e4, 1.0e7)
#: Relative detunings ``delta_omega / omega``, both signs and zero.
RELATIVE_DETUNINGS = (0.0, 3.1e-4, -3.1e-4, 0.3, -0.3)


def _omega_times_dt_samples() -> dict[str, np.ndarray]:
    """
    Per-cell ``omega * dt`` inputs the kernels must handle.

    Returns
    -------
    samples
        Named, strictly positive step arrays: one RF period with the jitter
        a ramped grid has, a log-uniform spread over eleven decades, and a
        single cell.
    """
    rng = np.random.default_rng(20260925)
    return {
        "one_rf_period": 2.0
        * np.pi
        * (1.0 + 1.0e-6 * rng.standard_normal(20_000)),
        "eleven_decades": 10.0 ** rng.uniform(-9.0, 2.0, 20_000),
        "single_cell": np.array([2.0 * np.pi]),
    }


class TestStepMultipliers(unittest.TestCase):
    """``B`` and ``W`` equal the NumPy propagator weights to the bit."""

    def _assert_bit_identical_to_numpy(self, omega_times_dt, q_loaded, rel):
        step_exponent = coarse_step_exponent(omega_times_dt, q_loaded, rel)
        expected_multiplier = exponential_voltage_multiplier(step_exponent)
        expected_weight = exponential_drive_weight(step_exponent)

        voltage_multiplier, drive_weight = step_multipliers(
            omega_times_dt, q_loaded, rel
        )

        self.assertTrue(
            np.array_equal(voltage_multiplier, expected_multiplier),
            msg=(
                "voltage multiplier differs in "
                f"{np.sum(voltage_multiplier != expected_multiplier)} cells"
            ),
        )
        self.assertTrue(
            np.array_equal(drive_weight, expected_weight),
            msg=(
                "drive weight differs in "
                f"{np.sum(drive_weight != expected_weight)} cells"
            ),
        )

    def test_bit_identical_to_the_numpy_weights(self):
        for name, omega_times_dt in _omega_times_dt_samples().items():
            for q_loaded in Q_LOADED_VALUES:
                for rel in RELATIVE_DETUNINGS:
                    with self.subTest(sample=name, Q_L=q_loaded, rel=rel):
                        self._assert_bit_identical_to_numpy(
                            omega_times_dt, q_loaded, rel
                        )

    def test_outputs_are_complex128_shaped_like_the_steps(self):
        omega_times_dt = np.full(7, 2.0 * np.pi)
        voltage_multiplier, drive_weight = step_multipliers(
            omega_times_dt, 1.29e4, 3.1e-4
        )
        for weights in (voltage_multiplier, drive_weight):
            self.assertEqual(weights.dtype, np.complex128)
            self.assertEqual(weights.shape, (7,))

    def test_an_empty_span_gives_empty_weights(self):
        voltage_multiplier, drive_weight = step_multipliers(
            np.empty(0), 1.29e4, 3.1e-4
        )
        self.assertEqual(voltage_multiplier.shape, (0,))
        self.assertEqual(drive_weight.shape, (0,))


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
