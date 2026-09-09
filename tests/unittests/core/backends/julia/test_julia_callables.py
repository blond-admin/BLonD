"""Compare the Julia backends against the python reference backend."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pytest

from blond.core.backends.julia.julia_env import is_julia_available
from blond.core.backends.python.callables import PythonSpecials
from blond.generals.cupy_.no_cupy_import import copy_to_cpu

CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None

RTOL = 1e-12


def _skip_if_julia_cannot_start(test_case: unittest.TestCase) -> None:
    """Skip when this process cannot load Julia at all.

    Parameters
    ----------
    test_case
        The running test case.
    """
    from blond.core.backends.julia.julia_env import (
        ensure_julia_environment,
    )

    try:
        ensure_julia_environment()
    except OSError as error:  # e.g. an incompatible libstdc++
        test_case.skipTest(str(error))


class _JuliaKernelChecks:
    """Kernel-by-kernel comparison against `PythonSpecials`.

    Mixed into a `unittest.TestCase` twice, once per Julia backend. The
    subclass provides `specials` (the Julia `Specials` instance) and
    `xp` (the array module the backend expects, numpy or cupy).
    """

    specials = None
    xp = np

    def array(self, values, dtype=np.float64):
        """Build a contiguous device array of `dtype`.

        Parameters
        ----------
        values
            Array-like input values.
        dtype
            Element type of the resulting array.

        Returns
        -------
        array
            Array on the device of the backend under test.
        """
        return self.xp.ascontiguousarray(
            self.xp.asarray(np.asarray(values, dtype=dtype))
        )

    def assert_close(self, julia_result, python_result, rtol=RTOL):
        """Compare a backend result against the python reference.

        Parameters
        ----------
        julia_result
            Result produced by the Julia backend.
        python_result
            Result produced by `PythonSpecials`.
        rtol
            Relative tolerance.
        """
        np.testing.assert_allclose(
            np.asarray(copy_to_cpu(julia_result)),
            np.asarray(python_result),
            rtol=rtol,
        )

    def test_get_max_threads(self) -> None:
        """The reported thread count must be a positive integer."""
        self.assertGreaterEqual(int(self.specials.get_max_threads()), 1)

    def test_kick_single_harmonic(self) -> None:
        """Single-harmonic kick must match the python backend."""
        dt_np = np.linspace(1e-9, 10e-9, 32)
        dE_np = np.linspace(1e9, 10e9, 32)
        dt, dE = self.array(dt_np), self.array(dE_np)
        args = (1e3, 2 * np.pi * 400e3, 0.3, 1.0, -1.0)
        self.specials.kick_single_harmonic(dt, dE, *args)
        PythonSpecials.kick_single_harmonic(dt_np, dE_np, *args)
        self.assert_close(dE, dE_np)

    def test_kick_multi_harmonic(self) -> None:
        """Multi-harmonic kick must match the python backend."""
        dt_np = np.linspace(1e-9, 10e-9, 32)
        dE_np = np.linspace(1e9, 10e9, 32)
        voltage_np = np.linspace(1e6, 5e6, 3)
        omega_np = np.linspace(200e6, 400e6, 3)
        phi_np = np.linspace(0.0, 2 * np.pi, 3)
        self.specials.kick_multi_harmonic(
            self.array(dt_np),
            (dE := self.array(dE_np)),
            self.array(voltage_np),
            self.array(omega_np),
            self.array(phi_np),
            1.0,
            3,
            -1.0,
        )
        PythonSpecials.kick_multi_harmonic(
            dt_np, dE_np, voltage_np, omega_np, phi_np, 1.0, 3, -1.0
        )
        self.assert_close(dE, dE_np)

    def test_drift_simple(self) -> None:
        """Simple drift must match the python backend."""
        dt_np = np.linspace(1e-9, 10e-9, 32)
        dE_np = np.linspace(1e9, 10e9, 32)
        dt = self.array(dt_np)
        args = (10.0, 0.3, 0.9, 10.0)
        self.specials.drift_simple(dt, self.array(dE_np), *args)
        PythonSpecials.drift_simple(dt_np, dE_np, *args)
        self.assert_close(dt, dt_np)

    def test_drift_exact(self) -> None:
        """Exact drift must match the python backend."""
        dt_np = np.linspace(1e-9, 10e-9, 32)
        dE_np = np.linspace(1e9, 10e9, 32)
        higher_alpha_np = np.array([1.0, 1.0])
        dt = self.array(dt_np)
        self.specials.drift_exact(
            dt,
            self.array(dE_np),
            10.0,
            1.0,
            self.array(higher_alpha_np),
            0.9,
            10.0,
        )
        PythonSpecials.drift_exact(
            dt_np, dE_np, 10.0, 1.0, higher_alpha_np, 0.9, 10.0
        )
        self.assert_close(dt, dt_np)

    def test_loss_box(self) -> None:
        """Out-of-box particles must be flagged like in python."""
        dt_np = np.linspace(0.0, 10.0, 21)
        dE_np = np.linspace(-5.0, 5.0, 21)
        flags_np = np.ones(21, dtype=np.int32)
        flags = self.array(flags_np, dtype=np.int32)
        args = (3.0, -3.0, 2.0, 8.0)
        self.specials.loss_box(
            *args, self.array(dt_np), self.array(dE_np), flags
        )
        PythonSpecials.loss_box(*args, dt_np, dE_np, flags_np)
        self.assert_close(flags, flags_np)

    def test_sum_1d_array(self) -> None:
        """Array sum must match the python backend."""
        values = np.linspace(-3.0, 7.0, 101)
        self.assert_close(
            self.specials.sum_1d_array(self.array(values)),
            PythonSpecials.sum_1d_array(values),
        )

    def test_dot_product_1d_array(self) -> None:
        """Dot product must match the python backend."""
        first = np.linspace(-3.0, 7.0, 101)
        second = np.linspace(2.0, -4.0, 101)
        self.assert_close(
            self.specials.dot_product_1d_array(
                self.array(first), self.array(second)
            ),
            PythonSpecials.dot_product_1d_array(first, second),
        )

    def test_histogram(self) -> None:
        """Histogram must match the python backend bin for bin."""
        values = np.linspace(-1.0, 2.0, 257)
        out_np = np.zeros(16)
        out = self.array(out_np)
        self.specials.histogram(self.array(values), out, 0.0, 1.0)
        PythonSpecials.histogram(values, out_np, 0.0, 1.0)
        self.assert_close(out, out_np)

    def test_beam_phase(self) -> None:
        """Beam phase must match the python backend."""
        hist_x = np.linspace(0.0, 5e-9, 64)
        hist_y = np.exp(-(((hist_x - 2.5e-9) / 1e-9) ** 2))
        args = (1e8, 2 * np.pi * 400e6, 0.3, float(hist_x[1] - hist_x[0]))
        julia_phase = self.specials.beam_phase(
            self.array(hist_x), self.array(hist_y), *args
        )
        python_phase = PythonSpecials.beam_phase(hist_x, hist_y, *args)
        self.assert_close(julia_phase, python_phase, rtol=1e-10)

    def test_kick_interpolated_dense(self) -> None:
        """Dense interpolated kick must match the python backend."""
        bin_centers_np = np.linspace(0.0, 10e-9, 33)
        voltage_np = np.linspace(-1e6, 1e6, 33)
        dt_np = np.linspace(-2e-9, 12e-9, 64)
        dE_np = np.zeros(64)
        dE = self.array(dE_np)
        self.specials.kick_interpolated(
            self.array(dt_np),
            dE,
            self.array(voltage_np),
            self.array(bin_centers_np),
            1.0,
            -1.0,
        )
        PythonSpecials.kick_interpolated(
            dt_np, dE_np, voltage_np, bin_centers_np, 1.0, -1.0
        )
        self.assert_close(dE, dE_np)

    def test_kick_interpolated_rejects_non_uniform_bin_centers(self) -> None:
        """Non-uniform `bin_centers` must raise the documented error."""
        bin_centers = np.array([0.0, 1e-9, 2e-9, 9e-9, 10e-9])
        with self.assertRaises(ValueError) as context:
            self.specials.kick_interpolated(
                self.array(np.zeros(4)),
                self.array(np.zeros(4)),
                self.array(np.zeros(5)),
                self.array(bin_centers),
                1.0,
                0.0,
            )
        self.assertIn("not uniformly spaced", str(context.exception))

    def _sparse_metadata(self):
        """Return sparse profile metadata shared by the sparse tests.

        Returns
        -------
        metadata
            Dictionary of the sparse-metadata keyword arguments.
        """
        return {
            "first_left_cut": 0.0,
            "left_cut_distance": 10e-9,
            "cut_width": 5e-9,
            "bins_per_profile": 8,
        }

    def test_kick_interpolated_sparse(self) -> None:
        """Sparse interpolated kick must match the python backend."""
        metadata = self._sparse_metadata()
        filling_pattern_np = np.array([True, False, True])
        bucket_to_memory_np = np.array([0, 0, 8], dtype=np.int32)
        n_bins = 16
        voltage_np = np.linspace(-1e6, 1e6, n_bins)
        bin_centers_np = np.concatenate(
            [
                np.linspace(0.3125e-9, 4.6875e-9, 8),
                np.linspace(20.3125e-9, 24.6875e-9, 8),
            ]
        )
        dt_np = np.linspace(-1e-9, 26e-9, 128)
        dE_np = np.zeros(128)
        dE = self.array(dE_np)
        self.specials.kick_interpolated(
            self.array(dt_np),
            dE,
            self.array(voltage_np),
            self.array(bin_centers_np),
            1.0,
            -1.0,
            filling_pattern=self.array(filling_pattern_np, dtype=np.bool_),
            bucket_index_to_memory_index=self.array(
                bucket_to_memory_np, dtype=np.int32
            ),
            **metadata,
        )
        PythonSpecials.kick_interpolated(
            dt_np,
            dE_np,
            voltage_np,
            bin_centers_np,
            1.0,
            -1.0,
            filling_pattern=filling_pattern_np,
            bucket_index_to_memory_index=bucket_to_memory_np,
            **metadata,
        )
        self.assert_close(dE, dE_np)

    def test_histogram_sparse(self) -> None:
        """Sparse histogram must match the python backend."""
        metadata = self._sparse_metadata()
        filling_pattern_np = np.array([True, False, True])
        bucket_to_memory_np = np.array([0, 0, 8], dtype=np.int32)
        x_np = np.linspace(-1e-9, 26e-9, 512)
        out_np = np.zeros(16)
        out = self.array(out_np)
        self.specials.histogram_sparse(
            self.array(x_np),
            out,
            metadata["first_left_cut"],
            metadata["left_cut_distance"],
            metadata["cut_width"],
            metadata["bins_per_profile"],
            2,
            self.array(filling_pattern_np, dtype=np.bool_),
            self.array(bucket_to_memory_np, dtype=np.int32),
        )
        PythonSpecials.histogram_sparse(
            x_np,
            out_np,
            metadata["first_left_cut"],
            metadata["left_cut_distance"],
            metadata["cut_width"],
            metadata["bins_per_profile"],
            2,
            filling_pattern_np,
            bucket_to_memory_np,
        )
        self.assert_close(out, out_np)

    def test_move_flagged_elements_to_end(self) -> None:
        """Flagged particles must end up at the array end."""
        flags_np = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int32)
        dt_np = np.arange(8, dtype=np.float64)
        dE_np = 10.0 + np.arange(8, dtype=np.float64)
        ids_np = np.arange(8, dtype=np.int32)
        flags = self.array(flags_np, dtype=np.int32)
        dt = self.array(dt_np)
        dE = self.array(dE_np)
        ids = self.array(ids_np, dtype=np.int32)
        n_new = self.specials.move_flagged_elements_to_end(
            0, flags, dt, dE, ids
        )
        n_new_python = PythonSpecials.move_flagged_elements_to_end(
            0, flags_np, dt_np, dE_np, ids_np
        )
        self.assertEqual(int(n_new), int(n_new_python))
        kept = int(n_new_python)
        ids_after = np.asarray(copy_to_cpu(ids))
        self.assertEqual(
            sorted(ids_after[:kept].tolist()),
            sorted(ids_np[:kept].tolist()),
        )
        flags_after = np.asarray(copy_to_cpu(flags))
        self.assertTrue(bool(np.all(flags_after[:kept] != 0)))
        self.assertTrue(bool(np.all(flags_after[kept:] == 0)))

    def test_wake_from_pole_residue(self) -> None:
        """Pole/residue wake must match the python backend."""
        n_bins = 32
        n_poles = 3
        profile_np = np.linspace(1.0, 2.0, n_bins)
        profile_dts_np = np.linspace(0.0, 3.1e-9, n_bins + 1)
        poles_np = np.array(
            [-1e8 + 2e9j, -2e8 + 0j, -3e8 + 5e9j], dtype=np.complex128
        )
        residues_np = np.array(
            [1e6 + 1e5j, 2e6 + 0j, 3e6 - 1e5j], dtype=np.complex128
        )
        signs_np = np.array([1.0, -1.0, 1.0])
        update_on_bin_np = np.array([0], dtype=np.int32)
        states_np = np.zeros(n_poles + 1, dtype=np.complex128)
        voltage_np = np.zeros(n_bins)

        states = self.array(states_np, dtype=np.complex128)
        voltage = self.array(voltage_np)
        self.specials.wake_from_pole_residue(
            self.array(profile_np),
            self.array(profile_dts_np),
            self.array(poles_np, dtype=np.complex128),
            self.array(residues_np, dtype=np.complex128),
            True,
            self.array(signs_np),
            self.array(update_on_bin_np, dtype=np.int32),
            1.0,
            states,
            voltage,
            self.array(
                np.zeros(
                    (int(self.specials.get_max_threads()), n_bins)
                ).ravel()
            ),
        )
        PythonSpecials.wake_from_pole_residue(
            profile_np,
            profile_dts_np,
            poles_np,
            residues_np,
            True,
            signs_np,
            update_on_bin_np,
            1.0,
            states_np,
            voltage_np,
            np.zeros((1, n_bins)),
        )
        self.assert_close(voltage, voltage_np, rtol=1e-10)
        self.assert_close(states, states_np, rtol=1e-10)

    def test_synchrotron_radiation_without_quantum_excitation(self) -> None:
        """The noiseless kick must match the python backend exactly."""
        dE_np = np.linspace(-1e6, 1e6, 64)
        dE = self.array(dE_np)
        args = (1e3, 100.0, 1e-3, 1e9)
        self.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
            dE, *args, disable_quantum_excitation=True
        )
        PythonSpecials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
            dE_np, *args, disable_quantum_excitation=True
        )
        self.assert_close(dE, dE_np)

    def test_synchrotron_radiation_noise_statistics(self) -> None:
        """With quantum excitation the noise statistics must be right."""
        n_macroparticles = 200_000
        longitudinal_damping_time = 100.0
        natural_energy_spread = 1e-3
        total_energy = 1e9
        dE = self.array(np.zeros(n_macroparticles))
        self.specials.apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
            dE,
            0.0,
            longitudinal_damping_time,
            natural_energy_spread,
            total_energy,
            disable_quantum_excitation=False,
        )
        expected_scale = (
            2.0
            * natural_energy_spread
            / np.sqrt(longitudinal_damping_time)
            * total_energy
        )
        result = np.asarray(copy_to_cpu(dE))
        self.assertAlmostEqual(
            float(np.std(result)) / expected_scale, 1.0, delta=0.05
        )
        self.assertLess(abs(float(np.mean(result))), 0.05 * expected_scale)


@pytest.mark.julia
class TestJuliaCpuSpecials(_JuliaKernelChecks, unittest.TestCase):
    """Run every kernel check on the `julia_cpu` backend."""

    def setUp(self) -> None:
        if not is_julia_available():
            self.skipTest("juliacall is not installed")
        from blond.core.backends.julia.callables import JuliaCpuSpecials

        self.specials = JuliaCpuSpecials()
        self.xp = np
        _skip_if_julia_cannot_start(self)

    def test_music_track(self) -> None:
        """MuSiC on the CPU must match the python backend."""
        n_macroparticles = 64
        beam_dt = np.linspace(1e-9, 10e-9, n_macroparticles)
        beam_dE_julia = np.linspace(1e6, 2e6, n_macroparticles)
        beam_dE_python = beam_dE_julia.copy()
        induced_voltage_julia = np.zeros(n_macroparticles)
        induced_voltage_python = np.zeros(n_macroparticles)
        parameter_array_julia = np.array([1.0, 0.0, 0.0])
        parameter_array_python = parameter_array_julia.copy()
        coefficients = (
            3.14e9,
            6.28e9,
            -1e3,
            -0.5,
            -2.0,
            3.0,
            0.5,
        )
        self.specials.music_track(
            beam_dt,
            beam_dE_julia,
            induced_voltage_julia,
            parameter_array_julia,
            *coefficients,
            10.0,
            True,
        )
        PythonSpecials.music_track(
            beam_dt,
            beam_dE_python,
            induced_voltage_python,
            parameter_array_python,
            *coefficients,
            10.0,
            True,
        )
        self.assert_close(induced_voltage_julia, induced_voltage_python)
        self.assert_close(beam_dE_julia, beam_dE_python)
        self.assert_close(parameter_array_julia, parameter_array_python)


@pytest.mark.julia
@pytest.mark.cupy
class TestJuliaGpuSpecials(_JuliaKernelChecks, unittest.TestCase):
    """Run every kernel check on the `julia_gpu` backend."""

    def setUp(self) -> None:
        if not is_julia_available():
            self.skipTest("juliacall is not installed")
        if not CUPY_AVAILABLE:
            self.skipTest("cupy is not installed")
        import cupy as cp  # type: ignore

        from blond.core.backends.julia.callables import JuliaGpuSpecials

        self.specials = JuliaGpuSpecials()
        self.xp = cp
        _skip_if_julia_cannot_start(self)

    def test_music_track_raises(self) -> None:
        """MuSiC is not implemented on the GPU."""
        with self.assertRaises(NotImplementedError):
            self.specials.music_track(
                self.array(np.zeros(4)),
                self.array(np.zeros(4)),
                self.array(np.zeros(4)),
                self.array(np.zeros(3)),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                False,
            )
