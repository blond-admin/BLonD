"""The tiled-GEMM far field of `CudaSpecials.no_parity`, against the kernel.

`wake_from_pole_residue_gemm` evaluates `Specials.wake_from_pole_residue`
as two GEMMs per slab of tiles and a scan over the tiles, on the grid a
`pole_tile_gemm_plan` was built for. It must reproduce the kernel: voltage
and hand-over state, call after call.
"""

import numpy as np
import pytest

from blond.testing.backend_testing import BLonDTestCase

BIN_DT = 2.5e-11


def _grid(rng, n_runs, run_length, last_run_length):
    """Runs of `run_length` bins (the last `last_run_length`), random gaps."""
    lengths = [run_length] * (n_runs - 1) + [last_run_length]
    offsets, position = [], 0
    for length in lengths:
        offsets.append(position + np.arange(length))
        position += length + int(rng.integers(1, 300))
    return np.concatenate(offsets)


def _poles(rng, n_poles):
    frequency = rng.uniform(1e8, 5e9, n_poles)
    quality = 10 ** rng.uniform(-0.2, 4.5, n_poles)  # down to fast poles
    poles = -2 * np.pi * frequency / (2 * quality) + 2j * np.pi * frequency
    real = np.arange(n_poles) % 5 == 0
    poles[real] = poles[real].real
    residues = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
    residues[real] = residues[real].real
    signs = np.where(np.arange(n_poles) % 3 == 0, -1.0, 1.0)
    return poles, residues, signs


@pytest.mark.cupy
class TestPoleTileGemm(BLonDTestCase):
    """The GEMM far field equals `CudaSpecials.wake_from_pole_residue`."""

    def setUp(self) -> None:
        try:
            import cupy as cp

            from blond.core.backends.cuda.callables import CudaSpecials
        except ImportError as exc:
            if "cupy" in str(exc):
                self.skipTest("Cupy not available")
            raise
        self.cp = cp
        self.specials = CudaSpecials

    def _plan(self, offsets, poles, residues, signs, **kwargs):
        cp = self.cp
        return self.specials.no_parity.pole_tile_gemm_plan(
            bin_offsets=offsets,
            poles=cp.asarray(poles, dtype=np.complex128),
            residues=cp.asarray(residues, dtype=np.complex128),
            counterrotating_pole_signs=cp.asarray(signs),
            bin_dt=BIN_DT,
            **kwargs,
        )

    def _check(self, offsets, n_poles, seed, **plan_kwargs):
        cp = self.cp
        rng = np.random.default_rng(seed)
        poles, residues, signs = _poles(rng, n_poles)
        plan = self._plan(offsets, poles, residues, signs, **plan_kwargs)
        self.assertIsNotNone(plan)
        states_in = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
        kernel_states = cp.asarray(states_in)
        gemm_states = cp.asarray(states_in)
        for call_i in range(3):
            carried_lag = float(rng.choice([1.0, 1.4, 2.0, 7.3, 300.0]))
            call = {
                "carried_charge": cp.asarray([rng.normal()]),
                "carried_is_counterrotating": bool(rng.random() < 0.5),
                "state_lag_dt": (carried_lag - 1.0) * BIN_DT,
                "carried_lag_dt": carried_lag * BIN_DT,
                "is_counterrotating_beam": bool(rng.random() < 0.5),
                "factor": float(rng.uniform(0.5, 2.0)),
            }
            profile = cp.asarray(rng.random(len(offsets)) * 1e3)
            expected = cp.zeros(len(offsets))
            self.specials.wake_from_pole_residue(
                profile_time=cp.asarray(3e-6 + offsets * BIN_DT),
                profile=profile,
                poles=cp.asarray(poles, dtype=np.complex128),
                residues=cp.asarray(residues, dtype=np.complex128),
                counterrotating_pole_signs=cp.asarray(signs),
                bin_dt=BIN_DT,
                states=kernel_states,
                voltage=expected,
                voltage_threaded=cp.zeros((1, len(offsets))),
                **call,
            )
            voltage = cp.full(len(offsets), np.nan)
            self.specials.no_parity.wake_from_pole_residue_gemm(
                plan=plan,
                profile=profile,
                states=gemm_states,
                voltage=voltage,
                **call,
            )
            expected = cp.asnumpy(expected)
            scale = float(np.max(np.abs(expected)))
            np.testing.assert_allclose(
                cp.asnumpy(voltage),
                expected,
                rtol=1e-9,
                atol=1e-9 * scale,
                err_msg=f"{call_i=}",
            )
            reference_states = cp.asnumpy(kernel_states)
            np.testing.assert_allclose(
                cp.asnumpy(gemm_states),
                reference_states,
                rtol=1e-9,
                atol=1e-9 * float(np.max(np.abs(reference_states))),
                err_msg=f"states {call_i=}",
            )

    def test_bunches_with_gaps(self) -> None:
        rng = np.random.default_rng(1)
        self._check(_grid(rng, 20, 128, 128), n_poles=37, seed=2)

    def test_partial_last_run(self) -> None:
        rng = np.random.default_rng(3)
        self._check(_grid(rng, 9, 64, 23), n_poles=21, seed=4)

    def test_single_run(self) -> None:
        self._check(np.arange(700), n_poles=24, seed=5)

    def test_single_bin(self) -> None:
        self._check(np.arange(1), n_poles=20, seed=6)

    def test_slabs_equal_one_pass(self) -> None:
        """A tiny memory budget forces one tile per slab; same result."""
        rng = np.random.default_rng(7)
        self._check(
            _grid(rng, 30, 64, 64), n_poles=33, seed=8, max_slab_bytes=1
        )

    def test_grid_without_common_tile_has_no_plan(self) -> None:
        offsets = np.concatenate((np.arange(10), 50 + np.arange(64)))
        poles, residues, signs = _poles(np.random.default_rng(9), 40)
        self.assertIsNone(self._plan(offsets, poles, residues, signs))

    def test_few_poles_have_no_plan(self) -> None:
        """Below 20 poles the kernel is faster; no plan is built."""
        poles, residues, signs = _poles(np.random.default_rng(10), 19)
        self.assertIsNone(self._plan(np.arange(256), poles, residues, signs))


@pytest.mark.cupy
class TestPoleNearField(BLonDTestCase):
    """`pole_near_field` equals the solver's near field and hand-over."""

    def setUp(self) -> None:
        try:
            import cupy as cp

            from blond.core.backends.cuda.callables import CudaSpecials
        except ImportError as exc:
            if "cupy" in str(exc):
                self.skipTest("Cupy not available")
            raise
        self.cp = cp
        self.specials = CudaSpecials

    def _inputs(self, seed, n_bins):
        rng = np.random.default_rng(seed)
        offsets = _grid(
            rng, 4, max(1, n_bins // 4), n_bins - 3 * (n_bins // 4)
        )
        profile = rng.random(len(offsets)) * 1e3
        adjacent = np.diff(offsets) == 1
        taps = rng.normal(size=3)
        voltage = rng.normal(size=len(offsets))
        return rng, profile, adjacent, taps, voltage

    @staticmethod
    def _near_field(profile, adjacent, taps, factor):
        near = factor * taps[1] * profile
        near[1:] += factor * taps[0] * adjacent * profile[:-1]
        near[:-1] += factor * taps[2] * adjacent * profile[1:]
        return near

    def _call(self, profile, adjacent, taps, voltage, carried, **kwargs):
        cp = self.cp
        voltage = cp.asarray(voltage)
        carried_charge = cp.asarray([carried])
        self.specials.no_parity.pole_near_field(
            profile=cp.asarray(profile),
            factor=1.7,
            taps=cp.asarray(taps),
            adjacent=cp.asarray(adjacent),
            carried_charge=carried_charge,
            voltage=voltage,
            **kwargs,
        )
        return cp.asnumpy(voltage), float(carried_charge[0])

    def test_constant_carried_wake(self) -> None:
        _, profile, adjacent, taps, voltage = self._inputs(1, 1000)
        result, carried = self._call(
            profile, adjacent, taps, voltage, carried=0.3, carried_wake=-2.5
        )
        expected = voltage + self._near_field(profile, adjacent, taps, 1.7)
        expected[0] += 0.3 * -2.5
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-9)
        self.assertAlmostEqual(carried, 1.7 * profile[-1], places=9)

    def test_carried_wake_summed_over_poles(self) -> None:
        cp = self.cp
        rng, profile, adjacent, taps, voltage = self._inputs(2, 517)
        poles, residues, _ = _poles(rng, 700)
        weights = rng.choice([-2.0, -1.0, 1.0, 2.0], size=700)
        shift_dt = 3.3 * BIN_DT
        result, _ = self._call(
            profile,
            adjacent,
            taps,
            voltage,
            carried=0.3,
            carried_poles=cp.asarray(poles),
            carried_residues=cp.asarray(residues),
            carried_weights=cp.asarray(weights),
            carried_shift_dt=shift_dt,
        )
        wake = np.sum(weights * (residues * np.exp(poles * shift_dt)).real)
        expected = voltage + self._near_field(profile, adjacent, taps, 1.7)
        expected[0] += 0.3 * wake
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-9)

    def test_single_bin(self) -> None:
        profile, voltage = np.array([5.0]), np.array([1.0])
        result, carried = self._call(
            profile,
            np.zeros(0, dtype=bool),
            np.array([1.0, 2.0, 3.0]),
            voltage,
            carried=0.5,
            carried_wake=4.0,
        )
        np.testing.assert_allclose(result, [1.0 + 1.7 * 2.0 * 5.0 + 2.0])
        self.assertAlmostEqual(carried, 1.7 * 5.0)
