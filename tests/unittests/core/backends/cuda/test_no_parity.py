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
