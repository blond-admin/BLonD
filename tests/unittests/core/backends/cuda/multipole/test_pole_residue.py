import unittest

import numpy as np
import pytest

from blond.testing.backend_testing import (
    BLonDTestCase,
    skip_if_cuda_unusable,
)


@pytest.mark.cupy
class TestPoleChunkCount(BLonDTestCase):
    """How many time chunks the CUDA pole kernel splits the bins into.

    A block holds 16 poles and the GPU wants 16 blocks per multiprocessor;
    chunks keep at least four tiles of 64 bins, and stay fewer than the
    square root of the tiles: each folds the maps of all earlier chunks.
    """

    def setUp(self) -> None:
        skip_if_cuda_unusable(self)
        from blond.core.backends.cuda.multipole.pole_residue import (
            pole_chunk_count,
        )

        self.chunk_count = pole_chunk_count

    def test_no_chunks_when_poles_fill_the_gpu(self) -> None:
        self.assertEqual(self.chunk_count(10_000, 16 * 16 * 6, 6), 1)

    def test_few_poles_get_one_chunk_per_missing_block(self) -> None:
        self.assertEqual(self.chunk_count(1_000_000, 1, 6), 96)
        self.assertEqual(self.chunk_count(1_000_000, 64, 6), 24)

    def test_chunks_keep_four_tiles(self) -> None:
        self.assertEqual(self.chunk_count(600, 1, 6), 2)

    def test_chunks_stay_fewer_than_the_root_of_the_tiles(self) -> None:
        self.assertEqual(self.chunk_count(100_000, 1, 132), 39)

    def test_short_profile_is_not_chunked(self) -> None:
        self.assertEqual(self.chunk_count(10, 1, 6), 1)
        self.assertEqual(self.chunk_count(1, 1, 6), 1)


@pytest.mark.cupy
class TestChunkedPoleKernel(BLonDTestCase):
    """The chunked launch reproduces the unchunked one."""

    def setUp(self) -> None:
        skip_if_cuda_unusable(self)
        import cupy as cp

        from blond.core.backends.cuda.multipole import pole_residue

        self.cp = cp
        self.pole_residue = pole_residue

    @staticmethod
    def _grid(name: str) -> np.ndarray:
        """Bin positions in whole bins, from the first bin."""
        index = np.arange(1000)
        if name == "bunches":  # runs of 40 bins, gaps of 5 bins
            return index + 5 * (index // 40)
        if name == "dense":  # one run, tiles of 64 and a partial last one
            return index
        # single-bin runs and gaps of one empty bin, between short runs
        steps = np.where(index[1:] % 7 == 0, 2, 1)
        steps[::11] = 1
        return np.concatenate(([0], np.cumsum(steps)))

    def _run(
        self, n_chunks: int, states_in, poles, residues, signs, grid="bunches"
    ):
        cp = self.cp
        bin_dt = 1e-9
        index = np.arange(1000)
        profile_time = 1e-6 + bin_dt * self._grid(grid)
        profile = 1.0 + np.cos(index / 23.0) ** 2
        states = cp.asarray(states_in)
        voltage = cp.zeros(len(index))
        bin_offsets = np.rint((profile_time - profile_time[0]) / bin_dt)
        self.pole_residue.wake_from_pole_residue_in_chunks(
            bin_offsets=cp.asarray(bin_offsets, dtype=np.int64),
            profile=cp.asarray(profile),
            carried_charge=cp.asarray([0.8]),
            carried_is_counterrotating=False,
            state_lag_dt=0.4 * bin_dt,
            carried_lag_dt=1.4 * bin_dt,
            poles=cp.asarray(poles),
            residues=cp.asarray(residues),
            is_counterrotating_beam=True,
            counterrotating_pole_signs=cp.asarray(signs),
            factor=1.3,
            bin_dt=bin_dt,
            states=states,
            voltage=voltage,
            n_chunks=n_chunks,
        )
        return cp.asnumpy(voltage), cp.asnumpy(states)

    def test_chunks_match_single_pass(self) -> None:
        rng = np.random.default_rng(3)
        n_poles = 37
        is_real = np.arange(n_poles) % 4 == 1
        poles = -rng.uniform(5e6, 5e8, n_poles) + 1j * np.where(
            is_real, 0.0, rng.uniform(1e8, 3e9, n_poles)
        )
        residues = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
        residues[is_real] = residues[is_real].real
        states_in = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
        states_in[is_real] = states_in[is_real].real
        signs = np.where(np.arange(n_poles) % 3 == 0, -1.0, 1.0)
        for grid in ("bunches", "dense", "single bins"):
            expected_voltage, expected_states = self._run(
                1, states_in, poles, residues, signs, grid
            )
            for n_chunks in (2, 7, 15):
                voltage, states = self._run(
                    n_chunks, states_in, poles, residues, signs, grid
                )
                np.testing.assert_allclose(
                    voltage,
                    expected_voltage,
                    rtol=1e-12,
                    atol=1e-12 * float(np.max(np.abs(expected_voltage))),
                    err_msg=f"{grid=} {n_chunks=}",
                )
                np.testing.assert_allclose(
                    states,
                    expected_states,
                    rtol=1e-12,
                    atol=1e-12 * float(np.max(np.abs(expected_states))),
                    err_msg=f"states {grid=} {n_chunks=}",
                )

    def test_single_pass_matches_python_reference(self) -> None:
        from blond.core.backends.python.callables import PythonSpecials

        rng = np.random.default_rng(4)
        n_poles = 21
        poles = -rng.uniform(5e6, 2e9, n_poles) + 1j * rng.uniform(
            1e8, 3e9, n_poles
        )
        poles[::5] = poles[::5].real
        residues = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
        residues[::5] = residues[::5].real
        states_in = rng.normal(size=n_poles) + 1j * rng.normal(size=n_poles)
        states_in[::5] = states_in[::5].real
        signs = np.where(np.arange(n_poles) % 3 == 0, -1.0, 1.0)
        bin_dt = 1e-9
        index = np.arange(1000)
        profile = 1.0 + np.cos(index / 23.0) ** 2
        for grid in ("bunches", "dense", "single bins"):
            voltage, states = self._run(
                1, states_in, poles, residues, signs, grid
            )
            expected_states = states_in.copy()
            expected_voltage = np.zeros(len(index))
            PythonSpecials.wake_from_pole_residue(
                profile_time=1e-6 + bin_dt * self._grid(grid),
                profile=profile,
                carried_charge=np.array([0.8]),
                carried_is_counterrotating=False,
                state_lag_dt=0.4 * bin_dt,
                carried_lag_dt=1.4 * bin_dt,
                poles=poles,
                residues=residues,
                is_counterrotating_beam=True,
                counterrotating_pole_signs=signs,
                factor=1.3,
                bin_dt=bin_dt,
                states=expected_states,
                voltage=expected_voltage,
                voltage_threaded=np.zeros((1, len(index))),
            )
            np.testing.assert_allclose(
                voltage,
                expected_voltage,
                rtol=1e-10,
                atol=1e-10 * float(np.max(np.abs(expected_voltage))),
                err_msg=f"{grid=}",
            )
            np.testing.assert_allclose(
                states,
                expected_states,
                rtol=1e-10,
                atol=1e-10 * float(np.max(np.abs(expected_states))),
                err_msg=f"states {grid=}",
            )


if __name__ == "__main__":
    unittest.main()
