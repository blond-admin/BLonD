import unittest

import numpy as np
import pytest

from blond.testing.backend_testing import BLonDTestCase


@pytest.mark.cupy
class TestPoleChunkCount(BLonDTestCase):
    """How many time chunks the CUDA pole kernel splits the bins into."""

    def setUp(self) -> None:
        try:
            from blond.core.backends.cuda.callables import _pole_chunk_count
        except ImportError as exc:
            if "cupy" in str(exc):
                self.skipTest("Cupy not available")
            raise
        self.chunk_count = _pole_chunk_count

    def test_no_chunks_when_poles_fill_the_gpu(self) -> None:
        self.assertEqual(self.chunk_count(10_000, 32 * 8 * 6, 6), 1)

    def test_few_poles_get_one_chunk_per_missing_warp(self) -> None:
        # 6 multiprocessors want 48 warps; one pole is one warp
        self.assertEqual(self.chunk_count(10_000, 1, 6), 48)
        self.assertEqual(self.chunk_count(10_000, 64, 6), 24)

    def test_chunks_keep_a_minimum_length(self) -> None:
        self.assertEqual(self.chunk_count(200, 1, 6), 3)

    def test_chunks_stay_longer_than_their_number(self) -> None:
        # folding the earlier chunks costs one step per chunk, so their
        # number is kept below the length of a chunk
        self.assertEqual(self.chunk_count(10_000, 1, 132), 100)

    def test_short_profile_is_not_chunked(self) -> None:
        self.assertEqual(self.chunk_count(10, 1, 6), 1)
        self.assertEqual(self.chunk_count(1, 1, 6), 1)


@pytest.mark.cupy
class TestChunkedPoleKernel(BLonDTestCase):
    """The chunked launch reproduces the unchunked one."""

    def setUp(self) -> None:
        try:
            import cupy as cp

            from blond.core.backends.cuda import callables
        except ImportError as exc:
            if "cupy" in str(exc):
                self.skipTest("Cupy not available")
            raise
        self.cp = cp
        self.callables = callables

    def _run(self, n_chunks: int, states_in, poles, residues, signs):
        cp = self.cp
        bin_dt = 1e-9
        index = np.arange(1000)
        profile_time = 1e-6 + bin_dt * (index + 5 * (index // 40))
        profile = 1.0 + np.cos(index / 23.0) ** 2
        states = cp.asarray(states_in)
        voltage = cp.zeros(len(index))
        bin_offsets = np.rint((profile_time - profile_time[0]) / bin_dt)
        self.callables._wake_from_pole_residue_in_chunks(
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
        expected_voltage, expected_states = self._run(
            1, states_in, poles, residues, signs
        )
        for n_chunks in (2, 7, 15):
            voltage, states = self._run(
                n_chunks, states_in, poles, residues, signs
            )
            np.testing.assert_allclose(
                voltage,
                expected_voltage,
                rtol=1e-12,
                atol=1e-12 * float(np.max(np.abs(expected_voltage))),
                err_msg=f"{n_chunks=}",
            )
            np.testing.assert_allclose(
                states,
                expected_states,
                rtol=1e-12,
                atol=1e-12 * float(np.max(np.abs(expected_states))),
                err_msg=f"states {n_chunks=}",
            )


if __name__ == "__main__":
    unittest.main()
