# coding: utf8
"""
Tests for the multi-pass resonator induced voltage on sparse profiles
(blond.impedances.sparse_impedance).

The reference behaviours are:
- `Resonators.get_decay_time` returns the time where the normalised
  wake envelope crosses the requested threshold;
- the induced voltage of `InducedVoltageSparseMultiPass` equals the
  direct double sum ``V[m] = factor * sum_k q[k] W(t_m - t_k)`` over all
  sparse window bins (with the wake's built-in half weight at t = 0);
- with an RFStation, a second call adds the previous turn's windows
  shifted by one revolution period, matching the direct sum over both
  turns;
- a pass older than the source's decay time is dropped: with a wake
  that decays within one turn, the second call reproduces the first.

Author:
Lina Valle
"""

import unittest

import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.sparse_profiles import SparseBatch
from blond.impedances.impedance_sources import Resonators
from blond.impedances.sparse_impedance import InducedVoltageSparseMultiPass
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from scipy.constants import e as e_charge

RING_CIRCUMFERENCE = 26658.883
SYNCHRONOUS_MOMENTUM = 450e9
HARMONIC_NUMBER = 35640
RF_VOLTAGE = 4e6
GAMMA_TRANSITION = 53.8

N_MACROPARTICLES = int(1e4)
BUNCH_INTENSITY = 1e11
BUNCH_SIGMA_DT = 0.5e-9

number_of_batches = 3
batch_spacing = 8  # buckets between batch starts
N_SLICES_PER_BUCKET = 32


def build_ring_and_rf(n_turns=2):
    ring = Ring(
        RING_CIRCUMFERENCE,
        1 / GAMMA_TRANSITION**2,
        SYNCHRONOUS_MOMENTUM,
        particle=Proton(),
        n_turns=n_turns,
    )
    rf_station = RFStation(ring, [HARMONIC_NUMBER], [RF_VOLTAGE], [0])
    return ring, rf_station


def build_beam(ring, rf_station, seed=1234):
    beam = Beam(
        ring,
        N_MACROPARTICLES * number_of_batches,
        BUNCH_INTENSITY * number_of_batches,
    )
    single = Beam(ring, N_MACROPARTICLES, BUNCH_INTENSITY)
    bigaussian(
        ring,
        rf_station,
        single,
        sigma_dt=BUNCH_SIGMA_DT,
        seed=seed,
        reinsertion=True,
    )
    for i in range(number_of_batches):
        sel = slice(i * N_MACROPARTICLES, (i + 1) * N_MACROPARTICLES)
        beam.dE[sel] = single.dE
        beam.dt[sel] = single.dt + i * batch_spacing * rf_station.t_rf[0, 0]
    return beam


def build_sparse_profile(beam, rf_station):
    batch_list = np.zeros(HARMONIC_NUMBER)
    for k in range(number_of_batches):
        batch_list[k * batch_spacing] = 1
    length_in_buckets = batch_spacing // 2
    sparse = SparseBatch(
        rf_station=rf_station,
        beam=beam,
        number_of_slices_per_profile=length_in_buckets * N_SLICES_PER_BUCKET,
        batch_list=batch_list,
        batch_length=length_in_buckets,
        tracker_mode="onebyone",
    )
    sparse.track()
    return sparse


def make_resonators():
    return Resonators(
        R_S=[5e6, 2e6], frequency_R=[200e6, 401e6], Q=[80.0, 35.0]
    )


def direct_sum_reference(resonators, beam, times, hists):
    """V[m] = factor * sum_k q[k] W(t_m - t_k) evaluated per target bin.

    `times` are the target bin centers; `hists` a list of
    (source_times, source_hist) pairs (e.g. windows of several turns).
    """
    factor = -beam.particle.charge * e_charge * beam.ratio
    voltage = np.zeros(len(times))
    for source_times, source_hist in hists:
        delta_t = times[:, np.newaxis] - source_times[np.newaxis, :]
        resonators.wake_calc(delta_t.flatten())
        wake = resonators.wake.reshape(delta_t.shape)
        voltage += factor * wake @ source_hist
    return voltage


class TestDecayTime(unittest.TestCase):
    def test_envelope_normalised_and_decaying(self):
        res = make_resonators()
        t, envelope = res.calculate_envelope()
        self.assertAlmostEqual(np.max(envelope), 1.0, places=12)
        self.assertTrue(np.all(np.diff(envelope) <= 0))

    def test_decay_time_crosses_threshold(self):
        res = make_resonators()
        threshold = 1e-3
        storage_time = res.get_decay_time(threshold)
        _, envelope = res.calculate_envelope(
            time_axis=np.array([0.0, storage_time])
        )
        self.assertAlmostEqual(envelope[-1], threshold, places=6)


class TestInducedVoltageSparseMultiPass(unittest.TestCase):
    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.sparse = build_sparse_profile(self.beam, self.rf)
        self.resonators = make_resonators()

    def _window_arrays(self):
        return [
            (
                self.sparse.profiles_list[p].bin_centers,
                self.sparse.profiles_list[p].n_macroparticles.astype(float),
            )
            for p in self.sparse.memory_time_order
        ]

    def test_single_turn_equals_direct_sum(self):
        iv = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, self.resonators
        )
        iv.induced_voltage_generation()

        windows = self._window_arrays()
        reference = np.concatenate(
            [
                direct_sum_reference(
                    self.resonators, self.beam, centers, windows
                )
                for centers, _ in windows
            ]
        )
        scale = np.max(np.abs(reference))
        np.testing.assert_allclose(
            iv.induced_voltage,
            reference,
            rtol=1e-6,
            atol=1e-9 * scale,
            err_msg="Single-turn sparse induced voltage differs from the "
            "direct double sum.",
        )

    def test_second_turn_equals_direct_sum_over_two_turns(self):
        # A wake that survives one turn: alpha * t_rev = 1
        t_rev = self.rf.t_rev[0]
        f_r = 200e6
        long_memory = Resonators(
            R_S=5e6, frequency_R=f_r, Q=np.pi * f_r * t_rev
        )
        iv = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, long_memory, rf_station=self.rf
        )
        iv.induced_voltage_generation()
        iv.induced_voltage_generation()

        windows = self._window_arrays()
        previous_turn = [(centers - t_rev, hist) for centers, hist in windows]
        reference = np.concatenate(
            [
                direct_sum_reference(
                    long_memory,
                    self.beam,
                    centers,
                    windows + previous_turn,
                )
                for centers, _ in windows
            ]
        )
        scale = np.max(np.abs(reference))
        np.testing.assert_allclose(
            iv.induced_voltage,
            reference,
            rtol=1e-6,
            atol=1e-9 * scale,
            err_msg="Second-turn sparse induced voltage differs from the "
            "direct double sum over both turns.",
        )

    def test_fully_decayed_pass_is_dropped(self):
        # alpha * t_rev ~ 700 with these resonators: the wake of turn 1
        # is fully decayed on turn 2, and the stored pass must be dropped
        iv = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, self.resonators, rf_station=self.rf
        )
        iv.induced_voltage_generation()
        first = iv.induced_voltage.copy()
        n_stored_after_first = len(iv._past_windows)
        iv.induced_voltage_generation()

        self.assertEqual(n_stored_after_first, len(self.sparse.profiles_list))
        self.assertEqual(len(iv._past_windows), len(self.sparse.profiles_list))
        np.testing.assert_allclose(
            iv.induced_voltage,
            first,
            rtol=1e-12,
            err_msg="With a fully decayed wake the second turn must "
            "reproduce the first.",
        )

    def test_process_resets_memory(self):
        iv = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, self.resonators, rf_station=self.rf
        )
        iv.induced_voltage_generation()
        first = iv.induced_voltage.copy()
        iv.process()
        iv.induced_voltage_generation()
        np.testing.assert_allclose(
            iv.induced_voltage,
            first,
            rtol=1e-12,
            err_msg="process() did not reset the pass memory.",
        )

    def test_kernel_matches_python_path(self):
        """The numba kernel and the numpy convolve path must agree, on
        the first turn and on a second turn with a surviving wake."""
        t_rev = self.rf.t_rev[0]
        f_r = 200e6
        long_memory = Resonators(
            R_S=5e6, frequency_R=f_r, Q=np.pi * f_r * t_rev
        )
        iv_kernel = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, long_memory, rf_station=self.rf
        )
        if not iv_kernel.use_numba_kernels:
            self.skipTest("numba kernels not available")
        iv_python = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, long_memory, rf_station=self.rf
        )
        iv_python.use_numba_kernels = False

        for turn in range(2):
            iv_kernel.induced_voltage_generation()
            iv_python.induced_voltage_generation()
            scale = np.max(np.abs(iv_python.induced_voltage))
            np.testing.assert_allclose(
                iv_kernel.induced_voltage,
                iv_python.induced_voltage,
                rtol=1e-7,
                atol=1e-9 * scale,
                err_msg="Numba kernel differs from the numpy path on "
                f"turn {turn}.",
            )

    def test_track_kicks_beam(self):
        iv = InducedVoltageSparseMultiPass(
            self.beam, self.sparse, self.resonators
        )
        dE_before = self.beam.dE.copy()
        iv.track()
        self.assertTrue(np.any(self.beam.dE != dE_before))


if __name__ == "__main__":
    unittest.main()
