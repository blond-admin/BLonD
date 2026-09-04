# coding: utf8
"""
Tests for the pole-residue induced voltage on sparse profiles with
multi-turn wake memory (blond.impedances.sparse_impedance).

The reference behaviours are:
- the pole/residue pairs of `Resonators.get_vectorfit` reproduce the
  analytic resonator wake of `Resonators.wake_calc`;
- the `wake_from_pole_residue` recursion on a contiguous grid equals the
  direct discrete convolution with the wake function (with the self-bin
  at half weight, following the beam-loading theorem);
- on a SparseBatch the recursion equals the contiguous solution evaluated
  at the window bins (the analytic decay across empty buckets replaces
  the zero-charge bins of the contiguous grid);
- calling the solver again one revolution period later equals a single
  contiguous solve over both turns.

Author:
Lina Valle
"""

import unittest

import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, Profile
from blond.beam.sparse_profiles import SparseBatch
from blond.impedances.impedance_sources import Resonators
from blond.impedances.sparse_impedance import (
    InducedVoltageSparseMTW,
    wake_from_pole_residue,
)
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring

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
N_SLICES_PER_BUCKET = 64


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


def build_standard_profile(beam, rf_station):
    n_buckets = batch_spacing * number_of_batches
    profile = Profile(
        beam,
        CutOptions(
            cut_left=0.0,
            cut_right=n_buckets * rf_station.t_rf[0, 0],
            n_slices=n_buckets * N_SLICES_PER_BUCKET,
        ),
    )
    profile.track()
    return profile


def reference_wake(resonators, time_array):
    """2 Re(sum A_k exp(s_k t)) for t >= 0."""
    poles, residues = resonators.get_vectorfit()
    wake = np.zeros(len(time_array))
    for p, r in zip(poles, residues):
        wake += 2 * np.real(r * np.exp(p * time_array))
    return wake


def make_resonators():
    return Resonators(
        R_S=[5e6, 2e6], frequency_R=[200e6, 401e6], Q=[80.0, 35.0]
    )


class TestGetVectorfit(unittest.TestCase):
    def test_reproduces_analytic_wake(self):
        res = make_resonators()
        t = np.linspace(1e-12, 100e-9, 2000)
        res.wake_calc(t)
        np.testing.assert_allclose(
            reference_wake(res, t),
            res.wake,
            rtol=1e-9,
            atol=1e-9 * np.max(np.abs(res.wake)),
            err_msg="Pole-residue wake differs from the analytic "
            "resonator wake.",
        )

    def test_pole_count(self):
        res = make_resonators()
        poles, residues = res.get_vectorfit()
        self.assertEqual(len(poles), res.n_resonators)
        self.assertEqual(len(residues), res.n_resonators)
        # Stable poles with positive oscillation frequency
        self.assertTrue(np.all(np.real(poles) < 0))
        self.assertTrue(np.all(np.imag(poles) > 0))


class TestWakeFromPoleResidue(unittest.TestCase):
    def _run_kernel(
        self,
        hist,
        centers,
        update_on_bin,
        poles,
        residues,
        factor=1.0,
        states=None,
    ):
        if states is None:
            states = np.zeros(len(poles) + 1, dtype=complex)
            states[-1] = centers[0] - (centers[1] - centers[0]) / 2
        voltage = np.zeros(len(hist))
        wake_from_pole_residue(
            profile=hist,
            profile_dts=centers,
            poles=poles,
            residues=residues,
            update_on_bin=update_on_bin,
            factor=factor,
            states=states,
            voltage=voltage,
        )
        return voltage, states

    def test_matches_direct_convolution(self):
        res = make_resonators()
        poles, residues = res.get_vectorfit()

        rng = np.random.default_rng(42)
        n_bins = 512
        dt = 0.5e-9
        centers = dt / 2 + dt * np.arange(n_bins)
        hist = rng.random(n_bins)

        voltage, _ = self._run_kernel(
            hist, centers, np.zeros(1, dtype=np.int64), poles, residues
        )

        wake = reference_wake(res, dt * np.arange(n_bins))
        wake[0] *= 0.5  # self-bin at half weight (beam-loading theorem)
        reference = np.convolve(hist, wake)[:n_bins]

        np.testing.assert_allclose(
            voltage,
            reference,
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(reference)),
            err_msg="Pole recursion differs from direct convolution on a "
            "contiguous grid.",
        )

    def test_gap_jump_equals_contiguous_zero_bins(self):
        res = make_resonators()
        poles, residues = res.get_vectorfit()

        rng = np.random.default_rng(7)
        n_window = 128
        n_gap = 300
        dt = 0.5e-9
        centers_full = dt / 2 + dt * np.arange(2 * n_window + n_gap)
        hist_full = np.zeros(len(centers_full))
        hist_full[:n_window] = rng.random(n_window)
        hist_full[n_window + n_gap :] = rng.random(n_window)

        v_full, _ = self._run_kernel(
            hist_full,
            centers_full,
            np.zeros(1, dtype=np.int64),
            poles,
            residues,
        )

        keep = np.concatenate(
            (
                np.arange(n_window),
                np.arange(n_window + n_gap, 2 * n_window + n_gap),
            )
        )
        v_sparse, _ = self._run_kernel(
            hist_full[keep],
            centers_full[keep],
            np.array([0, n_window], dtype=np.int64),
            poles,
            residues,
        )

        np.testing.assert_allclose(
            v_sparse,
            v_full[keep],
            rtol=1e-7,
            atol=1e-9 * np.max(np.abs(v_full)),
            err_msg="Analytic decay across the gap differs from stepping "
            "through zero-charge bins.",
        )

    def test_multi_turn_states_equal_contiguous_two_turns(self):
        res = make_resonators()
        poles, residues = res.get_vectorfit()

        rng = np.random.default_rng(3)
        n_bins = 256
        dt = 0.5e-9
        t_rev = 5e-6
        centers = dt / 2 + dt * np.arange(n_bins)
        hist = rng.random(n_bins)

        # Two sequential calls with persistent states, shifted by t_rev
        states = np.zeros(len(poles) + 1, dtype=complex)
        states[-1] = centers[0] - dt / 2
        v1, states = self._run_kernel(
            hist,
            centers,
            np.zeros(1, dtype=np.int64),
            poles,
            residues,
            states=states,
        )
        states[-1] -= t_rev
        v2, _ = self._run_kernel(
            hist,
            centers,
            np.zeros(1, dtype=np.int64),
            poles,
            residues,
            states=states,
        )

        # One contiguous solve over both turns
        centers_2t = np.concatenate((centers, centers + t_rev))
        hist_2t = np.concatenate((hist, hist))
        v_2t, _ = self._run_kernel(
            hist_2t,
            centers_2t,
            np.array([0, n_bins], dtype=np.int64),
            poles,
            residues,
        )

        np.testing.assert_allclose(
            v1, v_2t[:n_bins], rtol=1e-9, atol=1e-9 * np.max(np.abs(v_2t))
        )
        np.testing.assert_allclose(
            v2,
            v_2t[n_bins:],
            rtol=1e-7,
            atol=1e-9 * np.max(np.abs(v_2t)),
            err_msg="Second-turn voltage with persistent pole states "
            "differs from a contiguous two-turn solve.",
        )


class TestInducedVoltageSparseMTW(unittest.TestCase):
    def setUp(self):
        self.ring, self.rf = build_ring_and_rf()
        self.beam = build_beam(self.ring, self.rf)
        self.sparse = build_sparse_profile(self.beam, self.rf)
        self.standard = build_standard_profile(self.beam, self.rf)
        self.resonators = make_resonators()

    def test_sparse_equals_standard_profile(self):
        iv_sparse = InducedVoltageSparseMTW(
            self.beam, self.sparse, self.resonators
        )
        iv_sparse.induced_voltage_generation()

        # Reference: the same solve on the contiguous standard grid, with
        # the histogram restricted to the sparse windows (a few particle
        # tails land in the gap buckets; a sparse profile cannot see that
        # charge, so it must be excluded from the reference too)
        n_p = self.sparse.number_of_slices_per_profile
        window_start = {}
        hist_masked = np.zeros(len(self.standard.bin_centers))
        for k, p in enumerate(self.sparse.memory_time_order):
            profile = self.sparse.profiles_list[p]
            index = np.argmin(
                np.abs(self.standard.bin_centers - profile.bin_centers[0])
            )
            np.testing.assert_allclose(
                self.standard.bin_centers[index : index + n_p],
                profile.bin_centers,
                rtol=1e-10,
                atol=1e-15,
            )
            np.testing.assert_equal(
                self.standard.n_macroparticles[index : index + n_p],
                profile.n_macroparticles,
            )
            window_start[k] = index
            hist_masked[index : index + n_p] = profile.n_macroparticles

        from scipy.constants import e as e_charge

        factor = -self.beam.particle.charge * e_charge * self.beam.ratio
        states = np.zeros(len(iv_sparse.poles) + 1, dtype=complex)
        states[-1] = self.standard.bin_centers[0] - self.standard.bin_size / 2
        reference = np.zeros(len(hist_masked))
        wake_from_pole_residue(
            profile=hist_masked,
            profile_dts=self.standard.bin_centers,
            poles=iv_sparse.poles,
            residues=iv_sparse.residues,
            update_on_bin=np.zeros(1, dtype=np.int64),
            factor=factor,
            states=states,
            voltage=reference,
        )

        scale = np.max(np.abs(reference))
        for k in window_start:
            index = window_start[k]
            np.testing.assert_allclose(
                iv_sparse.induced_voltage[k * n_p : (k + 1) * n_p],
                reference[index : index + n_p],
                rtol=1e-6,
                atol=1e-9 * scale,
                err_msg="Sparse induced voltage differs from the standard "
                f"profile solution, window {k}",
            )

    def test_multi_turn_memory_adds_previous_turn(self):
        # A high-Q resonator whose wake survives one revolution period:
        # alpha * t_rev = omega_R / (2 Q) * t_rev ~ 1
        t_rev = self.rf.t_rev[0]
        f_r = 200e6
        Q = np.pi * f_r * t_rev  # alpha * t_rev = 1
        long_memory_resonator = Resonators(R_S=5e6, frequency_R=f_r, Q=Q)
        iv = InducedVoltageSparseMTW(
            self.beam,
            self.sparse,
            long_memory_resonator,
            rf_station=self.rf,
        )
        iv.induced_voltage_generation()
        first_turn = iv.induced_voltage.copy()
        iv.induced_voltage_generation()
        second_turn = iv.induced_voltage.copy()

        # The wake left over from the first turn must make a difference,
        # and the difference must be bounded by the decayed wake amplitude
        difference = second_turn - first_turn
        self.assertGreater(np.max(np.abs(difference)), 0.0)
        self.assertLess(np.max(np.abs(difference)), np.max(np.abs(first_turn)))

    def test_process_resets_memory(self):
        iv = InducedVoltageSparseMTW(
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
            err_msg="process() did not reset the multi-turn wake memory.",
        )

    def test_track_kicks_beam(self):
        iv = InducedVoltageSparseMTW(self.beam, self.sparse, self.resonators)
        dE_before = self.beam.dE.copy()
        iv.track()
        self.assertTrue(np.any(self.beam.dE != dE_before))


if __name__ == "__main__":
    unittest.main()
