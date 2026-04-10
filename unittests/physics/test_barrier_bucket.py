# General imports
import unittest

import numpy as np
import numpy.testing as nptest

# BLonD imports
import blond.physics.barrier_bucket as bbuck
from blond.core.backends.backend import CupyBackend, backend
from blond.testing.backend_testing import ArrayLikeScan, multi_backend_testcase


class TestBarrierBucketFunctions(unittest.TestCase):
    @multi_backend_testcase
    def test_simple_barrier(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        centers = np.linspace(0, 1000e-9, 5000)

        for inp_cast in ArrayLikeScan():
            barrier = bbuck.compute_sin_barrier(
                cent, width, ampl, inp_cast(centers), periodic=False
            )

            self.assertAlmostEqual(float(backend.max(barrier)), 1e3, places=1)

            left_pts = np.where(centers < cent - width / 2)[0]
            right_pts = np.where(centers > cent + width / 2)[0]

            self.assertListEqual(
                list(barrier[left_pts]), list(np.zeros_like(left_pts))
            )
            self.assertListEqual(
                list(barrier[right_pts]), list(np.zeros_like(right_pts))
            )

            self.assertAlmostEqual(
                float(backend.max(barrier[left_pts[-1] : right_pts[0]])),
                1e3,
                places=1,
            )
            self.assertAlmostEqual(
                float(backend.min(barrier[left_pts[-1] : right_pts[0]])),
                -1e3,
                places=1,
            )

    @multi_backend_testcase
    def test_periodic_barrier_right(self):
        cent = 1000e-9
        width = 100e-9
        ampl = 1e3

        centers = backend.linspace(0, 1000e-9, 20000)

        barrier = bbuck.compute_sin_barrier(
            cent, width, ampl, centers, periodic=True
        )

        self.assertAlmostEqual(float(backend.max(barrier)), 1e3, places=1)

        left_pts = backend.where(centers <= centers[0] + width / 2)[0]
        right_pts = backend.where(centers >= centers[-1] - width / 2)[0]

        if isinstance(backend, CupyBackend):
            barrier = barrier.get()
            left_pts = left_pts.get()
            right_pts = right_pts.get()

        nptest.assert_array_almost_equal(
            barrier[: int(left_pts[-1])] / ampl,
            -barrier[int(right_pts[0]) : -1] / ampl,
            decimal=2,
        )

        self.assertAlmostEqual(
            float(backend.max(barrier[left_pts])), 1e3, places=1
        )
        self.assertAlmostEqual(
            float(backend.min(barrier[right_pts])), -1e3, places=1
        )

    @multi_backend_testcase
    def test_periodic_barrier_left(self):
        cent = 0
        width = 100e-9
        ampl = 1e3

        centers = backend.linspace(0, 1000e-9, 20000)

        barrier = bbuck.compute_sin_barrier(
            cent, width, ampl, centers, periodic=True
        )

        self.assertAlmostEqual(float(backend.max(barrier)), 1e3, places=1)

        left_pts = backend.where(centers <= centers[0] + width / 2)[0]
        right_pts = backend.where(centers >= centers[-1] - width / 2)[0]

        if isinstance(backend, CupyBackend):
            barrier = barrier.get()
            left_pts = left_pts.get()
            right_pts = right_pts.get()

        nptest.assert_array_almost_equal(
            barrier[: int(left_pts[-1])] / ampl,
            -barrier[int(right_pts[0]) : -1] / ampl,
            decimal=2,
        )

        self.assertAlmostEqual(
            float(backend.max(barrier[left_pts])), 1e3, places=1
        )
        self.assertAlmostEqual(
            float(backend.min(barrier[right_pts])), -1e3, places=1
        )

    @multi_backend_testcase
    def test_wide_barrier(self):
        cent = 1000e-9
        width = 1100e-9
        ampl = 1e3

        centers = backend.linspace(0, 1000e-9, 5000)

        with self.assertRaises(ValueError):
            bbuck.compute_sin_barrier(cent, width, ampl, centers)

    @multi_backend_testcase
    def test_fourier_series(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        centers = backend.linspace(0, 1000e-9, 5000)

        barrier = bbuck.compute_sin_barrier(
            cent, width, ampl, centers, periodic=False
        )

        amps_exp = [
            19.9,
            39.0,
            56.6,
            72.1,
            84.9,
            94.6,
            101.0,
            103.9,
            103.5,
            100,
            93.7,
            85.0,
        ]
        phases_exp = [
            3.14,
            6.28,
            3.14,
            6.28,
            3.14,
            6.28,
            3.14,
            6.28,
            3.14,
            6.28,
            3.13,
            6.28,
        ]

        for inp_cast in ArrayLikeScan():
            amps, phases = bbuck.waveform_to_harmonics(
                inp_cast(barrier), inp_cast(list(range(1, 13)))
            )

            for a, a_exp, p, p_exp in zip(amps, amps_exp, phases, phases_exp):
                self.assertAlmostEqual(float(a), a_exp, places=1)
                self.assertAlmostEqual(float(p), p_exp, places=2)

    @multi_backend_testcase
    def test_sinc_filter(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        centers = backend.linspace(0, 1000e-9, 5000)

        barrier = bbuck.compute_sin_barrier(
            cent, width, ampl, centers, periodic=False
        )

        amps, _ = bbuck.waveform_to_harmonics(barrier, backend.arange(1, 13))
        amps = bbuck.sinc_filtering(amps, m=1)

        amps_exp = [
            19.4,
            35.3,
            45.1,
            47.4,
            42.3,
            31.6,
            17.7,
            3.6,
            -8.2,
            -16.0,
            -19.3,
            -18.4,
        ]

        for inp_cast in ArrayLikeScan():
            for a, a_exp in zip(inp_cast(amps), amps_exp):
                self.assertAlmostEqual(float(a), a_exp, places=1)

    @multi_backend_testcase
    def test_waveform_harmonics(self):
        harms = [1, 2, 3, 4, 5, 6, 7]
        set_amps = [4e3, 0, 3e3, 0, 2e3, 0, 1e3]
        set_phases = [0, 0, np.pi, 0, 0, 0, np.pi]

        t_rev = 1e-6
        centers = backend.linspace(0, t_rev, 5000)
        waveform = backend.zeros_like(centers)

        for h, a, p in zip(harms, set_amps, set_phases):
            waveform += a * np.sin(2 * np.pi * h * centers / t_rev + p)

        comp_amps, comp_phases = bbuck.waveform_to_harmonics(
            waveform, range(1, 8)
        )

        for i in range(len(harms)):
            self.assertAlmostEqual(comp_amps[i], set_amps[i], delta=1e1)

            if set_amps[i] > 0:
                # Use sin/cos comparison to avoid issues with 0 != 2pi
                self.assertAlmostEqual(
                    float(backend.cos(comp_phases[i])),
                    np.cos(set_phases[i]),
                    places=1,
                )
                self.assertAlmostEqual(
                    float(backend.sin(comp_phases[i])),
                    np.sin(set_phases[i]),
                    places=1,
                )

        if isinstance(backend, CupyBackend):
            waveform = waveform.get()

        for inp_cast in ArrayLikeScan():
            comp_wave = bbuck.harmonics_to_waveform(
                inp_cast(centers), harms, set_amps, set_phases, t_rev
            )

            if isinstance(backend, CupyBackend):
                comp_wave = comp_wave.get()

            nptest.assert_array_almost_equal(waveform, comp_wave, decimal=2)

    @multi_backend_testcase
    def test_negative_barrier(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        centers = backend.linspace(0, 1000e-9, 5000)

        pbarrier = bbuck.compute_sin_barrier(cent, width, ampl, centers)
        nbarrier = bbuck.compute_sin_barrier(cent, width, -ampl, centers)

        if isinstance(backend, CupyBackend):
            pbarrier = pbarrier.get()
            nbarrier = nbarrier.get()

        nptest.assert_array_equal(pbarrier, -nbarrier)

        pamps, pphases = bbuck.waveform_to_harmonics(pbarrier, range(1, 26))
        namps, nphases = bbuck.waveform_to_harmonics(nbarrier, range(1, 26))

        if isinstance(backend, CupyBackend):
            pamps = pamps.get()
            namps = namps.get()

        nptest.assert_array_equal(pamps, namps)

        precreated = bbuck.harmonics_to_waveform(
            centers, range(1, 26), pamps, pphases
        )
        nrecreated = bbuck.harmonics_to_waveform(
            centers, range(1, 26), namps, nphases
        )

        if isinstance(backend, CupyBackend):
            precreated = precreated.get()
            nrecreated = nrecreated.get()

        nptest.assert_array_almost_equal(precreated, -nrecreated)


class TestBarrierBucketGenerator(unittest.TestCase):
    def test_fixed_barrier(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        generator = bbuck.BarrierGenerator(cent, width, ampl)
        bin_cents = np.linspace(0, 1000e-9, 1000)
        wave = generator.waveform_at_turn_or_time(
            turn_i=0, reference_time=1, bin_centers=bin_cents
        )

        wave_pts = np.where(wave != 0)[0]
        self.assertAlmostEqual(
            bin_cents[wave_pts[-1]] - bin_cents[wave_pts[0]], width, places=2
        )
        self.assertAlmostEqual(np.mean(bin_cents[wave_pts]), cent, places=2)

    def test_variable_barrier(self):
        bin_cents = np.linspace(0, 1000e-9, 10000)

        peak = (np.array([0, 1]), np.array([1e3, 4e3]))
        t_cent = (np.array([0, 1]), np.array([200e-9, 800e-9]))
        t_width = (np.array([0, 1]), np.array([100e-9, 150e-9]))

        generator = bbuck.BarrierGenerator()
        generator.schedule("t_center", t_cent)
        generator.schedule("t_width", t_width)
        generator.schedule("peak", peak)

        for t in np.linspace(0, 1, 10):
            peak_exp = np.interp(t, peak[0], peak[1])
            cent_exp = np.interp(t, t_cent[0], t_cent[1])
            width_exp = np.interp(t, t_width[0], t_width[1])
            wave = generator.waveform_at_turn_or_time(0, t, bin_cents)

            self.assertAlmostEqual(np.max(wave), peak_exp, places=1)
            self.assertAlmostEqual(np.min(wave), -peak_exp, places=1)

            wave_pts = np.where(wave != 0)[0]
            self.assertAlmostEqual(
                bin_cents[wave_pts[-1]] - bin_cents[wave_pts[0]],
                width_exp,
                places=1,
            )
            self.assertAlmostEqual(
                np.mean(bin_cents[wave_pts]), cent_exp, places=1
            )

    def test_to_fourier_series_simple_times_only(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        generator = bbuck.BarrierGenerator(cent, width, ampl)

        times = np.linspace(0, 1, 10)
        t_rev = np.zeros_like(times) + 1000e-9
        harmonics = np.arange(1, 11)

        # Unfiltered
        harms, amps, phases = generator.to_fourier_series(
            t_rev, harmonics, None, times, m=0
        )

        self.assertEqual(len(harms), len(amps))
        self.assertEqual(len(harms), len(phases))

        for a, p in zip(amps, phases):
            self.assertEqual(len(a), len(times))
            self.assertEqual(len(p), len(times))
        self.assertListEqual(list(harms), list(harmonics))

        bin_width = t_rev[0] / (10 * harmonics[-1])
        n_bins = int(t_rev[0] / bin_width)
        bin_cents = np.linspace(0, t_rev[0], n_bins)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, bin_cents)
        amps_exp, phases_exp = bbuck.waveform_to_harmonics(barrier, harmonics)

        g_comp = bbuck._gain_compensation(
            bin_cents, barrier, harms, amps_exp, phases_exp, t_rev[0]
        )

        amps_exp /= g_comp

        for i, (a, p) in enumerate(zip(amps, phases)):
            self.assertEqual(a[0], amps_exp[i])
            self.assertEqual(p[0], phases_exp[i])

    def test_to_fourier_series_simple_turns_only(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        generator = bbuck.BarrierGenerator(cent, width, ampl)

        turns = np.arange(10)
        t_rev = np.zeros_like(turns) + 1000e-9
        harmonics = np.arange(1, 11)

        # Unfiltered
        harms, amps, phases = generator.to_fourier_series(
            t_rev, harmonics, turns, None, m=0
        )

        self.assertEqual(len(harms), len(amps))
        self.assertEqual(len(harms), len(phases))

        for a, p in zip(amps, phases):
            self.assertEqual(len(a), len(turns))
            self.assertEqual(len(p), len(turns))
        self.assertListEqual(list(harms), list(harmonics))

        bin_width = t_rev[0] / (10 * harmonics[-1])
        n_bins = int(t_rev[0] / bin_width)
        bin_cents = np.linspace(0, t_rev[0], n_bins)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, bin_cents)
        amps_exp, phases_exp = bbuck.waveform_to_harmonics(barrier, harmonics)

        g_comp = bbuck._gain_compensation(
            bin_cents, barrier, harms, amps_exp, phases_exp, t_rev[0]
        )

        amps_exp /= g_comp

        for i, (a, p) in enumerate(zip(amps, phases)):
            self.assertEqual(a[0], amps_exp[i])
            self.assertEqual(p[0], phases_exp[i])

    def test_to_fourier_series_simple_turns_times(self):
        cent = 500e-9
        width = 100e-9
        ampl = 1e3

        generator = bbuck.BarrierGenerator(cent, width, ampl)

        turns = np.arange(10)
        times = np.linspace(0, 1, 10)
        t_rev = np.zeros_like(turns) + 1000e-9
        harmonics = np.arange(1, 11)

        # Test exception
        with self.assertRaises(ValueError):
            harms, amps, phases = generator.to_fourier_series(
                t_rev, harmonics, times[:-1], turns, m=0
            )

        with self.assertRaises(ValueError):
            harms, amps, phases = generator.to_fourier_series(
                t_rev, harmonics, None, None, m=0
            )

        # Unfiltered
        harms, amps, phases = generator.to_fourier_series(
            t_rev, harmonics, turns, times, m=0
        )

        self.assertEqual(len(harms), len(amps))
        self.assertEqual(len(harms), len(phases))

        for a, p in zip(amps, phases):
            self.assertEqual(len(a), len(turns))
            self.assertEqual(len(p), len(turns))
        self.assertListEqual(list(harms), list(harmonics))

        bin_width = t_rev[0] / (10 * harmonics[-1])
        n_bins = int(t_rev[0] / bin_width)
        bin_cents = np.linspace(0, t_rev[0], n_bins)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, bin_cents)
        amps_exp, phases_exp = bbuck.waveform_to_harmonics(barrier, harmonics)

        g_comp = bbuck._gain_compensation(
            bin_cents, barrier, harms, amps_exp, phases_exp, t_rev[0]
        )

        amps_exp /= g_comp

        for i, (a, p) in enumerate(zip(amps, phases)):
            self.assertEqual(a[0], amps_exp[i])
            self.assertEqual(p[0], phases_exp[i])

    @multi_backend_testcase
    def test_to_fourier_series_complex(self):
        turns = np.arange(10, dtype=int)
        times = np.linspace(0, 1, len(turns))

        cent = (times, np.linspace(450e-9, 550e-9, 10))
        width = np.linspace(200e-9, 100e-9, len(turns))
        ampl = np.linspace(1e3, 2e3, len(turns))

        generator = bbuck.BarrierGenerator()

        generator.schedule("t_center", cent)
        generator.schedule("t_width", width)
        generator.schedule("peak", ampl)

        t_rev = np.linspace(1000e-9, 900e-9, 10)
        harmonics = np.arange(1, 21)

        harms, amps, phases = generator.to_fourier_series(
            t_rev, harmonics, turns, times, m=0
        )

        self.assertEqual(len(harms), len(amps))
        self.assertEqual(len(harms), len(phases))

        for c, w, a, t in zip(cent[1], width, ampl, t_rev):
            bin_width = t / (10 * harmonics[-1])
            n_bins = int(t / bin_width)
            bin_cents = np.linspace(0, t_rev[0], n_bins)
            barrier = bbuck.compute_sin_barrier(c, w, a, bin_cents)

            b_max = np.max(barrier)
            b_min = np.min(barrier)

            high = int(np.where(barrier == b_max)[0][0])
            low = int(np.where(barrier == b_min)[0][0])

            self.assertAlmostEqual(bin_cents[high], c + w / 4)
            self.assertAlmostEqual(bin_cents[low], c - w / 4)


if __name__ == "__main__":
    unittest.main()
