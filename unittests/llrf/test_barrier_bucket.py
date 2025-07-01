# General imports
import unittest
import numpy as np
import numpy.testing as nptest

# BLonD imports
import blond.llrf.barrier_bucket as bbuck


class TestBarrierBucketFunctions(unittest.TestCase):

    def test_simple_barrier(self):

        cent = 500E-9
        width = 100E-9
        ampl = 1E3

        centers = np.linspace(0, 1000E-9, 5000)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, centers,
                                            periodic = False)

        self.assertAlmostEqual(np.max(barrier), 1E3, places = 1)

        left_pts = np.where(centers < cent-width/2)[0]
        right_pts = np.where(centers > cent+width/2)[0]

        self.assertListEqual(list(barrier[left_pts]),
                             list(np.zeros_like(left_pts)))
        self.assertListEqual(list(barrier[right_pts]),
                             list(np.zeros_like(right_pts)))

        self.assertAlmostEqual(np.max(barrier[left_pts[-1]:right_pts[0]]),
                               1E3, places=1)
        self.assertAlmostEqual(np.min(barrier[left_pts[-1]:right_pts[0]]),
                               -1E3, places=1)

    def test_periodic_barrier(self):

        cent = 1000E-9
        width = 100E-9
        ampl = 1E3

        centers = np.linspace(0, 1000E-9, 5000)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, centers,
                                            periodic = True)

        self.assertAlmostEqual(np.max(barrier), 1E3, places = 1)

        left_pts = np.where(centers < centers[0]+width/2)[0]
        right_pts = np.where(centers > centers[-1]-width/2)[0]

        self.assertListEqual(list(barrier[left_pts[-1]:right_pts[0]]),
                             list(barrier[left_pts[-1]:right_pts[0]]))

        self.assertAlmostEqual(np.max(barrier[left_pts]),
                               1E3, places=1)
        self.assertAlmostEqual(np.min(barrier[right_pts]),
                               -1E3, places=1)

    def test_wide_barrier(self):

        cent = 1000E-9
        width = 1100E-9
        ampl = 1E3

        centers = np.linspace(0, 1000E-9, 5000)

        with self.assertRaises(ValueError):
            bbuck.compute_sin_barrier(cent, width, ampl, centers)

    def test_fourier_series(self):

        cent = 500E-9
        width = 100E-9
        ampl = 1E3

        centers = np.linspace(0, 1000E-9, 5000)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, centers,
                                            periodic = False)

        amps, phases = bbuck.waveform_to_harmonics(barrier, np.arange(1, 13))

        amps_exp = [19.9, 39.0, 56.6, 72.1, 84.9, 94.6, 101.0, 103.9, 103.5,
                    100, 93.7, 85.0]
        phases_exp = [3.14, 6.28, 3.14, 6.28, 3.14, 6.28, 3.14, 6.28, 3.14,
                      6.28, 3.13, 6.28]

        for a, a_exp, p, p_exp in zip(amps, amps_exp, phases, phases_exp):
            self.assertAlmostEqual(a, a_exp, places = 1)
            self.assertAlmostEqual(p, p_exp, places = 2)


    def test_sinc_filter(self):

        cent = 500E-9
        width = 100E-9
        ampl = 1E3

        centers = np.linspace(0, 1000E-9, 5000)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, centers,
                                            periodic = False)

        amps, _ = bbuck.waveform_to_harmonics(barrier, np.arange(1, 13))
        amps = bbuck.sinc_filtering(amps, m = 1)

        amps_exp = [19.4, 35.3, 45.1, 47.4, 42.3, 31.6, 17.7, 3.6, -8.2, -16.0,
                    -19.3, -18.4]

        for a, a_exp in zip(amps, amps_exp):
            self.assertAlmostEqual(a, a_exp, places = 1)


    def test_waveform_harmonics(self):

        harms = [1, 2, 3, 4, 5, 6, 7]
        set_amps = [4E3, 0, 3E3, 0, 2E3, 0, 1E3]
        set_phases = [0, 0, np.pi, 0, 0, 0, np.pi]

        t_rev = 1E-6
        centers = np.linspace(0, t_rev, 5000)
        waveform = np.zeros_like(centers)

        for h, a, p in zip(harms, set_amps, set_phases):
            waveform += a * np.sin(2*np.pi*h*centers / t_rev + p)

        comp_amps, comp_phases = bbuck.waveform_to_harmonics(waveform,
                                                             range(1, 8))

        for i in range(len(harms)):
            self.assertAlmostEqual(comp_amps[i], set_amps[i], delta = 1E1)

            if set_amps[i] > 0:
                # Use sin/cos comparison to avoid issues with 0 != 2pi
                self.assertAlmostEqual(np.cos(comp_phases[i]),
                                       np.cos(set_phases[i]), places = 1)
                self.assertAlmostEqual(np.sin(comp_phases[i]),
                                       np.sin(set_phases[i]), places = 1)

        comp_wave = bbuck.harmonics_to_waveform(centers, harms,
                                                set_amps, set_phases,
                                                t_rev)

        nptest.assert_array_almost_equal(waveform, comp_wave, decimal = 2)


class TestBarrierBucketGenerator(unittest.TestCase):

    def test_fixed_barrier(self):

        cent = 500E-9
        width = 100E-9
        ampl = 1E3

        generator = bbuck.BarrierGenerator(cent, width, ampl)
        bin_cents = np.linspace(0, 1000E-9, 1000)
        wave = generator.waveform_at_time(1, bin_cents)

        wave_pts = np.where(wave != 0)[0]
        self.assertAlmostEqual(bin_cents[wave_pts[-1]]
                               - bin_cents[wave_pts[0]], width, places=2)
        self.assertAlmostEqual(np.mean(bin_cents[wave_pts]), cent, places=2)

    def test_variable_barrier(self):

        bin_cents = np.linspace(0, 1000E-9, 10000)

        peak = [[0, 1], [1E3, 4E3]]
        t_cent = [[0, 1], [200E-9, 800E-9]]
        t_width = [[0, 1], [100E-9, 150E-9]]

        generator = bbuck.BarrierGenerator(t_cent, t_width, peak)

        for t in np.linspace(0, 1, 10):
            peak_exp = np.interp(t, peak[0], peak[1])
            cent_exp = np.interp(t, t_cent[0], t_cent[1])
            width_exp = np.interp(t, t_width[0], t_width[1])
            wave = generator.waveform_at_time(t, bin_cents)

            self.assertAlmostEqual(np.max(wave), peak_exp, places=1)
            self.assertAlmostEqual(np.min(wave), -peak_exp, places=1)

            wave_pts = np.where(wave != 0)[0]
            self.assertAlmostEqual(bin_cents[wave_pts[-1]]
                                - bin_cents[wave_pts[0]], width_exp, places=1)
            self.assertAlmostEqual(np.mean(bin_cents[wave_pts]), cent_exp,
                                   places=1)

    def test_for_rf_station(self):

        cent = 500E-9
        width = 100E-9
        ampl = 1E3

        generator = bbuck.BarrierGenerator(cent, width, ampl)

        times = np.linspace(0, 1, 10)
        t_rev = np.zeros_like(times) + 1000E-9
        harmonics = np.arange(1, 11)

        # Unfiltered
        harms, amps, phases = generator.for_rf_station(times, t_rev,
                                                       harmonics, m=0)

        for a, p in zip(amps, phases):
            self.assertListEqual(list(a[0]), list(times))
            self.assertListEqual(list(p[0]), list(times))
        self.assertListEqual(list(harms), list(harmonics))

        bin_width = t_rev[0]/(10*harmonics[-1])
        n_bins = int(t_rev[0]/bin_width)
        bin_cents = np.linspace(0, t_rev[0], n_bins)

        barrier = bbuck.compute_sin_barrier(cent, width, ampl, bin_cents)
        amps_exp, phases_exp = bbuck.waveform_to_harmonics(barrier,
                                                           harmonics)

        g_comp = bbuck._gain_compensation(bin_cents, barrier, harms, amps_exp,
                                          phases_exp, t_rev[0])

        amps_exp /= g_comp

        for i, (a, p) in enumerate(zip(amps, phases)):
            self.assertEqual(a[1, 0], amps_exp[i])
            self.assertEqual(p[1, 0], phases_exp[i])



if __name__ == "__main__":
    unittest.main()
