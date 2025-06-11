# General imports
import unittest
import numpy as np

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

        amps_exp = [19.9, 38.0, 51.3, 57.5, 55.8, 47.2, 33.7, 18.2, 3.6, -7.9,
                    -15.0, -17.5]

        for a, a_exp in zip(amps, amps_exp):
            self.assertAlmostEqual(a, a_exp, places = 1)


if __name__ == "__main__":
    unittest.main()
