import unittest

from blond.plots.plot_slices import plot_beam_profile, plot_beam_profile_derivative, \
    plot_beam_spectrum


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_plot_beam_profile(self):
        # TODO: implement test for `plot_beam_profile`
        plot_beam_profile(Profile=None, counter=None, style=None, dirname=None, show_plot=None)

    @unittest.skip
    def test_plot_beam_profile_derivative(self):
        # TODO: implement test for `plot_beam_profile_derivative`
        plot_beam_profile_derivative(Profile=None, counter=None, style=None, dirname=None, show_plot=None, modes=None)

    @unittest.skip
    def test_plot_beam_spectrum(self):
        # TODO: implement test for `plot_beam_spectrum`
        plot_beam_spectrum(Profile=None, counter=None, style=None, dirname=None, show_plot=None)
