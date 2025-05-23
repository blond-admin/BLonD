import unittest

from blond.plots.plot_beams import (
    plot_bunch_length_evol,
    plot_bunch_length_evol_gaussian,
    plot_energy_evol,
    plot_long_phase_space,
    plot_position_evol,
    plot_transmitted_particles,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_plot_bunch_length_evol(self):
        # TODO: implement test for `plot_bunch_length_evol`
        plot_bunch_length_evol(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_bunch_length_evol_gaussian(self):
        # TODO: implement test for `plot_bunch_length_evol_gaussian`
        plot_bunch_length_evol_gaussian(
            RFStation=None,
            Profile=None,
            h5data=None,
            output_freq=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_energy_evol(self):
        # TODO: implement test for `plot_energy_evol`
        plot_energy_evol(
            RFStation=None,
            h5data=None,
            output_freq=None,
            style=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_long_phase_space(self):
        # TODO: implement test for `plot_long_phase_space`
        plot_long_phase_space(
            Ring=None,
            RFStation=None,
            Beam=None,
            xmin=None,
            xmax=None,
            ymin=None,
            ymax=None,
            xunit=None,
            sampling=None,
            separatrix_plot=None,
            histograms_plot=None,
            dirname=None,
            show_plot=None,
            alpha=None,
            color=None,
        )

    @unittest.skip
    def test_plot_position_evol(self):
        # TODO: implement test for `plot_position_evol`
        plot_position_evol(
            RFStation=None,
            h5data=None,
            output_freq=None,
            style=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_transmitted_particles(self):
        # TODO: implement test for `plot_transmitted_particles`
        plot_transmitted_particles(
            RFStation=None,
            h5data=None,
            output_freq=None,
            style=None,
            dirname=None,
            show_plot=None,
        )
