import unittest

from blond.plots.plot_impedance import plot_impedance_vs_frequency, \
    plot_induced_voltage_vs_bin_centers, plot_wake_vs_time


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_plot_impedance_vs_frequency(self):
        # TODO: implement test for `plot_impedance_vs_frequency`
        plot_impedance_vs_frequency(induced_voltage_freq=None, figure_index=None, plot_total_impedance=None, plot_spectrum=None, plot_interpolated_impedances=None, style=None, cut_left_right=None, cut_up_down=None, dirname=None, show_plots=None)

    @unittest.skip
    def test_plot_induced_voltage_vs_bin_centers(self):
        # TODO: implement test for `plot_induced_voltage_vs_bin_centers`
        plot_induced_voltage_vs_bin_centers(total_induced_voltage=None, style=None, figure_index=None, dirname=None, show_plots=None)

    @unittest.skip
    def test_plot_wake_vs_time(self):
        # TODO: implement test for `plot_wake_vs_time`
        plot_wake_vs_time(induced_voltage_time=None, figure_index=None, plot_total_wake=None, plot_interpolated_wake=None, style=None, cut_left_right=None, cut_up_down=None, dirname=None, show_plots=None)
