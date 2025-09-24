import unittest

from blond.plots.plot import fig_folder, Plot


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_fig_folder(self):
        # TODO: implement test for `fig_folder`
        fig_folder(dirname=None)


class TestPlot(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.plot = Plot(
            Ring=None,
            RFStation=None,
            Beam=None,
            dt_plot=None,
            dt_bckp=None,
            xmin=None,
            xmax=None,
            ymin=None,
            ymax=None,
            xunit=None,
            sampling=None,
            show_plots=None,
            separatrix_plot=None,
            histograms_plot=None,
            Profile=None,
            h5file=None,
            output_frequency=None,
            PhaseLoop=None,
            LHCNoiseFB=None,
            format_options=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_reset_frame(self):
        # TODO: implement test for `reset_frame`
        self.plot.reset_frame(xmin=None, xmax=None, ymin=None, ymax=None)

    @unittest.skip
    def test_set_format(self):
        # TODO: implement test for `set_format`
        self.plot.set_format(format_options=None)

    @unittest.skip
    def test_track(self):
        # TODO: implement test for `track`
        self.plot.track()
