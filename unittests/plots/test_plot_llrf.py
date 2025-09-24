import unittest

from blond.plots.plot_llrf import (
    plot_COM_motion,
    plot_LHCNoiseFB,
    plot_LHCNoiseFB_FWHM,
    plot_LHCNoiseFB_FWHM_bbb,
    plot_PL_RF_freq,
    plot_PL_RF_phase,
    plot_PL_bunch_phase,
    plot_PL_freq_corr,
    plot_PL_phase_corr,
    plot_RF_phase_error,
    plot_RL_radial_error,
    plot_noise_spectrum,
    plot_phase_noise,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_plot_COM_motion(self):
        # TODO: implement test for `plot_COM_motion`
        plot_COM_motion(
            Ring=None,
            RFStation=None,
            h5data=None,
            output_freq=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_LHCNoiseFB(self):
        # TODO: implement test for `plot_LHCNoiseFB`
        plot_LHCNoiseFB(
            RFStation=None,
            LHCNoiseFB=None,
            h5data=None,
            output_freq=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_LHCNoiseFB_FWHM(self):
        # TODO: implement test for `plot_LHCNoiseFB_FWHM`
        plot_LHCNoiseFB_FWHM(
            RFStation=None,
            LHCNoiseFB=None,
            h5data=None,
            output_freq=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_LHCNoiseFB_FWHM_bbb(self):
        # TODO: implement test for `plot_LHCNoiseFB_FWHM_bbb`
        plot_LHCNoiseFB_FWHM_bbb(
            RFStation=None,
            LHCNoiseFB=None,
            h5data=None,
            output_freq=None,
            dirname=None,
            show_plot=None,
        )

    @unittest.skip
    def test_plot_PL_RF_freq(self):
        # TODO: implement test for `plot_PL_RF_freq`
        plot_PL_RF_freq(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_PL_RF_phase(self):
        # TODO: implement test for `plot_PL_RF_phase`
        plot_PL_RF_phase(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_PL_bunch_phase(self):
        # TODO: implement test for `plot_PL_bunch_phase`
        plot_PL_bunch_phase(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_PL_freq_corr(self):
        # TODO: implement test for `plot_PL_freq_corr`
        plot_PL_freq_corr(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_PL_phase_corr(self):
        # TODO: implement test for `plot_PL_phase_corr`
        plot_PL_phase_corr(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_RF_phase_error(self):
        # TODO: implement test for `plot_RF_phase_error`
        plot_RF_phase_error(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_RL_radial_error(self):
        # TODO: implement test for `plot_RL_radial_error`
        plot_RL_radial_error(
            RFStation=None, h5data=None, output_freq=None, dirname=None, show_plot=None
        )

    @unittest.skip
    def test_plot_noise_spectrum(self):
        # TODO: implement test for `plot_noise_spectrum`
        plot_noise_spectrum(
            frequency=None,
            spectrum=None,
            sampling=None,
            dirname=None,
            show_plot=None,
            figno=None,
        )

    @unittest.skip
    def test_plot_phase_noise(self):
        # TODO: implement test for `plot_phase_noise`
        plot_phase_noise(
            time=None,
            dphi=None,
            sampling=None,
            dirname=None,
            show_plot=None,
            figno=None,
        )
