import unittest

from blond.toolbox.filters_and_fitting import (
    beam_profile_filter_chebyshev,
    fwhm,
    fwhm_multibunch,
    rms_multibunch,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_beam_profile_filter_chebyshev(self):
        # TODO: implement test for `beam_profile_filter_chebyshev`
        beam_profile_filter_chebyshev(Y_array=None, X_array=None, filter_option=None)

    @unittest.skip
    def test_fwhm(self):
        # TODO: implement test for `fwhm`
        fwhm(Y_array=None, X_array=None, shift=None)

    @unittest.skip
    def test_fwhm_multibunch(self):
        # TODO: implement test for `fwhm_multibunch`
        fwhm_multibunch(
            Y_array=None,
            X_array=None,
            n_bunches=None,
            bunch_spacing_buckets=None,
            bucket_size_tau=None,
            bucket_tolerance=None,
            shift=None,
        )

    @unittest.skip
    def test_rms_multibunch(self):
        # TODO: implement test for `rms_multibunch`
        rms_multibunch(
            Y_array=None,
            X_array=None,
            n_bunches=None,
            bunch_spacing_buckets=None,
            bucket_size_tau=None,
            bucket_tolerance=None,
        )
