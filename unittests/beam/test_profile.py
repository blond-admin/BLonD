import unittest

from blond.beam.profile import CutOptions, Profile


class TestCutOptions(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.cut_options = CutOptions(
            cut_left=None,
            cut_right=None,
            n_slices=None,
            n_sigma=None,
            cuts_unit=None,
            RFSectionParameters=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_set_cuts(self):
        # TODO: implement test for `set_cuts`
        self.cut_options.set_cuts(Beam=None)

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.cut_options.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.cut_options.to_gpu(recursive=None)


class TestProfile(unittest.TestCase):
    @unittest.skip
    def setUp(self):
        # TODO: implement test for `__init__`
        self.profile = Profile(
            Beam=None,
            CutOptions=None,
            FitOptions=None,
            FilterOptions=None,
            OtherSlicesOptions=None,
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test__slice_smooth(self):
        # TODO: implement test for `_slice_smooth`
        self.profile._slice_smooth(reduce=None)

    @unittest.skip
    def test_beam_profile_derivative(self):
        # TODO: implement test for `beam_profile_derivative`
        self.profile.beam_profile_derivative(mode=None)

    @unittest.skip
    def test_fwhm_multibunch(self):
        # TODO: implement test for `fwhm_multibunch`
        self.profile.fwhm_multibunch(
            n_bunches=None,
            bunch_spacing_buckets=None,
            bucket_size_tau=None,
            bucket_tolerance=None,
            shift=None,
        )

    @unittest.skip
    def test_reduce_histo(self):
        # TODO: implement test for `reduce_histo`
        self.profile.reduce_histo(dtype=None)

    @unittest.skip
    def test_rms_multibunch(self):
        # TODO: implement test for `rms_multibunch`
        self.profile.rms_multibunch(
            n_bunches=None,
            bunch_spacing_buckets=None,
            bucket_size_tau=None,
            bucket_tolerance=None,
        )

    @unittest.skip
    def test_to_cpu(self):
        # TODO: implement test for `to_cpu`
        self.profile.to_cpu(recursive=None)

    @unittest.skip
    def test_to_gpu(self):
        # TODO: implement test for `to_gpu`
        self.profile.to_gpu(recursive=None)
