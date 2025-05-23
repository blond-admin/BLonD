import unittest

from blond.toolbox.diffusion import phase_noise_diffusion


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_phase_noise_diffusion(self):
        # TODO: implement test for `phase_noise_diffusion`
        phase_noise_diffusion(
            Ring=None,
            RFStation=None,
            spectrum=None,
            distribution=None,
            distributionBins=None,
            Ngrids=None,
            M=None,
            iterations=None,
            figdir=None,
        )
