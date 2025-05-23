import unittest

from blond.toolbox.action import (
    action_from_oscillation_amplitude,
    action_from_phase_amplitude,
    oscillation_amplitude_from_coordinates,
    phase_amplitude_from_tune,
    tune_from_phase_amplitude,
    x,
    x2,
)


class TestFunctions(unittest.TestCase):
    @unittest.skip
    def test_action_from_oscillation_amplitude(self):
        # TODO: implement test for `action_from_oscillation_amplitude`
        action_from_oscillation_amplitude(
            RFStation=None, dtmax=None, timestep=None, Np_histogram=None
        )

    @unittest.skip
    def test_action_from_phase_amplitude(self):
        # TODO: implement test for `action_from_phase_amplitude`
        action_from_phase_amplitude(x2=None)

    @unittest.skip
    def test_oscillation_amplitude_from_coordinates(self):
        # TODO: implement test for `oscillation_amplitude_from_coordinates`
        oscillation_amplitude_from_coordinates(
            Ring=None,
            RFStation=None,
            dt=None,
            dE=None,
            timestep=None,
            Np_histogram=None,
        )

    @unittest.skip
    def test_phase_amplitude_from_tune(self):
        # TODO: implement test for `phase_amplitude_from_tune`
        phase_amplitude_from_tune(tune=None)

    @unittest.skip
    def test_tune_from_phase_amplitude(self):
        # TODO: implement test for `tune_from_phase_amplitude`
        tune_from_phase_amplitude(phimax=None)

    @unittest.skip
    def test_x(self):
        # TODO: implement test for `x`
        x(phimax=None)

    @unittest.skip
    def test_x2(self):
        # TODO: implement test for `x2`
        x2(phimax=None)
