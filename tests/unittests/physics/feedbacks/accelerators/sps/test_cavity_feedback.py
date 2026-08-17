import copy
import unittest

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    proton,
)
from blond.physics.feedbacks.accelerators.sps import (
    SPSCavityFeedback,
    SPSCavityFeedbackCommissioning,
)


class TestSPSCavityFeedback(unittest.TestCase):
    def create_scenario(self):
        # TODO: implement
        pass

    def test_one_turn_delay_feedback(self):
        # TODO: implement
        pass


class TestSPSCavityFeedbackTransferFunction(unittest.TestCase):
    def create_scenario(self):
        # TODO: implement
        pass

    def test_open_loop_response(self):
        # TODO: implement
        pass

    def test_closed_loop_response(self):
        # TODO: implement
        pass

    def test_one_turn_delay_feedback_reponse(self):
        # TODO: implement
        pass
