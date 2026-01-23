import copy
import unittest

import numpy as np

from blond import (
    Ring,
    SynchrotronRadiationMaster,
)


class TestSynchrotronRadiationMaster(unittest.TestCase):
    def setUp(self):
        self.synchrotron_radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.SR_ring = Ring(
            10.0, radiation_integrals=self.synchrotron_radiation_integrals
        )
        self.ring = copy.deepcopy(self.SR_ring)

    def test_inputs(self):
        self.SRHandler = SynchrotronRadiationMaster(
            radiation_integrals=self.synchrotron_radiation_integrals,
        )

        self.SRHandler.generate_synchrotron_radiation_subclasses(
            ring=self.ring,
        )
