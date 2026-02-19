import unittest
from unittest.mock import Mock

from blond import Simulation, SingleHarmonicRFStation, StaticProfile
from blond.experimental.physics.feedbacks.cavity_feedback import (
    IQCavityFeedback,
)


class IQFDBKTester(IQCavityFeedback):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def circuit_track(self, no_beam: bool = False) -> None:
        pass

    def update_fb_variables(self) -> None:
        pass


class IQCavityFeedbackTest(unittest.TestCase):
    def setUp(self):
        self.profile = Mock(spec=StaticProfile)

        self.fdbk = IQFDBKTester(
            profile=self.profile,
            n_cavities=1,
            n_periods_coarse=1,
            harmonic_index=0,
        )

        self.fdbk._parent_rf_station = Mock(spec=SingleHarmonicRFStation)
        self.fdbk._parent_rf_station.harmonic = 50
        self.fdbk._parent_rf_station.omega_rf_design = 5e6
        self.fdbk._parent_rf_station.omega_rf = 5e6
        self.fdbk._parent_rf_station.phi_rf = 0
        self.fdbk.update_rf_variables()

    def test_discontinuity(self) -> None:
        assert self.fdbk.rf_centers

        pass
