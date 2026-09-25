import unittest

from blond import WakeField
from blond.generals.late_init import NotInitialisedError
from blond.physics.impedances.solvers import TimeDomainFftSolver
from blond.physics.impedances.sources import Resonators
from blond.testing.backend_testing import BLonDTestCase


class TestWakeFieldLateInit(BLonDTestCase):
    def setUp(self):
        self.wakefield = WakeField(
            sources=(
                Resonators(
                    shunt_impedances=1.0,
                    center_frequencies=1e9,
                    quality_factors=1.0,
                ),
            ),
            solver=TimeDomainFftSolver(),
        )

    def test_profile_before_init_raises(self):
        with self.assertRaisesRegex(NotInitialisedError, "on_init_simulation"):
            _ = self.wakefield.profile

    def test_induced_voltage_before_calc_raises(self):
        with self.assertRaisesRegex(
            NotInitialisedError, "calc_induced_voltage"
        ):
            _ = self.wakefield.induced_voltage


if __name__ == "__main__":
    unittest.main()
