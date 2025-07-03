import unittest

import numpy as np

from blond3 import RfStationParams
from blond3.physics.cavities import CavityBaseClass
from blond3.testing.simulation import SimulationTwoRfStations


class TestRfStationParams(unittest.TestCase):
    def setUp(self):
        self.rf_station_params = RfStationParams(
            harmonic=1,
            voltage=1,
            phi_rf=1,
            omega_rf=1,
            # phi_noise=[None],
            # phi_modulation=None,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_on_init_simulation(self):
        simulation = SimulationTwoRfStations().simulation
        cavity = simulation.ring.elements.get_element(CavityBaseClass, 1)

        for input in (
            1,
            1.0,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            np.linspace(0, 10, 10),
            (np.linspace(0, 10, 10), np.linspace(0, 10, 10)),
        ):
            rf_station_params = RfStationParams(
                harmonic=input,
                voltage=1,
                phi_rf=1,
                omega_rf=1,
                # phi_noise=[None],
                # phi_modulation=None,
            )
            rf_station_params.set_owner(
                cavity
            )
            rf_station_params.on_init_simulation(simulation=simulation)
            self.assertEqual(rf_station_params.harmonic.shape, (1, 10))

    @unittest.skip # TODO
    def test_track(self):
        # TODO: implement test for `track`
        self.rf_station_params.track()
