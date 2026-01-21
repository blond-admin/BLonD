import logging
import unittest

import numpy as np
import pytest

from blond import (
    Beam,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.core.backends.backend import Numpy32Bit, backend
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn


class TestDriftIntegration(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    def test_exec(self):
        circumference = 26658.883

        logging.basicConfig(level=logging.INFO)
        ring = Ring(circumference=circumference)

        cavity1 = SingleHarmonicRFStation(
            section_index=0,
            harmonic=35640,
            voltage=6e6,
            phi_rf=0,
        )
        cavity2 = SingleHarmonicRFStation(
            section_index=1,
            harmonic=35640,
            voltage=6e6,
            phi_rf=0,
        )

        N_TURNS = int(1e3)
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=circumference / 3,
            section_index=0,
        )

        drift2 = DriftSimple(
            orbit_length=circumference / 3,
            section_index=1,
        )

        drift3 = DriftSimple(
            orbit_length=circumference / 3,
            section_index=1,
        )
        drift1.transition_gamma = 55.759505
        drift2.transition_gamma = 55.759505
        drift3.transition_gamma = 55.759505
        beam1 = Beam(intensity=1e9, particle_type=proton)

        sim = Simulation.from_locals(locals())
        sim.ring.assert_circumference()
