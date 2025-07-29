import unittest
from unittest.mock import Mock

import numpy as np
from numpy._typing import NDArray as NumpyArray
from scipy.constants import speed_of_light as c0

from blond3 import WakeField, Simulation, proton
from blond3._core.backends.backend import backend, Numpy32Bit, Numpy64Bit
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.cavities import (
    CavityBaseClass,
    MultiHarmonicCavity,
    SingleHarmonicCavity,
)
from blond3.physics.feedbacks.base import LocalFeedback


class CavityBaseClassHelper(CavityBaseClass):
    def voltage_waveform_tmp(self, ts: NumpyArray):
        pass

    def calc_omega(self, beam_beta: float, ring_circumference: float):
        pass


class TestMultiHarmonicCavity(unittest.TestCase):
    def setUp(self):
        self.multi_harmonic_cavity = MultiHarmonicCavity.headless(
            section_index=0,
            voltage=np.array([1e6, 2e6], dtype=backend.float),
            phi_rf=np.array([0.1 * np.pi, np.pi], dtype=backend.float),
            harmonic=np.array([1, 5], dtype=backend.float),
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            total_energy=939,
            main_harmonic_idx=0,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self):
        from blond3._core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.particle_type = proton
        beam.reference_time = 0
        beam.reference_beta = 0.5
        beam.reference_velocity = beam.reference_beta * c0
        beam.reference_gamma = np.sqrt(1 - 0.25)  # beta**2
        beam.reference_total_energy = 938
        beam.dE = np.linspace(-1e6, 1e6, 10, dtype=backend.float)  # delta E
        # in eV
        beam.dt = np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)  # delta t
        # in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.multi_harmonic_cavity.track(beam=beam)

        self.assertEqual(beam.reference_total_energy, 939)  # incremented
        self.assertEqual(beam.reference_time, 0)  # unchanged
        print(beam.dE.tolist())
        np.testing.assert_allclose(  # changer/ test pinned to some value
            beam.dE,
            [
                -3553222.1295187217,
                229103.39306234661,
                -2334151.389566862,
                -1291443.680401674,
                1796893.796132672,
                -1195065.0503718334,
                1768699.6153487992,
                2588047.0010012407,
                -251122.31467230315,
                3259845.9525205432,
            ],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

        np.testing.assert_allclose(  # unchanged
            beam.dt,
            np.linspace(-1e-6, 1e-6, 10),
        )


class TestSingleHarmonicCavity(unittest.TestCase):
    def setUp(self):
        from blond3._core.backends.backend import backend, Numpy32Bit, Numpy64Bit

        self.single_harmonic_cavity = SingleHarmonicCavity.headless(
            section_index=0,
            voltage=1e6,
            phi_rf=np.pi * 0.3,
            harmonic=3.5,
            circumference=456,
            local_wakefield=None,
            cavity_feedback=None,
            total_energy=939,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self):
        from blond3._core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.particle_type = proton
        beam.reference_time = backend.float(0)
        beam.reference_beta = 0.5
        beam.reference_velocity = backend.float(beam.reference_beta * c0)
        beam.reference_gamma = backend.float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference_total_energy = backend.float(938)
        beam.dE = np.linspace(-1e6, 1e6, 10, dtype=backend.float)  # delta E in eV
        beam.dt = np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)  # delta t in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.single_harmonic_cavity.track(beam=beam)

        self.assertEqual(beam.reference_total_energy, 939)  # incremented
        self.assertEqual(beam.reference_time, 0)  # unchanged
        print(beam.dE.tolist())
        np.testing.assert_allclose(  # test pinned to some value
            beam.dE,
            [
                -1003263.8619856804,
                221697.39838640607,
                -623504.6270207566,
                -1327969.3279760184,
                27701.968069498133,
                1095854.844844814,
                124356.9102684273,
                -414301.0439499994,
                1055852.7198949838,
                1950042.1738763654,
            ],
            rtol=1e-5 if backend.float == np.float32 else 1e-12,
        )

        np.testing.assert_allclose(  # unchanged
            beam.dt,
            np.linspace(-1e-6, 1e-6, 10),
        )


if __name__ == "__main__":
    unittest.main()
