import unittest
from copy import deepcopy
from unittest.mock import Mock

import numpy as np
from numpy._typing import NDArray as NumpyArray
from scipy.constants import speed_of_light as c0

from blond import Simulation, proton
from blond._core.backends.backend import backend
from blond._core.base import DynamicParameter
from blond.physics.cavities import (
    CavityBaseClass,
    MultiHarmonicCavity,
    SingleHarmonicCavity,
)


class CavityBaseClassHelper(CavityBaseClass):
    def voltage_waveform_tmp(self, ts: NumpyArray):
        pass

    def calc_omega(self, beam_beta: float, ring_circumference: float):
        pass


class TestMultiHarmonicCavity(unittest.TestCase):
    def setUp(self) -> None:
        from blond._core.beam.base import BeamBaseClass

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

        self.beam = beam

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
        self.multi_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self) -> None:
        self.multi_harmonic_cavity.track(beam=self.beam)
        self.multi_harmonic_cavity._ring.average_transition_gamma = 5

        self.assertEqual(self.beam.reference_total_energy, 939)  # incremented
        self.assertEqual(self.beam.reference_time, 0)  # unchanged

        np.testing.assert_allclose(  # changer/ test pinned to some value
            self.beam.dE,
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
            self.beam.dt,
            np.linspace(-1e-6, 1e-6, 10),
        )

    def test_wrong_array(self) -> None:
        local_cav = MultiHarmonicCavity(n_harmonics=2, main_harmonic_idx=0, voltage=np.array([1, 2]),
                                        phi_rf=np.array([3, 4]), harmonic=np.array([5, 6]))
        np.testing.assert_allclose(local_cav.voltage, np.array([1, 2]))
        np.testing.assert_allclose(local_cav.phi_rf, np.array([3, 4]))
        np.testing.assert_allclose(local_cav.harmonic, np.array([5, 6]))

        with self.assertRaises(ValueError):
            _ = MultiHarmonicCavity(n_harmonics=2, main_harmonic_idx=0, voltage=np.array([1]),
                                    phi_rf=np.array([3, 4]), harmonic=np.array([5, 6]))
        with self.assertRaises(ValueError):
            _ = MultiHarmonicCavity(n_harmonics=2, main_harmonic_idx=0, voltage=np.array([1, 2]),
                                    phi_rf=np.array([3]), harmonic=np.array([5, 6]))
        with self.assertRaises(ValueError):
            _ = MultiHarmonicCavity(n_harmonics=2, main_harmonic_idx=0, voltage=np.array([1, 2]),
                                    phi_rf=np.array([3, 4]), harmonic=np.array([5]))

    def test_on_init_simulation_fails(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.voltage = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_general_getters(self) -> None:
        self.multi_harmonic_cavity._update_beam_based_attributes(beam=self.beam)
        assert self.multi_harmonic_cavity.get_main_harmonic() == self.multi_harmonic_cavity.harmonic[
            self.multi_harmonic_cavity.main_harmonic_idx]
        assert self.multi_harmonic_cavity.get_main_harmonic_t_rf_current() == 2 * np.pi / \
               self.multi_harmonic_cavity._omega_rf[self.multi_harmonic_cavity.main_harmonic_idx]
        assert self.multi_harmonic_cavity.calc_main_harmonic_t_rf(beam_beta=self.beam.reference_beta,
                                                                  ring_circumference=456) == self.multi_harmonic_cavity.get_main_harmonic_t_rf_current()

    def test_on_init_simulation_fails2(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.phi_rf = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails3(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.multi_harmonic_cavity.harmonic = None
            self.multi_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )


class TestSingleHarmonicCavity(unittest.TestCase):
    def setUp(self) -> None:
        from blond._core.beam.base import BeamBaseClass

        beam = Mock(BeamBaseClass)
        beam.particle_type = proton
        beam.reference_time = backend.float(0)
        beam.reference_beta = 0.5
        beam.reference_velocity = backend.float(beam.reference_beta * c0)
        beam.reference_gamma = backend.float(np.sqrt(1 - 0.25))  # beta**2
        beam.reference_total_energy = backend.float(938)
        beam.dE = np.linspace(
            -1e6, 1e6, 10, dtype=backend.float
        )  # delta E in eV
        beam.dt = np.linspace(
            -1e-6, 1e-6, 10, dtype=backend.float
        )  # delta t in s
        beam.read_partial_dt.return_value = beam.dt
        beam.write_partial_dE.return_value = beam.dE

        self.beam = beam
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
        self.single_harmonic_cavity._ring.section_lengths = [1, 2, 3]

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_track(self) -> None:

        self.single_harmonic_cavity.track(beam=self.beam)

        self.assertEqual(self.beam.reference_total_energy, 939)  # incremented
        self.assertEqual(self.beam.reference_time, 0)  # unchanged
        print(self.beam.dE.tolist())
        np.testing.assert_allclose(  # test pinned to some value
            self.beam.dE,
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
            self.beam.dt,
            np.linspace(-1e-6, 1e-6, 10),
        )

    def test_general_getters(self) -> None:
        self.single_harmonic_cavity._update_beam_based_attributes(beam=self.beam)
        assert self.single_harmonic_cavity.get_main_harmonic() == self.single_harmonic_cavity.harmonic
        assert self.single_harmonic_cavity.get_main_harmonic_t_rf_current() == 2 * np.pi / \
               self.single_harmonic_cavity._omega_rf
        assert self.single_harmonic_cavity.calc_main_harmonic_t_rf(beam_beta=self.beam.reference_beta,
                                                                   ring_circumference=456) == self.single_harmonic_cavity.get_main_harmonic_t_rf_current()

    def test_on_init_simulation_fails(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.voltage = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails2(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.phi_rf = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )

    def test_on_init_simulation_fails3(self) -> None:
        simulation = Mock(Simulation)
        simulation.turn_i = DynamicParameter(0)
        with self.assertRaises(ValueError):
            self.single_harmonic_cavity.harmonic = None
            self.single_harmonic_cavity.on_init_simulation(
                simulation=simulation
            )


if __name__ == "__main__":
    unittest.main()
