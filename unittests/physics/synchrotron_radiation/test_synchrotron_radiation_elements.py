import unittest
from functools import cached_property
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from numpy._typing import NDArray as NumpyArray
from scipy.constants import c, e
from scipy.constants import speed_of_light as c0

from blond import (
    SynchrotronRadiationBaseClass,
    WigglerMagnet,
    _SynchrotronRadiationDrift,
    _SynchrotronRadiationSection,
    electron,
)
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.backends.backend import backend
from blond.core.base import SimulationElementBase
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType

if TYPE_CHECKING:
    from numpy._typing import NDArray as NumpyArray


class BeamBaseClassTester(BeamBaseClass):
    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed=False,
    ):
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
            is_distributed=is_distributed,
        )
        # self.reference = Mock(ReferenceCoordinates)
        self.reference._particle_type = particle_type
        self.reference.time = 0
        self.reference_beta = 0.99
        self.reference_velocity = self.reference_beta * c0
        self.reference_gamma = np.sqrt(1 - 0.99**2)  # beta**2
        self.reference_total_energy = 20e9
        self.reference.total_energy = 20e9
        self._dE = np.linspace(-1e6, 1e6, 10, dtype=backend.float)  # delta E
        # in eV
        self._dt = np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)  # delta t
        # in s
        self._flags = np.zeros(10, dtype=np.int32)
        self._ids = np.arange(10, dtype=np.int32)

    @cached_property
    def ratio(self) -> float:
        return self.intensity / self.common_array_size

    def setup_beam(
        self,
        dt: NumpyArray,
        dE: NumpyArray,
        flags: NumpyArray = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
    ):
        """Sets beam array attributes for simulation

        Parameters
        ----------
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
        flags
            Macro-particle flags
        reference_time
            Time of the reference frame (global time), in [s]
        reference_total_energy
            Time of the reference frame (global total energy), in [eV]
        """
        pass

    def plot_hist2d(self):
        pass

    def dE_max(self) -> float:
        pass

    def dt_min(self) -> float:
        pass

    def dt_max(self) -> float:
        pass

    def dE_min(self) -> float:
        pass

    def common_array_size(self) -> int:
        pass

    def rms_emittance(self) -> int:
        pass


class TestSynchrotronRadiationBaseClass(unittest.TestCase):
    def setUp(self) -> None:
        self.radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.SRB = SynchrotronRadiationBaseClass(
            share_of_synchrotron_radiation_integrals=0.1
            * self.radiation_integrals,
            _disable_quantum_excitation=True,
        )

        self.SRD = _SynchrotronRadiationDrift(
            share_of_synchrotron_radiation_integrals=0.1
            * self.radiation_integrals,
            _disable_quantum_excitation=True,
        )
        self.SRS = _SynchrotronRadiationSection(
            share_of_synchrotron_radiation_integrals=0.1
            * self.radiation_integrals,
            _disable_quantum_excitation=True,
        )
        # To test the tracking methods
        self.SRB._simulation = Mock(SimulationElementBase)
        self.SRB._simulation.turn_i = 0
        self.SRD._simulation = Mock(SimulationElementBase)
        self.SRD._simulation.turn_i = 0
        self.SRS._simulation = Mock(SimulationElementBase)
        self.SRS._simulation.turn_i = 0

        self.beam = BeamBaseClassTester(
            intensity=1e12,
            particle_type=electron,
            is_counter_rotating=False,
            is_distributed=False,
        )

        self.decimal = 6 if backend.float == np.float32 else 12

        self.U0, self.tau_z, self.sigma0 = (
            gather_longitudinal_synchrotron_radiation_parameters(
                particle_type=self.beam.particle_type,
                energy=self.beam.reference.total_energy,
                synchrotron_radiation_integrals=self.radiation_integrals,
            )
        )

        self.seed = 500

    def test_inputs_SynchrotronRadiationBaseClass(self):
        np.testing.assert_array_equal(
            self.SRB._fractional_radiation_integrals,
            0.1 * self.radiation_integrals,
        )
        self.assertIsNone(self.SRB._energy_lost_due_to_synchrotron_radiation)
        self.assertIsNone(self.SRB._damping_time)
        self.assertIsNone(self.SRB._natural_energy_spread)

    def test_inputs_SynchrotronRadiationDrift(self):
        np.testing.assert_array_equal(
            self.SRD._fractional_radiation_integrals,
            0.1 * self.radiation_integrals,
        )
        self.assertIsNone(self.SRD._energy_lost_due_to_synchrotron_radiation)
        self.assertIsNone(self.SRD._damping_time)
        self.assertIsNone(self.SRD._natural_energy_spread)

    def test_inputs_SynchrotronRadiationSection(self):
        np.testing.assert_array_equal(
            self.SRS._fractional_radiation_integrals,
            0.1 * self.radiation_integrals,
        )
        self.assertIsNone(self.SRS._energy_lost_due_to_synchrotron_radiation)
        self.assertIsNone(self.SRS._damping_time)
        self.assertIsNone(self.SRS._natural_energy_spread)

    def test_calculate_kick_SynchrotronRadiationBaseClass(self):
        np.random.seed(seed=self.seed)
        _ = self.SRB._calculate_kick(
            beam=self.beam,
        )

        # SynchrotronRadiationBaseClass
        self.assertAlmostEqual(
            self.SRB._energy_lost_due_to_synchrotron_radiation,
            np.float64(133731.76297928384),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRB._damping_time,
            np.float64(149552.35530506275),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRB._natural_energy_spread,
            np.float64(0.0001675968578478592),
            places=self.decimal,
        )

    def test_calculate_kick_SynchrotronRadiationDrift(self):
        np.random.seed(seed=self.seed)
        _ = self.SRD._calculate_kick(
            beam=self.beam,
        )
        # Same outputs for the _SynchrotronRadiationDrift class
        self.assertAlmostEqual(
            self.SRD._energy_lost_due_to_synchrotron_radiation,
            np.float64(133731.76297928384),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRD._damping_time,
            np.float64(149552.35530506275),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRD._natural_energy_spread,
            np.float64(0.0001675968578478592),
            places=self.decimal,
        )

    def test_calculate_kick_SynchrotronRadiationSection(self):
        # Same outputs for the _SynchrotronRadiationSection class
        np.random.seed(seed=self.seed)
        _ = self.SRS._calculate_kick(
            beam=self.beam,
        )
        self.assertAlmostEqual(
            self.SRS._energy_lost_due_to_synchrotron_radiation,
            np.float64(133731.76297928384),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRS._damping_time,
            np.float64(149552.35530506275),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRS._natural_energy_spread,
            np.float64(0.0001675968578478592),
            places=self.decimal,
        )

    def test_all_energy_kicks_are_equal(self):
        np.random.seed(seed=self.seed)
        energy_kick_from_SynchrotronRadiationBaseClass = (
            self.SRB._calculate_kick(
                beam=self.beam,
            )
        )
        energy_kick_from_SynchrotronRadiationDrift = self.SRD._calculate_kick(
            beam=self.beam,
        )
        energy_kick_from_SynchrotronRadiationSection = (
            self.SRS._calculate_kick(
                beam=self.beam,
            )
        )
        np.testing.assert_array_equal(
            energy_kick_from_SynchrotronRadiationBaseClass,
            energy_kick_from_SynchrotronRadiationDrift,
        )

        np.testing.assert_array_equal(
            energy_kick_from_SynchrotronRadiationBaseClass,
            energy_kick_from_SynchrotronRadiationSection,
        )

        np.testing.assert_array_equal(
            energy_kick_from_SynchrotronRadiationDrift,
            energy_kick_from_SynchrotronRadiationSection,
        )

    def test_update_beam_energy(self):
        previous_energy = self.beam.read_partial_dE().copy()
        energy_kick = self.SRB._calculate_kick(
            beam=self.beam,
        )
        self.SRB._update_beam_energy(beam=self.beam)
        energy_after_one_kick = self.beam.read_partial_dE().copy()
        np.testing.assert_array_equal(
            energy_after_one_kick, previous_energy + energy_kick
        )

        second_energy_kick = self.SRB._calculate_kick(
            beam=self.beam,
        )
        self.SRB.track(beam=self.beam)
        energy_after_two_kicks = self.beam.read_partial_dE().copy()

        np.testing.assert_array_equal(
            energy_after_two_kicks, energy_after_one_kick + second_energy_kick
        )
        np.testing.assert_array_equal(
            energy_after_two_kicks,
            previous_energy + energy_kick + second_energy_kick,
        )


class TestWigglerMagnet(unittest.TestCase):
    def setUp(self) -> None:
        self.wiggler_magnet = WigglerMagnet(
            wiggler_type="sinusoidal",
            number=2,
            peak_field=1,
            pole_length=0.01,
            number_poles=50,
            section_index=0,
        )
        self.wiggler_magnet_none = WigglerMagnet(
            wiggler_type=None,
        )

    def test_inputs_and_properties(self):
        assert self.wiggler_magnet.section_index == 0
        assert self.wiggler_magnet._type == "sinusoidal"
        assert self.wiggler_magnet._number == 2
        assert self.wiggler_magnet._peak_field == 1
        assert self.wiggler_magnet._pole_length == 0.01
        assert self.wiggler_magnet._number_poles == 50

        self.assertEqual(self.wiggler_magnet.number_of_wigglers, 2)
        self.assertEqual(self.wiggler_magnet_none.number_of_wigglers, 1)

        self.assertEqual(self.wiggler_magnet.length_wiggler, 50 * 0.01)
        self.assertIsNone(self.wiggler_magnet_none.length_wiggler)

        self.assertEqual(self.wiggler_magnet.number_of_poles, 50)
        self.assertEqual(self.wiggler_magnet_none.number_of_poles, 43)

        self.assertEqual(self.wiggler_magnet.peak_magnetic_field, 1)
        self.assertEqual(self.wiggler_magnet_none.peak_magnetic_field, 1)

        self.assertEqual(self.wiggler_magnet.pole_length, 0.01)
        self.assertEqual(self.wiggler_magnet_none.pole_length, 0.095)

    def test_calculate_energy_contribution_to_synchrotron_radiation_integrals(
        self,
    ):
        energy_contribution_wiggler_integrals = self.wiggler_magnet._calculate_energy_contribution_to_synchrotron_radiation_integrals(
            reference_energy=20e9
        )
        var = 1 / (20e9 * e / c)
        expected_array = np.array(
            [
                var**2,
                var**2,
                var**3,
                var**3,
                var**5,
            ]
        )
        np.testing.assert_array_equal(
            energy_contribution_wiggler_integrals, expected_array
        )

    def test_calculate_contribution_to_synchrotron_radiation_integrals(self):
        (
            self.wiggler_magnet._calculate_contribution_to_synchrotron_radiation_integrals_without_beam_energy()
        )
        (
            self.wiggler_magnet_none._calculate_contribution_to_synchrotron_radiation_integrals_without_beam_energy()
        )

        self.assertIsNone(
            self.wiggler_magnet_none._contribution_to_synchrotron_radiation_integrals_without_energy,
        )

        self.assertEqual(
            self.wiggler_magnet._contribution_to_synchrotron_radiation_integrals_without_energy[
                0
            ],
            -1
            / 2
            * 2
            * 50
            * 0.01
            * (e * 1) ** 2
            * (50 * 0.01 / (2 * np.pi)) ** 2,
        )
        self.assertEqual(
            self.wiggler_magnet._contribution_to_synchrotron_radiation_integrals_without_energy[
                1
            ],
            1 / 2 * 2 * 50 * 0.01 * (e * 1) ** 2,
        )
        self.assertEqual(
            self.wiggler_magnet._contribution_to_synchrotron_radiation_integrals_without_energy[
                2
            ],
            4 / (3 * np.pi) * 2 * 50 * 0.01 * (e * 1) ** 3,
        )
        self.assertEqual(
            self.wiggler_magnet._contribution_to_synchrotron_radiation_integrals_without_energy[
                3
            ],
            0,
        )
        self.assertEqual(
            self.wiggler_magnet._contribution_to_synchrotron_radiation_integrals_without_energy[
                4
            ],
            2 * 0.01**2 * 50 * 0.01 / (15 * np.pi**3) * (e * 1) ** 5,
        )
