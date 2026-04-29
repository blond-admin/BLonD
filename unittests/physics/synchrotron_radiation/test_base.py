from __future__ import annotations

import unittest
from functools import cached_property
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.constants import speed_of_light as c0

from blond import (
    Numpy64Bit,
    Simulation,
    copy_to_cpu,
    electron,
)
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.core.backends.backend import backend
from blond.core.base import DynamicParameter, SimulationElementBase
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.synchrotron_radiation.base import (
    SynchrotronRadiationBaseClass,
    calculation_synchrotron_radiation_and_quantum_excitation_energy_kick,
)
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    _SynchrotronRadiationTracker,
)


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
        self._dE = DistributedArray(
            backend.linspace(-1e6, 1e6, 10, dtype=backend.float)
        )  #
        # delta E
        # in eV
        self._dt = DistributedArray(
            backend.linspace(-1e-6, 1e-6, 10, dtype=backend.float)
        )  # delta t
        # in s
        self._flags = backend.zeros(10, dtype=np.int32)
        self._ids = backend.arange(10, dtype=np.int32)

    @cached_property
    def ratio(self) -> float:
        return self.intensity / self.common_array_size

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        mpi_mode: Literal["root-distributes", "all-ranks"] = "all-ranks",
        **kwargs,
    ) -> None:
        """Sets beam array attributes for simulation

        Parameters
        ----------
        mpi_mode
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
        mpi_mode
            Specifies how the particle data is distributed across multiple ranks (processing
            units) in a parallel environment:

            - "root-distributes": The root node (rank 0) holds the full array and splits it
              into smaller chunks, which are then distributed to all ranks, including rank 0.
              Each rank stores its own chunk of the data. This mode is useful when loading
              large datasets (e.g., with `np.loadtxt(...)`) and distributing parts of the data
              across ranks.

            - "all-ranks": Each rank independently generates and stores a full copy of the data.
              While this mode uses more memory, it can be simpler to implement in scenarios where
              each rank needs to work with its own independent data (e.g., generating separate
              random distributions with `np.random.randn()`).
        **kwargs
            Keyword arguments to make the non-abstract implementation
            extendable.
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
            share_of_radiation_integrals=0.1 * self.radiation_integrals,
            disable_quantum_excitation=True,
        )

        self.SRD = _SynchrotronRadiationTracker(
            share_of_radiation_integrals=0.1 * self.radiation_integrals,
            disable_quantum_excitation=True,
        )
        self.SRS = _SynchrotronRadiationTracker(
            share_of_radiation_integrals=0.1 * self.radiation_integrals,
            disable_quantum_excitation=True,
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
                radiation_integrals=self.radiation_integrals,
            )
        )

        self.seed = 500

    def test_inputs_SynchrotronRadiationBaseClass(self):
        np.testing.assert_array_equal(
            self.SRB.share_of_radiation_integrals,
            0.1 * self.radiation_integrals,
        )
        self.assertIsNone(self.SRB._energy_lost_due_to_synchrotron_radiation)
        self.assertIsNone(self.SRB._damping_time)
        self.assertIsNone(self.SRB._natural_energy_spread)

    def test_inputs_SynchrotronRadiationDrift(self):
        np.testing.assert_array_equal(
            self.SRD.share_of_radiation_integrals,
            0.1 * self.radiation_integrals,
        )
        self.assertIsNone(self.SRD._energy_lost_due_to_synchrotron_radiation)
        self.assertIsNone(self.SRD._damping_time)
        self.assertIsNone(self.SRD._natural_energy_spread)

    def test_inputs_SynchrotronRadiationSection(self):
        np.testing.assert_array_equal(
            self.SRS.share_of_radiation_integrals,
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
            copy_to_cpu(energy_kick_from_SynchrotronRadiationBaseClass),
            copy_to_cpu(energy_kick_from_SynchrotronRadiationDrift),
        )

        np.testing.assert_array_equal(
            copy_to_cpu(energy_kick_from_SynchrotronRadiationBaseClass),
            copy_to_cpu(energy_kick_from_SynchrotronRadiationSection),
        )

        np.testing.assert_array_equal(
            copy_to_cpu(energy_kick_from_SynchrotronRadiationDrift),
            copy_to_cpu(energy_kick_from_SynchrotronRadiationSection),
        )

    def test_update_beam_energy(self):
        previous_energy = self.beam.read_partial_dE().copy()
        energy_kick = self.SRB._calculate_kick(
            beam=self.beam,
        )
        self.SRB._update_beam_energy(beam=self.beam)
        energy_after_one_kick = self.beam.read_partial_dE().copy()
        np.testing.assert_array_equal(
            copy_to_cpu(energy_after_one_kick),
            copy_to_cpu(previous_energy + energy_kick),
        )

        second_energy_kick = self.SRB._calculate_kick(
            beam=self.beam,
        )
        self.SRB.track(beam=self.beam)
        energy_after_two_kicks = self.beam.read_partial_dE().copy()

        np.testing.assert_array_equal(
            copy_to_cpu(energy_after_two_kicks),
            copy_to_cpu(energy_after_one_kick + second_energy_kick),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(energy_after_two_kicks),
            copy_to_cpu(previous_energy + energy_kick + second_energy_kick),
        )

    def test_energy_kick_with_quantum_excitation(self):
        rng = np.random.default_rng(seed=self.seed)
        energy_kick = calculation_synchrotron_radiation_and_quantum_excitation_energy_kick(
            beam_delta_energy_array=20e9 * np.ones(1000),
            energy_lost=13e6,
            longitudinal_damping_time=14955,
            natural_energy_spread=1e-3,
            total_energy=20e9,
            random_generator=rng,
            disable_quantum_excitation=False,
        )

        expected_energy_kick = (
            -13e6
            - 2.0 / 14955 * 20e9 * np.ones(1000)
            + 2.0
            * 1e-3
            / np.sqrt(14955)
            * 20e9
            * rng.standard_normal(size=1000)
        )
        np.testing.assert_allclose(
            copy_to_cpu(energy_kick),
            copy_to_cpu(expected_energy_kick),
            rtol=1,
        )


class TestSynchrotronRadiationBaseClassSchedulableRadiationIntegrals(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.number_of_turns = 100
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
            share_of_radiation_integrals=self.radiation_integrals,
            disable_quantum_excitation=True,
        )
        self.SRB.schedule(
            attribute="share_of_radiation_integrals",
            value=np.array(
                [
                    self.radiation_integrals * 1 / (k + 1)
                    for k in range(self.number_of_turns)
                ]
            ),
        )

        # To test the tracking methods
        self.simulation = Mock(Simulation)
        self.simulation.turn_i = Mock(DynamicParameter)
        self.simulation.turn_i.value = 0

        self.beam = BeamBaseClassTester(
            intensity=1e12,
            particle_type=electron,
            is_counter_rotating=False,
            is_distributed=False,
        )

        self.decimal = 6 if backend.float == np.float32 else 9

        self.U0, self.tau_z, self.sigma0 = (
            gather_longitudinal_synchrotron_radiation_parameters(
                particle_type=self.beam.particle_type,
                energy=self.beam.reference.total_energy,
                radiation_integrals=self.radiation_integrals,
            )
        )

        self.seed = 500

    def test_inputs_SynchrotronRadiationBaseClass(self):
        self.assertTrue(self.SRB.schedule_active)
        self.assertTrue(
            "share_of_radiation_integrals" in self.SRB.intended_for_scheduling
        )
        for k in range(self.number_of_turns):
            self.SRB.apply_schedules(
                turn_i=k,
                reference_time=float(self.beam.reference.time),
            )
            np.testing.assert_array_almost_equal(
                self.SRB.share_of_radiation_integrals,
                1 / (k + 1) * self.radiation_integrals,
                decimal=self.decimal,
            )
        self.assertIsNone(self.SRB._energy_lost_due_to_synchrotron_radiation)
        self.assertIsNone(self.SRB._damping_time)
        self.assertIsNone(self.SRB._natural_energy_spread)

    def test_calculate_kick_SynchrotronRadiationBaseClass(self):
        for k in range(self.number_of_turns):
            self.SRB.apply_schedules(
                turn_i=k,
                reference_time=float(self.beam.reference.time),
            )
            np.random.seed(seed=self.seed)
            _ = self.SRB._calculate_kick(
                beam=self.beam,
            )

            # SynchrotronRadiationBaseClass
            self.assertAlmostEqual(
                self.SRB._energy_lost_due_to_synchrotron_radiation,
                np.float64(1337317.6297928384) * 1 / (k + 1),
                places=self.decimal,
            )
            self.assertAlmostEqual(
                self.SRB._damping_time,
                np.float64(14955.235530506275) * (k + 1),
                places=self.decimal,
            )
            self.assertAlmostEqual(
                self.SRB._natural_energy_spread,
                np.float64(0.0001675968578478592),
                places=self.decimal,
            )

    def test_tracking_updates_SRI(self):
        turn_to_consider = 9
        self.SRB.on_init_simulation(simulation=self.simulation)
        self.simulation.turn_i.value = turn_to_consider
        self.SRB.track(beam=self.beam)
        np.testing.assert_array_almost_equal(
            self.SRB.share_of_radiation_integrals,
            1 / (turn_to_consider + 1) * self.radiation_integrals,
            decimal=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRB._energy_lost_due_to_synchrotron_radiation,
            np.float64(1337317.6297928384) * 1 / (turn_to_consider + 1),
            places=self.decimal,
        )

        self.assertAlmostEqual(
            self.SRB._damping_time,
            np.float64(14955.235530506275) * (turn_to_consider + 1),
            places=self.decimal,
        )
        self.assertAlmostEqual(
            self.SRB._natural_energy_spread,
            np.float64(0.0001675968578478592),
            places=self.decimal,
        )
