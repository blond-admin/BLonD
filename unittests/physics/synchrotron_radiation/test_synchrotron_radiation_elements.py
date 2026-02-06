from __future__ import annotations

import unittest
from functools import cached_property
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import c, e
from scipy.constants import speed_of_light as c0

from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType
from blond.generals.distributed.distributed_array import DistributedArray
from blond.physics.synchrotron_radiation.synchrotron_radiation_elements import (
    WigglerMagnet,
)

if TYPE_CHECKING:
    from typing import Literal

    from cupy.typing import NDArray as CupyArray
    from numpy.typing import NDArray as NumpyArray


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
            np.linspace(-1e6, 1e6, 10, dtype=backend.float)
        )  #
        # delta E
        # in eV
        self._dt = DistributedArray(
            np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)
        )  # delta t
        # in s
        self._flags = np.zeros(10, dtype=np.int32)
        self._ids = np.arange(10, dtype=np.int32)

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


class TestWigglerMagnet(unittest.TestCase):
    def setUp(self) -> None:
        self.wiggler_magnet = WigglerMagnet(
            wiggler_type="sinusoidal",
            number_of_wigglers=2,
            peak_field=1,
            pole_length=0.01,
            number_of_poles=50,
            section_index=0,
        )
        self.wiggler_magnet_none = WigglerMagnet(
            wiggler_type="",
            peak_field=1,
            pole_length=0.095,
            number_of_poles=43,
        )

    def test_inputs_and_properties(self):
        assert self.wiggler_magnet.section_index == 0
        assert self.wiggler_magnet._type == "sinusoidal"
        assert self.wiggler_magnet._number_of_wigglers == 2
        assert self.wiggler_magnet._peak_field == 1
        assert self.wiggler_magnet._pole_length == 0.01
        assert self.wiggler_magnet._number_of_poles == 50

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
