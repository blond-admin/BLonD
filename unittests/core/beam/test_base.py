from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from blond import Simulation, proton
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.distributed.distributed_array import DistributedArray

if TYPE_CHECKING:
    from typing import Literal

    from cupy.typing import NDArray as CupyArray
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
        self._dE = DistributedArray(
            backend.linspace(1, 10, 10, dtype=backend.float)
        )
        self._dt = DistributedArray(
            backend.linspace(20, 30, 10, dtype=backend.float)
        )
        self._flags = DistributedArray(backend.zeros(10, dtype=np.int32))
        self._ids = DistributedArray(backend.arange(10, dtype=np.int32))

    @property
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
    ):
        """Sets beam array attributes for simulation

        Parameters
        ----------
        dt
            Macro-particle time coordinates [s].
        dE
            Macro-particle energy coordinates [eV].
        flags
            Macro-particle flags.
        reference_time
            Time of the reference frame (global time), in [s].
        reference_total_energy
            Time of the reference frame (global total energy), in [eV].
        mpi_mode
            - "root-distributes": The array is distributed from the root node to all ranks.
            - "all-ranks":  All ranks setup the beam independently.
        **kwargs
            Unused - Keyword arguments to make the non-abstract implementation
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

    def rms_emittance(self):
        pass


class TestBeamBaseClass(unittest.TestCase):
    def setUp(self):
        self.beam_base_class = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip("Abstract method")
    def test_common_array_size(self):
        pass  # is abstract

    @unittest.skip("Abstract method")
    def test_dE_min(self):
        pass  # is abstract

    @unittest.skip("Abstract method")
    def test_dE_min(self):
        pass  # is abstract

    @unittest.skip("Abstract method")
    def test_dt_max(self):
        pass  # is abstract

    @unittest.skip("Abstract method")
    def test_dt_min(self):
        pass  # is abstract

    def test_is_counter_rotating(self):
        self.assertEqual(self.beam_base_class.is_counter_rotating, False)

    def test_is_distributed(self):
        self.assertEqual(self.beam_base_class.is_distributed, False)

    def test_n_macroparticles_partial(self):
        self.assertEqual(10, self.beam_base_class.n_macroparticles_partial())

    def test_on_init_simulation(self):
        simulation = Mock(spec=Simulation)
        self.beam_base_class.on_init_simulation(simulation=simulation)

    def test_missing_init_of_simulation(self):
        self.beam_base_class._dE = None
        with self.assertRaises(AttributeError):
            self.beam_base_class.n_macroparticles_partial()

    def test_on_run_simulation(self):
        simulation = Mock(spec=Simulation)
        beam = Mock(spec=BeamBaseClass)
        simulation.magnetic_cycle.get_target_total_energy.return_value = 11
        self.beam_base_class.on_run_simulation(
            simulation=simulation,
            n_turns=1,
            beam=beam,
        )

    def test_read_partial_dE(self):
        self.assertTrue(
            isinstance(self.beam_base_class.read_partial_dE(), backend.ndarray)
        )

    def test_read_partial_dt(self):
        self.assertTrue(
            isinstance(self.beam_base_class.read_partial_dt(), backend.ndarray)
        )

    def test_read_partial_ids(self):
        self.assertTrue(
            isinstance(
                self.beam_base_class.read_partial_ids(), backend.ndarray
            )
        )

    @unittest.skip("Abstract method")
    def test_setup_beam(self):
        self.beam_base_class.setup_beam(dt=None, dE=None, flags=None)

    def test_write_partial_dE(self):
        self.assertTrue(
            isinstance(
                self.beam_base_class.write_partial_dE(), backend.ndarray
            )
        )

    def test_write_partial_dt(self):
        self.assertTrue(
            isinstance(
                self.beam_base_class.write_partial_dt(), backend.ndarray
            )
        )

    def test_write_partial_flags(self):
        self.assertTrue(
            isinstance(
                self.beam_base_class.write_partial_flags(), backend.ndarray
            )
        )

    def test_purge_flagged_entries(self):
        ids_before = copy_to_cpu(self.beam_base_class._ids.array_local.copy())
        select = [0, 1, -1]

        self.beam_base_class._flags.array_local[select] = -500
        self.beam_base_class.purge_flagged_entries()
        self.assertTrue(
            backend.all(self.beam_base_class._flags.array_local != -500)
        )

        mask = np.ones(len(ids_before), dtype=bool)
        mask[select] = False
        ids_after = copy_to_cpu(self.beam_base_class._ids.array_local)
        np.testing.assert_equal(np.sort(ids_before[mask]), np.sort(ids_after))


if __name__ == "__main__":
    unittest.main()
