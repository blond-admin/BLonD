from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

from blond import Simulation, mu_plus, proton
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.flags import BeamFlags
from blond.core.beam.particle_types import ParticleType, mu_minus
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.distributed.distributed_array import DistributedArray
from blond.generals.distributed.helpers import (
    MPI_RANK,
    mpi_barrier,
    mpi_is_distributed,
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

    @property
    def common_array_size(self) -> int:
        return self._dt.global_size

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

    def test_counter_rotating_charge(self):
        beam_base_class_corot = BeamBaseClassTester(
            intensity=1, particle_type=mu_plus, is_counter_rotating=False
        )
        beam_base_class_counterrotating = BeamBaseClassTester(
            intensity=1, particle_type=mu_minus, is_counter_rotating=True
        )
        assert (
            beam_base_class_corot.signed_charge_with_direction()
            == beam_base_class_counterrotating.signed_charge_with_direction()
        )

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

    def test_add_coordinates(self):
        dt_1 = backend.linspace(0, 1e-6, 10, dtype=backend.float)
        dE_1 = backend.linspace(-1e6, 0, 10, dtype=backend.float)
        flags_1 = backend.zeros_like(dE_1, dtype=np.int32)
        ids_1 = backend.arange(len(dE_1), dtype=np.int32)

        dt_2 = backend.linspace(1e-6, 2e-6, 10, dtype=backend.float)
        dE_2 = backend.linspace(0, 1e6, 10, dtype=backend.float)
        flags_2 = backend.zeros_like(dE_2, dtype=np.int32)
        ids_2 = backend.arange(len(dE_2), dtype=np.int32)

        beam_1 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_1._dt = DistributedArray(dt_1)
        beam_1._dE = DistributedArray(dE_1)
        beam_1._flags = DistributedArray(flags_1)
        beam_1._ids = DistributedArray(ids_1)

        dist_dt = DistributedArray(dt_2)
        dist_dE = DistributedArray(dE_2)
        dist_flags = DistributedArray(flags_2)
        dist_ids = DistributedArray(ids_2)

        beam_1._add_coordinates(dist_dt, dist_dE, dist_flags, dist_ids)

        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dt.array_local),
            copy_to_cpu(np.concatenate((dt_1, dt_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dE.array_local),
            copy_to_cpu(np.concatenate((dE_1, dE_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._flags.array_local),
            copy_to_cpu(np.concatenate((flags_1, flags_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._ids.array_local),
            copy_to_cpu(np.concatenate((ids_1, ids_2))),
        )

        self.assertEqual(beam_1.intensity, 2e12)

    def test_add_particles(self):
        dt_1 = backend.linspace(0, 1e-6, 10, dtype=backend.float)
        dE_1 = backend.linspace(-1e6, 0, 10, dtype=backend.float)
        flags_1 = backend.zeros_like(dE_1, dtype=np.int32)
        ids_1 = backend.arange(len(dE_1), dtype=np.int32)

        dt_2 = backend.linspace(1e-6, 2e-6, 10, dtype=backend.float)
        dE_2 = backend.linspace(0, 1e6, 10, dtype=backend.float)
        flags_2 = (
            backend.zeros_like(dE_2, dtype=np.int32) + BeamFlags.ACTIVE.value
        )
        ids_2 = backend.arange(len(dE_2), dtype=np.int32) + len(dt_1)

        beam_1 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_1._dt = DistributedArray(dt_1)
        beam_1._dE = DistributedArray(dE_1)
        beam_1._flags = DistributedArray(flags_1)
        beam_1._ids = DistributedArray(ids_1)

        dist_dt = DistributedArray(dt_2)
        dist_dE = DistributedArray(dE_2)

        beam_1.add_particles(dist_dt, dist_dE)

        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dt.array_local),
            copy_to_cpu(np.concatenate((dt_1, dt_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dE.array_local),
            copy_to_cpu(np.concatenate((dE_1, dE_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._flags.array_local),
            copy_to_cpu(np.concatenate((flags_1, flags_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._ids.array_local),
            copy_to_cpu(np.concatenate((ids_1, ids_2))),
        )

        self.assertEqual(beam_1.intensity, 2e12)

        dist_dt = DistributedArray(dt_2[1:])
        dist_dE = DistributedArray(dE_2)

        with self.assertRaisesRegex(ValueError, "The dt and dE array sizes"):
            beam_1.add_particles(dist_dt, dist_dE)

    def test_add_beam_valid(self):
        dt_1 = backend.linspace(0, 1e-6, 10, dtype=backend.float)
        dE_1 = backend.linspace(-1e6, 0, 10, dtype=backend.float)
        flags_1 = backend.zeros_like(dE_1, dtype=np.int32)
        ids_1 = backend.arange(len(dE_1), dtype=np.int32)

        dt_2 = backend.linspace(1e-6, 2e-6, 10, dtype=backend.float)
        dE_2 = backend.linspace(0, 1e6, 10, dtype=backend.float)
        flags_2 = backend.zeros_like(dE_2, dtype=np.int32)
        ids_2 = backend.arange(len(dE_2), dtype=np.int32)

        beam_1 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_2 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_1._dt = DistributedArray(dt_1)
        beam_1._dE = DistributedArray(dE_1)
        beam_1._flags = DistributedArray(flags_1)
        beam_1._ids = DistributedArray(ids_1)

        beam_2._dt = DistributedArray(dt_2)
        beam_2._dE = DistributedArray(dE_2)
        beam_2._flags = DistributedArray(flags_2)
        beam_2._ids = DistributedArray(ids_2)

        beam_1.add_beam(beam_2)

        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dt.array_local),
            copy_to_cpu(np.concatenate((dt_1, dt_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dE.array_local),
            copy_to_cpu(np.concatenate((dE_1, dE_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._flags.array_local),
            copy_to_cpu(np.concatenate((flags_1, flags_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._ids.array_local),
            copy_to_cpu(np.arange(2 * len(dt_1), dtype=np.int32)),
        )

        self.assertEqual(beam_1.intensity, 2 * beam_2.intensity)
        self.assertEqual(beam_1.ratio, beam_2.ratio)
        self.assertEqual(
            beam_1.common_array_size, 2 * beam_2.common_array_size
        )

    def test_add_beam_errors(self):
        dt_1 = backend.linspace(0, 1e-6, 10, dtype=backend.float)
        dE_1 = backend.linspace(-1e6, 0, 10, dtype=backend.float)
        flags_1 = backend.zeros_like(dE_1, dtype=np.int32)
        ids_1 = backend.arange(len(dE_1), dtype=np.int32)

        dt_2 = backend.linspace(1e-6, 2e-6, 10, dtype=backend.float)
        dE_2 = backend.linspace(0, 1e6, 10, dtype=backend.float)
        flags_2 = backend.zeros_like(dE_2, dtype=np.int32)
        ids_2 = backend.arange(len(dE_2), dtype=np.int32)

        beam_1 = BeamBaseClassTester(
            intensity=20,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_2 = BeamBaseClassTester(
            intensity=10,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_1._dt = DistributedArray(dt_1)
        beam_1._dE = DistributedArray(dE_1)
        beam_1._flags = DistributedArray(flags_1)
        beam_1._ids = DistributedArray(ids_1)

        beam_2._dt = DistributedArray(dt_2)
        beam_2._dE = DistributedArray(dE_2)
        beam_2._flags = DistributedArray(flags_2)
        beam_2._ids = DistributedArray(ids_2)

        beam_1._is_distributed = True
        with self.assertRaisesRegex(
            RuntimeError, "A non-distributed beam cannot"
        ):
            beam_1.add_beam(beam_2)

        beam_1._is_distributed = False
        with self.assertRaisesRegex(ValueError, "Beams can only be added"):
            beam_1.add_beam(beam_2)

        beam_2.intensity = 20
        beam_2.reference._particle_type = mu_plus
        with self.assertRaisesRegex(
            ValueError, "Cannot add beams with mismatched"
        ):
            beam_1.add_beam(beam_2)

    def test_iadd(self):
        dt_1 = backend.linspace(0, 1e-6, 10, dtype=backend.float)
        dE_1 = backend.linspace(-1e6, 0, 10, dtype=backend.float)
        flags_1 = backend.zeros_like(dE_1, dtype=np.int32)
        ids_1 = backend.arange(len(dE_1), dtype=np.int32)

        dt_2 = backend.linspace(1e-6, 2e-6, 10, dtype=backend.float)
        dE_2 = backend.linspace(0, 1e6, 10, dtype=backend.float)
        flags_2 = backend.zeros_like(dE_2, dtype=np.int32)
        ids_2 = backend.arange(len(dE_2), dtype=np.int32)

        beam_1 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_2 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_1._dt = DistributedArray(dt_1)
        beam_1._dE = DistributedArray(dE_1)
        beam_1._flags = DistributedArray(flags_1)
        beam_1._ids = DistributedArray(ids_1)

        beam_2._dt = DistributedArray(dt_2)
        beam_2._dE = DistributedArray(dE_2)
        beam_2._flags = DistributedArray(flags_2)
        beam_2._ids = DistributedArray(ids_2)

        beam_1 += beam_2

        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dt.array_local),
            copy_to_cpu(np.concatenate((dt_1, dt_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._dE.array_local),
            copy_to_cpu(np.concatenate((dE_1, dE_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._flags.array_local),
            copy_to_cpu(np.concatenate((flags_1, flags_2))),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam_1._ids.array_local),
            copy_to_cpu(np.arange(2 * len(dt_1), dtype=np.int32)),
        )

        self.assertEqual(beam_1.intensity, 2 * beam_2.intensity)
        self.assertEqual(beam_1.ratio, beam_2.ratio)
        self.assertEqual(
            beam_1.common_array_size, 2 * beam_2.common_array_size
        )

    @pytest.mark.mpi
    def test_addition_mpi(self):
        mpi_active = mpi_is_distributed()

        if not mpi_active:
            return

        beam_1 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        beam_2 = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )

        dt_1 = DistributedArray(None)
        dE_1 = DistributedArray(None)
        dt_2 = DistributedArray(None)
        dE_2 = DistributedArray(None)
        flags = DistributedArray(None)
        ids = DistributedArray(None)

        if MPI_RANK == 0:
            dt_1.array_local = backend.array([0, 1, 2], dtype=backend.float)
            dE_1.array_local = backend.array([10, 11, 12], dtype=backend.float)
            dt_2.array_local = backend.array([3, 4, 5], dtype=backend.float)
            dE_2.array_local = backend.array([13, 14, 15], dtype=backend.float)
            flags.array_local = backend.zeros_like(
                dE_1.array_local, dtype=np.int32
            )
            ids.array_local = backend.array([1, 3, 5], dtype=np.int32)

        dt_1.mpi_scatter()
        dE_1.mpi_scatter()
        dt_2.mpi_scatter()
        dE_2.mpi_scatter()
        flags.mpi_scatter()
        ids.mpi_scatter()

        mpi_barrier()

        beam_1._dt = dt_1
        beam_1._dE = dE_1
        beam_1._flags = flags
        beam_1._ids = ids

        beam_2._dt = dt_2
        beam_2._dE = dE_2
        beam_2._flags = flags
        beam_2._ids = ids

        beam_1 += beam_2

        ids = beam_1.ids.mpi_gather()
        dt = beam_1.dt.mpi_gather()
        dE = beam_1.dE.mpi_gather()

        if MPI_RANK != 0:
            return

        gathered_ids = copy_to_cpu(ids)
        gathered_dts = copy_to_cpu(dt)
        gathered_dEs = copy_to_cpu(dE)

        particle_order = np.argsort(gathered_ids)

        np.testing.assert_array_equal(
            gathered_ids[particle_order], np.array([1, 3, 5, 7, 9, 11])
        )
        np.testing.assert_array_equal(
            gathered_dts[particle_order], np.array([0, 1, 2, 3, 4, 5])
        )
        np.testing.assert_array_equal(
            gathered_dEs[particle_order], np.array([10, 11, 12, 13, 14, 15])
        )


class TestSortByDt(unittest.TestCase):
    def _beam(self, dt, dE, ids, flags, is_distributed=False):
        beam = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_distributed=is_distributed,
        )
        beam._dt = DistributedArray(backend.array(dt, dtype=backend.float))
        beam._dE = DistributedArray(backend.array(dE, dtype=backend.float))
        beam._ids = DistributedArray(backend.array(ids, dtype=np.int32))
        beam._flags = DistributedArray(backend.array(flags, dtype=np.int32))
        return beam

    def test_permutes_all_arrays_consistently(self):
        # dt order: argsort([3,1,2,5,4]) -> [1,2,0,4,3]
        beam = self._beam(
            dt=[3.0, 1.0, 2.0, 5.0, 4.0],
            dE=[30.0, 10.0, 20.0, 50.0, 40.0],
            ids=[0, 1, 2, 3, 4],
            flags=[10, 11, 12, 13, 14],
        )
        beam.sort_by_dt()

        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_dt()), [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        # dE must follow dt
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_dE()),
            [10.0, 20.0, 30.0, 40.0, 50.0],
        )
        # ids/flags must follow the *same* permutation [1,2,0,4,3]
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_ids()), [1, 2, 0, 4, 3]
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_flags()), [11, 12, 10, 14, 13]
        )

    def test_already_sorted_is_idempotent(self):
        beam = self._beam(
            dt=[1.0, 2.0, 3.0],
            dE=[10.0, 20.0, 30.0],
            ids=[0, 1, 2],
            flags=[0, 0, 0],
        )
        beam.sort_by_dt()
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_dt()), [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_ids()), [0, 1, 2]
        )

    def test_reverse_sorted(self):
        beam = self._beam(
            dt=[3.0, 2.0, 1.0],
            dE=[30.0, 20.0, 10.0],
            ids=[0, 1, 2],
            flags=[0, 1, 2],
        )
        beam.sort_by_dt()
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_dt()), [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_ids()), [2, 1, 0]
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_dE()), [10.0, 20.0, 30.0]
        )

    def test_single_particle(self):
        beam = self._beam(dt=[7.0], dE=[1.0], ids=[3], flags=[1])
        beam.sort_by_dt()
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_dt()), [7.0]
        )
        np.testing.assert_array_equal(
            copy_to_cpu(beam.read_partial_ids()), [3]
        )

    def test_preserves_dtype(self):
        beam = self._beam(
            dt=[2.0, 1.0],
            dE=[20.0, 10.0],
            ids=[0, 1],
            flags=[0, 1],
        )
        beam.sort_by_dt()
        self.assertEqual(beam.read_partial_dt().dtype, backend.float)
        self.assertEqual(beam.read_partial_dE().dtype, backend.float)
        self.assertEqual(beam.read_partial_ids().dtype, np.int32)
        self.assertEqual(beam.read_partial_flags().dtype, np.int32)

    def test_raises_when_distributed(self):
        beam = self._beam(
            dt=[2.0, 1.0],
            dE=[20.0, 10.0],
            ids=[0, 1],
            flags=[0, 1],
            is_distributed=True,
        )
        with self.assertRaises(NotImplementedError):
            beam.sort_by_dt()


if __name__ == "__main__":
    unittest.main()
