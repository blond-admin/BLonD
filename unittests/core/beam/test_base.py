from __future__ import annotations

import copy
import unittest
from functools import cached_property
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from blond import Simulation, proton
from blond.core.backends.backend import backend
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType

if TYPE_CHECKING:
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
        self._dt = dt
        self._dE = dE
        self._ids = np.arange(len(dt), dtype=np.int32)
        self._flags = flags
        self.reference_time = reference_time
        self.reference_total_energy = reference_total_energy

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
        return len(self._dt)


class TestBeamBaseClass(unittest.TestCase):
    def setUp(self):
        self.beam_base_class = BeamBaseClassTester(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
            is_distributed=False,
        )
        self.beam_base_class.setup_beam(
            np.linspace(1, 10, 10, dtype=backend.float),
            np.linspace(20, 30, 10, dtype=backend.float),
            np.zeros(10, dtype=np.int32),
            np.arange(10, dtype=np.int32),
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

    def test_invalidate_cache(self):
        self.beam_base_class.invalidate_cache()

    def test_invalidate_cache_dE(self):
        self.beam_base_class.invalidate_cache_dE()

    def test_invalidate_cache_dt(self):
        self.beam_base_class.invalidate_cache_dt()

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
            turn_i_init=1,
            beam=beam,
        )

    def test_read_partial_dE(self):
        self.assertTrue(
            isinstance(self.beam_base_class.read_partial_dE(), np.ndarray)
        )

    def test_read_partial_dt(self):
        self.assertTrue(
            isinstance(self.beam_base_class.read_partial_dt(), np.ndarray)
        )

    def test_read_partial_ids(self):
        self.assertTrue(
            isinstance(self.beam_base_class.read_partial_ids(), np.ndarray)
        )

    @unittest.skip("Abstract method")
    def test_setup_beam(self):
        self.beam_base_class.setup_beam(dt=None, dE=None, flags=None)

    def test_write_partial_dE(self):
        self.assertTrue(
            isinstance(self.beam_base_class.write_partial_dE(), np.ndarray)
        )

    def test_write_partial_dt(self):
        self.assertTrue(
            isinstance(self.beam_base_class.write_partial_dt(), np.ndarray)
        )

    def test_write_partial_flags(self):
        self.assertTrue(
            isinstance(self.beam_base_class.write_partial_flags(), np.ndarray)
        )

    def test_purge_flagged_entries(self):
        ids_before = self.beam_base_class._ids.copy()
        select = [0, 1, -1]

        self.beam_base_class._flags[select] = -500
        self.beam_base_class.purge_flagged_entries()
        self.assertTrue(np.all(self.beam_base_class._flags != -500))

        mask = np.ones(len(ids_before), dtype=bool)
        mask[select] = False
        ids_after = self.beam_base_class._ids
        np.testing.assert_equal(np.sort(ids_before[mask]), np.sort(ids_after))

    def test_iadd_beam(self):
        intens = self.beam_base_class.intensity
        id_max = np.max(self.beam_base_class._ids)

        self.beam_base_class += self.beam_base_class

        self.assertEqual(self.beam_base_class.intensity, 2 * intens)
        self.assertEqual(self.beam_base_class._ids[-1], id_max * 2 + 1)
        self.assertEqual(
            len(self.beam_base_class._dt), len(self.beam_base_class._ids)
        )

    def test_iadd_particles(self):
        intens = self.beam_base_class.intensity
        id_max = np.max(self.beam_base_class._ids)

        new_dt = np.linspace(-10, -1, 10)
        new_dE = np.linspace(-10, -1, 10)

        self.beam_base_class += (new_dt, new_dE)

        self.assertEqual(self.beam_base_class.intensity, 2 * intens)
        self.assertEqual(self.beam_base_class._ids[-1], id_max * 2 + 1)
        self.assertEqual(
            len(self.beam_base_class._dt), len(self.beam_base_class._ids)
        )

        np.testing.assert_array_equal(self.beam_base_class._dt[10:], new_dt)
        np.testing.assert_array_equal(self.beam_base_class._dE[10:], new_dE)

    def test_iadd_exceptions(self):
        other_beam = BeamBaseClassTester(
            intensity=1e1,
            particle_type=copy.deepcopy(proton),
            is_counter_rotating=False,
            is_distributed=False,
        )
        other_beam.setup_beam(
            np.linspace(1, 10, 10, dtype=backend.float),
            np.linspace(20, 30, 10, dtype=backend.float),
            np.zeros(10, dtype=np.int32),
            np.arange(10, dtype=np.int32),
        )

        with self.assertRaises(ValueError):
            self.beam_base_class += other_beam

        other_beam.intensity = int(1e12)
        object.__setattr__(other_beam._particle_type, "charge", 2)

        with self.assertRaises(TypeError):
            self.beam_base_class += other_beam

        with self.assertRaises(ValueError):
            self.beam_base_class += (other_beam._dt[:5], other_beam._dE)


if __name__ == "__main__":
    unittest.main()
