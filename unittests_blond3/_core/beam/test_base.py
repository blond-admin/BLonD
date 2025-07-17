from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from cupy.typing import NDArray as CupyArray
from numpy._typing import NDArray as NumpyArray

from blond3 import proton, Simulation
from blond3._core.backends.backend import backend
from blond3._core.beam.base import BeamBaseClass
from blond3._core.beam.particle_types import ParticleType

if TYPE_CHECKING:
    from typing import Optional


class BeamBaseClassTester(BeamBaseClass):
    def __init__(
        self,
        n_particles: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed=False,
    ):
        super().__init__(
            n_particles=n_particles,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
            is_distributed=is_distributed,
        )
        self._dE = np.linspace(1, 10, 10)
        self._dt = np.linspace(20, 30, 10)
        self._flags = np.zeros(10, dtype=int)

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
        reference_time: Optional[float] = None,
        reference_total_energy: Optional[float] = None,
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

    def dE_max(self) -> backend.float:
        pass

    def dt_min(self) -> backend.float:
        pass

    def dt_max(self) -> backend.float:
        pass

    def dE_min(self) -> backend.float:
        pass

    def common_array_size(self) -> int:
        pass


class TestBeamBaseClass(unittest.TestCase):
    def setUp(self):
        self.beam_base_class = BeamBaseClassTester(
            n_particles=1e12,
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

    @unittest.skip("Abstract method")
    def test_plot_hist2d(self):
        # TODO: implement test for `plot_hist2d`
        self.beam_base_class.plot_hist2d()

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
        self.assertTrue(isinstance(self.beam_base_class.read_partial_dE(), np.ndarray))

    def test_read_partial_dt(self):
        self.assertTrue(isinstance(self.beam_base_class.read_partial_dt(), np.ndarray))

    @unittest.skip("Abstract method")
    def test_setup_beam(self):
        self.beam_base_class.setup_beam(dt=None, dE=None, flags=None)

    def test_write_partial_dE(self):
        self.assertTrue(isinstance(self.beam_base_class.write_partial_dE(), np.ndarray))

    def test_write_partial_dt(self):
        self.assertTrue(isinstance(self.beam_base_class.write_partial_dt(), np.ndarray))

    def test_write_partial_flags(self):
        self.assertTrue(
            isinstance(self.beam_base_class.write_partial_flags(), np.ndarray)
        )


if __name__ == "__main__":
    unittest.main()
