import unittest
from unittest import mock

import numpy as np
import pytest

from blond import (
    Beam,
    backend,
    copy_to_cpu,
    make_multibunch_beam,
    uranium_29,
)
from blond.beam_preparation.helpers import populate_beam
from blond.testing.backend_testing import BLonDTestCase


class TestCallables(BLonDTestCase):
    def test_make_multibunch_beam_fails(self):
        beam = Beam(
            intensity=1, particle_type=uranium_29, is_counter_rotating=False
        )
        with self.assertRaisesRegex(AssertionError, "Please set up beam"):
            make_multibunch_beam(
                beam=beam,
                n_times=3,
                t_distance=222,
                common_offset=111,
            )

    def test_make_multibunch_beam_executes(self):
        beam = Beam(
            intensity=1, particle_type=uranium_29, is_counter_rotating=False
        )
        beam.setup_beam(dt=[1, 2, 3], dE=[1e3, 2e3, 3e3])
        beam = make_multibunch_beam(
            beam=beam,
            n_times=3,
            t_distance=222,
            common_offset=111,
        )

        # Particles are grouped bunch by bunch, so that neighbouring
        # entries share a narrow ``dt`` range (cache-friendly histogramming)
        np.testing.assert_allclose(
            beam.dt.copy_as_numpy(),
            [
                112.0,  # dt[0] + common_offset
                113.0,  # dt[1] + common_offset
                114.0,  # dt[2] + common_offset
                334.0,  # dt[0] + common_offset + t_distance
                335.0,
                336.0,
                556.0,  # dt[0] + common_offset + 2 * t_distance
                557.0,
                558.0,
            ],
        )
        np.testing.assert_allclose(
            beam.dE.copy_as_numpy(),
            [
                1000.0,  # dE[0]
                2000.0,  # dE[1]
                3000.0,  # dE[2]
                1000.0,
                2000.0,
                3000.0,
                1000.0,
                2000.0,
                3000.0,
            ],
        )
        self.assertEqual(beam.particle_type, uranium_29)
        self.assertEqual(beam.is_counter_rotating, False)


class TestHelpersHandOverArrays(BLonDTestCase):
    def _setup_beam_spy(self):
        return mock.patch.object(
            Beam, "setup_beam", autospec=True, side_effect=Beam.setup_beam
        )

    def test_make_multibunch_beam_does_not_copy(self):
        beam = Beam(intensity=1, particle_type=uranium_29)
        beam.setup_beam(dt=[1, 2, 3], dE=[1e3, 2e3, 3e3])
        with self._setup_beam_spy() as setup_beam:
            make_multibunch_beam(
                beam=beam, n_times=2, t_distance=10, common_offset=0
            )
        self.assertIs(setup_beam.call_args.kwargs["copy"], False)

    def test_populate_beam_does_not_copy(self):
        beam = Beam(intensity=1, particle_type=uranium_29)
        time_grid, deltaE_grid = np.meshgrid(
            np.linspace(0, 1, 4), np.linspace(-1, 1, 5)
        )
        with self._setup_beam_spy() as setup_beam:
            populate_beam(
                beam=beam,
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                density_grid=np.ones_like(time_grid),
                n_macroparticles=100,
                seed=0,
            )
        self.assertIs(setup_beam.call_args.kwargs["copy"], False)


class TestPopulateBeamBackendDependentRandomNumbers(BLonDTestCase):
    def _dE(self, **kwargs):
        beam = Beam(intensity=1, particle_type=uranium_29)
        time_grid, deltaE_grid = np.meshgrid(
            np.linspace(0, 1, 4), np.linspace(-1, 1, 5)
        )
        populate_beam(
            beam=beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            density_grid=np.exp(-(time_grid**2) - deltaE_grid**2),
            n_macroparticles=1000,
            seed=6,
            **kwargs,
        )
        return copy_to_cpu(beam.dE.array_local)

    def test_change_nothing_on_cpu(self):
        if backend.is_gpu:
            self.skipTest("Only on a CPU backend")
        np.testing.assert_array_equal(
            self._dE(backend_dependent_random_numbers=True), self._dE()
        )


@pytest.mark.cupy
class TestPopulateBeamOnGpu(BLonDTestCase):
    def setUp(self):
        try:
            import cupy  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        from blond.core.backends.backend import Cupy64Bit

        original_backend = type(backend)
        backend.change_backend(Cupy64Bit)
        self.addCleanup(backend.change_backend, original_backend)

    def test_backend_dependent_random_numbers_draw_on_the_gpu(self):
        import cupy as cp

        beam = Beam(intensity=1, particle_type=uranium_29)
        time_grid, deltaE_grid = np.meshgrid(
            np.linspace(0, 1, 4), np.linspace(-1, 1, 5)
        )
        populate_beam(
            beam=beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            density_grid=np.exp(-(time_grid**2) - deltaE_grid**2),
            n_macroparticles=1000,
            seed=6,
            backend_dependent_random_numbers=True,
        )
        self.assertIsInstance(beam.dt.array_local, cp.ndarray)
        self.assertEqual(beam.dt.global_size, 1000)


if __name__ == "__main__":
    unittest.main()
