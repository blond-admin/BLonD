import unittest
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import Beam, Simulation, proton, uranium_29
from blond.core.beam.base import BeamBaseClass, BeamFlags
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.generals.distributed.distributed_array import DistributedArray
from blond.generals.distributed.helpers import (
    MPI_RANK,
    MPI_SIZE,
    mpi_is_distributed,
)


class TestBeam(unittest.TestCase):
    def setUp(self) -> None:
        self.beam = Beam(
            intensity=1e12, particle_type=proton, is_counter_rotating=False
        )
        self.beam.setup_beam(
            dE=np.linspace(1, 10, 10), dt=np.linspace(20, 30, 10)
        )

    def test_setup_beam(self) -> None:
        self.beam.setup_beam(
            dE=np.linspace(1, 10, 10),
            dt=np.linspace(20, 30, 10),
            reference_time=11,
            reference_total_energy=1e12,
        )
        self.assertEqual(self.beam.reference.time, 11.0)
        self.assertEqual(self.beam.reference.total_energy, 1e12)

    def test_setup_beam(self) -> None:
        with self.assertRaisesRegex(NameError, "Unknown"):
            self.beam.setup_beam(
                dE=np.linspace(1, 10, 10),
                dt=np.linspace(20, 30, 10),
                reference_time=11,
                reference_total_energy=1e12,
                mpi_mode="should fail",
            )

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_basic_getters(self):
        self.beam.dt  # NOQA
        self.beam.dE  # NOQA
        self.beam.flags  # NOQA
        self.beam.ids  # NOQA

        beam = Beam(intensity=123, particle_type=lead_82)
        with self.assertRaisesRegex(
            AttributeError, "not properly initialized"
        ):
            beam.dt  # NOQA
        with self.assertRaisesRegex(
            AttributeError, "not properly initialized"
        ):
            beam.dE  # NOQA
        with self.assertRaisesRegex(
            AttributeError, "not properly initialized"
        ):
            beam.ids  # NOQA
        with self.assertRaisesRegex(
            AttributeError, "not properly initialized"
        ):
            beam.flags  # NOQA

    def test_common_array_size(self) -> None:
        self.assertEqual(10, self.beam.common_array_size)

    def test_dE_min(self) -> None:
        self.assertEqual(1, self.beam.dE_min)

    def test_dE_max(self) -> None:
        self.assertEqual(10, self.beam.dE_max)

    def test_dt_min(self) -> None:
        self.assertEqual(20, self.beam.dt_min)

    def test_dt_max(self) -> None:
        self.assertEqual(30, self.beam.dt_max)

    def test_on_init_simulation(self) -> None:
        simulation = Mock(spec=Simulation)
        self.beam.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self) -> None:
        simulation = Mock(spec=Simulation)
        simulation.magnetic_cycle.get_target_total_energy.return_value = 11
        beam = Mock(spec=BeamBaseClass)

        self.beam.on_run_simulation(
            simulation=simulation,
            n_turns=10,
            beam=beam,
        )

    def test_plot_hist2d_fails(self) -> None:
        self.beam._dt = None
        self.beam._dE = None
        with self.assertRaises(ValueError):
            self.beam.plot_hist2d()

    def test_plot_hist2d_executes(self) -> None:
        self.beam.plot_hist2d()
        plt.gcf().clf()

    def test_plot_hist2d_executes_kwargs1(self) -> None:
        self.beam.plot_hist2d(cmap="viridis")

    def test_plot_hist2d_executes_kwargs2(self) -> None:
        self.beam.plot_hist2d(bins=12)

    def test_plot_hist(self):
        self.beam.plot_hist()
        plt.gcf().clf()

    @pytest.mark.cupy
    def test_plot_hist2d_executes_gpu(self) -> None:
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        beam = Mock(Beam)
        beam._dE = DistributedArray(cp.ones(10))
        beam._dt = DistributedArray(cp.ones(10))
        Beam.plot_hist2d(beam)
        plt.gcf().clf()

    def test_plot_scatter_raises(self) -> None:
        beam = Mock(Beam)
        beam._dE = None
        beam._dt = None
        with self.assertRaises(ValueError):
            Beam.plot_scatter(beam)

    def test_plot_scatter_executes_cpu(self) -> None:
        beam = Mock(Beam)
        beam._dE = DistributedArray(np.ones(10))
        beam._dt = DistributedArray(np.ones(10))
        Beam.plot_scatter(beam)
        plt.gcf().clf()

    @pytest.mark.cupy
    def test_plot_scatter_executes_gpu(self) -> None:
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        beam = Mock(Beam)
        beam._dE = DistributedArray(cp.ones(10))
        beam._dt = DistributedArray(cp.ones(10))
        Beam.plot_scatter(beam)
        plt.gcf().clf()

    def test_plot_hist_raises(self) -> None:
        beam = Mock(Beam)
        beam._dE = None
        beam._dt = None
        with self.assertRaises(ValueError):
            Beam.plot_hist(beam, axis=1)

    @pytest.mark.cupy
    def test_plot_hist_executes_gpu(self) -> None:
        try:
            import cupy as cp  # type: ignore
        except ModuleNotFoundError:
            self.skipTest("Cupy not available")
        beam = Mock(Beam)
        beam._dE = DistributedArray(cp.ones(10))
        beam._dt = DistributedArray(cp.ones(10))
        for axis in range(2):
            Beam.plot_hist(beam, axis=axis)
            plt.gcf().clf()
        with self.assertRaises(ValueError):
            Beam.plot_hist(beam, axis=10)

    def test_simple_gaussian(self):
        beam = Beam.simple_gaussian(
            n_macroparticles=10_000,
            intensity=1e10,
            particle_type=proton,
            dt_scale=1e-9,
            dE_scale=1e9,
            dE_offset=0.5e9,
            dt_offset=0.5e-9,
        )
        # places=1 because using random generator with low number of particles
        self.assertAlmostEqual(
            beam._dt.mean(), 5.194208272517843e-10, places=1
        )
        self.assertAlmostEqual(beam._dt.std(), 1.003710018454918e-09, places=1)
        self.assertAlmostEqual(
            beam._dE.mean() / 1e9, 510497638.7958076 / 1e9, places=1
        )
        self.assertAlmostEqual(
            beam._dE.std() / 1e9, 996310643.973366 / 1e9, places=1
        )

    def test_plot_hist_executes_kwargs(self) -> None:
        beam = Mock(Beam)
        beam._dE = DistributedArray(np.ones(10))
        beam._dt = DistributedArray(np.ones(10))
        Beam.plot_hist(beam, axis=0, bins=12)

    def test_plot_hist_executes_cpu(self) -> None:
        beam = Mock(Beam)
        beam._dE = DistributedArray(np.ones(10))
        beam._dt = DistributedArray(np.ones(10))
        for axis in range(2):
            Beam.plot_hist(beam, axis=axis)
            plt.gcf().clf()
        with self.assertRaises(ValueError):
            Beam.plot_hist(beam, axis=10)

    def test_setup_beam2(self) -> None:
        with self.assertRaises(AssertionError):
            self.beam.setup_beam(dE=np.ones(10), dt=np.ones(11))
        with self.assertRaises(AssertionError):
            self.beam.setup_beam(
                dE=np.ones(10), dt=np.ones(10), flags=np.ones(11)
            )

    @pytest.mark.mpi
    def test_setup_beam_mpi(self) -> None:
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.arange(12), dE=np.arange(12), mpi_mode="root-distributes"
        )
        if MPI_RANK == 0 and MPI_SIZE == 2:  # assume `mpirun -n 2`
            np.testing.assert_allclose(beam._dt.array_local, np.arange(0, 6))
            np.testing.assert_allclose(beam._dE.array_local, np.arange(0, 6))
            np.testing.assert_allclose(
                beam._flags.array_local, np.ones(6) * BeamFlags.ACTIVE.value
            )
            np.testing.assert_allclose(beam._ids.array_local, np.arange(0, 6))
        elif MPI_RANK == 1:
            np.testing.assert_allclose(beam._dt.array_local, np.arange(6, 12))
            np.testing.assert_allclose(beam._dE.array_local, np.arange(6, 12))
            np.testing.assert_allclose(
                beam._flags.array_local, np.ones(6) * BeamFlags.ACTIVE.value
            )
            np.testing.assert_allclose(beam._ids.array_local, np.arange(6, 12))

    @pytest.mark.mpi
    def test_plot_hist2d_warns(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.arange(12), dE=np.arange(12), mpi_mode="root-distributes"
        )
        with self.assertWarnsRegex(UserWarning, "Plotting MPI single node"):
            beam.plot_hist2d()

    @pytest.mark.mpi
    def test_plot_hist_warns(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.arange(12), dE=np.arange(12), mpi_mode="root-distributes"
        )
        with self.assertWarnsRegex(UserWarning, "Plotting MPI single node"):
            beam.plot_hist()

    @pytest.mark.mpi
    def test_plot_scatter_warns(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.arange(12), dE=np.arange(12), mpi_mode="root-distributes"
        )
        with self.assertWarnsRegex(UserWarning, "Plotting MPI single node"):
            beam.plot_scatter()


class TestProbeBunch(unittest.TestCase):
    def setUp(self) -> None:
        self.probe_bunch = ProbeBeam(particle_type=proton, dt=np.ones(10))

    def test___init__raises(self) -> None:
        with self.assertRaises(ValueError):
            self.probe_bunch = ProbeBeam(particle_type=proton)

    def test___init__raises2(self) -> None:
        with self.assertRaises(AssertionError):
            self.probe_bunch = ProbeBeam(
                particle_type=proton,
                dt=np.ones(10),
                dE=np.ones(11),
            )

    def test___init__raises3(self) -> None:
        with self.assertRaises(ValueError):
            self.probe_bunch = ProbeBeam(
                particle_type=proton,
                dt=None,
                dE=None,
            )

    def test___init__1(self) -> None:
        self.probe_bunch = ProbeBeam(particle_type=proton, dt=np.ones(10))

    def test___init__2(self) -> None:
        self.probe_bunch = ProbeBeam(particle_type=proton, dE=np.ones(10))

    def test___init__3(self) -> None:
        with self.assertRaises(ValueError):
            self.probe_bunch = ProbeBeam(particle_type=proton)


class TestWeightenedBeam(unittest.TestCase):
    @unittest.skip
    def setUp(self) -> None:
        # TODO: implement test for `__init__`
        self.weightened_beam = WeightenedBeam(
            intensity=None, particle_type=None
        )

    @unittest.skip
    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    @unittest.skip
    def test_from_beam(self) -> None:
        # TODO: implement test for `from_beam`
        self.weightened_beam.from_beam(beam=None)

    @unittest.skip
    def test_setup_beam(self) -> None:
        # TODO: implement test for `setup_beam`
        self.weightened_beam.setup_beam(
            dt=None, dE=None, flags=None, weights=None
        )


class TestBeamSaveLoad(unittest.TestCase):
    """JSON + ``.npz`` save/load round-trip tests for ``Beam``."""

    def _build_beam(self) -> Beam:
        beam = Beam(
            intensity=1e12,
            particle_type=proton,
            is_counter_rotating=False,
        )
        rng = np.random.default_rng(0)
        beam.setup_beam(
            dt=rng.normal(0, 1e-10, 64),
            dE=rng.normal(0, 1e6, 64),
            reference_time=0.0,
            reference_total_energy=450e9,
        )
        return beam

    def test_save_and_load_roundtrip(self):
        import os
        import tempfile

        original = self._build_beam()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "beam.npz")
            original.save(path)
            restored = Beam.load(path)

        self.assertEqual(original.intensity, restored.intensity)
        self.assertEqual(
            original.is_counter_rotating, restored.is_counter_rotating
        )
        self.assertEqual(
            original.reference.total_energy, restored.reference.total_energy
        )
        self.assertEqual(
            original.particle_type.mass, restored.particle_type.mass
        )
        self.assertEqual(
            original.particle_type.charge, restored.particle_type.charge
        )
        np.testing.assert_array_equal(
            original.read_partial_dt(), restored.read_partial_dt()
        )
        np.testing.assert_array_equal(
            original.read_partial_dE(), restored.read_partial_dE()
        )
        np.testing.assert_array_equal(
            original.read_partial_flags(), restored.read_partial_flags()
        )
        np.testing.assert_array_equal(
            original.read_partial_ids(), restored.read_partial_ids()
        )

    def test_load_rejects_mismatched_schema_version(self):
        """Loading a file with a wrong schema version should raise ValueError."""
        import json
        import os
        import tempfile

        original = self._build_beam()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "beam.npz")
            original.save(path)

            # Rewrite the file with a bumped schema version to simulate a
            # newer (or corrupted) BLonD format.
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                dt = archive["dt"]
                dE = archive["dE"]
                flags = archive["flags"]
                ids = archive["ids"]
            metadata["schema_version"] = 9999
            np.savez(
                path,
                metadata=np.array(json.dumps(metadata)),
                dt=dt,
                dE=dE,
                flags=flags,
                ids=ids,
            )

            with self.assertRaises(ValueError):
                Beam.load(path)


if __name__ == "__main__":
    unittest.main()
