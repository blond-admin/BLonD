import json
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import Beam, Simulation, backend, proton, uranium_29
from blond.core.backends.backend import NumpyBackend
from blond.core.beam.base import (
    _BEAM_SCHEMA_VERSION,
    BeamBaseClass,
    BeamFlags,
)
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.generals.distributed.distributed_array import DistributedArray
from blond.generals.distributed.helpers import (
    MPI_COMM_WORLD,
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

    def test_plot_scatter_with_explicit_ax(self) -> None:
        beam = Mock(Beam)
        beam._dE = DistributedArray(np.ones(10))
        beam._dt = DistributedArray(np.ones(10))
        fig, ax = plt.subplots()
        Beam.plot_scatter(beam, ax=ax)
        plt.close(fig)

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
            seed=1,
        )
        np.testing.assert_allclose(beam._dt.mean(), 4.890870988791548e-10)
        np.testing.assert_allclose(beam._dt.std(), 9.984960296298784e-10)
        np.testing.assert_allclose(beam._dE.mean(), 512871777.88223857)
        np.testing.assert_allclose(beam._dE.std(), 1005397644.8793652)

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
    def test_ratio_mpi_root_distributes(self) -> None:
        """Ratio must equal `intensity / global_n_macroparticles` on every rank."""
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        intensity = 1e12
        n_macroparticles = 12  # divisible by MPI_SIZE=2 to avoid truncation
        beam = Beam(intensity=intensity, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.arange(n_macroparticles, dtype=float),
            dE=np.arange(n_macroparticles, dtype=float),
            mpi_mode="root-distributes",
        )
        # Local chunks shrink with MPI_SIZE; global size stays the same.
        self.assertEqual(beam._dt.local_size, n_macroparticles // MPI_SIZE)
        self.assertEqual(beam.common_array_size, n_macroparticles)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            local_ratio = beam.ratio
        self.assertAlmostEqual(local_ratio, intensity / n_macroparticles)
        # Ratio must be identical on every rank (no silent divergence).
        all_ratios = MPI_COMM_WORLD.allgather(local_ratio)
        self.assertEqual(len(all_ratios), MPI_SIZE)
        for r in all_ratios:
            self.assertEqual(r, local_ratio)

    @pytest.mark.mpi
    def test_ratio_mpi_all_ranks(self) -> None:
        """`simple_gaussian` distributes locally; ratio must still be global."""
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        intensity = 1e10
        n_macroparticles = 1000  # divisible by MPI_SIZE=2
        beam = Beam.simple_gaussian(
            n_macroparticles=n_macroparticles,
            intensity=intensity,
            particle_type=proton,
            dt_scale=1e-9,
            dE_scale=1e9,
            seed=42,
        )
        # `all-ranks` mode: each rank holds only its local chunk.
        self.assertEqual(beam._dt.local_size, n_macroparticles // MPI_SIZE)
        self.assertEqual(beam.common_array_size, n_macroparticles)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            local_ratio = beam.ratio
        self.assertAlmostEqual(local_ratio, intensity / n_macroparticles)
        all_ratios = MPI_COMM_WORLD.allgather(local_ratio)
        for r in all_ratios:
            self.assertEqual(r, local_ratio)

    @pytest.mark.mpi
    def test_common_array_size_mpi(self) -> None:
        """`common_array_size` must report the *global* count, not the local one."""
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        n_macroparticles = 12
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        beam.setup_beam(
            dt=np.arange(n_macroparticles, dtype=float),
            dE=np.arange(n_macroparticles, dtype=float),
            mpi_mode="root-distributes",
        )
        sizes = MPI_COMM_WORLD.allgather(beam.common_array_size)
        for s in sizes:
            self.assertEqual(s, n_macroparticles)

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


def build_reference_beam() -> Beam:
    """Build a deterministic ``Beam`` (fixed RNG seed).

    Shared by the in-process round-trip tests, the committed golden-fixture
    generator, and the golden-fixture test so the on-disk fixture always
    corresponds to a beam the tests can reconstruct in memory.
    """
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


class TestBeamSaveLoad(unittest.TestCase):
    """JSON + ``.npz`` save/load round-trip tests for ``Beam``."""

    def _build_beam(self) -> Beam:
        return build_reference_beam()

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
            copy_to_cpu(original.read_partial_dt()),
            copy_to_cpu(restored.read_partial_dt()),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(original.read_partial_dE()),
            copy_to_cpu(restored.read_partial_dE()),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(original.read_partial_flags()),
            copy_to_cpu(restored.read_partial_flags()),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(original.read_partial_ids()),
            copy_to_cpu(restored.read_partial_ids()),
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


class TestBeamGoldenFixture(unittest.TestCase):
    """Guard the on-disk ``Beam`` format with a frozen, committed fixture.

    Unlike ``TestBeamSaveLoad`` (which saves and loads within a single process
    and therefore can only ever read a file the current code just wrote), this
    test loads a *static* ``golden_beam.npz`` that is committed to the repo and
    produced by ``generate_golden_beam_fixture.py``. That makes it a genuine
    backward-compatibility check on the save/load pipeline.

    The fixture forces a deliberate decision whenever the format changes. The
    test fails when:

    * ``_BEAM_SCHEMA_VERSION`` was bumped but the fixture was not regenerated
      -> ``Beam.load`` rejects the stale version. This is the intended forcing
      function: bumping the schema version is incomplete until the golden
      fixture is regenerated and committed.
    * the save/load pipeline changed incompatibly *without* a schema bump ->
      the committed file fails to load, or loads but no longer matches the
      reference beam. That is the signal that the schema version must be
      incremented (and the fixture regenerated).

    To fix a legitimate format change: bump ``_BEAM_SCHEMA_VERSION`` and run
    ``python tests/unittests/core/beam/fixtures/
    generate_golden_beam_fixture.py``, then commit the regenerated file.
    """

    FIXTURE_PATH = (
        Path(__file__).resolve().parent / "fixtures" / "golden_beam.npz"
    )

    def setUp(self):
        # The fixture is frozen at the default 64-bit NumPy backend; skip on
        # other backends rather than reporting a spurious mismatch.
        if (
            not isinstance(backend, NumpyBackend)
            or backend.float != np.float64
        ):
            self.skipTest(
                "Golden beam fixture is frozen at the 64-bit NumPy backend."
            )
        self.assertTrue(
            self.FIXTURE_PATH.exists(),
            f"Missing golden beam fixture {self.FIXTURE_PATH.name}; "
            "regenerate with ./fixtures/generate_golden_beam_fixture.py.",
        )

    def test_golden_fixture_schema_version_is_current(self):
        """The committed fixture must carry the current schema version.

        If this fails, ``_BEAM_SCHEMA_VERSION`` was changed without
        regenerating the fixture (or vice versa).
        """
        with np.load(self.FIXTURE_PATH, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        self.assertEqual(
            metadata.get("schema_version"),
            _BEAM_SCHEMA_VERSION,
            f"Golden beam fixture schema version "
            f"{metadata.get('schema_version')!r} != current "
            f"{_BEAM_SCHEMA_VERSION!r}. If you intentionally changed the "
            "on-disk beam format, regenerate the fixture with "
            "./fixtures/generate_golden_beam_fixture.py and commit it.",
        )

    def test_golden_fixture_loads_and_matches_reference(self):
        """The frozen fixture must still load into the reference beam.

        ``Beam.load`` enforces the schema version, so a stale fixture (after a
        bump) fails here loudly. A silently incompatible pipeline change
        instead surfaces as a load error or a mismatch against the in-memory
        reference beam.
        """
        restored = Beam.load(self.FIXTURE_PATH)
        expected = build_reference_beam()

        self.assertEqual(expected.intensity, restored.intensity)
        self.assertEqual(
            expected.is_counter_rotating, restored.is_counter_rotating
        )
        self.assertEqual(
            expected.reference.total_energy, restored.reference.total_energy
        )
        self.assertEqual(expected.reference.time, restored.reference.time)
        self.assertEqual(
            expected.particle_type.mass, restored.particle_type.mass
        )
        self.assertEqual(
            expected.particle_type.charge, restored.particle_type.charge
        )
        np.testing.assert_array_equal(
            copy_to_cpu(expected.read_partial_dt()),
            copy_to_cpu(restored.read_partial_dt()),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(expected.read_partial_dE()),
            copy_to_cpu(restored.read_partial_dE()),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(expected.read_partial_flags()),
            copy_to_cpu(restored.read_partial_flags()),
        )
        np.testing.assert_array_equal(
            copy_to_cpu(expected.read_partial_ids()),
            copy_to_cpu(restored.read_partial_ids()),
        )


if __name__ == "__main__":
    unittest.main()
