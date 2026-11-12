import copy
import importlib
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blond.core.backends.mpi_distributed.helpers import (
    MPI_RANK,
    MPI_SIZE,
    distributed_arange,
    mpi_aware_random_generator_cpu,
    mpi_is_distributed,
    mpi_is_root,
)
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.testing.backend_testing import BLonDTestCase


@pytest.mark.mpi
class TestCallablesWithMPI(BLonDTestCase):
    def setUp(self):
        is_distributed = mpi_is_distributed()
        if not is_distributed:
            self.skipTest("Only with MPI")

    def test_mpi_local_size(self):
        from blond.core.backends.mpi_distributed.helpers import mpi_local_size

        with self.assertWarnsRegex(
            UserWarning, "Because MPI is used, `global_size`"
        ):
            local_n = mpi_local_size(
                global_size=13,  # not divisible by 2 or 3 ranks
                warning_hint="global_size",
            )
            self.assertEqual(local_n, 13 // MPI_SIZE)

    def test_distributed_arange(self):
        from blond.core.backends.mpi_distributed.helpers import (
            distributed_arange,
        )

        da = distributed_arange(12, dtype=np.int32)
        np.testing.assert_allclose(
            da.copy_as_numpy(),
            np.arange(12 * da._rank, 12 * (da._rank + 1)),
            err_msg=f"{da._rank=} {da._size=}",
        )

    def test_distributed_zeros(self):
        from blond.core.backends.mpi_distributed.helpers import (
            distributed_zeros,
        )

        da = distributed_zeros(12, dtype=np.int32)
        np.testing.assert_array_equal(
            da.copy_as_numpy(),
            np.zeros(12),
            err_msg=f"{da._rank=} {da._size=}",
        )

    def test_mpi_is_root(self):
        da = distributed_arange(12, dtype=np.int32)
        self.assertEqual(mpi_is_root(), da._rank == 0)

    def test_simple_gaussian_shares_no_values_across_ranks(self):
        from mpi4py import MPI

        from blond import Beam, proton

        beam = Beam.simple_gaussian(
            n_macroparticles=200_000,
            intensity=1e10,
            particle_type=proton,
            dt_scale=1e-9,
            dE_scale=1e6,
            seed=0,
        )
        for coordinate in (beam.dt, beam.dE):
            local = coordinate.copy_as_numpy()
            gathered = MPI.COMM_WORLD.allgather(local)
            n_global = sum(array.size for array in gathered)
            n_unique = np.unique(np.concatenate(gathered)).size
            self.assertEqual(n_unique, n_global)


def _run_as_rank(function, rank: int, call):
    """Run `call` while `function` sees itself as MPI rank `rank`."""
    # Patch the globals the production function really uses: another
    # test re-imports the helpers module, so a fresh import could differ.
    # Only the rank: those globals belong to the whole helpers module, and
    # a size above 1 would make `mpi_is_distributed()` true everywhere, so
    # collectives would run without MPI.
    with patch.dict(function.__globals__, {"MPI_RANK": rank}):
        return call()


class TestRandomStreamsAcrossRanks(BLonDTestCase):
    """Each MPI rank must draw from its own, non-overlapping stream."""

    N_MACROPARTICLES = 200_000

    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Simulates ranks by patching; only without MPI")

    def test_rank_zero_reproduces_serial_stream(self):
        rng = _run_as_rank(
            mpi_aware_random_generator_cpu,
            rank=0,
            call=lambda: mpi_aware_random_generator_cpu(seed=42),
        )
        np.testing.assert_equal(
            rng.standard_normal(1000),
            np.random.default_rng(42).standard_normal(1000),
        )

    def test_seed_none_still_works(self):
        rng = _run_as_rank(
            mpi_aware_random_generator_cpu,
            rank=1,
            call=lambda: mpi_aware_random_generator_cpu(seed=None),
        )
        self.assertEqual(rng.standard_normal(10).shape, (10,))

    def _simple_gaussian_as_rank(self, rank):
        from blond import Beam, proton
        from blond.core.beam import beams

        return _run_as_rank(
            beams.mpi_aware_random_generator_cpu,
            rank=rank,
            call=lambda: Beam.simple_gaussian(
                n_macroparticles=self.N_MACROPARTICLES,
                intensity=1e10,
                particle_type=proton,
                dt_scale=1e-9,
                dE_scale=1e6,
                seed=0,
            ),
        )

    def test_simple_gaussian_ranks_share_no_values(self):
        beam_rank_0 = self._simple_gaussian_as_rank(0)
        beam_rank_1 = self._simple_gaussian_as_rank(1)
        for name in ("dt", "dE"):
            shared = np.intersect1d(
                getattr(beam_rank_0, name).copy_as_numpy(),
                getattr(beam_rank_1, name).copy_as_numpy(),
            )
            self.assertEqual(shared.size, 0, msg=f"shared {name} values")

    def _generator_used_by_particle_coordinates(self, rank):
        """Copy of the fresh generator `generate_particle_coordinates` got."""
        from blond.beam_preparation import helpers as preparation_helpers

        original = preparation_helpers.mpi_aware_random_generator_cpu
        received = []

        def recording(*args, **kwargs):
            generator = original(*args, **kwargs)
            received.append(copy.deepcopy(generator))
            return generator

        with patch.object(
            preparation_helpers, "mpi_aware_random_generator_cpu", recording
        ):
            self._particle_coordinates_as_rank(rank, original)
        self.assertEqual(len(received), 1)
        return received[0]

    def test_generate_particle_coordinates_streams_do_not_overlap(self):
        # The routine draws ~3 numbers per particle (`choice` and two
        # `triangular`) from one generator, so a rank's stream must not
        # reappear within the next rank's first draws.
        generator_rank_0 = self._generator_used_by_particle_coordinates(0)
        generator_rank_1 = self._generator_used_by_particle_coordinates(1)
        n_local = self.N_MACROPARTICLES // 2
        raw_rank_0 = generator_rank_0.bit_generator.random_raw(4 * n_local)
        raw_rank_1 = generator_rank_1.bit_generator.random_raw(4 * n_local)
        shared = np.intersect1d(raw_rank_0, raw_rank_1)
        self.assertEqual(shared.size, 0, msg="ranks share random draws")

    def _particle_coordinates_as_rank(self, rank, patched_function=None):
        from blond.beam_preparation import helpers as preparation_helpers

        if patched_function is None:
            patched_function = (
                preparation_helpers.mpi_aware_random_generator_cpu
            )
        time_axis = np.linspace(-1e-9, 1e-9, 64)
        energy_axis = np.linspace(-1e6, 1e6, 64)
        time_grid, deltaE_grid = np.meshgrid(time_axis, energy_axis)
        density_grid = np.exp(
            -(time_grid**2) / 0.3e-18 - deltaE_grid**2 / 0.3e12
        )
        return _run_as_rank(
            patched_function,
            rank=rank,
            call=lambda: preparation_helpers.generate_particle_coordinates(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                density_grid=density_grid,
                n_macroparticles=self.N_MACROPARTICLES,
                seed=0,
            ),
        )


class TestCallablesNoMPI(BLonDTestCase):
    def setUp(self):
        if mpi_is_distributed():
            self.skipTest("Only without MPI")

    def test_mpi_is_root(self):
        self.assertTrue(mpi_is_root())

    def test_mpi_local_size(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop(
                "blond.core.backends.mpi_distributed.helpers", None
            )
            from blond.core.backends.mpi_distributed.helpers import (
                mpi_local_size,
            )

            self.assertEqual(
                10, mpi_local_size(global_size=10, warning_hint="")
            )

    def test_distributed_arange(self):
        with patch.dict(sys.modules, {"mpi4py": None}):
            # trigger new import
            sys.modules.pop(
                "blond.core.backends.mpi_distributed.helpers", None
            )
            from blond.core.backends.mpi_distributed.helpers import (
                distributed_arange,
            )

            da = distributed_arange(12, dtype=np.int32)
            np.testing.assert_allclose(da.copy_as_numpy(), np.arange(0, 12))

    def test_mpi_is_distributed_size_one(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_SIZE", 1):
            result = mpi_is_distributed()
        self.assertFalse(result)

    def test_mpi_is_distributed_size_one_returns_bool(self):
        """With MPI size 1 the result is `False`, not an implicit `None`."""
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_SIZE", 1):
            result = mpi_is_distributed()
        self.assertIs(result, False)

    def test_mpi_is_distributed_size_two(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_SIZE", 2):
            result = mpi_is_distributed()
        self.assertIs(result, True)

    def test_mpi_is_root_on_rank_zero(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_RANK", 0):
            result = mpi_is_root()
        self.assertIs(result, True)

    def test_mpi_is_root_on_non_root_rank(self):
        with patch("blond.core.backends.mpi_distributed.helpers.MPI_RANK", 1):
            result = mpi_is_root()
        self.assertIs(result, False)


def _current_helpers():
    """
    Return the helpers module currently in `sys.modules`.

    Other tests pop and re-import it, so a reference taken at
    import time can be stale.
    """
    return importlib.import_module(
        "blond.core.backends.mpi_distributed.helpers"
    )


class TestMpiRunOnRoot(BLonDTestCase):
    """`mpi_run_on_root` runs on rank 0 and fails on all ranks together."""

    def _patch_rank(self, rank: int, size: int, broadcast_result=None):
        helpers = _current_helpers()
        comm = MagicMock()
        comm.bcast.return_value = broadcast_result
        patches = (
            patch.object(helpers, "MPI_RANK", rank),
            patch.object(helpers, "MPI_SIZE", size),
            patch.object(helpers, "MPI_COMM_WORLD", comm),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return helpers, comm

    def test_serial_runs_function_without_communication(self):
        helpers, comm = self._patch_rank(rank=0, size=1)
        function = MagicMock()
        helpers.mpi_run_on_root(function)
        function.assert_called_once_with()
        comm.bcast.assert_not_called()

    def test_serial_reraises_original_exception(self):
        helpers, _ = self._patch_rank(rank=0, size=1)
        function = MagicMock(side_effect=OSError("disk full"))
        with self.assertRaisesRegex(OSError, "disk full"):
            helpers.mpi_run_on_root(function)

    def test_root_broadcasts_success(self):
        helpers, comm = self._patch_rank(rank=0, size=2)
        function = MagicMock()
        helpers.mpi_run_on_root(function)
        function.assert_called_once_with()
        comm.bcast.assert_called_once_with(None, root=0)

    def test_root_broadcasts_failure_and_reraises(self):
        helpers, comm = self._patch_rank(rank=0, size=2)
        function = MagicMock(side_effect=OSError("disk full"))
        with self.assertRaisesRegex(OSError, "disk full"):
            helpers.mpi_run_on_root(function)
        comm.bcast.assert_called_once()
        message = comm.bcast.call_args.args[0]
        self.assertIn("disk full", message)
        self.assertIn("OSError", message)

    def test_non_root_skips_function(self):
        helpers, comm = self._patch_rank(rank=1, size=2)
        function = MagicMock()
        helpers.mpi_run_on_root(function)
        function.assert_not_called()
        comm.bcast.assert_called_once_with(None, root=0)

    def test_non_root_raises_when_root_failed(self):
        helpers, _ = self._patch_rank(
            rank=1, size=2, broadcast_result="OSError('disk full')"
        )
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            helpers.mpi_run_on_root(MagicMock())


class TestMpiLocalGpuIndex(BLonDTestCase):
    """Each rank of a node picks its own GPU."""

    def _patch(self, size: int, local_rank: int):
        helpers = _current_helpers()
        for patcher in (
            patch.object(helpers, "MPI_SIZE", size),
            patch.object(helpers, "MPI_LOCAL_RANK", local_rank),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return helpers

    def test_serial_keeps_current_device(self):
        helpers = self._patch(size=1, local_rank=0)
        self.assertIsNone(helpers.mpi_local_gpu_index(n_devices=4))

    def test_ranks_spread_over_devices(self):
        indices = [
            self._patch(size=8, local_rank=local_rank).mpi_local_gpu_index(
                n_devices=4
            )
            for local_rank in range(8)
        ]
        self.assertEqual(indices, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_one_visible_device_per_rank(self):
        """`CUDA_VISIBLE_DEVICES` per rank leaves one device: index 0."""
        helpers = self._patch(size=8, local_rank=5)
        self.assertEqual(helpers.mpi_local_gpu_index(n_devices=1), 0)

    def test_local_rank_without_mpi_is_zero(self):
        helpers = _current_helpers()
        if helpers.MPI is not None:
            self.skipTest("Only without MPI")
        self.assertEqual(helpers.MPI_LOCAL_RANK, 0)


class TestMpiCudaAware(BLonDTestCase):
    """`BLOND_MPI_CUDA_AWARE` opts in to passing GPU memory to MPI."""

    def _cuda_aware(self, environ: dict[str, str]) -> bool:
        with patch.dict(os.environ, environ, clear=True):
            return _current_helpers().mpi_cuda_aware()

    def test_default_is_not_cuda_aware(self):
        self.assertIs(self._cuda_aware({}), False)

    def test_true_and_false(self):
        self.assertIs(self._cuda_aware({"BLOND_MPI_CUDA_AWARE": "True"}), True)
        self.assertIs(
            self._cuda_aware({"BLOND_MPI_CUDA_AWARE": "False"}), False
        )

    def test_invalid_value_raises(self):
        with self.assertRaisesRegex(ValueError, "BLOND_MPI_CUDA_AWARE"):
            self._cuda_aware({"BLOND_MPI_CUDA_AWARE": "yes"})


@pytest.mark.mpi
class TestMpiLocalRankWithMPI(BLonDTestCase):
    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Only with MPI")

    def test_single_node_local_rank_equals_rank(self):
        """On one node (as in the test run), local and global rank agree."""
        helpers = _current_helpers()
        self.assertEqual(helpers.MPI_LOCAL_RANK, helpers.MPI_RANK)


@pytest.mark.mpi
class TestMpiRunOnRootWithMPI(BLonDTestCase):
    def setUp(self):
        if not mpi_is_distributed():
            self.skipTest("Only with MPI")

    def test_failure_on_root_raises_on_every_rank(self):
        from blond.core.backends.mpi_distributed.helpers import (
            mpi_run_on_root,
        )

        def fail():
            raise OSError("disk full")

        expected = OSError if mpi_is_root() else RuntimeError
        with self.assertRaisesRegex(expected, "disk full"):
            mpi_run_on_root(fail)


_LAUNCHER_ENV_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_RANK",
    "PMI_RANK",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
)


class TestMpiLaunched(BLonDTestCase):
    """`mpi_launched` decides whether MPI is initialised at all."""

    def _mpi_launched(self, environ):
        """Call `mpi_launched` with `environ` as the whole environment.

        `clear=True` matters: under `mpirun` the real environment already
        holds launcher variables.
        """
        from blond.core.backends.mpi_distributed.helpers import mpi_launched

        with patch.dict(os.environ, environ, clear=True):
            return mpi_launched()

    def test_plain_python_is_not_launched(self):
        self.assertIs(self._mpi_launched({}), False)

    def test_launcher_variables_are_detected(self):
        for key in _LAUNCHER_ENV_KEYS:
            with self.subTest(key=key):
                self.assertIs(self._mpi_launched({key: "0"}), True)

    def test_override_forces_mpi_on(self):
        self.assertIs(self._mpi_launched({"BLOND_LOAD_MPI": "True"}), True)

    def test_override_forces_mpi_off_under_launcher(self):
        environ = {"BLOND_LOAD_MPI": "False", "OMPI_COMM_WORLD_SIZE": "2"}
        self.assertIs(self._mpi_launched(environ), False)

    def test_invalid_override_raises(self):
        with self.assertRaises(ValueError):
            self._mpi_launched({"BLOND_LOAD_MPI": "yes"})


class TestImportDoesNotInitialiseMpi(BLonDTestCase):
    """Outside an MPI launcher, `import blond` must not call `MPI_Init`.

    `MPI_Init` costs ~0.5 s per import and sets up the MPI runtime in
    processes that never use MPI.
    """

    def test_import_blond_does_not_import_mpi4py_mpi(self):
        env = os.environ.copy()
        for key in (*_LAUNCHER_ENV_KEYS, "BLOND_LOAD_MPI", "PYCHARM_HOSTED"):
            env.pop(key, None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, blond; print('mpi4py.MPI' in sys.modules)",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.split()[-1], "False")


class TestImportMpiUnderLauncher(BLonDTestCase):
    """Module setup of `helpers` when an MPI launcher is detected.

    A fake `mpi4py` stands in for the real one, so that the test process
    itself never calls `MPI_Init`.
    """

    _HELPERS = "blond.core.backends.mpi_distributed.helpers"

    def _import_fresh_helpers(self, mpi4py_module):
        with (
            patch.dict(os.environ, {"BLOND_LOAD_MPI": "True"}),
            patch.dict(sys.modules, {"mpi4py": mpi4py_module}),
        ):
            sys.modules.pop(self._HELPERS, None)
            return importlib.import_module(self._HELPERS)

    def test_rank_and_size_come_from_comm_world(self):
        comm_world = MagicMock()
        comm_world.Get_rank.return_value = 1
        comm_world.Get_size.return_value = 4
        fake_mpi4py = MagicMock()
        fake_mpi4py.MPI.COMM_WORLD = comm_world

        helpers = self._import_fresh_helpers(fake_mpi4py)

        self.assertIs(helpers.MPI, fake_mpi4py.MPI)
        self.assertIs(helpers.MPI_COMM_WORLD, comm_world)
        self.assertEqual(helpers.MPI_RANK, 1)
        self.assertEqual(helpers.MPI_SIZE, 4)

    def test_missing_mpi4py_warns_and_runs_serially(self):
        with self.assertWarns(ImportWarning):
            helpers = self._import_fresh_helpers(None)

        self.assertIsNone(helpers.MPI)
        self.assertIsNone(helpers.MPI_COMM_WORLD)
        self.assertEqual(helpers.MPI_SIZE, 1)
