import unittest
import warnings
from unittest import mock
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import Beam, Simulation, backend, copy_to_cpu, proton, uranium_29
from blond.core.backends.mpi_distributed.distributed_array import (
    DistributedArray,
)
from blond.core.backends.mpi_distributed.helpers import (
    MPI_COMM_WORLD,
    MPI_RANK,
    MPI_SIZE,
    mpi_is_distributed,
)
from blond.core.beam.base import BeamBaseClass, BeamFlags
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82
from blond.generals.plotting_ import DEFAULT_CMAP
from blond.testing.backend_testing import BLonDTestCase


class TestBeam(BLonDTestCase):
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

    def test_setup_beam_unknown_mpi_mode(self) -> None:
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

    def test_plot_hist2d_uses_default_cmap(self) -> None:
        image = self.beam.plot_hist2d()
        self.assertEqual(image.get_cmap().name, DEFAULT_CMAP)
        plt.close(image.axes.figure)

    def test_plot_hist2d_keeps_user_cmap(self) -> None:
        image = self.beam.plot_hist2d(cmap="viridis")
        self.assertEqual(image.get_cmap().name, "viridis")
        plt.close(image.axes.figure)

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
        chunk = np.array_split(np.arange(12), MPI_SIZE)[MPI_RANK]
        for array in (beam._dt, beam._dE, beam._ids):
            np.testing.assert_allclose(copy_to_cpu(array.array_local), chunk)
        np.testing.assert_allclose(
            copy_to_cpu(beam._flags.array_local),
            np.full(chunk.size, BeamFlags.ACTIVE.value),
        )

    @pytest.mark.mpi
    def test_ratio_mpi_root_distributes(self) -> None:
        """Ratio must equal `intensity / global_n_macroparticles` on every rank."""
        if not mpi_is_distributed():
            self.skipTest("Only MPI")
        intensity = 1e12
        n_macroparticles = 12  # divisible by 2, 3 and 4 ranks
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
        n_macroparticles = 1200  # divisible by 2, 3 and 4 ranks
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


class TestProbeBunch(BLonDTestCase):
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


class TestWeightenedBeam(BLonDTestCase):
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


class TestBeamSetupBeamArguments(BLonDTestCase):
    def setUp(self):
        self.beam = Beam(intensity=1e10, particle_type=proton)

    def test_setup_beam_rejects_unknown_kwargs(self):
        with self.assertRaises(TypeError):
            self.beam.setup_beam(
                dt=np.zeros(4),
                dE=np.zeros(4),
                reference_totl_energy=450e9,  # typo
            )

    def test_setup_beam_applies_zero_reference_time(self):
        self.beam.reference.time = 1e-3
        self.beam.setup_beam(
            dt=np.zeros(4), dE=np.zeros(4), reference_time=0.0
        )
        self.assertEqual(self.beam.reference.time, 0.0)


@pytest.mark.mpi
class TestSetupBeamRootDistributes(BLonDTestCase):
    """Only rank 0 needs the full beam; the other ranks may pass `None`."""

    def setUp(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")

    def _expected_chunk(self, array):
        # chunk sizes as `np.array_split`, in rank order
        return np.array_split(array, MPI_SIZE)[MPI_RANK]

    def test_other_ranks_may_pass_none(self) -> None:
        dt = np.arange(13, dtype=float)  # not divisible: uneven chunks
        dE = 10 * np.arange(13, dtype=float)
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(
            dt=dt if MPI_RANK == 0 else None,
            dE=dE if MPI_RANK == 0 else None,
            mpi_mode="root-distributes",
        )
        for actual, expected in (
            (beam.dt, dt),
            (beam.dE, dE),
            (beam.ids, np.arange(13)),
            (beam.flags, np.full(13, BeamFlags.ACTIVE.value)),
        ):
            np.testing.assert_array_equal(
                actual.copy_as_numpy(), self._expected_chunk(expected)
            )
        self.assertEqual(beam.dt.array_local.dtype, backend.float)
        self.assertEqual(beam.flags.array_local.dtype, np.int32)
        self.assertEqual(beam.common_array_size, 13)

    def test_flags_from_rank_zero_are_scattered(self) -> None:
        flags = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(
            dt=np.zeros(8) if MPI_RANK == 0 else None,
            dE=np.zeros(8) if MPI_RANK == 0 else None,
            flags=flags if MPI_RANK == 0 else None,
            mpi_mode="root-distributes",
        )
        np.testing.assert_array_equal(
            beam.flags.copy_as_numpy(), self._expected_chunk(flags)
        )

    def test_invalid_input_on_rank_zero_raises_on_every_rank(self) -> None:
        beam = Beam(intensity=1e10, particle_type=proton)
        with self.assertRaises(AssertionError):
            beam.setup_beam(
                dt=np.zeros(8) if MPI_RANK == 0 else None,
                dE=np.zeros(9) if MPI_RANK == 0 else None,  # length mismatch
                mpi_mode="root-distributes",
            )


@pytest.mark.mpi
class TestBadInputOnOneRank(BLonDTestCase):
    """Input wrong on one rank only must raise on every rank, not hang."""

    def setUp(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")

    def tearDown(self) -> None:
        # a rank that skipped a collective the others made is out of step
        tags = MPI_COMM_WORLD.allgather(self.id())
        self.assertEqual(set(tags), {self.id()})

    def test_invalid_flags_on_one_rank(self) -> None:
        flags = np.full(4, BeamFlags.ACTIVE.value, dtype=np.int32)
        if MPI_RANK == MPI_SIZE - 1:
            flags[0] = BeamFlags.ACTIVE.value + 1
        beam = Beam(intensity=1e10, particle_type=proton)
        with self.assertRaises(AssertionError):
            beam.setup_beam(
                dt=np.zeros(4),
                dE=np.zeros(4),
                flags=flags,
                mpi_mode="all-ranks",
            )

    def test_local_lengths_mismatched_with_equal_totals(self) -> None:
        # rank 0: 3 dt, 2 dE; rank 1: 2 dt, 3 dE -> equal totals
        n_dt, n_dE = {0: (3, 2), 1: (2, 3)}.get(MPI_RANK, (4, 4))
        beam = Beam(intensity=1e10, particle_type=proton)
        with self.assertRaises(AssertionError):
            beam.setup_beam(
                dt=np.zeros(n_dt), dE=np.zeros(n_dE), mpi_mode="all-ranks"
            )

    def test_add_particles_mismatched_on_one_rank(self) -> None:
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(dt=np.zeros(4), dE=np.zeros(4), mpi_mode="all-ranks")
        # ranks 0 and 1 mismatch but cancel out; others match locally
        n_dt, n_dE = {0: (2, 3), 1: (3, 2)}.get(MPI_RANK, (2, 2))
        with self.assertRaises(ValueError):
            beam.add_particles(
                dt=DistributedArray(backend.zeros(n_dt, dtype=backend.float)),
                dE=DistributedArray(backend.zeros(n_dE, dtype=backend.float)),
            )


class TestBeamIsDistributed(BLonDTestCase):
    """`is_distributed` tells whether the particles are split over ranks."""

    def test_is_false_without_mpi(self) -> None:
        if mpi_is_distributed():
            self.skipTest("Runs only without `mpirun`")
        beam = Beam(intensity=1e10, particle_type=proton)
        self.assertFalse(beam.is_distributed)

    @pytest.mark.mpi
    def test_is_true_under_mpi(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")
        beam = Beam(intensity=1e10, particle_type=proton)
        self.assertTrue(beam.is_distributed)
        self.assertTrue(
            ProbeBeam(particle_type=proton, dt=np.zeros(4)).is_distributed
        )

    @pytest.mark.mpi
    def test_sort_by_dt_refuses_distributed_beam(self) -> None:
        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(
            dt=np.arange(12.0)[::-1],
            dE=np.zeros(12),
            mpi_mode="root-distributes",
        )
        with self.assertRaises(NotImplementedError):
            beam.sort_by_dt()

    @pytest.mark.mpi
    def test_profile_and_beam_observations_run_distributed(self) -> None:
        """These handle MPI already; `is_distributed` must not block them."""
        from blond import (
            BeamHist2dOncePerTurn,
            BeamObservationOncePerTurn,
            ConstantMagneticCycle,
            DriftSimple,
            Ring,
            SingleHarmonicRFStation,
            StaticProfile,
            momentum_compaction_factor,
        )

        if not mpi_is_distributed():
            self.skipTest("Runs only with `mpirun`")
        circumference = 26658.883  # [m]
        ring = Ring(circumference)
        ring.add_elements(
            [
                SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0),
                DriftSimple(
                    orbit_length=circumference,
                    momentum_compaction_factor=momentum_compaction_factor(
                        transition_gamma=55.759505
                    ),
                ),
                StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=64),
            ]
        )
        simulation = Simulation(
            ring=ring,
            magnetic_cycle=ConstantMagneticCycle(
                value=450e9, reference_particle=proton
            ),
        )
        beam = Beam.simple_gaussian(
            n_macroparticles=1000,
            intensity=1e9,
            particle_type=proton,
            dt_scale=0.1e-9,
            dE_scale=1e8,
            dt_offset=1.25e-9,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # rank-0 notes
            simulation.run_simulation(
                beams=(beam,),
                n_turns=2,
                show_progressbar=False,
                observe=(
                    BeamHist2dOncePerTurn(each_turn_i=1),
                    BeamObservationOncePerTurn(each_turn_i=1),
                ),
            )


class TestSetupBeamCopy(BLonDTestCase):
    """`setup_beam` copies its input unless told it owns the arrays."""

    def setUp(self) -> None:
        self.beam = Beam(intensity=1e12, particle_type=proton)
        self.dt = backend.linspace(20, 30, 10, dtype=backend.float)
        self.dE = backend.linspace(1, 10, 10, dtype=backend.float)

    def test_copies_by_default(self) -> None:
        self.beam.setup_beam(dt=self.dt, dE=self.dE)
        self.dt[0] = -1.0
        self.dE[0] = -1.0
        self.assertEqual(float(self.beam.dt.array_local[0]), 20.0)
        self.assertEqual(float(self.beam.dE.array_local[0]), 1.0)

    def test_two_beams_from_one_array_stay_independent(self) -> None:
        other = Beam(intensity=1e12, particle_type=proton)
        self.beam.setup_beam(dt=self.dt, dE=self.dE)
        other.setup_beam(dt=self.dt, dE=self.dE)
        self.assertIsNot(self.beam.dt.array_local, other.dt.array_local)

    def test_without_copy_keeps_the_arrays(self) -> None:
        self.beam.setup_beam(dt=self.dt, dE=self.dE, copy=False)
        self.assertIs(self.beam.dt.array_local, self.dt)
        self.assertIs(self.beam.dE.array_local, self.dE)

    def test_without_copy_still_converts_dtype(self) -> None:
        self.beam.setup_beam(dt=[1, 2, 3], dE=[4, 5, 6], copy=False)
        self.assertEqual(self.beam.dt.array_local.dtype, backend.float)
        self.assertEqual(self.beam.dE.array_local.dtype, backend.float)

    def test_without_copy_keeps_flags(self) -> None:
        flags = backend.ones(10, dtype=np.int32)
        self.beam.setup_beam(dt=self.dt, dE=self.dE, flags=flags, copy=False)
        self.assertIs(self.beam.flags.array_local, flags)

    def test_simple_gaussian_does_not_copy(self) -> None:
        with mock.patch.object(
            Beam, "setup_beam", autospec=True, side_effect=Beam.setup_beam
        ) as setup_beam:
            Beam.simple_gaussian(
                n_macroparticles=100,
                intensity=1e10,
                particle_type=proton,
                dt_scale=1e-9,
                dE_scale=1e9,
                seed=0,
            )
        self.assertIs(setup_beam.call_args.kwargs["copy"], False)

    def test_default_flags_are_active_int32(self) -> None:
        self.beam.setup_beam(dt=self.dt, dE=self.dE, copy=False)
        flags = copy_to_cpu(self.beam.flags.array_local)
        self.assertEqual(flags.dtype, np.int32)
        np.testing.assert_array_equal(flags, BeamFlags.ACTIVE.value)


class TestSimpleGaussianBackendDependentRandomNumbers(BLonDTestCase):
    def _dt(self, **kwargs):
        beam = Beam.simple_gaussian(
            n_macroparticles=1000,
            intensity=1e10,
            particle_type=proton,
            dt_scale=1e-9,
            dE_scale=1e9,
            seed=4,
            **kwargs,
        )
        return copy_to_cpu(beam.dt.array_local)

    def test_change_nothing_on_cpu(self):
        if backend.is_gpu:
            self.skipTest("Only on a CPU backend")
        np.testing.assert_array_equal(
            self._dt(backend_dependent_random_numbers=True), self._dt()
        )


if __name__ == "__main__":
    unittest.main()
