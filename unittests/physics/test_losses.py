import unittest

import matplotlib.pyplot as plt
import numpy as np
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.flags import BeamFlags
from blond.generals.cupy.no_cupy_import import copy_to_cpu
from blond.physics.losses import LossesBaseClass
from blond.testing.mocks import beam_mock, simulation_mock

from blond import Beam, BoxLosses, Simulation, proton, uranium_29


class LossesBaseClassHelper(LossesBaseClass):
    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs,
    ) -> None:
        pass


class TestLossesBaseClass(unittest.TestCase):
    def test_init(self):
        LossesBaseClassHelper(purge_flagged_macroparticles=True)

    def test_track(self):
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        flags = np.ones(10)
        flags[:5] = BeamFlags.LOST.value
        beam.setup_beam(dt=np.arange(10), dE=np.ones(10), flags=flags)
        LossesBaseClassHelper(
            purge_flagged_macroparticles=True
        )._purge_particles(beam=beam)
        self.assertEqual(beam.common_array_size, 5)
        np.testing.assert_almost_equal(
            np.sort(copy_to_cpu(beam.read_partial_dt())),
            np.sort(np.arange(10)[5:]),
        )
        np.testing.assert_equal(beam.intensity, 0.5)


class TestBoxLosses(unittest.TestCase):
    def setUp(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=True,
            t_min=-1,
            t_max=2,
            e_min=-10,
            e_max=20,
        )

    def test___init__(self):
        pass  # calls setUp()

    def test_on_init_simulation(self):
        self.box_losses.on_init_simulation(
            simulation=simulation_mock,
        )

    def test_on_run_simulation(self):
        self.box_losses.on_run_simulation(
            simulation=simulation_mock,
            beam=beam_mock,
            n_turns=1,
        )

    def test_track(self):
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 201),
            dE=np.linspace(-100, 100, 201),
        )
        self.box_losses.track(beam=beam)

        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dt()) >= -1, True
        )
        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) <= 2, True)
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) >= -10, True
        )
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) <= 20, True
        )

    def test_wrong_args(self):
        with self.assertRaises(AssertionError):
            self.box_losses = BoxLosses(
                purge_flagged_macroparticles=False,
                t_min=2,
                t_max=2,
                e_min=-10,
                e_max=20,
            )

    def test_wrong_arg2(self):
        with self.assertRaises(AssertionError):
            self.box_losses = BoxLosses(
                purge_flagged_macroparticles=False,
                t_min=1,
                t_max=2,
                e_min=30,
                e_max=20,
            )

    def test_track2(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=True,
            t_min=None,
            t_max=2,
            e_min=-10,
            e_max=20,
        )
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 201),
            dE=np.linspace(-100, 100, 201),
        )
        self.box_losses.track(beam=beam)
        print(beam._dt)
        print(beam._dE)

        # np.testing.assert_equal(beam._dt >= 1, True)
        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) <= 2, True)
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) >= -10, True
        )
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) <= 20, True
        )
        self.assertLess(beam.intensity, 1e12)

    def test_track3(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=True,
            t_min=1,
            t_max=None,
            e_min=-10,
            e_max=20,
        )
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 201),
            dE=np.linspace(-100, 100, 201),
        )
        self.box_losses.track(beam=beam)
        print(beam._dt)
        print(beam._dE)

        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) >= 1, True)
        # np.testing.assert_equal(copy_to_cpu()(beam._dt) <= 2, True)
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) >= -10, True
        )
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) <= 20, True
        )
        self.assertLess(beam.intensity, 1e12)

    def test_track4(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=True,
            t_min=1,
            t_max=2,
            e_min=None,
            e_max=20,
        )
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 201),
            dE=np.linspace(-100, 100, 201),
        )
        self.box_losses.track(beam=beam)
        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) >= 1, True)
        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) <= 2, True)
        # np.testing.assert_equal(copy_to_cpu()(beam._dE) >= -10, True)
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) <= 20, True
        )

    def test_track5(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=True,
            t_min=1,
            t_max=2,
            e_min=-10,
            e_max=None,
        )
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 201),
            dE=np.linspace(-100, 100, 201),
        )
        self.box_losses.track(beam=beam)
        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) >= 1, True)
        np.testing.assert_equal(copy_to_cpu(beam.read_partial_dt()) <= 2, True)
        np.testing.assert_equal(
            copy_to_cpu(beam.read_partial_dE()) >= -10, True
        )
        # np.testing.assert_equal(copy_to_cpu()(beam.read_partial_dE()) <= 20, True)
        self.assertLess(beam.intensity, 1e12)

    def test_track6(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=True,
            t_min=1,
            t_max=2,
            e_min=-10,
            e_max=20,
        )
        beam = Beam(intensity=1e12, particle_type=proton)
        dt_line = np.linspace(-10, 10, 20)
        dE_line = np.linspace(-100, 100, 20)
        dt, dE = np.meshgrid(dt_line, dE_line, indexing="ij")
        dt = dt.flatten()
        dE = dE.flatten()
        beam.setup_beam(
            dt=dt,
            dE=dE,
        )
        DEV_PLOT = False
        if DEV_PLOT:
            beam.plot_scatter()
        self.box_losses.track(beam=beam)
        if DEV_PLOT:
            beam.plot_scatter()
            plt.axhline(self.box_losses.e_max)
            plt.axhline(self.box_losses.e_min)
            plt.axvline(self.box_losses.t_max)
            plt.axvline(self.box_losses.t_min)
            plt.show()

        self.assertEqual(3, len(beam.read_partial_dt()))
        self.assertEqual(3, len(beam.read_partial_dE()))

        np.testing.assert_equal(
            (copy_to_cpu(beam.read_partial_dt()) < 1)
            | (copy_to_cpu(beam.read_partial_dt()) > 2)
            | (copy_to_cpu(beam.read_partial_dE()) < -10)
            | (copy_to_cpu(beam.read_partial_dE()) > 20),
            copy_to_cpu(~beam.read_partial_flags().astype(bool)),
        )
        self.assertLess(beam.intensity, 1e12)

    def test_track7(self):
        beam = Beam(intensity=1.0, particle_type=uranium_29)
        flags = np.ones(10)
        flags[:5] = BeamFlags.LOST.value
        beam.setup_beam(dt=np.arange(10), dE=np.ones(10), flags=flags)
        LossesBaseClassHelper(
            purge_flagged_macroparticles=True
        )._purge_particles(beam=beam)
        self.assertEqual(beam.common_array_size, 5)
        np.testing.assert_almost_equal(
            np.sort(copy_to_cpu(beam.read_partial_dt())),
            np.sort(np.arange(10)[5:]),
        )

        self.assertLess(beam.intensity, 1e12)


if __name__ == "__main__":
    unittest.main()
