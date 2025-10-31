import unittest

import numpy as np

from blond import Beam, BoxLosses, proton
from blond.testing.mocks import beam_mock, simulation_mock


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
            turn_i_init=0,
        )

    def test_track(self):
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 201),
            dE=np.linspace(-100, 100, 201),
        )
        self.box_losses.track(beam=beam)

        np.testing.assert_equal(beam._dt >= -1, True)
        np.testing.assert_equal(beam._dt <= 2, True)
        np.testing.assert_equal(beam._dE >= -10, True)
        np.testing.assert_equal(beam._dE <= 20, True)

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
        np.testing.assert_equal(beam._dt <= 2, True)
        np.testing.assert_equal(beam._dE >= -10, True)
        np.testing.assert_equal(beam._dE <= 20, True)

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

        np.testing.assert_equal(beam._dt >= 1, True)
        # np.testing.assert_equal(beam._dt <= 2, True)
        np.testing.assert_equal(beam._dE >= -10, True)
        np.testing.assert_equal(beam._dE <= 20, True)

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
        np.testing.assert_equal(beam._dt >= 1, True)
        np.testing.assert_equal(beam._dt <= 2, True)
        # np.testing.assert_equal(beam._dE >= -10, True)
        np.testing.assert_equal(beam._dE <= 20, True)

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
        np.testing.assert_equal(beam._dt >= 1, True)
        np.testing.assert_equal(beam._dt <= 2, True)
        np.testing.assert_equal(beam._dE >= -10, True)
        # np.testing.assert_equal(beam._dE <= 20, True)

    def test_track6(self):
        self.box_losses = BoxLosses(
            purge_flagged_macroparticles=False,
            t_min=1,
            t_max=2,
            e_min=-10,
            e_max=20,
        )
        beam = Beam(intensity=1e12, particle_type=proton)
        beam.setup_beam(
            dt=np.linspace(-10, 10, 21),
            dE=np.linspace(-100, 100, 21),
        )
        self.box_losses.track(beam=beam)

        self.assertEqual(21, len(beam._dt))
        self.assertEqual(21, len(beam._dE))

        np.testing.assert_equal(
            (beam._dt < 1)
            | (beam._dt > 2)
            | (beam._dE < -10)
            | (beam._dE > 20),
            ~beam._flags.astype(bool),
        )


if __name__ == "__main__":
    unittest.main()
