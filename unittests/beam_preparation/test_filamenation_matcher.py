# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import unittest

import numpy as np

from blond.beam_preparation.filamentation_matcher import (
    FilamentationMatcher,
)
from blond.testing.simulation import ExampleSimulation01


class TestFilamentationMatcher(unittest.TestCase):
    def setUp(self):
        self.simulation_ = ExampleSimulation01()
        self.sim = self.simulation_.simulation
        self.beam = self.simulation_.beam1

        self.time_limit = (0.0, 3e-9)
        self.energy_limit = (-1e9, 1e9)

    def test___init__(self):
        matcher = FilamentationMatcher(
            time_limit=self.time_limit,
            energy_limit=self.energy_limit,
            n_macroparticles=100,
            n_iter=10,
            animate=False,
        )
        self.assertEqual(matcher.n_macroparticles, 100)
        self.assertEqual(matcher.n_iter, 10)

    def test_prepare_beam_initialization(self):
        matcher = FilamentationMatcher(
            time_limit=self.time_limit,
            energy_limit=self.energy_limit,
            n_macroparticles=100,
            n_iter=0,  # no evolution
            animate=False,
        )

        matcher.prepare_beam(simulation=self.sim, beam=self.beam)

        dt = self.beam.read_partial_dt()
        dE = self.beam.read_partial_dE()

        # n part assert
        self.assertEqual(len(dt), 100)
        self.assertEqual(len(dE), 100)

        # check bounds
        self.assertTrue(np.all(dt >= self.time_limit[0]))
        self.assertTrue(np.all(dt <= self.time_limit[1]))
        self.assertTrue(np.all(dE >= self.energy_limit[0]))
        self.assertTrue(np.all(dE <= self.energy_limit[1]))

    def test_prepare_beam_with_purge(self):
        matcher = FilamentationMatcher(
            time_limit=self.time_limit,
            energy_limit=self.energy_limit,
            n_macroparticles=100,
            n_iter=2,
            animate=False,
            purge=True,
            purge_limit_time=(0.2e-9, 0.8e-9),
            purge_limit_energy=(-5e5, 5e5),
        )

        matcher.prepare_beam(simulation=self.sim, beam=self.beam)

        dt = self.beam.read_partial_dt()
        dE = self.beam.read_partial_dE()

        # all particles are within purge limits
        self.assertTrue(np.all(dt >= 0.2e-9))
        self.assertTrue(np.all(dt <= 0.8e-9))
        self.assertTrue(np.all(dE >= -5e5))
        self.assertTrue(np.all(dE <= 5e5))

    def test_intensity_preserved_after_purge(self):
        matcher = FilamentationMatcher(
            time_limit=self.time_limit,
            energy_limit=self.energy_limit,
            n_macroparticles=100,
            n_iter=1,
            animate=False,
            purge=True,
            purge_limit_time=self.time_limit,
            purge_limit_energy=self.energy_limit,
        )

        initial_intensity = self.beam.intensity

        matcher.prepare_beam(simulation=self.sim, beam=self.beam)

        self.assertEqual(self.beam.intensity, initial_intensity)

    def test_converges_towards_matched_bunch(self):
        simulation_1 = ExampleSimulation01()
        simulation_2 = ExampleSimulation01()
        simulation_1.simulation.turn_i.value = 0  # there is some cacheing
        simulation_2.simulation.turn_i.value = 0

        # --- Matcher with few iterations (poorly matched)
        matcher_few = FilamentationMatcher(
            time_limit=self.time_limit,
            energy_limit=self.energy_limit,
            n_macroparticles=1000,
            n_iter=5,
            animate=True,
            purge=True,
            purge_limit_time=self.time_limit,
            purge_limit_energy=self.energy_limit,
        )

        # --- Matcher with more iterations
        matcher_many = FilamentationMatcher(
            time_limit=self.time_limit,
            energy_limit=self.energy_limit,
            n_macroparticles=1000,
            n_iter=400,
            animate=True,
            purge=True,
            purge_limit_time=self.time_limit,
            purge_limit_energy=self.energy_limit,
        )

        matcher_few.prepare_beam(simulation_1.simulation, simulation_1.beam1)
        matcher_many.prepare_beam(simulation_2.simulation, simulation_2.beam1)

        # at injection
        dt_few_before = simulation_1.beam1.read_partial_dt().copy()
        dt_many_before = simulation_2.beam1.read_partial_dt().copy()

        simulation_1.simulation.run_simulation(
            beams=[simulation_1.beam1], n_turns=5, show_progressbar=False
        )
        simulation_2.simulation.run_simulation(
            beams=[simulation_2.beam1], n_turns=5, show_progressbar=False
        )

        dt_few_after = simulation_1.beam1.read_partial_dt()
        dt_many_after = simulation_2.beam1.read_partial_dt()

        change_few = np.std(dt_few_after - dt_few_before)
        change_many = np.std(dt_many_after - dt_many_before)

        # Better matched beam should change less
        self.assertLess(change_many, change_few)


if __name__ == "__main__":
    unittest.main()
