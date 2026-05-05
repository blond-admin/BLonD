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

        self.time_limit = (0.0, 1e-9)
        self.energy_limit = (-1e6, 1e6)

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

        # Check number of particles
        self.assertEqual(len(dt), 100)
        self.assertEqual(len(dE), 100)

        # Check bounds
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

        # Check that all particles are within purge limits
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


if __name__ == "__main__":
    unittest.main()