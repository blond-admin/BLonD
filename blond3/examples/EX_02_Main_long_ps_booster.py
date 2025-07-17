# pragma: no cover

# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example script to take into account intensity effects from impedance tables

:Authors: **Danilo Quartullo**
"""

import logging
import os

import numpy as np
from scipy.constants import m_p, c, e

from blond3 import (
    Beam,
    proton,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    DriftSimple,
    ConstantMagneticCycle,
    WakeField,
    StaticProfile,
    BiGaussian,
)
from blond3.physics.impedances.readers import (
    ExampleImpedanceReader1,
    ExampleImpedanceReader2,
)
from blond3.physics.impedances.sources import (
    ImpedanceTableFreq,
    InductiveImpedance,
)
from blond3.physics.impedances.solvers import (
    PeriodicFreqSolver,
    InductiveImpedanceSolver,
)

logging.basicConfig(level=logging.INFO)

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

E_0 = m_p * c**2 / e  # [eV]
tot_beam_energy = E_0 + 1.4e9  # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV / c]

ring = Ring(circumference=(2 * np.pi * 25))
energy_cycle = ConstantMagneticCycle(
    sync_momentum,
    reference_particle=proton,
)
cavity1 = SingleHarmonicCavity()
cavity1.harmonic = 1
cavity1.voltage = 8e3
cavity1.phi_rf = np.pi

drift = DriftSimple(
    share_of_circumference=1.0,
)
drift.transition_gamma = 4.4
beam1 = Beam(n_particles=1e11, particle_type=proton)
profile1 = StaticProfile(
    cut_left=-5.72984173562e-7, cut_right=5.72984173562e-7, n_bins=10_000
)
wakefield1 = WakeField(
    sources=(
        ImpedanceTableFreq.from_file(
            this_directory + "/resources/EX_02_Ekicker_1.4GeV.txt",
            ExampleImpedanceReader1(),
        ),
        ImpedanceTableFreq.from_file(
            this_directory + "/resources/EX_02_Finemet.txt",
            ExampleImpedanceReader2(),
        ),
        InductiveImpedance(34.6669349520904 / 10e9),
        InductiveImpedance(34.6669349520904 / 10e9),
    ),
    solver=PeriodicFreqSolver(t_periodicity=1 / 2e5),
)
wakefield2 = WakeField(
    sources=(InductiveImpedance(34.6669349520904 / 10e9),),
    solver=InductiveImpedanceSolver(),
)

sim = Simulation.from_locals(locals())
sim.prepare_beam(
    BiGaussian(sigma_dt=180e-9 / 4, reinsertion=False, seed=1, n_macroparticles=1001)
)
sim.run_simulation(n_turns=2)
