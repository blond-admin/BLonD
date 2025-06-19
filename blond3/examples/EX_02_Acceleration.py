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
    EnergyCycle,
    ConstantProgram,
    WakeField,
    StaticProfile,
)
from blond3.physics.impedances.sources import (
    ImpedanceTableFreq,
    ExampleImpedanceReader1,
    ExampleImpedanceReader2,
    InductiveImpedance,
)
from blond3.physics.impedances.sovlers import (
    PeriodicFreqSolver,
    InductiveImpedanceSolver,
)

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

E_0 = m_p * c**2 / e  # [eV]
tot_beam_energy = E_0 + 1.4e9  # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV / c]

ring = Ring(circumference=(2 * np.pi * 25))
energy_cycle = EnergyCycle(beam_energy_by_turn=sync_momentum * np.ones(3))
cavity1 = SingleHarmonicCavity(
    harmonic=1,
    rf_program=ConstantProgram(effective_voltage=8e3, phase=np.pi),
)
drift = DriftSimple(
    transition_gamma=4.4,
    share_of_circumference=1.0,
)

beam1 = Beam(n_particles=1e11, n_macroparticles=1001, particle_type=proton)
profile1 = StaticProfile(
    cut_left=-5.72984173562e-7, cut_right=5.72984173562e-7, n_bins=10000
)
wakefield1 = WakeField(
    sources=(
        ImpedanceTableFreq.from_file(
            this_directory + "../input_files/EX_02_Ekicker_1.4GeV.txt",
            ExampleImpedanceReader1(),
        ),
        ImpedanceTableFreq.from_file(
            this_directory + "../input_files/EX_02_Finemet.txt",
            ExampleImpedanceReader2(),
        ),
        InductiveImpedance(34.6669349520904 / 10e9),
        InductiveImpedance(34.6669349520904 / 10e9),
    ),
    solver=PeriodicFreqSolver(t_periodicity=12.0),
)
wakefield2 = WakeField(
    sources=(InductiveImpedance(34.6669349520904 / 10e9),),
    solver=InductiveImpedanceSolver(),
)

sim = Simulation(ring=ring, ring_attributes=locals())
sim.run_simulation(n_turns=2)
