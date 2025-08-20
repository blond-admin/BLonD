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
Example input for simulation with RF noise
No intensity effects

:Authors: **Helga Timko**
"""

import os

import numpy as np

from blond3 import (
    Beam,
    proton,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    DriftSimple,
    MagneticCyclePerTurn,
    BiGaussian,
    BoxLosses,
)
from blond3.cycles.noise_generators.vari_noise import VariNoise
from blond3.physics.losses import SeparatrixLosses
from blond3.physics.profiles import DynamicProfileConstNBins

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

ring = Ring(circumference=26658.883)
cavity1 = SingleHarmonicCavity()
cavity1.voltage = 6e6
cavity1.schedule(
    attribute="phi_rf",
    value=VariNoise().get_noise(n_turns=200),
    mode="per-turn",
)
cavity1.harmonic = 35640
energy_cycle = MagneticCyclePerTurn(
    value_init=450.0e9,
    values_after_turn=np.linspace(450.0e9, 450.0e9, 200),
    reference_particle=proton,
)
beam = Beam(
    n_particles=1.0e9,
    particle_type=proton,
)
profile = DynamicProfileConstNBins(n_bins=100)
# losses = BoxLosses(t_min=0, t_max=2.5e-9) # TODO implement
# losses2 = SeparatrixLosses()# TODO implement
drift = DriftSimple(
    transition_gamma=55.759505,
    orbit_length=ring.circumference,
)
sim = Simulation.from_locals(locals())
sim.print_one_turn_execution_order()
sim.prepare_beam(
    beam=beam,
    preparation_routine=BiGaussian(
        n_macroparticles=1001,
        sigma_dt=0.4e-9 / 4,
        sigma_dE=1e9,  # potentially very mismatched
        reinsertion=True,
        seed=1,
    ),
)
sim.run_simulation(
    beams=(beam,),
    n_turns=200,
)
