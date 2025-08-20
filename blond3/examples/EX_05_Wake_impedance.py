# pragma: no cover

# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
SPS simulation with intensity effects in time and frequency domains using
a table of resonators. The input beam has been cloned to show that the two
methods are equivalent (compare the two figure folders). Note that to create an
exact clone of the beam, the option seed=0 in the generation has been used.
This script shows also an example of how to use the class SliceMonitor (check
the corresponding h5 files).

:Authors: **Danilo Quartullo**
"""

import numpy as np
from matplotlib import pyplot as plt

from blond.impedances.induced_voltage_analytical import analytical_gaussian_resonator
from blond3 import (
    BiGaussian,
    Beam,
    proton,
    Ring,
    Simulation,
    MagneticCyclePerTurn,
    SingleHarmonicCavity,
    DriftSimple,
    BunchObservation,
    WakeField,
    StaticProfile,
)
from blond3.handle_results.helpers import callers_relative_path
from blond3.physics.impedances.solvers import (
    TimeDomainSolver,
    PeriodicFreqSolver,
)
from blond3.physics.impedances.sources import Resonators

sync_momentum = 25.92e9  # [eV / c]

resonator_data = np.loadtxt(
    callers_relative_path(
        "resources/EX_05_new_HQ_table.dat",
        stacklevel=1,
    ),
    comments="!",
)

R_shunt = resonator_data[:, 2] * 10**6
f_res = resonator_data[:, 0] * 10**9
Q_factor = resonator_data[:, 1]

for wake_solver in (
    TimeDomainSolver(),
    PeriodicFreqSolver(t_periodicity=None),  # todo
    # AnalyticSingleTurnResonatorSolver(), # todo implement
):
    ring = Ring(
        circumference=6911.56,
    )
    magnetic_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        values_after_turn=np.linspace(sync_momentum, sync_momentum, 2),
        value_init=sync_momentum,
        in_unit="momentum",
    )
    cavity1 = SingleHarmonicCavity(
        harmonic=4620,
        voltage=0.9e6,
        phi_rf=0.0,
    )
    beam = Beam(
        n_particles=1e10,
        particle_type=proton,
    )
    drift = DriftSimple(
        transition_gamma=22.82177322938192,
        orbit_length=1.0 * ring.circumference,
    )
    profile = StaticProfile.from_rad(
        0,
        2 * np.pi,
        2**8,
        magnetic_cycle.get_t_rev_init(
            ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=proton,
        )
        / 4620,
    )
    wakefield = WakeField(
        sources=(Resonators(R_shunt, f_res, Q_factor),),
        solver=wake_solver,
        profile=profile,
    )
    ring.add_elements(
        (drift, profile, cavity1, wakefield),
        reorder=True,
    )
    sim = Simulation(
        ring=ring,
        magnetic_cycle=magnetic_cycle,
    )
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=2e-9 / 4,
            seed=1,
            n_macroparticles=(5 * 1e6),
        ),
        beam=beam,
    )
    bunch_observable = BunchObservation(each_turn_i=10)
    sim.run_simulation(
        observe=(bunch_observable,),
        beams=(beam,),
        n_turns=1,
    )
    plt.subplot(2, 1, 1)
    plt.plot(
        profile.hist_x,
        profile.hist_y,
        label=f"{type(wake_solver).__name__}",
    )
    plt.subplot(2, 1, 2)
    plt.plot(
        profile.hist_x,
        wakefield.induced_voltage,
        label=f"{type(wake_solver).__name__}",
    )


# Analytic result-----------------------------------------------------------
VindGauss = np.zeros(len(profile.hist_x))
for r in range(len(Q_factor)):
    # Notice that the time-argument of inducedVoltageGauss is shifted by
    # mean(my_slices.hist_x), because the analytical equation assumes the
    # Gauss to be centered at t=0, but the line density is centered at
    # mean(my_slices.hist_x)
    tmp = analytical_gaussian_resonator(
        2e-9 / 4,
        Q_factor[r],
        R_shunt[r],
        2 * np.pi * f_res[r],
        profile.hist_x - np.mean(profile.hist_x),
        beam.n_particles,
    )
    VindGauss += tmp.real
plt.subplot(2, 1, 1)
plt.plot(profile.hist_x, profile.hist_y, label=f"analytical_gaussian_resonator")
plt.subplot(2, 1, 2)
plt.plot(profile.hist_x, VindGauss, label="analytical_gaussian_resonator")
plt.legend()
# plt.show()
