# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import mplhep
import numpy as np
from matplotlib import pyplot as plt

mplhep.style.use("CMS")

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.experimental.bch_matching.bch_matcher import (
    BCHMatcher,
)
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)

# # ------SPS
# charge_particles = 82.0
# circumference = 6911.5038  # Machine circumference [m]
# gamma_transition = 17.95142852  # Transition gamma
# p_s = 5e9 * charge_particles  # 17.07e9 * charge_particles
# momentum_compaction_factor_ = 1.0 / gamma_transition**2  # Momentum compaction array
# harmonic_number = 4653
# voltage1 = 10e6

# #-------LHC
order = 3

p_s = 450.0e9  # Synchronous momentum [eV]
harmonic_number = 35640  # Harmonic number
voltage1 = 1e8  # RF voltage, station 1 [eV]
voltage2 = 4e6  # RF voltage, station 2 [eV]
voltage3 = 2e6
phi_rf = 0  # Phase modulation/offset
momentum_compaction_factor_ = momentum_compaction_factor(
    transition_gamma=55.759505
)
circumference = 26588
n_turns = 1000

energy_cycle = ConstantMagneticCycle(
    value=p_s,
    reference_particle=proton,
)
ring = Ring(
    circumference=circumference,
)
beam = Beam(
    intensity=1.0e9,
    particle_type=proton,
)
observation = BeamObservationInRingElement(
    each_turn_i=1,
    section_index=0,
    n_turns=n_turns,
    folder="./",
)
one_turn_execution_order = (
    SingleHarmonicRFStation(
        harmonic=harmonic_number,
        phi_rf=0,
        voltage=voltage1,
        section_index=0,
    ),
    DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor_,
        orbit_length=circumference,
        section_index=0,
    ),
    observation,
)
ring.add_elements(one_turn_execution_order, reorder=False)


sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
beam.reference.total_energy = p_s

sim.prepare_beam(
    preparation_routine=BCHMatcher(
        simulation=sim,
        beam=beam,
        n_macroparticles=int(1e5),  # TODO handle int properly
        order=order,
        distribution="Gaussian",
        emittance=4e-9,
        time_window_limit=(0, 2.5e-9),
        energy_window_limit=(-2e9, 2e9),
    ),
    beam=beam,
)


sim.run_simulation(beams=[beam], n_turns=n_turns)


# plot
plt.title(f"order = {order}")
plt.scatter(observation.dts[0] * 1e9, observation.dEs[0], label="turn 0")
plt.scatter(
    observation.dts[-1] * 1e9 + 2, observation.dEs[-1], label="turn 100"
)
plt.legend()
plt.xlabel("Δt [ns]")
plt.ylabel("ΔE [eV]")
plt.show()

# plot
plt.title(f"order = {order}")
plt.hist(
    observation.dts[0] * 1e9,
    bins=np.linspace(0, 2.5, 100),
    label="turn 0",
    density=True,
)
plt.hist(
    observation.dts[-1] * 1e9,
    bins=np.linspace(0, 2.5, 100),
    label="turn 100",
    density=True,
)
plt.legend()
plt.xlabel("Δt [ns]")
plt.show()

dat = []
for i in range(100):
    dat.append(
        np.histogram(observation.dts[i * 1], bins=np.linspace(0, 2.5e-9, 100))[
            0
        ]
    )

plt.contourf(dat, levels=100, cmap="plasma")
plt.title(f"order = {order}")
plt.ylabel("Turn")
plt.xlabel("Δt [s]")
plt.show()
