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

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
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
from blond.physics.drifts import DriftExact

# mplhep.style.use("CMS")


order = 3
# LHC
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
n_turns = 2000

energy_cycle = MagneticCyclePerTurn.init_from_linspace(
    values=np.linspace(p_s, p_s * 1.05, n_turns + 1),
    reference_particle=proton,
    in_unit="momentum",
)

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
    # DriftSimple(
    #     momentum_compaction_factor=momentum_compaction_factor_,
    #     orbit_length=circumference,
    #     section_index=0,
    # ),
    DriftExact(
        momentum_compaction_factor=momentum_compaction_factor_,
        higher_order_alpha=[momentum_compaction_factor_ * 100],
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
        emittance=1e-9,
        time_window_limit=(0.0e-9, 2.5e-9),
        energy_window_limit=(-2e9, 2e9),
    ),
    beam=beam,
)


sim.run_simulation(beams=[beam], n_turns=n_turns)

plt.title(f"order = {order}")
plt.scatter(observation.dts[0] * 1e9, observation.dEs[0], label="turn 0")
plt.scatter(observation.dts[-1] * 1e9, observation.dEs[-1], label="turn 100")
plt.legend()
plt.xlabel("Δt [ns]")
plt.ylabel("ΔE [eV]")
plt.show()

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
    alpha=0.5,
)
plt.legend()
plt.xlabel("Δt [ns]")
plt.show()

plt.clf()
plt.title(f"order = {order}")

t_min, t_max = 0.5e-9, 2.0e-9

for i in range(n_turns):
    dt = observation.dts[i]

    mask = (dt > 0) & (dt < 2.5e-9)  # your window
    dt_masked = dt[mask]

    sigma_t = np.std(dt_masked)
    plt.scatter(i, sigma_t, color="r", s=4)
plt.ylim([1.8e-10, 2e-10])
plt.xlabel("Turn")
plt.ylabel("RMS Δt")
plt.tight_layout()
plt.show()
