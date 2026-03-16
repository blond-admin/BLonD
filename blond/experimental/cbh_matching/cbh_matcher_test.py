# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from matplotlib import pyplot as plt

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
from blond.experimental.cbh_matching.cbh_matcher import (
    CBHMatcher,
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
n_turns = 100

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
    observation,
    DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor_,
        orbit_length=circumference,
        section_index=0,
    ),
    SingleHarmonicRFStation(
        harmonic=harmonic_number,
        phi_rf=0,
        voltage=voltage1,
        section_index=0,
    ),
)
ring.add_elements(one_turn_execution_order, reorder=False)


sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
beam.reference.total_energy = p_s

sim.prepare_beam(
    preparation_routine=CBHMatcher(
        simulation=sim,
        beam=beam,
        n_macroparticles=10000,
        order=2,  # above order 3, takes a long time
        distribution="Gaussian",
    ),
    beam=beam,
)

sim.run_simulation(beams=[beam], n_turns=n_turns)
plt.scatter(observation.dts[0], observation.dEs[0])
# plt.scatter(observation.dts[1], observation.dEs[1])
plt.scatter(observation.dts[-1], observation.dEs[-1])
plt.show()
