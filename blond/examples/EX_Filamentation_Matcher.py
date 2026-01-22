# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example to show the BruteForceMatcher to find a quasi-matched distribution.
"""

from matplotlib import pyplot as plt

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.experimental.beam_preparation.filamentation_matcher import (
    FilamentationMatcher,
)
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)


def main():
    # Simulation parameters -------------------------------------------------------
    p_s = 450.0e9  # Synchronous momentum [eV]
    harmonic_number = 35640  # Harmonic number
    voltage1 = 2e6  # RF voltage, station 1 [eV]
    voltage2 = 4e6  # RF voltage, station 1 [eV]
    phi_rf = 0  # Phase modulation/offset
    transition_gamma = 55.759505  # Transition gamma

    energy_cycle = ConstantMagneticCycle(
        value=p_s,
        reference_particle=proton,
    )
    ring = Ring(
        circumference=26658.883,
    )
    beam = Beam(
        intensity=1.0e9,
        particle_type=proton,
    )

    observation = BeamObservationInRingElement(
        each_turn_i=1, section_index=0, n_turns=10, folder="./"
    )

    one_turn_execution_order = (
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=0.3 * ring.circumference,
            section_index=0,
        ),
        SingleHarmonicRFStation(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage1,
            section_index=0,
        ),
        observation,
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=0.7 * ring.circumference,
            section_index=1,
        ),
        SingleHarmonicRFStation(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage2,
            section_index=1,
        ),
    )
    ring.add_elements(one_turn_execution_order, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    sim.prepare_beam(
        preparation_routine=FilamentationMatcher(
            time_limit=[
                0.1e-9,
                4e-9,
            ],  # adjust this until the desired bunch is found
            energy_limit=[-4e8, 4e8],
            n_macroparticles=3000,
            n_iter=100,
            every_iter_to_plot=10,  # plot every 100/10 iterations
            animate=True,
            purge_limit_time=[0.1e-9, 4e-9],  #
            purge_limit_energy=[-4e8, 4e8],
            purge=True,
        ),
        beam=beam,
    )
    plt.show()

    sim.run_simulation(
        n_turns=20,
        beams=(beam,),
    )

    plt.close()
    plt.title("Injected beam into simulation")
    plt.scatter(observation.dts[0], observation.dEs[0], s=0.5)
    plt.scatter(observation.dts[0], observation.dEs[0], s=0.5)
    plt.xlabel("Energy [eV]")
    plt.ylabel("Time [s]")
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
