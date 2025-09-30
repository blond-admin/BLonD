# pragma: no cover
#
# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Example input for simulating a ring with multiple RF stations
No intensity effects

:Authors: **Helga Timko**
"""

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    proton,
)
from blond.physics.profiles import DynamicProfileConstNBins


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
        n_particles=1.0e9,
        particle_type=proton,
    )
    profile = DynamicProfileConstNBins(
        n_bins=100,
    )
    one_turn_execution_order = (
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=0.3 * ring.circumference,
            section_index=0,
        ),
        SingleHarmonicCavity(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage1,
            section_index=0,
        ),
        DriftSimple(
            transition_gamma=transition_gamma,
            orbit_length=0.7 * ring.circumference,
            section_index=1,
        ),
        SingleHarmonicCavity(
            harmonic=harmonic_number,
            phi_rf=phi_rf,
            voltage=voltage2,
            section_index=1,
        ),
        # TODO activate when losses are implemented
        # BoxLosses(
        #     t_min=0,
        #     t_max=2.5e-9,
        # ),
        # SeparatrixLosses(),
        profile,
    )
    ring.add_elements(one_turn_execution_order, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,
            reinsertion=True,
            seed=1,
            n_macroparticles=10001,
        ),
        beam=beam,
    )
    # profile_observable = DynamicProfileObservation(each_turn_i=10,
    # profile=profile)
    sim.run_simulation(
        n_turns=2000,
        # observe=(profile_observable,),
        beams=(beam,),
    )
    #############################################
    # plt.plot(profile_observable.turns_array, profile_observable.hist_y)


if __name__ == "__main__":  # pragma: no cover
    main()
