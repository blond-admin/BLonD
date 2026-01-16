# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from copy import deepcopy

import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCycleByTime,
    ReferenceEnergyChange,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    StaticProfileObservation,
    mu_plus,
)


def main():
    n_turns = 17
    # calculate parameters
    transition_gamma = 1 / np.sqrt(10.395e-4)
    n_sections = 1
    voltage_per_section = 865 * 30e6 / n_sections
    time_per_turn = 953.338 * 2 * np.pi / c

    # define energy ramp
    energy_ramp = np.linspace(63e9, 313.83e9 * 100, n_turns)
    phi_s = 135 * np.pi / 180

    # initiate ring
    ring = Ring(circumference=953.338 * 2 * np.pi)

    energy_cycle = MagneticCycleByTime(
        reference_particle=mu_plus,
        base_time=np.linspace(0, 18 * time_per_turn, n_turns),
        base_values=energy_ramp,
        in_unit="momentum",
    )

    one_turn_model = []
    observables = []
    for rf_station_i in range(n_sections):
        rf_station = SingleHarmonicRFStation(
            section_index=rf_station_i,
        )
        profile = StaticProfile(
            cut_left=0, cut_right=1, n_bins=256, section_index=rf_station_i
        )
        rf_station.voltage = voltage_per_section
        rf_station.phi_rf = phi_s
        rf_station.harmonic = 25900

        one_turn_model.extend(
            [
                rf_station,
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=rf_station_i,
                ),
                ReferenceEnergyChange(section_index=rf_station_i),
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=rf_station_i,
                ),
                ReferenceEnergyChange(section_index=rf_station_i),
                DriftSimple(
                    transition_gamma=transition_gamma,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=rf_station_i,
                ),
                profile,
            ]
        )
        observables.append(
            StaticProfileObservation(profile=profile, each_turn_i=2)
        )

    ring.add_elements(one_turn_model, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.print_one_turn_execution_order()

    beam1 = Beam(
        intensity=2.7e12,
        particle_type=mu_plus,
    )

    zmax = ring.circumference / (2 * 25900)  # maximum bunch length z

    total_rf_station = SingleHarmonicRFStation(
        section_index=rf_station_i,
    )

    total_rf_station.voltage = voltage_per_section
    total_rf_station.phi_rf = phi_s
    total_rf_station.harmonic = 25900

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=zmax / 43,
            reinsertion=False,
            seed=1,
            n_macroparticles=1e5,
        ),
    )

    def my_callback(
        simulation: Simulation, beam: Beam
    ) -> None:  # pragma: no cover
        """
        Empty callback example.

        Parameters
        ----------
        simulation
            Simulation context manager.
        beam
            Simulation `Beam` object.
        """
        pass

    beam2 = deepcopy(beam1)
    beam2._is_counter_rotating = True
    sim.run_simulation(
        beams=(beam1, beam2),
        n_turns=n_turns,
        callback=my_callback,  # not supported yet.
        observe=observables,
    )

    return


if __name__ == "__main__":
    main()  # pragma: no cover
