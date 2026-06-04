# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Downscaled muon collider example with a single beam, no intensity effects and reference energy changes in the drifts."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    DriftSimple,
    MagneticCycleByTime,
    ReferenceEnergyChange,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    mu_plus,
    setup_backend,
)
from blond.core.beam.particle_types import mu_minus
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.handle_results.observables import BeamStatisticsOncePerTurn
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")


def main():
    n_turns = 17
    # calculate parameters
    momentum_compaction_factor = 10.395e-4
    n_sections = 10
    voltage_per_section = 1500 * 30e6 / 32
    circumference = 953.338 * 2 * np.pi / 32
    time_per_turn = circumference / c

    # define energy ramp
    energy_ramp = np.linspace(63e9, (313.83e9 - 63e9) / 32 + 63e9, n_turns + 1)

    # initiate ring
    ring = Ring(circumference=circumference)

    energy_cycle = MagneticCycleByTime(
        reference_particle=mu_plus,
        reference_time=np.linspace(
            0, (n_turns + 1) * time_per_turn, n_turns + 1
        ),
        reference_values=energy_ramp,
        in_unit="total energy",
    )

    one_turn_model = []
    for rf_station_i in range(n_sections):
        rf_station = SingleHarmonicRFStation(
            section_index=rf_station_i,
        )
        rf_station.voltage = voltage_per_section
        rf_station.phi_rf_design = 0
        rf_station.harmonic = 25900

        one_turn_model.extend(
            [
                rf_station,
                DriftSimple(
                    momentum_compaction_factor=momentum_compaction_factor,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=rf_station_i,
                ),
                ReferenceEnergyChange(section_index=rf_station_i),
                DriftSimple(
                    momentum_compaction_factor=momentum_compaction_factor,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=rf_station_i,
                ),
                ReferenceEnergyChange(section_index=rf_station_i),
                DriftSimple(
                    momentum_compaction_factor=momentum_compaction_factor,
                    orbit_length=ring.circumference / n_sections / 3,
                    section_index=rf_station_i,
                ),
            ]
        )

    ring.add_elements(one_turn_model, reorder=False)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.print_one_turn_execution_order()

    beam = Beam(
        intensity=2.7e12,
        particle_type=mu_plus,
    )

    t_rf = sim.get_t_rev_init() / 25900

    sim.prepare_beam(
        beam=beam,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, t_rf),  # Time window around bunch [s]
            n_macroparticles=100_000,  # Number of macro-particles
            hamilton_to_density_kwargs=dict(
                density_modifier=2.0,  # Controls density profile sharpness
                hamilton_max=5000,  # Hamiltonian cutoff [eV]
            ),
            internal_grid_shape=(1023, 1023),  # Resolution of phase space grid
            seed=42,  # For reproducibility
            verbose=True,  # Print convergence info
            animate=False,  # Set True for live plotting
            until_section_index=1,
        ),
    )
    if not pytest_active():  # pragma: no cover
        plt.show()

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
        if not pytest_active():
            beam.plot_hist2d()
            plt.draw()
            plt.pause(0.5)
            plt.clf()

    beam_cr = deepcopy(beam)
    beam_cr.reference._particle_type = mu_minus
    beam_cr._is_counter_rotating = True
    beam_obs = BeamStatisticsOncePerTurn(each_turn_i=1)

    sim.run_simulation(
        beams=(beam, beam_cr),
        n_turns=n_turns,
        callbacks=my_callback,
        observe=beam_obs,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
