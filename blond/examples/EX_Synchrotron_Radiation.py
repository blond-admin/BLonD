# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    positron,
)
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.handle_results.observables import BeamStatisticsOncePerTurn
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    SynchrotronRadiationMaster,
)

logging.basicConfig(level=logging.INFO)


class SynchrotronRadiationSimulation:
    def __init__(
        self,
        n_turns: int,
    ):
        self.radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.circumference = 90.65874532 * 1e3
        self.momentum_compaction_factor = (
            self.radiation_integrals[0] / self.circumference
        )
        self.reference_energy = 20e9

        self.cavity = SingleHarmonicRFStation()
        self.cavity.harmonic = 242400
        self.cavity.voltage = 50.1e6
        self.cavity.phi_rf_design = 0

        self.n_turns = n_turns
        self.energy_cycle = MagneticCyclePerTurn(
            value_init=self.reference_energy,
            values_after_turn=np.linspace(
                self.reference_energy, self.reference_energy, self.n_turns
            ),
            reference_particle=positron,
            in_unit="total energy",
        )

        self.ring = Ring(
            self.circumference,
            radiation_integrals=self.radiation_integrals,
        )
        self.ring.add_element(self.cavity)

        # checks for rfcavity in each section prevents raising the following
        # number
        number_of_sections = 1
        for i in range(number_of_sections):
            drift = DriftSimple(
                name=f"drift{i + 1}",
                orbit_length=self.circumference / number_of_sections,
                momentum_compaction_factor=self.momentum_compaction_factor,
                section_index=i,
            )
            self.ring.add_element(
                drift,
                section_index=i,
            )

        self.SRHandler = SynchrotronRadiationMaster(
            # track_before_element_type = DriftBaseClass,
        )
        self.SRHandler.prepare_ring_for_synchrotron_radiation_tracking(
            ring=self.ring
        )
        self.ring.elements.print_order()

        beam = Beam(
            intensity=2.725e10,
            particle_type=positron,
        )

        self.beam = beam
        self.four_times_rms_bunch_length = 50 * 4e-3 / c
        self.energy_spread = 1e-3


def main(n_turns: int = 50000):
    params = SynchrotronRadiationSimulation(n_turns=n_turns)
    simulation = Simulation(
        ring=params.ring,
        magnetic_cycle=params.energy_cycle,
    )
    simulation.print_one_turn_execution_order()

    simulation.prepare_beam(
        beam=params.beam,
        preparation_routine=BiGaussian(
            sigma_dt=params.four_times_rms_bunch_length,
            sigma_dE=params.energy_spread * params.reference_energy,
            reinsertion=False,
            seed=1,
            n_macroparticles=1e5,
        ),
    )

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=params.cavity,
    )
    bunch_statistics = BeamStatisticsOncePerTurn(
        each_turn_i=1,
    )

    def custom_action(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_i.value is None or simulation.turn_i.value % 1 != 0:
            return

        artist = beam.plot_hist2d()
        plt.xlim([0, 1.5 * 1e-9])
        plt.ylim([-0.5 * 1e9, 0.5 * 1e9])
        plt.ylabel("DE [eV]")
        plt.xlabel("t [s]")
        plt.draw()
        plt.pause(1e-1)
        artist.remove()

    # custom_action(simulation, beam=params.beam)

    simulation.run_simulation(
        beams=(params.beam,),
        n_turns=params.n_turns,
        observe=(phase_observation, bunch_statistics),
        # callbacks=custom_action,
    )

    energy_loss_per_turn, damping_time, natural_energy_spread = (
        gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=params.beam.particle_type,
            energy=params.beam.reference.total_energy,
            radiation_integrals=params.radiation_integrals,
        )
    )
    fig, ax = plt.subplots(nrows=2, figsize=(8, 6), constrained_layout=True)
    synchronous_phase = np.pi - np.arcsin(
        energy_loss_per_turn / params.cavity.voltage
    )
    ax[0].plot(bunch_statistics.bunch_position * 1e9, label="Bunch position")
    ax[0].plot(
        synchronous_phase
        / params.cavity.omega_rf_design
        * 1e9
        * np.ones(len(bunch_statistics.bunch_position)),
        label="Synchronous position",
    )
    ax[0].set_xlabel("Turn number")
    ax[0].set_ylabel("Bunch position [ns]")
    ax[0].legend()

    ax[1].plot(
        bunch_statistics.energy_spread / 20e9 * 100,
        label="Energy spread evolution",
    )
    ax[1].plot(
        natural_energy_spread
        * np.ones(len(bunch_statistics.bunch_position))
        * 100,
        "r--",
        label="Natural energy spread",
    )
    ax[1].set_xlabel("Turn number")
    ax[1].set_ylabel("Energy spread [%]")
    ax[1].legend()
    os.makedirs("results/EX_Synchrotron_Radiation/", exist_ok=True)
    plt.savefig("results/EX_Synchrotron_Radiation/energy_spread_evolution.png")


if __name__ == "__main__":  # pragma: no cover
    main()
