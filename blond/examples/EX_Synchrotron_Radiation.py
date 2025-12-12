# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    RfStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    electron,
    positron,
)
from blond.acc_math.analytic.synchrotron_radiation.utilities import (
    gather_longitudinal_synchrotron_radiation_parameters,
)
from blond.physics.synchrotron_radiation.synchrotron_radiation import (
    SynchrotronRadiationMaster,
)


class SynchrotronRadiationSimulation:
    def __init__(self):
        self.synchrotron_radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.circumference = 90.65874532 * 1e3
        self.momentum_compaction_factor = 0.646747216157 * 90.65874532 * 1e3
        self.reference_energy = 20e9
        self.cavity = SingleHarmonicRfStation()
        self.cavity.harmonic = 242400
        self.cavity.voltage = 51e6
        self.cavity.phi_rf = 0

        self.n_turns = int(100)
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
            # check_section_indices=False,
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
            self.ring.add_element(drift, section_index=i)

        self.SRHandler = SynchrotronRadiationMaster(
            name="SynchrotronRadiationMaster",
            radiation_integrals=self.synchrotron_radiation_integrals,
            # track_before_element_type = DriftBaseClass,
        )
        self.ring.insert_element(self.SRHandler, insert_at=0, deepcopy=False)
        beam = Beam(
            intensity=1e9,
            particle_type=electron,
        )

        self.beam = beam
        self.four_times_rms_bunch_length = 4 * 4e-3 / c
        self.energy_spread = 1e-3


def main():
    params = SynchrotronRadiationSimulation()

    params.ring.elements.print_order()
    simulation = Simulation(
        ring=params.ring, magnetic_cycle=params.energy_cycle
    )
    params.ring.elements.print_order()
    simulation.print_one_turn_execution_order()

    simulation.prepare_beam(
        beam=params.beam,
        preparation_routine=BiGaussian(
            sigma_dt=params.four_times_rms_bunch_length,
            sigma_dE=params.energy_spread * params.reference_energy,
            reinsertion=False,
            seed=1,
            n_macroparticles=1e3,
        ),
    )
    phase_observation = RfStationPhaseObservation(
        each_turn_i=1,
        rf_station=params.cavity,
    )
    bunch_observation = BeamObservationOncePerTurn(
        each_turn_i=1, beam=params.beam
    )

    simulation.run_simulation(
        beams=(params.beam,),
        turn_i_init=0,
        n_turns=params.n_turns,
        observe=(phase_observation, bunch_observation),
    )

    energy_loss_per_turn, damping_time, natural_energy_spread = (
        gather_longitudinal_synchrotron_radiation_parameters(
            particle_type=params.beam.particle_type,
            energy=params.beam.reference_total_energy,
            synchrotron_radiation_integrals=params.synchrotron_radiation_integrals,
        )
    )
    fig, ax = plt.subplots(nrows=2, figsize=(8, 6), constrained_layout=True)
    bunch_position_evolution = np.average(bunch_observation.dts, axis=1)
    energy_spread_evolution = (
        np.average(bunch_observation.dEs, axis=1)
        / params.beam.reference_total_energy
    )
    # ax[0].scatter(bunch_observation.dts[0, :],  bunch_observation.dEs[0, :],
    #     label="Bunch position")
    ax[0].plot(bunch_position_evolution, label="Bunch position")

    ax[1].plot(energy_spread_evolution, label="Energy spread evolution")
    ax[1].plot(
        natural_energy_spread * np.ones(len(energy_spread_evolution)),
        "r--",
        label="Energy spread evolution",
    )
    plt.show()

    ANIMATE = False
    if ANIMATE:  # pragma: no cover
        plt.plot(phase_observation.phases)
        plt.figure()
        for i in range(params.n_turns):
            # plt.clf()
            plt.hist2d(
                bunch_observation.dts[i, :],
                bunch_observation.dEs[i, :],
                # bins=256,
                range=[[0, 2.5 * 1e-9], [-0.1 * 1e9, 0.1 * 1e9]],
            )
            plt.draw()
            plt.pause(0.1)

        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
