# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Several setups of simulations that are intended for testcases.

Authors
-------
Simon Lauber
"""

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MagneticCyclePerTurn,
    MultiHarmonicRfStation,
    RfStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    StaticProfile,
    WakeField,
    backend,
    proton,
)
from blond.physics.impedances.solvers import (
    TimeDomainFftSolver,
)
from blond.physics.impedances.sources import Resonators


class ExampleSimulation01:
    """Simulation with only one drift, one RF."""

    def __init__(self):
        ring = Ring(circumference=26658.883)

        rf_station = SingleHarmonicRfStation()
        rf_station.harmonic = 35640
        rf_station.voltage = 6e6
        rf_station.phi_rf_design = 0

        N_TURNS = 10
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
            in_unit="momentum",
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.transition_gamma = 55.759505

        beam1 = Beam(intensity=1e9, particle_type=proton)
        self.beam1 = beam1

        simulation = Simulation.from_locals(locals())
        simulation.print_one_turn_execution_order()

        simulation.prepare_beam(
            beam=beam1,
            preparation_routine=BiGaussian(
                sigma_dt=0.4e-9 / 4,
                sigma_dE=1e9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=10,
            ),
            turn_i=10,
        )

        phase_observation = RfStationPhaseObservation(
            each_turn_i=1, rf_station=rf_station
        )

        # bunch_observation = BunchObservation(each_turn_i=10, batch_size=)
        # batches
        def my_callback(simulation: Simulation, beam: Beam):
            if simulation.turn_i.value % 10 != 0:
                return

            plt.scatter(
                beam.read_partial_dt(),
                beam.read_partial_dE(),
            )
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        self.simulation = simulation


class SimulationTwoRfStations:
    """
    A simulation with two RF stations and according drifts.

    Parameters
    ----------
    below_transition_crossing
        Whether the beam is below the transition crossing.
    """

    def __init__(self, below_transition_crossing: bool = False):
        circumference = 26658.883
        ring = Ring(circumference=circumference)

        rf_station_1 = MultiHarmonicRfStation(
            harmonic=np.array(
                [35640],
            ),
            voltage=np.array(
                [6e6],
            ),
            phi_rf=np.array(
                [0.0],
            ),
            section_index=0,
            n_harmonics=1,
            main_harmonic_idx=0,
        )

        rf_station_2 = SingleHarmonicRfStation(
            section_index=1,
        )
        rf_station_2.harmonic = backend.float(35640)
        rf_station_2.voltage = backend.float(6e6)
        rf_station_2.phi_rf_design = backend.float(0)

        N_TURNS = int(1e6)
        energy_cycle = ConstantMagneticCycle(
            value=450e9,
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=0.5 * circumference,
            section_index=0,
        )
        drift1.transition_gamma = (
            855.759505 if below_transition_crossing else 55.759505
        )
        drift2 = DriftSimple(
            orbit_length=0.5 * circumference,
            section_index=1,
        )
        drift2.transition_gamma = (
            855.759505 if below_transition_crossing else 55.759505
        )
        beam1 = Beam(
            intensity=1e9,
            particle_type=proton,
        )

        simulation = Simulation.from_locals(locals())

        self.simulation = simulation
        self.beam1 = beam1


class SimulationTwoRfStationsWithWake:
    """
    A simulation with two RF stations and according drifts, plus wake.

    Parameters
    ----------
    below_transition_crossing
        Whether the beam is below the transition crossing.
    """

    def __init__(self, below_transition_crossing: bool = False):
        circumference = 26658.883
        ring = Ring(circumference=circumference)

        rf_station_1 = MultiHarmonicRfStation(
            harmonic=np.array(
                [35640],
            ),
            voltage=np.array(
                [6e6],
            ),
            phi_rf=np.array(
                [0.0],
            ),
            section_index=0,
            n_harmonics=1,
            main_harmonic_idx=0,
        )

        rf_station_2 = SingleHarmonicRfStation(
            section_index=1,
        )
        rf_station_2.harmonic = 35640
        rf_station_2.voltage = 6e6
        rf_station_2.phi_rf_design = 0

        N_TURNS = int(1e6)
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(
                450e9,
                450e9,
                N_TURNS,
            ),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=0.5 * circumference,
            section_index=0,
        )
        drift1.transition_gamma = (
            855.759505 if below_transition_crossing else 55.759505
        )
        drift2 = DriftSimple(
            orbit_length=0.5 * circumference,
            section_index=1,
        )
        drift2.transition_gamma = (
            855.759505 if below_transition_crossing else 55.759505
        )
        beam1 = Beam(
            intensity=1e9,
            particle_type=proton,
        )
        t_rev = energy_cycle.get_t_rev_init(
            circumference=ring.circumference,
            turn_i_init=0,
            t_init=0,
            particle_type=beam1.particle_type,
        )

        wakefield = WakeField(
            sources=(
                Resonators(
                    shunt_impedances=[
                        1e12,
                    ],
                    center_frequencies=[
                        400e6,
                    ],
                    quality_factors=[
                        1e5,
                    ],
                ),
            ),
            solver=TimeDomainFftSolver(),
            profile=StaticProfile(
                cut_left=0,
                cut_right=t_rev / 36540,
                n_bins=512,
            ),
        )

        simulation = Simulation.from_locals(locals())

        self.simulation = simulation
        self.beam1 = beam1
