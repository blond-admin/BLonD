# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Example of user interfacing with BLonD."""

# pragma: no cover

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    DriftSimple,
    InductiveImpedance,
    InductiveImpedanceSolver,
    MagneticCyclePerTurn,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    StaticProfile,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCycleBase


class Main:
    """Helper class to describe a simulation."""

    @staticmethod
    def describe_accelerator() -> tuple[Ring, MagneticCyclePerTurn, Beam]:
        """
        Describe the hardware that is simulated within the :class:`blond.core.ring.ring.Ring`.

        Returns
        -------
        my_ring
            Ring object representing the synchrotron.
        my_cycle
            Magnetic cycle describing energy evolution.
        my_beam
            Beam object for the simulation.
        """
        # Description of accelerator
        my_ring = Ring(circumference=6912)

        profile1 = StaticProfile(cut_left=0, cut_right=1, n_bins=128)
        rf_station = MultiHarmonicRFStation(
            voltage=backend.array([6e6, 2e6]),
            phi_rf=backend.array([0, 0]),
            harmonic=backend.array([4620, 4 * 4620]),
            n_harmonics=2,
            main_harmonic_idx=0,
        )
        one_turn_execution_order = (
            DriftSimple(
                orbit_length=1.0 * my_ring.circumference,
                momentum_compaction_factor=momentum_compaction_factor(
                    transition_gamma=21
                ),
            ),
            rf_station,
            WakeField(
                sources=(InductiveImpedance(34.6669349520904 / 10e9),),
                solver=InductiveImpedanceSolver(),
            ),
            profile1,
            # LocalFeedback(rf_station, profile1),
            # GlobalFeedback(profile1),
            # DriftXSuite(orbit_length=0.1 * my_ring.circumference), # TODO
        )

        my_cycle = MagneticCyclePerTurn(
            reference_particle=proton,
            values_after_turn=np.linspace(25e9, 30e9, 110),
            value_init=25e9,
        )

        my_beam = Beam(
            intensity=1e6,
            particle_type=proton,
        )

        my_ring.add_elements(one_turn_execution_order, reorder=False)

        return my_ring, my_cycle, my_beam

    @staticmethod
    def ready_simulation_and_beam(
        my_ring: Ring,
        my_cycle: MagneticCycleBase,
        my_beam: Beam,
    ) -> tuple:
        """
        Assemble the :class:`blond.core.simulation.simulation.Simulation` object and match the beam.

        Parameters
        ----------
        my_ring
            `Ring` a.k.a. synchrotron.
        my_cycle
            Container object to handle the scheduled energy gain
            per turn or by time.
        my_beam
            Simulation `Beam` object.

        Returns
        -------
        simulation
            `Simulation` object.
        my_beam
            `Beam` object, matched.
        """
        # Preparation of simulation
        # Here everything might be interconnected
        simulation = Simulation(ring=my_ring, magnetic_cycle=my_cycle)
        # Already minor simulation of single turn
        simulation.prepare_beam(
            preparation_routine=BiGaussian(
                n_macroparticles=100,
                sigma_dt=1e-9,
                sigma_dE=1e9,
            ),
            beam=my_beam,
        )
        return simulation, my_beam

    @staticmethod
    def run_simulation(
        simulation: Simulation,
        my_beam: Beam,
    ) -> None:
        """
        Run the simulation.

        Parameters
        ----------
        simulation
            `Simulation` object.
        my_beam
            `Beam` object, matched.
        """
        # Full simulation. everything here should be optimized
        simulation.run_simulation(n_turns=100, beams=(my_beam,))


def main() -> None:
    """Execute the predefined simulation."""
    my_ring, my_cycle, my_beam = Main.describe_accelerator()
    simulation, my_beam = Main.ready_simulation_and_beam(
        my_ring=my_ring,
        my_cycle=my_cycle,
        my_beam=my_beam,
    )
    Main.run_simulation(
        simulation=simulation,
        my_beam=my_beam,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
