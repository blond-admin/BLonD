# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)
from blond.handle_results.helpers import callers_relative_path
from blond.physics.drifts import DriftExact
from blond.testing.helpers import allclose_tolerances
from blond.utilities.separatrix.symbolic_serapartix import (
    SymbolicSeparatrixHelper,
)


class TestSymbolicSeparatrixHelper:
    def test_integration(self):
        DEV_DRAW = False
        ring = Ring(26658.883)

        rf_station1 = MultiHarmonicRFStation(
            section_index=0, n_harmonics=2, main_harmonic_idx=0
        )
        rf_station1.harmonic = np.array([35640, 4 * 35640])
        rf_station1.voltage = np.array([6e6, 6e6 / 2])
        rf_station1.phi_rf_design = np.array([0, 0])
        N_TURNS = int(1e3)

        energy_cycle = MagneticCyclePerTurn.init_from_linspace(
            values=np.linspace(450e9, 450e9, N_TURNS + 1),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            section_index=0,
            orbit_length=ring.circumference / 2,
        )
        drift1.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=55.759505
        )

        drift2 = DriftExact(
            orbit_length=ring.circumference / 2,
            section_index=1,
            momentum_compaction_factor=drift1.momentum_compaction_factor,
            higher_order_alpha=np.array(
                [drift1.alpha_0 * 2, drift1.alpha_0 * (-3)]
            ),
        )

        rf_station2 = SingleHarmonicRFStation(
            section_index=1,
            harmonic=35640,
            voltage=6e6,
            phi_rf=np.deg2rad(20),
        )

        ring.add_elements(
            (drift1, rf_station1), deepcopy=True, section_index=0
        )
        ring.add_elements(
            (drift2, rf_station2), deepcopy=True, section_index=1
        )

        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
        sim.print_one_turn_execution_order()
        t_rf = sim.get_t_rev_init() / 35640

        beam1 = Beam.simple_gaussian(
            n_macroparticles=1e5,
            intensity=1e9,
            particle_type=proton,
            dt_scale=0.4e-9 / 4,
            dE_scale=1e9 / 2,
            dt_offset=t_rf / 2,
            seed=1,
        )
        t0 = beam1.dt.min()
        t1 = beam1.dt.max()
        r = t1 - t0
        trange0_ = (t0 - 2 * r, t1 + 2 * r)
        plt.figure("Dynamic beam")
        plt.xlim(trange0_)

        def custom_action(
            simulation: Simulation, beam: Beam
        ):  # pragma: no cover
            plt.figure("Dynamic beam")
            if simulation.turn_i.value % 10 != 0:
                return

            dt = beam.read_partial_dt()
            plt.scatter(
                dt,
                beam.read_partial_dE(),
                s=1,
            )
            separatrix_dE = SymbolicSeparatrixHelper.from_simulation(
                simulation=sim
            ).get_separatrix(
                beam=beam,
                dt=np.linspace(*trange0_, 1000),
            )
            if simulation.turn_i.value == 0:
                separatrix_dE_pinned = np.loadtxt(
                    callers_relative_path(
                        "resources/separatrix_dE_pinned.txt", stacklevel=1
                    ),
                )
                np.testing.assert_allclose(
                    separatrix_dE,
                    separatrix_dE_pinned,
                    **allclose_tolerances(separatrix_dE_pinned),
                )
            if DEV_DRAW:
                sim.plot_separatrix(
                    beam=beam,
                    dt=np.linspace(*trange0_, 1000),
                )
                plt.xlim(trange0_)
                plt.ylim(-2e9, 2e9)

                plt.draw()
                plt.pause(0.1)
                plt.cla()

        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS if DEV_DRAW else 1,
            callbacks=custom_action,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
