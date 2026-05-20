# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

"""
Multi-turn wake effects in a PSB-like machine.

Compares single-turn (TimeDomainFftSolver) and multi-turn
(ContinuousMultiTurnTimeDomainSolver) induced voltages for a narrow-band
resonator.  Mirrors the legacy EX_17 multi-turn wake example.

Notes
-----
Authors:
Juan F. Esteban Mueller (legacy EX_17)
Simon Lauber
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import c, e, m_p

from blond import (
    AllowPlotting,
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.handle_results.observables import WakeFieldObservation
from blond.physics.impedances.solvers import (
    ContinuousMultiTurnTimeDomainSolver,
)

N_TURNS_MTW = 5  # turns included in the multi-turn window


def main():
    rest_energy = m_p * c**2 / e  # proton rest energy [eV]
    kin_energy = 1.4e9  # kinetic energy [eV]
    sync_momentum = np.sqrt(
        (rest_energy + kin_energy) ** 2 - rest_energy**2
    )  # [eV/c]

    circumference = 2 * np.pi * 25.0  # PSB circumference [m]

    solvers = [
        ("Single-turn (TimeDomainFft)", TimeDomainFftSolver()),
        (
            f"Multi-turn ({N_TURNS_MTW}T)",
            ContinuousMultiTurnTimeDomainSolver(N_TURNS_MTW),
        ),
    ]

    for label, solver in solvers:
        ring = Ring(circumference=circumference)
        energy_cycle = MagneticCyclePerTurn(
            reference_particle=proton,
            value_init=sync_momentum,
            values_after_turn=np.full(N_TURNS_MTW, sync_momentum),
            in_unit="momentum",
        )
        t_rev = energy_cycle.get_t_rev_init(circumference)

        profile = StaticProfile(cut_left=0, cut_right=t_rev, n_bins=2000)
        resonator = Resonators(
            shunt_impedances=5e3, center_frequencies=10e6, quality_factors=10
        )
        wakefield = WakeField(
            sources=(resonator,),
            solver=solver,
            profile=profile,
        )

        rf_station = SingleHarmonicRFStation(
            harmonic=1, voltage=8e3, phi_rf=np.pi
        )
        drift = DriftSimple(orbit_length=circumference)
        drift.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=4.4
        )

        ring.add_elements((drift, rf_station, wakefield), reorder=True)
        sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
        sim.print_one_turn_execution_order()

        beam = Beam(intensity=1e11, particle_type=proton)
        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=180e-9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=1001,
            ),
            beam=beam,
        )
        beam.dt.array_local += t_rev

        wake_obs = WakeFieldObservation(each_turn_i=1, wakefield=wakefield)

        sim.run_simulation(
            beams=(beam,), n_turns=N_TURNS_MTW, observe=wake_obs
        )

        with AllowPlotting():
            plt.figure()
            for i in range(wake_obs.induced_voltage.shape[0]):
                plt.plot(
                    (profile.hist_x + i * t_rev) * 1e9,
                    wake_obs.induced_voltage[i, :],
                    label=label if i == 0 else None,
                )
            plt.xlabel("Time [ns]")
            plt.ylabel("Induced voltage [V]")
            plt.title(
                "Multi-turn vs single-turn wake (PSB, narrow-band resonator)"
            )
            plt.legend()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
