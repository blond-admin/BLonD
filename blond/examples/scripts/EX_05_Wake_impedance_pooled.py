# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

"""
SPS simulation with intensity effects in time and frequency domains.

The input beam has been cloned to show that the two methods are equivalent
(compare the two figure folders). Note that to create an exact clone of the
beam, the option seed=0 in the generation has been used. This script shows
also an example of how to use the class SliceMonitor (check the corresponding
h5 files).

Notes
-----
Authors:
Simon Lauber
Danilo Quartullo
"""

import time

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
)
from blond.experimental import PooledInterpolationKick
from blond.handle_results.helpers import callers_relative_path

backend.set_specials("cpp")
pooling = False


def main():
    sync_momentum = 25.92e9  # [eV / c]

    resonator_data = np.loadtxt(
        callers_relative_path(
            "resources/EX_05_new_HQ_table.txt",
            stacklevel=1,
        ),
        comments="!",
    )

    R_shunt = resonator_data[:, 2] * 10**6
    f_res = resonator_data[:, 0] * 10**9
    Q_factor = resonator_data[:, 1]

    for wake_solver in (TimeDomainFftSolver(),):
        ring = Ring(
            circumference=6911.56,
        )
        magnetic_cycle = ConstantMagneticCycle(
            reference_particle=proton,
            value=sync_momentum,
            in_unit="momentum",
        )
        pooled_kick = PooledInterpolationKick()

        beam = Beam(
            intensity=1e10,
            particle_type=proton,
        )
        drift = DriftSimple(
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=22.82177322938192
            ),
            orbit_length=1.0 * ring.circumference,
        )
        profile = StaticProfile.from_rad(
            0,
            2 * np.pi,
            2**8,
            magnetic_cycle.get_t_rev_init(
                ring.circumference,
                particle_type=proton,
            )
            / 4620,
        )

        rf_station = SingleHarmonicRFStation(
            harmonic=4620,
            voltage=0.9e6,
            phi_rf=0.0,
            delayed_kick=pooled_kick if pooling else None,
            delayed_kick_time_axis=profile.hist_x if pooling else None,
        )

        wakefield = WakeField(
            sources=(Resonators(R_shunt, f_res, Q_factor),),
            solver=wake_solver,
            profile=profile,
            delayed_kick=pooled_kick if pooling else None,
        )

        ring.add_elements(
            (drift, rf_station, wakefield),
            reorder=True,
        )
        if pooling:
            ring.add_element(pooled_kick)
        sim = Simulation(
            ring=ring,
            magnetic_cycle=magnetic_cycle,
        )
        sim.prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=2e-9 / 4,
                seed=1,
                n_macroparticles=(5 * 1e6),
            ),
            beam=beam,
        )
        # sim.profiling(beams=beam, profile_n_turns=1000, sortby=SortKey.TIME)
        t0 = time.time()
        sim.run_simulation(beams=beam, n_turns=1000)

        print(f"{pooling=}", time.time() - t0, "s")


if __name__ == "__main__":  # pragma: no cover
    for b in (True, False):
        pooling = b
        main()
    plt.show()
