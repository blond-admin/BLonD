# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import logging

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    Beam,
    DriftSimple,
    EquidistantMultiProfile,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    momentum_compaction_factor,
    proton,
)

logging.basicConfig(level=logging.INFO)


def main():
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf_design = 0

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.momentum_compaction_factor = momentum_compaction_factor(55.759505)
    t_rev_init = energy_cycle.get_t_rev_init(ring.circumference)

    beam1 = Beam.simple_gaussian(
        n_macroparticles=int(1e5),
        dt_scale=0.4e-9 / 4,
        dE_scale=1e9 / 4,
        dt_offset=0.75 * t_rev_init,
        intensity=1e9,
        particle_type=proton,
    )
    profile_normal = StaticProfile(
        cut_left=0.5 * t_rev_init,
        cut_right=1.0 * t_rev_init,
        n_bins=512,
    )
    profile_sparse = EquidistantMultiProfile(
        filling_pattern=np.array([0, 1]),
        bins_per_profile=512,
    )
    ring.add_elements(
        (
            rf_station,
            drift1,
            profile_normal,
            profile_sparse,
        )
    )
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)
    sim.print_one_turn_execution_order()

    assert np.isclose(
        profile_normal.cut_left, profile_sparse.profiles[0].cut_left
    ), f"{(profile_normal.cut_left, profile_sparse.profiles[0].cut_left)}"
    assert np.isclose(
        profile_normal.cut_right, profile_sparse.profiles[0].cut_right
    ), f"{(profile_normal.cut_right, profile_sparse.profiles[0].cut_right)}"

    sim.run_simulation(
        beams=(beam1,),
        n_turns=1,
    )
    ax = plt.subplot(2, 1, 1)
    beam1.plot_scatter(label="particles", s=1)
    plt.legend()
    plt.subplot(2, 1, 2, sharex=ax)
    profile_normal.plot(label="profile_normal")
    profile_sparse.plot(linestyle="--", label="profile_sparse")
    plt.legend()
    assert (
        np.sum(profile_sparse._continuous_memory_hist_y)
        == beam1._dt.global_size
    ), f"""{
        (
            np.sum(profile_sparse._continuous_memory_hist_y),
            beam1._dt.global_size,
        )
    }"""

    assert (
        np.sum(profile_sparse.profiles[0].hist_y) == beam1._dt.global_size
    ), f"""{
        (
            np.sum(profile_sparse.profiles[0].hist_y),
            beam1._dt.global_size,
        )
    }"""
    assert np.allclose(
        profile_normal.hist_y, profile_sparse.profiles[0].hist_y
    ), f"{profile_normal.hist_y, profile_sparse.profiles[0].hist_y}"


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
