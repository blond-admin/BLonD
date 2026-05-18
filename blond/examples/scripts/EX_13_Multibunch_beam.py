# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

"""
Multi-bunch beam generation and tracking in the LHC.

Demonstrates make_multibunch_beam to replicate a single BiGaussian bunch
into an equidistant bunch train.  Mirrors the multi-bunch beam generation
workflow of the legacy EX_20 example.

Notes
-----
Authors:
Juan F. Esteban Mueller (legacy EX_20)
Simon Lauber
"""

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    AllowPlotting,
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    make_multibunch_beam,
    momentum_compaction_factor,
    proton,
)

N_BUNCHES = 4
BUCKET_SPACING = 10  # bunches separated by 10 RF buckets
N_TURNS = 10


def main():
    circumference = 26658.883  # LHC circumference [m]
    harmonic_number = 35640  # LHC harmonic number

    ring = Ring(circumference=circumference)
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=450e9,
        values_after_turn=np.full(N_TURNS, 450e9),
        in_unit="momentum",
    )
    t_rev = energy_cycle.get_t_rev_init(circumference)
    bucket_length = t_rev / harmonic_number
    bunch_spacing = BUCKET_SPACING * bucket_length

    rf_station = SingleHarmonicRFStation(
        harmonic=harmonic_number, voltage=6e6, phi_rf=0
    )
    drift = DriftSimple(orbit_length=circumference)
    drift.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )
    profile = StaticProfile(
        cut_left=0,
        cut_right=N_BUNCHES * bunch_spacing,
        n_bins=1024,
    )

    ring.add_elements((drift, rf_station, profile), reorder=True)
    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    single_bunch = Beam(intensity=1e11, particle_type=proton)
    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=0.4e-9 / 4,
            reinsertion=False,
            seed=1,
            n_macroparticles=1001,
        ),
        beam=single_bunch,
    )

    # Offset the train so the first bunch sits at the centre of its bucket group
    multibunch = make_multibunch_beam(
        beam=single_bunch,
        n_times=N_BUNCHES,
        t_distance=bunch_spacing,
        common_offset=bunch_spacing / 2,
    )
    print(
        f"Multibunch beam: {N_BUNCHES} bunches, "
        f"spacing = {BUCKET_SPACING} buckets ({bunch_spacing * 1e9:.2f} ns)"
    )

    sim.run_simulation(
        beams=(multibunch,),
        n_turns=N_TURNS,
    )

    with AllowPlotting():
        plt.figure()
        profile.plot(label=f"{N_BUNCHES}-bunch train")
        plt.xlabel("Time [s]")
        plt.ylabel("Macroparticles / bin")
        plt.title("Bunch train profile after simulation")
        plt.legend()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
