# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Match selected buckets of a ring with beam loading, one window per bucket.

PS-like flat top at 14 GeV/c on h=16 with the impedance of the 10 MHz
cavities (with their direct feedback). Only some buckets are filled, one of
them with a weaker pilot bunch. ``time_limit`` gets one window of
:math:`\\pm T_\\mathrm{rf}` around each bucket to fill: in every iteration
of the intensity effects, the matcher finds the bucket anew in its window,
so it follows the bucket while beam loading shifts it, and fills it up to
a fraction of its own separatrix.
"""

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    PeriodicFreqSolver,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

harmonic = 16
# buckets to fill, bucket k is centred at k * t_rf
filled_buckets = [1, 2, 3, 4, 5, 6, 9, 12]
pilot_bucket = 9  # half the intensity, filled less deep


def density_up_to_filling(
    time_grid,  # required, [s]
    deltaE_grid,  # required, [eV]
    hamilton_2D,  # required, [eV]
    filling,  # custom: fraction of the separatrix to fill
):
    r"""
    Parabolic density in H, up to ``filling`` times the separatrix.

    With windows, the grid is one bucket, cut out at its separatrix, and H
    is zero at the bottom of the bucket, so the separatrix is the lower of
    the two edges of the potential well :math:`H(t, \Delta E = 0)`.

    Parameters
    ----------
    time_grid
        Time coordinates of the grid, in [s]. Required by the matcher.
    deltaE_grid
        Energy coordinates of the grid, in [eV]. Required by the matcher.
    hamilton_2D
        Hamiltonian on the grid, in [eV]. Required by the matcher.
    filling
        Custom argument: fraction of the separatrix to fill.

    Returns
    -------
    density
        Density on the grid, zero above ``filling`` times the separatrix.
    """
    potential_well = hamilton_2D.min(axis=1)
    hamilton_separatrix = min(potential_well[0], potential_well[-1])
    return np.clip(1 - hamilton_2D / (filling * hamilton_separatrix), 0, 1)


def main(
    n_macroparticles: int = int(1e6),
    maxiter_intensity_effects: int = 100,
    internal_grid_shape: tuple[int, int] = (511, 511),
):
    circumference = 2 * np.pi * 100  # [m]
    intensity_per_bunch = 1.75e12  # [protons]
    rf_voltage = 100e3  # [V]
    # 10 MHz cavities, about 6 kOhm and Q=10 each, both divided by the
    # loop gain of the direct feedback (about 24 dB)
    n_cavities = 10
    feedback_reduction = 10 ** (24 / 20)

    ring = Ring(circumference=circumference)
    magnetic_cycle = ConstantMagneticCycle(
        value=14e9, reference_particle=proton
    )
    t_rev = magnetic_cycle.get_t_rev_init(circumference, proton)
    t_rf = t_rev / harmonic

    # above transition, phi_rf = pi centres bucket k at k * t_rf, so the
    # profile covers one turn from -t_rf / 2 on
    profile = StaticProfile(
        cut_left=-0.5 * t_rf,
        cut_right=t_rev - 0.5 * t_rf,
        n_bins=64 * harmonic,
    )
    ring.add_elements(
        [
            SingleHarmonicRFStation(
                harmonic=harmonic, voltage=rf_voltage, phi_rf=np.pi
            ),
            profile,
            WakeField(
                sources=(
                    Resonators(
                        n_cavities * 6e3 / feedback_reduction,
                        harmonic / t_rev,
                        10 / feedback_reduction,
                    ),
                ),
                solver=PeriodicFreqSolver(t_periodicity=t_rev),
                profile=profile,
            ),
            DriftSimple(
                orbit_length=circumference,
                momentum_compaction_factor=momentum_compaction_factor(6.1),
            ),
        ]
    )
    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

    bucket_weights = [
        0.5 if k == pilot_bucket else 1.0 for k in filled_buckets
    ]
    beam = Beam(
        intensity=intensity_per_bunch * sum(bucket_weights),
        particle_type=proton,
    )
    matcher = SemiEmpiricMatcher(
        # one window of +- t_rf around the nominal centre of each bucket
        time_limit=[((k - 1) * t_rf, (k + 1) * t_rf) for k in filled_buckets],
        n_macroparticles=n_macroparticles,
        hamilton_to_density_function=density_up_to_filling,
        # one dict per window, or one dict for all
        hamilton_to_density_kwargs=[
            {"filling": 0.4 if k == pilot_bucket else 0.8}
            for k in filled_buckets
        ],
        # relative intensity of each bunch
        bucket_weights=bucket_weights,
        internal_grid_shape=internal_grid_shape,
        maxiter_intensity_effects=maxiter_intensity_effects,
        increment_intensity_effects_until_iteration_i=min(
            10, maxiter_intensity_effects // 2
        ),
        tolerance_potential_well=1e-4,
        animate=True,
    )
    simulation.prepare_beam(beam=beam, preparation_routine=matcher)

    # where beam loading moved each bucket
    for k, (start, stop) in zip(
        filled_buckets, matcher.selected_buckets, strict=True
    ):
        shift = 0.5 * (start + stop) - k * t_rf
        print(
            f"bucket {k:2d}: centre shifted by {360 * shift / t_rf:+5.1f} deg"
        )

    if not pytest_active():  # pragma: no cover
        plt.figure("Matched beam")
        plt.hist(
            copy_to_cpu(beam.read_partial_dt()) / t_rf,
            bins=harmonic * 64,
            range=(-0.5, harmonic - 0.5),
        )
        plt.xlabel("Time (RF periods)")
        plt.ylabel("Macroparticles")
        plt.show()
    return beam, matcher, t_rf


if __name__ == "__main__":  # pragma: no cover
    main()
