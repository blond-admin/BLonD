# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

from typing import Callable  # NOQA
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

from blond import SingleHarmonicRFStation  # NOQA
from blond import (  # NOQA
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    StaticProfile,
    WakeField,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, backend
from blond.experimental.beam_preparation.bucket_filler_functions import (
    multibunch_match_metric_to_hamilton,
)
from blond.experimental.beam_preparation.density_functions import (
    gaussian_density,
)
from blond.experimental.beam_preparation.metric_functions import rms_emittance
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.physics.impedances.solvers import PeriodicFreqSolver
from blond.physics.impedances.sources import Resonators


def bucket_fill_by_emittance_gaussian(
    time_grid: NumpyArray | CupyArray,
    deltaE_grid: NumpyArray | CupyArray,
    hamilton_2D: NumpyArray | CupyArray,
    emittance_list: list[float],
    intensity_frac_list: list[float],
    n_buckets: int,
    max_emittance_diff: float = 0.01,
) -> NumpyArray | CupyArray:
    """
    Method for generating gaussian distributed bunches of specific RMS emittances.

    Notes
        Makes use of the generalized bucket filler method
        This is just an example of how the general method can be utilized.

        Parameters
        ----------
        time_grid
            Time coordinates of the hamiltonian, in [s].
        deltaE_grid
            Energy coordinate of the hamiltonian, in [eV].
        hamilton_2D
            Hamiltonian value, in [eV].
        emittance_list
            List of (rms) emittances for each bunch.
        intensity_frac_list
            Fraction of total intensity for each bunch to have.
        n_buckets
            Number of buckets spanned by the time axis.
        max_emittance_diff
            The maximum allowed difference between the desired
            emittance and the generated emittance, in [eVs].

        Returns
        -------
        _density
            2D array containing the density distribution of the beam.
    """

    _density = multibunch_match_metric_to_hamilton(
        time_grid,
        deltaE_grid,
        hamilton_2D,
        emittance_list,
        intensity_frac_list,
        n_buckets,
        max_metric_diff=max_emittance_diff,
        density_function=gaussian_density,
        metric_function=rms_emittance,
        free_parameter_guess=None,
        max_iterations=100,
    )

    return _density


def main():
    # Defining a 1 turn simulation with values typical of the Proton Synchrotron
    PHI_RF = np.array(2 * [0])
    GAMMA_T = np.array(2 * [6.2])
    MOMENTUM = np.array([7e9, (7 + 0.0001) * 1e9])
    CIRCUMFERENCE = 2 * np.pi * 100
    VOLTAGE = 200e3
    HARMONIC = 8
    INTENSITY = 1e13

    # Simulation specific parameters
    N_MACROS = 1e5
    N_BINS = 1000

    # Defining Some resonator impedance
    Q = 3
    F_RES = 4e6  # Hz
    R_SH = 10000  # Ohms
    PROFILE_LENGTH = 2.1e-06

    # Set up simulation
    ring = Ring(circumference=CIRCUMFERENCE)
    energy_cycle = MagneticCyclePerTurn(
        value_init=float(MOMENTUM[0]),
        values_after_turn=MOMENTUM[1:].copy(),
        reference_particle=proton,
    )

    cavity = SingleHarmonicRFStation()
    cavity.harmonic = HARMONIC
    cavity.voltage = VOLTAGE
    cavity.schedule("phi_rf_design", PHI_RF[:-1].copy()[:])

    drift = DriftSimple(
        orbit_length=CIRCUMFERENCE,
    )

    drift.schedule("momentum_compaction_factor", 1 / GAMMA_T[1:].copy() ** 2)

    beam = Beam(intensity=INTENSITY, particle_type=proton)

    profile = StaticProfile(0, PROFILE_LENGTH, N_BINS)
    wakefield = WakeField(
        sources=(
            Resonators(np.array([R_SH]), np.array([F_RES]), np.array([Q])),
        ),
        solver=PeriodicFreqSolver(PROFILE_LENGTH, allow_next_fast_len=True),
        profile=profile,
    )

    ring.add_elements(
        (
            cavity,
            drift,
            profile,
            wakefield,
        ),
        reorder=False,
        section_index=0,
    )

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    # Prepare the beam. The hamilton_to_density_function is a very general function that can be user defined and just needs
    # To return a density distribution as a function of a hamiltonian. bucket_fill_by_emittance_gaussian is a specialized
    # function for this purpose with a few keyword arguments giving flexibility without the user having to define a custom
    # hamilton_to_density_function. bucket_fill_by_emittance_gaussian is a specific application of the even more general
    # multibunch_match_metric_to_hamilton function.

    sim.prepare_beam(
        beam=beam,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(3e-8, 3e-8 + 2.1e-6),
            n_macroparticles=1e6,
            hamilton_to_density_function=bucket_fill_by_emittance_gaussian,
            hamilton_to_density_kwargs={
                "emittance_list": [
                    0.1,
                    0,
                    0.2,
                    0,
                    0,
                    0.1,
                    0,
                    0.3,
                ],  # RMS emittance for each bunch
                "intensity_frac_list": [
                    0.3,
                    0,
                    0.2,
                    0,
                    0,
                    0.1,
                    0,
                    0.4,
                ],  # Intensity fraction of each bunch
                "n_buckets": 8,  # Number of buckets within time_limit
                "max_emittance_diff": 0.01,
            },  # Greatest tolerated difference in generated emittance
            animate=True,
            tolerance=1e-8,
            increment_intensity_effects_until_iteration_i=10,
        ),
    )

    # plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
