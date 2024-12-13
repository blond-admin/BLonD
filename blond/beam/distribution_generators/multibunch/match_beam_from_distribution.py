"""
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import scipy
from packaging.version import Version

from .methods import match_a_bunch
from ...beam import Beam
from ...distribution_generators.singlebunch.methods import populate_bunch
from ....trackers.utilities import potential_well_cut
from ....utils import bmath as bm
from ....utils.legacy_support import handle_legacy_kwargs

if Version(scipy.__version__) >= Version("1.14"):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
else:
    from scipy.integrate import cumtrapz

if TYPE_CHECKING:
    from typing import Optional

    from ....input_parameters.ring import Ring
    from ...beam import Beam
    from ....impedances.impedance import TotalInducedVoltage
    from ....trackers.tracker import FullRingAndRF, MainHarmonicOptionType
    from ....utils.types import DistributionOptionsType


@handle_legacy_kwargs
def match_beam_from_distribution(beam: Beam, full_ring_and_rf: FullRingAndRF, ring: Ring,
                                 distribution_options: DistributionOptionsType,
                                 n_bunches: int,
                                 bunch_spacing_buckets: int,
                                 main_harmonic_option: MainHarmonicOptionType = 'lowest_freq',
                                 total_induced_voltage: Optional[TotalInducedVoltage] = None,
                                 n_iterations: int = 1,
                                 n_points_potential: int = int(1e4),
                                 dt_margin_percent: float = 0.40, seed: Optional[int] = None,
                                 ):
    """This generates n equally spaced bunches for a stationary distribution.


    *This function generates n equally spaced bunches for a stationary
    distribution and try to match them with intensity effects.*

    *The corresponding distributions are specified by their exponent:*

    .. math::
        g_0(J) \\sim (1-J/J_0)^{\\text{exponent}}}

    *Knowing the distribution, to generate the phase space:
    - Compute the potential U
    - The value of H can be computed thanks to U
    - The action J can be integrated over the whole phase space
    - 2piJ = emittance, this restricts the value of J0 (or H0)
    - with g0(H) we can randomize the macroparticles*

    Parameters
    ----------
    beam
        Class containing the beam properties.
    full_ring_and_rf
        Definition of the full ring and RF parameters in order to be able to have a full turn information
    ring
        Class containing the general properties of the synchrotron that are
        independent of the RF system or the beam.
    fit
        FitTableLineDensity or FitBunchLengthDistribution or list[FitTableLineDensity or list[FitBunchLengthDistribution]
    n_bunches
        # todo
    bunch_spacing_buckets
        # todo
    main_harmonic_option
        'lowest_freq', 'highest_voltage'
    total_induced_voltage
        # todo
    seed
        Random seed
    n_iterations
        # todo
    n_points_potential
        # todo
    dt_margin_percent
        # todo
    """
    # ------------------------------------------------------------------------
    # USEFUL VARIABLES
    # ------------------------------------------------------------------------
    # Slicing necessary only with intensity effects
    if total_induced_voltage is not None:
        profile = total_induced_voltage.profile

    # Ring information, Trev, energy, RF parameters ...
    rf_params = full_ring_and_rf.ring_and_rf_section[0].rf_params
    t_rev = rf_params.t_rev[0]
    n_rf = rf_params.n_rf
    beta = rf_params.beta[0]
    E = rf_params.energy[0]
    charge = rf_params.charge
    #    acceleration_kick = FullRingAndRF.ring_and_rf_section[0].acceleration_kick[0]

    # Minimum omega_rf is used to compute the size of the bucket
    omega_rf = []
    for i in range(n_rf):
        omega_rf += [rf_params.omega_rf[i][0]]
    omega_rf = np.array(omega_rf)

    eta_0 = rf_params.eta_0[0]

    # Coefficient of Kin and Pot part of the hamiltonian
    normalization_DeltaE = np.abs(eta_0) / (2. * beta ** 2 * E)
    normalization_potential = np.sign(eta_0) * charge / t_rev

    intensity_per_bunch = beam.intensity / n_bunches
    n_macro_per_bunch = int(beam.n_macroparticles / n_bunches)
    bucket_size_tau = 2 * np.pi / (np.min(omega_rf))

    # ------------------------------------------------------------------------
    # GENERATES N BUNCHES WITHOUT INTENSITY EFFECTS
    # ------------------------------------------------------------------------

    full_ring_and_rf.potential_well_generation(n_points=n_points_potential,
                                               dt_margin_percent=dt_margin_percent,
                                               main_harmonic_option=main_harmonic_option)

    # Restrict the potential well inside the separatrix and put min on 0
    potential_well_coordinates, potential_well = potential_well_cut(
        full_ring_and_rf.potential_well_coordinates,
        full_ring_and_rf.potential_well)
    potential_well = potential_well - np.min(potential_well)

    # Temporary beam, everything is done in the first bucket and then
    # shifted to plug into the real beam.
    temporary_beam = Beam(ring, n_macro_per_bunch, intensity_per_bunch)

    # Bunches placed in all the buckets without intensity effects
    # Loop the match function to have "different" bunches in each bucket
    matched_bunch_list = []
    for indexBunch in range(n_bunches):
        (time_grid, deltaE_grid, distribution, time_resolution,
         energy_resolution, single_profile) = match_a_bunch(
            normalization_DeltaE, temporary_beam,
            potential_well_coordinates,
            potential_well, seed, distribution_options,
            full_ring_and_rf=full_ring_and_rf)
        matched_bunch_list.append(
            (time_grid, deltaE_grid, distribution, time_resolution,
             energy_resolution, single_profile))

    print(str(n_bunches) + ' stationary bunches without intensity generated')
    # ------------------------------------------------------------------------
    # REMATCH THE BUNCHES WITH INTENSITY EFFECTS
    # ------------------------------------------------------------------------
    if total_induced_voltage is not None:
        print('Applying intensity effects ...')
        previous_well = potential_well
        for it in range(n_iterations):
            conv = 0.
            # Compute the induced voltage/potential for all the beam
            profile.n_macroparticles[:] = 0
            for indexBunch in range(n_bunches):
                profile.n_macroparticles += np.interp(
                    profile.bin_centers,
                    potential_well_coordinates +
                    indexBunch * bunch_spacing_buckets * bucket_size_tau,
                    matched_bunch_list[indexBunch][5],
                    left=0, right=0)
            profile.n_macroparticles[:] *= 1 / (np.sum(profile.n_macroparticles)) * beam.n_macroparticles

            total_induced_voltage.induced_voltage_sum()

            induced_voltage_coordinates = total_induced_voltage.time_array
            induced_voltage = total_induced_voltage.induced_voltage
            induced_potential = - normalization_potential * cumtrapz(
                induced_voltage,
                dx=float(induced_voltage_coordinates[1]
                         - induced_voltage_coordinates[0]),
                initial=0)

            for indexBunch in range(n_bunches):
                # Extract the induced potential for the specific bucket
                induced_potential_bunch = np.interp(potential_well_coordinates
                                                    + indexBunch * bunch_spacing_buckets * bucket_size_tau,
                                                    induced_voltage_coordinates, induced_potential)

                distorted_pot_well = potential_well + induced_potential_bunch
                distorted_pot_well -= np.min(distorted_pot_well)

                # Recompute the phase space distribution for the new
                # perturbed potential (containing induced_potential_bunch)
                matched_bunch_list[indexBunch] = match_a_bunch(
                    normalization_DeltaE, temporary_beam,
                    potential_well_coordinates,
                    distorted_pot_well, seed,
                    distribution_options,
                    full_ring_and_rf=full_ring_and_rf)

            conv = np.sqrt(np.sum((previous_well - distorted_pot_well) ** 2.)) / len(distorted_pot_well)
            previous_well = distorted_pot_well

            print('iteration ' + str(it + 1) + ', convergence parameter = ' + str(conv))

            profile.n_macroparticles[:] = 0
            for indexBunch in range(n_bunches):
                profile.n_macroparticles += np.interp(
                    profile.bin_centers,
                    potential_well_coordinates +
                    indexBunch * bunch_spacing_buckets * bucket_size_tau,
                    matched_bunch_list[indexBunch][5],
                    left=0, right=0)
            profile.n_macroparticles[:] *= 1 / (np.sum(profile.n_macroparticles)) * beam.n_macroparticles

            total_induced_voltage.induced_voltage_sum()

    for indexBunch in range(n_bunches):
        (time_grid, deltaE_grid, distribution, time_resolution,
         energy_resolution, single_profile) = matched_bunch_list[indexBunch]
        populate_bunch(temporary_beam, time_grid, deltaE_grid, distribution,
                       time_resolution, energy_resolution, seed)

        length_dt = len(temporary_beam.dt)
        length_dE = len(temporary_beam.dE)

        beam.dt[indexBunch * length_dt:(indexBunch + 1) * length_dt] = np.array(
            temporary_beam.dt) + (indexBunch * bunch_spacing_buckets * bucket_size_tau)
        beam.dE[indexBunch * length_dE:(indexBunch + 1) * length_dE] = np.array(
            temporary_beam.dE)

    beam.dt = beam.dt.astype(dtype=bm.precision.real_t, order='C', copy=False)
    beam.dE = beam.dE.astype(dtype=bm.precision.real_t, order='C', copy=False)
    gc.collect()
