"""
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scipy
from packaging.version import Version

from .match_beam_from_distribution import match_beam_from_distribution
from .methods import match_a_bunch
from ...beam import Beam
from ....trackers.utilities import potential_well_cut
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
def match_beam_from_distribution_multibatch(beam: Beam, full_ring_and_rf: FullRingAndRF, ring: Ring,
                                            distribution_options: DistributionOptionsType, n_bunches: int,
                                            bunch_spacing_buckets: int,
                                            n_batch: int,
                                            batch_spacing_buckets: int,
                                            main_harmonic_option: MainHarmonicOptionType = 'lowest_freq',
                                            total_induced_voltage: Optional[TotalInducedVoltage] = None,
                                            n_iterations: int = 1,
                                            n_points_potential: int = int(1e4),
                                            dt_margin_percent: float = 0.40, seed: Optional[int] = None):
    """This function generates n equally spaced bunches for a stationary distribution.


    *This function generates n equally spaced bunches for a stationary
    distribution and try to match them with intensity effects.*

    *Then it copies the batch n_batch times with spacing batch_spacing_buckets*

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
    bunch_spacing_buckets
        # todo
    n_batch
        # todo
    batch_spacing_buckets
        # todo
    main_harmonic_option
        'lowest_freq', 'highest_voltage'
    total_induced_voltage
        # todo
    n_iterations
        # todo
    n_points_potential
        # todo
    dt_margin_percent
        # todo
    seed
        Random seed
    """
    # ------------------------------------------------------------------------
    # USEFUL VARIABLES
    # ------------------------------------------------------------------------
    # Ring informations, Trev, energy, RF parameters ...
    rf_params = full_ring_and_rf.ring_and_rf_section[0].rf_params
    n_rf = rf_params.n_rf
    # Slicing necessary only with intensity effects
    if total_induced_voltage is not None:
        profile = total_induced_voltage.profile

        t_rev = rf_params.t_rev[0]
        beta = rf_params.beta[0]
        E = rf_params.energy[0]
        charge = rf_params.charge
        eta_0 = rf_params.eta_0[0]

        normalization_DeltaE = np.abs(eta_0) / (2. * beta ** 2 * E)
        normalization_potential = np.sign(eta_0) * charge / t_rev

    # Ring informations, Trev, energy, RF parameters ...
    #    beta = rf_params.beta[0]
    #    E = rf_params.energy[0]
    #    charge = rf_params.charge
    #    acceleration_kick = FullRingAndRF.ring_and_rf_section[0].acceleration_kick[0]

    # Minimum omega_rf is used to compute the size of the bucket
    omega_rf = []
    for i in range(n_rf):
        omega_rf += [rf_params.omega_rf[i][0]]
    omega_rf = np.array(omega_rf)

    #    eta_0 = rf_params.eta_0[0]

    #    # Coefficient of Kin and Pot part of the hamiltonian
    #    normalization_DeltaE = np.abs(eta_0) / (2.*beta**2*E)
    #    normalization_potential = np.sign(eta_0)*charge/t_rev

    intensity_per_bunch = beam.intensity / n_bunches / n_batch
    n_macro_per_bunch = int(beam.n_macroparticles / n_bunches / n_batch)
    bucket_size_tau = 2 * np.pi / (np.min(omega_rf))

    temporary_batch = Beam(ring, int(n_macro_per_bunch * n_bunches), (intensity_per_bunch * n_bunches))

    #    print(temporary_batch.dt)
    match_beam_from_distribution(temporary_batch, full_ring_and_rf, ring,
                                 distribution_options, n_bunches, bunch_spacing_buckets,
                                 total_induced_voltage=None, n_iterations=n_iterations,
                                 n_points_potential=n_points_potential)

    #    matched_from_distribution_density_multibunch(temporary_batch, GeneralParameters, FullRingAndRF, distribution_options,
    #                                          n_bunches, bunch_spacing_buckets,
    #                                          TotalInducedVoltage = TotalInducedVoltage,
    #                                          n_iterations_input = n_iterations)
    length_dt = len(temporary_batch.dt)
    print(length_dt)
    for index_batch in range(n_batch):
        start = index_batch * length_dt
        stop = (index_batch + 1) * length_dt
        beam.dt[start:stop] = (temporary_batch.dt
                               + index_batch * (n_bunches - 1) * bunch_spacing_buckets * bucket_size_tau
                               + (index_batch) * batch_spacing_buckets * bucket_size_tau)
        beam.dE[start:stop] = temporary_batch.dE

    plt.figure('copymultibatch')
    plt.plot(beam.dt[::100], beam.dE[::100], 'b.')
    plt.figure('temporarybatch')
    plt.plot(temporary_batch.dt[::100], temporary_batch.dE[::100], 'b.')
    plt.figure('profile before induced voltage')
    profile.track()
    plt.plot(profile.bin_centers, profile.n_macroparticles)
    plt.figure('beamInSlice')
    plt.plot(profile.beam.dt[::100], profile.beam.dE[::100], 'b.')
    # ------------------------------------------------------------------------
    # REMATCH THE BUNCHES WITH INTENSITY EFFECTS
    # ------------------------------------------------------------------------
    if total_induced_voltage is not None:
        #        TotalInducedVoltage.profile.beam.dt[:len(beam.dt)] = beam.dt
        #        TotalInducedVoltage.profile.beam.dE[:len(beam.dE)] = beam.dE
        print('Applying intensity effects ...')
        for it in range(n_iterations):
            conv = 0.
            # Compute the induced voltage/potential for all the beam
            profile.track()
            total_induced_voltage.induced_voltage_sum()

            plt.figure('profile before induced voltage')
            profile.track()
            plt.plot(profile.bin_centers, profile.n_macroparticles)
            #
            #            plt.figure('inducedvoltage before induced voltage')
            #            profile.track()
            #            plt.plot(TotalInducedVoltage.time_array,TotalInducedVoltage.induced_voltage)
            #
            induced_voltage_coordinates = total_induced_voltage.time_array
            induced_voltage = total_induced_voltage.induced_voltage
            induced_potential = (- normalization_potential
                                 * cumtrapz(induced_voltage,
                                            dx=float(induced_voltage_coordinates[1]
                                                     - induced_voltage_coordinates[0]),
                                            initial=0))

            plt.figure('testInducedVolt')
            plt.plot(induced_voltage_coordinates, induced_voltage)
            plt.figure('testInducedPot')
            plt.plot(induced_voltage_coordinates, induced_potential)

            full_ring_and_rf.potential_well_generation(n_points=n_points_potential,
                                                       dt_margin_percent=dt_margin_percent,
                                                       main_harmonic_option=main_harmonic_option)

            # Restrict the potential well inside the separatrix and put min on 0
            potential_well_coordinates, potential_well = potential_well_cut(
                full_ring_and_rf.potential_well_coordinates,
                full_ring_and_rf.potential_well)
            potential_well = potential_well - np.min(potential_well)

            temporary_beam = Beam(ring, n_macro_per_bunch, intensity_per_bunch)
            for indexBatch in range(n_batch):
                for indexBunch in range(n_bunches):
                    # Extract the induced potential for the specific bucket
                    induced_potential_bunch = np.interp(
                        potential_well_coordinates + indexBunch
                        * bunch_spacing_buckets * bucket_size_tau
                        + indexBatch * (batch_spacing_buckets
                                        + (n_bunches - 1)
                                        * bunch_spacing_buckets)
                        * bucket_size_tau, induced_voltage_coordinates,
                        induced_potential)

                    # Recompute the phase space distribution for the new
                    # perturbed potential (containing induced_potential_bunch)
                    match_a_bunch(normalization_DeltaE, temporary_beam,
                                  potential_well_coordinates,
                                  potential_well + induced_potential_bunch, seed,
                                  distribution_options,
                                  full_ring_and_rf=full_ring_and_rf)

                    dt = temporary_beam.dt
                    dE = temporary_beam.dE

                    # Compute RMS emittance to observe convergence
                    conv += np.pi * np.std(dt) * np.std(dE)

                    length_dt = len(dt)
                    length_dE = len(dE)
                    slice_start = (indexBunch + n_bunches * indexBatch) * length_dt
                    slice_stop = (indexBunch + n_bunches * indexBatch + 1) * length_dt
                    beam.dt[slice_start:slice_stop] = (
                            dt + (indexBunch * bunch_spacing_buckets
                                  * bucket_size_tau) + indexBatch
                            * (batch_spacing_buckets + (n_bunches - 1)
                               * bunch_spacing_buckets)
                            * bucket_size_tau)
                    slice_start = (indexBunch + n_bunches * indexBatch) * length_dE
                    slice_stop = (indexBunch + n_bunches * indexBatch + 1) * length_dE
                    beam.dE[slice_start:slice_stop] = dE

            print('iteration ' + str(it) + ', average RMS emittance (4sigma) = ' + str(4 * conv / n_bunches))
            profile.track()
            total_induced_voltage.induced_voltage_sum()
