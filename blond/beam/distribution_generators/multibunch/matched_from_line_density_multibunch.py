"""
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
"""

from __future__ import annotations

import copy
import gc
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ...beam import Beam
from ...distribution_generators.singlebunch.matched_from_line_density import FitTableLineDensity, FitBunchLengthLineDensity, \
    MatchedFromLineDensity
from ....utils import bmath as bm
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

    from ....input_parameters.ring import Ring
    from ....impedances.impedance import TotalInducedVoltage
    from ....trackers.tracker import FullRingAndRF, MainHarmonicOptionType
    from ....utils.types import HalfOptionType


@handle_legacy_kwargs
def matched_from_line_density_multibunch(beam: Beam, ring: Ring,
                                         full_ring_and_rf: FullRingAndRF,
                                         fit: FitTableLineDensity | FitBunchLengthLineDensity |
                                              list[FitTableLineDensity | FitBunchLengthLineDensity],
                                         n_bunches: int,
                                         bunch_spacing_buckets: int,
                                         intensity_list: list | NDArray | None = None,
                                         minimum_n_macroparticles: Optional[list[int | float]] = None,
                                         main_harmonic_option: MainHarmonicOptionType = 'lowest_freq',
                                         total_induced_voltage: Optional[TotalInducedVoltage] = None,
                                         half_option: HalfOptionType = 'first',
                                         plot_option: bool = False,
                                         seed: Optional[int] = None,
                                         ):
    """
    *Function to generate a multi-bunch beam using the matched_from_distribution_density
    function for each bunch. The extra parameters to include are the number of
    bunches and the spacing between two bunches (assumed constant presently).
    Moreover, the line_density_options_list corresponds to the distribution_options
    of the matched_from_line_density function. It can be inputted as
    a dictionary just like the matched_from_line_density function (assuming
    the same parameters for all bunches), or as a list of length n_bunches
    to have different parameters for each bunch.*

    Parameters
    ----------
    beam
        Class containing the beam properties.
    ring
        Class containing the general properties of the synchrotron that are
        independent of the RF system or the beam.
    full_ring_and_rf
        Definition of the full ring and RF parameters in order to be able to have a full turn information
    fit
        FitTableLineDensity or FitBunchLengthDistribution or list[FitTableLineDensity or list[FitBunchLengthDistribution]
    n_bunches
        # todo
    bunch_spacing_buckets
        # todo
    intensity_list
        # todo
    minimum_n_macroparticles
        # todo
    main_harmonic_option
        'lowest_freq', 'highest_voltage'
    total_induced_voltage
        # todo
    half_option
        TODO
        'first', 'second', or 'both'
    plot_option
        Whether to plot the output
    seed
        Random seed

    """

    if intensity_list is None:
        intensity_per_bunch = beam.intensity / n_bunches * np.ones(n_bunches)
        n_macroparticles_per_bunch = beam.n_macroparticles / n_bunches * np.ones(n_bunches)
    else:
        intensity_per_bunch = np.array(intensity_list)
        if minimum_n_macroparticles is None:
            n_macroparticles_per_bunch = np.round(beam.n_macroparticles / beam.intensity * intensity_per_bunch)
        else:
            n_macroparticles_per_bunch = np.round(minimum_n_macroparticles
                                                  / np.min(intensity_per_bunch)
                                                  * intensity_per_bunch)

    if np.sum(intensity_per_bunch) != beam.intensity:
        print(
            'WARNING !! The total intensity per bunch does not match the total intensity of the beam, the beam.intensity will be overwritten')
        beam.intensity = np.sum(intensity_per_bunch)

    if np.sum(n_macroparticles_per_bunch) != beam.n_macroparticles:
        print(
            'WARNING !! The number of macroparticles per bunch does not match the total number of the beam, the beam.n_macroparticles will be overwritten')
        beam.n_macroparticles = int(np.sum(n_macroparticles_per_bunch))

    voltages = np.array([])
    harmonics = np.array([])

    for RingAndRFSectionElement in full_ring_and_rf.ring_and_rf_section:
        for rf_system in range(RingAndRFSectionElement.rf_params.n_rf):
            voltages = np.append(voltages, RingAndRFSectionElement.rf_params.voltage[rf_system, 0])
            harmonics = np.append(harmonics, RingAndRFSectionElement.rf_params.harmonic[rf_system, 0])

    if main_harmonic_option == 'lowest_freq':
        main_harmonic = np.min(harmonics)
    elif main_harmonic_option == 'highest_voltage':
        main_harmonic = np.min(harmonics[voltages == np.max(voltages)])
    elif isinstance(main_harmonic_option, int) or isinstance(main_harmonic_option, float):
        if harmonics[harmonics == main_harmonic_option].size == 0:
            # GenerationError
            raise RuntimeError('The desired harmonic to compute the potential well does not match the RF parameters...')
        main_harmonic = np.min(harmonics[harmonics == main_harmonic_option])
    else:
        raise ValueError(f"main_harmonic_option did not match any option! {main_harmonic_option=}")

    bucket_size_tau = 2 * np.pi / (main_harmonic * ring.omega_rev[0])

    beam_iteration = Beam(ring, 1, 0.)

    extra_voltage_dict = None

    if total_induced_voltage is not None:
        total_induced_voltage_iteration = copy.deepcopy(total_induced_voltage)
        total_induced_voltage_iteration.profile.Beam = beam_iteration

    for indexBunch in range(0, n_bunches):

        print('Generating bunch no %d' % (indexBunch + 1))

        bunch = Beam(ring,
                     int(n_macroparticles_per_bunch[indexBunch]),
                     float(intensity_per_bunch[indexBunch]))

        if isinstance(fit, list):
            _fit_bunch_i = fit[indexBunch]
        else:
            _fit_bunch_i = fit

        full_ring_and_rf.potential_well_generation(
            main_harmonic_option=main_harmonic_option,

            n_points=int(1e3),  # default values from legacy matched_from_line_density
            dt_margin_percent=0.40,  # default values from legacy matched_from_line_density

            # Commented parameters that are available,
            # but unused by current implementation:

            # turn = 0
            # time_array = None
        )

        m = MatchedFromLineDensity(
            beam=bunch,
            full_ring_and_rf=full_ring_and_rf,
            fit=_fit_bunch_i,
            total_induced_voltage=total_induced_voltage,
            extra_voltage_dict=extra_voltage_dict,
        )
        # Commented parameters that are available,
        # but unused by current implementation:

        # m.plot = plot
        # m.figdir = figdir
        # m.process_pot_well = process_pot_well
        # m.n_points_abel = int(n_points_abel)
        # m.n_points_grid = n_points_grid
        m.half_option = half_option
        m.seed = seed

        m.match_beam()

        if indexBunch == 0:
            beam_iteration.dt = bunch.dt
            beam_iteration.dE = bunch.dE
        else:
            beam_iteration.dt = np.append(beam_iteration.dt, bunch.dt +
                                          (indexBunch * bunch_spacing_buckets * bucket_size_tau))
            beam_iteration.dE = np.append(beam_iteration.dE, bunch.dE)

        beam_iteration.n_macroparticles = int(np.sum(n_macroparticles_per_bunch[:indexBunch + 1]))
        beam_iteration.intensity = np.sum(intensity_per_bunch[:indexBunch + 1])
        beam_iteration.ratio = beam_iteration.intensity / beam_iteration.n_macroparticles

        if total_induced_voltage is not None:
            total_induced_voltage_iteration.profile.track()
            total_induced_voltage_iteration.induced_voltage_sum()

            bucket_tolerance = 0.40

            left_edge = (indexBunch + 1) * bunch_spacing_buckets * bucket_size_tau - bucket_tolerance * bucket_size_tau
            right_edge = (((indexBunch + 1) * bunch_spacing_buckets + 1)
                          * bucket_size_tau + bucket_tolerance * bucket_size_tau)

            tau_induced_voltage_next_bunch = total_induced_voltage_iteration.profile.bin_centers[
                (total_induced_voltage_iteration.profile.bin_centers > left_edge)
                * (total_induced_voltage_iteration.profile.bin_centers < right_edge)]
            induced_voltage_next_bunch = total_induced_voltage_iteration.induced_voltage[
                (total_induced_voltage_iteration.profile.bin_centers > left_edge)
                * (total_induced_voltage_iteration.profile.bin_centers < right_edge)]

            time_induced_voltage_next_bunch = (tau_induced_voltage_next_bunch
                                               - (indexBunch + 1)
                                               * bunch_spacing_buckets
                                               * bucket_size_tau)

            extra_voltage_dict = {'time_array': time_induced_voltage_next_bunch,
                                  'voltage_array': induced_voltage_next_bunch}

    if plot_option:
        plt.figure('Bunch train + induced voltage')
        plt.clf()
        plt.plot(total_induced_voltage_iteration.profile.bin_centers,
                 total_induced_voltage_iteration.profile.n_macroparticles
                 / (1. * np.max(total_induced_voltage_iteration.profile.n_macroparticles))
                 * np.max(total_induced_voltage_iteration.induced_voltage))
        plt.plot(total_induced_voltage_iteration.profile.bin_centers,
                 total_induced_voltage_iteration.induced_voltage)
        plt.show()

    beam.dt = beam_iteration.dt.astype(dtype=bm.precision.real_t, order='C', copy=False)
    beam.dE = beam_iteration.dE.astype(dtype=bm.precision.real_t, order='C', copy=False)
    gc.collect()
