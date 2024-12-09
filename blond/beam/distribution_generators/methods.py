# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to generate distributions**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**, **Theodoros Argyropoulos**,
          **Joel Repond**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..profile import CutOptions, Profile
from ...utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import NDArray

    from ..beam import Beam
    from ...trackers.tracker import FullRingAndRF
    from ...utils.types import (BunchLengthFitTypes)


@handle_legacy_kwargs
def x0_from_bunch_length(bunch_length: float,
                         bunch_length_fit: BunchLengthFitTypes,
                         X_grid: NDArray,
                         sorted_X_dE0: NDArray,
                         n_points_grid: int,
                         time_potential_low_res: NDArray,
                         distribution_function_: Callable,
                         # TODO this is just distribution_function with strange way, is this intended?
                         distribution_type: str,
                         distribution_exponent: float,
                         beam: Beam,
                         full_ring_and_rf: FullRingAndRF
                         ) -> float:
    """
    Function to find the corresponding H0 or J0 for a given bunch length.
    Used by matched_from_distribution_function()
    """  # todo documentation
    # todo improve readability
    tau = 0.0

    # Initial values for iteration
    x_low = sorted_X_dE0[0]
    x_hi = sorted_X_dE0[-1]
    X_min = sorted_X_dE0[0]
    X_max = sorted_X_dE0[-1]
    X_accuracy = (sorted_X_dE0[1] - sorted_X_dE0[0]) / 2.0

    bin_size = (time_potential_low_res[1] - time_potential_low_res[0])

    # Iteration to find H0/J0 from the bunch length
    while np.abs(bunch_length - tau) > bin_size:
        # Takes middle point of the interval [X_low,X_hi]
        X0 = 0.5 * (x_low + x_hi)

        if bunch_length_fit == 'full':
            bunch_indices = np.where(np.sum(X_grid <= X0, axis=0))[0]
            tau = float(time_potential_low_res[bunch_indices][-1] -
                        time_potential_low_res[bunch_indices][0])
        else:
            # Calculating the line density for the parameter X0
            density_grid = distribution_function_(
                X_grid, distribution_type, X0, distribution_exponent
            )

            density_grid = density_grid / np.sum(density_grid)
            line_density_ = np.sum(density_grid, axis=0)

            # Calculating the bunch length of that line density
            if (line_density_ > 0).any():
                tau = 4.0 * np.sqrt(np.sum((time_potential_low_res
                                            - np.sum(line_density_
                                                     * time_potential_low_res)
                                            / np.sum(line_density_)) ** 2
                                           * line_density_)
                                    / np.sum(line_density_))

                if bunch_length_fit is not None:
                    profile = Profile(
                        beam, cut_options=CutOptions(
                            cut_left=(time_potential_low_res[0] - 0.5
                                      * bin_size),
                            cut_right=(time_potential_low_res[-1]
                                       + 0.5 * bin_size),
                            n_slices=n_points_grid,
                            rf_station=full_ring_and_rf.ring_and_rf_section[0].rf_params
                        )
                    )
                    #                     profile = Profile(
                    #                       full_ring_and_RF.RingAndRFSection_list[0].rf_params,
                    #                       beam, n_points_grid, cut_left=time_potential_low_res[0] -
                    #                       0.5*bin_size , cut_right=time_potential_low_res[-1] +
                    #                       0.5*bin_size)

                    profile.n_macroparticles = line_density_

                    if bunch_length_fit == 'gauss':
                        raise NotImplementedError()
                        # FIXME this wont work as gaussian_fit is undefined. What shall this function do?
                        profile.bl_gauss = tau
                        profile.bp_gauss = (np.sum(line_density_ *
                                                   time_potential_low_res)
                                            / np.sum(line_density_))
                        profile.gaussian_fit()  # FIXME
                        tau = profile.bl_gauss
                    elif bunch_length_fit == 'fwhm':
                        profile.fwhm()
                        tau = profile.bunchLength

        # Update of the interval for the next iteration
        if tau >= bunch_length:
            x_hi = X0
        else:
            x_low = X0

        if (X_max - X0) < X_accuracy:
            print('WARNING: The bucket is too small to have the ' +
                  'desired bunch length! Input is %.2e, ' % bunch_length +
                  'the generation gave %.2e, ' % tau +
                  'the error is %.2e' % (bunch_length - tau))
            break

        if (X0 - X_min) < X_accuracy:
            print('WARNING: The desired bunch length is too small ' +
                  'to be generated accurately!')

    #    return 0.5 * (X_low + X_hi)
    return X0
