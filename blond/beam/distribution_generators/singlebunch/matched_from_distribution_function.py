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

import copy
import gc
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy
from packaging.version import Version

if Version(scipy.__version__) >= Version("1.14"):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
else:
    from scipy.integrate import cumtrapz
from .methods import populate_bunch
from ..methods import x0_from_bunch_length
from ...profile import Profile
from ....trackers.utilities import potential_well_cut
from ....utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray

    from ...beam import Beam
    from ....impedances.impedance import TotalInducedVoltage
    from ....trackers.tracker import FullRingAndRF, MainHarmonicOptionType
    from ....utils.types import (DistributionUserTableType, ExtraVoltageDictType, DistributionVariableType,
                                 BunchLengthFitTypes, DistTypeDistFunction, DistributionFunctionTypeHint)


def __distribution_function_by_exponent(action_array: NDArray, exponent: float,
                                        length: float) -> NDArray:
    warnings.filterwarnings("ignore")
    distribution_function_ = (1 - action_array / length) ** exponent
    warnings.filterwarnings("default")
    distribution_function_[action_array > length] = 0
    return distribution_function_


def distribution_function(action_array: NDArray,
                          dist_type: DistTypeDistFunction,
                          length: float,
                          exponent: Optional[float] = None) -> NDArray:
    """
    *Distribution function (formulas from Laclare).*
    """

    if dist_type == 'waterbag':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'parabolic_amplitude':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 1
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'parabolic_line':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        exponent = 0.5
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'binomial':
        assert exponent is not None, "Please specify exponent"
        distribution_ = __distribution_function_by_exponent(action_array,
                                                            exponent, length)

    elif dist_type == 'gaussian':
        if exponent is not None:
            warnings.warn(f"exponent is ignored for {dist_type=}")
        distribution_ = np.exp(- 2 * action_array / length)

    else:
        # DistributionError
        raise RuntimeError(
            f"'dist_type' must be in ['waterbag','parabolic_amplitude','parabolic_line','binomial','gaussian'], not '{dist_type}'")

    return distribution_


@handle_legacy_kwargs
def matched_from_distribution_function(
        beam: Beam, full_ring_and_rf: FullRingAndRF,
        distribution_function_input: DistributionFunctionTypeHint = distribution_function,
        distribution_user_table: Optional[DistributionUserTableType] = None,
        main_harmonic_option: MainHarmonicOptionType = 'lowest_freq',
        total_induced_voltage: Optional[TotalInducedVoltage] = None,
        n_iterations: int = 1,
        n_points_potential: float = 1e4,
        n_points_grid: int = int(1e3),
        dt_margin_percent: float = 0.40,
        extra_voltage_dict: Optional[ExtraVoltageDictType] = None,
        seed: Optional[int] = None,
        distribution_exponent: Optional[float] = None,
        distribution_type: Optional[DistTypeDistFunction] = None,
        emittance: Optional[float] = None,
        bunch_length: Optional[float] = None,
        bunch_length_fit: Optional[BunchLengthFitTypes] = None,
        distribution_variable: DistributionVariableType = 'Hamiltonian',
        process_pot_well: bool = True,
        turn_number: int = 0
) -> tuple[list[NDArray], TotalInducedVoltage] | list[NDArray]:
    """Function to generate a beam by inputting the distribution function

    *Function to generate a beam by inputting the distribution function by
    choosing the type of distribution and the emittance.
    The potential well is preprocessed to check for the min/max and center
    the frame around the separatrix.
    An error will be raised if there is not a full potential well (2 max
    and 1 min at least), or if there are several wells (more than 2 max and
    1 min, this case will be treated in the future).
    An adjustable margin (40% by default) is applied in order to be able to
    catch the min/max of the potential well that might be on the edge of the
    frame. The slippage factor should be updated to take the higher orders.
    Outputs should be added in order for the user to check step by step if
    his bunch is going to be well generated. More detailed 'step by step'
    documentation should be implemented
    The user can input a custom distribution function by setting the parameter
    distribution_type = 'user_input' and passing the function in the
    parameter distribution_options['function'], with the following definition:
    distribution_function(action_array, dist_type, length, exponent=None).
    The user can also add an input table by setting the parameter
    distribution_type = 'user_input_table',
    distribution_options['user_table_action'] = array of action (in H or in J)
    and distribution_options['user_table_distribution']*


    Parameters
    ----------
    beam
        Class containing the beam properties.
    full_ring_and_rf
        Definition of the full ring and RF parameters in order to be able to have a full turn information
    distribution_user_table
        Dictionary holding the arrays action and distribution
        Tip: Use distribution_function_input OR distribution_user_table
    main_harmonic_option
        'lowest_freq', 'highest_voltage'
    total_induced_voltage
        TODO
    n_iterations
        Number of iterations to match distribution
    n_points_potential
        Number of points to be used in the potential well calculation
    n_points_grid
        Internal grid resolution
    dt_margin_percent
        See FullRingAndRF.potential_well_generation
    extra_voltage_dict
        Extra potential from previous bunches (for multi-bunch generation).
        Offsets the total_potential.
        (total_potential = potential_well + induced_potential + extra_potential)
    seed
        Random seed
    distribution_exponent
        Distribution exponent required for distribution_type='binomial'
    distribution_type
        Must be specified when not using 'distribution_user_table'
        'waterbag', 'parabolic_amplitude', 'parabolic_line', 'binomial', 'gaussian'
    emittance
        Beam emittance to calculate density grid.
        Use either emittance OR bunch_length.
        When using 'distribution_user_table', 'emittance' is ignored!
    bunch_length
        Bunch length to calculate density grid.
        Use either emittance OR bunch_length.
        When using 'distribution_user_table', 'bunch_length' is ignored!
    bunch_length_fit
        Used in combination with bunch_length
        None, 'full', 'gauss', 'fwhm'
    distribution_variable
        'Action' # TODO
        'Hamiltonian' # TODO
    process_pot_well
        If true, process the potential well in order
        to take a frame around the separatrix
    turn_number
        Used to calculate the EOM??

    Returns
    -------
    if total_induced_voltage is not None:
        [time_potential_low_res, line_density_], induced_voltage_object
    else:
        [time_potential_low_res, line_density_]


    Examples
    --------
    >>> import numpy as np

    >>> from blond.beam.beam import Beam, Proton
    >>> from blond.beam.distributions import (matched_from_distribution_function)
    >>> from blond.input_parameters.rf_parameters import RFStation
    >>> from blond.input_parameters.ring import Ring
    >>> from blond.trackers.tracker import FullRingAndRF, RingAndRFTracker

    >>> ring = Ring(
    >>>     ring_length=2 * np.pi * 4242.89,
    >>>     alpha_0=1 / 55.76 ** 2,  # alpha by gamma_transition
    >>>     synchronous_data=7e12,
    >>>     particle=Proton(),
    >>>     n_turns=int(1e4),
    >>> )

    >>> RF_sct_par = RFStation(
    >>>     ring=ring,
    >>>     harmonic=[35640.0],
    >>>     voltage=[16e6],
    >>>     phi_rf_d=[0],
    >>>     n_rf=1,
    >>> )

    >>> beam = Beam(
    >>>     ring=ring,
    >>>     n_macroparticles=int(1e6),
    >>>     intensity=int(1e11),
    >>> )
    >>>
    >>> full_tracker = FullRingAndRF(
    >>>     ring_and_rf_section=[RingAndRFTracker(RF_sct_par, beam)],
    >>> )
    >>>
    >>> matched_from_distribution_function(
    >>>     beam=beam,
    >>>     full_ring_and_rf=full_tracker,
    >>>     distribution_type='parabolic_line',
    >>>     distribution_exponent=None,  # Automatically set for parabolic line
    >>>     bunch_length=0.5e-9,
    >>>     bunch_length_fit='full',
    >>>     distribution_variable='Action',
    >>>     seed=18,
    >>> )

    """

    # Generate potential well
    full_ring_and_rf.potential_well_generation(turn=turn_number,
                                               n_points=int(n_points_potential),
                                               dt_margin_percent=dt_margin_percent,
                                               main_harmonic_option=main_harmonic_option)
    if distribution_user_table is not None:
        fit = FitTableDistribution(distribution_user_table=distribution_user_table)
    elif emittance is not None:
        fit = FitEmittanceDistribution(
            emittance=emittance,
            distribution_type=distribution_type,
            distribution_exponent=distribution_exponent,
        )
        if distribution_function_input is not None:
            fit.distribution_function_input = distribution_function_input
    elif bunch_length is not None:
        fit = FitBunchLengthDistribution(
            bunch_length=bunch_length,
            bunch_length_fit=bunch_length_fit,
            distribution_type=distribution_type,
            distribution_exponent=distribution_exponent,
        )
        if distribution_function_input is not None:
            fit.distribution_function_input = distribution_function_input
    else:
        raise KeyError(
            "'distribution_user_table','emittance', or 'bunch_length' must be given to matched_from_distribution_function()!")
    m = MatchedFromDistributionFunction(
        beam=beam,
        full_ring_and_rf=full_ring_and_rf,
        fit=fit,
        total_induced_voltage=total_induced_voltage,
        n_iterations=n_iterations,
        extra_voltage_dict=extra_voltage_dict,
        turn_number=turn_number,
    )
    m.n_points_grid = n_points_grid
    m.seed = seed

    m.distribution_variable = distribution_variable
    m.process_pot_well = process_pot_well
    m.main_harmonic_option = main_harmonic_option

    m.match_beam()
    if total_induced_voltage is not None:
        return [m._time_potential_low_res, m._line_density_], m._induced_voltage_object
    else:
        return [m._time_potential_low_res, m._line_density_]


class FitEmittanceDistribution:
    """Parameters for MatchedFromDistributionFunction to fit emittance

    Parameters
    ----------
    emittance
        Beam emittance to calculate density grid.
        Use either emittance.
    distribution_type
        'waterbag', 'parabolic_amplitude', 'parabolic_line', 'binomial', 'gaussian'
    distribution_exponent
        Distribution exponent required for distribution_type='binomial'

    Attributes
    ----------
    distribution_function_input
        Distribution function
    emittance
        Beam emittance to calculate density grid.
    distribution_type
        'waterbag', 'parabolic_amplitude', 'parabolic_line', 'binomial', 'gaussian'
    distribution_exponent
        Distribution exponent required for distribution_type='binomial'

    """

    def __init__(self,
                 emittance: float,
                 distribution_type: DistTypeDistFunction,
                 distribution_exponent: Optional[float] = None,
                 ):

        super().__init__()
        assert isinstance(distribution_function, Callable)
        self.distribution_function_input = distribution_function
        self.emittance = float(emittance)
        self.distribution_type: DistTypeDistFunction = str(distribution_type)
        if self.distribution_type != 'binomial':
            if distribution_exponent is not None:
                warnings.warn("'distribution_exponent' is only required for distribution_type='binomial'!")
            self.distribution_exponent = None
        else:
            self.distribution_exponent = float(distribution_exponent)


class FitBunchLengthDistribution:
    """Parameters for MatchedFromDistributionFunction to fit emittance

        Parameters
        ----------
        bunch_length
            Bunch length to calculate density grid.
        distribution_type
            'waterbag', 'parabolic_amplitude', 'parabolic_line', 'binomial', 'gaussian'
        distribution_exponent
            Distribution exponent required for distribution_type='binomial'
        bunch_length_fit
            None, 'full', 'gauss', 'fwhm'


        Attributes
        ----------
        distribution_function_input
            Distribution function
        bunch_length
            Bunch length to calculate density grid.
        distribution_type
            'waterbag', 'parabolic_amplitude', 'parabolic_line', 'binomial', 'gaussian'
        distribution_exponent
            Distribution exponent required for distribution_type='binomial'
        bunch_length_fit
            None, 'full', 'gauss', 'fwhm'


        """

    def __init__(self,
                 bunch_length: float,
                 distribution_type: DistTypeDistFunction,
                 distribution_exponent: Optional[float] = None,
                 bunch_length_fit: BunchLengthFitTypes = None):
        super().__init__()
        assert isinstance(distribution_function, Callable)
        self.distribution_function_input = distribution_function
        self.bunch_length = float(bunch_length)
        self.distribution_type: DistTypeDistFunction = str(distribution_type)
        if self.distribution_type != 'binomial':
            if distribution_exponent is not None:
                warnings.warn("'distribution_exponent' is only required for distribution_type='binomial'!")
            self.distribution_exponent = None

        else:
            self.distribution_exponent = float(distribution_exponent)

        self.bunch_length_fit = bunch_length_fit


class FitTableDistribution:
    def __init__(self, distribution_user_table: DistributionUserTableType):
        super().__init__()
        assert isinstance(distribution_user_table, dict)
        self.distribution_user_table = distribution_user_table


class MatchedFromDistributionFunction:
    """Function to generate a beam by inputting the distribution function

        *Function to generate a beam by inputting the distribution function by
        choosing the type of distribution and the emittance.
        The potential well is preprocessed to check for the min/max and center
        the frame around the separatrix.
        An error will be raised if there is not a full potential well (2 max
        and 1 min at least), or if there are several wells (more than 2 max and
        1 min, this case will be treated in the future).
        An adjustable margin (40% by default) is applied in order to be able to
        catch the min/max of the potential well that might be on the edge of the
        frame. The slippage factor should be updated to take the higher orders.
        Outputs should be added in order for the user to check step by step if
        his bunch is going to be well generated. More detailed 'step by step'
        documentation should be implemented
        The user can input a custom distribution function by setting the parameter
        distribution_type = 'user_input' and passing the function in the
        parameter distribution_options['function'], with the following definition:
        distribution_function(action_array, dist_type, length, exponent=None).
        The user can also add an input table by setting the parameter
        distribution_type = 'user_input_table',
        distribution_options['user_table_action'] = array of action (in H or in J)
        and distribution_options['user_table_distribution']*


        Parameters
        ----------
        beam
           Class containing the beam properties.
        full_ring_and_rf
           Definition of the full ring and RF parameters in order to be able to have a full turn information
        total_induced_voltage
           TODO
        n_iterations
           Number of iterations to match distribution
        extra_voltage_dict
           Extra potential from previous bunches (for multi-bunch generation).
           Offsets the total_potential.
           (total_potential = potential_well + induced_potential + _extra_potential)
        turn_number
           Used to calculate the EOM??

        Attributes
        ----------
        n_points_grid
           Internal grid resolution
        seed
            Random seed
        fit
            Fit options
        distribution_variable
            'Action' TODO documentation
            'Hamiltonian' TODO documentation
        process_pot_well
            If true, process the potential well in order
            to take a frame around the separatrix
        main_harmonic_option
            'lowest_freq', 'highest_voltage'
       """

    def __init__(self,
                 beam: Beam,
                 full_ring_and_rf: FullRingAndRF,
                 fit: FitTableDistribution | FitEmittanceDistribution | FitBunchLengthDistribution,
                 total_induced_voltage: Optional[TotalInducedVoltage] = None,
                 extra_voltage_dict: Optional[ExtraVoltageDictType] = None,
                 turn_number: int = 0,
                 n_iterations: int = 1,
                 ):
        assert full_ring_and_rf.potential_well is not None, "Please call full_ring_and_rf.potential_well_generation() before using it for beam matching!"

        self._beam = beam
        self._eom_factor_dE, self._eom_factor_potential = self._calc_eom(full_ring_and_rf=full_ring_and_rf,
                                                                         turn_number=turn_number)
        self._n_points_potential = len(full_ring_and_rf.potential_well_coordinates)

        self._full_ring_and_rf = full_ring_and_rf
        self._potential_well: NDArray = full_ring_and_rf.potential_well
        self._time_potential: NDArray = full_ring_and_rf.potential_well_coordinates

        self._induced_potential: float | NDArray = 0

        self._extra_potential = self._calc_extra_potential(extra_voltage_dict=extra_voltage_dict)

        self._total_potential = self._potential_well + self._induced_potential + self._extra_potential

        if total_induced_voltage is None:
            if n_iterations != 1:
                warnings.warn("When given 'total_induced_voltage', '_n_iterations' is overwritten as 1!", UserWarning)
            n_iterations = 1
            self._induced_voltage_object: TotalInducedVoltage | None = None
            self._profile: Profile | None = None
        else:
            self._induced_voltage_object: TotalInducedVoltage | None = copy.deepcopy(total_induced_voltage)
            self._profile: Profile | None = self._induced_voltage_object.profile
        self._n_iterations = n_iterations  # might be reset by total_induced_voltage to 1
        self._dE_trajectory = np.zeros(self._n_points_potential)

        ##########################
        # Attributes for the user
        ##########################

        # other options that can be adjusted by the user before using match_beam()
        self.n_points_grid: int = int(1e3)
        self.seed: Optional[int] = None
        self.fit = fit
        self.distribution_variable: DistributionVariableType = 'Hamiltonian'
        self.process_pot_well: bool = True
        self.main_harmonic_option: MainHarmonicOptionType = 'lowest_freq'
        #########################

        self._time_potential_low_res: NDArray = None  # set by _outer_loop
        self._line_density_: NDArray = None  # set by _outer_loop

    def match_beam(self):
        """Match the beam to the fit parameters"""
        self._outer_loop()
        self._ready_results()

    def _ready_results(self):
        """Convenience function to document the behaviour of legacy matched_from_distribution_function()"""
        if self._profile is not None:
            self._time_potential_low_res = self._time_potential_low_res
            self._line_density_ = self._line_density_
            self._induced_voltage_object = self._induced_voltage_object
        else:
            self._time_potential_low_res = self._time_potential_low_res
            self._line_density_ = self._line_density_

    def _outer_loop(self):
        for i in range(self._n_iterations):
            old_potential = copy.deepcopy(self._total_potential)

            # Adding the induced potential to the RF potential
            self._total_potential = (self._potential_well + self._induced_potential +
                                     self._extra_potential)

            sse = np.sqrt(np.sum((old_potential - self._total_potential) ** 2))

            print('Matching the bunch... (iteration: ' + str(i) + ' and sse: ' +
                  str(sse) + ')')

            # Process the potential well in order to take a frame around the separatrix
            if not self.process_pot_well:
                time_potential_sep, potential_well_sep = self._time_potential, self._total_potential
            else:
                time_potential_sep, potential_well_sep = potential_well_cut(
                    self._time_potential, self._total_potential)

            # Potential is shifted to put the minimum on 0
            potential_well_sep = potential_well_sep - np.min(potential_well_sep)

            # Compute deltaE frame corresponding to the separatrix
            max_potential = np.max(potential_well_sep)
            max_deltaE = np.sqrt(max_potential / self._eom_factor_dE)

            # Initializing the grids by reducing the resolution to a
            # n_points_grid*n_points_grid frame
            self._time_potential_low_res = np.linspace(float(time_potential_sep[0]),
                                                       float(time_potential_sep[-1]),
                                                       self.n_points_grid)
            time_resolution_low = (float(self._time_potential_low_res[1]) -
                                   float(self._time_potential_low_res[0]))
            deltaE_coord_array = np.linspace(-float(max_deltaE), float(max_deltaE),
                                             self.n_points_grid)
            potential_well_low_res = np.interp(self._time_potential_low_res,
                                               time_potential_sep,
                                               potential_well_sep)
            time_grid, deltaE_grid = np.meshgrid(self._time_potential_low_res,
                                                 deltaE_coord_array)
            potential_well_grid = np.meshgrid(potential_well_low_res,
                                              potential_well_low_res)[0]

            # Computing the action J by integrating the dE trajectories

            J_array_dE0 = self._inner_loop(
                i=i,
                full_ring_and_rf2=copy.deepcopy(self._full_ring_and_rf),
                potential_well_low_res=potential_well_low_res,
            )

            # Sorting the H and J functions to be able to interpolate J(H)
            H_array_dE0 = potential_well_low_res
            sorted_H_dE0 = H_array_dE0[H_array_dE0.argsort()]
            sorted_J_dE0 = J_array_dE0[H_array_dE0.argsort()]

            # Calculating the H and J grid
            H_grid = self._eom_factor_dE * deltaE_grid ** 2 + potential_well_grid
            J_grid = np.interp(H_grid, sorted_H_dE0, sorted_J_dE0, left=0,
                               right=np.inf)

            # Choice of either H or J as the variable used
            if self.distribution_variable == 'Action':
                sorted_X_dE0 = sorted_J_dE0
                X_grid = J_grid
            elif self.distribution_variable == 'Hamiltonian':
                sorted_X_dE0 = sorted_H_dE0
                X_grid = H_grid
            else:
                raise RuntimeError(
                    f"'distribution_variable' must be 'Action' or 'Hamiltonian', not '{self.distribution_variable}'")

            # Computing the density grid
            # Computing bunch length as a function of H/J if needed
            # Bunch length can be calculated as 4-rms, Gaussian fit, or FWHM
            if isinstance(self.fit, FitBunchLengthDistribution):
                _fit: FitBunchLengthDistribution = self.fit
                X0 = x0_from_bunch_length(_fit.bunch_length, _fit.bunch_length_fit,
                                          X_grid, sorted_X_dE0, self.n_points_grid,
                                          self._time_potential_low_res,
                                          _fit.distribution_function_input,
                                          _fit.distribution_type, _fit.distribution_exponent,
                                          self._beam, self._full_ring_and_rf)
                density_grid = _fit.distribution_function_input(X_grid, _fit.distribution_type, X0,
                                                                _fit.distribution_exponent)
            elif isinstance(self.fit, FitEmittanceDistribution):
                _fit: FitEmittanceDistribution = self.fit
                X0 = self.x0_from_emittance(emittance=_fit.emittance, sorted_H_dE0=sorted_H_dE0,
                                            sorted_J_dE0=sorted_J_dE0)
                density_grid = _fit.distribution_function_input(X_grid, _fit.distribution_type, X0,
                                                                _fit.distribution_exponent)
            elif isinstance(self.fit, FitTableDistribution):
                _fit: FitTableDistribution = self.fit
                density_grid = np.interp(
                    X_grid,
                    _fit.distribution_user_table['user_table_action'],
                    _fit.distribution_user_table['user_table_distribution']
                )
            else:
                raise TypeError(
                    f"Expected FitBunchLength, FitEmittance, or FitDistributionUserTable, not {type(self.fit)}")
            # Normalizing the grid
            density_grid[H_grid > np.max(H_array_dE0)] = 0
            density_grid = density_grid / np.sum(density_grid)

            # Calculating the line density
            self._line_density_ = np.sum(density_grid, axis=0)
            self._line_density_ *= self._beam.n_macroparticles / np.sum(self._line_density_)

            # Induced voltage contribution
            if self._profile is not None:
                self._induced_potential = self._calc_induced_potential(time_resolution_low)
            gc.collect()
        # Populating the bunch
        populate_bunch(self._beam, time_grid, deltaE_grid, density_grid,
                       time_resolution_low, float(deltaE_coord_array[1]
                                                  - deltaE_coord_array[0]), self.seed)

    def _inner_loop(self, i, full_ring_and_rf2, potential_well_low_res):
        J_array_dE0 = np.zeros(self.n_points_grid)

        for j in range(self.n_points_grid):
            # Find left and right time coordinates for a given hamiltonian
            # value
            time_indexes = np.where(potential_well_low_res <=
                                    potential_well_low_res[j])[0]
            left_time = self._time_potential_low_res[np.max((0, time_indexes[0]))]
            right_time = self._time_potential_low_res[np.min((time_indexes[-1],
                                                              self.n_points_grid - 1))]
            # Potential well calculation with high resolution in that frame
            time_potential_high_res = np.linspace(float(left_time),
                                                  float(right_time),
                                                  self._n_points_potential)
            full_ring_and_rf2.potential_well_generation(
                n_points=self._n_points_potential,
                time_array=time_potential_high_res,
                main_harmonic_option=self.main_harmonic_option)
            pot_well_high_res = full_ring_and_rf2.potential_well

            if self._profile is not None and i != 0:
                induced_potential_hires = np.interp(
                    time_potential_high_res,
                    self._time_potential, self._induced_potential +
                                          self._extra_potential, left=0, right=0)
                pot_well_high_res += induced_potential_hires
                pot_well_high_res -= pot_well_high_res.min()

            # Integration to calculate action
            self._dE_trajectory[pot_well_high_res <= potential_well_low_res[j]] = \
                np.sqrt((potential_well_low_res[j]
                         - pot_well_high_res[pot_well_high_res
                                             <= potential_well_low_res[j]])
                        / self._eom_factor_dE)
            self._dE_trajectory[pot_well_high_res > potential_well_low_res[j]] = 0
            J_array_dE0[j] = (1 / np.pi
                              * np.trapezoid(self._dE_trajectory,
                                             dx=time_potential_high_res[1]
                                                - time_potential_high_res[0]))
        return J_array_dE0

    def _calc_induced_potential(self, time_resolution_low: float) -> NDArray:
        # Inputting new line density
        self._profile.cut_options.cut_left = (
                self._time_potential_low_res[0] - 0.5 * time_resolution_low)
        self._profile.cut_options.cut_right = (
                self._time_potential_low_res[-1] + 0.5 * time_resolution_low)
        self._profile.cut_options.n_slices = self.n_points_grid
        self._profile.cut_options.cuts_unit = 's'
        self._profile.cut_options.set_cuts()
        self._profile.set_slices_parameters()
        self._profile.n_macroparticles = self._line_density_
        # Re-calculating the sources of wakes/impedances according to this
        # slicing
        self._induced_voltage_object.reprocess()
        # Calculating the induced voltage
        self._induced_voltage_object.induced_voltage_sum()
        induced_voltage = self._induced_voltage_object.induced_voltage
        # Calculating the induced potential
        induced_potential_low_res = -(self._eom_factor_potential
                                      * cumtrapz(induced_voltage,
                                                 dx=time_resolution_low,
                                                 initial=0))
        induced_potential = np.interp(self._time_potential,
                                      self._time_potential_low_res,
                                      induced_potential_low_res,
                                      left=0, right=0)
        return induced_potential

    def _calc_eom(self,
                  full_ring_and_rf: FullRingAndRF,
                  turn_number: int
                  ) -> (float, float):
        """Initialize variables depending on the accelerator parameters"""
        slippage_factor = full_ring_and_rf.ring_and_rf_section[0].rf_params.eta_0[turn_number]
        beta = full_ring_and_rf.ring_and_rf_section[0].rf_params.beta[turn_number]
        energy = full_ring_and_rf.ring_and_rf_section[0].rf_params.energy[turn_number]
        eom_factor_dE = abs(slippage_factor) / (2 * beta ** 2. * energy)
        eom_factor_potential = (np.sign(slippage_factor) * self._beam.particle.charge
                                / (full_ring_and_rf.ring_and_rf_section[0].rf_params.t_rev[turn_number]))
        return eom_factor_dE, eom_factor_potential

    def _calc_extra_potential(self, extra_voltage_dict: dict[str, NDArray],
                              ) -> float | NDArray:
        """Extra potential from previous bunches (for multi-bunch generation)"""
        extra_potential = 0
        if extra_voltage_dict is not None:
            extra_voltage_time_input = extra_voltage_dict['time_array']
            extra_voltage_input = extra_voltage_dict['voltage_array']
            extra_potential_input = -(self._eom_factor_potential
                                      * cumtrapz(extra_voltage_input,
                                                 dx=(float(extra_voltage_time_input[1])
                                                     - float(extra_voltage_time_input[0])),
                                                 initial=0)
                                      )
            extra_potential = np.interp(self._time_potential, extra_voltage_time_input, extra_potential_input)
        return extra_potential

    def x0_from_emittance(self,
                          emittance: float,
                          sorted_H_dE0: NDArray,
                          sorted_J_dE0: NDArray
                          ) -> float:
        if self.distribution_variable == 'Action':
            X0 = emittance / (2 * np.pi)
        elif self.distribution_variable == 'Hamiltonian':
            X0 = np.interp(emittance / (2 * np.pi), sorted_J_dE0, sorted_H_dE0)
        else:
            raise RuntimeError(
                f"'distribution_variable' must be 'Action' or 'Hamiltonian', not '{self.distribution_variable}'")
        return X0
