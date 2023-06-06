# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Helga Timko**, **Alexandre Lasheen**, **Danilo Quartullo**,
    **Simon Albright**
'''

from __future__ import division

from builtins import range, str

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep

from ..plots.plot import fig_folder


class RFStationOptions:
    r""" Class to preprocess the RF data (voltage, phase, harmonic) for
    RFStation, interpolating it to every turn.

    Parameters
    ----------
    interpolation : str
        Interpolation options for the data points. Available options are
        'linear' (default) and 'cubic'
    smoothing : float
        Smoothing value for 'cubic' interpolation
    plot : bool
        Option to plot interpolated arrays; default is False
    figdir : str
        Directory to save optional plot; default is 'fig'
    figname : list of str
        Figure name to save optional plot; default is 'data', different arrays
        will have figures with different indices
    sampling : int
        Decimation value for plotting; default is 1

    """

    def __init__(self, interpolation='linear', smoothing=0, plot=False,
                 figdir='fig', figname=['data'], sampling=1):

        if interpolation in ['linear', 'cubic']:
            self.interpolation = str(interpolation)
        else:
            # InterpolationError
            raise RuntimeError(
                "ERROR: Interpolation scheme in" +
                " RFStationOptions not recognised. Aborting...")

        self.smoothing = float(smoothing)

        if isinstance(plot, bool):
            self.plot = bool(plot)
        else:
            # TypeError
            raise RuntimeError("ERROR: plot value in PreprocessRamp" +
                               " not recognised. Aborting...")

        self.figdir = str(figdir)
        self.figname = figname  # str(figname)

        if sampling > 0:
            self.sampling = int(sampling)
        else:
            # TypeError
            raise RuntimeError("ERROR: sampling value in PreprocessRamp" +
                               " not recognised. Aborting...")

    def reshape_data(self, input_data, n_turns, n_rf, interp_time,
                     t_start=0):
        r"""Checks whether the user input is consistent with the expectation
        for the RFStation object. The possibilites are detailed in the
        documentation of the RFStation object.

        Parameters
        ----------
        input_data : Ring.synchronous_data, Ring.alpha_0,1,2
            Main input data to reshape
        n_turns : RFStation.n_turns
            Number of turns the simulation should be. Note that if
            the input_data is passed as a tuple it is expected that the
            input_data is a program. This parameter is only relevant if the
            rf program is passed as an array/list with the rigth size
            (see interp_time otherwise)
        n_rf : RFStation.n_rf
            The number of rf harmonics in the station. The simulation is
            stopped if the input_data shape does not correspond to the expected
            number of rf harmonics.
        interp_time : Ring.cycle_time
            Ensure that the rf program is interpolated on the same time basis
            as the Ring.momentum program
        t_start : Ring.RingOptions.t_start or float
            Uses the same t_start as the one used to interpolate the momentum
            program in Ring. This value can nevertheless be changed to a custom
            value if necessary.

        Returns
        -------
        output_data
            Returns the data with the adequate shape for the RStation object

        """

        # TO BE IMPLEMENTED: if you pass a filename the function reads the file
        # and reshape the data
        if isinstance(input_data, str):
            pass

        # If single float, expands the value to match the input number of turns
        # and rf harmonics
        if isinstance(input_data, float) or isinstance(input_data, int):
            output_data = input_data * np.ones((n_rf, n_turns + 1))

        # If tuple, separate time and synchronous data and check data
        elif isinstance(input_data, tuple):

            output_data = []

            # Hot fix to safely treat t_start
            if t_start is None:
                t_start = 0

            interp_time = interp_time + t_start

            # If there is only one rf harmonic, it is expected that the user
            # passes a tuple with (time, data). However, the user can also pass
            # a tuple which size is the number of section as ((time, data), ).
            # and this if condition takes this into account
            if (n_rf == 1) and (len(input_data) > 1):
                input_data = (input_data, )

            if len(input_data) != n_rf:
                # InputDataError
                raise RuntimeError("ERROR in RFStation: the input data " +
                                   "does not match the number of rf harmonics")

            # Loops over all the rf harmonics to interpolate the programs,
            # appends the results on the output_data list which is afterwards
            # converted to a numpy.array
            for index_rf in range(n_rf):
                input_data_time = input_data[index_rf][0]
                input_data_values = input_data[index_rf][1]

                if len(input_data_values) \
                        != len(input_data_time):
                    # InputDataError
                    raise RuntimeError("ERROR in RFStation: synchronous " +
                                       "data does not match the time data")

                if self.interpolation == 'linear':
                    output_data.append(np.interp(interp_time,
                                                 input_data_time,
                                                 input_data_values))
                elif self.interpolation == 'cubic':
                    interp_funtion = splrep(input_data_time, input_data_values,
                                            s=self.smoothing)
                    output_data.append(splev(interp_time, interp_funtion))

            output_data = np.array(output_data, ndmin=2, dtype=float)

            # Plot original and interpolated data
            if self.plot:
                # Directory where plots will be stored
                fig_folder(self.figdir)

                # Plot
                for index_rf in range(n_rf):
                    input_data_time = input_data[index_rf][0]
                    input_data_values = input_data[index_rf][1]

                    plt.figure('RFStationOptions', figsize=(8, 6))
                    plt.clf()
                    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
                    ax.plot(interp_time[::self.sampling],
                            output_data[index_rf][::self.sampling],
                            label='Interpolated data')
                    ax.plot(input_data_time, input_data_values, '.',
                            label='Input data', color='r')
                    ax.set_xlabel('Time [s]')
                    ax.set_ylabel("%s" % self.figname[index_rf])
                    ax.legend = plt.legend(
                        bbox_to_anchor=(0., 1.02, 1., .102),
                        loc=3, ncol=2, mode='expand', borderaxespad=0.)

                    # Save figure
                    fign = self.figdir + '/preprocess_' "%s" % self.figname[index_rf] + \
                        '.png'
                    plt.savefig(fign)

        # If array/list, compares with the input number of turns and
        # if synchronous_data is a single value converts it into a (n_turns+1)
        # array
        elif isinstance(input_data, np.ndarray) or \
                isinstance(input_data, list):

            input_data = np.array(input_data, ndmin=2, dtype=float)
            output_data = np.zeros((n_rf, n_turns + 1), dtype=float)

            # If the number of points is exactly the same as n_rf, this means
            # that the rf program for each harmonic is constant, reshaping
            # the array so that the size is [n_sections,1] for successful
            # reshaping
            if input_data.size == n_rf:
                input_data = input_data.reshape((n_rf, 1))

            if len(input_data) != n_rf:
                # InputDataError
                raise RuntimeError("ERROR in RFStation: the input data " +
                                   "does not match the number of rf harmonics")

            for index_rf in range(len(input_data)):
                if len(input_data[index_rf]) == 1:
                    output_data[index_rf] = input_data[index_rf] * \
                        np.ones(n_turns + 1)

                elif len(input_data[index_rf]) == (n_turns + 1):
                    output_data[index_rf] = np.array(
                        input_data[index_rf])

                else:
                    # InputDataError
                    raise RuntimeError("ERROR in Ring: The input data " +
                                       "does not match the proper length " +
                                       "(n_turns+1)")

        return output_data


def combine_rf_functions(function_list, merge_type='linear', resolution=1e-3,
                         Ring=None, main_h=True):
    r"""Function to combine different RF programs. Each program is passed in a
    tuple with complete function (single valued or numpy array) and 2-list
    [start_time, stop_time].

    Parameters
    ----------
    function_list : list of tuples
        each tuple has form (function, [start_time, stop_time])
        function can be a numpy.ndarray of format [time, value] or single valued
        if function is single valued it will be assumed constant from start_time to stop_time
        if function is numpy.ndarray it will be truncated to start_time, stop_time
    merge_type : str
        string signifying type of merge available, options are:
            linear : function will be linearly interpolated from function_1[stop_time] to function_2[start_time]
            isoadiabatic : designed for voltage functions and intended to maintain adiabaticity during change of voltage, best suited to flat momentum sections
            linear_tune : for use with voltages, provides a linear change in the tune from function_1[stop_time] to function_2[start_time]
    resolution : float
        the time in seconds between points of the interpolation
    Ring : class
        A Ring type class, only used with linear_tune merge_type
    main_h : boolean
        if main_h is True dE is considered in linear_tune merge_type, otherwise dE is set to 0

    Returns
    -------
    2 dimensional numpy.ndarray containing [time, value] of merged functions

    """

    nFunctions = len(function_list)

    if not isinstance(merge_type, list):
        merge_type = (nFunctions - 1) * [merge_type]
    if not isinstance(resolution, list):
        resolution = (nFunctions - 1) * [resolution]

    if len(merge_type) != nFunctions:
        # InputDataError
        raise RuntimeError("ERROR: merge_type list wrong length")
    if len(resolution) != nFunctions:
        # InputDataError
        raise RuntimeError("ERROR: resolution list wrong length")

    timePoints = []
    for i in range(nFunctions):
        timePoints += function_list[i][1]
    if not np.all(np.diff(timePoints)) > 0:
        # InputDataError
        raise RuntimeError("ERROR: in combine_rf_functions, times are not" +
                           " monotonically increasing!")

    fullFunction = []
    fullTime = []

    # Determines if 1st function is single valued or array and stores values
    if not isinstance(function_list[0][0], np.ndarray):
        fullFunction += 2 * [function_list[0][0]]
        fullTime += function_list[0][1]

    else:
        start = np.where(function_list[0][0][0] > function_list[0][1][0])[0][0]
        stop = np.where(function_list[0][0][0] > function_list[0][1][1])[0][0]

        funcTime = [function_list[0][1][0]] + \
            function_list[0][0][0][start:stop].tolist() + \
            [function_list[0][1][1]]
        funcProg = np.interp(funcTime, function_list[0][0][0],
                             function_list[0][0][1])

        fullFunction += funcProg.tolist()
        fullTime += funcTime

    # Loops through remaining functions merging them as requested and
    # storing results
    for i in range(1, nFunctions):

        if merge_type[i - 1] == 'linear':

            if not isinstance(function_list[i][0], np.ndarray):
                fullFunction += 2 * [function_list[i][0]]
                fullTime += function_list[i][1]

            else:
                start = np.where(function_list[i][0][0] >=
                                 function_list[i][1][0])[0][0]
                stop = np.where(function_list[i][0][0] >=
                                function_list[i][1][1])[0][0]

                funcTime = [function_list[i][1][0]] + \
                    function_list[i][0][0][start:stop].tolist() + \
                    [function_list[i][1][1]]
                funcProg = np.interp(funcTime, function_list[i][0][0],
                                     function_list[i][0][1])

                fullFunction += funcProg.tolist()
                fullTime += funcTime

        elif merge_type[i - 1] == 'isoadiabatic':

            if not isinstance(function_list[i][0], np.ndarray):

                tDur = function_list[i][1][0] - fullTime[-1]
                Vinit = fullFunction[-1]
                Vfin = function_list[i][0]
                k = (1. / tDur) * (1 - (1. * Vinit / Vfin)**0.5)

                nSteps = int(tDur / resolution[i - 1])
                time = np.linspace(float(fullTime[-1]),
                                   float(function_list[i][1][0]), nSteps)
                volts = Vinit / ((1 - k * (time - time[0]))**2)

                fullFunction += volts.tolist() + 2 * [function_list[i][0]]
                fullTime += time.tolist() + function_list[i][1]

            else:

                start = np.where(function_list[i][0][0] >=
                                 function_list[i][1][0])[0][0]
                stop = np.where(function_list[i][0][0] >=
                                function_list[i][1][1])[0][0]

                funcTime = [function_list[i][1][0]] + \
                    function_list[i][0][0][start:stop].tolist() + \
                    [function_list[i][1][1]]
                funcProg = np.interp(funcTime, function_list[i][0][0],
                                     function_list[i][0][1])

                tDur = funcTime[0] - fullTime[-1]
                Vinit = fullFunction[-1]
                Vfin = funcProg[0]
                k = (1. / tDur) * (1 - (1. * Vinit / Vfin)**0.5)

                nSteps = int(tDur / resolution[i - 1])
                time = np.linspace(float(fullTime[-1]), float(funcTime[0]), nSteps)
                volts = Vinit / ((1 - k * (time - time[0]))**2)

                fullFunction += volts.tolist() + funcProg.tolist()
                fullTime += time.tolist() + funcTime

        elif merge_type[i - 1] == 'linear_tune':

            # harmonic, charge and 2pi are constant so can be ignored
            if not isinstance(function_list[i][0], np.ndarray):

                initPars = Ring.parameters_at_time(fullTime[-1])
                finalPars = Ring.parameters_at_time(function_list[i][1][0])

                vInit = fullFunction[-1]
                vFin = function_list[i][0]

                if main_h is False:
                    initPars['delta_E'] = 0.
                    finalPars['delta_E'] = 0.

                initTune = np.sqrt(
                    (vInit * np.abs(initPars['eta_0']) *
                     np.sqrt(1 - (initPars['delta_E'] / vInit)**2)) /
                    (initPars['beta']**2 * initPars['energy']))

                finalTune = np.sqrt(
                    (vFin * np.abs(finalPars['eta_0']) *
                     np.sqrt(1 - (finalPars['delta_E'] / vFin)**2)) /
                    (finalPars['beta']**2 * finalPars['energy']))

                tDur = function_list[i][1][0] - fullTime[-1]
                nSteps = int(tDur / resolution[i - 1])
                time = np.linspace(float(fullTime[-1]), float(function_list[i][1][0]),
                                   nSteps)
                tuneInterp = np.linspace(float(initTune), float(finalTune), nSteps)

                mergePars = Ring.parameters_at_time(time)

                if main_h is False:
                    mergePars['delta_E'] *= 0

                volts = np.sqrt(
                    ((tuneInterp**2 * mergePars['beta']**2 *
                      mergePars['energy']) / (np.abs(mergePars['eta_0'])))**2 +
                    mergePars['delta_E']**2)

                fullFunction += volts.tolist() + 2 * [function_list[i][0]]
                fullTime += time.tolist() + function_list[i][1]

            else:

                start = np.where(function_list[i][0][0] >=
                                 function_list[i][1][0])[0][0]
                stop = np.where(function_list[i][0][0] >=
                                function_list[i][1][1])[0][0]

                funcTime = [function_list[i][1][0]] + \
                    function_list[i][0][0][start:stop].tolist() + \
                    [function_list[i][1][1]]
                funcProg = np.interp(funcTime, function_list[i][0][0],
                                     function_list[i][0][1])

                tDur = funcTime[0] - fullTime[-1]
                nSteps = int(tDur / resolution[i - 1])
                time = np.linspace(float(fullTime[-1]), float(funcTime[0]), nSteps)

                initPars = Ring.parameters_at_time(fullTime[-1])
                finalPars = Ring.parameters_at_time(funcTime[0])

                if main_h is False:
                    initPars['delta_E'] = 0.
                    finalPars['delta_E'] = 0.

                vInit = fullFunction[-1]
                vFin = funcProg[0]

                initTune = np.sqrt(
                    (vInit * np.abs(initPars['eta_0']) *
                     np.sqrt(1 - (initPars['delta_E'] / vInit)**2)) /
                    (initPars['beta']**2 * initPars['energy']))

                finalTune = np.sqrt(
                    (vFin * np.abs(finalPars['eta_0']) *
                     np.sqrt(1 - (finalPars['delta_E'] / vFin)**2)) /
                    (finalPars['beta']**2 * finalPars['energy']))

                tuneInterp = np.linspace(float(initTune), float(finalTune), nSteps)

                mergePars = Ring.parameters_at_time(time)

                if main_h is False:
                    mergePars['delta_E'] *= 0

                volts = np.sqrt(
                    ((tuneInterp**2 * mergePars['beta']**2 *
                      mergePars['energy']) / (np.abs(mergePars['eta_0'])))**2 +
                    mergePars['delta_E']**2)

                fullFunction += volts.tolist() + funcProg.tolist()
                fullTime += time.tolist() + funcTime

        else:
            # InputDataError
            raise RuntimeError("ERROR: merge_type not recognised")

    returnFunction = np.zeros([2, len(fullTime)])
    returnFunction[0] = fullTime
    returnFunction[1] = fullFunction

    return returnFunction
