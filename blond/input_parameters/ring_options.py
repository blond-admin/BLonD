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
from scipy.constants import c
from scipy.interpolate import splev, splrep

from ..plots.plot import fig_folder


class RingOptions:
    r""" Class to preprocess the synchronous data for Ring, interpolating it to
    every turn.

    Parameters
    ----------
    interpolation : str
        Interpolation options for the data points. Available options are
        'linear' (default), 'cubic', and 'derivative'
    smoothing : float
        Smoothing value for 'cubic' interpolation
    flat_bottom : int
        Number of turns to be added on flat bottom; default is 0. Constant
        extrapolation is used for the synchronous data
    flat_top : int
        Number of turns to be added on flat top; default is 0. Constant
        extrapolation is used for the synchronous data
    t_start : int
        Starting index from which the time array input should be taken into
        account; default is 0
    t_end : int
        Last index up to which the time array input should be taken into
        account; default is -1
    plot : bool
        Option to plot interpolated arrays; default is False
    figdir : str
        Directory to save optional plot; default is 'fig'
    figname : str
        Figure name to save optional plot; default is 'preprocess_ramp'
    sampling : int
        Decimation value for plotting; default is 1

    """

    def __init__(self, interpolation='linear', smoothing=0, flat_bottom=0,
                 flat_top=0, t_start=None, t_end=None, plot=False,
                 figdir='fig', figname='preprocess_ramp', sampling=1):

        if interpolation in ['linear', 'cubic', 'derivative']:
            self.interpolation = str(interpolation)
        else:
            # InputDataError
            raise RuntimeError("ERROR: Interpolation scheme in " +
                               "PreprocessRamp not recognised. Aborting...")

        self.smoothing = float(smoothing)

        if flat_bottom < 0:
            # MomentumError
            raise RuntimeError("ERROR: flat_bottom value in PreprocessRamp" +
                               " not recognised. Aborting...")
        else:
            self.flat_bottom = int(flat_bottom)

        if flat_top < 0:
            # MomentumError
            raise RuntimeError("ERROR: flat_top value in PreprocessRamp" +
                               " not recognised. Aborting...")
        else:
            self.flat_top = int(flat_top)

        self.t_start = t_start
        self.t_end = t_end

        if (plot is True) or (plot is False):
            self.plot = bool(plot)
        else:
            # TypeError
            raise RuntimeError("ERROR: plot value in PreprocessRamp" +
                               " not recognised. Aborting...")

        self.figdir = str(figdir)
        self.figname = str(figname)
        if sampling > 0:
            self.sampling = int(sampling)
        else:
            # TypeError
            raise RuntimeError("ERROR: sampling value in PreprocessRamp" +
                               " not recognised. Aborting...")

    def reshape_data(self, input_data, n_turns, n_sections,
                     interp_time='t_rev', input_to_momentum=False,
                     synchronous_data_type='momentum', mass=None, charge=None,
                     circumference=None, bending_radius=None):
        r"""Checks whether the user input is consistent with the expectation
        for the Ring object. The possibilites are detailed in the documentation
        of the Ring object.


        Parameters
        ----------
        input_data : Ring.synchronous_data, Ring.alpha_0,1,2
            Main input data to reshape
        n_turns : Ring.n_turns
            Number of turns the simulation should be. Note that if
            the input_data is passed as a tuple it is expected that the
            input_data is a program. Hence, the number of turns may not
            correspond to the input one and will be overwritten
        n_sections : Ring.n_sections
            The number of sections of the ring. The simulation is stopped
            if the input_data shape does not correspond to the expected number
            of sections.
        interp_time : str or float or float array [n_turns+1]
            Optional : defines the time on which the program will be
            interpolated. If 't_rev' is passed and if the input_data is
            momentum (see input_to_momentum option) the momentum program
            is interpolated on the revolution period (see preprocess()
            function). If a float or a float array is passed, the program
            is interpolated on that input ; default is 't_rev'
        input_to_momentum : bool
            Optional : flags if the input_data is the momentum program, the
            options defined below become necessary for conversion
        synchronous_data_type : str
            Optional : to be passed to the convert_data function if
            input_to_momentum ; default is 'momentum'
        mass : Ring.Particle.mass
            Optional : the mass of the particles in [eV/c**2] ; default is None
        charge : Ring.Particle.charge
            Optional : the charge of the particles in units of [e] ;
            default is None
        circumference : Ring.circumference
            Optional : the circumference of the ring ; default is None
        bending_radius : Ring.bending_radis
            Optional : the bending radius of magnets ; default is None

        Returns
        -------
        output_data
            Returns the data with the adequate shape for the Ring object

        """

        # TO BE IMPLEMENTED: if you pass a filename the function reads the file
        # and reshape the data
        if isinstance(input_data, str):
            pass

        # If single float, expands the value to match the input number of turns
        # and sections
        if isinstance(input_data, float) or isinstance(input_data, int):
            input_data = float(input_data)
            if input_to_momentum:
                input_data = convert_data(input_data, mass, charge,
                                          synchronous_data_type,
                                          bending_radius)
            output_data = input_data * np.ones((n_sections, n_turns + 1))

        # If tuple, separate time and synchronous data and check data
        elif isinstance(input_data, tuple):

            output_data = []

            # If there is only one section, it is expected that the user passes
            # a tuple with (time, data). However, the user can also pass a
            # tuple which size is the number of section as ((time, data), ).
            # and this if condition takes this into account
            if (n_sections == 1) and (len(input_data) > 1):
                input_data = (input_data, )

            if len(input_data) != n_sections:
                # InputDataError
                raise RuntimeError("ERROR in Ring: the input data " +
                                   "does not match the number of sections")

            # Loops over all the sections to interpolate the programs, appends
            # the results on the output_data list which is afterwards
            # converted to a numpy.array
            for index_section in range(n_sections):
                input_data_time = input_data[index_section][0]
                input_data_values = input_data[index_section][1]

                if input_to_momentum:
                    input_data_values = convert_data(input_data_values, mass,
                                                     charge,
                                                     synchronous_data_type,
                                                     bending_radius)

                if len(input_data_time) \
                        != len(input_data_values):
                    # InputDataError
                    raise RuntimeError("ERROR in Ring: synchronous data " +
                                       "does not match the time data")

                if input_to_momentum and (interp_time == 't_rev'):
                    output_data.append(self.preprocess(
                        mass,
                        circumference,
                        input_data_time,
                        input_data_values)[1])

                elif isinstance(interp_time, float) or \
                        isinstance(interp_time, int):
                    interp_time = float(interp_time)
                    interp_time = np.arange(
                        input_data_time[0],
                        input_data_time[-1],
                        interp_time)

                    output_data.append(np.interp(
                        interp_time,
                        input_data_time,
                        input_data_values))

                elif isinstance(interp_time, np.ndarray):
                    output_data.append(np.interp(
                        interp_time,
                        input_data_time,
                        input_data_values))

            output_data = np.array(output_data, ndmin=2, dtype=float)

        # If array/list, compares with the input number of turns and
        # if synchronous_data is a single value converts it into a (n_turns+1)
        # array
        elif isinstance(input_data, np.ndarray) or \
                isinstance(input_data, list):

            input_data = np.array(input_data, ndmin=2, dtype=float)

            if input_to_momentum:
                input_data = convert_data(input_data, mass, charge,
                                          synchronous_data_type,
                                          bending_radius)

            output_data = np.zeros((n_sections, n_turns + 1), dtype=float)

            # If the number of points is exactly the same as n_rf, this means
            # that the rf program for each harmonic is constant, reshaping
            # the array so that the size is [n_sections,1] for successful
            # reshaping
            if input_data.size == n_sections:
                input_data = input_data.reshape((n_sections, 1))

            if len(input_data) != n_sections:
                # InputDataError
                raise RuntimeError("ERROR in Ring: the input data " +
                                   "does not match the number of sections")

            for index_section in range(len(input_data)):
                if len(input_data[index_section]) == 1:
                    output_data[index_section] = input_data[index_section] * \
                        np.ones(n_turns + 1)

                elif len(input_data[index_section]) == (n_turns + 1):
                    output_data[index_section] = np.array(
                        input_data[index_section])

                else:
                    # InputDataError
                    raise RuntimeError("ERROR in Ring: The input data " +
                                       "does not match the proper length " +
                                       "(n_turns+1)")

        return output_data

    def preprocess(self, mass, circumference, time, momentum):
        r"""Function to pre-process acceleration ramp data, interpolating it to
        every turn. Currently it works only if the number of RF sections is
        equal to one, to be extended for multiple RF sections.

        Parameters
        ----------
        mass : float
            Particle mass [eV]
        circumference : float
            Ring circumference [m]
        time : float array
            Time points [s] corresponding to momentum data
        momentum : float array
            Particle momentum [eV/c]

        Returns
        -------
        float array
            Cumulative time [s]
        float array
            Interpolated momentum [eV/c]

        """

        # Some checks on the options
        if ((self.t_start is not None) and (self.t_start < time[0])) or \
                ((self.t_end is not None) and (self.t_end > time[-1])):
            # InputDataError
            raise RuntimeError("ERROR: [t_start, t_end] should be " +
                               "included in the passed time array.")

        # Obtain flat bottom data, extrapolate to constant
        beta_0 = np.sqrt(1 / (1 + (mass / momentum[0])**2))
        T0 = circumference / (beta_0 * c)  # Initial revolution period [s]
        shift = time[0] - self.flat_bottom * T0
        time_interp = shift + T0 * np.arange(0, self.flat_bottom + 1)
        beta_interp = beta_0 * np.ones(self.flat_bottom + 1)
        momentum_interp = momentum[0] * np.ones(self.flat_bottom + 1)

        time_interp = time_interp.tolist()
        beta_interp = beta_interp.tolist()
        momentum_interp = momentum_interp.tolist()

        time_start_ramp = np.max(time[momentum == momentum[0]])
        time_end_ramp = np.min(time[momentum == momentum[-1]])

        # Interpolate data recursively
        if self.interpolation == 'linear':

            time_interp.append(time_interp[-1]
                               + circumference / (beta_interp[0] * c))

            i = self.flat_bottom
            for k in range(1, len(time)):

                while time_interp[i + 1] <= time[k]:

                    momentum_interp.append(
                        momentum[k - 1] + (momentum[k] - momentum[k - 1]) *
                        (time_interp[i + 1] - time[k - 1]) /
                        (time[k] - time[k - 1]))

                    beta_interp.append(
                        np.sqrt(1 / (1 + (mass / momentum_interp[i + 1])**2)))

                    time_interp.append(
                        time_interp[i + 1] + circumference / (beta_interp[i + 1] * c))

                    i += 1

        elif self.interpolation == 'cubic':

            interp_funtion_momentum = splrep(
                time[(time >= time_start_ramp) * (time <= time_end_ramp)],
                momentum[(time >= time_start_ramp) * (time <= time_end_ramp)],
                s=self.smoothing)

            i = self.flat_bottom

            time_interp.append(
                time_interp[-1] + circumference / (beta_interp[0] * c))

            while time_interp[i] <= time[-1]:

                if (time_interp[i + 1] < time_start_ramp):

                    momentum_interp.append(momentum[0])

                    beta_interp.append(
                        np.sqrt(1 / (1 + (mass / momentum_interp[i + 1])**2)))

                    time_interp.append(
                        time_interp[i + 1] + circumference / (beta_interp[i + 1] * c))

                elif (time_interp[i + 1] > time_end_ramp):

                    momentum_interp.append(momentum[-1])

                    beta_interp.append(
                        np.sqrt(1 / (1 + (mass / momentum_interp[i + 1])**2)))

                    time_interp.append(
                        time_interp[i + 1] + circumference / (beta_interp[i + 1] * c))

                else:

                    momentum_interp.append(
                        splev(time_interp[i + 1], interp_funtion_momentum))

                    beta_interp.append(
                        np.sqrt(1 / (1 + (mass / momentum_interp[i + 1])**2)))

                    time_interp.append(
                        time_interp[i + 1] + circumference / (beta_interp[i + 1] * c))

                i += 1

        # Interpolate momentum in 1st derivative to maintain smooth B-dot
        elif self.interpolation == 'derivative':

            momentum_initial = momentum_interp[0]
            momentum_derivative = np.gradient(momentum) / np.gradient(time)

            momentum_derivative_interp = [0] * self.flat_bottom + \
                [momentum_derivative[0]]
            integral_point = momentum_initial

            i = self.flat_bottom

            time_interp.append(
                time_interp[-1] + circumference / (beta_interp[0] * c))

            while time_interp[i] <= time[-1]:

                derivative_point = np.interp(time_interp[i + 1], time,
                                             momentum_derivative)
                momentum_derivative_interp.append(derivative_point)
                integral_point += (time_interp[i + 1] - time_interp[i]) \
                    * derivative_point

                momentum_interp.append(integral_point)
                beta_interp.append(
                    np.sqrt(1 / (1 + (mass / momentum_interp[i + 1])**2)))

                time_interp.append(
                    time_interp[i + 1] + circumference / (beta_interp[i + 1] * c))

                i += 1

            # Adjust result to get flat top energy correct as derivation and
            # integration leads to ~10^-8 error in flat top momentum
            momentum_interp = np.asarray(momentum_interp)
            momentum_interp -= momentum_interp[0]
            momentum_interp /= momentum_interp[-1]
            momentum_interp *= momentum[-1] - momentum[0]

            momentum_interp += momentum[0]

        time_interp.pop()
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)

        # Obtain flat top data, extrapolate to constant
        if self.flat_top > 0:
            time_interp = np.append(
                time_interp,
                time_interp[-1] + circumference * np.arange(1, self.flat_top + 1)
                / (beta_interp[-1] * c))

            beta_interp = np.append(
                beta_interp, beta_interp[-1] * np.ones(self.flat_top))

            momentum_interp = np.append(
                momentum_interp,
                momentum_interp[-1] * np.ones(self.flat_top))

        # Cutting the input momentum on the desired cycle time
        if self.t_start is not None:
            initial_index = np.min(np.where(time_interp >= self.t_start)[0])
        else:
            initial_index = 0
        if self.t_end is not None:
            final_index = np.max(np.where(time_interp <= self.t_end)[0]) + 1
        else:
            final_index = len(time_interp)
        time_interp = time_interp[initial_index:final_index]
        momentum_interp = momentum_interp[initial_index:final_index]

        if self.plot:
            # Directory where longitudinal_plots will be stored
            fig_folder(self.figdir)

            # Plot
            plt.figure(1, figsize=(8, 6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(time_interp[::self.sampling],
                    momentum_interp[::self.sampling],
                    label='Interpolated momentum')
            ax.plot(time, momentum, '.', label='input momentum', color='r',
                    markersize=0.5)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("p [eV]")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                   ncol=2, mode="expand", borderaxespad=0.)

            # Save figure
            fign = self.figdir + '/preprocess_' + self.figname
            plt.savefig(fign)
            plt.clf()

        return time_interp, momentum_interp


def convert_data(synchronous_data, mass, charge,
                 synchronous_data_type='momentum', bending_radius=None):
    """ Function to convert synchronous data (i.e. energy program of the
    synchrotron) into momentum.

    Parameters
    ----------
    synchronous_data : float array
        The synchronous data to be converted to momentum
    mass : float or Particle.mass
        The mass of the particles in [eV/c**2]
    charge : int or Particle.charge
        The charge of the particles in units of [e]
    synchronous_data_type : str
        Type of input for the synchronous data ; can be 'momentum',
        'total energy', 'kinetic energy' or 'bending field' (last case
        requires bending_radius to be defined)
    bending_radius : float
        Bending radius in [m] in case synchronous_data_type is
        'bending field'

    Returns
    -------
    momentum : float array
        The input synchronous_data converted into momentum [eV/c]

    """

    if synchronous_data_type == 'momentum':
        momentum = synchronous_data
    elif synchronous_data_type == 'total energy':
        momentum = np.sqrt(synchronous_data**2 - mass**2)
    elif synchronous_data_type == 'kinetic energy':
        momentum = np.sqrt((synchronous_data + mass)**2 - mass**2)
    elif synchronous_data_type == 'bending field':
        if bending_radius is None:
            # InputDataError
            raise RuntimeError("ERROR in Ring: bending_radius is not " +
                               "defined and is required to compute " +
                               "momentum")
        momentum = synchronous_data * bending_radius * charge * c
    else:
        # InputDataError
        raise RuntimeError("ERROR in Ring: Synchronous data" +
                           " type not recognized!")

    return momentum


def load_data(filename, ignore=0, delimiter=None):
    r"""Helper function to load column-by-column data from a txt file to numpy
    arrays.

    Parameters
    ----------
    filename : str
        Name of the file containing the data.
    ignore : int
        Number of lines to ignore from the head of the file.
    delimiter : str
        Delimiting character between columns.

    Returns
    -------
    list of arrays
        Input data, column by column.

    """

    data = np.loadtxt(str(filename), skiprows=int(ignore),
                      delimiter=str(delimiter))

    return [np.ascontiguousarray(data[:, i]) for i in range(len(data[0]))]
