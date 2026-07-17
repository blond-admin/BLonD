
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to plot different bunch features**

:Authors: **Helga Timko**, **Danilo Quartullo**
'''

from __future__ import division

from builtins import range

import matplotlib.pyplot as plt
import numpy as np

from ..impedances.impedance_sources import InputTable
import matplotlib.lines as mlines

# Automatic layouting of plots
plt.rc('figure', autolayout=True)
plt.rc('savefig', bbox='tight')
plt.rc('savefig', pad_inches=0.1)
plt.rc('figure', figsize = [8,6])

def plot_impedance_vs_frequency(induced_voltage_freq, figure_index = 0,
                                plot_total_impedance = True, plot_spectrum=False,
                                plot_interpolated_impedances = False, style='-',
                                cut_left_right=None, cut_up_down=None,
                                dirname='fig', show_plots = False):
    """
    Plots Impedance in frequency domain. Given an InducedVoltageFreq object, i.e. one
    that was created from a list of impedances:
     - can plot either the total, summed impedance or each individual one from the list
     - can additionally plot the beam spectrum
     - and can plot the impedances on an interpolated frequency axis, if the table
        given to induced_voltage_time has been interpolated on a new axis

    Additionally, either outputs the plots in the console, or save them on disk.

    :param induced_voltage_freq:  InducedVoltageFreq object
        InducedVoltageFreq object which impedances are to be plotted
    :param figure_index:  int
        Index of the figure, e.g. the turn number if creating a plot for every turn
        Default: 0
    :param plot_total_impedance: bool
        If True, plots the total, summed impedance which is calculated internally in the
        induced_voltage_time.
        If False, plots the raw impedances from the table given to induced_voltage_time
        Default: True
    :param plot_spectrum:  bool
        If True, plots the beam spectrum in addition to the impedances.
        If False, no beam spectrum is plotted
        Default: False
    :param plot_interpolated_impedances: bool
        If True, the impedance table in induced_voltage_time must have been interpolated
        to a new frequency axis. Will then plot the impedances on that new frequency axis.
        If False, plots the raw impedances on the original frequency axis
        given by the table in induced_voltage_time
        Default: False
    :param style: str
        Matplotlib plot style string
        Default: "-"
    :param cut_left_right: 2-tuple of floats
        x-axis limits of the plots (frequency in Hz)
        Default: None
    :param cut_up_down: 2-tuple of floats
        y-axis limits of the plots (resistance in Ohm)
        Default: None
    :param dirname: str
        Path to directory in which the plots are to be saved.
        Default: "fig"
    :param show_plots: bool
        If True, will output the plots directly in the console
        If False, will save the plots to directory given in dirname
    """

    if plot_total_impedance:

        ax1 = plt.subplots()[1]
        # Renormalize Impedances to the cont. Fourier Transform values, i.e. Ohm.
        # In the InducedVoltage Objects they are normalized to the DFT values by deviding
        # by bin size. This is reverted here when plotting to get sensible units

        ax1.plot(induced_voltage_freq.freq,
                 induced_voltage_freq.total_impedance.real * induced_voltage_freq.profile.bin_size, style, label = "Real Impedance")
        ax1.plot(induced_voltage_freq.freq,
                 induced_voltage_freq.total_impedance.imag * induced_voltage_freq.profile.bin_size, style, label = "Imaginary Impedance")

        ax1.set_xlim(cut_left_right)
        ax1.set_ylim(cut_up_down)
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Impedance [Ohm]")
        if plot_spectrum:
            ax2 = ax1.twinx()
            ax2.plot(induced_voltage_freq.profile.beam_spectrum_freq,
                     np.abs(induced_voltage_freq.profile.beam_spectrum), label = "Beam Spectrum")

            ax2.set_xlim(cut_left_right)
            ax2.set_ylim(cut_up_down)
            ax2.set_ylabel("Beam Spectrum [a.u.]")
        if plot_spectrum:
            ax2.legend()
        else:
            ax1.legend()
        fign = dirname + '/sum_imp_vs_freq_fft' "%d" % figure_index + '.png'

        if show_plots:
            plt.show()
        else:
            plt.savefig(fign)
            plt.clf()

    else:

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        # The impedance sources themselves are properly normalized already.
        for i in range(len(induced_voltage_freq.impedance_source_list)):
            if isinstance(induced_voltage_freq.impedance_source_list[i],
                          InputTable) and not plot_interpolated_impedances:
                ax0.plot(induced_voltage_freq.impedance_source_list[i].frequency_array_loaded,
                         induced_voltage_freq.impedance_source_list[i].Re_Z_array_loaded, style)
                ax0.set_xlim(cut_left_right)
                ax0.set_ylim(cut_up_down)
                ax1.plot(induced_voltage_freq.impedance_source_list[i].frequency_array_loaded,
                         induced_voltage_freq.impedance_source_list[i].Im_Z_array_loaded, style)
                ax1.set_xlim(cut_left_right)
                ax1.set_ylim(cut_up_down)
            elif plot_interpolated_impedances:
                ax0.plot(induced_voltage_freq.impedance_source_list[i].frequency_array,
                         induced_voltage_freq.impedance_source_list[i].impedance.real, style)
                ax0.set_xlim(cut_left_right)
                ax0.set_ylim(cut_up_down)
                ax1.plot(induced_voltage_freq.impedance_source_list[i].frequency_array,
                         induced_voltage_freq.impedance_source_list[i].impedance.imag, style)
                ax1.set_xlim(cut_left_right)
                ax1.set_ylim(cut_up_down)

        if plot_interpolated_impedances:
            fig_suffix = "interpolated_freq"
        else:
            fig_suffix = "table_freq"

        ax0.set_xlabel("Frequency [Hz]")
        ax0.set_ylabel("Real impedance [Ohm]")

        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Imaginary Impedance [Ohm]")


        fign1 = dirname + '/real_imp_vs_' + fig_suffix + '_' "%d" % figure_index + '.png'
        if plot_spectrum:
            ax2 = ax0.twinx()
            spectrum, = ax2.plot(induced_voltage_freq.profile.beam_spectrum_freq,
                     np.abs(induced_voltage_freq.profile.beam_spectrum), label = "Beam Spectrum",color = "r")
            ax2.set_xlim(cut_left_right)
            ax2.set_ylim(cut_up_down)
            ax2.set_ylabel("Beam Spectrum [a.u.]")
            ax2.legend(handles=[spectrum])



        plt.figure(0)
        if show_plots:
            plt.show()
        else:
            plt.savefig(fign1)
            plt.clf()
        fign2 = dirname + '/imag_imp_vs_' + fig_suffix + '_' "%d" % figure_index + '.png'


        if plot_spectrum:
            ax3 = ax1.twinx()
            spectrum, = ax3.plot(induced_voltage_freq.profile.beam_spectrum_freq,
                     np.abs(induced_voltage_freq.profile.beam_spectrum), label = "Beam Spectrum", color = "r")
            ax3.set_xlim(cut_left_right)
            ax3.set_ylim(cut_up_down)
            ax3.set_ylabel("Beam Spectrum [a.u.]")

            ax3.legend(handles=[spectrum])
        plt.figure(1)
        if show_plots:
            plt.show()
        else:
            plt.savefig(fign2)
            plt.clf()

def plot_wake_vs_time(induced_voltage_time, figure_index = 0,
                      plot_total_wake = True,
                      plot_interpolated_wake = False, style='-',
                      cut_left_right=None, cut_up_down=None,
                      dirname='fig', show_plots = False):
    """
    Plots wakes in time domain. Given an InducedVoltageTime object, i.e. one
    that was created from a list of wakes:
     - can plot either the total, summed wake or each individual one from the list
     - and can plot the wakes  on an interpolated wake axis, if the table
        given to induced_voltage_time has been interpolated on a new axis

    Additionally, either outputs the plots in the console, or save them on disk.

    :param induced_voltage_time:  InducedVoltageTime object
        InducedVoltageTime object which wakes are to be plotted
    :param figure_index:  int
        Index of the figure, e.g. the turn number if creating a plot for every turn
        Default: 0
    :param plot_total_wake: bool
        If True, plots the total, summed wake which is calculated internally in the
        induced_voltage_time.
        If False, plots the raw wakes from the table of wakes given to induced_voltage_time
        Default: True
    :param plot_interpolated_wake: bool
        If True, the wake table in induced_voltage_time must have been interpolated
        to a new frequency axis. Will then plot the wakes on that new frequency axis.
        If False, plots the raw wakes on the original frequency axis
        given by the table in induced_voltage_time
        Default: False
    :param style: str
        Matplotlib plot style string
        Default: "-"
    :param cut_left_right: 2-tuple of floats
        x-axis limits of the plots (frequency in Hz)
        Default: None
    :param cut_up_down: 2-tuple of floats
        y-axis limits of the plots (resistance in Ohm)
        Default: None
    :param dirname: str
        Path to directory in which the plots are to be saved.
        Default: "fig"
    :param show_plots: bool
        If True, will output the plots directly in the console
        If False, will save the plots to directory given in dirname
    """

    if plot_total_wake:

        ax1 = plt.subplots()[1]

        ax1.plot(induced_voltage_time.time,
                 induced_voltage_time.total_wake, style, label ="Wake")

        ax1.set_xlim(cut_left_right)
        ax1.set_ylim(cut_up_down)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Wake [Ohm/s]")
        ax1.legend()
        fign = dirname + '/sum_wake_vs_table_times' "%d" % figure_index + '.png'

        if show_plots:
            plt.show()
        else:
            plt.savefig(fign)
            plt.clf()

    else:

        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)

        for i in range(len(induced_voltage_time.wake_source_list)):
            if isinstance(induced_voltage_time.wake_source_list[i],
                          InputTable) and not plot_interpolated_wake:
                ax0.plot(induced_voltage_time.wake_source_list[i].time_array,
                         induced_voltage_time.wake_source_list[i].wake_array, style)
                ax0.set_xlim(cut_left_right)
                ax0.set_ylim(cut_up_down)
            elif plot_interpolated_wake:
                ax0.plot(induced_voltage_time.wake_source_list[i].new_time_array,
                         induced_voltage_time.wake_source_list[i].wake, style)
                ax0.set_xlim(cut_left_right)
                ax0.set_ylim(cut_up_down)

        if plot_interpolated_wake:
            fig_suffix = "interpolated_times"
        else:
            fig_suffix = "table_times"

        ax0.set_xlabel("Time [s]")
        ax0.set_ylabel("Wake [Ohm/s]")


        fign1 = dirname + '/wake_vs_' + fig_suffix + '_' "%d" % figure_index + '.png'



        plt.figure(0)
        if show_plots:
            plt.show()
        else:
            plt.savefig(fign1)
            plt.clf()

def plot_induced_voltage_vs_bin_centers(total_induced_voltage, style='-', figure_index = 0,
                                        dirname='fig', show_plots = False):
    """
    Plots the total induced voltage calculated in the TotalInducedVoltage object given.

    :param total_induced_voltage: TotalInducedVoltage object
        TotalInducedVoltage object containing the total induced voltage to be plotted
    :param style: str
        Matplotlib style string
        Default: "-"
    :param figure_index: int
        Index of the figure, e.g. the turn number if creating a plot for every turn
        Default: 0
    :param dirname: str
        Directory to save the plots to
        Default: "fig"
    :param show_plots: bool
        If True, outputs the plots directly to console
        If False, saved the plots to disk in given directory
        Default: False
    """

    fig0 = plt.figure(0)
    fig0.set_size_inches(8, 6)
    ax0 = plt.axes()
    plt.plot(total_induced_voltage.profile.bin_centers, total_induced_voltage.induced_voltage, style)
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Induced voltage [V]")

    # Save plot
    fign = dirname + '/induced_voltage_' "%d" % figure_index + '.png'
    if show_plots:
        plt.show()
    else:
        plt.savefig(fign)
        plt.clf()
