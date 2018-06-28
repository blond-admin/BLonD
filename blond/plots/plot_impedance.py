
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
from ..impedances.impedance_sources import *



def plot_impedance_vs_frequency(counter, general_params, ind_volt_from_imp, 
                                option1 = "sum", option2 = "no_spectrum", 
                                option3 = "freq_fft", style = '-', 
                                cut_left_right = None, cut_up_down = None, 
                                dirname = 'fig'):

    """
    Plot of impedance vs frequency.
    """

    if option1 == "sum":
        
        ax1 = plt.subplots()[1]
        ax1.plot(ind_volt_from_imp.frequency_array, 
                 ind_volt_from_imp.total_impedance.real, style)
        ax1.plot(ind_volt_from_imp.frequency_array, 
                 ind_volt_from_imp.total_impedance.imag, style)
        if option2 == "spectrum":
            ax2 = ax1.twinx()
            ax2.plot(ind_volt_from_imp.profile.beam_spectrum_freq, 
                     np.abs(ind_volt_from_imp.profile.beam_spectrum))
        fign = dirname +'/sum_imp_vs_freq_fft' "%d" %counter + '.png'
        plt.show()
        plt.savefig(fign)
        plt.clf()
            
    elif option1 == "single":
        
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        
        for i in range(len(ind_volt_from_imp.impedance_source_list)):
            if isinstance(ind_volt_from_imp.impedance_source_list[i], 
                          InputTable) and option3 == "freq_table":
                ax0.plot(ind_volt_from_imp.impedance_source_list[i].frequency_array_loaded, 
                         ind_volt_from_imp.impedance_source_list[i].Re_Z_array_loaded, style)
                ax0.set_xlim(cut_left_right)
                ax0.set_ylim(cut_up_down) 
                ax1.plot(ind_volt_from_imp.impedance_source_list[i].frequency_array_loaded, 
                         ind_volt_from_imp.impedance_source_list[i].Im_Z_array_loaded, style)
                ax1.set_xlim(cut_left_right)
                ax1.set_ylim(cut_up_down) 
            else:
                ax0.plot(ind_volt_from_imp.frequency_array, 
                         ind_volt_from_imp.impedance_source_list[i].impedance.real, style)
                ax0.set_xlim(cut_left_right)
                ax0.set_ylim(cut_up_down) 
                ax1.plot(ind_volt_from_imp.frequency_array, 
                         ind_volt_from_imp.impedance_source_list[i].impedance.imag, style)
                ax1.set_xlim(cut_left_right)
                ax1.set_ylim(cut_up_down) 
        
        fign1 = dirname +'/real_imp_vs_'+option3+'_' "%d" %counter + '.png'
        if option2 == "spectrum":
            ax2 = ax0.twinx()
            ax2.plot(ind_volt_from_imp.profile.beam_spectrum_freq, 
                     np.abs(ind_volt_from_imp.profile.beam_spectrum))
            ax2.set_xlim(cut_left_right)
            ax2.set_ylim(cut_up_down)
        ax0.set_xlabel("Frequency [Hz]")
        ax0.set_ylabel ("Real impedance [Ohm]")
        plt.figure(0)
        
        plt.savefig(fign1)
        plt.clf()
        fign2 = dirname +'/imag_imp_vs_'+option3+'_' "%d" %counter + '.png'
        
        if option2 == "spectrum":            
            ax3 = ax1.twinx()
            ax3.plot(ind_volt_from_imp.profile.beam_spectrum_freq, 
                     np.abs(ind_volt_from_imp.profile.beam_spectrum))
            ax3.set_xlim(cut_left_right)
            ax3.set_ylim(cut_up_down)
        plt.figure(1)
        
        plt.savefig(fign2)
        plt.clf()

   
   
def plot_induced_voltage_vs_bin_centers(counter, general_params, 
                                         total_voltage, style = '-', 
                                         dirname = 'fig'):

    """
    Plot of induced voltage vs bin centers.
    """
   
    fig0 = plt.figure(0)
    fig0.set_size_inches(8,6)
    ax0 = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    plt.plot(total_voltage.profile.bin_centers, total_voltage.induced_voltage, style)
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel ("Induced voltage [V]")

    # Save plot
    fign = dirname +'/induced_voltage_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()
