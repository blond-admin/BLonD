'''
**Module to plot different bunch features **

:Authors: **Helga Timko**, **Danilo Quartullo**

'''

from __future__ import division
import os
import subprocess
import sys
import warnings
import matplotlib.pyplot as plt
from impedances.longitudinal_impedance import *



if os.path.exists('fig'):    
    if "lin" in sys.platform:
        subprocess.Popen("rm -rf fig", shell = True, executable = "/bin/bash")
    elif "win" in sys.platform:
        os.system('del /s/q '+ os.getcwd() +'\\fig>null')
    else:
        warnings.warn("You have not a Windows or Linux operating system. Aborting...")

    
def fig_folder(dirname):
    
    # Try to create directory
    try:
        os.makedirs(dirname)
    # Check whether already exists/creation failed
    except OSError:
        if os.path.exists(dirname):
            pass
        else:
            raise



def plot_impedance_vs_frequency(counter, general_params, ind_volt_from_imp, 
                                option1 = "sum", option2 = "no_spectrum", 
                                option3 = "freq_fft", style = '-', dirname = 'fig'):

    """
    Plot of impedance vs frequency.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    if option1 == "sum":
        
        ax1 = plt.subplots()[1]
        ax1.plot(ind_volt_from_imp.frequency_fft, 
                 ind_volt_from_imp.impedance_array.real, style)
        ax1.plot(ind_volt_from_imp.frequency_fft, 
                 ind_volt_from_imp.impedance_array.imag, style)
        if option2 == "spectrum":
            ax2 = ax1.twinx()
            ax2.plot(ind_volt_from_imp.frequency_fft, 
                     np.abs(ind_volt_from_imp.spectrum))
        fign = dirname +'/sum_imp_vs_freq_fft' "%d" %counter + '.png'
        plt.savefig(fign, dpi=300)
        plt.clf()
    
    elif option1 == "single":
        
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        for i in range(len(ind_volt_from_imp.impedance_sum)):
                if isinstance(ind_volt_from_imp.impedance_sum[i], 
                              Longitudinal_table) and option3 == "freq_table":
                    ax0.plot(ind_volt_from_imp.impedance_sum[i].frequency_array, 
                             ind_volt_from_imp.impedance_sum[i].Re_Z_array, style)
                    ax1.plot(ind_volt_from_imp.impedance_sum[i].frequency_array, 
                             ind_volt_from_imp.impedance_sum[i].Im_Z_array, style) 
                else:
                    ax0.plot(ind_volt_from_imp.frequency_fft, 
                             ind_volt_from_imp.impedance_sum[i].impedance.real, style)
                    ax1.plot(ind_volt_from_imp.frequency_fft, 
                             ind_volt_from_imp.impedance_sum[i].impedance.imag, style)
        
        fign1 = dirname +'/real_imp_vs_'+option3+'_' "%d" %counter + '.png'
        if option2 == "spectrum":
            ax2 = ax0.twinx()
            ax2.plot(ind_volt_from_imp.frequency_fft, 
                     np.abs(ind_volt_from_imp.spectrum))
        plt.figure(0)
        plt.savefig(fign1, dpi=300)
        plt.clf()
        fign2 = dirname +'/imag_imp_vs_'+option3+'_' "%d" %counter + '.png'
        plt.figure(1)
        if option2 == "spectrum":
            ax3 = ax1.twinx()
            ax3.plot(ind_volt_from_imp.frequency_fft, 
                     np.abs(ind_volt_from_imp.spectrum))
        plt.savefig(fign2, dpi=300)
        plt.clf()
        
   
   
def plot_induced_voltage_vs_bins_centers(counter, general_params, 
                                         ind_volt_from_imp, style = '-', 
                                         dirname = 'fig'):

    """
    Plot of induced voltage vs bin centers.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    plt.plot(ind_volt_from_imp.slices.bins_centers, ind_volt_from_imp.ind_vol, style)
             
    # Save plot
    fign = dirname +'/induced_voltage_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()



