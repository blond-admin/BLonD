'''
Created on 12.06.2014

@author: Helga Timko
'''

from __future__ import division
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e
from trackers.longitudinal_utilities import separatrix
import sys
from impedances.longitudinal_impedance import *

if os.path.exists('temp'):
    os.system('del /s/q '+ os.getcwd() +'\\temp>null')
    

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


def plot_long_phase_space(counter, beam, General_parameters, RingAndRFSection, xmin,
                          xmax, ymin, ymax, xunit = None, yunit = None, perc_plotted_points = 100, 
                          separatrix_plot = False, histograms_plot = True, dirname = 'temp'):

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    # Calculate the final index of coordinate array according to perc_plotted_points
    index = int(perc_plotted_points * beam.n_macroparticles / 100) + 1
    
    # Conversion from metres to nanoseconds
    if xunit == 'ns':
        coeff = 1.e9 * General_parameters.ring_radius / (beam.beta_rel * c)
    elif xunit == 'm':
        coeff = - General_parameters.ring_radius
    ycoeff = beam.beta_rel**2 * beam.energy

    # Definitions for placing the axes
    left, width = 0.1, 0.63
    bottom, height = 0.1, 0.63
    bottom_h = left_h = left+width+0.03
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # Prepare plot
    plt.figure(1, figsize=(8,8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # Main plot: longitudinal distribution
    if xunit == None or xunit == 'rad':
        axScatter.set_xlabel('theta [rad]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta[0:index], beam.dE[0:index]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.theta[0:index], beam.delta[0:index], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)           
    elif xunit == 'm':
        axScatter.set_xlabel('z [m]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.z[0:index], beam.dE[0:index]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.z[0:index], beam.delta[0:index], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)              
    elif xunit == 'ns':
        axScatter.set_xlabel('Time [ns]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta[0:index]*coeff, beam.dE[0:index]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.theta[0:index]*coeff, beam.delta[0:index], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)           
        
    axScatter.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    
    if xunit == None or xunit == 'rad':
        axScatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axScatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95,0.95,'%d turns' %counter, fontsize=16, ha='right', 
                va='center') 

    # Separatrix
    if separatrix_plot:
        x_sep = np.linspace(xmin, xmax, 1000)
        if xunit == None or xunit == 'rad':
            y_sep = separatrix(General_parameters, RingAndRFSection, x_sep)
        elif xunit == 'm' or xunit == 'ns':
            y_sep = separatrix(General_parameters, RingAndRFSection, x_sep/coeff)
        if yunit == None or yunit == 'MeV':
            axScatter.plot(x_sep, y_sep/1.e6, 'r')
            axScatter.plot(x_sep, -1.e-6*y_sep, 'r')       
        else:
            axScatter.plot(x_sep, y_sep/ycoeff, 'r')
            axScatter.plot(x_sep, -1.*y_sep/ycoeff, 'r')
    
    # Phase and momentum histograms
    if histograms_plot:
        xbin = (xmax - xmin)/200.
        xh = np.arange(xmin, xmax + xbin, xbin)
        ybin = (ymax - ymin)/200.
        yh = np.arange(ymin, ymax + ybin, ybin)
      
        if xunit == None or xunit == 'rad':
            axHistx.hist(beam.theta[0:index], bins=xh, histtype='step')
        elif xunit == 'm':
            axHistx.hist(beam.z[0:index], bins=xh, histtype='step')       
        elif xunit == 'ns':
            axHistx.hist(beam.theta[0:index]*coeff, bins=xh, histtype='step')
        if yunit == None or yunit == 'MeV':
            axHisty.hist(beam.dE[0:index]/1.e6, bins=yh, histtype='step', orientation='horizontal')
        if yunit == '1':
            axHisty.hist(beam.delta[0:index], bins=yh, histtype='step', orientation='horizontal')
        axHistx.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axHisty.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axHistx.axes.get_xaxis().set_visible(False)
        axHisty.axes.get_yaxis().set_visible(False)
        axHistx.set_xlim(xmin, xmax)
        axHisty.set_ylim(ymin, ymax)
        labels = axHisty.get_xticklabels()
        for label in labels:
            label.set_rotation(-90) 
 
    # Save plot
    fign = dirname +'/long_distr_'"%d"%counter+'.png'
    plt.savefig(fign)
    plt.clf()


def plot_bunch_length_evol(counter, beam, h5file, General_parameters, unit = None, dirname = 'temp'):

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)

    # Get bunch length data in metres or nanoseconds
    t = range(1, General_parameters.n_turns + 1)
    
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/sigma_theta"], dtype = np.double)
    if unit == None or unit == 'ns':
        bl *= 4.e9 / beam.beta_rel / c * General_parameters.ring_radius
    elif unit == 'm':
        bl *= 4 * General_parameters.ring_radius

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(t, bl[0:General_parameters.n_turns], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'ns':
        ax.set_ylabel (r"Bunch length, $4\sigma$ r.m.s. [ns]")
    elif unit == 'm':
        ax.set_ylabel (r"Bunch length, $4\sigma$ r.m.s. [m]")
    
    # Save plot
    fign = dirname +'/bunch_length_evolution_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()


def plot_bunch_length_evol_gaussian(beam, h5file, General_parameters, unit = None, dirname = 'temp'):

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)

    # Get bunch length data in metres or nanoseconds
    t = range(1, General_parameters.n_turns + 1) 
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/bunch_length_gauss_theta"], dtype=np.double)
    if unit == 'ns':
        bl *= 1.e9/c/beam.beta_rel * General_parameters.ring_radius

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(t, bl[0:General_parameters.n_turns], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'theta':
        ax.set_ylabel (r"Bunch length, $4\sigma$ Gaussian fit [rad]")
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $4\sigma$ Gaussian fit [ns]")
    
    # Save plot    
    fign = dirname +'/bunch_length_evolution_Gaussian_' "%d" %General_parameters.n_turns + '.png'
    plt.savefig(fign)
    plt.clf()


def plot_impedance_vs_frequency(counter, general_params, ind_volt_from_imp, option1 = "sum", 
                                option2 = "no_spectrum", option3 = "freq_fft", style = '-', dirname = 'temp'):

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    if option1 == "sum":
        
        ax1 = plt.subplots()[1]
        ax1.plot(ind_volt_from_imp.frequency_fft, ind_volt_from_imp.impedance_array.real, style)
        ax1.plot(ind_volt_from_imp.frequency_fft, ind_volt_from_imp.impedance_array.imag, style)
        if option2 == "spectrum":
            ax2 = ax1.twinx()
            ax2.plot(ind_volt_from_imp.frequency_fft, np.abs(ind_volt_from_imp.spectrum))
        fign = dirname +'/sum_imp_vs_freq_fft' "%d" %counter + '.png'
        plt.savefig(fign, dpi=300)
        plt.clf()
    
    elif option1 == "single":
        
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        for i in range(len(ind_volt_from_imp.impedance_sum)):
                if isinstance(ind_volt_from_imp.impedance_sum[i], Longitudinal_table) and option3 == "freq_table":
                    ax0.plot(ind_volt_from_imp.impedance_sum[i].frequency_array, ind_volt_from_imp.impedance_sum[i].Re_Z_array, style)
                    ax1.plot(ind_volt_from_imp.impedance_sum[i].frequency_array, ind_volt_from_imp.impedance_sum[i].Im_Z_array, style) 
                else:
                    ax0.plot(ind_volt_from_imp.frequency_fft, ind_volt_from_imp.impedance_sum[i].impedance.real, style)
                    ax1.plot(ind_volt_from_imp.frequency_fft, ind_volt_from_imp.impedance_sum[i].impedance.imag, style)
        
        fign1 = dirname +'/real_imp_vs_'+option3+'_' "%d" %counter + '.png'
        if option2 == "spectrum":
            ax2 = ax0.twinx()
            ax2.plot(ind_volt_from_imp.frequency_fft, np.abs(ind_volt_from_imp.spectrum))
        plt.figure(0)
        plt.savefig(fign1, dpi=300)
        plt.clf()
        fign2 = dirname +'/imag_imp_vs_'+option3+'_' "%d" %counter + '.png'
        plt.figure(1)
        if option2 == "spectrum":
            ax3 = ax1.twinx()
            ax3.plot(ind_volt_from_imp.frequency_fft, np.abs(ind_volt_from_imp.spectrum))
        plt.savefig(fign2, dpi=300)
        plt.clf()
        
   
def plot_induced_voltage_vs_bins_centers(counter, general_params, ind_volt_from_imp, style = '-', dirname = 'temp'):

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    plt.plot(ind_volt_from_imp.slices.bins_centers, ind_volt_from_imp.ind_vol, style)
             
    # Save plot
    fign = dirname +'/induced_voltage_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()


def plot_beam_profile(counter, general_params, slices, style = '-', dirname = 'temp'):
    
    fig_folder(dirname)
    plt.plot(slices.bins_centers, slices.n_macroparticles, style)
    fign = dirname +'/beam_profile_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()


def plot_beam_profile_derivative(counter, general_params, slices, style = '-', dirname = 'temp', numbers = [3]):
    
    fig_folder(dirname)
    if 1 in numbers:
        x1, derivative1 = slices.beam_profile_derivative(1)
        plt.plot(x1, derivative1, style)
    if 2 in numbers:
        x2, derivative2 = slices.beam_profile_derivative(2)
        plt.plot(x2, derivative2, style)
    if 3 in numbers:
        x3, derivative3 = slices.beam_profile_derivative(3)
        plt.plot(x3, derivative3, style)
    fign = dirname +'/beam_profile_derivative_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()
         
    
