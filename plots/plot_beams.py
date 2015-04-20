
# Copyright 2014 CERN. This software is distributed under the
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
import h5py
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.constants import c
from trackers.utilities import separatrix
from plots.plot_settings import fig_folder


def plot_long_phase_space(beam, General_parameters, RFSectionParameters, xmin,
                          xmax, ymin, ymax, xunit = None, yunit = None, 
                          sampling = 1, separatrix_plot = False, 
                          histograms_plot = True, dirname = 'fig'):
    """
    Plot of longitudinal phase space. Optional use of histograms and separatrix.
    Choice of units: xunit = rad, ns, m; yunit = MeV, 1.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    # Conversion from metres to nanoseconds
    if xunit == 'ns':
        coeff = 1.e9 * General_parameters.ring_radius / (beam.beta_r*c)
    elif xunit == 'm':
        coeff = - General_parameters.ring_radius
    ycoeff = beam.beta_r**2 * beam.energy

    # Definitions for placing the axes
    left, width = 0.1, 0.63
    bottom, height = 0.1, 0.63
    bottom_h = left_h = left+width+0.03
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # Prepare plot
    fig = plt.figure(1)
    fig.set_size_inches(8,8)
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # Main plot: longitudinal distribution
    if xunit == None or xunit == 'rad':
        axScatter.set_xlabel(r"$\vartheta$ [rad]")
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta[::sampling], beam.dE[::sampling]/1.e6, 
                              s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]")
        elif yunit == '1': 
            axScatter.scatter(beam.theta[::sampling], beam.delta[::sampling], 
                              s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]")           
    elif xunit == 'm':
        axScatter.set_xlabel('z [m]')
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.z[::sampling], beam.dE[::sampling]/1.e6, 
                              s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]")
        elif yunit == '1': 
            axScatter.scatter(beam.z[::sampling], beam.delta[::sampling], 
                              s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]")              
    elif xunit == 'ns':
        axScatter.set_xlabel('Time [ns]')
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta[::sampling]*coeff, 
                              beam.dE[::sampling]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]")
        elif yunit == '1': 
            axScatter.scatter(beam.theta[::sampling]*coeff, 
                              beam.delta[::sampling], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]")      
    axScatter.yaxis.labelpad = 1     
        
    axScatter.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    
    if xunit == None or xunit == 'rad':
        axScatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axScatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95,0.95,'%d turns' %RFSectionParameters.counter[0], 
                fontsize=16, ha='right', va='center') 
            
    # Separatrix
    if separatrix_plot:
        x_sep = np.linspace(xmin, xmax, 1000)
        if xunit == None or xunit == 'rad':
            y_sep = separatrix(General_parameters, RFSectionParameters, x_sep)
        elif xunit == 'm' or xunit == 'ns':
            y_sep = separatrix(General_parameters, RFSectionParameters, x_sep/coeff)
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
            axHistx.hist(beam.theta[::sampling], bins=xh, histtype='step')
        elif xunit == 'm':
            axHistx.hist(beam.z[::sampling], bins=xh, histtype='step')       
        elif xunit == 'ns':
            axHistx.hist(beam.theta[::sampling]*coeff, bins=xh, histtype='step')
        if yunit == None or yunit == 'MeV':
            axHisty.hist(beam.dE[::sampling]/1.e6, bins=yh, histtype='step', 
                         orientation='horizontal')
        if yunit == '1':
            axHisty.hist(beam.delta[::sampling], bins=yh, histtype='step', 
                         orientation='horizontal')
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
    fign = dirname +'/long_distr_'"%d"%RFSectionParameters.counter[0]+'.png'
    plt.savefig(fign)
    plt.clf()
    #plt.close()


def plot_bunch_length_evol(h5file, General_parameters, time_step, 
                           output_freq = 1, unit = None, dirname = 'fig'):
    """
    Plot of r.m.s. 4-sigma bunch length as a function of time.
    Choice of units: unit = rad, ns, m.
    """

    # Directory where plots will be stored
    fig_folder(dirname)

    # Get bunch length data in metres or nanoseconds
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*np.arange(ndata)    
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/sigma_theta"], dtype = np.double)
    
    if unit == None or unit == 'rad':
        bl *= 4 
    elif unit == 'ns':
        bl *= 4.e9 * General_parameters.ring_radius / (General_parameters.beta_r[0]*c) 
    elif unit == 'm':
        bl *= 4*General_parameters.ring_radius 
    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: unit of plot_bunch_length not recognized!")   

    # Plot
    
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, bl[0:ndata], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'rad':
        ax.set_ylabel (r"Bunch length, $\vartheta_{4\sigma}$ r.m.s. [rad]")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $\tau_{4\sigma}$ r.m.s. [ns]")
    elif unit == 'm':
        ax.set_ylabel (r"Bunch length, z$_{4\sigma}$ r.m.s. [m]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Save plot
    fign = dirname +'/bunch_length.png'
    plt.savefig(fign)
    plt.clf()
    #plt.close()
    

def plot_bunch_length_evol_gaussian(h5file, General_parameters, slices, 
                                    time_step, output_freq = 1, unit = None,
                                    dirname = 'fig'):

    """
    Plot of Gaussian 4-sigma bunch length as a function of time; requires slices.
    Choice of units: unit = rad, ns, m.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)

    # Get bunch length data in metres or nanoseconds
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq)
    t = output_freq*range(0, ndata)    
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/bunch_length_gauss_theta"], dtype=np.double)

    if slices.slicing_coord == "theta":
        if unit == 'ns':
            bl *= 1.e9 * General_parameters.ring_radius / (General_parameters.beta_r[0]*c) 
        elif unit == 'm':
            bl *= General_parameters.ring_radius 
    elif slices.slicing_coord == "tau":
        if unit == None or unit == 'rad':
            bl *= (General_parameters.beta_r[0]*c) / General_parameters.ring_radius 
        elif unit == 'ns':
            bl *= 1.e9 
        elif unit == 'm':
            bl *= General_parameters.beta_r[0]*c 
    elif slices.slicing_coord == "z":    
        if unit == None or unit == 'rad':
            bl /= General_parameters.ring_radius 
        elif unit == 'ns':
            bl *= 1.e9 / (General_parameters.beta_r[0]*c) 
    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: unit of plot_bunch_length_gaussian not recognized!")   

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, bl[0:ndata], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'rad':
        ax.set_ylabel (r"Bunch length, $\vartheta_{4\sigma}$ Gaussian fit [rad]")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $\tau_{4\sigma}$ Gaussian fit [ns]")
    elif unit == 'm':
        ax.set_ylabel (r"Bunch length, z$_{4\sigma}$ Gaussian fit [m]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
   
    # Save plot    
    fign = dirname +'/bunch_length_Gaussian.png'
    plt.savefig(fign)
    plt.clf()
    #plt.close()
    

def plot_position_evol(h5file, General_parameters, time_step,
                       output_freq = 1, unit = None, style = '-', 
                       dirname = 'fig'): 
 
    # Directory where longitudinal_plots will be stored 
    fig_folder(dirname) 
 
    # Get position data in metres or nanoseconds 
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*range(1, ndata + 1)    
    storeddata = h5py.File(h5file + '.h5', 'r') 
    pos = np.array(storeddata["/Bunch/mean_theta"], dtype = np.double) 
    
    if unit == 'ns': 
        pos *= 1.e9 / (General_parameters.beta_r[0]*c) * General_parameters.ring_radius 
    elif unit == 'm': 
        pos *= General_parameters.ring_radius 
         
    pos[time_step:] = np.nan 
 
    # Plot 
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot(t, pos[0:ndata], style) 
    ax.set_xlabel(r"No. turns [T$_0$]") 
    ax.set_xlim((1,General_parameters.n_turns + 1)) 
    if unit == None or unit == 'rad': 
        ax.set_ylabel (r"Bunch mean position, $\vartheta$ [rad]") 
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    elif unit == 'ns': 
        ax.set_ylabel (r"Bunch mean position, $\tau$ [ns]") 
    elif unit == 'm': 
        ax.set_ylabel ("Bunch mean position, z [m]") 
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
     
    # Save plot 
    fign = dirname+'/bunch_mean_position.png'
    plt.savefig(fign) 
    plt.clf() 
    #plt.close()

def plot_energy_evol(h5file, General_parameters, time_step,
                     output_freq = 1, unit = None, style = '-', 
                     dirname = 'fig'): 
 
    # Directory where longitudinal_plots will be stored 
    fig_folder(dirname) 
 
    # Get position data in metres or nanoseconds 
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*range(1, ndata + 1)    
    storeddata = h5py.File(h5file + '.h5', 'r') 
    nrg = np.array(storeddata["/Bunch/mean_dE"], dtype = np.double)/1.e6 
            
    nrg[time_step:] = np.nan 
 
    # Plot 
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8]) 
    ax.plot(t, nrg[0:ndata], style) 
    ax.set_xlabel(r"No. turns [T$_0$]") 
    ax.set_xlim((1,General_parameters.n_turns + 1)) 
    ax.set_ylabel (r"Bunch mean energy, $\Delta E$ [MeV]") 
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
     
    # Save plot 
    fign = dirname+'/bunch_mean_energy.png'
    plt.savefig(fign) 
    plt.clf() 
    #plt.close()

