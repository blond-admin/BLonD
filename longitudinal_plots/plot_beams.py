'''
**Module to plot different bunch features **

:Authors: **Helga Timko**, **Danilo Quartullo**

'''

from __future__ import division
import h5py
import os
import subprocess
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from trackers.longitudinal_utilities import separatrix




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
        coeff = 1.e9*General_parameters.ring_radius/beam.beta_r/c
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
    plt.figure(1, figsize=(8,8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # Main plot: longitudinal distribution
    if xunit == None or xunit == 'rad':
        axScatter.set_xlabel(r"$\vartheta$ [rad]", fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta[::sampling], beam.dE[::sampling]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.theta[::sampling], beam.delta[::sampling], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)           
    elif xunit == 'm':
        axScatter.set_xlabel('z [m]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.z[::sampling], beam.dE[::sampling]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.z[::sampling], beam.delta[::sampling], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)              
    elif xunit == 'ns':
        axScatter.set_xlabel('Time [ns]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta[::sampling]*coeff, beam.dE[::sampling]/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.theta[::sampling]*coeff, beam.delta[::sampling], s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)           
        
    axScatter.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    
    if xunit == None or xunit == 'rad':
        axScatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axScatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95,0.95,'%d turns' %RFSectionParameters.counter[0], fontsize=16, ha='right', 
                va='center') 

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
            axHisty.hist(beam.dE[::sampling]/1.e6, bins=yh, histtype='step', orientation='horizontal')
        if yunit == '1':
            axHisty.hist(beam.delta[::sampling], bins=yh, histtype='step', orientation='horizontal')
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



def plot_bunch_length_evol(beam, h5file, General_parameters, time_step, 
                           unit = None, dirname = 'fig'):
    """
    Plot of r.m.s. 4-sigma bunch length as a function of time.
    Choice of units: unit = rad, ns, m.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)

    # Get bunch length data in metres or nanoseconds
    t = range(time_step)
    
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/sigma_theta"], dtype = np.double)
    if unit == None or unit == 'rad':
        bl *= 4 
    elif unit == 'ns':
        bl *= 4.e9*General_parameters.ring_radius/beam.beta_r/c 
    elif unit == 'm':
        bl *= 4*General_parameters.ring_radius 
    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: unit of plot_bunch_length not recognized!")   

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(t, bl[0:time_step], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'rad':
        ax.set_ylabel (r"Bunch length, $\vartheta_{4\sigma}$ r.m.s. [rad]")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $\tau_{4\sigma}$ r.m.s. [ns]")
    elif unit == 'm':
        ax.set_ylabel (r"Bunch length, z$_{4\sigma}$ r.m.s. [m]")
    
    # Save plot
    fign = dirname +'/bunch_length.png'
    plt.savefig(fign)
    plt.clf()



def plot_bunch_length_evol_gaussian(beam, h5file, General_parameters, slices, 
                                    time_step, unit = None, dirname = 'fig'):

    """
    Plot of Gaussian 4-sigma bunch length as a function of time; requires slices.
    Choice of units: unit = rad, ns, m.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)

    # Get bunch length data in metres or nanoseconds
    t = range(time_step) 
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/bunch_length_gauss_theta"], dtype=np.double)

    if slices.coord == "theta":
        if unit == 'ns':
            bl *= 1.e9*General_parameters.ring_radius/beam.beta_r/c 
        elif unit == 'm':
            bl *= General_parameters.ring_radius 
    elif slices.coord == "tau":
        if unit == None or unit == 'rad':
            bl *= beam.beta_r*c/General_parameters.ring_radius 
        elif unit == 'ns':
            bl *= 1.e9 
        elif unit == 'm':
            bl *= beam.beta_r*c 
    elif slices.coord == "z":    
        if unit == None or unit == 'rad':
            bl /= General_parameters.ring_radius 
        elif unit == 'ns':
            bl *= 1.e9/beam.beta_r/c 
    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: unit of plot_bunch_length_gaussian not recognized!")   

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(t, bl[0:time_step], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'rad':
        ax.set_ylabel (r"Bunch length, $\vartheta_{4\sigma}$ Gaussian fit [rad]")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $\tau_{4\sigma}$ Gaussian fit [ns]")
    elif unit == 'm':
        ax.set_ylabel (r"Bunch length, z$_{4\sigma}$ Gaussian fit [m]")
   
    # Save plot    
    fign = dirname +'/bunch_length_Gaussian.png'
    plt.savefig(fign)
    plt.clf()



def plot_position_evol(counter, beam, h5file, General_parameters, unit = None, style = '-', dirname = 'fig'): 
 
    # Directory where longitudinal_plots will be stored 
    fig_folder(dirname) 
 
    # Get position data in metres or nanoseconds 
    t = range(1, General_parameters.n_turns + 1) 
     
    storeddata = h5py.File(h5file + '.h5', 'r') 
    pos = np.array(storeddata["/Bunch/mean_theta"], dtype = np.double) 
    if unit == None or unit == 'ns': 
        pos *= 1.e9 / beam.beta_r / c * General_parameters.ring_radius 
    elif unit == 'm': 
        pos *= General_parameters.ring_radius 
         
    pos[counter:] = np.nan 
 
    # Plot 
    plt.figure(1, figsize=(8,6)) 
    ax = plt.axes([0.12, 0.1, 0.82, 0.8]) 
    ax.plot(t[0:counter], pos[0:counter], style) 
    ax.set_xlabel(r"No. turns [T$_0$]") 
    ax.set_xlim((1,General_parameters.n_turns + 1)) 
    if unit == None or unit == 'ns': 
        ax.set_ylabel (r"Position [ns]") 
    elif unit == 'm': 
        ax.set_ylabel (r"Position [m]") 
     
    # Save plot 
    fign = 'fig/position_evolution_' "%d" %counter + '.png' 
    plt.savefig(fign) 
    plt.clf() 



