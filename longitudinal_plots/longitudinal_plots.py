'''
Created on 12.06.2014

@author: Helga Timko
'''

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e


def fig_folder():
    # Directory where longitudinal_plots will be stored
    dirname = 'fig'

    # Try to create directory
    try:
        os.makedirs(dirname)
    # Check whether already exists/creation failed
    except OSError:
        if os.path.exists(dirname):
            pass
        else:
            raise


def plot_long_phase_space(ring, beam, nturns, xmin, xmax, ymin, ymax, 
                          xunit=None, yunit=None):

    # Directory where longitudinal_plots will be stored
    fig_folder()

    # Conversion from metres to nanoseconds
    if xunit == 'ns':
        coeff = 1.e9 * beam.radius / (ring.beta_f(beam) * c)
    elif xunit == 'm':
        coeff = - beam.radius * beam.harmonic[0]
#     if yunit == None or yunit == 'MeV':
#         ymin *= 1.e6
#         ymax *= 1.e6

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
            axScatter.scatter(beam.theta, beam.dE/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.theta, beam.delta, s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)           
    elif xunit == 'm':
        axScatter.set_xlabel('z [m]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.z, beam.dE/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.z, beam.delta, s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)              
    elif xunit == 'ns':
        axScatter.set_xlabel('Time [ns]', fontsize=14)
        if yunit == None or yunit == 'MeV':
            axScatter.scatter(beam.theta*coeff, beam.dE/1.e6, s=1, edgecolor='none')
            axScatter.set_ylabel(r"$\Delta$E [MeV]", fontsize=14)
        elif yunit == '1': 
            axScatter.scatter(beam.theta*coeff, beam.delta, s=1, edgecolor='none') 
            axScatter.set_ylabel(r"$\Delta$p/p$_0$ [1]", fontsize=14)           
        
    axScatter.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    
    if xunit == None or xunit == 'rad':
        axScatter.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axScatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95,0.95,'%d turns' %(nturns+1), fontsize=16, ha='right', 
                va='center') 

    # Separatrix
    x_sep = np.linspace(xmin, xmax, 1000)
    if xunit == None or xunit == 'rad':
        y_sep = ring.separatrix(beam, x_sep)
    elif xunit == 'm' or xunit == 'ns':
        y_sep = ring.separatrix(beam, x_sep/coeff)
    if yunit == None or yunit == 'MeV':
        axScatter.plot(x_sep, y_sep/1.e6, 'r')
        axScatter.plot(x_sep, -1.e-6*y_sep, 'r')       
    else:
        axScatter.plot(x_sep, y_sep, 'r')
        axScatter.plot(x_sep, -1.*y_sep, 'r')
    
    # Phase and momentum histograms
    xbin = (xmax - xmin)/200.
    xh = np.arange(xmin, xmax + xbin, xbin)
    ybin = (ymax - ymin)/200.
    yh = np.arange(ymin, ymax + ybin, ybin)

    if xunit == None or xunit == 'rad':
        axHistx.hist(beam.theta, bins=xh, histtype='step')
    elif xunit == 'm':
        axHistx.hist(beam.z, bins=xh, histtype='step')       
    elif xunit == 'ns':
        axHistx.hist(beam.theta*coeff, bins=xh, histtype='step')
    if yunit == None or yunit == 'MeV':
        axHisty.hist(beam.dE/1.e6, bins=yh, histtype='step', orientation='horizontal')
    if yunit == '1':
        axHisty.hist(beam.delta, bins=yh, histtype='step', orientation='horizontal')
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
    fign = 'fig/long_distr_'"%d"%nturns+'.png'
    plt.savefig(fign)
    plt.clf()


def plot_bunch_length_evol(bunch, h5file, nturns, unit=None):

    # Directory where longitudinal_plots will be stored
    fig_folder()

    # Get bunch length data in metres or nanoseconds
    t = range(1, nturns + 1) 
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/sigma_dz"], dtype=np.double)
    if unit == None or unit == 'm':
        bl *= 4. # 4-sigma bunch length
    elif unit == 'ns':
        bl *= 4.e9/c/bunch.beta # 4-sigma bunch length

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(t, bl[0:nturns], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'm':
        ax.set_ylabel (r"Bunch length, $4\sigma$ r.m.s. [m]")
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $4\sigma$ r.m.s. [ns]")
    fign = "fig/bunch_length_evolution.png"
    plt.savefig(fign)
    plt.clf()


def plot_bunch_length_evol_gaussian(bunch, h5file, nturns, unit=None):

    # Directory where longitudinal_plots will be stored
    fig_folder()

    # Get bunch length data in metres or nanoseconds
    t = range(1, nturns + 1) 
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Bunch/bl_gauss"], dtype=np.double)
    if unit == 'ns':
        bl *= 1.e9/c/bunch.beta # 4-sigma bunch length

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(t, bl[0:nturns], '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    if unit == None or unit == 'm':
        ax.set_ylabel (r"Bunch length, $4\sigma$ Gaussian fit [m]")
    elif unit == 'ns':
        ax.set_ylabel (r"Bunch length, $4\sigma$ Gaussian fit [ns]")
    fign = "fig/bunch_length_evolution_Gaussian.png"
    plt.savefig(fign)
    plt.clf()
