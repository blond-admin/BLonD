import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from trackers.longitudinal_tracker import *


def fig_folder():
    # Directory where plots will be stored
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


def plot_long_phase_space(beam, cavity, nturns, xmin, xmax, ymin, ymax, 
                          unit=None):

    # Directory where plots will be stored
    fig_folder()

    # Conversion from metres to nanoseconds
    if unit == 'ns':
        coeff = 1.e9/c/beam.beta

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
    if unit == None or unit == 'm':
        axScatter.scatter(beam.z, beam.dp, s=1, edgecolor='none')
        axScatter.set_xlabel('z [m]', fontsize=14)
    elif unit == 'ns':
        axScatter.scatter(beam.z*coeff, beam.dp, s=1, edgecolor='none')
        axScatter.set_xlabel('Time [ns]', fontsize=14)
    axScatter.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    axScatter.set_ylabel(r"$\Delta$p/p [1]", fontsize=14)
    axScatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95,0.95,'%d turns' %(nturns+1), fontsize=16, ha='right', 
                va='center') 

    # Separatrix
    x_sep = np.linspace(xmin, xmax, 1000)
    if unit == None or unit == 'm':
        y_sep = cavity.separatrix(x_sep, beam)
    elif unit == 'ns':
        y_sep = cavity.separatrix(x_sep/coeff, beam)
    axScatter.plot(x_sep, y_sep, 'r')
    axScatter.plot(x_sep, -1.*y_sep, 'r')
    
    # Phase and momentum histograms
    xbin = (xmax - xmin)/200.
    xh = np.arange(xmin, xmax + xbin, xbin)
    ybin = (ymax - ymin)/200.
    yh = np.arange(ymin, ymax + ybin, ybin)

    if unit == None or unit == 'm':
        axHistx.hist(beam.z, bins=xh, histtype='step')
    elif unit == 'ns':
        axHistx.hist(beam.z*coeff, bins=xh, histtype='step')
    axHisty.hist(beam.dp, bins=yh, histtype='step', orientation='horizontal')
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


def plot_bunch_length_evol(beam, h5file, nturns, unit=None):

    # Directory where plots will be stored
    fig_folder()

    # Get bunch length data in metres or nanoseconds
    t = range(1, nturns + 1) 
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Beam/sigma_dz"], dtype=np.double)
    if unit == None or unit == 'm':
        bl *= 4. # 4-sigma bunch length
    elif unit == 'ns':
        bl *= 4.e9/c/beam.beta # 4-sigma bunch length

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


def plot_bunch_length_evol_gaussian(beam, h5file, nturns, unit=None):

    # Directory where plots will be stored
    fig_folder()

    # Get bunch length data in metres or nanoseconds
    t = range(1, nturns + 1) 
    storeddata = h5py.File(h5file + '.h5', 'r')
    bl = np.array(storeddata["/Beam/bl_gauss"], dtype=np.double)
    if unit == 'ns':
        bl *= 1.e9/c/beam.beta # 4-sigma bunch length

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
