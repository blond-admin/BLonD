
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
import numpy as np
import matplotlib.pyplot as plt
from ..trackers.utilities import separatrix
from matplotlib.layout_engine import PlaceHolderLayoutEngine



def plot_long_phase_space(Ring, RFStation, Beam, xmin,
                          xmax, ymin, ymax, xunit = 's', sampling = 1, 
                          separatrix_plot = False, histograms_plot = True, 
                          dirname = 'fig', show_plot = False, alpha = 1, color = 'b'):
    """
    Plot of longitudinal phase space. Optional use of histograms and separatrix.
    Choice of units: xunit = s, rad.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Conversion from particle arrival time to RF phase
    if xunit == 'rad':
        omega_RF = RFStation.omega_rf[0,RFStation.counter[0]]
        phi_RF = RFStation.phi_rf[0,RFStation.counter[0]]

    # Definitions for placing the axes
    left, width = 0.115, 0.63
    bottom, height = 0.115, 0.63
    bottom_h = left_h = left+width+0.03
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # Prepare plot
    fig = plt.figure(1)
    fig.set_layout_engine(PlaceHolderLayoutEngine(False,False))
    fig.set_size_inches(8,8)
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # Main plot: longitudinal distribution
    indlost = np.where( Beam.id[::sampling] == 0 )[0] # particles lost
    indalive = np.where( Beam.id[::sampling] != 0 )[0] # particles transmitted
    if xunit == 's':
        axScatter.set_xlabel(r"$\Delta t$ [s]")
        axScatter.scatter(Beam.dt[indalive], Beam.dE[indalive], 
                          s=1, edgecolor='none', alpha=alpha, color=color)
        if len(indlost) > 0:
            axScatter.scatter(Beam.dt[indlost], Beam.dE[indlost], color='0.5',
                              s=1, edgecolor='none')
    elif xunit == 'rad':
        axScatter.set_xlabel(r"$\varphi$ [rad]")
        axScatter.scatter(omega_RF*Beam.dt[indalive] + phi_RF, 
                          Beam.dE[indalive], s=1, edgecolor='none', 
                          alpha=alpha, color=color)
        if len(indlost) > 0:
            axScatter.scatter(omega_RF*Beam.dt[indlost] + phi_RF, 
                              Beam.dE[indlost], color='0.5', s=1, 
                              edgecolor='none')
    axScatter.set_ylabel(r"$\Delta$E [eV]")
    axScatter.yaxis.labelpad = 1     
        
    axScatter.set_xlim(xmin, xmax)
    axScatter.set_ylim(ymin, ymax)
    
    axScatter.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95,0.95,'%d turns' %RFStation.counter[0], 
                fontsize=16, ha='right', va='center') 
            
    # Separatrix
    if separatrix_plot:
        x_sep = np.linspace(float(xmin), float(xmax), 1000)
        if xunit == 's':
            y_sep = separatrix(Ring, RFStation, x_sep)
        elif xunit == 'rad':
            y_sep = separatrix(Ring, RFStation, (x_sep - phi_RF)/omega_RF)
        axScatter.plot(x_sep, y_sep, 'r')
        axScatter.plot(x_sep, - y_sep, 'r')       
   
    # Phase and momentum histograms
    if histograms_plot:
        xbin = (xmax - xmin)/200.
        xh = np.arange(xmin, xmax + xbin, xbin)
        ybin = (ymax - ymin)/200.
        yh = np.arange(ymin, ymax + ybin, ybin)
      
        if xunit == 's':
            axHistx.hist(Beam.dt[::sampling], bins=xh, histtype='step', color=color)
        elif xunit == 'rad':
            axHistx.hist(omega_RF*Beam.dt[::sampling] + phi_RF, bins=xh, histtype='step', color=color)       
        axHisty.hist(Beam.dE[::sampling], bins=yh, histtype='step', 
                     orientation='horizontal', color=color)
        axHistx.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axHisty.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axHistx.axes.get_xaxis().set_visible(False)
        axHisty.axes.get_yaxis().set_visible(False)
        axHistx.set_xlim(xmin, xmax)
        axHisty.set_ylim(ymin, ymax)
        labels = axHisty.get_xticklabels()
        for label in labels:
            label.set_rotation(-90)
                     
    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/long_distr_'"%d"%RFStation.counter[0]+'.png'
        plt.savefig(fign)
    plt.clf()



def plot_bunch_length_evol(RFStation, h5data, output_freq = 1, 
                           dirname = 'fig', show_plot = False):
    """
    Plot of r.m.s. 4-sigma bunch length [s] as a function of time.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]
    

    # Get bunch length data in metres or seconds
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq)
    t = output_freq*np.arange(ndata)  
    bl = np.array(h5data["/Beam/sigma_dt"][0:ndata], dtype = np.double)
    bl *= 4
    bl[time_step:] = np.nan
    
    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, bl, '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel (r"Bunch length, $\Delta t_{4\sigma}$ r.m.s. [s]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/bunch_length.png'
        plt.savefig(fign)
    plt.clf()

    

def plot_bunch_length_evol_gaussian(RFStation, Profile, h5data, 
                                    output_freq = 1, dirname = 'fig', show_plot = False):

    """
    Plot of Gaussian 4-sigma bunch length [s] as a function of time.
    Requires profile.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Get bunch length data in metres or nanoseconds
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq)
    t = output_freq*np.arange(ndata)
    
    bl = np.array(h5data["/Beam/bunch_length"][0:ndata], dtype=np.double)
    
    bl[time_step:] = np.nan

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, bl, '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel (r"Bunch length, $\Delta t_{4\sigma}$ Gaussian fit [s]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
   
    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/bunch_length_Gaussian.png'
        plt.savefig(fign)
    plt.clf()

    

def plot_position_evol(RFStation, h5data, output_freq = 1, 
                       style = '.', dirname = 'fig', show_plot = False):
 
    # Time step of plotting
    time_step = RFStation.counter[0]
    
    # Get position data [s] 
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq)
    t = output_freq*np.arange(ndata)    
    pos = np.array(h5data["/Beam/mean_dt"][0:ndata], dtype = np.double)          
    pos[time_step:] = np.nan 
 
    # Plot 
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, pos, style) 
    ax.set_xlabel(r"No. turns [T$_0$]") 
    ax.set_ylabel (r"Bunch mean position, $<\Delta t>$ [s]") 
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
     
    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname+'/bunch_mean_position.png'
        plt.savefig(fign)
    plt.clf() 



def plot_energy_evol(RFStation, h5data, output_freq = 1, style = '.', 
                     dirname = 'fig', show_plot = False):
 
    # Time step of plotting
    time_step = RFStation.counter[0]

    # Get position data in metres or nanoseconds 
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq)
    t = output_freq*np.arange(ndata)  
    nrg = np.array(h5data["/Beam/mean_dE"][0:ndata], dtype = np.double)          
    nrg[time_step:] = np.nan 
 
    # Plot 
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, nrg, style) 
    ax.set_xlabel(r"No. turns [T$_0$]") 
    ax.set_ylabel (r"Bunch mean energy, $<\Delta E>$ [eV]") 
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
     
    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname+'/bunch_mean_energy.png'
        plt.savefig(fign)
    plt.clf() 


def plot_transmitted_particles(RFStation, h5data, output_freq = 1, 
                               style = '.', dirname = 'fig', show_plot = False):
 
    # Time step of plotting
    time_step = RFStation.counter[0]

    # Get position data in metres or nanoseconds 
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq)
    t = output_freq*np.arange(ndata)
    prtcls = np.array(h5data["/Beam/n_macroparticles_alive"][0:ndata], 
                      dtype = np.double)          
    prtcls[time_step:] = np.nan 
 
    # Plot 
    plt.figure(1, figsize=(8,6)) 
    ax = plt.axes()
    ax.plot(t, prtcls, style) 
    ax.set_xlabel(r"No. turns [T$_0$]") 
    ax.set_ylabel (r"Transmitted macro-particles [1]") 
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname+'/bunch_transmitted_particles.png'
        plt.savefig(fign)
    plt.clf() 
