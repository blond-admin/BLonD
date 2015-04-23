
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to plot different LLRF features**

:Authors: **Helga Timko**, **Danilo Quartullo**
'''

from __future__ import division
import matplotlib.pyplot as plt
import h5py
import numpy as np
from trackers.utilities import separatrix



def plot_noise_spectrum(frequency, spectrum, sampling = 1, dirname = 'fig', 
                        figno = 0):
    
    """
    Plot of the phase noise spectrum.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.set_xlim([0, 300])    
    ax.plot(frequency[::sampling], spectrum[::sampling])
    ax.set_xlabel("Frequency [Hz]")
    params = {'text.usetex': False, 'mathtext.default' : 'sf'}
    plt.rcParams.update(params)
    ax.set_ylabel(r"Noise spectrum [$\frac{rad^2}{Hz}$]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Save figure
    fign = dirname +'/noise_spectrum_' "%d" %figno +'.png'
    plt.savefig(fign)
    plt.clf()

        
    
def plot_phase_noise(time, dphi, sampling = 1, dirname = 'fig', figno = 0):
    
    """
    Plot of phase noise as a function of time.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(time[::sampling], dphi[::sampling])
    ax.set_xlabel("Time [s]")    
    ax.set_ylabel (r"Phase noise [rad]")

    # Save figure
    fign = dirname +'/phase_noise_' "%d" %figno +'.png'
    plt.savefig(fign)
    plt.clf()     

        

def plot_PL_bunch_phase(RFSectionParameters, PhaseLoop, h5data, output_freq = 1, 
                        dirname = 'fig'):
    
    """
    Plot of bunch phase measured by the Phase Loop as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFSectionParameters.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*np.arange(1, ndata)    
    dphi = np.array(h5data["/Beam/PL_bunch_phase"][1:ndata], dtype = np.double)
    dphi[time_step:] = np.nan
    
    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, dphi,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"PL $\phi_{\mathsf{bunch}}$ [rad]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Save figure
    fign = dirname +'/PL_bunch_phase.png'
    plt.savefig(fign)
    plt.clf()     

               

def plot_PL_phase_corr(RFSectionParameters, PhaseLoop, h5data, output_freq = 1, 
                       dirname = 'fig'):
    
    """
    Plot of phase correction applied by the Phase Loop as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFSectionParameters.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*np.arange(ndata)   
    dphi = np.array(h5data["/Beam/PL_phase_corr"][0:ndata], dtype = np.double)
    dphi[time_step:] = np.nan
    
    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, dphi,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"PL $\phi$ correction [rad]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Save figure
    fign = dirname +'/PL_phase_corr.png'
    plt.savefig(fign)
    plt.clf()     

               

def plot_PL_freq_corr(RFSectionParameters, PhaseLoop, h5data, output_freq = 1, 
                      dirname = 'fig'):
    
    """
    Plot of frequency correction applied by the Phase Loop as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFSectionParameters.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*np.arange(ndata)    
    dphi = np.array(h5data["/Beam/PL_omegaRF_corr"][0:ndata], dtype = np.double)
    dphi[time_step:] = np.nan
    
    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, dphi,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"PL $\omega_{RF}$ correction [1/s]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Save figure
    fign = dirname +'/PL_freq_corr.png'
    plt.savefig(fign)
    plt.clf()     

        

def plot_COM_motion(GeneralParameters, RFSectionParameters, h5data, xmin,
                    xmax, ymin, ymax, separatrix_plot = False, dirname = 'fig'):
    """
    Evolution of bunch C.O.M. in longitudinal phase space. 
    Optional use of histograms and separatrix.
    """

    # Time step of plotting
    ndata = RFSectionParameters.counter[0] + 1

    # Load data
    mean_dt = np.array(h5data["/Beam/mean_dt"][0:ndata+1], dtype = np.double)
    mean_dE = np.array(h5data["/Beam/mean_dE"][0:ndata+1], dtype = np.double)   

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,8)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.scatter(mean_dt, mean_dE, s=5, edgecolor='none')
       
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"$\Delta$E [eV]")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.figtext(0.95, 0.95, 'C.O.M. evolution', fontsize=16, ha='right', 
                va='center') 
            
    # Separatrix
    if separatrix_plot:
        x_sep = np.linspace(xmin, xmax, 1000)
        y_sep = separatrix(GeneralParameters, RFSectionParameters, x_sep)
        ax.plot(x_sep, y_sep, 'r')
        ax.plot(x_sep, - y_sep, 'r')       
                        
    # Save plot
    fign = dirname +'/COM_evolution.png'
    plt.savefig(fign)
    plt.clf()

    

def plot_LHCNoiseFB(RFSectionParameters, LHCNoiseFB, h5data, output_freq = 1, 
                    dirname = 'fig'):
    
    """
    Plot of the phase noise multiplication factor as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFSectionParameters.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*np.arange(ndata)    
    x = np.array(h5data["/Beam/LHC_noise_FB_factor"][0:ndata], dtype = np.double)
    x[time_step:] = np.nan
    
    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, x,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"LHC noise FB scaling factor [1]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Save figure
    fign = dirname +'/LHC_noise_FB.png'
    plt.savefig(fign)
    plt.clf()         


    
def plot_LHCNoiseFB_FWHM(RFSectionParameters, LHCNoiseFB, h5data, 
                         output_freq = 1, dirname = 'fig'):
    
    """
    Plot of the FWHM bunch length used in Noise FB as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFSectionParameters.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step/output_freq) + 1
    t = output_freq*np.arange(ndata)    
    x = np.array(h5data["/Beam/LHC_noise_FB_bl"][0:ndata], dtype = np.double)
    x[time_step:] = np.nan
    
    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(t, x,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"4-sigma FWHM bunch length [s]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Save figure
    fign = dirname +'/LHC_noise_FB_bl.png'
    plt.savefig(fign)
    plt.clf()         


    
