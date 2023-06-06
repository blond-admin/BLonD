
# Copyright 2016 CERN. This software is distributed under the
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

from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from pylab import cm



def plot_noise_spectrum(frequency, spectrum, sampling = 1, dirname = 'fig', show_plot = False,
                        figno = 0):
    
    """
    Plot of the phase noise spectrum.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.set_xlim([0, 300])    
    ax.plot(frequency[::sampling], spectrum[::sampling])
    ax.set_xlabel("Frequency [Hz]")
    params = {'text.usetex': False, 'mathtext.default': 'sf'}
    plt.rcParams.update(params)
    ax.set_ylabel(r"Noise spectrum [$\frac{rad^2}{Hz}$]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/noise_spectrum_' "%d" %figno +'.png'
        plt.savefig(fign)
    plt.clf()

        
    
def plot_phase_noise(time, dphi, sampling = 1, dirname = 'fig', show_plot = False,
                     figno = 0):
    
    """
    Plot of phase noise as a function of time.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(time[::sampling], dphi[::sampling])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Phase noise [rad]")

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/phase_noise_' "%d" %figno +'.png'
        plt.savefig(fign)
    plt.clf()     


def plot_PL_bunch_phase(RFStation, h5data, output_freq = 1, 
                        dirname = 'fig', show_plot = False):
    
    """
    Plot of bunch phase measured by the Phase Loop as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/PL_bunch_phase"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, dphi,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"PL $\phi_{\mathsf{bunch}}$ [rad]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/PL_bunch_phase.png'
        plt.savefig(fign)
    plt.clf()     


def plot_PL_RF_phase(RFStation, h5data, output_freq = 1, 
                    dirname = 'fig', show_plot = False):
    
    """
    Plot of RF phase; monitored with Phase Loop.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/PL_phiRF"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes()
    ax.plot(t, dphi, '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel(r"RF phase $\phi_{\mathsf{RF}}$ [rad]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/RF_phase.png'
        plt.savefig(fign)
    plt.clf()     


def plot_PL_phase_corr(RFStation, h5data, output_freq = 1, 
                       dirname = 'fig', show_plot = False):
    
    """
    Plot of phase correction applied by the Phase Loop as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/PL_phase_corr"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, dphi,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"PL $\phi$ correction [rad]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/PL_phase_corr.png'
        plt.savefig(fign)
    plt.clf()     


def plot_PL_RF_freq(RFStation, h5data, output_freq = 1, 
                    dirname = 'fig', show_plot = False):
    
    """
    Plot of RF revolution frequency; monitored with Phase Loop.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/PL_omegaRF"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes()
    ax.plot(t, dphi, '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel(r"RF revolution frequency $\omega_{\mathsf{RF}}$ [1/s]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/RF_freq.png'
        plt.savefig(fign)
    plt.clf()     


def plot_PL_freq_corr(RFStation, h5data, output_freq = 1, 
                      dirname = 'fig', show_plot = False):
    
    """
    Plot of frequency correction applied by the Phase Loop as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/PL_omegaRF_corr"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, dphi,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"PL $\omega_{\mathsf{RF}}$ correction [1/s]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/PL_freq_corr.png'
        plt.savefig(fign)
    plt.clf()     


def plot_RF_phase_error(RFStation, h5data, output_freq = 1, 
                       dirname = 'fig', show_plot = False):
    
    """
    Plot of accumulated RF phase error; the Synchro Loop can act on this.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/SL_dphiRF"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes()
    ax.plot(t, dphi, '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel(r"RF phase error $\Delta \phi_{\mathsf{RF}}$ [rad]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/RF_phase_error.png'
        plt.savefig(fign)
    plt.clf()    
    
    
     
def plot_RL_radial_error(RFStation, h5data, output_freq = 1, 
                         dirname = 'fig', show_plot = False):
    
    """
    Plot of relative radial error; monitored with Phase Loop.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    dphi = np.array(h5data["/Beam/RL_drho"][0:ndata], dtype=np.double)
    dphi[time_step:] = np.nan

    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes()
    ax.plot(t, dphi, '.')
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel(r"Relative radial error [1]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/RL_radial_error.png'
        plt.savefig(fign)
    plt.clf()     
    
               

def plot_COM_motion(Ring, RFStation, h5data,  
                    output_freq = 1, dirname = 'fig', show_plot = False):
    """
    Evolution of bunch C.O.M. in longitudinal phase space. 
    Optional use of histograms and separatrix.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)

    # Load data
    mean_dt = np.array(h5data["/Beam/mean_dt"][0:ndata], dtype=np.double)
    mean_dE = np.array(h5data["/Beam/mean_dE"][0:ndata], dtype=np.double)

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,8)
    ax = plt.axes()
    ax.scatter(mean_dt, mean_dE, s=5, edgecolor='none')

    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel(r"$\Delta$E [eV]")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlim((-0.7e-6, 0.7e-6))
    plt.figtext(0.95, 0.95, 'C.O.M. evolution', fontsize=16, ha='right', 
                va='center') 
            
    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/COM_evolution.png'
        plt.savefig(fign)
    plt.clf()


def plot_LHCNoiseFB(RFStation, LHCNoiseFB, h5data, output_freq = 1, 
                    dirname = 'fig', show_plot = False):
    
    """
    Plot of the phase noise multiplication factor as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    x = np.array(h5data["/Beam/LHC_noise_FB_factor"][0:ndata], dtype=np.double)
    x[time_step:] = np.nan

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, x,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"LHC noise FB scaling factor [1]")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/LHC_noise_FB.png'
        plt.savefig(fign)
    plt.clf()         


    
def plot_LHCNoiseFB_FWHM(RFStation, LHCNoiseFB, h5data, 
                         output_freq = 1, dirname = 'fig', show_plot = False):
    
    """
    Plot of the FWHM bunch length used in LHCNoiseFB as a function of time.
    For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    x = np.array(h5data["/Beam/LHC_noise_FB_bl"][0:ndata], dtype=np.double)
    x[time_step:] = np.nan

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(t, x,'.')
    ax.set_xlabel(r"No. turns [T$_0$]")    
    ax.set_ylabel (r"4-sigma FWHM bunch length [s]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/LHC_noise_FB_bl.png'
        plt.savefig(fign)
    plt.clf()         



def plot_LHCNoiseFB_FWHM_bbb(RFStation, LHCNoiseFB, h5data, 
                             output_freq = 1, dirname = 'fig', show_plot = False):
    
    """
    Plot of bunch-by-bunch FWHM bunch length used in LHCNoiseFB as a function 
    of time. For large amount of data, monitor with larger 'output_freq'.
    """

    # Time step of plotting
    time_step = RFStation.counter[0]

    # Load/create data
    if output_freq < 1:
        output_freq = 1
    ndata = int(time_step / output_freq)
    t = output_freq * np.arange(ndata)
    x = np.array(h5data["/Beam/LHC_noise_FB_bl_bbb"][0:ndata, :], ndmin=2)
    x[time_step:, :] = np.nan
    nbunches = x.shape[1]

    # Plot
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    for i in range(nbunches):
        ax.plot(t, x[:, i], '.', color=cm.get_cmap('jet')(i / nbunches), label="Bunch %d" % i)
    ax.set_xlabel(r"No. turns [T$_0$]")
    ax.set_ylabel(r"4-sigma FWHM bunch length [s]")
    if time_step > 100000:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend()

    # Output plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/LHC_noise_FB_bl_bbb.png'
        plt.savefig(fign)
    plt.clf()         
