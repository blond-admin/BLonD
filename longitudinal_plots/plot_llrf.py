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



def plot_noise_spectrum(frequency, spectrum, sampling = 1, dirname = 'fig'):
    
    """
    Plot of the phase noise spectrum.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.set_xlim([0, 300])    
    ax.plot(frequency[::sampling], spectrum[::sampling])
    ax.set_xlabel("Frequency [Hz]", fontsize=20, fontweight='bold')
    params = {'text.usetex': False, 'mathtext.default' : 'sf'}
    plt.rcParams.update(params)
    ax.set_ylabel (r"Noise spectrum [$\frac{rad^2}{Hz}$]", fontsize=20, fontweight='bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_fontweight('bold')
    fign = dirname +'/noise_spectrum.png'
    plt.savefig(fign)
    plt.clf()
    
    
    
def plot_phase_noise(time, dphi, sampling = 1, dirname = 'fig'):
    
    """
    Plot of the phase noise as a function of time.
    For large amount of data, use "sampling" to plot a fraction of the data.
    """

    # Directory where longitudinal_plots will be stored
    fig_folder(dirname)
    
    # Plot
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    ax.plot(time[::sampling], dphi[::sampling])
    ax.set_xlabel("Time [s]", fontsize=20, fontweight='bold')
    
    ax.set_ylabel (r"Phase noise [rad]", fontsize=20, fontweight='bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_fontweight('bold')
    fign = dirname +'/phase_noise.png'
    plt.savefig(fign)
    plt.clf()     
    
    
       
