
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
import matplotlib.pyplot as plt
import numpy as np



def plot_beam_profile(Profile, counter, style = '-', dirname = 'fig', show_plot = False):
    
    """
    Plot of longitudinal beam profile
    """
 
    fig = plt.figure(1)
    fig.set_size_inches(8,6)
    ax = plt.axes()
    ax.plot(Profile.bin_centers, Profile.n_macroparticles, style)
    
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel('Beam profile [arb. units]')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.figtext(0.95, 0.95, '%d turns' %counter, fontsize=16, ha='right', 
                va='center') 
    
    # Save plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/beam_profile_' "%d" %counter + '.png'
        plt.savefig(fign)
    plt.clf()



def plot_beam_profile_derivative(Profile, counter, style='-', dirname='fig', show_plot = False,
                                 modes=['diff']):
    """
    Plot of the derivative of the longitudinal beam profile.
    Modes list should contain 1 or more of the elements below:
    1) 'filter1d', 2) 'gradient', 3) 'diff'
    """
    for mode in modes:
        x, derivative = Profile.beam_profile_derivative(mode)
        plt.plot(x, derivative, style, label=mode)
    plt.legend()
    if show_plot:
        plt.show()
    else:
        fign = dirname + '/beam_profile_derivative_' "%d" % counter + '.png'
        plt.savefig(fign)
    plt.clf()
    

def plot_beam_spectrum(Profile, counter, style = '-', dirname = 'fig', show_plot = False):
    
    """
    Plot of longitudinal beam profile
    """
 
    plt.figure(1, figsize=(8,6))
    ax = plt.axes()
    ax.plot(Profile.beam_spectrum_freq, np.absolute(Profile.beam_spectrum), style)
    
    ax.set_xlabel(r"Frequency [Hz]")
    ax.set_ylabel('Beam spectrum, \n absolute value [arb. units]')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlim(0,5.e9)
    
    plt.figtext(0.95, 0.95, '%d turns' %counter, fontsize=16, ha='right', 
                va='center') 
    
    # Save plot
    if show_plot:
        plt.show()
    else:
        fign = dirname +'/beam_spectrum_' "%d" %counter + '.png'
        plt.savefig(fign)
    plt.clf()
