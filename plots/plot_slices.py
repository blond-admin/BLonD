
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
import matplotlib.pyplot as plt
from plots.plot_settings import fig_folder


def plot_beam_profile(counter, general_params, slices, style = '-', 
                      dirname = 'fig'):
    
    """
    Plot of longitudinal beam profile
    """
 
    # Directory where plots will be stored 
    fig_folder(dirname)
    
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])    
    ax.plot(slices.bins_centers, slices.n_macroparticles, style)
    
    if slices.slicing_coord == 'theta': 
        ax.set_xlabel(r"$\vartheta$ [rad]")
    elif slices.slicing_coord == 'z':
        ax.set_xlabel('z [m]')
    elif slices.slicing_coord == 'tau':
        ax.set_xlabel('Time [s]')
    ax.set_ylabel('Beam profile [arb. units]')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.figtext(0.95, 0.95, '%d turns' %counter, fontsize=16, ha='right', 
                va='center') 
    
    # Save plot
    fign = dirname +'/beam_profile_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()
    #plt.close()


def plot_beam_profile_derivative(counter, general_params, slices, style = '-', 
                                 dirname = 'fig', numbers = [3]):

    """
    Plot of the derivative of the longitudinal beam profile.
    """

    
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
    #plt.close()        
    

