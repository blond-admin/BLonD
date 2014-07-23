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



def plot_beam_profile(counter, general_params, slices, style = '-', 
                      dirname = 'fig'):
    
    """
    Plot of longitudinal beam profile
    """
 
    fig_folder(dirname)
    plt.plot(slices.bins_centers, slices.n_macroparticles, style)
    fign = dirname +'/beam_profile_' "%d" %counter + '.png'
    plt.savefig(fign)
    plt.clf()



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
         
    

