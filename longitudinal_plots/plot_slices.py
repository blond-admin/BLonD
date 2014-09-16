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
 
    # Directory where plots will be stored 
    fig_folder(dirname)
    
    plt.figure(1, figsize=(8,6))
    ax = plt.axes([0.12, 0.1, 0.82, 0.8])
    
    ax.plot(slices.bins_centers, slices.n_macroparticles, style, linewidth=4)
    if slices.slicing_coord == 'theta': 
        ax.set_xlabel(r"$\vartheta$ [rad]", fontsize=20, fontweight='bold')
    elif slices.slicing_coord == 'z':
        ax.set_xlabel('z [m]', fontsize=20, fontweight='bold')
    elif slices.slicing_coord == 'tau':
        ax.set_xlabel('Time [s]', fontsize=20, fontweight='bold')
    ax.set_ylabel('Beam profile [arb. units]', fontsize=20, fontweight='bold')
#     ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), fontweight='bold')
#     ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), fontweight='bold')
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label1.set_fontweight('bold')

    plt.figtext(0.95, 0.95, '%d turns' %counter, fontsize=16, ha='right', 
                va='center') 
    
    # Save plot
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
         
    

