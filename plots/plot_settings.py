
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Class to customize plot layout**

:Authors: **Helga Timko**

'''

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


class PlotSettings(object):
    
    def __init__(self, linewidth=2, markersize=6, labelsize=18, 
                 fontfamily='sans-serif', fontweight='normal', dpi=100):
        '''
        Initialize custom plot formatting. For more options, see
        
        http://matplotlib.org/1.3.1/users/customizing.html
        '''

        self.lwidth = linewidth
        self.msize = markersize
        self.lsize = labelsize
        self.ffamily = fontfamily
        self.fweight = fontweight
        self.dpi = dpi
        
        # Ticksize
        self.tsize = self.lsize - 2
        
        
    def set_plot_format(self):
        
        # Set size of x- and y-grid numbers
        plt.rc('xtick', labelsize=self.tsize) 
        plt.rc('ytick', labelsize=self.tsize)
        
        # Set x- and y-grid labelsize and weight
        plt.rc('axes', labelsize=self.lsize)
        plt.rc('axes', labelweight=self.fweight)

        # Set linewidth for continuous, markersize for discrete plotting
        plt.rc('lines', linewidth=self.lwidth, markersize=self.msize)
        
        # Set figure resolution, font
        plt.rc('figure', dpi=self.dpi)  
        plt.rc('savefig', dpi=self.dpi)  
        plt.rc('font', family=self.ffamily)  
         
