
# Copyright 2014 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_p, m_e, e, c

from plots.plot_settings import fig_folder


def preprocess_ramp(particle_type, nturns, circumference, time, momentum, 
                    data=None, interpolation='linear', flat_bottom=0, plot=True, 
                    figdir='fig', figname='data', user_mass=None, user_charge=None):
    '''
    Pre-process acceleration ramp data to create input for simulation parameters.
    Input absolute time [s] and corresponding momentum and optional data (e.g. 
    voltage) arrays; the revolution period will be calculated based on the p_s.
    Interpolate data points to simulation time steps.
    Constant extrapolation of first/last data point for flat bottom/top.
    Use flat_bottom to adjust the number of turns spent in flat bottom; flat top
    will be defined by total number of turns minus ramp and flat bottom. 
    '''

    # Define output arrays
    time_interp = np.ones(nturns+1) # accumulated revolution periods [s]
    ps_interp = np.ones(nturns+1)   # interpolated synchronous momentum [eV]
    data_interp = np.ones(nturns+1) # interpolated 

    # Attribution of mass and charge with respect to particle_type
    if particle_type is 'proton':
        mass =  m_p # [Kg]
        charge = e
    elif particle_type is 'electron':
        mass =  m_e # [Kg]
        charge = -e
    elif particle_type is 'user_input':
        mass = user_mass # [Kg]
        charge = user_charge
    else:
        raise RuntimeError('ERROR: Particle type not recognized!')

    # Obtain flat bottom data, extrapolate to constant
    beta = np.sqrt(1/(1 + (mass*c**2)**2/(momentum[0]*charge)**2))
    T0 = circumference/(beta*c)
    for i in xrange(0, flat_bottom+1):        
        time_interp[i] = i*T0
    ps_interp[0:flat_bottom+1] = momentum[0]
    data_interp[0:flat_bottom+1] = data[0] 

    # Shift input time to start of the ramp
    time += time_interp[flat_bottom] - time[0]   
    ni = len(time)              # Input array length
    
    if interpolation=='linear':
        
        i = flat_bottom + 1
        for k in xrange(1,ni):      
            while time_interp[i-1] <= time[k] and i < nturns+1:

                # Obtain next momentum/data point based on time passed during previous turns
                ps_interp[i] = (momentum[k-1]*(time[k]-time_interp[i-1]) + momentum[k]
                                *(time_interp[i-1]-time[k-1])) / (time[k] - time[k-1])
                  
                data_interp[i] = (data[k-1]*(time[k]-time_interp[i-1]) + data[k]
                                  *(time_interp[i-1]-time[k-1])) / (time[k] - time[k-1])
                    
                # Calculate the next revolution period based on update momentum              
                beta = np.sqrt(1/(1 + (mass*c**2)**2/(ps_interp[i]*charge)**2))
                T0 = circumference/(beta*c)
                time_interp[i] = time_interp[i-1] + T0
                
                i += 1
                kmax = k
                
    else:
        
        raise RuntimeError("WARNING: Interpolation scheme in preprocess_arrays \
                           not recognized. Aborting...")

    # Obtain flat top data, extrapolate to constant
    for m in xrange(i,nturns+1): 
        time_interp[m] = time_interp[m-1] + T0
    ps_interp[i:nturns+1] = momentum[kmax]
    data_interp[i:nturns+1] = data[kmax]
      
      
    if plot:
        # Directory where longitudinal_plots will be stored
        fig_folder(figdir)
        
        # Plot
        plt.figure(1, figsize=(8,6))
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        ax.plot(time_interp, ps_interp, label='Interpolated momentum')
        ax.plot(time[0:kmax], momentum[0:kmax], 'ro', label='Input momentum')
        ax.set_xlabel("Time [s]")    
        ax.set_ylabel (r"Momentum [eV]")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.legend()
    
        # Save figure
        fign = figdir +'/preprocess_momentum.png'
        plt.savefig(fign)
        plt.clf()   

        # Plot
        plt.figure(2, figsize=(8,6))
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        ax.plot(time_interp, data_interp, label='Interpolated data')
        ax.plot(time[0:kmax], data[0:kmax], 'ro', label='Input data')
        ax.set_xlabel("Time [s]")    
        ax.set_ylabel (r"Data [user-defined]")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.legend()
    
        # Save figure
        fign = figdir +'/preprocess_'"%s"%figname+'.png'
        plt.savefig(fign)
        plt.clf()   
   
    return ps_interp, data_interp  
        
        
        