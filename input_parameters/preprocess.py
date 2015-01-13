
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3), 
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities 
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
import time
import sys, copy


def loaddata(filename, ignore=0):

    """
    Loading column-by-column data from file to numpy arrays.
    Ignore x lines from the head of the file.
    """
    
    data = np.loadtxt(filename, skiprows=ignore)

    return [ np.ascontiguousarray(data[:,i]) for i in range(len(data[0])) ]


def preprocess_ramp(particle_type, circumference, time_input, data_input, 
                    data_type='momentum', interpolation='linear', 
                    flat_bottom=0, flat_top=0, 
                    plot=True, figdir='fig', figname='data', sampling=1, 
                    user_mass=None, user_charge=None):
    '''
    Pre-process acceleration ramp data to create input for simulation parameters.
    Input: absolute time [s] and corresponding momentum [eV] or gamma or beta; 
    Interpolate data points to simulation time steps.
    'interpolation': restricted to linear at the moment.
    'flat_bottom/top': extra time can be be added in units of time steps;
    constant extrapolation of the first/last data point is used in this case. 
    'plot': optional plotting of interpolated array with 'sampling' frequency; 
    saved with name 'figname' into 'figdir'.
    '''

    # Definitions
    Nd = len(time_input)
    if len(data_input) != Nd:
        raise RuntimeError(str(data_input)+' does not match the length of '+str(time_input))
    time_interp = []                   # accumulated revolution periods [s]
    data_interp = []                   # interpolated synchronous momentum [eV]

    # Attribution of mass and charge with respect to particle_type
    if particle_type is 'proton':
        mass =  m_p # [kg]
        charge = e
    elif particle_type is 'electron':
        mass =  m_e # [kg]
        charge = -e
    elif particle_type is 'user_input':
        mass = user_mass # [kg]
        charge = user_charge
    else:
        raise RuntimeError('ERROR: Particle type in preprocess_rampnot recognized!')

    # Convert data_input to beta, if necessary
    if data_type == 'momentum':
        beta = np.sqrt(1/(1 + (mass*c**2)**2/(data_input*charge)**2))
    elif data_type == 'gamma':
        beta = np.sqrt(1 - 1/(data_input*data_input))
    elif data_type != 'beta':
        raise RuntimeError('ERROR: Ramp data in preprocess_ramp not recognized!')
    
    # Obtain flat bottom data, extrapolate to constant  
    T0 = circumference/(beta[0]*c)     # Initial revolution period 
    shift = time_input[0] - flat_bottom*T0
    time_interp = np.append(time_interp, shift + T0*np.arange(0, flat_bottom+1))
    data_interp = np.append(data_interp, beta[0]*np.ones(flat_bottom+1))
    
    # Interpolate data recursively
    if interpolation=='linear':
        
        time_interp = time_interp.tolist()
        data_interp = data_interp.tolist()
        i = flat_bottom+1     # Counter for time steps
        
        for k in xrange(1,Nd):         # Counter for data points 
            while time_interp[i-1] <= time_input[k]:
                
                data_interp.append((beta[k-1]*(time_input[k]
                                         -time_interp[i-1]) + beta[k]*(time_interp[i-1]
                                         -time_input[k-1])) / (time_input[k] - time_input[k-1]) )
                time_interp.append(time_interp[i-1] + circumference/(data_interp[i]*c) )     
                
                i += 1
                
        time_interp=np.asarray(time_interp)
        data_interp=np.asarray(data_interp) 
        
    else:
         
        raise RuntimeError("WARNING: Interpolation scheme in preprocess_arrays \
                           not recognized. Aborting...")
    
    # Obtain flat top data, extrapolate to constant
    if flat_top > 0:
        T0 = circumference/(beta[-1]*c)
        time_interp = np.append(time_interp, 
                                time_interp[-1] + T0*np.arange(1, flat_top+1))
        data_interp = np.append(data_interp, beta[-1]*np.ones(flat_top))
    
    # Convert data back, if necessary
    if data_type == 'momentum':
        data_interp = mass*c**2/charge \
                    *np.sqrt( 1/(1/(data_interp*data_interp) - 1) )
    elif data_type == 'gamma':
        data_interp = np.sqrt( 1/(1 - data_interp*data_interp) )
      
    if plot:
        # Directory where longitudinal_plots will be stored
        fig_folder(figdir)
        
        # Plot
        plt.figure(1, figsize=(8,6))
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        ax.plot(time_interp[::sampling], data_interp[::sampling], 
                label='Interpolated data')
        ax.plot(time_input[0:Nd], data_input[0:Nd], 'ro', label='Input data')
        ax.set_xlabel("Time [s]")    
        ax.set_ylabel ("%s" %figname)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                               ncol=2, mode="expand", borderaxespad=0.)
        
        # Save figure
        fign = figdir + '/preprocess_' "%s" %figname +'.png'
        plt.savefig(fign)
        plt.clf()     
  
    return data_interp  
        


def preprocess_rf_params(general_params, time_input, data_input, interpolation='linear', 
                         flat_bottom=0, 
                         plot=True, figdir='fig', figname='data', sampling=1):
    
    """
    Pre-process 'data_input' to be input into RF parameters, such as RF voltage, 
    phase, harmonic as a function of 'time'.
    'data' can be a single np array or a list of arrays corresponding to 'time'.
    Use 'loaddata' function to load data with correct format.
    Pre-requisite: general parameters need to be set up.
    'interpolation': restricted to linear at the moment.
    'flat_bottom': extrapolation to flat vector during given time steps;
    Flat top time automatically adjusted.
    'plot': optional plotting of interpolated array with 'sampling' frequency; 
    saved with name 'figname' into 'figdir'.
    """
    
    T0 = general_params.t_rev          # Revolution period
    Nt = general_params.n_turns        # Number of turns
    Nd = len(time_input)                     # Number of data points

    # Check input format of 'data_input'
    # Single numpy array
    if isinstance(data_input, np.ndarray) and data_input.ndim == 1: 
        Na = 1                         # Number of arrays
        data_input = np.array(data_input, ndmin =2)
        if data_input.size != Nd:
            raise RuntimeError(str(data_input)+' does not match the length of '+str(time_input))
    # List of numpy arrays
    elif isinstance(data_input, list) and isinstance(data_input[0], np.ndarray): 
        data_input = np.array(data_input, ndmin =2)
        Na = len(data_input)                 # Number of arrays
        if data_input[0].size != Nd: 
            raise RuntimeError(str(data_input)+' does not match the length of '+str(time_input))
    else:
        raise RuntimeError('Data format not recognized in preprocess_rf_params()')
            
    # Initialise data; constant at flat bottom
    data_interp = np.zeros((Na, Nt+1))
    time_interp = np.zeros(Nt+1)
    data_interp[:,0:flat_bottom+1] = data_input[:,0]
    shift = time_input[0] - flat_bottom*T0[0]
    time_interp[0:flat_bottom+1] = shift + T0[0]*np.arange(0, flat_bottom+1)
    
    # Interpolate data
    if interpolation=='linear':
        
        i = flat_bottom+1
        for k in xrange(1,Nd):      
            while time_interp[i-1] <= time_input[k]: 

                # Obtain next data point based on time passed during previous turns
                data_interp[:,i] = (data_input[:,k-1]*(time_input[k]-time_interp[i-1]) + data_input[:,k]
                                 *(time_interp[i-1]-time_input[k-1])) / (time_input[k] - time_input[k-1])
                                     
                # Update time array              
                time_interp[i] = time_interp[i-1] + T0[i-1]              
                i += 1

    else:
        
        raise RuntimeError("WARNING: Interpolation scheme in preprocess_arrays \
                           not recognized. Aborting...")
        
    # Obtain flat top data, extrapolate to constant
    flat_top = Nt+1-i
    print "    In preprocess_rf_params, adding %d flat top time steps" %flat_top
    if flat_top > 0:
        time_interp[i:] = time_interp[i-1] + T0[i]*np.arange(1, Nt+2-i)
        data_interp[:,i:] = data_interp[:,i-1]
    if flat_top < 0:
        raise RuntimeError("Number of turns contradicts data length in preprocess_rf_params!")
    
    # Plot original and interpolated data       
    if plot:
        # Directory where longitudinal_plots will be stored
        fig_folder(figdir)
        
        # Plot
        for i in xrange(0,Na):
            plt.figure(1, figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(time_interp[::sampling], data_interp[i,::sampling], 
                    label='Interpolated data')
            ax.plot(time_input[0:Nd], data_input[i,0:Nd], 'ro', label='Input data')
            ax.set_xlabel("Time [s]")    
            ax.set_ylabel ("%s" %figname)
            ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                                   ncol=2, mode="expand", borderaxespad=0.)

            # Save figure
            fign = figdir + '/preprocess_' "%s" %figname + '_' "%d" %i +'.png'
            plt.savefig(fign)
            plt.clf()     
 
    return data_interp



        