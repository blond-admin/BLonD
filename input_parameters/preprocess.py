
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3), 
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Helga Timko**, **Alexandre Lasheen**, **Danilo Quartullo**
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from plots.plot import fig_folder
from scipy.constants import m_p, m_e, c, e
from scipy.interpolate import splrep, splev


def loaddata(filename, ignore=0, delimiter=None):

    """
    Loading column-by-column data from file to numpy arrays.
    Ignore x lines from the head of the file.
    """
    
    data = np.loadtxt(filename, skiprows=ignore, delimiter=delimiter)

    return [ np.ascontiguousarray(data[:,i]) for i in range(len(data[0])) ]



def preprocess_ramp(particle_type, circumference, time, data, 
                    data_type='momentum', interpolation='linear', smoothing = 0,
                    flat_bottom=0, flat_top=0, t_start=0, t_end=-1,
                    plot=False, figdir='fig', figname='data', sampling=1, 
                    user_mass=None, user_charge=None):
    '''
    Pre-process acceleration ramp data to create input for simulation parameters.
    Input: absolute time [s] and corresponding momentum [eV/c] or total energy [eV] or kinetic energy [eV].
    Output: cumulative time array [s], interpolated momentum [eV/c].
    'interpolation': restricted to linear and cubic at the moment.
    'flat_bottom/top': extra time can be be added in units of time steps;
    constant extrapolation of the first/last data point is used in this case. 
    't_start/end': cutting the inputed momentum program to the times to be simulated
    'plot': optional plotting of interpolated array with 'sampling' frequency; 
    saved with name 'figname' into 'figdir'.
    '''
    
    # Definitions
    Nd = len(time)
    if len(data) != Nd:
        raise RuntimeError(str(data)+' does not match the length of '+str(time))

    # Attribution of mass and charge with respect to particle_type
    if particle_type is 'proton':
        mass =  m_p*c**2/e # [eV]
    elif particle_type is 'electron':
        mass =  m_e*c**2/e # [eV]
    elif particle_type is 'user_input':
        mass = user_mass # [eV]
    else:
        raise RuntimeError('ERROR: Particle type in preprocess_ramp not recognized!')

    # Convert data to momentum, if necessary
    if data_type == 'momentum':
        momentum = data
    elif data_type == 'total energy':
        momentum = np.sqrt(data**2-mass**2)
    elif data_type == 'kinetic energy':
        momentum = np.sqrt((data+mass)**2-mass**2)
    else:
        raise RuntimeError('ERROR: Data type in preprocess_ramp not recognized!')
    
    
    # Obtain flat bottom data, extrapolate to constant
    beta_0 = np.sqrt(1/(1 + (mass/momentum[0])**2))
    T0 = circumference/(beta_0*c) # Initial revolution period [s]
    shift = time[0] - flat_bottom*T0
    time_interp = shift + T0*np.arange(0, flat_bottom+1)
    beta_interp = beta_0*np.ones(flat_bottom+1)
    momentum_interp = momentum[0]*np.ones(flat_bottom+1)
        
    time_interp = time_interp.tolist()
    beta_interp = beta_interp.tolist()
    momentum_interp = momentum_interp.tolist()
    
    time_start_ramp = np.max(time[momentum==momentum[0]])
    time_end_ramp = np.min(time[momentum==momentum[-1]])

    # Interpolate data recursively
    if interpolation=='linear':
        
        time_interp.append(time_interp[0]
                                     + circumference/(beta_interp[0]*c) )
        i = flat_bottom 
        for k in xrange(1,Nd): 
            while time_interp[i+1] <= time[k]:
                
                momentum_interp.append(momentum[k-1] + (momentum[k] - momentum[k-1]) * (time_interp[i+1] - time[k-1])
                                    / (time[k] - time[k-1])) 
                
                beta_interp.append(np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2))) 
                
                time_interp.append(time_interp[i+1]
                                     + circumference/(beta_interp[i+1]*c) )               
            
                i += 1
            
        time_interp.pop()        
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)   
                    
    
    elif interpolation=='cubic':
        
        interp_funtion_momentum = splrep(time[(time>=time_start_ramp)*(time<=time_end_ramp)], 
                                         momentum[(time>=time_start_ramp)*(time<=time_end_ramp)], 
                                         s=smoothing)
                  
        i = flat_bottom
       
        time_interp.append(time_interp[0]
                         + circumference/(beta_interp[0]*c) )
        
        while time_interp[i] <= time[-1]:

            if (time_interp[i+1] < time_start_ramp) :

                momentum_interp.append(momentum[0]) 
                
                beta_interp.append(np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2))) 
                
                time_interp.append(time_interp[i+1]
                                     + circumference/(beta_interp[i+1]*c) )
                                     
            elif (time_interp[i+1] > time_end_ramp):
                
                momentum_interp.append(momentum[-1]) 
                
                beta_interp.append(np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2))) 
                
                time_interp.append(time_interp[i+1]
                                     + circumference/(beta_interp[i+1]*c) )
                
            else:     

                momentum_interp.append(splev(time_interp[i+1], interp_funtion_momentum))
                
                beta_interp.append(np.sqrt(1/(1 + (mass/momentum_interp[i+1])**2))) 
                
                time_interp.append(time_interp[i+1]
                                         + circumference/(beta_interp[i+1]*c) )

            i += 1
        
        
        time_interp.pop()       
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)
        
                
    else:
        
        raise RuntimeError("WARNING: Interpolation scheme in preprocess_arrays \
                           not recognized. Aborting...")
    
    
    # Obtain flat top data, extrapolate to constant
    if flat_top > 0:
        time_interp = np.append(time_interp, time_interp[-1] + circumference*np.arange(1, flat_top+1)/(beta_interp[-1]*c))
        beta_interp = np.append(beta_interp, beta_interp[-1]*np.ones(flat_top))
        momentum_interp = np.append(momentum_interp, momentum_interp[-1]*np.ones(flat_top))
 
        
    # Cutting the input momentum on the desired cycle time
    if (t_start != 0) or (t_end != -1):
        if t_end == -1:
            t_end = time[-1]   
        momentum_interp = momentum_interp[(time_interp>=t_start)*(time_interp<=t_end)]
        time_interp = time_interp[(time_interp>=t_start)*(time_interp<=t_end)]
        
    if plot:
        # Directory where longitudinal_plots will be stored
        fig_folder(figdir)
        
        # Plot
        plt.figure(1, figsize=(8,6))
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        ax.plot(time_interp[::sampling], momentum_interp[::sampling], 
                label='Interpolated momentum')
        ax.plot(time, momentum, '.', label='input momentum', color='r', markersize=0.5)
        ax.set_xlabel("Time [s]")    
        ax.set_ylabel ("p [eV]")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                               ncol=2, mode="expand", borderaxespad=0.)
        
        # Save figure
        fign = figdir + '/preprocess_momentum.png'
        plt.savefig(fign)
        plt.clf()     
        
    return (time_interp, momentum_interp)
        


def preprocess_rf_params(general_params, time_arrays, data_arrays, interpolation='linear', smoothing = 0,
                         plot=True, figdir='fig', figname=['data'], sampling=1):
    
    """
    Pre-process RF programs to be input into RF parameters, such as RF voltage [V], 
    phase [rad], harmonic as a function of time [s].
    time_arrays and data_arrays are two lists of numpy arrays: thi first array of time_arrays
    corresponds to the first array of data_arrays and so on.
    Use 'loaddata' function to load data with correct format.
    Pre-requisite: general parameters need to be set up.
    'interpolation': restricted to linear at the moment.
    'flat_bottom': extrapolation to flat vector during given time steps;
    Flat top time automatically adjusted.
    'plot': optional plotting of interpolated array with 'sampling' frequency; 
    saved with name 'figname' into 'figdir'. Note that figname has to be a list of string where 
    each string corresponds to an interpolated array.
    """
    
    
    cumulative_times = general_params.cycle_time
    
    data_interp = []
    
    for i in range(len(time_arrays)):
        if len(time_arrays[i])!=len(data_arrays[i]):
            raise RuntimeError(str(data_arrays[i])+' does not match the length of '+str(time_arrays[i]))
        if interpolation=='linear':
            data_interp.append(np.interp(cumulative_times, time_arrays[i], data_arrays[i]))
        elif interpolation=='cubic':
            interp_funtion = splrep(time_arrays[i], data_arrays[i], s=smoothing)
            data_interp.append(splev(cumulative_times, interp_funtion))
            
    # Plot original and interpolated data       
    if plot:
        # Directory where longitudinal_plots will be stored
        fig_folder(figdir)
        
        # Plot
        for i in range(len(time_arrays)):
            plt.figure(1, figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(cumulative_times[::sampling], data_interp[i][::sampling], 
                    label='Interpolated data')
            ax.plot(time_arrays[i], data_arrays[i], '.', label='Input data', color='r')
            ax.set_xlabel("Time [s]")    
            ax.set_ylabel ("%s" %figname[i])
            ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                                   ncol=2, mode="expand", borderaxespad=0.)

        
            # Save figure
            fign = figdir + '/preprocess_' "%s" %figname[i] + '_' "%d" %i +'.png'
            plt.savefig(fign)
            plt.clf()     
 
    return data_interp



        