
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public License version 3 (GPL Version 3), 
# copied verbatim in the file LICENSE.md.
# In applying this license, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Function(s) for pre-processing input data**

:Authors: **Helga Timko**, **Alexandre Lasheen**, **Danilo Quartullo**, 
    **Simon Albright**
'''

from __future__ import division
from builtins import str, range
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



class PreprocessRamp(object):
    r""" Class to preprocess the synchronous data for GeneralParameters
    
    Parameters
    ----------
    interpolation : string
        Interpolation options for the data points. Available options are 
        'linear' (default), 'cubic', and 'derivative'.
    smoothing : float
        Smoothing value for 'cubic' interpolation
    flat_bottom : int
        Number of turns to be added on flat bottom; default is 0. Constant
        extrapolation is used for the synchronous data
    flat_top : int
        Number of turns to be added on flat top; default is 0. Constant
        extrapolation is used for the synchronous data
    t_start : int
        Starting index from which the time array input should be taken into
        account; default is 0
    t_end : int
        Last index up to which the time array input should be taken into
        account; default is -1
    plot : boolean
        Option to plot interpolated arrays; default is False
    figdir : string
        Directory to save optional plot; default is 'fig'
    figname : string
        Figure name to save optional plot; default is 'preprocess_ramp'
    sampling : int
        Decimation value for plotting; default is 1
    
    """
    def __init__(self, interpolation='linear', smoothing = 0, flat_bottom = 0, 
                 flat_top = 0, t_start = 0, t_end = -1, plot = False, 
                 figdir= 'fig', figname = 'preprocess_ramp', sampling = 1):
        
        if interpolation in ['linear', 'cubic', 'derivative']:
            self.interpolation = str(interpolation)
        else:    
            raise RuntimeError('ERROR: Interpolation scheme in PreprocessRamp'+
                               ' not recognized. Aborting...')
        self.smoothing = float(smoothing)
        if flat_bottom < 0:
            raise RuntimeError('ERROR: flat_bottom value in PreprocessRamp'+
                               ' not recognized. Aborting...')
        else:          
            self.flat_bottom = int(flat_bottom)
        if flat_top < 0:
            raise RuntimeError('ERROR: flat_top value in PreprocessRamp'+
                               ' not recognized. Aborting...')
        else:          
            self.flat_top = int(flat_top)            
        self.t_start = int(t_start)
        self.t_end = int(t_end)
        if plot == True or plot == False:
            self.plot = bool(plot)
        else: 
            raise RuntimeError('ERROR: plot value in PreprocessRamp'+
                               ' not recognized. Aborting...')            
        self.figdir = str(figdir)
        self.figname = str(figname)
        if sampling > 0:
            self.sampling = int(sampling)
        else:
            raise RuntimeError('ERROR: sampling value in PreprocessRamp'+
                               ' not recognized. Aborting...')            
            

    def preprocess(self, mass, circumference, time, momentum):#, #data, #data_type='momentum', 
#                    interpolation='linear', smoothing = 0,
#                    flat_bottom=0, flat_top=0, t_start=0, t_end=-1,
#                    plot=False, figdir='fig', figname='data', sampling=1): #, 
#                    user_mass=None, user_charge=None):
        """
        Pre-process acceleration ramp data to create input for simulation parameters.
        Input: absolute time [s] and corresponding momentum [eV/c] or total energy [eV] or kinetic energy [eV].
        Output: cumulative time array [s], interpolated momentum [eV/c].
        """
#        'interpolation': restricted to linear and cubic at the moment.
#        'flat_bottom/top': extra time can be be added in units of time steps;
#        constant extrapolation of the first/last data point is used in this case. 
#        't_start/end': cutting the inputed momentum program to the times to be simulated
#        'plot': optional plotting of interpolated array with 'sampling' frequency; 
#        saved with name 'figname' into 'figdir'.
#        '''
        
        # Definitions
    #    Nd = len(time)
    #    if len(data) != Nd:
    #        raise RuntimeError(str(data)+' does not match the length of '+str(time))
    
        # Attribution of mass and charge with respect to particle_type
    #     if particle_type is 'proton':
    #         mass =  m_p*c**2/e # [eV]
    #     elif particle_type is 'electron':
    #         mass =  m_e*c**2/e # [eV]
    #     elif particle_type is 'user_input':
    #         mass = user_mass # [eV]
    #     else:
    #         raise RuntimeError('ERROR: Particle type in preprocess_ramp not recognized!')
    
        # Convert data to momentum, if necessary
    #     if data_type == 'momentum':
    #         momentum = data
    #     elif data_type == 'total energy':
    #         momentum = np.sqrt(data**2-mass**2)
    #     elif data_type == 'kinetic energy':
    #         momentum = np.sqrt((data+mass)**2-mass**2)
    #     else:
    #         raise RuntimeError('ERROR: Data type in preprocess_ramp not recognized!')
        
        # Some checks on the options
        if self.t_start < 0 or self.t_start > len(time)-1:
            raise RuntimeError('ERROR: t_start value in PreprocessRamp'+
                               ' does not match the time array length')            
        if np.abs(self.t_end) > len(time)-1:   
            raise RuntimeError('ERROR: t_end value in PreprocessRamp'+
                               ' does not match the time array length')            
        
        # Obtain flat bottom data, extrapolate to constant
        beta_0 = np.sqrt(1/(1 + (mass/momentum[0])**2))
        T0 = circumference/(beta_0*c) # Initial revolution period [s]
        shift = time[0] - self.flat_bottom*T0
        time_interp = shift + T0*np.arange(0, self.flat_bottom+1)
        beta_interp = beta_0*np.ones(self.flat_bottom+1)
        momentum_interp = momentum[0]*np.ones(self.flat_bottom+1)
        
        time_interp = time_interp.tolist()
        beta_interp = beta_interp.tolist()
        momentum_interp = momentum_interp.tolist()
        
        time_start_ramp = np.max(time[momentum==momentum[0]])
        time_end_ramp = np.min(time[momentum==momentum[-1]])
    
        # Interpolate data recursively
        if self.interpolation=='linear':
            
            time_interp.append(time_interp[-1]
                               + circumference/(beta_interp[0]*c) )
    
            i = self.flat_bottom 
            for k in range(1,len(time)): 
                while time_interp[i+1] <= time[k]:
                    
                    momentum_interp.append(momentum[k-1] + (momentum[k] 
                        - momentum[k-1]) * (time_interp[i+1] - time[k-1])
                        / (time[k] - time[k-1])) 
                    
                    beta_interp.append(np.sqrt(1/(1 
                        + (mass/momentum_interp[i+1])**2))) 
                    
                    time_interp.append(time_interp[i+1]
                        + circumference/(beta_interp[i+1]*c) )               
                
                    i += 1
                
        elif self.interpolation=='cubic':
            
            interp_funtion_momentum = splrep(time[(time>=time_start_ramp) \
                *(time<=time_end_ramp)], momentum[(time>=time_start_ramp) \
                *(time<=time_end_ramp)], s=self.smoothing)
                      
            i = self.flat_bottom
           
            time_interp.append(time_interp[-1] + circumference/
                               (beta_interp[0]*c))
            
            while time_interp[i] <= time[-1]:
    
                if (time_interp[i+1] < time_start_ramp):
    
                    momentum_interp.append(momentum[0]) 
                    
                    beta_interp.append(np.sqrt(1/(1 
                        + (mass/momentum_interp[i+1])**2))) 
                    
                    time_interp.append(time_interp[i+1]
                        + circumference/(beta_interp[i+1]*c) )
                                         
                elif (time_interp[i+1] > time_end_ramp):
                    
                    momentum_interp.append(momentum[-1]) 
                    
                    beta_interp.append(np.sqrt(1/(1 
                        + (mass/momentum_interp[i+1])**2))) 
                    
                    time_interp.append(time_interp[i+1]
                        + circumference/(beta_interp[i+1]*c) )
                    
                else:     
    
                    momentum_interp.append(splev(time_interp[i+1], 
                        interp_funtion_momentum))
                    
                    beta_interp.append(np.sqrt(1/(1 
                        + (mass/momentum_interp[i+1])**2))) 
                    
                    time_interp.append(time_interp[i+1]
                        + circumference/(beta_interp[i+1]*c) )
    
                i += 1
            
        # Interpolate momentum in 1st derivative to maintain smooth B-dot
        elif self.interpolation == 'derivative':
    
            momentum_initial = momentum_interp[0]
            momentum_derivative = np.gradient(momentum)/np.gradient(time)
    
            momentum_derivative_interp = [0]*self.flat_bottom + \
                [momentum_derivative[0]]
            integral_point = momentum_initial
    
            i = self.flat_bottom
    
            time_interp.append(time_interp[-1]
                             + circumference/(beta_interp[0]*c) )
    
            while time_interp[i] <= time[-1]:
    
                derivative_point = np.interp(time_interp[i+1], time, 
                                             momentum_derivative)
                momentum_derivative_interp.append(derivative_point)
                integral_point += (time_interp[i+1] - time_interp[i]) \
                    * derivative_point
    
                momentum_interp.append(integral_point)
                beta_interp.append(np.sqrt(1/(1 + (mass/
                                                   momentum_interp[i+1])**2)))
                time_interp.append(time_interp[i+1]
                    + circumference/(beta_interp[i+1]*c) )
    
                i += 1
    
            #Adjust result to get flat top energy correct as derivation and
            #integration leads to ~10^-8 error in flat top momentum
            momentum_interp = np.asarray(momentum_interp)
            momentum_interp -= momentum_interp[0]
            momentum_interp /= momentum_interp[-1]
            momentum_interp *= momentum[-1] - momentum[0]
    
            momentum_interp += momentum[0]
           
     
      
        time_interp.pop()
        time_interp = np.asarray(time_interp)
        beta_interp = np.asarray(beta_interp)
        momentum_interp = np.asarray(momentum_interp)
    
        # Obtain flat top data, extrapolate to constant
        if self.flat_top > 0:
            time_interp = np.append(time_interp, time_interp[-1] 
                + circumference*np.arange(1, self.flat_top+1)
                /(beta_interp[-1]*c))
            beta_interp = np.append(beta_interp, beta_interp[-1]
                                    *np.ones(self.flat_top))
            momentum_interp = np.append(momentum_interp, 
                momentum_interp[-1]*np.ones(self.flat_top))
     
            
        # Cutting the input momentum on the desired cycle time
        if (self.t_start != 0) or (self.t_end != -1):
            if self.t_end == -1:
                t_end = time[-1]   
            momentum_interp = momentum_interp[(time_interp>=self.t_start) \
                *(time_interp<=t_end)]
            time_interp = time_interp[(time_interp>=self.t_start) \
                *(time_interp<=t_end)]
            
        if self.plot:
            # Directory where longitudinal_plots will be stored
            fig_folder(self.figdir)
            
            # Plot
            plt.figure(1, figsize=(8,6))
            ax = plt.axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(time_interp[::self.sampling], 
                    momentum_interp[::self.sampling], 
                    label='Interpolated momentum')
            ax.plot(time, momentum, '.', label='input momentum', color='r', 
                    markersize=0.5)
            ax.set_xlabel("Time [s]")    
            ax.set_ylabel ("p [eV]")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, 
                                   ncol=2, mode="expand", borderaxespad=0.)
            
            # Save figure
            fign = self.figdir + '/preprocess_momentum.png'
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



def combine_rf_functions(function_list, merge_type = 'linear', resolution = 1E-3, general_params = None, mainH = True):

    """
    function to merge different programs in case e.g. different fixed bucket areas are required at different points in time.
    function_list contains 2-tuples in the form (program, start/stop time), where program is a 2-D numpy array or a single
    value and start/stop time is a list of length 2, with the first member the start time and the second the stop time.
    merge_type can be 'linear' or 'adiabatic' or a list in case different merge types are required between different functions.
    resolution determines the time points along the merge for non-linear merge types and can also be a list if required.
    """

    nFunctions = len(function_list)

    if not isinstance(merge_type, list):
        merge_type = (nFunctions-1)*[merge_type]
    if not isinstance(resolution, list):
        resolution = (nFunctions-1)*[resolution]
    
    timePoints = []
    for i in range(nFunctions):
        timePoints += function_list[i][1]
    
    if not np.all(np.diff(timePoints)) > 0:
        print("Error, times not monotonically increasing")
    
    fullFunction = []
    fullTime = []
    
    if not isinstance(function_list[0][0], np.ndarray):
        fullFunction += 2*[function_list[0][0]]
        fullTime += function_list[0][1]
    
    else:
        start = np.where(function_list[0][0][0] > function_list[0][1][0])[0][0]
        stop = np.where(function_list[0][0][0] > function_list[0][1][1])[0][0]

        funcTime = [function_list[0][1][0]] + function_list[0][0][0][start:stop].tolist() + [function_list[0][1][1]]
        funcProg = np.interp(funcTime, function_list[0][0][0], function_list[0][0][1])
        
        fullFunction += funcProg.tolist()
        fullTime += funcTime

    
    for i in range(1, nFunctions):
        
        if merge_type[i-1] == 'linear':
            
            if not isinstance(function_list[i][0], np.ndarray):
                fullFunction += 2*[function_list[i][0]]
                fullTime += function_list[i][1]
                
            else:
                start = np.where(function_list[i][0][0] >= function_list[i][1][0])[0][0]
                stop = np.where(function_list[i][0][0] >= function_list[i][1][1])[0][0]
                
                funcTime = [function_list[i][1][0]] + function_list[i][0][0][start:stop].tolist() + [function_list[i][1][1]]
                funcProg = np.interp(funcTime, function_list[i][0][0], function_list[i][0][1])
                
                fullFunction += funcProg.tolist()
                fullTime += funcTime
                
        elif merge_type[i-1] == 'isoadiabatic':
            
            if not isinstance(function_list[i][0], np.ndarray):
                
                tDur = function_list[i][1][0] - fullTime[-1]
                Vinit = fullFunction[-1]
                Vfin = function_list[i][0]
                k = (1./tDur)*(1-(1.*Vinit/Vfin)**0.5)
                
                nSteps = int(tDur/resolution[i-1])
                time = np.linspace(fullTime[-1], function_list[i][1][0], nSteps)
                volts = Vinit/((1-k*(time-time[0]))**2)
                
                fullFunction += volts.tolist() + 2*[function_list[i][0]]
                fullTime += time.tolist() + function_list[i][1]
                
            else:
                
                start = np.where(function_list[i][0][0] >= function_list[i][1][0])[0][0]
                stop = np.where(function_list[i][0][0] >= function_list[i][1][1])[0][0]
                
                funcTime = [function_list[i][1][0]] + function_list[i][0][0][start:stop].tolist() + [function_list[i][1][1]]
                funcProg = np.interp(funcTime, function_list[i][0][0], function_list[i][0][1])
                
                tDur = funcTime[0] - fullTime[-1]
                Vinit = fullFunction[-1]
                Vfin = funcProg[0]
                k = (1./tDur)*(1-(1.*Vinit/Vfin)**0.5)
                
                nSteps = int(tDur/resolution[i-1])
                time = np.linspace(fullTime[-1], funcTime[0], nSteps)
                volts = Vinit/((1-k*(time-time[0]))**2)
                
                fullFunction += volts.tolist() + funcProg.tolist()
                fullTime += time.tolist() + funcTime
                
        elif merge_type[i-1] == 'linear_tune':
            
            #harmonic, charge and 2pi are constant so can be ignored
            if not isinstance(function_list[i][0], np.ndarray):
                
                initPars = general_params.parameters_at_time(fullTime[-1])
                finalPars = general_params.parameters_at_time(function_list[i][1][0])
                
                vInit = fullFunction[-1]
                vFin = function_list[i][0]
                
                if mainH is False:
                    initPars['delta_E'] = 0.
                    finalPars['delta_E'] = 0.
                    
                initTune = np.sqrt( (vInit * np.abs(initPars['eta_0']) * np.sqrt(1 - (initPars['delta_E']/vInit)**2)) / (initPars['beta']**2 * initPars['energy']) )
                finalTune = np.sqrt( (vFin * np.abs(finalPars['eta_0']) * np.sqrt(1 - (finalPars['delta_E']/vFin)**2)) / (finalPars['beta']**2 * finalPars['energy']) )
                
                tDur = function_list[i][1][0] - fullTime[-1]
                nSteps = int(tDur/resolution[i-1])
                time = np.linspace(fullTime[-1], function_list[i][1][0], nSteps)
                tuneInterp = np.linspace(initTune, finalTune, nSteps)
                
                mergePars = general_params.parameters_at_time(time)
                
                if mainH is False:
                    mergePars['delta_E'] *= 0
                    
                volts = np.sqrt( ((tuneInterp**2 * mergePars['beta']**2 * mergePars['energy']) / (np.abs(mergePars['eta_0'])))**2 + mergePars['delta_E']**2)
                
                fullFunction += volts.tolist() + 2*[function_list[i][0]]
                fullTime += time.tolist() + function_list[i][1]
                
            else:
                
                start = np.where(function_list[i][0][0] >= function_list[i][1][0])[0][0]
                stop = np.where(function_list[i][0][0] >= function_list[i][1][1])[0][0]
                
                funcTime = [function_list[i][1][0]] + function_list[i][0][0][start:stop].tolist() + [function_list[i][1][1]]
                funcProg = np.interp(funcTime, function_list[i][0][0], function_list[i][0][1])
                
                tDur = funcTime[0] - fullTime[-1]
                nSteps = int(tDur/resolution[i-1])
                time = np.linspace(fullTime[-1], funcTime[0], nSteps)
                
                initPars = general_params.parameters_at_time(fullTime[-1])
                finalPars = general_params.parameters_at_time(funcTime[0])
                
                if mainH is False:
                    initPars['delta_E'] = 0.
                    finalPars['delta_E'] = 0.
                    
                vInit = fullFunction[-1]
                vFin = funcProg[0]
                
                initTune = np.sqrt( (vInit * np.abs(initPars['eta_0']) * np.sqrt(1 - (initPars['delta_E']/vInit)**2)) / (initPars['beta']**2 * initPars['energy']) )
                finalTune = np.sqrt( (vFin * np.abs(finalPars['eta_0']) * np.sqrt(1 - (finalPars['delta_E']/vFin)**2)) / (finalPars['beta']**2 * finalPars['energy']) )
                tuneInterp = np.linspace(initTune, finalTune, nSteps)
                
                mergePars = general_params.parameters_at_time(time)
                
                if mainH is False:
                    mergePars['delta_E'] *= 0
                    
                volts = np.sqrt( ((tuneInterp**2 * mergePars['beta']**2 * mergePars['energy']) / (np.abs(mergePars['eta_0'])))**2 + mergePars['delta_E']**2)
                
                fullFunction += volts.tolist() + funcProg.tolist()
                fullTime += time.tolist() + funcTime
                
                
    returnFunction = np.zeros([2, len(fullTime)])
    returnFunction[0] = fullTime
    returnFunction[1] = fullFunction
    
    return returnFunction
