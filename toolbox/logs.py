
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Logs routines, storing parameters of the simulations and status**

:Authors: **Joel Repond**
'''

from __future__ import division
import yaml
import numpy as np
import h5py
#from builtins import range
#import numpy as np
#from scipy.special import ellipk, ellipe


class logs(object):
    def __init__(self, logFile, GeneralParameters, RFSectionParameters, Beam, Slices, userInputDic=None):
        
        self.logFile = logFile
        self.rf_params = RFSectionParameters#  To obtain nturns through counter
        
        self.nturns = int(GeneralParameters.n_turns)
        self.particle_type = GeneralParameters.particle_type
        self.circumference = float(GeneralParameters.ring_circumference)
        self.n_rf_systems = int(RFSectionParameters.n_rf)
#        self.gamma_transition = GeneralParameters
        self.harmonic_numbers_1 = int(RFSectionParameters.harmonic[0][0])
        self.n_macroparticles = int(Beam.n_macroparticles)
        self.intensity = float(Beam.intensity)
        self.n_slices = int(Slices.n_slices)
        self.cut_left = float(Slices.cut_left)
        self.cut_right = float(Slices.cut_right)
        
        self.dicParams = {'nturns total':self.nturns,
                     'particle type':self.particle_type,
                     'circumference':self.circumference,
                     'n RF system':self.n_rf_systems,
                     'harmonic number':self.harmonic_numbers_1,
                     'intensity totale':self.intensity,
                     'n slices total':self.n_slices,
                     'cut left':self.cut_left,
                     'cut right':self.cut_right}
        if userInputDic is not None:
            self.userInputDic=userInputDic
        else:
            self.userInputDic=dict()
#    yaml.dump({'nturns':10000,'parameters':{'circumference':4,'momentum':25},'userdefined':{'aaa':4,'bbb':'basdasd'}},default_flow_style=False)
    def save_log_file(self):
        with open(self.logFile,'w') as log_file:
            yaml.dump({'nturns':self.rf_params.counter, 'userDefined':self.userInputDic, 'params':self.dicParams}, log_file, default_flow_style=False)
            
    def track(self):
        self.save_log_file()

class variablesLogs(object):
    def __init__(self, h5file, attributeName):
        self.attributeLength = attributeName
        self.h5file = h5file
        
        self.dumpCounter = int(0)
        for key in self.attributeLength.keys():
            setattr(self,key,np.zeros(shape=attributeName[key]))

    def update(self, attributeValue):
        for key in self.attributeLength.keys():
            if self.attributeLength[key][0] > np.atleast_1d(attributeValue[key]).shape[0]:
                x = getattr(self,key)
                x[self.dumpCounter] = attributeValue[key]
            else:
                setattr(self,key,attributeValue[key])
        self.dumpCounter += 1
    
    def saveH5(self):
        with h5py.File(self.h5file, 'w') as h5file:
            for key in self.attributeLength.keys():
                x = h5file.create_dataset(key, self.attributeLength[key], compression="gzip", compression_opts=9, dtype=float, shuffle=True)
                y = getattr(self,key)
                x[:] = y
    
    def get(self,key):
        return getattr(self,key)
#
#    def loadH5(self):
#        loadedResults = h5py.File('../results/emit_%.2f_int_%.1f_dist_%.1f_V200_%.1f_V800_%.1f_imp_%s/saved_result.hdf5' %(emittance, intensity/1e11, exponent, voltage_program_1, voltage_program_2, impedance), 'r')
#                measurementTime = loadedResults['cycle_time_h5'][:]
