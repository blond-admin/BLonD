
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module to save beam statistics in h5 files**

:Authors: **Danilo Quartullo**, **Helga Timko**
'''

from builtins import object
import h5py as hp
import numpy as np



class BunchMonitor(object):
    
    ''' Class able to save bunch data into h5 file. Use 'buffer_time' to select 
        the frequency of saving to file in number of turns.
        If in the constructor a Profile object is passed, that means that one
        wants to save the gaussian-fit bunch length as well (obviously the 
        Profile object has to have the fit_option set to 'gaussian').
    '''
    
    def __init__(self, Ring, RFParameters, Beam, filename, 
                 buffer_time = None, 
                 Profile = None, PhaseLoop = None, LHCNoiseFB = None):
        
        self.filename = filename
        self.n_turns = Ring.n_turns
        self.i_turn = 0
        self.buffer_time = buffer_time
        if buffer_time == None:
            self.buffer_time = self.n_turns
        self.rf_params = RFParameters
        self.beam = Beam
        self.profile = Profile
        if self.profile:
            if self.profile.fit_option!=None:
                self.fit_option = True
            else:
                self.fit_option = False
        else:
            self.fit_option = False
        self.PL = PhaseLoop
        self.LHCNoiseFB = LHCNoiseFB

        # Initialise data and save initial state
        self.init_data( self.filename, (self.n_turns + 1,))
        
        # Track at initialisation
        self.track()          

    
    def track(self):
        
        self.beam.statistics()

        # Write buffer with i_turn = RFcounter - 1                
        self.write_buffer()

        # Synchronise to i_turn = RFcounter
        self.i_turn += 1

        if self.i_turn > 0 and ( self.i_turn % self.buffer_time ) == 0:
            self.open()
            self.write_data(self.h5file['Beam'], (self.n_turns + 1,))
            self.close()
            self.init_buffer()    

    
    def init_data(self, filename, dims):
         
        # Prepare data
        self.beam.statistics()

        # Open file
        self.h5file = hp.File(filename + '.h5', 'w')
        self.h5file.require_group('Beam')
         
        # Create datasets and write first data points  
        h5group = self.h5file['Beam']
               
        h5group.create_dataset("n_macroparticles_alive", shape = dims, 
                               dtype = 'f', 
                               compression = "gzip", compression_opts = 9)
        h5group["n_macroparticles_alive"][0] = self.beam.n_macroparticles_alive
        
        h5group.create_dataset("mean_dt", shape = dims, dtype = 'f',
                               compression = "gzip", compression_opts = 9)
        h5group["mean_dt"][0]   = self.beam.mean_dt
         
        h5group.create_dataset("mean_dE", shape = dims, dtype = 'f',
                               compression = "gzip", compression_opts = 9)
        h5group["mean_dE"][0]  = self.beam.mean_dE
         
        h5group.create_dataset("sigma_dt", shape = dims, dtype = 'f',
                               compression = "gzip", compression_opts = 9)
        h5group["sigma_dt"][0]  = self.beam.sigma_dt 
         
        h5group.create_dataset("sigma_dE", shape = dims, dtype = 'f',
                               compression = "gzip", compression_opts = 9)
        h5group["sigma_dE"][0] = self.beam.sigma_dE
         
        h5group.create_dataset("epsn_rms_l", shape = dims, dtype = 'f',
                               compression = "gzip", compression_opts = 9)
        h5group["epsn_rms_l"][0]   = self.beam.epsn_rms_l
         
        if self.fit_option == True:

            h5group.create_dataset("bunch_length", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["bunch_length"][0] = self.profile.bunchLength
             
        if self.PL:
             
            h5group.create_dataset("PL_omegaRF", shape = dims, 
                                   dtype = np.float64,
                                   compression = "gzip", compression_opts = 9)
            h5group["PL_omegaRF"][0] = self.rf_params.omega_rf[0,0]

            h5group.create_dataset("PL_phiRF", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["PL_phiRF"][0] = self.rf_params.phi_rf[0,0]

            h5group.create_dataset("PL_bunch_phase", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["PL_bunch_phase"][0] = self.PL.phi_beam
 
            h5group.create_dataset("PL_phase_corr", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["PL_phase_corr"][0] = self.PL.dphi
 
            h5group.create_dataset("PL_omegaRF_corr", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["PL_omegaRF_corr"][0] = self.PL.domega_rf

            h5group.create_dataset("SL_dphiRF", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["SL_dphiRF"][0] = self.rf_params.dphi_rf[0]

            h5group.create_dataset("RL_drho", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["RL_drho"][0] = self.PL.drho
             
        if self.LHCNoiseFB:
             
            h5group.create_dataset("LHC_noise_FB_factor", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["LHC_noise_FB_factor"][0] = self.LHCNoiseFB.x
             
            h5group.create_dataset("LHC_noise_FB_bl", shape = dims, 
                                   dtype = 'f',
                                   compression = "gzip", compression_opts = 9)
            h5group["LHC_noise_FB_bl"][0] = self.LHCNoiseFB.bl_meas
            
            if self.LHCNoiseFB.bl_meas_bbb != None:
                
                h5group.create_dataset("LHC_noise_FB_bl_bbb", 
                                       shape = (self.n_turns + 1, 
                                                len(self.LHCNoiseFB.bl_meas_bbb)), 
                                       dtype = 'f', compression = "gzip", 
                                       compression_opts = 9)
                h5group["LHC_noise_FB_bl_bbb"][0,:] = self.LHCNoiseFB.bl_meas_bbb[:]
                 
        # Close file
        self.close() 
        
        # Initialise buffer for next turn
        self.init_buffer()       
        

    def init_buffer(self):
        
        self.b_np_alive = np.zeros(self.buffer_time)
        self.b_mean_dt = np.zeros(self.buffer_time)
        self.b_mean_dE = np.zeros(self.buffer_time)
        self.b_sigma_dt = np.zeros(self.buffer_time)
        self.b_sigma_dE = np.zeros(self.buffer_time)
        self.b_epsn_rms = np.zeros(self.buffer_time)
         
        if self.fit_option == True:
             
            self.b_bl = np.zeros(self.buffer_time)
             
        if self.PL:
             
            self.b_PL_omegaRF = np.zeros(self.buffer_time)
            self.b_PL_phiRF = np.zeros(self.buffer_time)
            self.b_PL_bunch_phase = np.zeros(self.buffer_time)
            self.b_PL_phase_corr = np.zeros(self.buffer_time)
            self.b_PL_omegaRF_corr = np.zeros(self.buffer_time)
            self.b_SL_dphiRF = np.zeros(self.buffer_time)
            self.b_RL_drho = np.zeros(self.buffer_time)
             
        if self.LHCNoiseFB:
             
            self.b_LHCnoiseFB_factor = np.zeros(self.buffer_time)
            self.b_LHCnoiseFB_bl = np.zeros(self.buffer_time)
            if self.LHCNoiseFB.bl_meas_bbb != None:
                self.b_LHCnoiseFB_bl_bbb = np.zeros((self.buffer_time, 
                                           len(self.LHCNoiseFB.bl_meas_bbb)))
                

    def write_buffer(self):
        
        i = self.i_turn % self.buffer_time

        self.b_np_alive[i] = self.beam.n_macroparticles_alive
        self.b_mean_dt[i] = self.beam.mean_dt
        self.b_mean_dE[i] = self.beam.mean_dE
        self.b_sigma_dt[i] = self.beam.sigma_dt
        self.b_sigma_dE[i] = self.beam.sigma_dE
        self.b_epsn_rms[i] = self.beam.epsn_rms_l
         
        if self.fit_option == True:
             
            self.b_bl[i] = self.profile.bunchLength
             
        if self.PL:
             
            self.b_PL_omegaRF[i] = self.rf_params.omega_rf[0,self.i_turn]
            self.b_PL_phiRF[i] = self.rf_params.phi_rf[0,self.i_turn]
            self.b_PL_bunch_phase[i] = self.PL.phi_beam
            self.b_PL_phase_corr[i] = self.PL.dphi
            self.b_PL_omegaRF_corr[i] = self.PL.domega_rf
            self.b_SL_dphiRF[i] = self.rf_params.dphi_rf[0]
            self.b_RL_drho[i] = self.PL.drho
                         
        if self.LHCNoiseFB:
             
            self.b_LHCnoiseFB_factor[i] = self.LHCNoiseFB.x
            self.b_LHCnoiseFB_bl[i] = self.LHCNoiseFB.bl_meas
            if self.LHCNoiseFB.bl_meas_bbb != None:
                self.b_LHCnoiseFB_bl_bbb[i,:] = self.LHCNoiseFB.bl_meas_bbb[:]


    def write_data(self, h5group, dims):
        
        i1 = self.i_turn - self.buffer_time
        i2 = self.i_turn

        h5group.require_dataset("n_macroparticles_alive", shape = dims, 
                                dtype = 'f')
        h5group["n_macroparticles_alive"][i1:i2] = self.b_np_alive[:]

        h5group.require_dataset("mean_dt", shape = dims, dtype = 'f')
        h5group["mean_dt"][i1:i2]   = self.b_mean_dt[:]
         
        h5group.require_dataset("mean_dE", shape = dims, dtype = 'f')
        h5group["mean_dE"][i1:i2]  = self.b_mean_dE[:]
         
        h5group.require_dataset("sigma_dt", shape = dims, dtype = 'f')
        h5group["sigma_dt"][i1:i2]  = self.b_sigma_dt[:] 
         
        h5group.require_dataset("sigma_dE", shape = dims, dtype = 'f')
        h5group["sigma_dE"][i1:i2] = self.b_sigma_dE[:]
         
        h5group.require_dataset("epsn_rms_l", shape = dims, dtype = 'f')
        h5group["epsn_rms_l"][i1:i2]   = self.b_epsn_rms[:]
         
        if self.fit_option == True:
             
                h5group.require_dataset("bunch_length", shape = dims, 
                                        dtype = 'f')
                h5group["bunch_length"][i1:i2] = self.b_bl[:]
             
        if self.PL:
 
            h5group.require_dataset("PL_omegaRF", shape = dims, 
                                    dtype = np.float64)
            h5group["PL_omegaRF"][i1:i2] = self.b_PL_omegaRF[:]

            h5group.require_dataset("PL_phiRF", shape = dims, 
                                    dtype = 'f')
            h5group["PL_phiRF"][i1:i2] = self.b_PL_phiRF[:]

            h5group.require_dataset("PL_bunch_phase", shape = dims, 
                                    dtype = 'f')
            h5group["PL_bunch_phase"][i1:i2] = self.b_PL_bunch_phase[:]
             
            h5group.require_dataset("PL_phase_corr", shape = dims, 
                                    dtype = 'f')
            h5group["PL_phase_corr"][i1:i2] = self.b_PL_phase_corr[:]
             
            h5group.require_dataset("PL_omegaRF_corr", shape = dims, 
                                    dtype = 'f')
            h5group["PL_omegaRF_corr"][i1:i2] = self.b_PL_omegaRF_corr[:]

            h5group.require_dataset("SL_dphiRF", shape = dims, 
                                    dtype = 'f')
            h5group["SL_dphiRF"][i1:i2] = self.b_SL_dphiRF[:]

            h5group.require_dataset("RL_drho", shape = dims, 
                                    dtype = 'f')
            h5group["RL_drho"][i1:i2] = self.b_RL_drho[:]
             
        if self.LHCNoiseFB:
             
            h5group.require_dataset("LHC_noise_FB_factor", shape = dims, 
                                    dtype = 'f')
            h5group["LHC_noise_FB_factor"][i1:i2] = self.b_LHCnoiseFB_factor[:]
             
            h5group.require_dataset("LHC_noise_FB_bl", shape = dims, 
                                    dtype = 'f')        
            h5group["LHC_noise_FB_bl"][i1:i2] = self.b_LHCnoiseFB_bl[:]

            if self.LHCNoiseFB.bl_meas_bbb != None:
                h5group.require_dataset("LHC_noise_FB_bl_bbb", shape = (self.n_turns + 1, 
                                        len(self.LHCNoiseFB.bl_meas_bbb)), 
                                        dtype = 'f')        
                h5group["LHC_noise_FB_bl_bbb"][i1:i2,:] = self.b_LHCnoiseFB_bl_bbb[:,:]


    def open(self):
        self.h5file = hp.File(self.filename + '.h5', 'r+')
        self.h5file.require_group('Beam')
        

    def close(self):
        self.h5file.close()



class SlicesMonitor(object):

    ''' Class able to save the bunch profile, i.e. the histogram derived from
        the slicing.
    '''
    
    def __init__(self, filename, n_turns, profile):
        
        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_turns = n_turns
        self.i_turn = 0
        self.profile = profile
        self.h5file.create_group('Slices')

    
    def track(self, bunch):
        
        if not self.i_turn:
            self.create_data(self.h5file['Slices'], (self.profile.n_slices, 
                                                     self.n_turns))
            self.write_data(self.profile, self.h5file['Slices'], self.i_turn)
        else:
            self.write_data(self.profile, self.h5file['Slices'], self.i_turn)

        self.i_turn += 1

    
    def create_data(self, h5group, dims):
        
        h5group.create_dataset("n_macroparticles", dims, compression="gzip", 
                               compression_opts=9)
        
        
    def write_data(self, bunch, h5group, i_turn):
        
        h5group["n_macroparticles"][:, i_turn] = self.profile.n_macroparticles
        
    def close(self):
        self.h5file.close()
