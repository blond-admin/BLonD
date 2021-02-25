#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import gc
import os
import h5py as hp
import numpy as np
#import pathlib

# My custom functions for BLonD
from blond.beam.beam_tools import BeamTools
from blond.beam.profile_tools import ProfileTools

###############################################################################

beam_stats_keys        = ['mean_dt', 'mean_dE', 'sigma_dt', 'sigma_dE', 'epsnrmsl', 'mean_dtOff_ctr', 'mean_dtOff_fit']
profile_stats_FWHMRMS_key_list = ['bunchPosition', 'bunchPositionOff_ctr', 'bunchPositionOff_fit', 'bunchLength', 'bunchEnergySpread', 'bunchEmittance']
profile_stats_BRF_key_list     = ['bunchPositionOff', 'bunchFormFactor']

class MonitorFull(object):
    '''
        Based on BLonD's BunchMonitor
    '''

    def __init__(self,
                 outdir,
                 ring_obj, rfstation_obj,
                 beam_obj, profile_obj,  # Re-arranged arguments, and profile_obj no longer optional (why would it be, it's the most important!)
                 beampattern_obj,
                 profilepattern_obj,
                 Nt_mn1, Nt_mn2, Nt_sav,
                 totalinducedvoltage_obj=None,
                 cavityfeedback_obj=None,
                 beamfeedback_obj=None,
                 lhcnoisefb_obj=None): # totalinducedvoltage_obj=None, beamfeedback_obj=None, lhcnoise_obj=None):

        self.filename = f'{outdir}/monitor_full'

        # Monitored every Nt_mn1 turns
        self.beam      = beam_obj
        self.profile   = profile_obj

        # Monitored every Nt_mn2 turns
        self.rfstation = rfstation_obj
        self.beampattern = beampattern_obj
        self.totalinducedvoltage = totalinducedvoltage_obj
        self.cavityfeedback = cavityfeedback_obj
        self.beamfeedback = beamfeedback_obj
        self.lhcnoisefb = lhcnoisefb_obj

        self.nbf = self.beampattern.nbf # len(self.idxP['BF']['bunch']) # = beampattern_obj.bucket_centres_0
        self.nbs = self.beampattern.nbs

        self.idxB = beampattern_obj.idxB
        self.idxP = profilepattern_obj.idxP

        # Inital no. of macroparticles per bunch/beam from Profile (not needed a similar function from Beam, as the initial no. is known from the array sizes and indices corresponding to each bunch):
        self.profile_nmacrop_bunch_0 = ProfileTools.nmacrop_bunch(self.profile, self.idxP['BF']['bunch'], self.idxP['BS']['bunch'], self.idxP['BG']['bunch'])
        self.profile_nmacrop_beam_0  = ProfileTools.nmacrop_beam(self.profile_nmacrop_bunch_0) #self.profile, self.idxP['BF']['beam'], self.idxP['BS']['beam'], self.idxP['BG']['beam'])
        # print(f'profile_nmacrop_bunch_0 = {self.profile_nmacrop_bunch_0}')
        # print(f'profile_nmacrop_beam_0 = {self.profile_nmacrop_beam_0}')

        # Saving and monitoring intervals
        self.Nt_sav = Nt_sav
        self.Nt_mn1 = Nt_mn1  # Interval (in no. of turns) between computing Beam/Profile stats and saving in buffer: e.g. every 2 turns
        self.Nt_mn2 = Nt_mn2  # Interval (in no. of turns) between reading RFStation/TotalInducedVoltage/CavityFeedback/BeamFeedback/LHCNoiseFB attributes and saving in buffer: e.g. every 1 turn

        # Total no. of turns to be monitored and saved, and no. of turns in before before saving (Note that self.rfstation.n_turns is already 1 turn lower than Nt_trk):

        # Total no. of turns to be monitored and saved from Beam and Profile
        self.n_turns_tot_mn1 = (self.rfstation.n_turns-1) // self.Nt_mn1
        if((self.rfstation.n_turns-1) % self.Nt_mn1 > 0): self.n_turns_tot_mn1 += 2 # Add space for 2 turns: turn 0 and the last turn
        else:                                             self.n_turns_tot_mn1 += 1 # Add space for 1 turn: turn 0 (the last turn is already included)

        # Total no. of turns to be monitored and saved from RFStation/TotalInducedVoltage/CavityFeedback/BeamFeedback/LHCNoiseFB
        self.n_turns_tot_mn2 = (self.rfstation.n_turns-1) // self.Nt_mn2
        if((self.rfstation.n_turns-1) % self.Nt_mn2 > 0): self.n_turns_tot_mn2 += 2 # Add space for 2 turns: turn 0 and the last turn
        else:                                             self.n_turns_tot_mn2 += 1 # Add space for 1 turn: turn 0 (the last turn is already included)

        # Total no. of turns to be stored in buffer before saving:
        self.n_turns_buff_mn1 = self.Nt_sav // self.Nt_mn1 # Size (no. of turns) of buffer for Beam/Profile
        self.n_turns_buff_mn2 = self.Nt_sav // self.Nt_mn2 # Size (no. of turns) of buffer for RFStation/TotalInducedVoltage/CavityFeedback/BeamFeedback/LHCNoiseFB

        # "Current" turn
        self.i1 = -1 # Current turn in buffer: -1 indicated initialisation (i.e. before tracking)
        self.i2 = -1 # Current turn in buffer: -1 indicated initialisation (i.e. before tracking)

        # Shape of full arrays, and buffered arrays:

        self.shape_tot_mn1_plus1           = (self.n_turns_tot_mn1 + 1,)
        self.shape_tot_mn1_plus1_x_nbf     = (self.n_turns_tot_mn1 + 1, self.nbf)

        self.shape_tot_mn2_plus1           = (self.n_turns_tot_mn2 + 1,)
        self.shape_tot_mn2_plus1_x_nbf     = (self.n_turns_tot_mn2 + 1, self.nbf)
        self.shape_tot_mn2_plus1_x_nrf     = (self.n_turns_tot_mn2 + 1, self.rfstation.n_rf)
        self.shape_tot_mn2_plus1_x_nslices = (self.n_turns_tot_mn2 + 1, self.profile.n_slices)

        self.shape_buff_mn1           = (self.n_turns_buff_mn1,)              # For beam data
        self.shape_buff_mn1_x_nbf     = (self.n_turns_buff_mn1, self.nbf)     # For bunch data

        self.shape_buff_mn2_x_nbf     = (self.n_turns_buff_mn2, self.nbf)
        self.shape_buff_mn2           = (self.n_turns_buff_mn2,)
        self.shape_buff_mn2_x_nrf     = (self.n_turns_buff_mn2, self.rfstation.n_rf)
        self.shape_buff_mn2_x_nslices = (self.n_turns_buff_mn2, self.profile.n_slices)

        # Initialise data and save initial state:

        self.init_data() # Create groups and datasets, and initilize buffers for RFStation/TotalInducedVoltage/CavityFeedback/BeamFeedback/LHCNoiseFB

        # No need to track now. The original stated before tracking has already been saved as turn '-1' in the previous command. Turn 0 in tracking will call .track on its own

    def init_data(self):

        # Check and delete if file exists
        if os.path.exists(self.filename + '.h5'):
          os.remove(self.filename + '.h5')
        else:
          pass

        # Create file
        with hp.File(self.filename + '.h5', 'w') as self.h5file:

            bucket_centres_0 = self.beampattern.bucket_centres_0
            maxbktl_0        = self.beampattern.maxbktl_0
            marginoffset_0   = self.beampattern.marginoffset_0

            # BEAM

            # Macroparticles and stats
            #print(f'beam.id (in monitor init_data) = {self.beam.id}, shape = {self.beam.id.shape}')
           ##print(f'beam.dE (in monitor init_data) = {self.beam.dE}, shape = {self.beam.dE.shape}')
            #print(f'beam.dt (in monitor init_data) = {self.beam.dt}, shape = {self.beam.dt.shape}')
            beam_nmacrop_bunch_0 = BeamTools.nmacrop_bunch(self.beam, self.idxB)
            beam_nmacrop_beam_0  = BeamTools.nmacrop_beam(beam_nmacrop_bunch_0) # self.beam)
            beam_stats_bunch_0 = BeamTools.stats_bunch(self.beam, self.idxB, bucket_centres_0)
            beam_stats_beam_0  = BeamTools.stats_beam(beam_stats_bunch_0)
            #print(f'beam_nmacrop_bunch_0[alive] = {beam_nmacrop_bunch_0["alive"]}')
            #print(f'beam_stats_bunch_0[mean_dt] = {beam_stats_bunch_0["mean_dt"]}')

            # Create groups
            self.h5file.require_group('Beam')
            for bunchbeam in ['bunch', 'beam']:
                self.h5file.require_group(f'Beam/{bunchbeam}')
            h5group = self.h5file['Beam']

            # Create datasets
            h5group.create_dataset('turns', shape=self.shape_tot_mn1_plus1, dtype='i')
            h5group['turns'][0] = self.i1

            # Write first data points

            for key in ['alive', 'lost']:
                h5group.create_dataset(f'bunch/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='i') #, compression="gzip", compression_opts=9)
                h5group.create_dataset(f'beam/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='i') #, compression="gzip", compression_opts=9)
                h5group[f'bunch/{key}'][0] = beam_nmacrop_bunch_0[key]
                h5group[f'beam/{key}' ][0] = beam_nmacrop_beam_0[ key]
                self.h5file.flush()

            for key in beam_stats_keys:
                h5group.create_dataset(f'bunch/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='f') #, compression="gzip", compression_opts=9)
                h5group.create_dataset(f'beam/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='f') #, compression="gzip", compression_opts=9)
                h5group[f'bunch/{key}'][0] = beam_stats_bunch_0[key]
                h5group[f'beam/{key}' ][0] = beam_stats_beam_0[ key]
                self.h5file.flush()

            gc.collect()

            # PROFILE

            # Macroparticles and stats
            #print(f'profile.n_macroparticles (in monitor init_data) = {self.profile.n_macroparticles}, shape = {self.profile.n_macroparticles.shape}')
            profile_nmacrop_bunch_0 = ProfileTools.nmacrop_bunch(self.profile, self.idxP['BF']['bunch'], self.idxP['BS']['bunch'], self.idxP['BG']['bunch'], initial=self.profile_nmacrop_bunch_0)
            profile_nmacrop_beam_0  = ProfileTools.nmacrop_beam(profile_nmacrop_bunch_0) # self.profile, self.idxP['BF']['beam'],  self.idxP['BS']['beam'],  self.idxP['BG']['beam'],, initial=self.profile_nmacrop_beam_0)
            profile_stats_bunch_0 = ProfileTools.stats_bunch(self.profile, self.idxP['BF']['bunch'], bucket_centres_0, self.nbs, maxbktl_0, marginoffset_0, self.rfstation)
            profile_stats_beam_0  = ProfileTools.stats_beam(profile_stats_bunch_0)
            #print(f'profile_nmacrop_bunch_0[BF][alive] = {profile_nmacrop_bunch_0["BF"]["alive"]}')
            #print(f'profile_stats_bunch_0[FWHM][bunchPosition] = {profile_stats_bunch_0["FWHM"]["bunchPosition"]}')
            #print('')

            # Create groups
            self.h5file.require_group('Profile')
            for bunchbeam in ['bunch', 'beam']:
                for opt in ['BF', 'BS', 'BG', 'FWHM', 'RMS']:
                    self.h5file.require_group(f'Profile/{bunchbeam}/{opt}')
            h5group = self.h5file['Profile']

            # Create datasets
            h5group.create_dataset('turns', shape=self.shape_tot_mn1_plus1, dtype='i')
            h5group['turns'][0] = self.i1

            # Write first data points

            for opt in ['BF', 'BS', 'BG']:
                for key in ['alive', 'lost']:
                    h5group.create_dataset(f'bunch/{opt}/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='i') #, compression="gzip", compression_opts=9)
                    h5group.create_dataset(f'beam/{opt}/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='i') #, compression="gzip", compression_opts=9)
                    #print(opt, key, self_shape_tot_mn1_plus1_x_nbf, profile_nmacrop_bunch_0[opt][key], self.shape_tot_mn1_plus1, profile_nmacrop_beam_0[ opt][key])
                    h5group[f'bunch/{opt}/{key}'][0] = profile_nmacrop_bunch_0[opt][key]
                    h5group[f'beam/{opt}/{key}'][0]  = profile_nmacrop_beam_0[ opt][key]
                    self.h5file.flush()

            for opt in ['FWHM', 'RMS', 'BRF']:
                if opt == 'BRF': profile_stats_opt_keys = profile_stats_BRF_key_list[:]
                else:            profile_stats_opt_keys = profile_stats_FWHMRMS_key_list[:]
                for key in profile_stats_opt_keys:
                    h5group.create_dataset(f'bunch/{opt}/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='f') #, compression="gzip", compression_opts=9)
                    h5group.create_dataset(f'beam/{opt}/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='f') #, compression="gzip", compression_opts=9)
                    h5group[f'bunch/{opt}/{key}'][0] = profile_stats_bunch_0[opt][key]
                    h5group[f'beam/{opt}/{key}'][0]  = profile_stats_beam_0[ opt][key]
                    self.h5file.flush()

            gc.collect()

            # OTHERS

            h5group = self.h5file['/']

            h5group.create_dataset('turns', shape=self.shape_tot_mn2_plus1, dtype='i')
            h5group['turns'][0] = self.i2

            # # RFSTATION -- RF paramters are saved in the BeamFeedback group, as they only change (and thus are of interest) when feedbacks are present

            # self.h5file.require_group('RFStation')
            # h5group = self.h5file['RFStation']

            # h5group.create_dataset("omega_rf", shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') #, compression="gzip", compression_opts=9)
            # h5group.create_dataset("dphi_rf",  shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') #, compression="gzip", compression_opts=9)
            # h5group.create_dataset("phi_rf",   shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') #, compression="gzip", compression_opts=9)
            # h5group["omega_rf"][0] = self.rfstation.omega_rf[:,0]   # <-> "PL_omegaRF" (old) -> 'PL/omega_rf'
            # h5group["dphi_rf"][0]  = self.rfstation.dphi_rf         # <-> "SL_dphiRF" (old)  -> 'SL/phi_rf'
            # h5group["phi_rf"][0]   = self.rfstation.phi_rf[:,0]     # <-> "PL_phiRF" (old)   -> 'PL/phi_rf'

            # TOTALINDUCEDVOLTAGE

            if self.totalinducedvoltage is not None:

                # Create groups
                self.h5file.require_group('TotalInducedVoltage')
                h5group = self.h5file['TotalInducedVoltage']

                # Create datasets
                # h5group.create_dataset('induced_voltage_minabs', shape=self.shape_tot_mn2_plus1, dtype='f') #, compression="gzip", compression_opts=9)
                # h5group.create_dataset('induced_voltage_aveabs', shape=self.shape_tot_mn2_plus1, dtype='f') #, compression="gzip", compression_opts=9)
                h5group.create_dataset('induced_voltage_maxabs', shape=self.shape_tot_mn2_plus1, dtype='f') #, compression="gzip", compression_opts=9)
                # h5group['induced_voltage_minabs'][0] =     min(np.abs(self.totalinducedvoltage.induced_voltage))
                # h5group['induced_voltage_aveabs'][0] = np.mean(np.abs(self.totalinducedvoltage.induced_voltage))
                h5group['induced_voltage_maxabs'][0] =     max(np.abs(self.totalinducedvoltage.induced_voltage))
                self.h5file.flush()

            # CAVITYFEEDBACK

            if self.cavityfeedback is not None:

                # Create groups
                self.h5file.require_group('CavityFeedback')
                h5group = self.h5file['CavityFeedback']

                # Create datasets
                for key in ['V_sum', 'V_corr', 'phi_corr']:
                    if(key == 'V_sum'): dtypekey = np.dtype('complex64') # instead of the default 'complex128')
                    else:               dtypekey = 'f'
                    h5group.create_dataset(key, shape=self.shape_tot_mn2_plus1_x_nslices, dtype=dtypekey) #, compression="gzip", compression_opts=9)
                    h5group[key][0] = getattr(self.cavityfeedback, key)
                    self.h5file.flush()

            # BEAMFEEDBACK

            if self.beamfeedback is not None:

                # Create groups
                self.h5file.require_group('BeamFeedback')
                for PRSL in ['PL', 'RL', 'SL']:
                    self.h5file.require_group(f'BeamFeedback/{PRSL}')

                # Create datasets: Phase loop
                h5group.create_dataset('PL/omega_rf',  shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') #, compression="gzip", compression_opts=9) # CHECK: dtype=np.float64 ??
                h5group.create_dataset('PL/phi_rf',    shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') #, compression="gzip", compression_opts=9)
                h5group.create_dataset('PL/phi_beam',  shape=self.shape_tot_mn2_plus1,    dtype='f') #, compression="gzip", compression_opts=9)
                h5group.create_dataset('PL/dphi',      shape=self.shape_tot_mn2_plus1,    dtype='f') #, compression="gzip", compression_opts=9)
                h5group.create_dataset('PL/domega_rf', shape=self.shape_tot_mn2_plus1,    dtype='f') #, compression="gzip", compression_opts=9)
                h5group['PL/omega_rf'][0]  = self.rfstation.omega_rf[:, 0]        # "PL_omegaRF" (old)
                h5group['PL/phi_rf'][0]    = self.rfstation.phi_rf[:, 0]          # "PL_phiRF" (old)
                h5group['PL/phi_beam'][0]  = self.beamfeedback.phi_beam           # "PL_bunch_phase" (old)
                h5group['PL/dphi'][0]      = self.beamfeedback.dphi               # "PL_phase_corr" (old)
                h5group['PL/domega_rf'][0] = self.beamfeedback.domega_rf          # "PL_omegaRF_corr" (old)
                # Create datasets: Synchro loop
                h5group.create_dataset('SL/dphi_rf', shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') #, compression="gzip", compression_opts=9)
                h5group['SL/dphi_rf'][0] = self.rfstation.dphi_rf                 # "SL_dphiRF" (old)
                # Create datasets: Radial loop
                h5group.create_dataset('RL/drho', shape=self.shape_tot_mn2_plus1, dtype='f') #, compression="gzip", compression_opts=9)
                h5group['RL/drho'][0] = self.beamfeedback.drho                    # "RL_drho" (old)
                self.h5file.flush()

            # LHCNOISEFB

            if self.lhcnoisefb is not None:

                # Create groups
                self.h5file.require_group('LHCNoiseFB')
                h5group = self.h5file['LHCNoiseFB']

                # Create datasets
                h5group.create_dataset('x',       shape=self.shape_tot_mn2_plus1, dtype='f') #, compression="gzip", compression_opts=9)
                h5group.create_dataset('bl_meas', shape=self.shape_tot_mn2_plus1, dtype='f') #, compression="gzip", compression_opts=9)
                h5group['x'][0]       = self.lhcnoisefb.x                          # "LHC_noise_FB_factor" (old)
                h5group['bl_meas'][0] = self.lhcnoisefb.bl_meas                    # "LHC_noise_FB_bl" (old)
                if self.lhcnoisefb.bl_meas_bbb is not None:
                    h5group.create_dataset("bl_meas_bbb", shape=self.shape_tot_mn2_plus1_x_nbf, dtype='f') #, compression="gzip", compression_opts=9)
                    h5group["bl_meas_bbb"][0,:] = self.lhcnoisefb.bl_meas_bbb[:]   # "LHC_noise_FB_bl_bbb"
                self.h5file.flush()

            #

            gc.collect()

        # Close file
        self.close()

        print(f'* MONITOR FULL SIZE: {os.path.getsize(self.filename + ".h5")/1024./1024.:.3f} MB')

        # Initialise buffer for next turn
        self.init_buffer()

    def close(self):
        try:    self.h5file.close()
        except: pass

    def init_buffer(self):

        # BEAM

        setattr(self, 'buffer_beam_turns', np.zeros(self.shape_buff_mn1, dtype=int) )

        for key in ['alive', 'lost']:
            setattr(self, f'buffer_beam_bunch_{key}', np.zeros(self.shape_buff_mn1_x_nbf, dtype=int))
            setattr(self, f'buffer_beam_beam_{key}',  np.zeros(self.shape_buff_mn1, dtype=int) )

        for key in beam_stats_keys:
            setattr(self, f'buffer_beam_bunch_{key}', np.zeros(self.shape_buff_mn1_x_nbf, dtype='f'))
            setattr(self, f'buffer_beam_beam_{key}',  np.zeros(self.shape_buff_mn1, dtype='f') )

        # PROFILE

        setattr(self, 'buffer_profile_turns', np.zeros(self.shape_buff_mn1, dtype=int) )

        for opt in ['BF', 'BS', 'BG']:
            for key in ['alive', 'lost']:
                setattr(self, f'buffer_profile_bunch_{opt}_{key}', np.zeros(self.shape_buff_mn1_x_nbf, dtype=int))
                setattr(self, f'buffer_profile_beam_{opt}_{key}',  np.zeros(self.shape_buff_mn1, dtype=int) )

        for opt in ['FWHM', 'RMS', 'BRF']:
            if opt == 'BRF': profile_stats_opt_keys = profile_stats_BRF_key_list[:]
            else:            profile_stats_opt_keys = profile_stats_FWHMRMS_key_list[:]
            for key in profile_stats_opt_keys:
                setattr(self, f'buffer_profile_bunch_{opt}_{key}', np.zeros(self.shape_buff_mn1_x_nbf, dtype='f'))
                setattr(self, f'buffer_profile_beam_{opt}_{key}',  np.zeros(self.shape_buff_mn1, dtype='f') )

        # OTHERS

        self.buffer_turns = np.zeros(self.shape_buff_mn2, dtype=int)

        # # RFSTATION

        # if self.rfstation:

        #     self.buffer_rfstation_omega_rf = np.zeros(self.shape_buff_mn2_x_nrf)
        #     self.buffer_rfstation_dphi_rf  = np.zeros(self.shape_buff_mn2_x_nrf)
        #     self.buffer_rfstation_phi_rf   = np.zeros(self.shape_buff_mn2_x_nrf)

        # TOTALINDUCEDVOLTAGE

        if self.totalinducedvoltage is not None:

            # self.buffer_totalinducedvoltage_induced_voltage_minabs = np.zeros(self.shape_buff_mn2)
            # self.buffer_totalinducedvoltage_induced_voltage_aveabs = np.zeros(self.shape_buff_mn2)
            self.buffer_totalinducedvoltage_induced_voltage_maxabs = np.zeros(self.shape_buff_mn2, dtype='f')

        # CAVITYFEEDBACK

        if self.cavityfeedback is not None:

            for key in ['V_sum', 'V_corr', 'phi_corr']:
                if(key == 'V_sum'): dtypekey = np.dtype('complex64') # instead of the default 'complex128')
                else:               dtypekey = 'f'
                setattr(self, f'buffer_cavityfeedback_{key}', np.zeros(self.shape_buff_mn2_x_nslices, dtype=dtypekey))

        # BEAMFEEDBACK

        if self.beamfeedback is not None:

            # Phase loop
            self.buffer_beamfeedback_PL_omega_rf  = np.zeros(self.shape_buff_mn2_x_nrf, dtype='f')
            self.buffer_beamfeedback_PL_phi_rf    = np.zeros(self.shape_buff_mn2_x_nrf, dtype='f')
            self.buffer_beamfeedback_PL_phi_beam  = np.zeros(self.shape_buff_mn2, dtype='f')
            self.buffer_beamfeedback_PL_dphi      = np.zeros(self.shape_buff_mn2, dtype='f')
            self.buffer_beamfeedback_PL_domega_rf = np.zeros(self.shape_buff_mn2, dtype='f')
            # Synchro loop
            self.buffer_beamfeedback_SL_dphi_rf = np.zeros(self.shape_buff_mn2_x_nrf, dtype='f')
            # Radial loop
            self.buffer_beamfeedback_RL_drho = np.zeros(self.shape_buff_mn2, dtype='f')

        # LHCNOISEFB

        if self.lhcnoisefb is not None:

            self.buffer_lhcnoisefb_x       = np.zeros(self.shape_buff_mn2, dtype='f')
            self.buffer_lhcnoisefb_bl_meas = np.zeros(self.shape_buff_mn2, dtype='f')
            if self.lhcnoisefb.bl_meas_bbb is not None:
                self.buffer_lhcnoisefb_bl_meas_bbb = np.zeros(self.shape_buff_mn2_x_nbf, dtype='f')

    def track(self):
        ''' To be called after trackers.track() '''

        # Compute Beam and Profile stats every mn1 turns
        if(((self.rfstation.counter[-1]-1) % self.Nt_mn1 == 0) or ((self.rfstation.counter[-1]-1) == self.rfstation.n_turns-1)):
            #print('Tracking monitorfull...')

            bucket_centres_i  = self.beampattern.bucket_centres
            maxbktl_i         = self.beampattern.maxbktl
            marginoffset_i    = self.beampattern.marginoffset
            #
            #print(f'beam.id (in monitor track) = {self.beam.id}, shape = {self.beam.id.shape}')
           ##print(f'beam.dE (in monitor track) = {self.beam.dE}, shape = {self.beam.dE.shape}')
            #print(f'beam.dt (in monitor track) = {self.beam.dt}, shape = {self.beam.dt.shape}')
            beam_nmacrop_bunch_i = BeamTools.nmacrop_bunch(self.beam, self.idxB)
            beam_nmacrop_beam_i  = BeamTools.nmacrop_beam(beam_nmacrop_bunch_i) # self.beam)
            beam_stats_bunch_i = BeamTools.stats_bunch(self.beam, self.idxB, bucket_centres_i)
            beam_stats_beam_i  = BeamTools.stats_beam(beam_stats_bunch_i)
            #print(f'beam_nmacrop_bunch_i[alive] = {beam_nmacrop_bunch_i["alive"]}')
            #print(f'beam_stats_bunch_i[mean_dt] = {beam_stats_bunch_i["mean_dt"]}')
            #
            #print(f'profile.n_macroparticles (in monitor track) = {self.profile.n_macroparticles}, shape = {self.profile.n_macroparticles.shape}')
            profile_nmacrop_bunch_i = ProfileTools.nmacrop_bunch(self.profile, self.idxP['BF']['bunch'], self.idxP['BS']['bunch'], self.idxP['BG']['bunch'], initial=self.profile_nmacrop_bunch_0)
            profile_nmacrop_beam_i  = ProfileTools.nmacrop_beam(profile_nmacrop_bunch_i) # self.profile, self.idxP['BF']['beam'],  self.idxP['BS']['beam'],  self.idxP['BG']['beam'], initial=self.profile_nmacrop_beam_0)
            profile_stats_bunch_i = ProfileTools.stats_bunch(self.profile, self.idxP['BF']['bunch'], bucket_centres_i, self.nbs, maxbktl_i, marginoffset_i, self.rfstation) # TODO: check if it shouldn't be static
            profile_stats_beam_i  = ProfileTools.stats_beam(profile_stats_bunch_i)
            #print(f'profile_nmacrop_bunch_i[BF][alive] = {profile_nmacrop_bunch_i["BF"]["alive"]}')
            #print(f'profile_stats_bunch_i[FWHM][bunchPosition] = {profile_stats_bunch_i["FWHM"]["bunchPosition"]}')
            #print('')
            #
            self.write_buffer_1(beam_nmacrop_bunch_i,
                                beam_nmacrop_beam_i,
                                beam_stats_bunch_i,
                                beam_stats_beam_i,
                                profile_nmacrop_bunch_i,
                                profile_nmacrop_beam_i,
                                profile_stats_bunch_i,
                                profile_stats_beam_i)
            self.i1 += 1
            gc.collect()

        # Read parameters from the other objects (they have already been computed internally in the corresponding track calls) every mn2 turns
        if(((self.rfstation.counter[-1]-1) % self.Nt_mn2 == 0) or ((self.rfstation.counter[-1]-1) == self.rfstation.n_turns-1)):
            #print(f'track for BeamFeedBack and lhcnoise_obj: {self.rfstation.counter[-1]-1}, i2 = {self.i1}')
            self.write_buffer_2()
            self.i2 += 1

        ###

        # Save the data every Nt_sav
        if(((self.rfstation.counter[-1]-1) % self.Nt_sav == 0) or ((self.rfstation.counter[-1]-1) == self.rfstation.n_turns-1)):
            #self.open()
            self.write_data()
            #self.close()
            #print(f'* MONITOR FULL SIZE: {os.path.getsize(self.filename + ".h5")/1024./1024.:.3f} MB')
            self.init_buffer() # Restart the buffer

    def write_buffer_1(self, beam_nmacrop_bunch_i,
                             beam_nmacrop_beam_i,
                             beam_stats_bunch_i,
                             beam_stats_beam_i,
                             profile_nmacrop_bunch_i,
                             profile_nmacrop_beam_i,
                             profile_stats_bunch_i,
                             profile_stats_beam_i):
        ''' Write Beam and Profile buffers '''

        # Place in the buffer (turn no.)
        if(self.i1 == -1): j = 0 # Turn 0
        else:              j = self.i1 % self.n_turns_buff_mn1 # = current turn % size of buffer (no. of turns)
        #print(f'Writing buffer: j1 = {j}')

        # BEAM

        getattr(self, 'buffer_beam_turns')[j] = int(self.rfstation.counter[-1]-1)
        #print(f'self.buffer_beam_turns = {self.buffer_beam_turns}')

        for key in ['alive', 'lost']:
            getattr(self, f'buffer_beam_bunch_{key}')[j] = beam_nmacrop_bunch_i[key]
            getattr(self, f'buffer_beam_beam_{key}' )[j] = beam_nmacrop_beam_i[ key]

        for key in beam_stats_keys:
            getattr(self, f'buffer_beam_bunch_{key}')[j] = beam_stats_bunch_i[key]
            getattr(self, f'buffer_beam_beam_{key}' )[j] = beam_stats_beam_i[ key]

        # PROFILE

        getattr(self, 'buffer_profile_turns')[j] = int(self.rfstation.counter[-1]-1)

        for opt in ['BF', 'BS', 'BG']:
            for key in ['alive', 'lost']:
                getattr(self, f'buffer_profile_bunch_{opt}_{key}')[j] = profile_nmacrop_bunch_i[opt][key]
                getattr(self, f'buffer_profile_beam_{opt}_{key}' )[j] = profile_nmacrop_beam_i[ opt][key]

        for opt in ['FWHM', 'RMS', 'BRF']:
            if opt == 'BRF': profile_stats_opt_keys = profile_stats_BRF_key_list[:]
            else:            profile_stats_opt_keys = profile_stats_FWHMRMS_key_list[:]
            for key in profile_stats_opt_keys:
                getattr(self, f'buffer_profile_bunch_{opt}_{key}')[j] = profile_stats_bunch_i[opt][key]
                getattr(self, f'buffer_profile_beam_{opt}_{key}')[j]  = profile_stats_beam_i[ opt][key]

    def write_buffer_2(self):
        ''' Write other buffers (RFStation/TotalInducedVoltage/CavityFeedback/BeamFeedback/LHCNoiseFB) '''

        # Current turn in tracking
        i = self.rfstation.counter[-1]-1 # instead of self.i_turn_sav

        # Place in the buffer (turn no.)
        if(self.i2 == -1): j = 0 # Turn 0
        else:              j = self.i2 % self.n_turns_buff_mn2 # = current turn % size of buffer (no. of turns)
        #print(f'Writing buffer: j2 = {j}')

        self.buffer_turns[j] = int(self.rfstation.counter[-1]-1)
        #print(f'self.buffer_turns = {self.buffer_turns}')

        # RFSTATION

        # if self.rfstation:

        #     self.buffer_rfstation_omega_rf[j] = self.rfstation.omega_rf[:, i]
        #     self.buffer_rfstation_dphi_rf[j]  = self.rfstation.dphi_rf
        #     self.buffer_rfstation_phi_rf[j]   = self.rfstation.phi_rf[:, i]

        # TOTALINDUCEDVOLTAGE

        if self.totalinducedvoltage is not None:

            # self.buffer_totalinducedvoltage_induced_voltage_minabs[j] =     min(np.abs(self.totalinducedvoltage.induced_voltage))
            # self.buffer_totalinducedvoltage_induced_voltage_aveabs[j] = np.mean(np.abs(self.totalinducedvoltage.induced_voltage))
            self.buffer_totalinducedvoltage_induced_voltage_maxabs[j] =     max(np.abs(self.totalinducedvoltage.induced_voltage))

        # CAVITYFEEDBACK

        if self.cavityfeedback is not None:
            for key in ['V_sum', 'V_corr', 'phi_corr']:
                getattr(self, f'buffer_cavityfeedback_{key}')[j] = getattr(self.cavityfeedback, key)

        # BEAMFEEDBACK

        if self.beamfeedback is not None:

            # Phase loop
            self.buffer_beamfeedback_PL_omega_rf[j]  = self.rfstation.omega_rf[:, i]
            self.buffer_beamfeedback_PL_phi_rf[j]    = self.rfstation.phi_rf[:, i]
            self.buffer_beamfeedback_PL_phi_beam[j]  = self.beamfeedback.phi_beam
            self.buffer_beamfeedback_PL_dphi[j]      = self.beamfeedback.dphi
            self.buffer_beamfeedback_PL_domega_rf[j] = self.beamfeedback.domega_rf
            # Synchro loop
            self.buffer_beamfeedback_SL_dphi_rf[j] = self.rfstation.dphi_rf
            # Radial loop
            self.buffer_beamfeedback_RL_drho[j] = self.beamfeedback.drho

        # LHCNOISEFB

        if self.lhcnoisefb is not None:

            self.buffer_beamfeedback_x[j]       = self.lhcnoisefb.x
            self.buffer_beamfeedback_bl_meas[j] = self.lhcnoisefb.bl_meas
            if self.lhcnoisefb.bl_meas_bbb is not None:
                self.buffer_beamfeedback_bl_meas_bbb[j, :] = self.lhcnoisefb.bl_meas_bbb[:]

    def open(self):
        self.h5file = hp.File(self.filename + '.h5', 'r+')

    def write_data(self):

        with hp.File(self.filename + '.h5', 'r+') as self.h5file:

            # print('')
            # print(f'Writting data..')
            # Places (turns) in buffer to be written into datasets:
            if(  self.rfstation.counter[-1]-1 == 0):
                #print('case A')
                j1f = 1 # so we only take the first element stored in the Beam/Profile buffers
                j2f = 1 # so we only take the first element stored in the RFStation/TotalInducedVoltage/CavityFeedback/BeamFeedback/LHCNoiseFB buffers
            elif(self.rfstation.counter[-1]-1 == self.rfstation.n_turns-1):
                #print('case C')
                j1f = (self.i1-1) % self.n_turns_buff_mn1 +1  # so we only take the elements with data (the rest are zero, as the buffer might contain more turns that the no. turn left in the tracking for the last buffer)
                j2f = (self.i2-1) % self.n_turns_buff_mn2 +1  # so we only take the elements with data (the rest are zero, as the buffer might contain more turns that the no. turn left in the tracking for the last buffer)
            else:
                #print('case B')
                j1f = self.n_turns_buff_mn1
                j2f = self.n_turns_buff_mn2
            # Places in datasets where to write buffer data (we need one extra +1 as the first entry is '-1')
            i1i = self.i1-j1f+1+1
            i2i = self.i2-j2f+1+1
            i1f = self.i1+1+1
            i2f = self.i2+1+1
            # print('')
            # print(f'datasets 1: [i1i:i1f] = [{i1i}:{i1f}]')
            # print(f'buffers 1:  [j1i:j1f] = [0:{j1f}]')
            # print('')
            # print(f'datasets 2: [i2i:i2f] = [{i2i}:{i2f}]')
            # print(f'buffers 2:  [j2i:j2f] = [0:{j2f}]')
            # print('')

            h5group = self.h5file['/']

            # BEAM

            h5group.require_dataset('Beam/turns', shape=self.shape_tot_mn1_plus1, dtype='i')
            h5group['Beam']['turns'][i1i:i1f] = getattr(self, 'buffer_beam_turns')[0:j1f]

            for key in ['alive', 'lost']:
                h5group.require_dataset(f'Beam/bunch/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='i')
                h5group.require_dataset(f'Beam/beam/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='i')
                h5group['Beam']['bunch'][key][i1i:i1f] = getattr(self, f'buffer_beam_bunch_{key}')[0:j1f]
                h5group['Beam']['beam' ][key][i1i:i1f] = getattr(self, f'buffer_beam_beam_{key}' )[0:j1f]
                self.h5file.flush()

            for key in beam_stats_keys:
                h5group.require_dataset(f'Beam/bunch/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='f')
                h5group.require_dataset(f'Beam/beam/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='f')
                h5group['Beam']['bunch'][key][i1i:i1f] = getattr(self, f'buffer_beam_bunch_{key}')[0:j1f]
                h5group['Beam']['beam' ][key][i1i:i1f] = getattr(self, f'buffer_beam_beam_{key}' )[0:j1f]
                self.h5file.flush()

            # PROFILE

            h5group.require_dataset('Profile/turns', shape=self.shape_tot_mn1_plus1, dtype='i')
            h5group['Profile']['turns'][i1i:i1f] = getattr(self, 'buffer_profile_turns')[0:j1f]

            for opt in ['BF', 'BS', 'BG']:
                for key in ['alive', 'lost']:
                    h5group.require_dataset(f'Profile/bunch/{opt}/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='i')
                    h5group.require_dataset(f'Profile/beam/{opt}/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='i')
                    h5group['Profile']['bunch'][opt][key][i1i:i1f] = getattr(self, f'buffer_profile_bunch_{opt}_{key}')[0:j1f]
                    h5group['Profile']['beam' ][opt][key][i1i:i1f] = getattr(self, f'buffer_profile_beam_{opt}_{key}' )[0:j1f]
                    self.h5file.flush()

            for opt in ['FWHM', 'RMS', 'BRF']:
                if opt == 'BRF': profile_stats_opt_keys = profile_stats_BRF_key_list[:]
                else:            profile_stats_opt_keys = profile_stats_FWHMRMS_key_list[:]
                for key in profile_stats_opt_keys:
                    h5group.require_dataset(f'Profile/bunch/{opt}/{key}', shape=self.shape_tot_mn1_plus1_x_nbf, dtype='f')
                    h5group.require_dataset(f'Profile/beam/{opt}/{key}',  shape=self.shape_tot_mn1_plus1,  dtype='f')
                    h5group['Profile']['bunch'][opt][key][i1i:i1f] = getattr(self, f'buffer_profile_bunch_{opt}_{key}')[0:j1f]
                    h5group['Profile']['beam' ][opt][key][i1i:i1f] = getattr(self, f'buffer_profile_beam_{opt}_{key}' )[0:j1f]
                    self.h5file.flush()

            # OTHERS

            h5group.require_dataset('turns', shape=self.shape_tot_mn2_plus1, dtype='i')
            h5group['turns'][i2i:i2f] = getattr(self, 'buffer_turns')[0:j2f]

            # # RFSTATION

            # h5group.require_dataset("RFStation/omega_rf", shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f')
            # h5group.require_dataset("RFStation/dphi_rf",  shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f')
            # h5group.require_dataset("RFStation/phi_rf",   shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f')
            # h5group["RFStation/omega_rf"][i2i:i2f] = self.buffer_rfstation_omega_rf[0:j2f]
            # h5group["RFStation/dphi_rf"][i2i:i2f]  = self.buffer_rfstation_dphi_rf[0:j2f]
            # h5group["RFStation/phi_rf"][i2i:i2f]   = self.buffer_rfstation_phi_rf[0:j2f]

            # TOTALINDUCEDVOLTAGE

            if self.totalinducedvoltage is not None:

                # h5group.require_dataset('TotalInducedVoltage/induced_voltage_minabs', shape=self.shape_tot_mn2_plus1, dtype='f')
                # h5group.require_dataset('TotalInducedVoltage/induced_voltage_aveabs', shape=self.shape_tot_mn2_plus1, dtype='f')
                h5group.require_dataset('TotalInducedVoltage/induced_voltage_maxabs', shape=self.shape_tot_mn2_plus1, dtype='f')
                # h5group['TotalInducedVoltage/induced_voltage_minabs'][i2i:i2f] = self.buffer_totalinducedvoltage_induced_voltage_minabs[0:j2f]
                # h5group['TotalInducedVoltage/induced_voltage_aveabs'][i2i:i2f] = self.buffer_totalinducedvoltage_induced_voltage_aveabs[0:j2f]
                h5group['TotalInducedVoltage/induced_voltage_maxabs'][i2i:i2f] = self.buffer_totalinducedvoltage_induced_voltage_maxabs[0:j2f]
                self.h5file.flush()

            # CAVITYFEEDBACK

            if self.cavityfeedback is not None:
                for key in ['V_sum', 'V_corr', 'phi_corr']:
                    if(key == 'V_sum'): dtypekey = np.dtype('complex64') # instead of the default 'complex128')
                    else:               dtypekey = 'f'
                    h5group.require_dataset(f'CavityFeedback/{key}', shape=self.shape_tot_mn2_plus1_x_nslices, dtype=dtypekey)
                    h5group['CavityFeedback'][key][i2i:i2f] = getattr(self, f'buffer_cavityfeedback_{key}')[0:j2f]
                    self.h5file.flush()

            # BEAMFEEDBACK

            if self.beamfeedback is not None:

                # Phase loop
                h5group.require_dataset("BeamFeedback/PL/omega_rf",  shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f') # CHECK: dtype=np.float64 ??
                h5group.require_dataset("BeamFeedback/PL/phi_rf",    shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f')
                h5group.require_dataset("BeamFeedback/PL/phi_beam",  shape=self.shape_tot_mn2_plus1,    dtype='f')
                h5group.require_dataset("BeamFeedback/PL/dphi",      shape=self.shape_tot_mn2_plus1,    dtype='f')
                h5group.require_dataset("BeamFeedback/PL/domega_rf", shape=self.shape_tot_mn2_plus1,    dtype='f')
                h5group["BeamFeedback/PL/omega_rf"][i2i:i2f]  = self.buffer_beamfeedback_PL_omega_rf[0:j2f]
                h5group["BeamFeedback/PL/phi_rf"][i2i:i2f]    = self.buffer_beamfeedback_PL_phi_rf[0:j2f]
                h5group["BeamFeedback/PL/phi_beam"][i2i:i2f]  = self.buffer_beamfeedback_PL_phi_beam[0:j2f]
                h5group["BeamFeedback/PL/dphi"][i2i:i2f]      = self.buffer_beamfeedback_PL_dphi[0:j2f]
                h5group["BeamFeedback/PL/domega_rf"][i2i:i2f] = self.buffer_beamfeedback_PL_domega_rf[0:j2f]
                # Synchro loop
                h5group.require_dataset("BeamFeedback/SL/dphi_rf", shape=self.shape_tot_mn2_plus1_x_nrf, dtype='f')
                h5group["BeamFeedback/SL/dphi_rf"][i2i:i2f] = self.buffer_beamfeedback_SL_dphi_rf[0:j2f]
                # Radial loop
                h5group.require_dataset("BeamFeedback/RL/drho", shape=self.shape_tot_mn2_plus1, dtype='f')
                h5group["BeamFeedback/RL/drho"][i2i:i2f] = self.buffer_beamfeedback_RL_drho[0:j2f]
                self.h5file.flush()

            # LHCNOISEFB

            if self.lhcnoisefb is not None:

                h5group.require_dataset('LHCNoiseFB/x',       shape=self.shape_tot_mn2_plus1, dtype='f')
                h5group.require_dataset('LHCNoiseFB/bl_meas', shape=self.shape_tot_mn2_plus1, dtype='f')
                h5group['LHCNoiseFB/x'][i2i:i2f]       = self.buffer_lhcnoisefb_x[0:j2f]
                h5group['LHCNoiseFB/bl_meas'][i2i:i2f] = self.buffer_lhcnoisefb_bl_meas[0:j2f]
                if self.lhcnoisefb.bl_meas_bbb is not None:
                    h5group.require_dataset('LHCNoiseFB/bl_meas_bbb', shape=self.shape_tot_mn2_plus1_x_nbf, dtype='f')
                    h5group['LHCNoiseFB/bl_meas_bbb'][i2i:i2f,:] = self.buffer_lhcnoisefb_bl_meas_bbb[0:j2f, :]
                self.h5file.flush()

        # Close file
        self.close()

###############################################################################


class MonitorProfile(object):
    '''
        Based on BlonD's SlicesMonitor
    '''

    def __init__(self, outdir, profile_obj, rfstation_obj, Nt_mn3, Nt_sav):

        self.filename = f'{outdir}/monitor_profile'

        self.profile = profile_obj
        self.rfstation  = rfstation_obj

        # Saving and monitoring intervals
        self.Nt_sav = Nt_sav
        self.Nt_mn3 = Nt_mn3

        # Total no. of turns to be monitored and saved (Note that self.rfstation.n_turns is already 1 turn lower than Nt_trk):
        self.n_turns_tot_mn3 = (self.rfstation.n_turns-1) // self.Nt_mn3
        if((self.rfstation.n_turns-1) % self.Nt_mn3 > 0): self.n_turns_tot_mn3 += 2 # Add space for 2 turns: turn 0 and the last turn
        else:                                             self.n_turns_tot_mn3 += 1 # Add space for 1 turn: turn 0 (the last turn is already included
        # Total no. of turns to be stored in buffer before saving:
        self.n_turns_buff_mn3 = self.Nt_sav // self.Nt_mn3 # Size (no. of turns) of buffer for Beam/Profile

        # "Current" turn
        self.i1 = -1

        # Shape of full arrays, and buffered arrays:

        self.shape_tot_mn3_plus1_x_nslices       = (self.n_turns_tot_mn3 + 1, self.profile.n_slices)   # For 'n_macroparticles' and 'bin_centers'
        self.shape_tot_mn3_plus1_x_nslices_plus1 = (self.n_turns_tot_mn3 + 1, self.profile.n_slices+1) # For 'edges'
        self.shape_tot_mn3_plus1                 = (self.n_turns_tot_mn3 + 1,)                         # For turns

        self.shape_buff_mn3_x_nslices       = (self.n_turns_buff_mn3, self.profile.n_slices)   # For 'n_macroparticles' and 'bin_centers'
        self.shape_buff_mn3_x_nslices_plus1 = (self.n_turns_buff_mn3, self.profile.n_slices+1) # For 'edges'
        self.shape_buff_mn3                 = (self.n_turns_buff_mn3,)                         # For turns

        # Initialise:

        self.init_data()

        # No need to track now. The original stated before tracking has already been saved as turn '-1' in the previous command. Turn 0 in tracking will call .track on its own

    def init_data(self):

        # Check and delete if file exists
        if os.path.exists(self.filename + '.h5'):
          os.remove(self.filename + '.h5')
        else:
          pass

        # Open file
        with hp.File(self.filename + '.h5', 'w') as self.h5file:

            # Create datasets
            self.h5file.require_group('Profile')
            h5group = self.h5file['Profile']

            # Create datasets
            h5group.create_dataset('turns', shape=self.shape_tot_mn3_plus1, dtype='i')
            h5group['turns'][0] = self.i1

            for key in ['bin_centers', 'n_macroparticles']: #, 'edges'
                if(key == 'n_macroparticles'): dtypekey = 'i'
                else:                          dtypekey = 'f'
                h5group.create_dataset(f'{key}', shape=self.shape_tot_mn3_plus1_x_nslices, dtype=dtypekey) #, compression="gzip", compression_opts=9)
                h5group[f'{key}'][0] = getattr(self.profile, key)
                #print(0, key, getattr(self.profile, key))
                self.h5file.flush()

            gc.collect()

        # Close file
        self.close()

        print(f'* MONITOR PROFILE SIZE: {os.path.getsize(self.filename + ".h5")/1024./1024.:.3f} MB')

        # Initialise buffer for next turn
        self.init_buffer()

    def close(self):
        try:    self.h5file.close()
        except: pass

    def init_buffer(self):

        setattr(self, 'buffer_profile_turns', np.zeros(self.shape_buff_mn3, dtype=int) )

        for key in ['bin_centers', 'n_macroparticles']: #, 'edges'
            setattr(self, f'buffer_profile_{key}', np.zeros(self.shape_buff_mn3_x_nslices))

    def track(self):

        if(((self.rfstation.counter[-1]-1) % self.Nt_mn3 == 0) or ((self.rfstation.counter[-1]-1) == self.rfstation.n_turns-1)):
            #print('Tracking monitorprofile...')
            self.write_buffer_1()
            self.i1 += 1

        # Save the data every Nt_sav
        if(((self.rfstation.counter[-1]-1) % self.Nt_sav == 0) or ((self.rfstation.counter[-1]-1) == self.rfstation.n_turns-1)):
            #self.open()
            self.write_data()
            #self.close()
            #print(f'* MONITOR PROFILE SIZE: {os.path.getsize(self.filename + ".h5")/1024./1024.:.3f} MB')
            self.init_buffer() # Restart the buffer

    def write_buffer_1(self):
        ''' Write buffer (Profile) '''

        # Place in the buffer (turn no.)
        if(self.i1 == -1): j = 0 # Turn 0
        else:              j = self.i1 % self.n_turns_buff_mn3 # = current turn % size of buffer (no. of turns)
        #print(f'Writing buffer: j2 = {j}')

        getattr(self, 'buffer_profile_turns')[j] = int(self.rfstation.counter[-1]-1)

        for key in ['bin_centers', 'n_macroparticles']:
            getattr(self, f'buffer_profile_{key}')[j] = getattr(self.profile, key)

    def open(self):
        self.h5file = hp.File(self.filename + '.h5', 'r+')

    def write_data(self):

        with hp.File(self.filename + '.h5', 'r+') as self.h5file:

            # Places (turns) in buffer to be written into datasets:
            if(  self.rfstation.counter[-1]-1 == 0):
                j1f = 1 # so we only take the first element stored in the Profile buffer
            elif(self.rfstation.counter[-1]-1 == self.rfstation.n_turns-1):
                j1f = (self.i1-1) % self.n_turns_buff_mn3 +1  # so we only take the elements with data (the rest are zero, as the buffer might contain more turns that the no. turn left in the tracking for the last buffer)
            else:
                j1f = self.n_turns_buff_mn3
            # Places in datasets where to write buffer data (we need one extra +1 as the first entry is '-1')
            i1i = self.i1-j1f+1+1
            i1f = self.i1+1+1

            h5group = self.h5file['/']

            h5group.require_dataset('Profile/turns', shape=self.shape_tot_mn3_plus1, dtype='i')
            h5group['Profile']['turns'][i1i:i1f] = getattr(self, 'buffer_profile_turns')[0:j1f]

            for key in ['bin_centers', 'n_macroparticles']: # 'bin_centers'
                if(key == 'n_macroparticles'): dtypekey = 'i'
                else:                          dtypekey = 'f'
                h5group.require_dataset(f'Profile/{key}', shape=self.shape_tot_mn3_plus1_x_nslices, dtype=dtypekey)
                h5group['Profile'][key][i1i:i1f] = getattr(self, f'buffer_profile_{key}')[0:j1f]
                self.h5file.flush()

            # print(self.buffer_profile_bin_centers)
            # print(self.buffer_profile_n_macroparticles)

        # Close file
        self.close()

###############################################################################
