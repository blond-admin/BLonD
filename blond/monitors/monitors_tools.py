#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import gc
import os
import sys
import h5py as hp
import numpy as np
#import pathlib
import pickle as pkl

from blond.llrf.cavity_feedback import get_power_gen_0, get_power_gen_VI, get_power_gen_I2, get_power_gen_V2

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


class MonitorOTFB(object):

    def __init__(self, cavityfeedback, fillpattern_beam, fillpattern_profile, list_params='short', i0=0, hof=0.5, profile=None,):

        # For the computation for beam-average parameters:
        # hof = 0.5# Take the second Half of the batch
        # hof = 0.0 # Take the Full batch

        self.cavityfeedback = cavityfeedback

        self.hof = hof
        self.profile = profile # If not None, the bunch profiles will be plotted too

        self.with_FF = bool( self.cavityfeedback.OTFB_1.open_FF or self.cavityfeedback.OTFB_2.open_FF ) # The flags are inversed at self.cavityfeedback instatiation, so 1.0 -> True  means WITH ACTIVE FF. One of the two OTFB_1 having active FF is enought to create the FF parameters for all

        self.time_arrays_dict = {}

        for ot in ['1','2']:

            # n_samples:
            setattr(self, f'n_mov_av_fine_{ot}',   getattr(self.cavityfeedback, f'OTFB_{ot}').n_mov_av_fine)
            setattr(self, f'n_mov_av_coarse_{ot}', getattr(self.cavityfeedback, f'OTFB_{ot}').n_mov_av_coarse)
            setattr(self, f'n_fine_{ot}',          getattr(self.cavityfeedback, f'OTFB_{ot}').n_fine)
            setattr(self, f'n_coarse_{ot}',        getattr(self.cavityfeedback, f'OTFB_{ot}').n_coarse)
            if not self.cavityfeedback.nollrf:
                setattr(self, f'n_fine_long_{ot}',     getattr(self.cavityfeedback, f'OTFB_{ot}').n_fine_long) # = n_fine_{ot} + n_mov_av_coarse_{ot} * profilepattern.Ns
                setattr(self, f'n_coarse_long_{ot}',   getattr(self.cavityfeedback, f'OTFB_{ot}').n_coarse_long)
            if self.with_FF:
                setattr(self, f'n_coarseFF_{ot}', getattr(self.cavityfeedback, f'OTFB_{ot}').n_coarseFF)

            # Generator-, beam-, and total (gen+beam) charges/currents/induced voltages:
            for qiv in ['Q', 'I', 'Vind']:
                gbt_list = ['gen', 'beam'] if qiv in ['Q', 'I'] else ['gen', 'beam', 'tot']
                for gbt in gbt_list:
                    #fc_list = ['coarse'] if gbt in ['gen'] and qiv in ['Q', 'I'] else ['coarse', 'fine']
                    fc_list = ['coarse', 'fine']
                    for fc in fc_list:
                        if qiv == 'Vind':
                            # i.e. for Vind_[gen|beam|tot]_[fine|coarse]
                            samples = getattr(self, f'n_{fc}_{ot}')
                            self.time_arrays_dict[f'OTFB_{ot}_{qiv}_{gbt}_{fc}'] = f'OTFB_{ot}_t_{fc}'
                        else:
                            if gbt == 'gen' and not self.cavityfeedback.nollrf:
                                # i.e. for [Q,I]_gen_[fine|coarse]
                                samples = getattr(self, f'n_{fc}_long_{ot}')
                                self.time_arrays_dict[f'OTFB_{ot}_{qiv}_{gbt}_{fc}'] = f'OTFB_{ot}_t_{fc}_long'
                            else: # i.e. beam
                                # i.e. for [Q,I]_beam_[fine|coarse]
                                samples = getattr(self, f'n_{fc}_{ot}')
                                self.time_arrays_dict[f'OTFB_{ot}_{qiv}_{gbt}_{fc}'] = f'OTFB_{ot}_t_{fc}'

                        setattr(self, f'OTFB_{ot}_{qiv}_{gbt}_{fc}', np.empty( shape=(0,samples) ))
                        #print(f'OTFB_{ot}_{qiv}_{gbt}_{fc} = {getattr(self, f"OTFB_{ot}_{qiv}_{gbt}_{fc}")}, shape = {getattr(self, f"OTFB_{ot}_{qiv}_{gbt}_{fc}").shape}')

            # FF parameters:
            if self.with_FF:
                for qiv in ['Q', 'dV']:
                    gbt = 'ff'
                    fc_list = ['coarseFF'] if qiv in ['Q'] else ['coarseFF', 'fine']
                    for fc in fc_list:
                        samples = getattr(self, f'n_{fc}_{ot}')
                        self.time_arrays_dict[f'OTFB_{ot}_{qiv}_{gbt}_{fc}'] = f'OTFB_{ot}_t_{fc}'
                        setattr(self, f'OTFB_{ot}_{qiv}_{gbt}_{fc}', np.empty( shape=(0,samples) ))
                # Missing interval FF parameters:
                #setattr(self, f'OTFB_{ot}_dV_coarseFF_ff_del', np.empty( shape=(0,samples) ))
                #setattr(self, f'OTFB_{ot}_dV_coarse_ff',       np.empty( shape=(0,getattr(self, f'n_coarse_{ot}')) ))

            # Set-point voltage and different stages of the calculation of the generator voltage in LLRF:
            for param in ['V_set', 'dV_err', 'dV_err_gain', 'dV_comb', 'dV_del', 'dV_mod', 'dV_Hcav', 'dV_gen', 'V_gen']:
                fc_list = ['coarse']
                for fc in fc_list:
                    setattr(self, f'OTFB_{ot}_{param}_{fc}', np.empty( shape=getattr(self, f'OTFB_{ot}_Vind_gen_{fc}').shape ))
                    self.time_arrays_dict[f'OTFB_{ot}_{param}_{fc}'] = f'OTFB_{ot}_t_{fc}'

            # Generator power:
            for fc in ['coarse', 'fine']:
                ### Power from current
                #setattr(self, f'OTFB_{ot}_P_gen_{fc}', np.empty( shape=getattr(self, f'OTFB_{ot}_Q_gen_{fc}').shape ))
                #self.time_arrays_dict[f'OTFB_{ot}_P_gen_{fc}'] = f't_{fc}_long_{ot}'
                ## Power from current
                setattr(self, f'OTFB_{ot}_P_gen_i2_{fc}', np.empty( shape=getattr(self, f'OTFB_{ot}_Q_gen_{fc}').shape ))
                if not self.cavityfeedback.nollrf: self.time_arrays_dict[f'OTFB_{ot}_P_gen_i2_{fc}'] = f'OTFB_{ot}_t_{fc}_long'
                else:                         self.time_arrays_dict[f'OTFB_{ot}_P_gen_i2_{fc}'] = f'OTFB_{ot}_t_{fc}'
                # Power from voltage
                setattr(self, f'OTFB_{ot}_P_gen_v2_{fc}', np.empty( shape=getattr(self, f'OTFB_{ot}_Vind_gen_{fc}').shape ))
                self.time_arrays_dict[f'OTFB_{ot}_P_gen_v2_{fc}'] = f'OTFB_{ot}_t_{fc}'

        # Sum (OTFB1+OTFB2) of total (gen+beam) induced voltages :
        setattr(self, 'OTFB_sum_Vind_tot_fine', np.empty( shape=(0,self.n_fine_1) )) # self.n_fine_1 = profile.n_slices = self.n_fine_2  # Only in fine grid
        self.time_arrays_dict['OTFB_sum_Vind_tot_fine'] = 't_fine'

        # Actual RF voltage and actual RF voltage + induced voltage from impedance model (all except TWCs):
        setattr(self, 'tracker_Vrf_fine', np.empty( shape=(0,self.n_fine_1) )) # Only in fine grid
        self.time_arrays_dict['tracker_Vrf_fine'] = 'profile_bin_centers' #'t_fine'
        setattr(self, 'tracker_Vtot_fine', np.empty( shape=(0,self.n_fine_1) )) # Only in fine grid
        self.time_arrays_dict['tracker_Vtot_fine'] = 'profile_bin_centers' #'t_fine'
        setattr(self, 'tracker_Vrfnocorr_fine', np.empty( shape=(0,self.n_fine_1) )) # Only in fine grid
        self.time_arrays_dict['tracker_Vrfnocorr_fine'] = 'profile_bin_centers' #'t_fine'

        # Indices of beam/no-beam segments
        self.indices_beam_fine   = fillpattern_profile[ int(hof*len(fillpattern_profile)): ] # idxP_nomargin does not include the samples of spacing buckets between filled buckets. We use idxP_nomargin instead of idxP to ignore the margins around the bucket (margins meant for beam stats and losses)
        self.indices_beam_coarse = fillpattern_beam[ int(hof*len(fillpattern_beam)): ]
        if self.with_FF:
            nbs = fillpattern_beam[1] - fillpattern_beam[0]
            self.indices_beam_coarseFF = (self.indices_beam_coarse/nbs).astype(int)
        # print(self.indices_beam_fine)
        # print(self.indices_beam_coarse)
        # print(self.indices_beam_coarseFF)
        # quit()
        fc_list = ['fine', 'coarse', 'coarseFF'] if self.with_FF else ['fine', 'coarse']
        for fc in fc_list:
            # Note that while the indices in the beam segment have jumps between the bunches contanied in beam,
            # in the no beam segment the indices will span a continuosly around the 3/4-portion of the ring
            setattr(self, f'n_samples_beam_{fc}', len( getattr(self, f'indices_beam_{fc}')))
            setattr(self, f'indices_nobeam_{fc}', np.arange( int(0.75*getattr(self, f'n_{fc}_1') - 0.5*getattr(self, f'n_samples_beam_{fc}')),
                                                             int(0.75*getattr(self, f'n_{fc}_1') + 0.5*getattr(self, f'n_samples_beam_{fc}')) ))


        # Time arrays:
        for ot in ['1','2']:
            setattr(self, f'OTFB_{ot}_t_fine',        np.empty( shape=(0,getattr(self, f'n_fine_{ot}')) ))
            setattr(self, f'OTFB_{ot}_t_coarse',      np.empty( shape=(0,getattr(self, f'n_coarse_{ot}')) ))
            if not self.cavityfeedback.nollrf:
                setattr(self, f'OTFB_{ot}_t_fine_long',   np.empty( shape=(0,getattr(self, f'n_fine_long_{ot}')) ))
                setattr(self, f'OTFB_{ot}_t_coarse_long', np.empty( shape=(0,getattr(self, f'n_coarse_long_{ot}')) ))
            if self.with_FF:
                setattr(self, f'OTFB_{ot}_t_coarseFF', np.empty( shape=(0,getattr(self, f'n_coarseFF_{ot}')) ))
        self.t_fine   = np.empty( shape=(0,self.n_fine_1) )   # 1 and 2 are by definition the same, doesn't apply for fine long
        self.t_coarse = np.empty( shape=(0,self.n_coarse_1) ) # 1 and 2 are by definition the same, doesn't apply for coarse long
        if self.with_FF:
            self.t_coarseFF = np.empty( shape=(0,self.n_coarseFF_1) )

        self.turns = np.empty( shape=(0,), dtype=int )

        # Profile (for testing):
        if self.profile is not None:
            self.profile_bin_centers      = np.empty( shape=(0,self.profile.n_slices) )
            self.profile_n_macroparticles = np.empty( shape=(0,self.profile.n_slices) )

        #

        if list_params == 'short':

            self.list_params  = ['OTFB_1_V_set_coarse',
                                        'OTFB_1_dV_err_coarse',
                                        # 'OTFB_1_dV_err_gain_coarse',
                                        # 'OTFB_1_dV_comb_coarse',
                                        # 'OTFB_1_dV_del_coarse',
                                        # 'OTFB_1_dV_mod_coarse',
                                        # 'OTFB_1_dV_Hcav_coarse',
                                        # 'OTFB_1_dV_gen_coarse',
                                        # 'OTFB_1_V_gen_coarse',
                                        'OTFB_1_Q_gen_coarse',
                                        # 'OTFB_1_I_gen_coarse',
                                        'OTFB_1_Vind_gen_coarse',
                                        # 'OTFB_1_Vind_gen_fine',
                                        'OTFB_1_P_gen_i2_coarse',
                                        # 'OTFB_1_P_gen_v2_coarse',
                                        # 'OTFB_1_P_gen_0_fine',
                                        'OTFB_1_Q_beam_coarse',
                                        'OTFB_1_Q_beam_fine',
                                        # 'OTFB_1_I_beam_coarse',
                                        # 'OTFB_1_I_beam_fine',
                                        'OTFB_1_Q_ff_coarseFF',
                                        # 'OTFB_1_dV_ff_coarseFF',
                                        'OTFB_1_dV_ff_fine',
                                        # 'OTFB_1_Vind_beam_coarse',
                                        'OTFB_1_Vind_beam_fine',
                                        # 'OTFB_1_Vind_tot_coarse',
                                        'OTFB_1_Vind_tot_fine',
                                        'OTFB_2_V_set_coarse',
                                        'OTFB_2_dV_err_coarse',
                                        # 'OTFB_2_dV_err_gain_coarse',
                                        # 'OTFB_2_dV_comb_coarse',
                                        # 'OTFB_2_dV_del_coarse',
                                        # 'OTFB_2_dV_mod_coarse',
                                        # 'OTFB_2_dV_Hcav_coarse',
                                        # 'OTFB_2_dV_gen_coarse',
                                        # 'OTFB_2_V_gen_coarse',
                                        'OTFB_2_Q_gen_coarse',
                                        # 'OTFB_2_I_gen_coarse',
                                        'OTFB_2_Vind_gen_coarse',
                                        # 'OTFB_2_Vind_gen_fine',
                                        'OTFB_2_P_gen_i2_coarse',
                                        # 'OTFB_2_P_gen_v2_coarse',
                                        # 'OTFB_2_P_gen_0_fine',
                                        'OTFB_2_Q_beam_coarse',
                                        'OTFB_2_Q_beam_fine',
                                        # 'OTFB_2_I_beam_coarse',
                                        #'OTFB_2_I_beam_fine',
                                        'OTFB_2_Q_ff_coarseFF',
                                        # 'OTFB_2_dV_ff_coarseFF',
                                        'OTFB_2_dV_ff_fine',
                                        # 'OTFB_2_Vind_beam_coarse',
                                        'OTFB_2_Vind_beam_fine',
                                        # 'OTFB_2_Vind_tot_coarse',
                                        'OTFB_2_Vind_tot_fine',
                                        'OTFB_sum_Vind_tot_fine'] #,
                                        # 'tracker_Vrfnocorr_fine',
                                        # 'tracker_Vrf_fine',
                                        # 'tracker_Vtot_fine']

        else:

            sys.exit('\n[!] ERROR in MonitorOTFB: list_params to be implemeneted!\n')

        # Check the list_params and remove FF params if not active:

        if not self.with_FF:
            # Remove ff parameters
            self.list_params = [param for param in self.list_params if 'ff' not in param]
        #print(self.list_params)
        #quit()

        # Create associated beam and nobeam parameters:

        for param in self.list_params:
            setattr(self, f'{param}_ave_beam',   np.empty(shape=(0,), dtype=complex))
            setattr(self, f'{param}_ave_nobeam', np.empty(shape=(0,), dtype=complex))

        #

        self.i0 = i0


    def track(self, i):
        ''' i is the current turn in the "for" tracking loop '''

        # All data is saved on a turn-by-tun basis below turn no. monitorotfb_i0. AAbova that thereshold, it's only saved
        # every Nt_mn2 turns, and at the last turn

        # #if (i % MAC['Nt_mn2'] == 0 or i == MAC['Nt_trk'] - 2):
        # synch_period = rfstation.omega_s0[rfstation.counter[0]]/2./np.pi
        # n_synch_period = int(MAC['Nt_trk']//synch_period)
        # sets_of_turns_to_saveplot = []
        # #print(f'synch_period = {synch_period}')
        # #print(f'n_synch_period = {n_synch_period}')
        # if n_synch_period > 0 and MAC['Nt_trk']-1 >= int(n_synch_period*synch_period)+4:
        #     for nsp_i in range(n_synch_period):
        #         sets_of_turns_to_saveplot.append( list(range(int(nsp_i*synch_period), int(nsp_i*synch_period)+4)) )
        # # print(f'sets_of_turns_to_saveplot = {sets_of_turns_to_saveplot}')

        # sets_of_turns_to_saveplot_flatten = [turni for seti in sets_of_turns_to_saveplot for turni in seti]
        # # print(f'sets_of_turns_to_saveplot_flatten = {sets_of_turns_to_saveplot_flatten}')
        # #quit()

        #if i in sets_of_turns_to_saveplot_flatten:
        #if i % MAC['Nt_mn2'] == 0:
        #if True:

        # print(f'profile.bin_centers = {profile.bin_centers}, shape = {profile.bin_centers.shape}')
        # print(f'self.cavityfeedback.OTFB_1.n_cavities = {self.cavityfeedback.OTFB_1.n_cavities}')
        # print(f'self.cavityfeedback.OTFB_1.V_gen_coarse = {self.cavityfeedback.OTFB_1.V_gen_coarse}, shape = {self.cavityfeedback.OTFB_1.V_gen_coarse.shape}')
        # print(f'self.cavityfeedback.OTFB_1.Q_gen_coarse = {self.cavityfeedback.OTFB_1.Q_gen_coarse}, shape = {self.cavityfeedback.OTFB_1.Q_gen_coarse.shape}')
        # print(f'self.cavityfeedback.OTFB_1.V_ind_gen_coarse = {self.cavityfeedback.OTFB_1.V_ind_gen_coarse}, shape = {self.cavityfeedback.OTFB_1.V_ind_gen_coarse.shape}')
        # print(f'self.cavityfeedback.OTFB_1.Q_beam_fine = {self.cavityfeedback.OTFB_1.Q_beam_fine}, shape = {self.cavityfeedback.OTFB_1.Q_beam_fine.shape}')
        # print(f'self.cavityfeedback.OTFB_1.Q_beam_coarse = {self.cavityfeedback.OTFB_1.Q_beam_coarse}, shape = {self.cavityfeedback.OTFB_1.Q_beam_coarse.shape}')
        # print(f'self.cavityfeedback.OTFB_1.V_ind_beam_fine = {self.cavityfeedback.OTFB_1.V_ind_beam_fine}, shape = {self.cavityfeedback.OTFB_1.V_ind_beam_fine.shape}')
        # print(f'self.cavityfeedback.OTFB_1.V_ind_beam_coarse = {self.cavityfeedback.OTFB_1.V_ind_beam_coarse}, shape = {self.cavityfeedback.OTFB_1.V_ind_beam_coarse.shape}')
        # print(f'self.cavityfeedback.OTFB_1.V_tot_fine = {self.cavityfeedback.OTFB_1.V_tot_fine}, shape = {self.cavityfeedback.OTFB_1.V_tot_fine.shape}')
        # print(f'self.cavityfeedback.OTFB_1.V_tot_coarse = {self.cavityfeedback.OTFB_1.V_tot_coarse}, shape = {self.cavityfeedback.OTFB_1.V_tot_coarse.shape}')

        self.turns = np.concatenate( (self.turns, np.array([int(i)])) )

        # e = 1.60217662e-19
        # print(f'OTFB_1_Q_beam_fine   = {self.cavityfeedback.OTFB_1.Q_beam_fine},   sum = {np.sum(self.cavityfeedback.OTFB_1.Q_beam_fine)},   abs(sum) = {abs(np.sum(self.cavityfeedback.OTFB_1.Q_beam_fine))},   abs(sum)/e/nbf = {abs(np.sum(self.cavityfeedback.OTFB_1.Q_beam_fine))/e/beampattern.nbf},   sum(abs) = {np.sum(np.abs(self.cavityfeedback.OTFB_1.Q_beam_fine))},   sum(abs)/e/nbf = {np.sum(np.abs(self.cavityfeedback.OTFB_1.Q_beam_fine))/e/beampattern.nbf}')
        # print(f'OTFB_1_Q_beam_coarse = {self.cavityfeedback.OTFB_1.Q_beam_coarse}, sum = {np.sum(self.cavityfeedback.OTFB_1.Q_beam_coarse)}, abs(sum) = {abs(np.sum(self.cavityfeedback.OTFB_1.Q_beam_coarse))}, abs(sum)/e/nbf = {abs(np.sum(self.cavityfeedback.OTFB_1.Q_beam_coarse))/e/beampattern.nbf}, sum(abs) = {np.sum(np.abs(self.cavityfeedback.OTFB_1.Q_beam_coarse))}, sum(abs)/e/nbf = {np.sum(np.abs(self.cavityfeedback.OTFB_1.Q_beam_coarse))/e/beampattern.nbf}')
        # quit()

        for ot in ['1','2']:

            # ## Extrapolation of Q_gen_coarse (coarse) to create Q_gen_fine -- To compare with running directly with use_genfine
            # OTFB_ot_Q_gen_fine = np.zeros( getattr(self, f'n_fine_long_{ot}'), dtype=complex)
            # OTFB_ot_I_gen_fine = np.zeros( getattr(self, f'n_fine_long_{ot}'), dtype=complex)
            # for j in range( 0, int(getattr(self, f'n_coarse_long_{ot}')), int(beampattern.nbs)):
            #     indices_j = np.arange( int((j+0.5)*profilepattern.Ns - 0.5*profilepattern.Ns), int((j+0.5)*profilepattern.Ns + 0.5*profilepattern.Ns) )
            #     #print(j, indices_j, len(getattr(self.cavityfeedback, f'OTFB_{ot}').Q_gen_coarse) )
            #     OTFB_ot_Q_gen_fine[ indices_j ] = getattr(self.cavityfeedback, f'OTFB_{ot}').Q_gen_coarse[j] / profilepattern.Ns
            #     OTFB_ot_I_gen_fine[ indices_j ] = getattr(self.cavityfeedback, f'OTFB_{ot}').Q_gen_coarse[j] / profilepattern.Ns # At this point I_gen_coarse is still charge

            setattr(self, f'OTFB_{ot}_t_fine',         np.vstack( (getattr(self, f'OTFB_{ot}_t_fine'),        getattr(self.cavityfeedback, f'OTFB_{ot}').t_fine[:]) ))
            setattr(self, f'OTFB_{ot}_t_coarse',       np.vstack( (getattr(self, f'OTFB_{ot}_t_coarse'),      getattr(self.cavityfeedback, f'OTFB_{ot}').t_coarse[:]) ))
            setattr(self, f'OTFB_{ot}_t_fine_long',    np.vstack( (getattr(self, f'OTFB_{ot}_t_fine_long'),   getattr(self.cavityfeedback, f'OTFB_{ot}').t_fine_long[:]) ))
            setattr(self, f'OTFB_{ot}_t_coarse_long',  np.vstack( (getattr(self, f'OTFB_{ot}_t_coarse_long'), getattr(self.cavityfeedback, f'OTFB_{ot}').t_coarse_long[:]) ))
            if self.with_FF:
                setattr(self, f'OTFB_{ot}_t_coarseFF', np.vstack( (getattr(self, f'OTFB_{ot}_t_coarseFF'),    getattr(self.cavityfeedback, f'OTFB_{ot}').t_coarseFF[:]) ))
            # print('self:', ot)
            # print( getattr(self, f'OTFB_{ot}_t_fine') )
            # print( getattr(self, f'OTFB_{ot}_t_fine_long') )
            # print( getattr(self, f'OTFB_{ot}_t_coarse') )
            # print( getattr(self, f'OTFB_{ot}_t_coarse_long') )
            if ot == '1':
                self.t_fine   = np.vstack( (self.t_fine,   self.OTFB_1_t_fine[-1]) )
                self.t_coarse = np.vstack( (self.t_coarse, self.OTFB_1_t_coarse[-1]) )
                if self.with_FF:
                    self.t_coarseFF = np.vstack( (self.t_coarseFF, self.OTFB_1_t_coarseFF[-1]) )
                # print( self.t_fine )
                # print( self.t_coarse )
                # print( self.t_coarseFF )
                #
                if self.profile is not None:
                    self.profile_bin_centers      = np.vstack( (self.profile_bin_centers,      self.profile.bin_centers[:]) )
                    self.profile_n_macroparticles = np.vstack( (self.profile_n_macroparticles, self.profile.n_macroparticles[:]) )
                    # print( self.profile_bin_centers )
                    # print( self.profile_n_macroparticles )

            #
            setattr(self, f'OTFB_{ot}_V_set_coarse',       np.vstack( (getattr(self, f'OTFB_{ot}_V_set_coarse'),       getattr(self.cavityfeedback, f'OTFB_{ot}').V_set_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_err_coarse',      np.vstack( (getattr(self, f'OTFB_{ot}_dV_err_coarse'),      getattr(self.cavityfeedback, f'OTFB_{ot}').dV_err_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_err_gain_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_dV_err_gain_coarse'), getattr(self.cavityfeedback, f'OTFB_{ot}').dV_err_gain_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_comb_coarse',     np.vstack( (getattr(self, f'OTFB_{ot}_dV_comb_coarse'),     getattr(self.cavityfeedback, f'OTFB_{ot}').dV_comb_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_del_coarse',      np.vstack( (getattr(self, f'OTFB_{ot}_dV_del_coarse'),      getattr(self.cavityfeedback, f'OTFB_{ot}').dV_del_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_mod_coarse',      np.vstack( (getattr(self, f'OTFB_{ot}_dV_mod_coarse'),      getattr(self.cavityfeedback, f'OTFB_{ot}').dV_mod_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_Hcav_coarse',     np.vstack( (getattr(self, f'OTFB_{ot}_dV_Hcav_coarse'),     getattr(self.cavityfeedback, f'OTFB_{ot}').dV_Hcav_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_dV_gen_coarse',      np.vstack( (getattr(self, f'OTFB_{ot}_dV_gen_coarse'),      getattr(self.cavityfeedback, f'OTFB_{ot}').dV_gen_coarse)[:] ))
            setattr(self, f'OTFB_{ot}_V_gen_coarse',       np.vstack( (getattr(self, f'OTFB_{ot}_V_gen_coarse'),       getattr(self.cavityfeedback, f'OTFB_{ot}').V_gen_coarse)[:] ))
            #
            # Note that generator- charge and current are for all cavitites/generators (to divide by no. cavities if using for power):
            # setattr(self, f'OTFB_{ot}_Q_gen_fine',   np.vstack( (getattr(self, f'OTFB_{ot}_Q_gen_fine'),   OTFB_ot_Q_gen_fine ) ))                                                         # Derived OTFB_{ot}_Vind_gen_fine (below), assuming all slices having the same charge (rectangular beam): at the end, this is equivalent to Q_beam_cooarse T_s_coarse = Ns * T_s_fine
            # setattr(self, f'OTFB_{ot}_I_gen_fine',   np.vstack( (getattr(self, f'OTFB_{ot}_I_gen_fine'),   OTFB_ot_Q_gen_fine / getattr(self.cavityfeedback, f'OTFB_{ot}').T_s_fine ) )) # Derived OTFB_{ot}_Vind_gen_fine (below), assuming all slices having the same charge (rectangular beam): at the end, this is equivalent to I_beam_cooarse T_s_coarse = Ns * T_s_fine
            setattr(self, f'OTFB_{ot}_Q_gen_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_Q_gen_coarse'), getattr(self.cavityfeedback, f'OTFB_{ot}').Q_gen_coarse[:] ) ))                                                                                # Qn the coarse grid, the time step is equal to t_rf (at present step); in the fine grid, the time step is T_s_fine
            setattr(self, f'OTFB_{ot}_I_gen_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_I_gen_coarse'), getattr(self.cavityfeedback, f'OTFB_{ot}').Q_gen_coarse[:] / getattr(self.cavityfeedback, f'OTFB_{ot}').T_s_coarse ) )) # In the coarse grid, the time step is equal to t_rf (at present step); in the fine grid, the time step is T_s_fine
            if False:
                # print(f'OTFB_{ot}_Q_gen_fine   = {getattr(self, f"OTFB_{ot}_Q_gen_fine")[-1]}, shape = {getattr(self, f"OTFB_{ot}_Q_gen_fine")[-1].shape}')
                print(f'OTFB_{ot}_Q_gen_coarse = {getattr(self, f"OTFB_{ot}_Q_gen_coarse")[-1]}, shape = {getattr(self, f"OTFB_{ot}_Q_gen_coarse")[-1].shape}')
                # print(f'OTFB_{ot}_I_gen_fine   = {getattr(self, f"OTFB_{ot}_I_gen_fine")[-1]}, shape = {getattr(self, f"OTFB_{ot}_I_gen_fine")[-1].shape}')
                # print(f'OTFB_{ot}_I_gen_coarse = {getattr(self, f"OTFB_{ot}_I_gen_coarse")[-1]}, shape = {getattr(self, f"OTFB_{ot}_I_gen_coarse")[-1].shape}')
                #quit()
            # Note that generator- induced voltage is for all cavities/generators (to divide by no. cavities if using for power):
            # setattr(self, f'OTFB_{ot}_Vind_gen_fine',   np.vstack( (getattr(self, f'OTFB_{ot}_Vind_gen_fine'),   getattr(self.cavityfeedback, f'OTFB_{ot}').V_tot_fine[:] - getattr(self.cavityfeedback, f'OTFB_{ot}').V_ind_beam_fine[:] ) )) # Doesn't exit: has to be interpolated (this is actually done internally in SPSCavityFeedback, but it is not saved)
            setattr(self, f'OTFB_{ot}_Vind_gen_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_Vind_gen_coarse'), getattr(self.cavityfeedback, f'OTFB_{ot}').V_ind_gen_coarse[:]) ))
            if False:
                print(f'OTFB_{ot}_Vind_gen_coarse = {getattr(self, f"OTFB_{ot}_Vind_gen_coarse")}, shape = {getattr(self, f"OTFB_{ot}_Vind_gen_coarse").shape}')

            #

            P_gen_0_cav_coarse_ot  = get_power_gen_0(  getattr(self, f'OTFB_{ot}_Vind_gen_coarse')[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.Z_0, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.R_gen )
            # P_gen_0_cav_fine_ot    = get_power_gen_0(  getattr(self, f'OTFB_{ot}_Vind_gen_fine'  )[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.Z_0, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.R_gen )

            P_gen_i2_cav_coarse_ot = get_power_gen_I2( getattr(self, f'OTFB_{ot}_I_gen_coarse')[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.Z_0 )
            # P_gen_i2_cav_fine_ot   = get_power_gen_I2( getattr(self, f'OTFB_{ot}_I_gen_fine'  )[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.Z_0 )

            P_gen_vi_cav_coarse_ot = get_power_gen_VI( getattr(self, f'OTFB_{ot}_Vind_gen_coarse')[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self, f'OTFB_{ot}_I_gen_coarse')[-1][ getattr(self, f'n_mov_av_coarse_{ot}'): ] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities )
            # P_gen_vi_cav_fine_ot   = get_power_gen_VI( getattr(self, f'OTFB_{ot}_Vind_gen_fine')  [-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self, f'OTFB_{ot}_I_gen_fine'  )[-1][ getattr(self, f'n_mov_av_fine_{ot}'):   ] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities )

            P_gen_v2_cav_coarse_ot = get_power_gen_V2( getattr(self, f'OTFB_{ot}_Vind_gen_coarse')[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.Z_0, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.R_gen, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.tau, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.d_omega )
            # P_gen_v2_cav_fine_ot   = get_power_gen_V2( getattr(self, f'OTFB_{ot}_Vind_gen_fine'  )[-1] / getattr(self.cavityfeedback, f'OTFB_{ot}').n_cavities, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.Z_0, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.R_gen, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.tau, getattr(self.cavityfeedback, f'OTFB_{ot}').TWC.d_omega )

            ##################################################################
            # if False and (i%MAC["Nt_plt"] == 0 or i == MAC['Nt_trk']-2):

            #     # print(f'ot = {ot}')
            #     # print(f'P_gen_0_cav_coarse_ot = {P_gen_0_cav_coarse_ot}, shape = {P_gen_0_cav_coarse_ot.shape}')
            #     # # print(f'P_gen_0_cav_fine_ot   = {P_gen_0_cav_fine_ot}, shape = {P_gen_0_cav_fine_ot.shape}')
            #     # print(f'P_gen_i2_cav_coarse_ot = {P_gen_i2_cav_coarse_ot}, shape = {P_gen_i2_cav_coarse_ot.shape}')
            #     # # print(f'P_gen_i2_cav_fine_ot   = {P_gen_i2_cav_fine_ot}, shape = {P_gen_i2_cav_fine_ot.shape}')
            #     # print(f'P_gen_vi_cav_coarse_ot = {P_gen_vi_cav_coarse_ot}, shape = {P_gen_vi_cav_coarse_ot.shape}')
            #     # # print(f'P_gen_vi_cav_fine_ot   = {P_gen_vi_cav_fine_ot}, shape = {P_gen_vi_cav_fine_ot.shape}')
            #     # print(f'P_gen_v2_cav_coarse_ot = {P_gen_v2_cav_coarse_ot}, shape = {P_gen_v2_cav_coarse_ot.shape}')
            #     # # print(f'P_gen_v2_cav_fine_ot   = {P_gen_v2_cav_fine_ot}, shape = {P_gen_v2_cav_fine_ot.shape}')

            #     nptsplot = int(1e6) # int(0.25*len(Pgen_opt1)) or a large number e.g. int(1e6) for all
            #     fig, ax = plt.subplots() #2, 1)
            #     fig.set_size_inches(8.0, 6.0) #18.0,6.0)
            #     #fig.subplots_adjust(left=0.090, right=0.70)
            #     #
            #     # ax2 = ax.twinx()
            #     # ax3 = ax.twinx()
            #     # ax4 = ax.twinx()
            #     # ax5 = ax.twinx()
            #     #
            #     # ax3.spines["right"].set_position(("axes", 1.10))
            #     # ax4.spines["right"].set_position(("axes", 1.25))
            #     # ax5.spines["right"].set_position(("axes", 1.35))

            #     # l5a, = ax5.plot(P_gen_i2_coarse_ot[-getattr(self, f'n_coarse_{ot}'):], '-', c='lime', label='P_gen_i2_coarse_ot', alpha=0.5) # Remove the delay in the current signal from which P is derived
            #     # # l5b, = ax5.plot(np.arange(0, len(P_gen_i2_fine_ot))/profilepattern.Ns, P_gen_i2_fine_ot, ':',  c='green', label='P_gen_i2_coarse_ot', alpha=0.5)
            #     # #ax3.tick_params(axis='y', colors='magenta')
            #     # #
            #     # l3a, = ax3.plot(Pgen_opt0a[-getattr(self, f'n_coarse_{ot}'):], '--', c='magenta', label='Pgen_opt0a', alpha=0.5) # Remove the delay in the current signal from which P is derived
            #     # # l3b, = ax3.plot(np.arange(0, len(Pgen_opt0b))/profilepattern.Ns, Pgen_opt0b, '-',  c='r',      label='Pgen_opt0a', alpha=0.5)
            #     # #ax3.tick_params(axis='y', colors='r')
            #     # #
            #     # l1a, = ax.plot(Pgen_opt1, c='k', label='Pgen_opt1', alpha=0.5)
            #     # #
            #     # # l2a, = ax2.plot(Pgen_opt2, '-',  c='b',    label='Pgen_opt2', alpha=0.5)
            #     # # l2b, = ax2.plot(Pgen_opt5, '--', c='cyan', label='Pgen_opt5', alpha=0.5)
            #     # # # ax2.plot(Pgen_opt3,  label='Pgen_opt3')
            #     # # # ax2.plot(Pgen_opt4,  label='Pgen_opt4')
            #     # #
            #     # l4a, = ax4.plot(Pgen_opt6, c='orange', label='Pgen_opt6', alpha=0.5)
            #     # #ax4.tick_params(axis='y', colors='g')
            #     # #
            #     # # lines = [l1a, l2a, l2b, l3a, l3b, l4a, l5a, l5b]
            #     # # lines = [l1a, l2a, l2b, l3a, l4a, l5a]
            #     # lines = [l1a, l3a, l4a, l5a]
            #     # ax.legend(lines, [l.get_label() for l in lines])
            #     # #
            #     # # for axi, pi in [(ax2, l2a), (ax3, l3a), (ax4, l4a), (ax5, l5a)]:
            #     # for axi, pi in [(ax3, l3a), (ax4, l4a), (ax5, l5a)]:
            #     #     axi.set_frame_on(True)
            #     #     axi.patch.set_visible(False)
            #     #     plt.setp(axi.spines.values(), visible=False)
            #     #     axi.spines["right"].set_visible(True)
            #     #     axi.spines["right"].set_edgecolor(pi.get_color())

            #     # # for axi, pi in [(ax, l1a), (ax2, l2a), (ax3, l3a), (ax4, l4a), (ax5, l5a)]:
            #     # for axi, pi in [(ax, l1a), (ax3, l3a), (ax4, l4a), (ax5, l5a)]:
            #     #     axi.yaxis.label.set_color(pi.get_color())
            #     #     axi.tick_params(axis='y', colors=pi.get_color())
            #     #     axi.ticklabel_format(useOffset=False, style='plain') # disable offset (e.g. +4.309e5 at the top of the yaxis) and scientific notation (e.g. 1e-7 at top of yaxis())
            #     #

            #     ax.plot( np.arange(getattr(self, f'n_coarse_{ot}')),                                                    P_gen_0_cav_coarse_ot/ 1e6, ':',  label='P_gen_0_cav_coarse_ot')
            #     ax.plot( np.arange(getattr(self, f'n_coarse_long_{ot}'))-getattr(self, f'n_mov_av_coarse_{ot}'), P_gen_i2_cav_coarse_ot/1e6, '-',  label='P_gen_i2_cav_coarse_ot')
            #     ax.plot( np.arange(getattr(self, f'n_coarse_{ot}')),                                                    P_gen_v2_cav_coarse_ot/1e6, '--', label='P_gen_v2_cav_coarse_ot')
            #     # ax2.plot(P_gen_vi_cav_coarse_ot/1e6, '--', c='grey', label='P_gen_vi_cav_coarse_ot')
            #     ax.legend(loc=1)
            #     # ax2.legend(loc=7)

            #     ax.set_xlabel('RF bucket')
            #     ax.set_ylabel('RF generator power [MW]')

            #     ax.set_title(f'OTFB{ot}, turn {i}')
            #     # ax.set_xlim(0, int(2*beampattern.fillpattern[-1]))
            #     fig.tight_layout()
            #     fname = f'{MAC["outdir"]}/plot_track_P_gen_{ot}_turn_{i}.png'
            #     print(f'Saving {fname} ... (turn {i})')
            #     fig.savefig(fname)
            #     plt.close()
            #     #
            # quit()
            ##################################################################

            setattr(self, f'OTFB_{ot}_P_gen_i2_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_P_gen_i2_coarse'),  P_gen_i2_cav_coarse_ot) ))
            setattr(self, f'OTFB_{ot}_P_gen_v2_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_P_gen_v2_coarse'),  P_gen_v2_cav_coarse_ot) ))
            if False:
                print(f'OTFB_{ot}_P_gen_i2_coarse = {getattr(self, f"OTFB_{ot}_P_gen_i2_coarse")}, shape = {getattr(self, f"OTFB_{ot}_P_gen_i2_coarse").shape}')
                print(f'OTFB_{ot}_P_gen_v2_coarse = {getattr(self, f"OTFB_{ot}_P_gen_v2_coarse")}, shape = {getattr(self, f"OTFB_{ot}_P_gen_v2_coarse").shape}')

            setattr(self, f'OTFB_{ot}_Q_beam_fine',      np.vstack( (getattr(self, f'OTFB_{ot}_Q_beam_fine'),      getattr(self.cavityfeedback, f'OTFB_{ot}').Q_beam_fine[:] )   ))
           #setattr(self, f'OTFB_{ot}_I_beam_fine',      np.vstack( (getattr(self, f'OTFB_{ot}_I_beam_fine'),      getattr(self.cavityfeedback, f'OTFB_{ot}').Q_beam_fine[:] / getattr(self.cavityfeedback, f'OTFB_{ot}').T_s_fine ) ))
            setattr(self, f'OTFB_{ot}_Q_beam_coarse',    np.vstack( (getattr(self, f'OTFB_{ot}_Q_beam_coarse'),    getattr(self.cavityfeedback, f'OTFB_{ot}').Q_beam_coarse[:] ) )) # Q_beam_coarse was properly computed as the sum of all the Q_bean_fine contributions per coarse point: Indeed, the full charge per coarse sample (i.e. bucket) is np.abs(getattr(self.cavityfeedback, f'OTFB_{ot}').Q_beam_coarse[0] = np.abs(np.sum(getattr(self.cavityfeedback, f'OTFB_{ot}').Q_beam_fine[:64]))
           #setattr(self, f'OTFB_{ot}_I_beam_coarse',    np.vstack( (getattr(self, f'OTFB_{ot}_I_beam_coarse'),    getattr(self.cavityfeedback, f'OTFB_{ot}').Q_beam_coarse[:] / getattr(self.cavityfeedback, f'OTFB_{ot}').T_s_coarse ) )) # I_beam_coarse was properly computed as the sum of all the I_bean_fine contributions per coarse point: Indeed, the full charge per coarse sample (i.e. bucket) is np.abs(getattr(self.cavityfeedback, f'OTFB_{ot}').I_beam_coarse[0] = np.abs(np.sum(getattr(self.cavityfeedback, f'OTFB_{ot}').I_beam_fine[:64]))
            if self.with_FF:
                setattr(self, f'OTFB_{ot}_Q_ff_coarseFF',  np.vstack( (getattr(self, f'OTFB_{ot}_Q_ff_coarseFF'),  getattr(self.cavityfeedback, f'OTFB_{ot}').Q_coarseFF_ff[:]) ))
                setattr(self, f'OTFB_{ot}_dV_ff_coarseFF', np.vstack( (getattr(self, f'OTFB_{ot}_dV_ff_coarseFF'), getattr(self.cavityfeedback, f'OTFB_{ot}').dV_coarseFF_ff[:]) ))
                setattr(self, f'OTFB_{ot}_dV_ff_fine',     np.vstack( (getattr(self, f'OTFB_{ot}_dV_ff_fine'),     getattr(self.cavityfeedback, f'OTFB_{ot}').dV_fine_ff[:]) ))
                if False:
                    print(f'OTFB_{ot}_Q_ff_coarseFF  = {getattr(self, f"OTFB_{ot}_Q_ff_coarseFF")}, shape = {getattr(self, f"OTFB_{ot}_Q_ff_coarseFF").shape}')
                    print(f'OTFB_{ot}_dV_ff_coarseFF = {getattr(self, f"OTFB_{ot}_dV_ff_coarseFF")}, shape = {getattr(self, f"OTFB_{ot}_dV_ff_coarseFF").shape}')
                    print(f'OTFB_{ot}_dV_ff_fine     = {getattr(self, f"OTFB_{ot}_dV_ff_fine")}, shape = {getattr(self, f"OTFB_{ot}_dV_ff_fine").shape}')
            setattr(self, f'OTFB_{ot}_Vind_beam_fine',   np.vstack( (getattr(self, f'OTFB_{ot}_Vind_beam_fine'),   getattr(self.cavityfeedback, f'OTFB_{ot}').V_ind_beam_fine[:]) ))
            #setattr(self, f'OTFB_{ot}_Vind_beam_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_Vind_beam_coarse'), getattr(self.cavityfeedback, f'OTFB_{ot}').V_ind_beam_coarse[:]) ))
            if False:
                print(f'OTFB_{ot}_Q_beam_fine   = {getattr(self, f"OTFB_{ot}_Q_beam_fine")}, shape = {getattr(self, f"OTFB_{ot}_Q_beam_fine").shape}')
                #print(f'OTFB_{ot}_Q_beam_coarse = {getattr(self, f"OTFB_{ot}_Q_beam_coarse")}, shape = {getattr(self, f"OTFB_{ot}_Q_beam_coarse").shape}')
                #print(f'OTFB_{ot}_I_beam_fine   = {getattr(self, f"OTFB_{ot}_I_beam_fine")}, shape = {getattr(self, f"OTFB_{ot}_I_beam_fine").shape}')
                #print(f'OTFB_{ot}_I_beam_coarse = {getattr(self, f"OTFB_{ot}_I_beam_coarse")}, shape = {getattr(self, f"OTFB_{ot}_I_beam_coarse").shape}')
                print(f'OTFB_{ot}_Vind_beam_fine = {getattr(self, f"OTFB_{ot}_Vind_beam_fine")}, shape = {getattr(self, f"OTFB_{ot}_Vind_beam_fine").shape}')

            setattr(self, f'OTFB_{ot}_Vind_tot_fine',   np.vstack( (getattr(self, f'OTFB_{ot}_Vind_tot_fine'),   getattr(self.cavityfeedback, f'OTFB_{ot}').V_tot_fine[:]) ))
            #setattr(self, f'OTFB_{ot}_Vind_tot_coarse', np.vstack( (getattr(self, f'OTFB_{ot}_Vind_tot_coarse'), getattr(self.cavityfeedback, f'OTFB_{ot}').V_tot_coarse[:]) ))

        self.OTFB_sum_Vind_tot_fine = np.vstack( (self.OTFB_sum_Vind_tot_fine, self.cavityfeedback.V_sum) )

        if self.profile is not None:
            # For testing, not mantained anymore
            # We need to add zeros at 1st turn for the ringandrftracker quantities since we are tracking self.cavityfeedback before beamfeedback
            self.tracker_Vrf_fine  = np.vstack( (self.tracker_Vrf_fine,  ringandrftracker.rf_voltage    if i > 0 else np.zeros(self.profile.n_slices)) )
            self.tracker_Vtot_fine = np.vstack( (self.tracker_Vtot_fine, ringandrftracker.total_voltage if i > 0 else np.zeros(self.profile.n_slices) ) )
            # This is the voltage that would be created in the trakcer if we don't have self.cavityfeedback
            tmp_voltage  = np.ascontiguousarray(rfstation.voltage[:, rfstation.counter[0]])
            tmp_omega_rf = np.ascontiguousarray(rfstation.omega_rf[:, rfstation.counter[0]])
            tmp_phi_rf   = np.ascontiguousarray(rfstation.phi_rf[:, rfstation.counter[0]])
            self.tracker_Vrfnocorr_fine = np.vstack( (self.tracker_Vrfnocorr_fine, (tmp_voltage[0] * bm.sin(tmp_omega_rf[0]*self.profile.bin_centers + tmp_phi_rf[0]) + bm.rf_volt_comp(tmp_voltage[1:], tmp_omega_rf[1:], tmp_phi_rf[1:], self.profile.bin_centers) if i > 0 else np.zeros(self.profile.n_slices)) ) )

        #print(self.turns)


    def track_ave(self):

        for param in self.list_params: # all of the above, regardless of if they are to be plotted or not (beacuse we are removing to save space)

            ot = '1' if '1' in param else '2'
            if   'fine'     in param: fc = 'fine'
            elif 'coarseFF' in param: fc = 'coarseFF'
            else:                     fc = 'coarse'

            param_ii = getattr(self, param)[-1]
            # print(f'{param}:, {param_ii}, shape = {param_ii}')
            indices_beam_ii   = np.copy(getattr(self, f'indices_beam_{fc}'))
            indices_nobeam_ii = np.copy(getattr(self, f'indices_nobeam_{fc}'))
            # For quantities in the long arrays, we need to shift this indices by n_mov_av_[fine|coarse] (the TWC tau)
            if 'long' in self.time_arrays_dict[param]:
                indices_beam_ii += getattr(self, f'n_mov_av_{fc}_{ot}')
            param_ii_ave_beam   = np.average(param_ii[ indices_beam_ii ])
            param_ii_ave_nobeam = np.average(param_ii[ indices_nobeam_ii ])

            #print(f'{i} {param}')
            #print(f'param_ii = {param_ii}, shape = {param_ii.shape}')
            #print(f'param_ii_ave_beam   = {param_ii_ave_beam}')
            #print(f'param_ii_ave_nobeam = {param_ii_ave_nobeam}')

            setattr(self, f'{param}_ave_beam',   np.hstack( (getattr(self, f'{param}_ave_beam'),   np.array(param_ii_ave_beam)) ))
            setattr(self, f'{param}_ave_nobeam', np.hstack( (getattr(self, f'{param}_ave_nobeam'), np.array(param_ii_ave_nobeam)) ))

            #print(f'{param}_ave_beam   = {getattr(self, f"{param}_ave_beam")}')
            #print(f'{param}_ave_nobeam = {getattr(self, f"{param}_ave_nobeam")}')

            # #if MAC['Nt_trk'] > monitorotfb_i0:
            # if i > monitorotfb_i0:
            #     ##### CAREFULLLLL HERE!
            #     # When testing with amny turns, we can only look at the average beam data, since it's to heavy to keep the full data in memory..
            #     #print(f'{param} = {getattr(self, f"{param}")}, shape = {getattr(self, f"{param}").shape}')
            #     #setattr(self, param, np.empty( shape=(getattr(self, param).shape) )) # WRONG: the arrays are rewritten as empty, but they still have the same size at the current step (i.e. after appending) so we are not really emptying memory)
            #     setattr(self, param, np.empty( shape=(0,getattr(self, param).shape[1]) ))
            #     #print(f'{param} = {getattr(self, f"{param}")}, shape = {getattr(self, f"{param}").shape}')

        gc.collect()

    def save_pickle(self, outdir):

        dicmonitorotfb = {}
        dicmonitorotfb['turns'] = np.copy(self.turns)

        # for key in list_monitorotfb_params:
        #     if '1' in key:
        #         dicmonitorotfb[key] = getattr(monitorotfb, key)
        # for key in [param for param in list(dir(monitorotfb)) if 't_'  in param]:
        #     if '1' in key:
        #         dicmonitorotfb[key] = getattr(monitorotfb, key)

        for key in self.list_params:
            if 'ave' in key:
                dicmonitorotfb[key] = getattr(self, key)

        with open(f'{outdir}/monitor_otfb.pkl', 'wb') as foutgen:
            pkl.dump(dicmonitorotfb, foutgen)

        print(f'dicmonitorotfb:')
        for key in dicmonitorotfb.keys():
            if key == 'turns':
                print(f'{key}: {dicmonitorotfb[key]}, shape = {dicmonitorotfb[key].shape}')
            else:
                for i in [0,-1]:
                    print(f'{key}[{i}]: {dicmonitorotfb[key][i]}, abs = {np.abs(dicmonitorotfb[key][i])}, ang = {np.angle(dicmonitorotfb[key][i])}')
