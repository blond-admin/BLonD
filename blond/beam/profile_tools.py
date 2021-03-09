#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import gc
import numpy as np
from scipy import optimize, integrate
from scipy.constants import c as clight
#from scipy.stats import linregress
#import time
from warnings import filterwarnings
import pathlib

import matplotlib.pyplot as plt

from blond.toolbox import filters_and_fitting as ffroutines
from blond.utils import bmath as bm

dirhome = str(pathlib.Path.home())

if(  dirhome.startswith('/afs')  ):       myenv = 'afs'       # RUNNING LOCALLY (AFS)
elif(dirhome.startswith('/pool') ):       myenv = 'batch'     # RUNNING WITH HTCONDOR
elif(dirhome.startswith('/hpcscratch') ): myenv = 'hpc'       # RUNNING WITH SLURM
elif(dirhome.startswith('/home') ):       myenv = 'Ubuntu'    # RUNNING LOCALLY (UBUNTU)
elif(dirhome.startswith('/Users')):       myenv = 'Mac'       # RUNNING LOCALLY (MAC)

if(  myenv == 'afs'):   dirhome = '/afs/cern.ch/work/l/lmedinam' # When running locally in AFS, re-assign dirhome so that we use the original version of the scripts in AFS work
elif(myenv == 'batch'): dirhome = '/afs/cern.ch/work/l/lmedinam' # When running with HTCondor,  re-assign dirhome so that we use the original version of the scripts in AFS work (we do not transfer input files, as this way it's faster)
elif(myenv == 'hpc'):   pass                                     # When running with Slurm, no need to re-assign dirhome, as a local copy of the full BLonD_simulations exist in the Slurh home directory
elif(myenv == 'Mac'):   pass                                     # When running with Slurm, no need to re-assign dirhome. The BLonD_simulations directory must exist there
else:                   sys.exit('\n[!] ERROR in plot_beam_lm: NOT IMPLEMENTED!\n')

if myenv in ['afs', 'batch', 'hpc']:
    import matplotlib as mpl
    mpl.use('Agg')

###############################################################################

def linear_fit(x, m, b):
    return m * x + b

class ProfileTools(object):

    # No. of macroparticles ===================================================

    def nmacrop_bunch(profile_obj, idxP_BF_bunch, idxP_BS_bunch, idxP_BG_bunch, initial=None):

        nbf = len(idxP_BF_bunch)
        BF_alive_array = np.array([]).astype(int)
        BS_alive_array = np.array([]).astype(int)
        BG_alive_array = np.array([]).astype(int)
        for i in range(nbf):
            BF_alive_array = np.append(BF_alive_array, int(sum(profile_obj.n_macroparticles[idxP_BF_bunch[i]])))
            BS_alive_array = np.append(BS_alive_array, int(sum(profile_obj.n_macroparticles[idxP_BS_bunch[i]])))
            BG_alive_array = np.append(BG_alive_array, int(sum(profile_obj.n_macroparticles[idxP_BG_bunch[i]])))
        result = {'BF': {'alive': BF_alive_array.astype(int)},
                  'BS': {'alive': BS_alive_array.astype(int)},
                  'BG': {'alive': BG_alive_array.astype(int)}}
        if(initial is not None):
            BF_lost_array = initial['BF']['alive']-BF_alive_array
            BS_lost_array = initial['BS']['alive']-BS_alive_array
            BG_lost_array = initial['BG']['alive']-BG_alive_array
            result['BF']['lost'] = BF_lost_array.astype(int)
            result['BS']['lost'] = BS_lost_array.astype(int)
            result['BG']['lost'] = BG_lost_array.astype(int)
        gc.collect()
        return result

    def nmacrop_beam(profile_nmacrop_bunch_OR_profile_obj, idxP_BF_beam=None, idxP_BS_beam=None, idxP_BG_beam=None, initial=None):

        if(isinstance(profile_nmacrop_bunch_OR_profile_obj, dict)):
            result = {}
            for opt in profile_nmacrop_bunch_OR_profile_obj.keys():
                result[opt] = {}
                for key in profile_nmacrop_bunch_OR_profile_obj[opt].keys():
                    result[opt][key] = np.sum(profile_nmacrop_bunch_OR_profile_obj[opt][key])
        else:
            BF_alive = int(sum(profile_nmacrop_bunch_OR_profile_obj.n_macroparticles[idxP_BF_beam]))
            BS_alive = int(sum(profile_nmacrop_bunch_OR_profile_obj.n_macroparticles[idxP_BS_beam]))
            BG_alive = int(sum(profile_nmacrop_bunch_OR_profile_obj.n_macroparticles[idxP_BG_beam]))
            result = {'BF': {'alive': BF_alive},
                      'BS': {'alive': BS_alive},
                      'BG': {'alive': BG_alive}}
            if(initial is not None):
                BF_lost = initial['BF']['alive']-BF_alive
                BS_lost = initial['BS']['alive']-BS_alive
                BG_lost = initial['BG']['alive']-BG_alive
                result['BF']['lost'] = int(BF_lost)
                result['BS']['lost'] = int(BS_lost)
                result['BG']['lost'] = int(BG_lost)
        gc.collect()
        return result

    # Statistics (position, length) ===========================================

    def stats_bunch(profile_obj, idxP_BF_bunch, bucket_centres, nbs, maxbktl, marginoffset0, rfstation_obj, Fbsymmetric=True):
        '''
        Calculation of the FWHM and RMS bunch length, bunch position, and
        bunch position offsets (from bucket centres, fit, and beam phase - RF)

        Input:
        - profile_obj
        - bucket_centres
        - nbs
        - maxbktl
        - marginoffset0
        - rfstation_obj
        - idxP_BF_bunch

        Returns:
        - FWHM bunchPosition
        - FWHM bunchLength
        - FWHM bunchPositionOff_ctr (w.r.t. bucket centres (single RF, no impedance model, but takes into account beam phase shift due to beam feedback))
        - FWHM bunchPositionOff_fit (w.r.t. batch fit evaluated at the corresponding bunch no., returns [NaN] for single bunch)
        - FWHM bunchPositionOff_brf (beam phase - RF)
        - RMS bunchPosition
        - RMS bunchLength
        - RMS bunchPositionOff
        - RMS bunchPositionOff_ctr (w.r.t. bucket centres (single RF, no impedance model, but takes into account beam phase shift due to beam feedback))
        - RMS bunchPositionOff_fit (w.r.t. batch fit evaluated at the corresponding bunch no., returns [NaN] for single bunch)
        - RMS bunchPositionOff_brf (beam phase - RF)
        '''

        nbf = len(bucket_centres)

        # indices = np.where(profile_obj.bin_centers > profile_obj.cut_left + marginoffset0)[0]
        # print(f'profile_obj.bin_centers = {profile_obj.bin_centers}, shape = {profile_obj.bin_centers.shape}')
        # print(f'profile_obj.cut_left = {profile_obj.cut_left}')
        # print(f'marginoffset0 = {marginoffset0}')
        # print(f'indices = {indices}, shape = {indices.shape}')

        profile_cut_left_to_zero = profile_obj.cut_left + marginoffset0

        # print('> Calling profile stats_bunch...')
        # print(f'profile_obj.bin_centers = [{profile_obj.bin_centers[0]},...]')
        # print(f'> Correcting by {profile_cut_left_to_zero}...')
        profile_obj.bin_centers -= profile_cut_left_to_zero
        # print(f'profile_obj.bin_centers = [{profile_obj.bin_centers[0]},...]')
        if(nbf == 1):
            profile_obj.fwhm() # gives profile_obj.bunchPosition, profile_obj.bunchLength measured in Gaussian RMS (the bunch length has a factor of 4)
            profile_obj_bunchPosition_RMS, profile_obj_bunchLength_RMS = ffroutines.rms(profile_obj.n_macroparticles, profile_obj.bin_centers) # in terms of the actual RMS of the distribution (the bunch length has a factor of 4)
        else:
            profile_obj.fwhm_multibunch(nbf, nbs, maxbktl)
            profile_obj_bunchPosition_RMS, profile_obj_bunchLength_RMS = ffroutines.rms_multibunch(profile_obj.n_macroparticles, profile_obj.bin_centers, nbf,nbs, maxbktl) #bucket_tolerance)
        #print('> Results...')
        #print(f'profile_obj.bunchPosition     = {profile_obj.bunchPosition}')
        #print(f'profile_obj_bunchPosition_RMS = {profile_obj_bunchPosition_RMS}')
        #quit()

        profile_obj.bin_centers += profile_cut_left_to_zero
        profile_obj.bunchPosition     += profile_cut_left_to_zero
        profile_obj_bunchPosition_RMS += profile_cut_left_to_zero
        # print('> Reverting margin in profile and displacement result by same margin...')
        # print(f'profile_obj.bin_centers       = [{profile_obj.bin_centers[0]},...]')

        # # print(f'profile_obj.bunchPosition     = [{profile_obj.bunchPosition[0]},...]')
        # # print(f'profile_obj_bunchPosition_RMS = [{profile_obj_bunchPosition_RMS[0]},...]')
        # print(f'profile_obj.bunchPosition     = {profile_obj.bunchPosition}')
        # print(f'profile_obj_bunchPosition_RMS = {profile_obj_bunchPosition_RMS}')

        # BUNCH POSITION OFFSETS:

        bunchlist = np.arange(nbf)
        #print(f'bunchlist = {bunchlist}')
        idxbunches = np.logical_not( np.logical_or(np.isnan(profile_obj.bunchPosition), np.isnan(profile_obj_bunchPosition_RMS)) )
        bunchlist = bunchlist[idxbunches]
        #print(f'idxbunches = {idxbunches}')
        #print(f'bunchlist = {bunchlist}')
        if(len(bunchlist) > 0):
            #print('[!] NaN bunches:', bunchlist[ np.logical_or(np.isnan(profile_obj.bunchPosition), np.isnan(profile_obj_bunchPosition_RMS)) ] )
            pass

        # w.r.t. bucket centres:

        profile_obj_bunchPositionOff_ctr_FWHM = profile_obj.bunchPosition     - bucket_centres
        profile_obj_bunchPositionOff_ctr_RMS  = profile_obj_bunchPosition_RMS - bucket_centres
        # print(f'profile_obj_bunchPositionOff_ctr_FWHM = {profile_obj_bunchPositionOff_ctr_FWHM/1e-12}ps, shape = {profile_obj_bunchPositionOff_ctr_FWHM.shape}')
        # print(f'profile_obj_bunchPositionOff_ctr_RMS  = {profile_obj_bunchPositionOff_ctr_RMS/1e-12}ps,  shape = {profile_obj_bunchPositionOff_ctr_RMS.shape}')

        # w.r.t. fit: New method for computation of mean_dtOff_fit (the same that the one beam_profiles, where other options where studied first)

        if nbf == 1:

            profile_obj_bunchPositionOff_fit_FWHM = np.array([ np.NaN ])
            profile_obj_bunchPositionOff_fit_RMS  = np.array([ np.NaN ])

        else:

            ## print('SciPy optimize')
            ###m0 = (bucket_centres[-1]-bucket_centres[0])/nbf  # bucket_centres is actually NOT needed, as these m0 and b0 are only used as initial guess for the fit in the new method of computation of mean_dtOff_fit. Moreover, we could compute very similar inital guesses from mean_dt
            ###b0 =  bucket_centres[0]
            m0 = (profile_obj.bunchPosition[-1]-profile_obj.bunchPosition[0])/nbf
            b0 =  profile_obj.bunchPosition[0]
            # print('> Guess for fit...')
            # print(f'bucket_centres = [{bucket_centres[0]},...]')
            # print(f'm0 = {m0}')
            # print(f'b0 = {b0}')

            fit_m,     fit_b     = optimize.curve_fit(linear_fit, np.arange(nbf), profile_obj.bunchPosition[idxbunches],     p0=[m0, b0])[0] # alternatively, we could add beampattern to the arguments of the fucntion, and use fillpattern instead of np.arange(nbf)
            fit_m_RMS, fit_b_RMS = optimize.curve_fit(linear_fit, np.arange(nbf), profile_obj_bunchPosition_RMS[idxbunches], p0=[m0, b0])[0] # (same)
            fit     = linear_fit(np.arange(nbf), fit_m,     fit_b)
            fit_RMS = linear_fit(np.arange(nbf), fit_m_RMS, fit_b_RMS)
            # print('> Results fit...')
            # print(f'fit_m,     fit_b     = {fit_m}, {fit_b}')
            # print(f'fit_m_RMS, fit_b_RMS = {fit_m_RMS}, {fit_b_RMS}')
            # print(f'fit     = {fit}')
            # print(f'fit_RMS = {fit_RMS}')
            # print('')

            profile_obj_bunchPositionOff_fit_FWHM = profile_obj.bunchPosition     - fit
            profile_obj_bunchPositionOff_fit_RMS  = profile_obj_bunchPosition_RMS - fit_RMS
            # print(f'profile_obj_bunchPositionOff_fit_FWHM = {profile_obj_bunchPositionOff_fit_FWHM/1e-12}ps, shape = {profile_obj_bunchPositionOff_fit_FWHM.shape}')
            # print(f'profile_obj_bunchPositionOff_fit_RMS  = {profile_obj_bunchPositionOff_fit_RMS/1e-12}ps,  shape = {profile_obj_bunchPositionOff_fit_RMS.shape}')

        # beam phase - RF:

        # Based on llrf/beam_feedback/ beam_phase and_phase difference:
        # * Beam phase measured at the main RF frequency and phase. The beam is
        # convolved with the window function of the band-pass filter of the
        # machine. The coefficients of sine and cosine components determine the
        # beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
        # phase is already w.r.t. the instantaneous RF phase. Then:
        # * Phase difference between beam and RF phase of the main RF system.

        # Main RF frequency at the present turn
        counter  = rfstation_obj.counter[0]
        omega_rf = rfstation_obj.omega_rf[0, counter] # single RF
        phi_rf   = rfstation_obj.phi_rf[  0, counter] # single RF
        phi_s    = rfstation_obj.phi_s[counter]
        dphi_rf  = rfstation_obj.dphi_rf[0]
        # print(f'counter = {counter}')
        # print(f'omega_rf = {omega_rf}')
        # print(f'phi_rf = {phi_rf}')
        # print(f'phi_s = {phi_s}')
        # print(f'print = {dphi_rf}')

        profile_obj_bunchPositionOff_brf = np.empty(nbf)

        # Fbsymmetric = False # For tests
        if not Fbsymmetric:
            # Half is OK (i.e. = 2 for nbs = 5) to resolve in frequency well
            # enough for the computation of the complex form factor:
            nbext = 0.5*(nbs-1)
            fr = 200.222e6
            # fr = 200.100e6
        profile_obj_bunchFormFactor_brf = np.empty(nbf)

        for i in range(nbf):

            # print(i)
            # if time_offset is None:
            indexes = idxP_BF_bunch[i]
            # print(f'indexes = {indexes}')

            profile_bin_centers      = profile_obj.bin_centers[indexes]
            profile_n_macroparticles = profile_obj.n_macroparticles[indexes]
            # print(f'profile_bin_centers = {profile_bin_centers}, shape = {profile_bin_centers.shape}')
            # print(f'profile_obj.bin_centers[indexes-profile_bin_centers[0] = {profile_bin_centers-profile_bin_centers[0]}, shape = (same)')
            # print(f'profile_n_macroparticles = {profile_n_macroparticles}, shape = {profile_n_macroparticles.shape}, sum = {np.sum(profile_n_macroparticles)}')
            # print(f'profile_obj.bin_size = {profile_obj.bin_size}')

            # Window cofficient is always zero (= single bunch). Time offset is
            # always zero (since we are already feeding the computation of the
            # beam phase with the extracted profile for  each single bunch. It
            # is actually not used as we have implemented the simplified
            # formula where this is the assumption (no time_offset):
            alpha = 0
            time_offset = 0

            coeff = bm.beam_phase(profile_bin_centers, #-profile_bin_centers[0],
                                  profile_n_macroparticles,
                                  alpha, omega_rf, phi_rf,
                                  profile_obj.bin_size)
            # print(f'coeff = {coeff}')

            # Project bunch phase to (pi/2,3pi/2) range
            phiOff = np.arctan(coeff) + np.pi
            # print(f'phiOff = {phiOff}rad = {phiOff/np.pi*180.}deg = {phiOff/2./np.pi*maxbktl/1e-12}ps')

            # Correct for design stable phase
            phiOff -= phi_s
            # print(f'phiOff = {phiOff}rad = {phiOff/np.pi*180.}deg = {phiOff/2./np.pi*maxbktl/1e-12}ps')

            profile_obj_bunchPositionOff_brf[i] = phiOff/2./np.pi*maxbktl

            # RELATIVE BUNCH FACTOR:

            if Fbsymmetric:

                # Method 1: Simplified for even-symmetric profile

                # Normalized profile: Formally, we have to take into account the
                # profile.bin_centers[i+1]-profile_obj.bin_centers[i] i.e.:
                # profile_n_macroparticles_norm = profile_n_macroparticles/np.sum( profile_n_macroparticles * profile_obj.bin_size )
                # Fbc = np.sum( profile_n_macroparticles_norm * np.cos(omega_rf*profile_bin_centers) ) * profile_obj.bin_size
                # (see below for FBc and Fbs), but since all bins have the same profile_obj.bin_size we simplify:
                profile_n_macroparticles_norm = profile_n_macroparticles/np.sum( profile_n_macroparticles )

                Fbc = np.sum( profile_n_macroparticles_norm * np.cos(omega_rf*profile_bin_centers + dphi_rf) )
                Fbs = np.sum( profile_n_macroparticles_norm * np.sin(omega_rf*profile_bin_centers + dphi_rf) )
                # print(f'Fbc = {Fbc}')
                # print(f'Fbs = {Fbs}')
                Fb  = np.sqrt( Fbc**2 + Fbs**2 )
                #print(f'Fb = {Fb}')

            else:

                # Method 2: General

                # Extended profile: Formally, nbext*len(profile_bin_centers)-1)
                # will go extend beyond the actual sparation (in time because
                # len(profile_bin_centers) include the margins, but it's fine
                # since we just want to add zeros to the bunch profile at both sides
                t_left  = np.linspace(profile_bin_centers[ 0]-(nbext*len(profile_bin_centers)-1)*profile_obj.bin_size, profile_bin_centers[ 0], int(nbext*len(profile_bin_centers)))
                t_right = np.linspace(profile_bin_centers[-1], profile_bin_centers[-1]+(nbext*len(profile_bin_centers)-1)*profile_obj.bin_size, int(nbext*len(profile_bin_centers)))
                delta_t_left  = t_left[1] -t_left[0]  if len(t_left)  > 0 else 0
                delta_t_right = t_right[1]-t_right[0] if len(t_right) > 0 else 0
                # print(f't_left = {t_left}, shape = {t_left.shape}, delta_t_left = {delta_t_left}')
                # print(f't_right = {t_right}, shape = {t_right.shape}, delta_t_right = {delta_t_right}')
                n_left_right  = np.zeros(int(nbext*len(profile_bin_centers)))
                # print(f'n_left_right = {n_left_right}, shape = {n_left_right.shape}')

                # Extend:
                profile_bin_centers_ext = np.hstack( (t_left-profile_obj.bin_size, profile_bin_centers))
                profile_bin_centers_ext = np.hstack( (profile_bin_centers_ext,     t_right+profile_obj.bin_size))
                profile_n_macroparticles_ext = np.hstack( (n_left_right,                 profile_n_macroparticles))
                profile_n_macroparticles_ext = np.hstack( (profile_n_macroparticles_ext, n_left_right))
                # print(f'profile_bin_centers_ext = {profile_bin_centers_ext}, shape = {profile_bin_centers_ext.shape}')
                # print(f'profile_n_macroparticles_ext = {profile_n_macroparticles_ext}, shape = {profile_n_macroparticles_ext.shape}, sum = {np.sum(profile_n_macroparticles_ext)}')

                npts = len(profile_n_macroparticles_ext)
                T = profile_bin_centers_ext[-1] - profile_bin_centers_ext[0]
                fres = 1./T
                f = np.arange(npts)*fres
                F = f[-1]
                nptsh = int(0.5*npts)
                # print(f'npts = {npts}, nptsh = {nptsh}')
                # print(f't = profile_bin_centers_ext = {profile_bin_centers_ext}, shape = {profile_bin_centers_ext.shape}')
                # print(f'T = {T/1e-9}ns, tres = profile_obj.bin_size = {profile_obj.bin_size/1e-9}ns')
                # print(f'f = {f}, shape = {f.shape}')
                # print(f'F = {F/1e6}MHz, fres = {fres/1e6}MHz')
                FT = np.fft.fft( profile_n_macroparticles_ext )
                # print(f'FT = {FT}, shape = {FT.shape}')
                FT_val = FT[:nptsh+1]/npts # complex
                FT_frq = f[:nptsh+1]
                # print(f'FT_val = {FT_val}, shape = {FT_val.shape}')
                # print(f'FT_frq = {FT_frq}, shape = {FT_frq.shape}')
                idx_fr = np.argmax( FT_frq >= fr )
                idx_f0  = 0 # = np.argmax( FT_frq >= 0 ))
                # print(f'idx_fr   = {idx_fr}, FT_frq[idx_fr] = {FT_frq[idx_fr]}, FT_val[idx_fr] = {FT_val[idx_fr]}')
                # print(f'idx_fr-1 = {idx_fr-1}, FT_frq[idx_fr-1] = {FT_frq[idx_fr-1]}, FT_val[idx_fr-1] = {FT_val[idx_fr-1]}')
                # print(f'idx_f0   = {idx_f0}, FT_frq[idx_f0] = {FT_frq[idx_f0]}, FT_val[idx_f0] = {FT_val[idx_f0]}')
                # With interpolation
                #FT_val_fr = (FT_val[idx_fr] - FT_val[idx_fr-1])/(FT_frq[idx_fr] - FT_frq[idx_fr-1])*(fr - FT_frq[idx_fr-1]) + FT_val[idx_fr-1]
                # Without interpolation
                FT_val_fr = FT_val[idx_fr]
                #
                FT_val_f0 = FT_val[idx_f0]
                Fb_complex = FT_val_fr/FT_val_f0 # Sometimes it is defined with a factor of 2
                Fb = np.absolute(Fb_complex)
                # print(f'FT_val_fr = {FT_val_fr}')
                # print(f'FT_val_f0 = {FT_val_f0}')
                # print(f'Fb_complex = {Fb_complex}')
                # print(f'Fb = {Fb}')

                # fig, ax = plt.subplots(1,3)
                # fig.set_size_inches(8.00*3, 6.00*1)
                # ax[0].plot(profile_bin_centers_ext/1e-9, profile_n_macroparticles_ext)
                # ax[1].plot(np.absolute(FT))
                # ax[2].plot(FT_frq/1e6, np.absolute(FT_val))
                # ax[0].set_xlabel('Time [ns]')
                # ax[1].set_xlabel('Samples')
                # ax[2].set_xlabel('Frequency [MHz]')
                # ax[0].set_ylabel('Bunch profile $\lambda(t)$ [macroparticles]')
                # ax[1].set_ylabel('$|S(f)| = |F[\lambda(t)]|$')
                # ax[2].set_ylabel('$|S(f)| = |F[\lambda(t)]|/n$')
                # fig.tight_layout()
                # fig.savefig('tmp.png')
                # plt.cla()
                # plt.close(fig)
                # quit()

                #

            profile_obj_bunchFormFactor_brf[i] = Fb

        # print(f'profile_obj_bunchPositionOff_brf = {profile_obj_bunchPositionOff_brf/1e-12}ps, shape = {profile_obj_bunchPositionOff_brf.shape}')
        # print(f'profile_obj_bunchPositionOff_brf -mean(profile_obj_bunchPositionOff_brf) = {(profile_obj_bunchPositionOff_brf-profile_obj_bunchPositionOff_brf.mean())/1e-12}ps, shape = {profile_obj_bunchPositionOff_brf.shape}')

        # print(f'profile_obj_bunchFormFactor_brf = {profile_obj_bunchFormFactor_brf}, shape = {profile_obj_bunchFormFactor_brf.shape}')
        # quit()

        #

        # Emittances:

        energy   = rfstation_obj.energy[counter]
        energy_k = energy - rfstation_obj.Particle.mass
        omega_s0 = rfstation_obj.omega_s0[counter]
        beta     = rfstation_obj.beta[counter]
        beta_sq  = beta**2
        eta_0    = rfstation_obj.eta_0[counter]
        voltage  = rfstation_obj.voltage[0,counter] # single RF
        harmonic = rfstation_obj.harmonic[0,counter] # single RF

        profile_obj_bunchEnergySpread_FWHM = np.empty(nbf)
        profile_obj_bunchEnergySpread_RMS  = np.empty(nbf)

        profile_obj_bunchEmittance_FWHM = np.empty(nbf)
        profile_obj_bunchEmittance_RMS  = np.empty(nbf)

        for i in range(nbf):

            #print(i)
            # Based on emittance_from_bunch_length (bunch lengths: 4-sigma
            # from FWHM bunch length and RMS bunch length [s]):

            if nbf == 1:
                #print(f'profile_obj.bunchLength     = {profile_obj.bunchLength/1e-9} ns')
                #print(f'profile_obj_bunchLength_RMS = {profile_obj_bunchLength_RMS/1e-9} ns')
                # Emittance contour in phase [rad]
                phi_b     = omega_rf * 0.5*profile_obj.bunchLength
                phi_b_RMS = omega_rf * 0.5*profile_obj_bunchLength_RMS

            else:
                #print(f'profile_obj.bunchLength[i]     = {profile_obj.bunchLength[i]/1e-9} ns')
                #print(f'profile_obj_bunchLength_RMS[i] = {profile_obj_bunchLength_RMS[i]/1e-9} ns')
                # Emittance contour in phase [rad]
                phi_b     = omega_rf * 0.5*profile_obj.bunchLength[i]
                phi_b_RMS = omega_rf * 0.5*profile_obj_bunchLength_RMS[i]

            # print(f'phi_b     = {phi_b} rad')
            # print(f'phi_b_RMS = {phi_b_RMS} rad')

            # Emittance contour in energy offset [eV]: \Delta E_b
            dE_b     = np.sqrt(beta_sq * energy * voltage*(1 - np.cos(phi_b    )) / (np.pi*harmonic*eta_0))
            dE_b_RMS = np.sqrt(beta_sq * energy * voltage*(1 - np.cos(phi_b_RMS)) / (np.pi*harmonic*eta_0))

            # print(f'dE_b     = {dE_b/1e6} MeV')
            # print(f'dE_b_RMS = {dE_b_RMS/1e6} MeV')
            # Emittance contour in relative momentum [1]: Delta d_p
            delta_b     = dE_b    /(beta_sq * energy)
            delta_b_RMS = dE_b_RMS/(beta_sq * energy)
            # print(f'delta_b     = {delta_b}')
            # print(f'delta_b_RMS = {delta_b_RMS}')
            #  RMS energy spread [MeV]: # [1] (we don't want it relative)
            sigmaE     = np.str(0.5*dE_b) #    /energy_k)
            sigmaE_RMS = np.str(0.5*dE_b_RMS) #/energy_k)
            # print(f'sigmaE     = {sigmaE}')
            # print(f'sigmaE_RMS = {sigmaE_RMS}')

            profile_obj_bunchEnergySpread_FWHM[i] = sigmaE
            profile_obj_bunchEnergySpread_RMS[i]  = sigmaE_RMS

            # Emittance:
            filterwarnings("ignore")
            integral     = integrate.quad(lambda x: np.sqrt(2.*(np.cos(x) -  np.cos(phi_b    ))), 0, phi_b    )[0]
            integral_RMS = integrate.quad(lambda x: np.sqrt(2.*(np.cos(x) -  np.cos(phi_b_RMS))), 0, phi_b_RMS)[0]
            filterwarnings("default")
            emit     = 4. * energy * omega_s0 * beta_sq * integral     / (omega_rf**2 * eta_0)
            emit_RMS = 4. * energy * omega_s0 * beta_sq * integral_RMS / (omega_rf**2 * eta_0)
            # print(f'emit     = 4*{emit/4.} eVs = {emit} eVs')
            # print(f'emit_RMS = 4*{emit_RMS/4.} eVs = {emit_RMS} eVs')

            profile_obj_bunchEmittance_FWHM[i] = emit
            profile_obj_bunchEmittance_RMS[i]  = emit_RMS

        #

        gc.collect()

        result = {'FWHM': {'bunchPosition':        profile_obj.bunchPosition,
                           'bunchLength':          profile_obj.bunchLength,
                           'bunchPositionOff_ctr': profile_obj_bunchPositionOff_ctr_FWHM,
                           'bunchPositionOff_fit': profile_obj_bunchPositionOff_fit_FWHM,
                           'bunchEnergySpread':    profile_obj_bunchEnergySpread_FWHM,
                           'bunchEmittance':       profile_obj_bunchEmittance_FWHM,
                           },
                  'RMS':  {'bunchPosition':        profile_obj_bunchPosition_RMS,
                           'bunchLength':          profile_obj_bunchLength_RMS,
                           'bunchPositionOff_ctr': profile_obj_bunchPositionOff_ctr_RMS,
                           'bunchPositionOff_fit': profile_obj_bunchPositionOff_fit_RMS,
                           'bunchEnergySpread':    profile_obj_bunchEnergySpread_RMS,
                           'bunchEmittance':       profile_obj_bunchEmittance_RMS,
                           },
                  'BRF':  {'bunchPositionOff':     profile_obj_bunchPositionOff_brf,
                           'bunchFormFactor':      profile_obj_bunchFormFactor_brf
                           }
                  }
        return result

    def stats_beam(profile_stats_bunch):

        result = {}
        for opt in profile_stats_bunch.keys():
            result[opt] = {}
            for key in profile_stats_bunch[opt].keys():
                result[opt][key] = np.mean(profile_stats_bunch[opt][key])
        gc.collect()
        return result


###############################################################################

class ProfilePattern(object):

    def __init__(self, profile, Ns, Nsmargfrac, beampattern_obj):

        satellite_threshold = 25e-9 # [s]
        nbf = beampattern_obj.nbf

        # Total no. of samples and per bucket
        self.Ns     = Ns
        self.Ns_tot = profile.n_slices
        #print(f'self.Ns_tot = {self.Ns_tot}')
        #print(f'self.Ns = {self.Ns}')

        # Very fine sampling (to be used in separatrix computation)
        self.Nsfinefactor = 8
        self.Nsfine     = self.Nsfinefactor*self.Ns
        self.Nsfine_tot = self.Nsfinefactor*self.Ns_tot

        # No. of samples for filled bucket margins
        self.Nsmargfrac = Nsmargfrac
        self.Nsmargin   = int(Nsmargfrac*self.Ns)
        #print(f'self.Nsmargfrac = {self.Nsmargfrac}')
        #print(f'self.Nsmargin = {self.Nsmargin}')

        # Windows around the bucket centre to assign sample indices as belonging to filled buckets, or satellite buckets
        self.maxbklt = int(2*self.Nsmargin) + self.Ns  # Full bucket size (in samples) including margins
        self.sathwin = int(satellite_threshold/profile.bin_size)   # 25ns window ("half"-window, full window is 50ns centered around the bucket centre) to be applied left and right of the bucket centres to define indices corresponding to satellite buckets.
        #print(f'self.maxbklt = {self.maxbklt}')
        #print(f'self.sathwin = {self.sathwin}')

        # Sample index marking corresponding to the bucket centre
        self.bucket_centres = self.Ns*(beampattern_obj.fillpattern+0.5)
        self.bucket_centres = self.bucket_centres.astype(int)
        #print(f'bucket_centres = {self.bucket_centres}')

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # Samples corresponding to each FILLED bucket:

        # Per main bunch:
        if self.Ns % 2 == 0: istart_list = self.bucket_centres - int((self.Ns  )/2) - self.Nsmargin # Even no. of slices per bunch (then the bucket centre is half-bucket in front of the real centre: e.g. 6: m m o o o x o o m m )
        else:                istart_list = self.bucket_centres - int((self.Ns-1)/2) - self.Nsmargin # Odd  no. of slices per bunch (then the bucket centre is *really* at the centre:                  e.g. 5: m m o o x o o m m )
        #print(f'istart_list = {istart_list}')
        idx_BF_bunch = []
        for i in range(nbf):
            istart = istart_list[i]
            iend   = istart_list[i] + self.maxbklt
            #print(istart_list[i] + self.maxbklt)
            if(istart < 0):          istart = 0 # Check that we don't start below the zero index (e.g. when no margin was requested)
            if(iend >= self.Ns_tot): iend   = self.Ns_tot # Check that we don't go above the max. no. of indices (e.g. when no margin was requested)
            idx_BF_bunch.append( np.arange(istart, iend ).astype('int32') )
        idx_BF_bunch = np.array(idx_BF_bunch)
        #print(f'idx_BF_bunch = {idx_BF_bunch}, shape = {idx_BF_bunch.shape}, dtype = {idx_BF_bunch.dtype}')

        # Per beam
        idx_BF_beam = np.hstack(idx_BF_bunch) #.astype('int32')
        idx_BF_beam = list(set(idx_BF_beam)) # Remove possible overlaps
        idx_BF_beam.sort()
        idx_BF_beam = np.array(idx_BF_beam)
        #print(f'idx_BF_beam = {idx_BF_beam}, shape = {idx_BF_beam.shape}, dtype = {idx_BF_beam.dtype}')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # For some application, we might want the samples of filled bunches
        # WITHOUT the margin (for example, to calculate the average of a given
        # parameter signal like rf beam current in the beam segment) -- necessary?

        # Per main bunch:
        if self.Ns % 2 == 0: istart_list = self.bucket_centres - int((self.Ns  )/2) - self.Nsmargin # Even no. of slices per bunch (then the bucket centre is half-bucket in front of the real centre: e.g. 6: m m o o o x o o m m )
        else:                istart_list = self.bucket_centres - int((self.Ns-1)/2) - self.Nsmargin # Odd  no. of slices per bunch (then the bucket centre is *really* at the centre:                  e.g. 5: m m o o x o o m m )
        #print(f'istart_list = {istart_list}')
        idx_nomargin_BF_bunch = []
        for i in range(nbf):
            istart = istart_list[i]
            iend   = istart_list[i] + self.maxbklt
            #print(istart_list[i] + self.maxbklt)
            if(istart < 0):          istart = 0 # Check that we don't start below the zero index (e.g. when no margin was requested)
            if(iend >= self.Ns_tot): iend   = self.Ns_tot # Check that we don't go above the max. no. of indices (e.g. when no margin was requested)
            idx_nomargin_BF_bunch.append( np.arange(istart, iend ).astype('int32') )
        idx_nomargin_BF_bunch = np.array(idx_nomargin_BF_bunch)
        #print(f'idx_nomargin_BF_bunch = {idx_nomargin_BF_bunch}, shape = {idx_nomargin_BF_bunch.shape}, dtype = {idx_nomargin_BF_bunch.dtype}')

        # Per beam
        idx_nomargin_BF_beam = np.hstack(idx_nomargin_BF_bunch) #.astype('int32')
        idx_nomargin_BF_beam = list(set(idx_nomargin_BF_beam)) # Remove possible overlaps
        idx_nomargin_BF_beam.sort()
        idx_nomargin_BF_beam = np.array(idx_nomargin_BF_beam)
        #print(f'idx_nomargin_BF_beam = {idx_nomargin_BF_beam}, shape = {idx_nomargin_BF_beam.shape}, dtype = {idx_nomargin_BF_beam.dtype}')

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # Samples corresponding to SATELLITE buckets:

        # Per main bunch:
        if self.Ns % 2 == 0: istart_list = self.bucket_centres   - self.sathwin
        else:                istart_list = self.bucket_centres-1 - self.sathwin
        #print(f'istart_list = {istart_list}')
        idx_BS_bunch = []
        for i in range(nbf):
            istart = istart_list[i]
            iend   = istart_list[i] + int(2*self.sathwin)
            #print(istart_list[i] + self.maxbklt)
            if(istart < 0):          istart = 0 # Check that we don't start below the zero index (e.g. when no margin was requested)
            if(iend >= self.Ns_tot): iend   = self.Ns_tot # Check that we don't go above the max. no. of indices (e.g. when no margin was requested)
            idx_BS_bunch.append( np.arange(istart, iend ).astype('int32') )
        # If overlap between the satellites associated to consecutve filled buckets, split in the middle
        for i in range(len(idx_BS_bunch) - 1):
            if idx_BS_bunch[i][-1] >= idx_BS_bunch[i+1][0]:
                middlepoint = int(0.5*(idx_BS_bunch[i][-1]+idx_BS_bunch[i+1][0]))
                ##print(i)
                ##print(f'idx_BS_bunch[i  ] = {idx_BS_bunch[i  ]}')
                ##print(f'idx_BS_bunch[i+1] = {idx_BS_bunch[i+1]}')
                ##print(middlepoint)
                idx_BS_bunch[i  ] = np.arange(idx_BS_bunch[i][0], middlepoint+1)
                idx_BS_bunch[i+1] = np.arange(middlepoint+1,      idx_BS_bunch[i+1][-1]+1)
                ##print(f'idx_BS_bunch[i  ] = {idx_BS_bunch[i  ]}')
                ##print(f'idx_BS_bunch[i+1] = {idx_BS_bunch[i+1]}')
                ##print(')
        ##print(f'idx_BS_bunch = {idx_BS_bunch}')
        # Remove indices of the corresponding filled buckets:
        for i in range(len(idx_BS_bunch)):
            idx_BS_bunch[i] = np.setdiff1d( idx_BS_bunch[i], idx_BF_bunch[i] ) # Return the elements in idx_BS_bunch[i] not in idx_BF_bunch[i]
        ##print(f'idx_BS_bunch = {idx_BS_bunch}')
        # If the 50ns window corresponding satellites extends so far that it actually reaches nearby filled buckets, remove those samples
        for i in range(len(idx_BS_bunch)):
            jlist = list(range(nbf))
            jlist.pop(jlist.index(i))
            ##print(i, jlist)
            for j in jlist:
                idx_BS_bunch[i] = np.setdiff1d( idx_BS_bunch[i], idx_BF_bunch[j] ).astype('int32') # Return the elements in idx_BS_bunch[i] not in idx_BF_bunch[j]  # np.copy(idx_BS_bunch_i_tmp)
                ##print(f'{i} {j} idx_BS_bunch[i] = {idx_BS_bunch[i]}')
        idx_BS_bunch = np.array(idx_BS_bunch)
        #print(f'idx_BS_bunch = {idx_BS_bunch}, shape = {idx_BS_bunch.shape}, dtype = {idx_BS_bunch.dtype}')

        # Per beam:
        idx_BS_beam = np.hstack(idx_BS_bunch) #.astype('int32')
        idx_BS_beam = list(set(idx_BS_beam)) # Remove possible overlaps
        idx_BS_beam.sort()
        idx_BS_beam = np.array(idx_BS_beam)
        #print(f'idx_BS_beam = {idx_BS_beam}, shape = {idx_BS_beam.shape}, dtype = {idx_BS_beam.dtype}')

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # Samples corresponding to GHOST buckets:

        # Per main bunch:
        idx_BG_bunch = []
        for i in range(nbf):
            if(i == 0): istart = 0
            else:       istart = int(0.5*(self.bucket_centres[i-1] + self.bucket_centres[i]))
            if(i == nbf-1): iend = self.Ns_tot
            else:           iend = int(0.5*(self.bucket_centres[i] + self.bucket_centres[i+1]))
            #print(i, self.bucket_centres[i], istart, iend)
            idx_BG_bunch.append( np.arange(istart, iend).astype('int32') )
        #print(f'idx_BG_bunch = {idx_BG_bunch}')
        # Remove indices of the corresponding satellite buckets:
        for i in range(len(idx_BG_bunch)):
            idx_BG_bunch[i] = np.setdiff1d( idx_BG_bunch[i], idx_BS_bunch[i] ) # Return the elements in idx_BG_bunch[i] not in idx_BS_bunch[i]
        ##print(f'idx_BG_bunch = {idx_BG_bunch}')
        # # If the window corresponding ghost extends so far that it actually reaches nearby satellite buckets, remove those samples
        # for i in range(len(idx_BG_bunch)):
        #     jlist = list(range(len(idx_BG_bunch)))
        #     jlist.pop(jlist.index(i))
        #     ##print(i, jlist)
        #     for j in jlist:
        #         idx_BG_bunch[i] = np.setdiff1d( idx_BG_bunch[i], idx_BS_bunch[j] ) # Return the elements in idx_BG_bunch[i] not in idx_BS_bunch[j]  # np.copy(idx_BG_bunch_i_tmp)
        #         #print(f'{i} {j} idx_BG_bunch[i] = {idx_BG_bunch[i]}')
        # ##print(f'idx_BG_bunch = {idx_BG_bunch}')
        # Remove indices of the corresponding filled buckets:
        for i in range(len(idx_BG_bunch)):
            idx_BG_bunch[i] = np.setdiff1d( idx_BG_bunch[i], idx_BF_bunch[i] ) # Return the elements in idx_BG_bunch[i] not in idx_BF_bunch[i]
        ##print(f'idx_BG_bunch = {idx_BG_bunch}')
        # # If the window corresponding ghost extends so far that it actually reaches nearby filled buckets, remove those samples
        # for i in range(len(idx_BG_bunch)):
        #     jlist = list(range(len(idx_BG_bunch)))
        #     jlist.pop(jlist.index(i))
        #     ##print(i, jlist)
        #     for j in jlist:
        #         idx_BG_bunch[i] = np.setdiff1d( idx_BG_bunch[i], idx_BF_bunch[j] ).astype('int32') # Return the elements in idx_BG_bunch[i] not in idx_BF_bunch[j]  # np.copy(idx_BG_bunch_i_tmp)
        #         ##print(f'{i} {j} idx_BG_bunch[i] = {idx_BG_bunch[i]}')
        idx_BG_bunch = np.array(idx_BG_bunch) #.astype('int32')
        ##print(f'idx_BG_bunch = {idx_BG_bunch}, shape = {idx_BG_bunch.shape}, dtype = {idx_BG_bunch.dtype}')

        # Per beam:
        idx_BG_beam = np.hstack(idx_BG_bunch) #.astype('int32')
        idx_BG_beam = list(set(idx_BG_beam)) # Remove possible overlaps
        idx_BG_beam.sort()
        idx_BG_beam = np.array(idx_BG_beam)
        ##print(f'idx_BG_beam = {idx_BG_beam}, shape = {idx_BG_beam.shape}, dtype = {idx_BG_beam.dtype}')

        self.idxP = {'BF': {'bunch': idx_BF_bunch,
                            'beam':  idx_BF_beam},
                     'BS': {'bunch': idx_BS_bunch,
                            'beam':  idx_BS_beam},
                     'BG': {'bunch': idx_BG_bunch,
                            'beam':  idx_BG_beam} }

        # Without sample margins
        self.idxP_nomargin = {'BF': {'bunch': idx_nomargin_BF_bunch,
                                     'beam':  idx_nomargin_BF_beam} }

#     # idxP (for BF/BS, per bunch/beam) ========================================

#     def dict_idxP(self, beampattern_obj):
#         ''' INDEX OF SAMPLES IN FULL PROFILE CORRESPONDING TO FILLED/SATELLITE/GHOST BUCKETS '''

#         result = {'BF': {}, 'BS': {}, 'BG': {}}

#         # Filled buckets (per bunch)
#         result['BF']['bunch'] = np.array([np.arange(i*self.Ns-self.Nsmargin, (i+1)*self.Ns+self.Nsmargin) for i in beampattern_obj.fillpattern]).astype(int)
#         #
#         # Separation buckets (per "bunch", i.e. per segmend of nbs-1 in between the filled buckets, and to the left and right of the first and last filled bucket)
#         idxP_BS_bunch = [np.arange(0, (0+1)*self.Ns-self.Nsmargin)]
#         for i in beampattern_obj.fillpattern[:-1]:
#             #print(i, i+1, i+nbs)
#             idxP_BS_bunch.append(np.arange((i+1)*self.Ns+self.Nsmargin, (i+beampattern_obj.nbs)*self.Ns-self.Nsmargin))
#         idxP_BS_bunch.append(np.arange((beampattern_obj.fillpattern[-1]+1)*self.Ns+self.Nsmargin, self.Ns_tot))
#         result['BS']['bunch'] = np.array(idxP_BS_bunch) #.astype(int)



#         # Per bunch (all bunckets between filled bunches are actually countes as a "bunch") - array has +1 "bunch" than nbf

#         #
#         # Per beam
#         result['BS']['beam'] = np.array([i for i in np.arange(self.Ns_tot) if i not in result['BF']['beam']]).astype(int)
#         result['BF']['beam'] = np.array([item for sublist in result['BF']['bunch'] for item in sublist]).astype(int)

#         gc.collect()

#         return result


# ###############################################################################
