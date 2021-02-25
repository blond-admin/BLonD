#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import gc
import numpy as np
from scipy import optimize
#from scipy.stats import linregress
#import time

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

###############################################################################

def linear_fit(x, m, b):
    return m * x + b
        
class BeamTools(object):

    # No. of macroparticles ===================================================

    def nmacrop_bunch(beam_obj, idxB):

        nbf = len(idxB)
        alive_array = np.array([]).astype(int)
        lost_array  = np.array([]).astype(int)
        for i in range(nbf):
            beam_id_array_i = beam_obj.id[idxB[i]]
            alive_array = np.append(alive_array, len(np.where(beam_id_array_i!=0)[0]))
            lost_array  = np.append(lost_array,  len(np.where(beam_id_array_i==0)[0]))
        gc.collect()
        return {'alive': alive_array.astype(int),
                'lost':  lost_array.astype(int)}

    def nmacrop_beam(beam_nmacrop_bunch_OR_beam_obj):

        if(isinstance(beam_nmacrop_bunch_OR_beam_obj, dict)):
            result = {}
            for key in beam_nmacrop_bunch_OR_beam_obj.keys():
                result[key] = np.sum(beam_nmacrop_bunch_OR_beam_obj[key])
        else:
            result = {'alive': int(beam_nmacrop_bunch_OR_beam_obj.n_macroparticles_alive),
                      'lost':  int(beam_nmacrop_bunch_OR_beam_obj.n_macroparticles_lost)}
        gc.collect()
        return result

    # Statistics (mean, rms) ==================================================

    def stats_bunch(beam_obj, idxB, bucket_centres): #, return_dict=False):
        '''
        Calculation of the bunch length (RMS of bunch time), bunch energy spread
        (RMS of bunch energy), bunch position in time (MEAN of bunch time), bunch
        position in energy (MEAN of bunch energy), bunch emittance (numerical
        factor times product of RMSs), and bunch time position offsets (from
        bucket centres, fit, and beam phase - RF)
        

        Input:
        - beam_obj
        - idxB
        - bucket_centres

        Returns:
        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        - epsnrmsl
        - mean_dtOff_ctr (w.r.t. bucket centres (single RF, no impedance model, but takes into account beam phase shift due to beam feedback))
        - mean_dtOff_fit (w.r.t. batch fit evaluated at the corresponding bunch no., returns [NaN] for single bunch)
        '''

        nbf = len(idxB)

        mean_dt_array  = np.array([])
        mean_dE_array  = np.array([])
        sigma_dt_array = np.array([])
        sigma_dE_array = np.array([])
        epsnrmsl_array = np.array([])

        for i in range(nbf):

            beam_dt_array_i = beam_obj.dt[idxB[i]]
            beam_dE_array_i = beam_obj.dE[idxB[i]]
            beam_id_array_i = beam_obj.id[idxB[i]]

            # Statistics only for particles that are not flagged as lost. For bunch length, multiply by 4 to compare with RMS bunch length from profile
            itemindex  = np.where(beam_id_array_i != 0)[0]
            mean_dt_i  = np.mean( beam_dt_array_i[itemindex])
            mean_dE_i  = np.mean( beam_dE_array_i[itemindex])
            sigma_dt_i = np.std(  beam_dt_array_i[itemindex])
            sigma_dE_i = np.std(  beam_dE_array_i[itemindex])

            # RMS emittance in Gaussian approximation (multiply by 4 to compare with RMS emittance from profile)
            epsnrmsl_i = np.pi*sigma_dE_i*sigma_dt_i # in eVs

            mean_dt_array  = np.append(mean_dt_array,  mean_dt_i )
            mean_dE_array  = np.append(mean_dE_array,  mean_dE_i )
            sigma_dt_array = np.append(sigma_dt_array, sigma_dt_i)
            sigma_dE_array = np.append(sigma_dE_array, sigma_dE_i)
            epsnrmsl_array = np.append(epsnrmsl_array, epsnrmsl_i)

        # BUNCH POSITION OFFSETS:
        
        # w.r.t. bucket centres:
            
        mean_dtOff_ctr_array = mean_dt_array - bucket_centres
        # print(f'mean_dtOff_ctr_array = {mean_dtOff_ctr_array/1e-12}ps, shape = {mean_dtOff_ctr_array.shape}')
       
        # w.r.t. fit: New method for computation of mean_dtOff_fit (other options to perform the linear fit above, but this one is efficient)
        
        if nbf == 1:
        
            mean_dtOff_fit_array = np.array([ np.NaN ])
            
        else:
        
            # # print('SciPy stats.linregress')
            # # proctime0 = time.process_time()
            # # slope, intercept, r_value, p_value, std_err = linregress(np.arange(nbf), mean_dt_array)
            # # print(f'slope = {slope}, intercept = {intercept}, r_value = {r_value}, p_value = {p_value}, std_err = {std_err}')
            # # print(time.process_time() - proctime0)
            # # print('')
            
            # # print('NumPy linalg.lstsq')
            # # proctime0 = time.process_time()
            # # A = np.vstack([np.arange(nbf), np.ones(nbf)*(mean_dt_array[-1]-mean_dt_array[0])/nbf ]).T
            # # m, c = np.linalg.lstsq(A, mean_dt_array, rcond=None)[0]
            # # print(time.process_time() - proctime0)
            # # print(f'A = {A}, m = {m}, c = {c}')
            # # print('')

            # # print('NumPy polyfit')    
            # # proctime0 = time.process_time()
            # # p = np.polyfit(np.arange(nbf), mean_dt_array, 1)
            # # z = np.poly1d(p)
            # # print(time.process_time() - proctime0)
            # # print(f'p = {p}, z = {z}')
            # # print('')
            
            # # print('SciPy optimize') 
            # # proctime0 = time.process_time()
            m0 = (bucket_centres[-1]-bucket_centres[0])/nbf  # bucket_centres is actually NOT needed, as these m0 and b0 are only used as initial guess for the fit in the new method of computation of mean_dtOff_fit. Moreover, we could compute very similar inital guesses from mean_dt
            b0 = bucket_centres[0]
            fit_m, fit_b = optimize.curve_fit(linear_fit, np.arange(nbf), mean_dt_array, p0=[m0, b0])[0] # alternatively, we could add beampattern to the arguments of the fucntion, and use fillpattern instead of np.arange(nbf) 
            fit = linear_fit(np.arange(nbf), fit_m, fit_b)
            
            mean_dtOff_fit_array = mean_dt_array - fit
            # print(f'mean_dtOff_fit_array = {mean_dtOff_fit_array/1e-12}ps, shape = {mean_dtOff_fit_array.shape}')
            
        # Plot and checking...
        
        # mean_dtOff_ctr_array_ave = np.mean(mean_dtOff_ctr_array)
        # mean_dtOff_fit_array_ave = np.mean(mean_dtOff_fit_array) # Formally, this is ~0, and therefore there's should not be need of substracting afterwards...
        # mean_dtOff_ctr_array_shift = mean_dtOff_ctr_array - mean_dtOff_ctr_array_ave
        # mean_dtOff_fit_array_shift = mean_dtOff_fit_array - mean_dtOff_fit_array_ave 
        # diff       = mean_dtOff_fit_array       - mean_dtOff_ctr_array
        # diff_shift = mean_dtOff_fit_array_shift - mean_dtOff_ctr_array_shift
        # # print(time.process_time() - proctime0)
        # print('')
        # print(f'mean_dt_array = {mean_dt_array/1e-9} ns')
        # print('')
        # print(f'bucket_centres = {bucket_centres/1e-9} ns')
        # print(f'fit = {fit/1e-9} ns, (fit_m = {fit_m}, fit_b = {fit_b})')
        # print('')
        # print(f'mean_dtOff_ctr_array = {mean_dtOff_ctr_array/1e-9} ns')
        # print(f'mean_dtOff_fit_array = {mean_dtOff_fit_array/1e-9} ns')
        # print('')
        # print(f'mean_dtOff_ctr_array_ave = {mean_dtOff_ctr_array_ave/1e-9} ns')
        # print(f'mean_dtOff_fit_array_ave = {mean_dtOff_fit_array_ave/1e-9} ns')
        # print('')
        # print(f'mean_dtOff_ctr_array_shift = {mean_dtOff_ctr_array_shift/1e-9} ns')
        # print(f'mean_dtOff_fit_array_shift = {mean_dtOff_fit_array_shift/1e-9} ns')
        # print('')
        # print(f'diff = {diff}')
        # print(f'diff_shift = {diff_shift}')
        # print('')
        
        # fig, ax = plt.subplots(4, 1, sharex=True)
        # fig.set_size_inches(7.5, 10.0)
        # #
        # ax[0].grid()
        # ax[0].set_ylabel('Reference [ns]')
        # ax[0].plot(np.arange(nbf), bucket_centres/1e-9, 'x--', label=f'bucket_centres ("orig."), m = {m0/1e-9:.3f} ns/bunch (const.)')
        # ax[0].plot(np.arange(nbf), fit/1e-9,            '.:',  label=f'fit ("alt."), m = {fit_m/1e-9:.3f} ns/bunch')
        # ax[0].legend(loc=2)
        # #
        # ax[1].grid()
        # ax[1].set_ylabel('Bunch position offset\n[ns]')
        # l10 = ax[1].plot(np.arange(nbf), mean_dtOff_ctr_array/1e-9, 'x--', label='mean_dtOff_ctr_array')
        # l11 = ax[1].plot(np.arange(nbf), mean_dtOff_fit_array/1e-9, '.:',  label='mean_dtOff_fit_array')
        # ax[1].axhline(mean_dtOff_ctr_array_ave/1e-9, ls='--', color=l10[0].get_color(), alpha=0.50) #, label='mean_dtOff_ctr_array_ave')
        # ax[1].axhline(mean_dtOff_fit_array_ave/1e-9, ls=':',  color=l11[0].get_color(), alpha=0.75) #, label='mean_dtOff_fit_array_ave')
        # ax[1].legend(loc=1)
        # #
        # ax[2].grid()
        # ax[2].set_ylabel('Bunch position offset\n(shift. by beam-ave.) [ps]')
        # ax[2].plot(np.arange(nbf), mean_dtOff_ctr_array_shift/1e-12, 'x--', label='mean_dtOff_ctr_array_shift')
        # ax[2].plot(np.arange(nbf), mean_dtOff_fit_array_shift/1e-12, '.:',  label='mean_dtOff_fit_array_shift')
        # ax[2].legend(loc=1)
        # #
        # ax[3].grid()
        # ax[3].set_ylabel('Difference\nalt. - orig. [ps]')
        # ax[3].plot(np.arange(nbf), diff/1e-12,       'x--', label='diff')
        # ax[3].plot(np.arange(nbf), diff_shift/1e-12, '.:',  label='diff_shift')
        # ax[3].legend(loc=4)
        # ax[3].set_xlabel('Bunch no.')
        # #
        # fig.tight_layout()
        # fig.savefig(f'tmp.png')
        # plt.cla()
        # plt.close(fig)
        
        gc.collect()
        
        return {'mean_dt':    mean_dt_array,
                'mean_dE':    mean_dE_array,
                'sigma_dt':   sigma_dt_array,
                'sigma_dE':   sigma_dE_array,
                'epsnrmsl':   epsnrmsl_array,
                'mean_dtOff_ctr': mean_dtOff_ctr_array,
                'mean_dtOff_fit': mean_dtOff_fit_array,
                }

    def stats_beam(beam_stats_bunch):

        result = {}
        for key in beam_stats_bunch.keys():
            result[key] = np.mean(beam_stats_bunch[key])
        gc.collect()
        return result

    # Losses (time cuts) ======================================================

    def losses_dt_cut_bunch(beam_obj, idxB, buncket_centres, LR, dt_0):
        '''Beam losses based on longitudinal cuts (multibunch)'''

        nbf = len(idxB)

        if(LR == 'L'):

            for i in range(nbf):
                beam_dt_array_i = beam_obj.dt[idxB[i]]
                itemindex = np.where( ((beam_dt_array_i-buncket_centres[i]) - dt_0 ) < 0 )[0]
                if itemindex.size != 0:
                    beam_obj.id[ itemindex+idxB[i][0] ] = 0
        else:

            for i in range(nbf):
                beam_dt_array_i = beam_obj.dt[idxB[i]]
                itemindex = np.where( (dt_0 - (beam_dt_array_i-buncket_centres[i]) ) < 0 )[0]
                if itemindex.size != 0:
                    beam_obj.id[ itemindex+idxB[i][0] ] = 0

        gc.collect()

    # Losses (energy cuts) ====================================================

    def losses_dE_cut_bunch(beam_obj, idxB, buncket_centres, DU, dE_0):
        '''Beam losses based on energy cuts, e.g. on collimators (multibunch)'''

        nbf = len(idxB)

        if(DU == 'D'):

            for i in range(nbf):
               #beam_dt_array_i = beam_obj.dt[idxB[i]]
                beam_dE_array_i = beam_obj.dE[idxB[i]]
               #beam_id_array_i = beam_obj.id[idxB[i]]
                itemindex = np.where( ((beam_dE_array_i-buncket_centres[i]) - dE_0) < 0 )[0]
                if itemindex.size != 0:
                    beam_obj.id[ itemindex+idxB[i][0] ] = 0
        else:

            for i in range(nbf):
               #beam_dt_array_i = beam_obj.dt[idxB[i]]
                beam_dE_array_i = beam_obj.dE[idxB[i]]
               #beam_id_array_i = beam_obj.id[idxB[i]]
                itemindex = np.where( (dE_0 - (beam_dE_array_i-buncket_centres[i])) < 0 )[0]
                if itemindex.size != 0:
                    beam_obj.id[ itemindex+idxB[i][0] ] = 0

        gc.collect()

    # def losses_longitudinal_cut_bunch(beam_obj, idxB, buncket_centres, dt_min, dt_max):
    #     '''Beam losses based on longitudinal cuts (multibunch)'''

    #     nbf = len(idxB)

    #     for i in range(nbf):
    #         beam_dt_array_i = beam_obj.dt[idxB[i]]
    #         #beam_dE_array_i = beam_obj.dE[idxB[i]]
    #         #beam_id_array_i = beam_obj.id[idxB[i]]
    #         itemindex = np.where( ((beam_dt_array_i-buncket_centres[i]) - dt_min)*(dt_max - (beam_dt_array_i-buncket_centres[i])) < 0 )[0]
    #         if itemindex.size != 0:
    #             beam_obj.id[ itemindex+idxB[i][0] ] = 0

    # def losses_energy_cut_bunch(beam_obj, idxB, buncket_centres, dE_min, dE_max):
    #     '''Beam losses based on energy cuts, e.g. on collimators (multibunch)'''

    #     nbf = len(idxB)

    #     for i in range(nbf):
    #         #beam_dt_array_i = beam_obj.dt[idxB[i]]
    #         beam_dE_array_i = beam_obj.dE[idxB[i]]
    #         #beam_id_array_i = beam_obj.id[idxB[i]]
    #         itemindex = np.where( ((beam_dE_array_i-buncket_centres[i]) - dE_min)*(dE_max - (beam_dE_array_i-buncket_centres[i])) < 0 )[0]
    #         if itemindex.size != 0:
    #             beam_obj.id[ itemindex+idxB[i][0] ] = 0

    # Losses (master) =========================================================

    def losses(losstype, beam_obj, idxB, losses_opt): #, mpi=False):

        # Tests for MPI, or imp conducted before calling the function, possibly
        # resulting in a reassignment of losstype, which is feed to this function

        if(losstype == 'cuts'):

            # Losses based on energy and/or longitudinal cuts
            # -----------------------------------------------------------------

            outmost_sepy_i   = losses_opt['outmost_sepy_i']
            bucket_centres_i = losses_opt['bucket_centres_i']
            maxbktl_i        = losses_opt['maxbktl_i']
            losscut          = losses_opt['losscut']
            i                = losses_opt['i']
            Nt_trk           = losses_opt['Nt_trk']

            nbf = len(idxB)

            max_abs_outmost_sepy_i = max(abs(outmost_sepy_i))
            #print('max_abs_outmost_sepy_i =', max_abs_outmost_sepy_i)
            #print('')

            if(losscut['dtL'] is not None): BeamTools.losses_dt_cut_bunch(beam_obj, idxB, bucket_centres_i, 'L', losscut['dtL']*maxbktl_i)              # /1.00*0.80*(Nt_trk-i/2.)/Nt_trk)
            if(losscut['dtR'] is not None): BeamTools.losses_dt_cut_bunch(beam_obj, idxB, bucket_centres_i, 'R', losscut['dtR']*maxbktl_i)              # /1.20*0.80*(Nt_trk-i/2.)/Nt_trk)
            if(losscut['dED'] is not None): BeamTools.losses_dE_cut_bunch(beam_obj, idxB, np.zeros(nbf),    'D', losscut['dED']*max_abs_outmost_sepy_i) # /1.00*0.80*(Nt_trk-i/2.)/Nt_trk)
            if(losscut['dEU'] is not None): BeamTools.losses_dE_cut_bunch(beam_obj, idxB, np.zeros(nbf),    'U', losscut['dEU']*max_abs_outmost_sepy_i) # /1.00*0.80*(Nt_trk-i/2.)/Nt_trk)

            # Always makes a longitudinal cut at the end of the tracking (everything outside the batch --with a threshold-- is set to zero)
            if(i == Nt_trk-2):
                #print('>> Longitudinal cut at the end to remove anything outside the batch (+/- 1.00*maxbktl_i)...')
                #print('Left margin:',  bucket_centres_i[ 0]-1.00*maxbktl_i )
                #print('Right margin:', bucket_centres_i[-1]+1.00*maxbktl_i)
                # Using the standard BLonD beam method for losses_longitudinal_cut:
                beam_obj.losses_longitudinal_cut(bucket_centres_i[0]-1.00*maxbktl_i, bucket_centres_i[-1]+1.00*maxbktl_i)
                #print('')

            gc.collect()

        elif(losstype == 'with_inteff'):

            # Losses based on separatrix WITH intensity effects
            # -----------------------------------------------------------------

            outmost_sepx_i = losses_opt['outmost_sepx_i']
            outmost_sepy_i = losses_opt['outmost_sepy_i']
            Nsfine         = losses_opt['Nsfine']

            idxL_outmost_sepx_i_beam_dt_partID = np.digitize(beam_obj.dt, outmost_sepx_i)-1
            #print(f'idxL_outmost_sepx_i_beam_dt_partID = {idxL_outmost_sepx_i_beam_dt_partID}')

            idxL_lostleft  = np.where(idxL_outmost_sepx_i_beam_dt_partID < 0)[0] # indices of particles that left the left side of the profile
            idxL_lostright = np.where(idxL_outmost_sepx_i_beam_dt_partID > len(outmost_sepx_i)-2)[0] # indices of particles that left the left side of the profile. # it could eb minus one, but then we need a >=

            idxR_outmost_sepx_i_beam_dt_partID = idxL_outmost_sepx_i_beam_dt_partID + 1
            # print('idxL_outmost_sepx_i_beam_dt_partID =', idxL_outmost_sepx_i_beam_dt_partID, len(idxL_outmost_sepx_i_beam_dt_partID))
            # print('idxR_outmost_sepx_i_beam_dt_partID =', idxR_outmost_sepx_i_beam_dt_partID, len(idxR_outmost_sepx_i_beam_dt_partID))
            # print('')

            idxL_outmost_sepx_i_beam_dt_partID[ idxL_lostleft  ] = 0
            idxL_outmost_sepx_i_beam_dt_partID[ idxL_lostright ] = 0
            idxR_outmost_sepx_i_beam_dt_partID[ idxL_lostleft  ] = 0
            idxR_outmost_sepx_i_beam_dt_partID[ idxL_lostright ] = 0
            # print('idxL_outmost_sepx_i_beam_dt_partID =', idxL_outmost_sepx_i_beam_dt_partID, len(idxL_outmost_sepx_i_beam_dt_partID))
            # print('idxR_outmost_sepx_i_beam_dt_partID =', idxR_outmost_sepx_i_beam_dt_partID, len(idxR_outmost_sepx_i_beam_dt_partID))
            # print('')

            # Find the corresponding value in the separatrix (energy axis) of each of those bin pairs
            sep_dEL_partID = outmost_sepy_i[idxL_outmost_sepx_i_beam_dt_partID]
            sep_dER_partID = outmost_sepy_i[idxR_outmost_sepx_i_beam_dt_partID]
            #print('sep_dEL_partID =', sep_dEL_partID, len(sep_dEL_partID))
            #print('sep_dER_partID =', sep_dER_partID, len(sep_dEL_partID))

            sep_dEL_partID[ idxL_lostleft  ] = 0
            sep_dEL_partID[ idxL_lostright ] = 0
            sep_dER_partID[ idxL_lostleft  ] = 0
            sep_dER_partID[ idxL_lostright ] = 0

            if(Nsfine < 128*8+1):  # Nsfine = Ns*8, therefore <=> Usually  now SPS: Ns = 160 / Nsfine = 1280 and LHC: Ns = 320 / 2560

                #print('Interp')
                # Linearly interpolated values at each particle's dt coordinate:
                sep_dE_partID = (sep_dER_partID-sep_dEL_partID)/(outmost_sepx_i[idxR_outmost_sepx_i_beam_dt_partID]-outmost_sepx_i[idxL_outmost_sepx_i_beam_dt_partID])*(beam_obj.dt-outmost_sepx_i[idxL_outmost_sepx_i_beam_dt_partID])+sep_dEL_partID
                #print('sep_dE_partID =', sep_dE_partID, len(sep_dE_partID))
                #print('')

                # Check for each particle's, if its dE coordinate is larger than the value in the energy axis of the separatrix at that corresponding time coordinate
                idx_beam_dE_outside = np.fabs(beam_obj.dE) > np.fabs(sep_dE_partID)
                #print('idx_beam_dE_outside =', idx_beam_dE_outside, len(idx_beam_dE_outside))
                #print('')

            else:

                #print('NO interp')
                idx_beam_dE_outside = np.fabs(beam_obj.dE) > np.fabs(sep_dEL_partID)

            # Set the id of those particles larger than their corresponding separatrix to zero
            beam_obj.id[ idx_beam_dE_outside ] = 0.
            #print('beam_obj.id =', beam_obj.id)
            #print('')

            gc.collect()

        else:  # i.e. losstype == 'without_inteff'

            # Losses based on separatrix WITHOUT intensity effects
            # -----------------------------------------------------------------

            #if mpi: beam_obj.gather()
            beam_obj.losses_separatrix( losses_opt['ring_obj'], losses_opt['rfstation_obj'] )
            #if mpi: beam_obj.split()

            gc.collect()

    #  Insert dt,dE,id data into beam object
    # (including dt/dE injection errors and dt/dE offsets) ====================

    def dump_dt_dE_id_into_beam_obj(beam_obj, dt_array, dE_array, id_array, maxbktl_0=None, injerr_dt=0, injerr_dE=0, offset_dt=0, offset_dE=0):
        if(maxbktl_0 is None): injerr_dt = 0
        else:                  injerr_dt *= maxbktl_0/360.
        beam_obj.dt = dt_array + injerr_dt - offset_dt
        beam_obj.dE = dE_array + injerr_dE - offset_dE
        beam_obj.id = id_array
        beam_obj.dt = np.ascontiguousarray(beam_obj.dt)
        beam_obj.dE = np.ascontiguousarray(beam_obj.dE)
        beam_obj.id = np.ascontiguousarray(beam_obj.id)
        ### or in one line...
        ##beam_obj.dt = np.ascontiguousarray( dt_array + injerr_dt - offset_dt )
        ##beam_obj.dE = np.ascontiguousarray( dE_array + injerr_dE - offset_dE )
        ##beam_obj.id = np.ascontiguousarray( id_array )
        print(f'beam_obj.dt = {beam_obj.dt}, shape = {beam_obj.dt.shape}, dtype = {beam_obj.dt.dtype}, ave = {beam_obj.dt.mean()}')
        print(f'beam_obj.dE = {beam_obj.dE}, shape = {beam_obj.dE.shape}, dtype = {beam_obj.dE.dtype}, ave = {beam_obj.dE.mean()}')
        print(f'beam_obj.id = {beam_obj.id}, shape = {beam_obj.id.shape}, dtype = {beam_obj.id.dtype},')
        print('')
        gc.collect()


###############################################################################

class BeamPattern(object):

    def __init__(self, rfstation_obj, nbf, nbs, nbm, Np_list_0, phaseoffset=None):   # dphi_rf cooresponds to the accumulated phase error in the lowest harmonic = rfstation.dphi_rf[0]

        self.rfstation = rfstation_obj
        self.nbf = nbf
        self.nbs = nbs
        self.nbm = nbm
        self.nbt = self.nbm + ((self.nbf - 1)*self.nbs + 1 ) + self.nbm
        self.bucketlist  = np.array([i for i in range(self.nbt)])
        self.fillpattern = np.array([j*self.nbs for j in range(self.nbf)]) + self.nbm

        self.Np_list_0 = np.copy(Np_list_0)
        
        self.get_idxB()
        self.idxB_0 = np.copy(self.idxB)
        
        self.phaseoffset      = phaseoffset
        self.timeoffset       = 0.
        self.timeoffset_delta = 0.
        self.timeoffset_tmp   = 0.
        
        self.track()
        
        self.maxbktl_0         = self.maxbktl
        self.bucket_centres_0  = np.copy(self.bucket_centres)
        self.marginoffset_0    = self.marginoffset
        
        #self.phaseoffset_delta_0 = self.phaseoffset_delta
        #self.marginoffset        = self.marginoffset_0 
        
    def get_idxB(self):

        self.idxB = []
        idx0 = 0
        for i in range(self.nbf):
            idxf = idx0 + self.Np_list_0[i]
            self.idxB.append( np.arange(idx0, idxf) )
            idx0 = idxf
        self.idxB = np.array(self.idxB)
        
        
        
    def get_idxB_MPI(self, beam_obj_split_id):
        
        self.idxB  = np.array([ np.intersect1d(self.idxB_0[i], beam_obj_split_id-1) for i in range(len(self.idxB_0)) ])
        #print(f'self.idxB = {self.idxB}')
        self.idxB -= beam_obj_split_id[0]-1
        #print(f'self.idxB = {self.idxB}')
        

    def track(self):

        if self.phaseoffset is None:
            self.maxbktl           = 2*np.pi / self.rfstation.omega_rf_d[0][ self.rfstation.counter[-1] ] # = self.rfstation.t_rf[0][0]                          if RF freq is not changing. Also = C/(h*beta*c) = Trev/h = 1/(h*frev) = 1/(frf) = 2pi/omegarf
            self.bucket_centres    = (self.fillpattern+0.5)*self.maxbktl                                  # self.bucket_centres = np.vstack((self.bucket_centres, np.array([(j*self.nbs+0.5)+self.maxbktl for j in range(self.nbf)])))
            self.marginoffset      = self.nbm * self.maxbktl
            
        else:
            self.maxbktl           = 2*np.pi / self.rfstation.omega_rf[  0][ self.rfstation.counter[-1] ] # = self.rfstation.t_rf[0][self.rfstation.counter[-1]] if RF freq is not changing.
            
            self.timeoffset        = self.rfstation.dphi_rf[0]/(2.*np.pi) * self.maxbktl 
            self.timeoffset_delta  = self.timeoffset - self.timeoffset_tmp
            self.timeoffset_tmp    = self.timeoffset
            self.marginoffset      -= self.timeoffset_delta
            self.bucket_centres    -= self.timeoffset_delta
            
            
###############################################################################







    ###########################################################################

#    def losses_separatrix(Ring, RFStation, beam_obj):
#        '''Beam losses based on separatrix'''
#
#        # is_in_separatrix
#
#        counter = RFStation.counter[0]
#
#        print('')
#        print(counter)
#        print(np.pi)
#        print(RFStation.phi_s[counter])
#        print(RFStation.phi_rf_d[0,counter])
#        print(RFStation.dphi_rf[0])
#        print(RFStation.omega_rf[0,counter])
#
#        dt_sep = (np.pi - RFStation.phi_s[counter]  - RFStation.phi_rf_d[0,counter] - RFStation.dphi_rf[0]) / RFStation.omega_rf[0,counter]
#
#        Hsep = hamiltonian(Ring, RFStation, beam_obj, dt_sep, 0, total_voltage=None)
#
#        print('dt_sep =', dt_sep*1e12, 'ps, Hsep =', Hsep)
#        print('')
#        #quit()
#
#        #import matplotlib.pyplot as plt
#        #fig, ax = plt.subplots()
#        #ax.plot(beam_obj.dt, hamiltonian(Ring, RFStation, beam_obj,  beam_obj.dt, beam_obj.dE, total_voltage=None))
#        #fig.savefig('tmp.png')
#        #quit()
#
#        isin = np.fabs(hamiltonian(Ring, RFStation, beam_obj,  beam_obj.dt, beam_obj.dE, total_voltage=None)) < np.fabs(Hsep)
#
#        # losses_separatrix
#
#        itemindex = np.where(isin == False)[0]
#
#        if itemindex.size != 0:
#            beam_obj.id[itemindex] = 0

    # def lost_multibunch(beam_obj, nbf):

    #     Np = int(beam_obj.n_total_macroparticles/nbf)
    #     lost_array = np.array([]).astype(int)
    #     for i in range(nbf):
    #         idx0 =   i  * Np
    #         idxf = (i+1)* Np
    #         beam_id_array_i = beam_obj.id[idxB[i]]
    #         lost_array = np.append(lost_array, len(np.where(beam_id_array_i==0)[0]))
    #     return lost_array.astype(int)

    # def alive_multibunch(beam_obj, nbf):

    #     Np = int(beam_obj.n_total_macroparticles/nbf)
    #     alive_array = np.array([]).astype(int)
    #     for i in range(nbf):
    #         idx0 =   i  * Np
    #         idxf = (i+1)* Np
    #         beam_id_array_i = beam_obj.id[idxB[i]]
    #         alive_array = np.append(alive_array, len(np.where(beam_id_array_i!=0)[0]))
    #     return alive_array.astype(int)



