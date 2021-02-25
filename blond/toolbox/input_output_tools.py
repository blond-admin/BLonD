#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import os
import shutil
import numpy as np

###############################################################################

def move_directory_content(root_src_dir, root_dst_dir):
    #root_src_dir = 'Src Directory\\'
    #root_dst_dir = 'Dst Directory\\'
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # in case of the src and dst are the same file
                if os.path.samefile(src_file, dst_file):
                    continue
                os.remove(dst_file)
            shutil.move(src_file, dst_dir)
            
# Print object parameters =====================================================

def print_object_attributes(obj, ok=True, onlyattr=False):
    r'''
    Function to print all attributes on an class object.

    Parameters
    ----------
    obj : class
        Object from which attributes will be print
    onlyattr:  bool
        True: Only print attribute names
        False: Print attribute names and their values

    Returns
    -------
    None
    '''

    if obj is not None and ok:
        print(obj.__class__.__name__+':')
        print(obj)
        if(onlyattr):
            for attr in obj.__dict__:
                print('*', attr, '= ...')
        else:
            for attr, value in obj.__dict__.items():
                print('*', attr, '=', value)
        print('')

    return None

###############################################################################

# NPY files ===================================================================

def mynpsave(fname, nparraydata, formatstr='float32', verbatim=False):
    if fname[-4:] != '.npy': fname += '.npy'
    if verbatim:
        #print(f'Saving {fname} ...')
        pass
    with open(fname, 'wb') as fout:
        np.save(fout, nparraydata.astype(formatstr))
    ##fout.close()

def mynpload(fname, formatstr='float32', verbatim=False):
    if fname[-4:] != '.npy': fname += '.npy'
    if verbatim:
        #print(f'Loading {fname} ...')
        pass
    with open(fname, 'rb') as fin:
        nparraydata = np.load(fin).astype(formatstr)
    ##fin.close()
    return nparraydata

# H5 files ====================================================================

# def myh5save(fname, nparraydata, formatstr='float32', verbatim=False):
#     # if(fname[-4:] == '.npy'):
#     #     fname = fname[:-4] + '.h5'
#     # elif(fname[-3:] != '.h5'):
#     #     fname += '.h5'
#     if(fname[-3:] != '.h5'):
#         fname += '.h5'
#     if verbatim:
#         #print(f'Saving {fname} ...')
#         pass
#     with h5py.File(fname, 'w') as fout:
#         fout.create_dataset('var', data=nparraydata.astype(formatstr)) #, dtype=formatstr) # nparraydata.shape is the default shape, compression="gzip"

# def myh5load(fname, formatstr='float32', verbatim=False):
#     # if(fname[-4:] == '.npy'):
#     #     fname = fname[:-4] + '.h5'
#     # elif(fname[-3:] != '.h5'):
#     #     fname += '.h5'
#     if(fname[-3:] != '.h5'):
#         fname += '.h5'
#     if verbatim:
#         #print(f'Loading {fname} ...')
#         pass
#     with h5py.File(fname, 'r') as fin:
#         nparraydata = fin['var'][:]
#         #print(nparraydata, nparraydata.dtype)
#     return nparraydata.astype(formatstr)

###############################################################################

# def print_basic_info_at_turn(i, nbf, showproctime=False):

#     proctime0 = time.process_time()

#     #print('')
#     #print('{0:24}: {1:9d}'.format(      'Time step',               i ))
#     #print('{0:24}: {1:9d}'.format(      'RF tracker counter',  rfstation.counter[0]))
#     #print('{0:24}: {1:9.1f} s'.format(  'Simulation time',         t0))
#     #print('{0:24}: {1:9.3f} GeV'.format('Beam momentum',           beam.momentum/1e9))
#     #print('{0:24}: {1:9.3f} GeV'.format('Beam energy',             beam.energy/1e9))
#     #print('{0:24}: {1:9.3f} rad'.format('delta_E',                 rfstation.delta_E[i]))
#     #print('{0:24}: {1:9.3f} MHz'.format('Design RF rev. freq.',    rfstation.omega_rf_d[0,i]/1e6))
#     #print('{0:24}: {1:9.3f} MHz'.format('RF rev. freq.',           rfstation.omega_rf[0,i]/1e6))
#     #print('{0:24}: {1:9.3f} rad'.format('RF phase',                rfstation.phi_rf[0,i]))
#     if(nbf == 1):
#        ##print('{0:24}: {1:.4e} s'.format(  'Mean bunch position',     beam.mean_dt))
#        ##print('{0:24}: {1:.4e} s'.format(  '4sigma rms bunch length', 4.*beam.sigma_dt))
#         #print('{0:24}: {1:9.3f} ns'.format('Bunch position',       profile.bunchPosition/1e-9))
#         #print('{0:24}: {1:9.3f} ns'.format('Bunch length',         profile.bunchLength/1e-9))
#         pass
#     else:
#         #print('{0:24}: '.format('Bunch position'), end='')
#         for j in range(nbf):
#             if(j == 0):
#                 #print(        '{0:9.3f} ns'.format(profile.bunchPosition[j]/1e-9))
#                 pass
#             else:
#                 #print(25*' ', '{0:9.3f} ns'.format(profile.bunchPosition[j]/1e-9))
#                 pass
#         #print('{0:24}: '.format('Bunch length'), end='')
#         for j in range(nbf):
#             if(j == 0):
#                 #print(        '{0:9.3f} ns'.format(profile.bunchLength[j]/1e-9))
#                 pass
#             else:
#                 #print(25*' ', '{0:9.3f} ns'.format(profile.bunchLength[j]/1e-9))
#                 pass
#         if(j != 0):
#             #print('{0:>24}: {1:9.3f} ns'.format('ave.', np.mean(profile.bunchLength)/1e-9))
#             pass
#     #print('')

#     if showproctime:
#         proctime = time.process_time()
#         print(f'Proc. time I: {proctime-proctime0:.3f} s')

# def print_final_results(stage, i, nbf):

#     if(stage == 'SPS-FT'):
#         #print('\n\n'+80*'*'+'\n'+'*** RESULTS:', SPS_yamlname, '\n'+80*'*')
#         pass
#     elif(stage == 'LHC-INJ' or stage == 'LHC-FB'):
#         #print('\n\n'+80*'*'+'\n'+'*** RESULTS:', SPS_yamlname, '->', LHC_yamlname, '\n'+80*'*')
#         pass
#     elif(stage == 'LHC-RAMP'):
#         #print('\n\n'+80*'*'+'\n'+'*** RESULTS:', SPS_yamlname, '->', LHC_yamlname, '->', LHCramp_yamlname, '\n'+80*'*')
#         pass

#     for i in [0, -1]:

#         if  (i ==  0):
#             #print('FIRST TURN:', beam_n_macroparticles_turns[i], '\n')
#            ##print('First turn:', profile_n_macroparticles_turns[i], '\n') # should be the same
#             pass
#         elif(i == -1):
#             #print('LAST TURN:',  beam_n_macroparticles_turns[i], '\n')
#            ##print('Last turn:',  profile_n_macroparticles_turns[i], '\n') # should be the same
#             pass

#         #print('Losses from BEAM')

#         #print(176*'-')

#         #print('{0:>14}'.format('a_OrigAveNp'), end='')
#         #print('{0:>14}'.format('b_ActuAveNp'), end='')
#         #print('{0:>6}'.format('nb'), end='')
#         #print('{0:>14}'.format('c_OrigNpi'), end='')
#         #print('{0:>14}'.format('d_ActNpi'), end='')
#         #print(' |', end='')
#         #print('{0:>14}'.format('Alive'), end='')
#         #print('{0:>10}'.format('Alive_a'), end='')
#         #print('{0:>10}'.format('Alive_b'), end='')
#         #print('{0:>10}'.format('Alive_c'), end='')
#         #print('{0:>10}'.format('Alive_d'), end='')
#         #print(' |', end='')
#         #print('{0:>14}'.format('Lost'), end='')
#         #print('{0:>10}'.format('Lost_a'), end='')
#         #print('{0:>10}'.format('Lost_b'), end='')
#         #print('{0:>10}'.format('Lost_c'), end='')
#         #print('{0:>10}'.format('Lost_d'), end='')
#         #print('')

#         #print(176*'-')

#         for j in range(nbf):

#             #print('{0:14.6e}'.format(Np_tmp), end='')                                                        # a) Original (ave) no. of macroparticles per bunch at the start of the simulation
#             #print('{0:14.6e}'.format(Np), end='')                                                            # b) Actual   (ave) no. of macroparticles per bunch at the start of the simulation. For SPS, it coincides with the one above. For LHC-INJ/LHC-FB and LHC-RAMP, it is computed as no. of alive particles over the number of bunches (an average). DO NOT USE IN OPERATIONS WITH THE BEAM object! Use the list with the actual number PER bunch

#             #print('{0:6}'.format(j), end='')
#             #print('{0:14.6e}'.format(Np_list_tmp[j]), end='')                                                # c) Original no. of macroparticles per bunch (list) at the start of the simulation
#             #print('{0:14.6e}'.format(Np_list[j]), end='')                                                    # d) Actual   no. of macroparticles per bunch (list) at the start of the simulation. For SPS, it coincides with the one above. For LHC-INJ/LHC-FB and LHC-RAMP, it matches with the bunch intensity at the end of the simulation of the previous stage (i.e. SPS and LHC-INJ/LHC-FB)
#             #print(' |', end='')
#             #print('{0:14.6e}'.format(beam_nmacrop_bunch_alive[i][j]), end='')                       # No. of alive macroparticles per bunch
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_alive[i][j]/Np_tmp), end='')           # Ratio of alive macroparticles per bunch w.r.t. (a) the original ave no. of macroparticles per bunch    # Should coincide with [*]
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_alive[i][j]/Np), end='')               # Ratio of alive macroparticles per bunch w.r.t. (b) the actual   ave no. of macroparticles per bunch    # DO NOT USE
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_alive[i][j]/Np_list_tmp[j]), end='')   # Ratio of alive macroparticles per bunch w.r.t. (c) the original no. of macroparticles per bunch (list) # [*]
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_alive[i][j]/Np_list[j]), end='')       # Ratio of alive macroparticles per bunch w.r.t. (d) the actual   no. of macroparticles per bunch (list)
#             #print(' |', end='')
#             #print('{0:14.6e}'.format(beam_nmacrop_bunch_lost[i][j]), end='')                        # No. of alive macroparticles per bunch
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_lost[i][j]/Np_tmp), end='')            # Ratio of lost macroparticles per bunch w.r.t. (a) the original ave no. of macroparticles per bunch     # Should coincide with [*]
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_lost[i][j]/Np), end='')                # Ratio of lost macroparticles per bunch w.r.t. (b) the actual   ave no. of macroparticles per bunch     # DO NOT USE
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_lost[i][j]/Np_list_tmp[j]), end='')    # Ratio of lost macroparticles per bunch w.r.t. (c) the original no. of macroparticles per bunch (list)  # [*]
#             #print('{0:10.4f}'.format(100.*beam_nmacrop_bunch_lost[i][j]/Np_list[j]))                # Ratio of lost macroparticles per bunch w.r.t. (d) the actual   no. of macroparticles per bunch (list)
#             pass

#         #print(176*'-')

#         #print('{0:14}'.format(''), end='')
#         #print('{0:14}'.format(''), end='')
#         #print('{0:6}'.format('Total'), end='')
#         #print('{0:14.6e}'.format(Np_tot_tmp), end='')
#         #print('{0:14.6e}'.format(Np_tot), end='')
#         #print(' |', end='')
#         #print('{0:14.6e}'.format(beam_nmacrop_beam_alive[i]), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10.4f}'.format(100.*beam_nmacrop_beam_alive[i]/Np_tot_tmp), end='')
#         #print('{0:10.4f}'.format(100.*beam_nmacrop_beam_alive[i]/Np_tot), end='')
#         #print(' |', end='')
#         #print('{0:14.6e}'.format(beam_nmacrop_beam_lost[i]), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10.4f}'.format(100.*beam_nmacrop_beam_lost[i]/Np_tot_tmp), end='')
#         #print('{0:10.4f}'.format(100.*beam_nmacrop_beam_lost[i]/Np_tot))
#         #print('\n')


#         #print('Losses from PROFILE')

#         #print(232*'-')

#         #print('{0:>14}'.format('a_OrigAveNp'), end='')
#         #print('{0:>14}'.format('b_ActuAveNp'), end='')
#         #print('{0:>6}'.format('nb'), end='')
#         #print('{0:>14}'.format('c_OrigNpi'), end='')
#         #print('{0:>14}'.format('d_ActNpi'), end='')
#         #print(' |', end='')
#         #print('{0:>14}'.format('inbf'), end='')
#         #print('{0:>10}'.format('inbf_a'), end='')
#         #print('{0:>10}'.format('inbf_b'), end='')
#         #print('{0:>10}'.format('inbf_c'), end='')
#         #print('{0:>10}'.format('inbf_d'), end='')
#         #print(' |', end='')
#         #print('{0:>14}'.format('outbf'), end='')
#         #print('{0:>10}'.format('outbf_a'), end='')
#         #print('{0:>10}'.format('outbf_b'), end='')
#         #print('{0:>10}'.format('outbf_c'), end='')
#         #print('{0:>10}'.format('outbf_d'), end='')
#         #print(' |', end='')
#         #print('{0:>14}'.format('inbs'), end='')
#         #print('{0:>10}'.format('inbs_a'), end='')
#         #print('{0:>10}'.format('inbs_b'), end='')
#         #print('{0:>10}'.format('inbs_c'), end='')
#         #print('{0:>10}'.format('inbs_d'), end='')
#         #print('')

#         #print(232*'-')

#         for j in range(nbf):

#             #print('{0:14.6e}'.format(Np_tmp), end='')                                                                # a) Original (ave) no. of macroparticles per bunch at the start of the simulation
#             #print('{0:14.6e}'.format(Np), end='')                                                                    # b) Actual   (ave) no. of macroparticles per bunch at the start of the simulation. For SPS, it coincides with the one above. For LHC-INJ/LHC-FB and LHC-RAMP, it is computed as no. of alive particles over the number of bunches (an average). DO NOT USE IN OPERATIONS WITH THE BEAM object! Use the list with the actual number PER bunch

#             #print('{0:6}'.format(j), end='')
#             #print('{0:14.6e}'.format(Np_list_tmp[j]), end='')                                                        # c) Original no. of macroparticles per bunch (list) at the start of the simulation
#             #print('{0:14.6e}'.format(Np_list[j]), end='')                                                            # d) Actual   no. of macroparticles per bunch (list) at the start of the simulation. For SPS, it coincides with the one above. For LHC-INJ/LHC-FB and LHC-RAMP, it matches with the bunch intensity at the end of the simulation of the previous stage (i.e. SPS and LHC-INJ/LHC-FB)
#             #print(' |', end='')
#             #print('{0:14.6e}'.format(profile_nmacrop_bunch_inbf[i][j]), end='')                             # No. of macroparticles per bunch in filled buckets
#             #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbf[i][j]/Np_tmp), end='')                 # Ratio of macroparticles per bunch in filled buckets w.r.t. (a) the original ave no. of macroparticles per bunch    # Should coincide with [*]
#             #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbf[i][j]/Np), end='')                     # Ratio of macroparticles per bunch in filled buckets w.r.t. (b) the actual   ave no. of macroparticles per bunch    # DO NOT USE
#             #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbf[i][j]/Np_list_tmp[j]), end='')         # Ratio of macroparticles per bunch in filled buckets w.r.t. (c) the original no. of macroparticles per bunch (list) # [*]
#             #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbf[i][j]/Np_list[j]), end='')             # Ratio of macroparticles per bunch in filled buckets w.r.t. (d) the actual   no. of macroparticles per bunch (list)
#             #print(' |', end='')
#             #print('{0:14.6e}'.format((Np_list_tmp[j]-profile_nmacrop_bunch_inbf[i][j])), end='')                        # No. of macroparticles per bunch lost from filled buckets # WITH RESPECT TO (d)
#             #print('{0:10.4f}'.format(100.*(Np_list_tmp[j]-profile_nmacrop_bunch_inbf[i][j])/Np_tmp), end='')            # Ratio of macroparticles per bunch lost from filled buckets w.r.t. (a) the original ave no. of macroparticles per bunch     # Should coincide with [*]
#             #print('{0:10.4f}'.format(100.*(Np_list_tmp[j]-profile_nmacrop_bunch_inbf[i][j])/Np), end='')                # Ratio of macroparticles per bunch lost from filled buckets w.r.t. (b) the actual   ave no. of macroparticles per bunch     # DO NOT USE
#             #print('{0:10.4f}'.format(100.*(Np_list_tmp[j]-profile_nmacrop_bunch_inbf[i][j])/Np_list_tmp[j]), end='')    # Ratio of macroparticles per bunch lost from filled buckets w.r.t. (c) the original no. of macroparticles per bunch (list)  # [*]
#             #print('{0:10.4f}'.format(100.*(Np_list_tmp[j]-profile_nmacrop_bunch_inbf[i][j])/Np_list[j]))                # Ratio of macroparticles per bunch lost from filled buckets w.r.t. (d) the actual   no. of macroparticles per bunch (list)
#             # #print(' |', end='')
#             # #print('{0:14.6e}'.format(profile_nmacrop_bunch_inbs[i][j]), end='')                           # No. of macroparticles per bunch in separation buckets
#             # #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbs[i][j]/Np_tmp), end='')               # Ratio of macroparticles per bunch in separation buckets w.r.t. (a) the original ave no. of macroparticles per bunch    # Should coincide with [*]
#             # #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbs[i][j]/Np), end='')                   # Ratio of macroparticles per bunch in separation buckets w.r.t. (b) the actual   ave no. of macroparticles per bunch    # DO NOT USE
#             # #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbs[i][j]/Np_list_tmp[j]), end='')       # Ratio of macroparticles per bunch in separation buckets w.r.t. (c) the original no. of macroparticles per bunch (list) # [*]
#             # #print('{0:10.4f}'.format(100.*profile_nmacrop_bunch_inbs[i][j]/Np_list[j]), end='')           # Ratio of macroparticles per bunch in separation buckets w.r.t. (d) the actual   no. of macroparticles per bunch (list)
#             pass

#         #print(232*'-')

#         #print('{0:14}'.format(''), end='')
#         #print('{0:14}'.format(''), end='')
#         #print('{0:6}'.format('Total'), end='')
#         #print('{0:14.6e}'.format(Np_tot_tmp), end='')
#         #print('{0:14.6e}'.format(Np_tot), end='')
#         #print(' |', end='')
#         #print('{0:14.6e}'.format(profile_nmacrop_beam_inbf[i]), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10.4f}'.format(100.*profile_nmacrop_beam_inbf[i]/Np_tot_tmp), end='')
#         #print('{0:10.4f}'.format(100.*profile_nmacrop_beam_inbf[i]/Np_tot), end='')
#         #print(' |', end='')
#         #print('{0:14.6e}'.format((Np_tot_tmp-profile_nmacrop_beam_inbf[i])), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10.4f}'.format(100.*(Np_tot_tmp-profile_nmacrop_beam_inbf[i])/Np_tot_tmp), end='')
#         #print('{0:10.4f}'.format(100.*(Np_tot_tmp-profile_nmacrop_beam_inbf[i])/Np_tot), end='')
#         #print(' |', end='')
#         #print('{0:14.6e}'.format(profile_nmacrop_beam_inbs[i]), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10}'.format(''), end='')
#         #print('{0:10.4f}'.format(100.*profile_nmacrop_beam_inbs[i]/Np_tot_tmp), end='')
#         #print('{0:10.4f}'.format(100.*profile_nmacrop_beam_inbs[i]/Np_tot))
#         #print('\n')
