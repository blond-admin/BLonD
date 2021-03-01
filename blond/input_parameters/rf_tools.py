#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

TOOLS BASED ON BLOND_COMMON, NOT BLOND

@author: medinamluis
"""

import gc
import numpy as np
import pathlib

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter #AutoLocator


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


# BLonD-based tools
from blond_common.rf_functions.potential import rf_voltage_generation
from blond_common.rf_functions.potential import rf_potential_generation_cubic
from blond_common.rf_functions.potential import find_potential_wells_cubic
from blond_common.rf_functions.potential import potential_well_cut_cubic
from blond_common.rf_functions.potential import trajectory_area_cubic

###############################################################################

class RFTools(object):

    # Separatrix with intensity effects (incl. all sub-separatrices) ----------

    def separatrix(ring_obj,
                   rfstation_obj,
                   profile_obj,
                   totalinducedvoltage_obj,   # Mandatory - we are interested in separatrices WITH intensity effects afterwall, but it can be None if you insist!
                   stepi, nbt,
                   Ns=1024,
                   omit_conjoined_wells=False,
                   relative_max_val_precision_limit=None,
                   outdir=None,
                   verbose=False):
        '''
        Function to compute the separatrices including intensity effects. It first computes
        the RF voltage and total voltage (with induced voltage), and then the total potential
        from it. The separatrices are cumputed from the total potential cut into individual
        wells  (max and min are found to do the cuts). For each well, the Hamiltonian,
        area, half maximum height (energy) and full bucket length (time) are also computed.

        Parameters
        ----------
        ...
        '''

        # Import RF and ring_obj parameters at this moment stepi
        q      = ring_obj.Particle.charge
        trev   = ring_obj.t_rev[stepi]
        beta   = ring_obj.beta[0,stepi]
        energy = rfstation_obj.energy[stepi]
        Vrf    = rfstation_obj.voltage[:,stepi]
        h      = rfstation_obj.harmonic[:,stepi]
        phirf  = rfstation_obj.phi_rf[:,stepi]
        trf    = rfstation_obj.t_rf[:,stepi]
        eta0   = rfstation_obj.eta_0[stepi]
        deltaE = rfstation_obj.delta_E[stepi]     # instead of:   try: deltaE = rfstation_obj.delta_E[stepi]; except: deltaE = rfstation_obj.delta_E[-1]
        xmin   = profile_obj.cut_left
        xmax   = profile_obj.cut_right
        if verbose:
            print('stepi      =', stepi)
            print('q      =', q)
            print('trev   =', trev)
            print('Vrf    =', Vrf)
            print('h      =', h)
            print('phirf  =', phirf)
            print('eta0   =', eta0)
            print('deltaE =', deltaE)
            print('')

        time_array = np.linspace(xmin, xmax, Ns*nbt)  # copy.copy(profile_objBinCenters)
        timerange  = [xmin, xmax]
        deltat = (xmax-xmin)/nbt

        # Time array (voltages and potentials have all the same time array)

        if verbose:
            print('time_array =', time_array, len(time_array)) # [s]
            print('timerange =', timerange)
            print('deltat    =', deltat)
            print('')

        # Induced voltage

        if totalinducedvoltage_obj is None:
            TotIndVolt_time = None
            TotIndVolt      = None
            vind_array = np.zeros(len(time_array))
        else:
            TotIndVolt_time = totalinducedvoltage_obj.time_array
            TotIndVolt      = totalinducedvoltage_obj.induced_voltage
            vind_array = np.interp(time_array, TotIndVolt_time, TotIndVolt)

        if verbose:
            print(f'TotIndVolt_time = {TotIndVolt_time}')
            print(f'TotIndVolt = {TotIndVolt}') # [V]
            print(f'vind_array = {vind_array}, len = {len(vind_array)}') # [V]
            print('')

        # RF voltage

        timetmp_array, vrf_array = rf_voltage_generation(len(time_array),
                                                         trev,
                                                         Vrf,
                                                         h,
                                                         phirf,
                                                         time_bounds=timerange)

        if verbose:
            print('vrf_array =', vrf_array, len(vrf_array)) # [V]
            print('')

        # Total voltage

        vtot_array = vrf_array + vind_array # [V]

        if verbose:
            print('vtot_array =', vtot_array, len(vtot_array)) # [V]
            print('')

        if outdir is not None:
            figtmp, axtmp = plt.subplots()
            figtmp.set_size_inches(8.0+nbt*16.0/48., 8.0)
           #axtmp.plot(time_array/1.e-9,      vind_array/1.e6, '-', label='vind (interp)')
           #axtmp.plot(TotIndVolt_time/1.e-9, TotIndVolt/1.e6, '-', label='vind')
            axtmp.plot(time_array /1.e-9,     vrf_array /1.e6, '-', label='vrf')
            axtmp.plot(time_array /1.e-9,     vtot_array/1.e6, '-', label='vtot')
            axtmp.set_xlabel('Time [ns]')
            axtmp.set_ylabel('Voltage [MV]')
           #axtmp.set_xlim(0., 50.)
            axtmp.legend()
            figtmp.tight_layout()
            figtmp.savefig(f'{outdir}/plot_voltage_rf_total_vs_time.png')
            figtmp.clf()

        # Total potential

        #if TotIndVolt is None:
        #
        #    # Without intensity effects
        #    timetmp_array, utot_array = rf_potential_generation(len(time_array),
        #                                                        trev,
        #                                                        vtot_array,
        #                                                        h,
        #                                                        phirf,
        #                                                        eta0,
        #                                                        q,
        #                                                        deltaE,
        #                                                        time_bounds=timerange)
        #
        #else:

        # With or without intensity effects (the added vind woould have been zero)
        timetmp_array, utot_array = rf_potential_generation_cubic(time_array,
                                                                  vtot_array,
                                                                  eta0,
                                                                  q,
                                                                  trev,
                                                                  deltaE)[0:2] # TODO: Check what's in the remaining entries, for reference...

        if verbose:
            print('utot_array =', utot_array, len(utot_array)) # [eV]
            print('')

        gc.collect()

        # Estimate number of max and min of potential wells

        mest = int(2*nbt-1)

        if verbose:
            print('nbt  =', nbt)
            print('mest =', mest)
            print('')

        # Arrays extended with one more point to the left and to the right (the value of each point is taken as the double of the corresponding left/right point in the original array)
        ext_time_array = np.insert(time_array, 0, np.array([2.*time_array[0]]))  # the first element has to be different of zero
        ext_utot_array = np.insert(utot_array, 0, np.array([   utot_array[0]]))
        ext_time_array = np.insert(ext_time_array, len(ext_time_array), np.array([2.*ext_time_array[-1]]))
        ext_utot_array = np.insert(ext_utot_array, len(ext_utot_array), np.array([   ext_utot_array[-1]]))

        if verbose:
            print('time_array =', time_array, len(time_array))
            print('utot_array =', utot_array, len(utot_array))
            print('ext_time_array =', ext_time_array, len(ext_time_array))
            print('ext_utot_array =', ext_utot_array, len(ext_utot_array))
            print('')

        #print('find_potential_wells_cubic...\n')

        if relative_max_val_precision_limit is None: relative_max_val_precision_limit=1.e-1
        #if(stepi == 0 or TotIndVolt is None): relative_max_val_precision_limit=1.e-1
        #else:                                 relative_max_val_precision_limit=1.e-4 #4

        utot_max_t, utot_max, utot_max_inner, utot_min_t, utot_min = find_potential_wells_cubic(time_array, #ext_time_array,
                                                                                                utot_array, #ext_utot_array,
                                                                                                relative_max_val_precision_limit=relative_max_val_precision_limit,
                                                                                                mest=mest,
                                                                                                verbose=False)

        if verbose:
            print('utot_max_inner =', utot_max_inner, len(utot_max_inner))
            print('')
            print('utot_max_t =', utot_max_t, len(utot_max_t)) # pair of points (left/right)
            print('utot_max   =', utot_max  , len(utot_max)  )
            print('utot_min_t =', utot_min_t, len(utot_min_t)) # single point (centre)
            print('utot_min   =', utot_min  , len(utot_min)  )
            print('')

        if outdir is not None:
            figtmp, axtmp = plt.subplots()
            figtmp.set_size_inches(8.0+nbt*16.0/48., 8.0)
            axtmp.plot(time_array/1.e-9,  utot_array/1.e3)

            # Pairs (left/right) of maximums
            for i in range(len(utot_max_t)):
                axtmp.plot(np.array(utot_max_t)[i]/1.e-9, np.array(utot_max)[i]/1.e3, 'x')
            # Minimums (centre)
            axtmp.plot(np.array(utot_min_t)/1.e-9, np.array(utot_min)/1.e3, 'kx')

            axtmp.xaxis.set_major_locator(MultipleLocator(25))
            axtmp.xaxis.set_minor_locator(MultipleLocator(5))
            axtmp.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axtmp.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axtmp.grid()
            axtmp.set_xlabel('Time [ns]')
            axtmp.set_ylabel('RF potential [keV]')
            figtmp.tight_layout()
            figtmp.savefig(f'{outdir}/plot_potential_total_vs_time.png')
            figtmp.clf()

        gc.collect()

        # Cut potential wells

        list_utot_array_time, list_utot_array = potential_well_cut_cubic(time_array,
                                                                         utot_array,
                                                                         utot_max_t)

        if verbose:
            print('list_utot_array_time =', '...') #list_utot_array_time, len(list_utot_array_time))
            print('list_utot_array     =',  '...') #list_utot_array,      len(list_utot_array)     )
            print('')

        if outdir is not None:
            figtmp, axtmp = plt.subplots()
            figtmp.set_size_inches(8.0+nbt*16.0/48., 8.0)
            axtmp.plot(time_array/1.e-9,  utot_array/1.e3, '-')

            # All individual wells (including those longer than one bucket of the main harmonic)
            for i in range(len(list_utot_array_time)):
                axtmp.plot(list_utot_array_time[i]/1.e-9, list_utot_array[i]/1.e3+i*0.05)

            axtmp.set_xlabel('Time [ns]')
            axtmp.set_ylabel('RF potential [keV]')
            figtmp.tight_layout()
            figtmp.savefig(f'{outdir}/plot_potential_rf_cut_vs_time.png')
            figtmp.clf()

        # Ignore potential wells that extend over more than one bucket (allow margins)

        if omit_conjoined_wells:
            i_omit = []
            for i in range(len(list_utot_array_time)):
                welllength = list_utot_array_time[i][-1] - list_utot_array_time[i][0]
                print('welllength =', '%6.2f' %(welllength/1.e-9), 'ns', end=' ')
                if(abs(welllength-trf[0])/trf[0] > 0.5):
                    i_omit.append(i)
                    print('*')
                else:
                    print('')

            list_utot_array_time = np.delete(list_utot_array_time, i_omit)
            list_utot_array      = np.delete(list_utot_array,      i_omit)

            if verbose:
                print('i_omit =', i_omit)
                print('list_utot_array_time =', '...') #list_utot_array_time, len(list_utot_array_time))
                print('list_utot_array     =',  '...') #list_utot_array,      len(list_utot_array))
                print('')

        gc.collect()

        # Separatrices and acceptance

        list_separatrix_time  = [] # Separatrix    of each bucket (time) -- redf
        list_separatrix       = [] # Separatrix    of each bucket (val)
        list_H                = [] # Hamiltonian   of each bucket
        list_A                = [] # Enclosed area of each bucket
        list_BucketHalfHeight = [] # Half bucket height (energy)
        list_BucketFullLength = [] # Full bucket length (time)

        # For each well:
        for i in range(len(list_utot_array_time)):
            separatrix_time_i, separatrix_i, H_i, A_i, BucketHalfHeight_i, BucketFullLength_i = trajectory_area_cubic(list_utot_array_time[i],
                                                                                                                      list_utot_array[i],
                                                                                                                      eta0,
                                                                                                                      beta,
                                                                                                                      energy)
            list_separatrix_time.append(np.ascontiguosarray(separatrix_time_i))
            list_separatrix.append(np.ascontiguosarray(separatrix_i))
            list_H.append(H_i)
            list_A.append(A_i)
            list_BucketHalfHeight.append(BucketHalfHeight_i)
            list_BucketFullLength.append(BucketFullLength_i)

        if verbose:
            print('list_separatrix_time =', '...') #list_separatrix_time, len(list_separatrix_time)) # [s]
            print('list_separatrix      =', '...') #list_separatrix, len(list_separatrix))           # [eV]
            print('list_H =', list_H, len(list_H))                                                   # [eV]
            print('list_A =', list_A, len(list_A))                                                   # [eV.s]
            print('list_BucketHalfHeight =', list_BucketHalfHeight, len(list_BucketHalfHeight))      # [eV]
            print('list_BucketFullLength =', list_BucketFullLength, len(list_BucketFullLength))      # [s]

        if outdir is not None:
            figtmp, axtmp = plt.subplots()
            figtmp.set_size_inches(8.0+nbt*16.0/48., 8.0)
            colors = plt.cm.rainbow( np.array(range(len(list_separatrix_time)))/(len(list_separatrix_time)-1) )

            for i in range(len(list_separatrix_time)):
                plt.plot(        list_separatrix_time[i]/1.e-9,  list_separatrix[i]/1.e6,                           color=colors[i])
                plt.plot(        list_separatrix_time[i]/1.e-9, -list_separatrix[i]/1.e6,                           color=colors[i])
                plt.fill_between(list_separatrix_time[i]/1.e-9,  list_separatrix[i]/1.e6, -list_separatrix[i]/1.e6, color=colors[i], alpha=0.3)
            axtmp.set_xlabel('Time [ns]')
            axtmp.set_ylabel('Energy $\\Delta E$ [MeV]')
            axtmp.set_xlim(np.array(timerange)/1.e-9)
            figtmp.tight_layout()
            figtmp.savefig(f'{outdir}/plot_separatrix_vs_time.png')
            figtmp.clf()

        gc.collect()

        #quit()
        return list_separatrix_time, list_separatrix, time_array, vrf_array, vind_array, vtot_array, utot_array, list_H, list_A, list_BucketHalfHeight


    # Outmost separatrix with intensity effects -------------------------------

    def separatrix_outmost(ring_obj,
                           rfstation_obj,
                           profile_obj,
                           totalinducedvoltage_obj,
                           stepi,
                           nbt,
                           align_time_array, # An array of x-coordinates (time) is needed to align the sub-sepatrices - passing align_time_array=sepx_wo_inteff_i seems the obvious option (be sure it has a high resolution!)...
                           Ns=1024,
                           omit_conjoined_wells=False,
                           relative_max_val_precision_limit=1e0,
                           outdir=None,
                           verbose=False):

        sepx_w_inteff_i, sepy_w_inteff_i,\
        time_array_i, vrf_array_i, vind_array_i, vtot_array_i, utot_array_i,\
        list_H_i, list_A_i, list_BucketHalfHeight_i = RFTools.separatrix(ring_obj,
                                                                         rfstation_obj,
                                                                         profile_obj,
                                                                         totalinducedvoltage_obj,
                                                                         stepi,
                                                                         nbt,
                                                                         Ns=Ns,
                                                                         omit_conjoined_wells=omit_conjoined_wells,
                                                                         relative_max_val_precision_limit=1e0,
                                                                         outdir=outdir,
                                                                         verbose=False)
        del(time_array_i)
        del(vrf_array_i)
        del(vind_array_i)
        del(vtot_array_i)
        del(utot_array_i)
        del(list_H_i)
        del(list_A_i)
        del(list_BucketHalfHeight_i)
        #print('')
        ##print('sepx_w_inteff_i =', sepx_w_inteff_i[0], '...', sepx_w_inteff_i[-1], len(sepx_w_inteff_i), len(sepx_w_inteff_i[0]), '...', len(sepx_w_inteff_i[-1]))
        ##print('sepy_w_inteff_i =', sepy_w_inteff_i[0], '...', sepy_w_inteff_i[-1], len(sepy_w_inteff_i), len(sepy_w_inteff_i[0]), '...', len(sepy_w_inteff_i[-1]))
        #print('sepx_w_inteff_i =', '...', len(sepx_w_inteff_i), len(sepx_w_inteff_i[0]), '...', len(sepx_w_inteff_i[-1]))
        #print('sepy_w_inteff_i =', '...', len(sepy_w_inteff_i), len(sepy_w_inteff_i[0]), '...', len(sepy_w_inteff_i[-1]))
        #print('')

        gc.collect()

        # List with the corresponding indices of the first x-coordinates of each sub-separatrix in the x-array used for alignment (i.e. the TIME
        # ARRAY OF THE SEPARATRIX W/O INT. EFF. -- the resolution of both the separatrices w/ and w/o most be the same, which more or less is.
        #  Of course, the higher the resolution the better!)
        list_idx_align_sep_w = np.array([]).astype(int)
        for idxs in range(len(sepx_w_inteff_i)): # For all sub-separatrices
            list_idx_align_sep_w = np.append(list_idx_align_sep_w, np.digitize(sepx_w_inteff_i[idxs][0], align_time_array))
        #print('list_idx_align_sep_w =', list_idx_align_sep_w, len(list_idx_align_sep_w))
        #print('list_idx_align_sep_w = ...', len(list_idx_align_sep_w))
        #print('')

        # On zero arrays of the same length than the separatrix w/o int. eff, we paste the sub-separatrices in their corresponding positions using the position of the first element found above
        ext_sepy_w_inteff_i = []
        for idxs in range(len(sepy_w_inteff_i)): # For all sub-separatrices
            idxt0 = list_idx_align_sep_w[idxs]
            sepy_w_inteff_i_idxs = np.zeros(len(align_time_array))
            for idxt in range(len(sepy_w_inteff_i[idxs])):
                sepy_w_inteff_i_idxs[idxt0+idxt] = sepy_w_inteff_i[idxs][idxt]
            ext_sepy_w_inteff_i.append(sepy_w_inteff_i_idxs)
        ext_sepy_w_inteff_i = np.array(ext_sepy_w_inteff_i)
        ##print('')
        ##print('ext_sepy_w_inteff_i =', ext_sepy_w_inteff_i, len(ext_sepy_w_inteff_i))
        #print('ext_sepy_w_inteff_i = ...', len(ext_sepy_w_inteff_i))
        #print('')
        ##print('sepy_w_inteff_i[2][10] =', sepy_w_inteff_i[2][10])
        ##print('ext_sepy_w_inteff_i[2]['+str(list_idx_align_sep_w[2])+'+10] =', ext_sepy_w_inteff_i[2][list_idx_align_sep_w[2]+10])
        ##print('')#

        del(list_idx_align_sep_w)

        del(sepx_w_inteff_i)
        del(sepy_w_inteff_i)

        gc.collect()

        # Once all sub-separatrices extend over the same number of x points, we can compare them all against each other element-by-element.
        # The outmost separatrix is formed with the largest y-value among all separatrices for a given x.

        outmost_sepx_i = np.copy(align_time_array)
        outmost_sepy_i = np.array([])

        outmost_sepy_i = np.copy(ext_sepy_w_inteff_i[0])
        if(len(ext_sepy_w_inteff_i) > 1):
            for idxs in range(1, len(ext_sepy_w_inteff_i)):
                ##print(idxs)
                outmost_sepy_i = np.fmax(outmost_sepy_i, ext_sepy_w_inteff_i[idxs])
        #print('outmost_sepx_i =', outmost_sepx_i, len(outmost_sepx_i))
        #print('outmost_sepy_i =', outmost_sepy_i, len(outmost_sepy_i))
        #print('')

        del(ext_sepy_w_inteff_i)

        # Interpolate for better accuracy used to be here... However, it is not
        # the case (and thus not returned) in this function. Interpolation is
        # only computed (in the BeamTools module) if beam losses based on
        # separatrix w/ int. eff. is requested

        #if outdir is not None:
            # # Plot of all sub-separatrices
            # figtmp = plt.figure()
            # figtmp.set_size_inches(60.0,8.0)
            # for idxs in range(len(ext_sepy_w_inteff_i)):
            #     #print(idxs)
            #     plt.plot(align_time_array, ext_sepy_w_inteff_i[idxs]+idxs*0.2e8)
            # #print('Saving', outdir+'/tmp.png ...')
            # figtmp.savefig(outdir+'/tmp.png')
            # #quit()

            # # Plot of outer-most separatrix
            # figtmp = plt.figure()
            # figtmp.set_size_inches(60.0,8.0)
            # plt.plot(outmost_sepx_i,  outmost_sepy_i)
            # #plt.plot(outmost_sepx_i, -outmost_sepy_i)
            # figtmp.savefig(f'{outdir}/plot_separatrix_outmost_vs_time.png')
            # #quit()

        outmost_sepx_i = np.ascontiguousarray(outmost_sepx_i)
        outmost_sepy_i = np.ascontiguousarray(outmost_sepy_i)

        gc.collect()

        return outmost_sepx_i, outmost_sepy_i

###############################################################################

