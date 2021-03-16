#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

###############################################################################
# GENERAL PYTHON IMPORTS
###############################################################################

import gc
import sys
import numpy as np
import pathlib

###############################################################################
# BLOND AND BLOND-BASED IMPORTS
###############################################################################

# BLOND =======================================================================

from blond.impedances.impedance import InducedVoltageFreq, InducedVoltageTime
from blond.impedances.impedance_sources import InputTable

# CUSTOM MODULES ==============================================================

# BLonD-based tools -----------------------------------------------------------

from blond.toolbox.input_output_tools import print_object_attributes
from blond.impedances.impedance_tools_sps_reduction import reduce_impedance_feedforward_feedback #, remove_TWCs

# Tools for SPS impedance (original from JR)-----------------------------------

dirhome = str(pathlib.Path.home())

if(  dirhome.startswith('/afs')  ):       myenv = 'afs'       # RUNNING LOCALLY (AFS)
elif(dirhome.startswith('/pool') ):       myenv = 'batch'     # RUNNING WITH HTCONDOR
elif(dirhome.startswith('/hpcscratch') ): myenv = 'hpc'       # RUNNING WITH SLURM
elif(dirhome.startswith('/home') ):       myenv = 'Ubuntu'    # RUNNING LOCALLY (UBUNTU)
elif(dirhome.startswith('/Users')):       myenv = 'Mac'       # RUNNING LOCALLY (MAC)

if(  myenv == 'afs'):     dirhome = '/afs/cern.ch/work/l/lmedinam' # When running locally in AFS, re-assign dirhome so that we use the original version of the scripts in AFS work
elif(myenv == 'batch'):   dirhome = '/afs/cern.ch/work/l/lmedinam' # When running with HTCondor,  re-assign dirhome so that we use the original version of the scripts in AFS work (we do not transfer input files, as this way it's faster)
elif(myenv == 'hpc'):     pass                                     # When running with Slurm, no need to re-assign dirhome, as a local copy of the full BLonD_simulations exist in the Slurh home directory
elif(myenv == 'Ubuntu'):  pass                                     # When running with Slurm, no need to re-assign dirhome. The BLonD_simulations directory must exist there
elif(myenv == 'Mac'):     pass                                     # When running with Slurm, no need to re-assign dirhome. The BLonD_simulations directory must exist there
else:                     sys.exit('\n[!] ERROR in plot_beam_lm: NOT IMPLEMENTED!\n')

if myenv in ['afs', 'batch', 'hpc']:
    import matplotlib as mpl
    mpl.use('Agg')

dirimp = f'{dirhome}/BLonD_simulations/sps_lhc_losses/sps_lhc_losses/rep/Impedance'

sys.path.insert(0, f'{dirimp}/SPS_jr/SPS-clean-up')
import impedance_scenario

###############################################################################

class ImpedanceTools(object):

    # induced_voltage_sps =====================================================

    def induced_voltage_sps(llrf, imp, modelDir, rfstation_obj, beam_obj, profile_obj, freqRes, freqtime='freq', Gfb=None, gff=None, outdir=None, okprintobjattr=False): # keepTWCs=True,

        # You might one to remove the TWcs to create inducedvoltagefreq (and, in turn, totalinducedvoltage) object(s) w/o them for the tracking stage of the simulation when the new OTFB (which internally performs its own voltage calculation from the TWCs) is included
        # On the other hand, you might one to keep them to generate the matched beam including their effect on the total induced voltage, as the effect of OTFB only comes afterwards in the tracking.

        #if(('OTFB' in llrf) and (keepTWCs == False)):
        if('OTFB' in str(llrf)):
            imp += '_noMain200TWC'
        if okprintobjattr: print(f'Loading imp = {imp}.txt...', end='\n\n')

        # Load SPS scenario ---------------------------------------------------

        impedancescenario = impedance_scenario.scenario(f'{imp}.txt', folder=f'{modelDir}/')   # The model directory, up to one level before the last direcory 'scenarios'
        print_object_attributes(impedancescenario, ok=okprintobjattr, onlyattr=True)
        #print('Showing only selected attributes...')
        #print('impedancescenario.scenarioFileName =', impedancescenario.scenarioFileName)
        #print('impedancescenario.baseFolder       =', impedancescenario.baseFolder)
        #print('impedancescenario.scenarioFolder   =', impedancescenario.scenarioFolder)
        ##print('impedancescenario.table_impedance =', '...', len(impedancescenario.table_impedance))
        ##print('impedancescenario.table_impedance[0] =', impedancescenario.table_impedance[0])

        if(str(llrf) == 'FBFF'):
            reduce_impedance_feedforward_feedback(profile_obj, freqRes, impedancescenario, outdir=outdir, Gfb=Gfb, gff=gff)  # Defaults in this function: Gfb=10. and gff=0.5, with default Gfb changed 7.5 to 10. on 05.Aug.2020
        #elif(('OTFB' in llrf) and (keepTWCs == False)):
        #    remove_TWCs(impedancescenario)
        else:
            #print('\nNo static impedance reduction.
            pass
        ##print('impedancescenario.table_impedance[0] =',impedancescenario.table_impedance[0])

        # Convert scenario into BLonD model -----------------------------------

        impedancescenarioblond = impedance_scenario.impedance2blond(impedancescenario.table_impedance)
        print_object_attributes(impedancescenarioblond, ok=okprintobjattr, onlyattr=True)
        # Show impedance list:
        #print('impedancescenarioblond.impedanceList =')
        # for i in range(len(impedancescenarioblond.impedanceList)):
        #     #print(i, impedancescenarioblond.impedanceList[i])
        #     pass

        if(str(llrf) == 'FBFF'):
            # A new element was included: the sum of the cavities with the FB and/or FF reduction
            #print('\n[!] New element in impedance list with the sum of the cavities with', llrf, 'reduction:\n')
            print_object_attributes(impedancescenarioblond.impedanceList[-1], ok=okprintobjattr)

        print_object_attributes(impedancescenario.table_impedance[0]['impedance'], ok=okprintobjattr)

        gc.collect()

        # Induced voltage calculated by the 'frequency' method ----------------

        if(freqtime == 'freq'):

            inducedvoltagefreq_full = InducedVoltageFreq(beam_obj,
                                                         profile_obj,
                                                         impedancescenarioblond.impedanceList,
                                                         freqRes,
                                                         RFParams=rfstation_obj) # Note that InducedVoltageFreq results are scaled by the bin_size
            inducedvoltagefreq_full.freq            = np.ascontiguousarray(inducedvoltagefreq_full.freq)
            inducedvoltagefreq_full.total_impedance = np.ascontiguousarray(inducedvoltagefreq_full.total_impedance)
            print_object_attributes(inducedvoltagefreq_full, ok=okprintobjattr)

            # InputTable to speed-up  - - - - - - - - - - - - - - - - - - - - -

            impedancesources_inputtable = InputTable(inducedvoltagefreq_full.freq,
                                                     inducedvoltagefreq_full.total_impedance.real*profile_obj.bin_size,
                                                     inducedvoltagefreq_full.total_impedance.imag*profile_obj.bin_size)
            impedancesources_inputtable.frequency_array_loaded = np.ascontiguousarray(impedancesources_inputtable.frequency_array_loaded)
            impedancesources_inputtable.Re_Z_array_loaded      = np.ascontiguousarray(impedancesources_inputtable.Re_Z_array_loaded)
            impedancesources_inputtable.Im_Z_array_loaded      = np.ascontiguousarray(impedancesources_inputtable.Im_Z_array_loaded)
            impedancesources_inputtable.impedance_loaded       = np.ascontiguousarray(impedancesources_inputtable.impedance_loaded)
            print_object_attributes(impedancesources_inputtable, ok=okprintobjattr)

            # InducedVoltageFreq object - - - - - - - - - - - - - - - - - - - -

            inducedvoltagefreq = InducedVoltageFreq(beam_obj,
                                                    profile_obj,
                                                    [impedancesources_inputtable],
                                                    freqRes, #)
                                                    RFParams=rfstation_obj) # InducedVoltageFreq gives returns scaled by the bin_size
            inducedvoltagefreq.freq            = np.ascontiguousarray(inducedvoltagefreq.freq)
            inducedvoltagefreq.total_impedance = np.ascontiguousarray(inducedvoltagefreq.total_impedance)
            print_object_attributes(inducedvoltagefreq, ok=okprintobjattr)

            gc.collect()

            return inducedvoltagefreq

        # Induced voltage calculated by the 'time' method ---------------------

        elif(freqtime == 'time'):

            wakeLength = profile_obj.cut_right - profile_obj.cut_left
            inducedvoltagetime_full = InducedVoltageTime(beam_obj,
                                                         profile_obj,
                                                         impedancescenarioblond.wakeList,
                                                         wake_length=wakeLength,
                                                         RFParams=rfstation_obj)
            print_object_attributes(inducedvoltagetime_full, ok=okprintobjattr)

            # InputTable to speed-up  - - - - - - - - - - - - - - - - - - - - -

            inducedvoltagetime_inputtable = InputTable(inducedvoltagetime_full.time,
                                                       inducedvoltagetime_full.total_wake)
            print_object_attributes(inducedvoltagetime_inputtable, ok=okprintobjattr)

            # InducedVoltageTime object - - - - - - - - - - - - - - - - - - - -

            inducedvoltagetime = InducedVoltageTime(beam_obj,
                                                    profile_obj,
                                                    [inducedvoltagetime_inputtable],
                                                    RFParams=rfstation_obj)
            inducedvoltagetime.time       = np.ascontiguousarray(inducedvoltagetime.time)
            inducedvoltagetime.total_wake = np.ascontiguousarray(inducedvoltagetime.total_wake)
            print_object_attributes(inducedvoltagetime, ok=okprintobjattr)

            # if outdir is not None:

                # # Save impedance (wake) from time method  - - - - - - - - - - - -

                # total_wake_vs_time = np.c_[inducedvoltagetime.time, inducedvoltagetime.total_wake]
                # mynpsave(f'{outdir}/wake_vs_time', total_wake_vs_time)

                # #inducedvoltagetime.process()

                # # PLOT

                # plot_wake_vs_time(inducedvoltagetime.time,
                #                   inducedvoltagetime.total_wake,
                #                   imp.replace('_', '-'),
                #                   outdir)

            gc.collect()

            return inducedvoltagetime

        else:

            sys.error(f'[!] ERROR: freqtime = {freqtime} not an option!')


    # induced_voltage_lhc =====================================================

    def induced_voltage_lhc(imp, modelDir, freqcut, rfstation_obj, beam_obj, profile_obj, freqRes, freqtime='freq', outdir=None, okprintobjattr=False): # outdir not really used, kept just for symmetry with sps

        # Induced voltage calculated by the 'frequency' method ----------------

        if(freqtime == 'freq'):

            impedancetable_freq_full, impedancetable_ReZlong_full, impedancetable_ImZlong_full = np.loadtxt(f'{modelDir}/scenarios/{imp}.dat', delimiter="\t", skiprows=1, unpack=True) #, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
            impedancetable_freq_full    = np.ascontiguousarray(impedancetable_freq_full)
            impedancetable_ReZlong_full = np.ascontiguousarray(impedancetable_ReZlong_full)
            impedancetable_ImZlong_full = np.ascontiguousarray(impedancetable_ImZlong_full)
            #print('* impedancetable_freq_full    =', impedancetable_freq_full,   len(impedancetable_freq_full))
            #print('* impedancetable_ReZlong_full =', impedancetable_ReZlong_full)
            #print('* impedancetable_ImZlong_full =', impedancetable_ImZlong_full)
            #print(f'freqcut = {MAC["freqcut"]:.1e}')
            #print('')

            # Remove frequencies higher than cut-off, to speed  - - - - - - - -

            if(freqcut is not None):
                #myidxtocut = 60000
                myidxtocut = np.argmax(impedancetable_freq_full > freqcut) # 13.5e9 to be the same than the cut I did for present LHC
                #print('myidxtocut =', myidxtocut, impedancetable_freq_full[myidxtocut])
                #quit()

                #if(  imp == 'Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1'): myidxtocut = 60000 # at idx = 55300, the frequency in the model array is 10.0041925 GHz, that is, f >~ 10e9. I've used 60000 for a round number, corresponding to 13.5138052 GHz
                #elif(imp == 'Zlong_Allthemachine_450GeV_B1_HLLHC_inj_450GeV_TDIS-Gr_55.0mm_5umMo+MoC_IP7'): myidxtocut = 0 # 677498 gives the threshold above 10e9: 10.000207139 GHz. I can use 686197 to give the same limit used for the present LHC: 13.515075611 GHz

                idxtocut = np.arange(myidxtocut, len(impedancetable_freq_full)).astype(int)

                impedancetable_freq_cut    = np.delete(impedancetable_freq_full,    idxtocut, 0)
                impedancetable_ReZlong_cut = np.delete(impedancetable_ReZlong_full, idxtocut, 0)
                impedancetable_ImZlong_cut = np.delete(impedancetable_ImZlong_full, idxtocut, 0)
                #impedancetable_freq_cut    = np.copy(impedancetable_freq_full)
                #impedancetable_ReZlong_cut = np.copy(impedancetable_ReZlong_full)
                #impedancetable_ImZlong_cut = np.copy(impedancetable_ImZlong_full)
                #print('* impedancetable_freq_cut    =', impedancetable_freq_cut,   len(impedancetable_freq_cut))
                #print('* impedancetable_ReZlong_cut =', impedancetable_ReZlong_cut)
                #print('* impedancetable_ImZlong_cut =', impedancetable_ImZlong_cut)
                #print('')
            else:
                impedancetable_freq_cut    = np.copy(impedancetable_freq_full)
                impedancetable_ReZlong_cut = np.copy(impedancetable_ReZlong_full)
                impedancetable_ImZlong_cut = np.copy(impedancetable_ImZlong_full)

            del(impedancetable_freq_full)
            del(impedancetable_ReZlong_full)
            del(impedancetable_ImZlong_full)

            impedancetable_freq    = np.arange(0.0, impedancetable_freq_cut[-1], freqRes)
            impedancetable_ReZlong = np.interp(impedancetable_freq, impedancetable_freq_cut, impedancetable_ReZlong_cut)
            impedancetable_ImZlong = np.interp(impedancetable_freq, impedancetable_freq_cut, impedancetable_ImZlong_cut)
            impedancetable_freq    = np.ascontiguousarray(impedancetable_freq)
            impedancetable_ReZlong = np.ascontiguousarray(impedancetable_ReZlong)
            impedancetable_ImZlong = np.ascontiguousarray(impedancetable_ImZlong)
            #print('* impedancetable_freq    =', impedancetable_freq,   len(impedancetable_freq))
            #print('* impedancetable_ReZlong =', impedancetable_ReZlong)
            #print('* impedancetable_ImZlong =', impedancetable_ImZlong)
            #print('')
            #quit()
           #impedancetable_ReZlong = impedancetable_ReZlong #/impedancetable_freq*ring.f_rev[0]
           #impedancetable_ImZlong = impedancetable_ImZlong #/impedancetable_freq*ring.f_rev[0]
            impedancetable_ReZlong[0] = 0e0
            impedancetable_ImZlong[0] = 0e0
            #print('* impedancetable_ReZlong =', impedancetable_ReZlong)
            #print('* impedancetable_ImZlong =', impedancetable_ImZlong)
            #print('')
            #quit()
            del(impedancetable_freq_cut)
            del(impedancetable_ReZlong_cut)
            del(impedancetable_ImZlong_cut)

            # InputTable to speed-up  - - - - - - - - - - - - - - - - - - - - -

            print(impedancetable_freq)

            impedancesources_inputtable = InputTable(impedancetable_freq,    # inducedvoltagefreq_full.freq
                                                     impedancetable_ReZlong, # inducedvoltagefreq_full.total_impedance.real*profile.bin_size,
                                                     impedancetable_ImZlong) # inducedvoltagefreq_full.total_impedance.imag*profile.bin_size,
            impedancesources_inputtable.frequency_array_loaded = np.ascontiguousarray(impedancesources_inputtable.frequency_array_loaded)
            impedancesources_inputtable.Re_Z_array_loaded      = np.ascontiguousarray(impedancesources_inputtable.Re_Z_array_loaded)
            impedancesources_inputtable.Im_Z_array_loaded      = np.ascontiguousarray(impedancesources_inputtable.Im_Z_array_loaded)
            impedancesources_inputtable.impedance_loaded       = np.ascontiguousarray(impedancesources_inputtable.impedance_loaded)
            print_object_attributes(impedancesources_inputtable, ok=okprintobjattr)

            gc.collect()

            # InducedVoltageFreq object - - - - - - - - - - - - - - - - - - - -

            inducedvoltagefreq = InducedVoltageFreq(beam_obj,
                                                    profile_obj,
                                                    [impedancesources_inputtable],
                                                    freqRes, #)
                                                    RFParams=rfstation_obj) # InducedVoltageFreq gives returns scaled by the bin_size
            inducedvoltagefreq.freq            = np.ascontiguousarray(inducedvoltagefreq.freq)
            inducedvoltagefreq.total_impedance = np.ascontiguousarray(inducedvoltagefreq.total_impedance)
            print_object_attributes(inducedvoltagefreq, ok=okprintobjattr)

            print(inducedvoltagefreq.freq)

            gc.collect()

            return inducedvoltagefreq

        # Induced voltage calculated by the 'time' method ---------------------

        elif(freqtime == 'time'):

            sys.error(f'[!] ERROR: freqtime = {freqtime} not an option!')

        else:

            sys.error(f'[!] ERROR: freqtime = {freqtime} not an option!')

###############################################################################
