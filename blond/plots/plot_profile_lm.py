#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import gc
import sys
import numpy as np
import h5py as hp
import pickle
import os
import pathlib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator #  AutoLocator, FormatStrFormatter,
from cycler import cycler

# BLonD-based tools
from blond.toolbox.input_output_tools import mynpload

#-------------

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
else:                   sys.exit('\n[!] ERROR in plot_profile_lm: NOT IMPLEMENTED!\n')

dirbase = 'BLonD_simulations/sps_lhc_losses'
dirin   = f'{dirhome}/{dirbase}/sps_lhc_losses'
dirinp  = f'{dirin}/inp'
dirinpbench  = f'{dirinp}/benchmark'

#-------------

###############################################################################

# Profile macroparticles OR stats vs turn -------------------------------------

def plot_profile_vs_turn(attr, outdir, scalefactor=None, blmeas=None):  # Formally, we should pass NbperNp_list_0 instead of NbperNp_tot_0

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_Profile_turns = h5File['/Profile/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_Profile_turns)
        data_Profile_turns = h5File['/Profile/turns'][:idxlastturn+1]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if(  attr == 'nmacrop'):
            opt_list = ['BF', 'BS', 'BG']
            key_list = ['alive', 'lost']  # Order is important, lave should come before lost!
            n_key_list = int(len(opt_list)*len(key_list))
        elif(attr == 'stats'):
            opt_list = ['FWHM', 'RMS', 'BRF']
            FWHMRMS_key_list = ['bunchPosition', 'bunchPositionOff_ctr', 'bunchPositionOff_fit', 'bunchLength', 'bunchEnergySpread', 'bunchEmittance']
            BRF_key_list     = ['bunchPositionOff', 'bunchFormFactor']
            n_key_list = int( (len(opt_list)-1)*len(FWHMRMS_key_list) + 1*len(BRF_key_list))

        fig, ax1 = plt.subplots(n_key_list, 2, sharex=True)
        #ax2 = ax1.twinx() # [ax1[spi, 0].twinx() for spi in range(n_key_list)]
        ax2 = [[ax1[spi,spj].twinx() for spj in range(2)] for spi in range(n_key_list)]
        fig.set_size_inches(2*8.0,n_key_list*2.0)

        gc.collect()

        spi = 0
        for opt in opt_list:
            if(attr == 'stats'):
                if opt == 'BRF': key_list = BRF_key_list
                else:            key_list = FWHMRMS_key_list
            for key in key_list:

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                data_Profile_bunch_opt_key_fst = h5File[f'/Profile/bunch/{opt}/{key}'][0]
                data_Profile_beam_opt_key_fst  = h5File[f'/Profile/beam/{opt}/{key}' ][0]

                data_Profile_bunch_opt_key = h5File[f'/Profile/bunch/{opt}/{key}'][:idxlastturn+1]
                data_Profile_beam_opt_key  = h5File[f'/Profile/beam/{opt}/{key}' ][:idxlastturn+1]

                nbf = len(data_Profile_bunch_opt_key[0])

                # - - - - - - - - - - - - - - - - -

                # if(attr == 'nmacrop'): # nmacrop (alive and lost) for beam will be plotted as the average over all bunches
                #     data_Profile_beam_opt_key_fst = data_Profile_beam_opt_key_fst.astype(float)
                #     data_Profile_beam_opt_key_fst = data_Profile_beam_opt_key_fst/nbf
                #     data_Profile_beam_opt_key = data_Profile_beam_opt_key.astype(float)
                #     data_Profile_beam_opt_key = data_Profile_beam_opt_key/nbf

                if(opt == 'BF' and key == 'alive'): # Needed for plotting 'lost' in percent:
                    data_Profile_bunch_BF_alv_fst = np.copy(data_Profile_bunch_opt_key_fst)
                    data_Profile_beam_BF_alv_fst  = np.copy(data_Profile_beam_opt_key_fst)
                    # data_Profile_bunch_BF_alv = np.copy(data_Profile_bunch_opt_key)
                    # data_Profile_beam_BF_alv  = np.copy(data_Profile_beam_opt_key)

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                mycm = plt.get_cmap('coolwarm')
                if(nbf > 1): color_cycle_nb = [mycm(nb/(nbf-1.)) for nb in range(nbf)]
                else:        color_cycle_nb = [mycm(0.)]

                if(len(data_Profile_turns) == 2): mymarker = '.' # Turns -1 and 0
                else:                             mymarker = '-'

                ax1[spi, 0].set_prop_cycle(cycler('color', color_cycle_nb))
                ax1[spi, 1].set_prop_cycle(cycler('color', color_cycle_nb))
                if(attr == 'nmacrop'):
                    ax2[spi][0].set_prop_cycle(cycler('color', color_cycle_nb))
                    ax2[spi][1].set_prop_cycle(cycler('color', color_cycle_nb))

                ax1[spi, 0].grid()
                ax1[spi, 1].grid()

                if(spi == n_key_list-1):
                    ax1[spi, 0].set_xlabel('Turns')
                    ax1[spi, 1].set_xlabel('Turns')

                if(  key == 'alive'):
                    unitfact_spi_0_ax1 = 1
                    unitfact_spi_0_ax2 = 1e11
                    if(  opt == 'BF'):
                        ax1[spi, 0].set_ylabel('Alive macrop\n(BF)')
                        ax2[spi][0].set_ylabel('Intensity\n[$10^{11}$ ppb]')
                    elif(opt == 'BS'):
                        ax1[spi, 0].set_ylabel('Alive macrop\n(BS)')
                        ax2[spi][0].set_ylabel('Satellites\n[$10^{11}$ ppb]')
                    elif(opt == 'BG'):
                        ax1[spi, 0].set_ylabel('Alive macrop\n(BG)')
                        ax2[spi][0].set_ylabel('Ghosts\n[$10^{11}$ ppb]')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    if(  opt == 'BF'):
                        ax1[spi, 1].set_ylabel('Alive macrop\ngrowth (BF) [%]')
                        ax2[spi][1].set_ylabel('Intensity\ngrowth [%]')
                    elif(opt == 'BS'):
                        ax1[spi, 1].set_ylabel('Alive macrop\ngrowth (BS)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Satellites\ngrowth [% w.r.t tot.]')
                    elif(opt == 'BG'):
                        ax1[spi, 1].set_ylabel('Alive macrop\ngrowth (BG)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Ghosts\ngrowth [% w.r.t tot.]')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                elif(key == 'lost'):
                    unitfact_spi_0_ax1 = 1
                    unitfact_spi_0_ax2 = 1e9
                    if(  opt == 'BF'):
                        ax1[spi, 0].set_ylabel('Lost macrop\n(BF)')
                        ax2[spi][0].set_ylabel('Lost intensity\n[$10^{9}$ ppb]')
                    elif(opt == 'BS'):
                        ax1[spi, 0].set_ylabel('Lost macrop\n(BS)')
                        ax2[spi][0].set_ylabel('Lost satellites\n[$10^{9}$ ppb]')
                    elif(opt == 'BG'):
                        ax1[spi, 0].set_ylabel('Lost macrop\n(BG)')
                        ax2[spi][0].set_ylabel('Lost ghosts\n[$10^{9}$ ppb]')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    if(  opt == 'BF'):
                        ax1[spi, 1].set_ylabel('Lost macrop\ngrowth (BF) [%]')
                        ax2[spi][1].set_ylabel('Lost intensity\ngrowth [%]')
                    elif(opt == 'BS'):
                        ax1[spi, 1].set_ylabel('Lost macrop\ngrowth (BS)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Lost satellites\ngrowth [% w.r.t tot.]')
                    elif(opt == 'BG'):
                        ax1[spi, 1].set_ylabel('Lost macrop\ngrowth (BG)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Lost ghosts\ngrowth [% w.r.t tot.]')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                elif(key == 'bunchPosition'): # eq. mean_dt
                    unitfact_spi_0_ax1 = 1e-9
                    unitfact_spi_0_ax2 = 1e3
                    ax1[spi, 0].set_ylabel('Position\n('+opt+') [ns]')
                    ax2[spi][0].set_ylabel('Position\n('+opt+') [$10^3$ deg]')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    #
                    unitfact_spi_1_ax1 = 1e-12
                    unitfact_spi_1_ax2 = 1.
                    ax1[spi, 1].set_ylabel('Position diff\n('+opt+') [ps]')
                    ax2[spi][1].set_ylabel('Position diff\n('+opt+') [deg]')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                elif('bunchPositionOff' in key):
                    unitfact_spi_0_ax1 = 1e-12
                    unitfact_spi_0_ax2 = 1.
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                    unitfact_spi_1_ax1 = 1e-12
                    unitfact_spi_1_ax2 = 1.
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    if('ctr' in key): # eq. mean_dtOff_ctr
                        ax1[spi, 0].set_ylabel('Pos off wrt ctr\n('+opt+') [ps]')
                        ax2[spi][0].set_ylabel('Pos off wrt ctr\n('+opt+') [deg]')
                        ax1[spi, 1].set_ylabel('Pos off wrt ctr\ndiff ('+opt+') [ps]')
                        ax2[spi][1].set_ylabel('Pos off wrt ctr\ndiff ('+opt+') [deg]')
                    elif('fit' in key): # eq. mean_dtOff_fit
                        ax1[spi, 0].set_ylabel('Pos off wrt fit\n('+opt+') [ps]')
                        ax2[spi][0].set_ylabel('Pos off wrt fit\n('+opt+') [deg]')
                        ax1[spi, 1].set_ylabel('Pos off wrt fit\ndiff ('+opt+') [ps]')
                        ax2[spi][1].set_ylabel('Pos off wrt fit\ndiff ('+opt+') [deg]')
                    else: # i.e. elif('brf' in key), no equiv in beam
                        ax1[spi, 0].set_ylabel('Pos off wrt brf\n[ps]')
                        ax2[spi][0].set_ylabel('Pos off wrt brf\n[deg]')
                        ax1[spi, 1].set_ylabel('Pos off wrt brf\ndiff [ps]')
                    #
                elif(key == 'bunchLength'): # eq. sigma_dt
                    unitfact_spi_0_ax1 = 1e-9
                    unitfact_spi_0_ax2 = 1e-9
                    ax1[spi, 0].set_ylabel('Length\n('+opt+') [ns]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Length growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                elif(key == 'bunchEnergySpread'): # eq. sigma_dE
                    unitfact_spi_0_ax1 = 1e6
                    unitfact_spi_0_ax2 = 1e6
                    ax1[spi, 0].set_ylabel('Energy spread\n('+opt+') [MeV]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Energy spread growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                elif(key == 'bunchEmittance'): # eq. epsnrmsl
                    unitfact_spi_0_ax1 = 1.
                    unitfact_spi_0_ax2 = 1.
                    ax1[spi, 0].set_ylabel('Emittance\n('+opt+') [eV$\cdot$s]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Emittance growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                elif(key == 'bunchFormFactor'): # no equiv in beam
                    unitfact_spi_0_ax1 = 1.
                    unitfact_spi_0_ax2 = 1.
                    ax1[spi, 0].set_ylabel('Bunch form fact\n[eV$\cdot$s]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Form fact growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                # BUNCH:

                for nb in range(nbf):
                    if(nb == 0 or nb == nbf-1):
                        label_nb = f'Bunch {nb}'
                        if(opt == 'BG' and nb == 0):     label_nb += ' (L)'
                        if(opt == 'BG' and nb == nbf-1): label_nb += ' (R)'
                    else:
                        label_nb = None

                    ax1[spi, 0].plot(data_Profile_turns, data_Profile_bunch_opt_key[:,nb]/unitfact_spi_0_ax1, mymarker, alpha=0.20, label=label_nb)

                    if(attr == 'nmacrop'):
                        if(opt == 'BF'):
                            if(key == 'alive'):
                                # Percent growth w.r.t. its own value before tracking (turn -1)
                                ax1[spi, 1].plot(data_Profile_turns, data_Profile_bunch_opt_key[:,nb]/data_Profile_bunch_opt_key_fst[nb]/unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)
                            else:
                                # Percent growth w.r.t. "bunch-alive" value before tracking (turn -1)
                                ax1[spi, 1].plot(data_Profile_turns, data_Profile_bunch_opt_key[:,nb]/data_Profile_bunch_BF_alv_fst[nb] /unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)
                        else:
                            # Percent growth w.r.t. "beam-alive" value before tracking (turn -1)
                            ax1[spi, 1].plot(data_Profile_turns, data_Profile_bunch_opt_key[:,nb]/data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)

                    else:
                        if('Position' in key):
                            # Difference growth w.r.t. its own value before tracking (turn -1)
                            ax1[spi, 1].plot(data_Profile_turns, (data_Profile_bunch_opt_key[:,nb]-data_Profile_bunch_opt_key_fst[nb])/unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)
                        else:
                            # Percent growth w.r.t. its own value before tracking (turn -1)
                            ax1[spi, 1].plot(data_Profile_turns,  data_Profile_bunch_opt_key[:,nb]/data_Profile_bunch_opt_key_fst[nb] /unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)

                # BEAM:

                if(attr == 'nmacrop'):
                    if(opt == 'BF'):
                        ax1[spi, 0].plot(data_Profile_turns, (data_Profile_beam_opt_key/nbf)/unitfact_spi_0_ax1, mymarker+'-k', label='Bunch ave.') # instead of 'beam'
                        if(key == 'alive'): ax1[spi, 1].plot(data_Profile_turns, (data_Profile_beam_opt_key/nbf)/(data_Profile_beam_opt_key_fst/nbf)/unitfact_spi_1_ax1, mymarker+'-k', label='Bunch ave.')
                        else:               ax1[spi, 1].plot(data_Profile_turns, (data_Profile_beam_opt_key/nbf)/(data_Profile_beam_BF_alv_fst /nbf)/unitfact_spi_1_ax1, mymarker+'-k', label='Bunch ave.')
                    else:
                        ax1[spi, 0].plot(data_Profile_turns,  data_Profile_beam_opt_key     /unitfact_spi_0_ax1, mymarker+'-k', label='Total')
                        ax1[spi, 1].plot(data_Profile_turns, data_Profile_beam_opt_key/data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, mymarker+'-k', label='Total')
                else:
                    ax1[spi, 0].plot(data_Profile_turns,  data_Profile_beam_opt_key     /unitfact_spi_0_ax1, mymarker+'-k', label='Bunch ave.')
                    if('Position' in key):
                        ax1[spi, 1].plot(data_Profile_turns, (data_Profile_beam_opt_key-data_Profile_beam_opt_key_fst)/unitfact_spi_1_ax1, mymarker+'-k', label='Bunch ave.')
                    else:
                        ax1[spi, 1].plot(data_Profile_turns,  data_Profile_beam_opt_key/data_Profile_beam_opt_key_fst /unitfact_spi_1_ax1, mymarker+'-k', label='Bunch ave.')

                # SPECIAL CASES:

                # if (key == 'bunchPosition' and len(data_Profile_turns) > 2 and bucket_centres_i is not None):
                #     for nb in range(nbf):
                #         ax1[spi, 0].plot(data_Profile_turns, np.ones(len(data_Profile_turns))*bucket_centres_i[nb]/unitfact_spi_0_ax1, mymarker+'.', color='#888888')
                if(key == 'bunchLength' and opt == 'FWHM' and blmeas is not None):
                    ax1[spi, 0].axhline(blmeas/unitfact_spi_0_ax1, ls='-.', color='#00cc00')
                   #ax1[spi, 1].axhline(100.,                 ls='-.', color='#00cc00')

                # SECONDARY AXIS:

                ax1_spi_0_yticks = ax1[spi, 0].get_yticks()
                ax1[spi][0].set_ylim( ax1_spi_0_yticks[0], ax1_spi_0_yticks[-1] )
                if(unitfact_spi_0_ax1 == unitfact_spi_0_ax2): scalefactor_tmp = 1
                else:                                         scalefactor_tmp = scalefactor
                ax2_spi_0_yticks = ax1_spi_0_yticks*unitfact_spi_0_ax1 * scalefactor_tmp /unitfact_spi_0_ax2
                ax2[spi][0].set_ylim( ax2_spi_0_yticks[0], ax2_spi_0_yticks[-1] )
                ax2[spi][0].set_yticks(ax2_spi_0_yticks)
                #print(f'key: {key}: {ax1_spi_0_yticks} (unit: {unitfact_spi_0_ax1}) -> (scale: {scalefactor_tmp}) -> {ax2_spi_0_yticks} (unit: {unitfact_spi_0_ax2})')
                #
                ax1_spi_1_yticks = ax1[spi, 1].get_yticks()
                ax1[spi][1].set_ylim( ax1_spi_1_yticks[0], ax1_spi_1_yticks[-1] )
                if(unitfact_spi_1_ax1 == unitfact_spi_1_ax2): scalefactor_tmp = 1
                else:                                         scalefactor_tmp = scalefactor
                ax2_spi_1_yticks = ax1_spi_1_yticks*unitfact_spi_1_ax1 * scalefactor_tmp /unitfact_spi_1_ax2
                ax2[spi][1].set_ylim( ax2_spi_1_yticks[0], ax2_spi_1_yticks[-1] )
                ax2[spi][1].set_yticks(ax2_spi_1_yticks)
                #print(f'key: {key}: {ax1_spi_1_yticks} (unit: {unitfact_spi_1_ax1}) -> (scale: {scalefactor_tmp}) -> {ax2_spi_1_yticks} (unit: {unitfact_spi_1_ax2})')

                # LEGEND:

                if(attr == 'nmacrop'):
                    if(spi == 0):
                        ax1[spi, 0].legend(loc=3, ncol=3)
                else:
                    if(spi == 0):
                        ax1[spi, 0].legend(loc=2, ncol=3)

                spi += 1
                gc.collect()

        fig.tight_layout()
        fig.savefig(f'{outdir}/plot_profile_{attr}_vs_turn.png')
        plt.cla()
        plt.close(fig)

    try:    h5File.close()
    except: pass


# Profile macroparticles OR stats vs bunch no. --------------------------------

def plot_profile_vs_bn(attr, outdir, turn_bn=None, scalefactor=None, blmeas=None):  # Formally, we should pass NbperNp_list_0 instead of NbperNp_tot_0

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_Profile_turns = h5File['/Profile/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_Profile_turns)
        lastturn = data_Profile_turns[idxlastturn]
        idxzeroturn = 0
        zeroturn = data_Profile_turns[idxzeroturn]
        data_Profile_turns = h5File['/Profile/turns'][:idxlastturn+1]

        if(turn_bn is not None):
            # if(lastturn > 5000): turn_bn = 5000
            # idxfirstturn = np.argmax(data_Profile_turns >= turn_bn)
            # #print(f'data_Profile_turns = {data_Profile_turns}')
            # #print(f'idxfirstturn = {idxfirstturn}')
            if(lastturn > turn_bn):
                idxfirstturn = np.argmax(data_Profile_turns >= turn_bn)
            else:
                idxfirstturn = 0
        else:
            idxfirstturn = 0
        firstturn = data_Profile_turns[idxfirstturn]
        data_Profile_turns = h5File['/Profile/turns'][firstturn:idxlastturn+1]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if(  attr == 'nmacrop'):
            opt_list = ['BF', 'BS', 'BG']
            key_list = ['alive', 'lost']  # Order is important, lave should come before lost!
            n_key_list = int(len(opt_list)*len(key_list))
        elif(attr == 'stats'):
            opt_list = ['FWHM', 'RMS', 'BRF']
            FWHMRMS_key_list = ['bunchPosition', 'bunchPositionOff_ctr', 'bunchPositionOff_fit', 'bunchLength', 'bunchEnergySpread', 'bunchEmittance']
            BRF_key_list     = ['bunchPositionOff', 'bunchFormFactor']
            n_key_list = int( (len(opt_list)-1)*len(FWHMRMS_key_list) + 1*len(BRF_key_list))

        fig, ax1 = plt.subplots(n_key_list, 2, sharex=True)
        #ax2 = ax1.twinx() # [ax1[spi, 0].twinx() for spi in range(n_key_list)]
        ax2 = [[ax1[spi,spj].twinx() for spj in range(2)] for spi in range(n_key_list)]
        fig.set_size_inches(2*8.0,n_key_list*2.0)

        gc.collect()

        spi = 0
        for opt in opt_list:
            if(attr == 'stats'):
                if opt == 'BRF': key_list = BRF_key_list
                else:            key_list = FWHMRMS_key_list
            for key in key_list:

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                data_Profile_bunch_opt_key_fst = h5File[f'/Profile/bunch/{opt}/{key}'][0]
                data_Profile_beam_opt_key_fst  = h5File[f'/Profile/beam/{opt}/{key}'][0]

                data_Profile_bunch_opt_key = h5File[f'/Profile/bunch/{opt}/{key}'][idxfirstturn:idxlastturn+1]
                data_Profile_beam_opt_key  = h5File[f'/Profile/beam/{opt}/{key}'][idxfirstturn:idxlastturn+1]

                nmntturns = len(data_Profile_bunch_opt_key)
                nbf       = len(data_Profile_bunch_opt_key[0])

                # - - - - - - - - - - - - - - - - -

                data_Profile_bunch_opt_key_min = np.minimum.accumulate(data_Profile_bunch_opt_key)
                data_Profile_bunch_opt_key_max = np.maximum.accumulate(data_Profile_bunch_opt_key)
                data_Profile_bunch_opt_key_ave = np.add.accumulate(data_Profile_bunch_opt_key)/ np.arange(1,nmntturns+1).reshape((nmntturns,1)) # the divisor equivalent to np.array([ np.ones(nbf)*i for i in np.arange((1,nmntturns+1) ])
              ##data_Profile_bunch_opt_key_min = np.array([ np.array([    min(data_Profile_bunch_opt_key[:i+1,j]) for j in range(nbf)]) for i in range(nmntturns)])
              ##data_Profile_bunch_opt_key_ave = np.array([ np.array([np.mean(data_Profile_bunch_opt_key[:i+1,j]) for j in range(nbf)]) for i in range(nmntturns)])
              ##data_Profile_bunch_opt_key_max = np.array([ np.array([    max(data_Profile_bunch_opt_key[:i+1,j]) for j in range(nbf)]) for i in range(nmntturns)])

               #data_Profile_beam_opt_key_min = np.minimum.accumulate(data_Profile_beam_opt_key)
               #data_Profile_beam_opt_key_max = np.maximum.accumulate(data_Profile_beam_opt_key)
                data_Profile_beam_opt_key_ave = np.add.accumulate(data_Profile_beam_opt_key) / np.arange(1,nmntturns+1)
              ##data_Profile_beam_opt_key_min = np.array([     min(data_Profile_beam_opt_key[:i+1]) for i in range(nmntturns)])
              ##data_Profile_beam_opt_key_ave = np.array([ np.mean(data_Profile_beam_opt_key[:i+1]) for i in range(nmntturns)])
              ##data_Profile_beam_opt_key_max = np.array([     max(data_Profile_beam_opt_key[:i+1]) for i in range(nmntturns)])

                # - - - - - - - - - - - - - - - - -

                # if(attr == 'nmacrop'): # nmacrop (alive and lost) for beam will be plotted as the average over all bunches
                #     data_Profile_beam_opt_key_fst = data_Profile_beam_opt_key_fst.astype(float)
                #     data_Profile_beam_opt_key_fst = data_Profile_beam_opt_key_fst/nbf
                #     data_Profile_beam_opt_key = data_Profile_beam_opt_key.astype(float)
                #     data_Profile_beam_opt_key = data_Profile_beam_opt_key/nbf
                #     data_Profile_beam_opt_key_ave = data_Profile_beam_opt_key_ave.astype(float)
                #     data_Profile_beam_opt_key_ave = data_Profile_beam_opt_key_ave/nbf

                if(opt == 'BF' and key == 'alive'): # Needed for plotting 'lost' in percent:
                    data_Profile_bunch_BF_alv_fst = np.copy(data_Profile_bunch_opt_key_fst)
                    data_Profile_beam_BF_alv_fst  = np.copy(data_Profile_beam_opt_key_fst)
                    # data_Profile_bunch_BF_alv = np.copy(data_Profile_bunch_opt_key)
                    # data_Profile_beam_BF_alv  = np.copy(data_Profile_beam_opt_key)
                    # data_Profile_bunch_BF_alv_min = np.copy(data_Profile_bunch_opt_key_min)
                    # data_Profile_bunch_BF_alv_ave = np.copy(data_Profile_bunch_opt_key_ave)
                    # data_Profile_bunch_BF_alv_max = np.copy(data_Profile_bunch_opt_key_max)
                    # data_Profile_beam_BF_alv_min = np.copy(data_Profile_beam_opt_key_min)
                    # data_Profile_beam_BF_alv_ave = np.copy(data_Profile_beam_opt_key_ave)
                    # data_Profile_beam_BF_alv_max = np.copy(data_Profile_beam_opt_key_max)

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                ax1[spi, 0].grid()
                ax1[spi, 1].grid()

                if(spi == n_key_list-1):
                    ax1[spi, 0].set_xlabel('Bunch no.')
                    ax1[spi, 1].set_xlabel('Bunch no.')

                if(  key == 'alive'):
                    unitfact_spi_0_ax1 = 1
                    unitfact_spi_0_ax2 = 1e11
                    if(  opt == 'BF'):
                        ax1[spi, 0].set_ylabel('Alive macrop\n(BF)')
                        ax2[spi][0].set_ylabel('Intensity\n[$10^{11}$ ppb]')
                    elif(opt == 'BS'):
                        ax1[spi, 0].set_ylabel('Alive macrop\n(BS)')
                        ax2[spi][0].set_ylabel('Satellites\n[$10^{11}$ ppb]')
                    elif(opt == 'BG'):
                        ax1[spi, 0].set_ylabel('Alive macrop\n(BG)')
                        ax2[spi][0].set_ylabel('Ghosts\n[$10^{11}$ ppb]')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    if(  opt == 'BF'):
                        ax1[spi, 1].set_ylabel('Alive macrop\ngrowth (BF) [%]')
                        ax2[spi][1].set_ylabel('Intensity\ngrowth [%]')
                    elif(opt == 'BS'):
                        ax1[spi, 1].set_ylabel('Alive macrop\ngrowth (BS)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Satellites\ngrowth [% w.r.t tot.]')
                    elif(opt == 'BG'):
                        ax1[spi, 1].set_ylabel('Alive macrop\ngrowth (BG)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Ghosts\ngrowth [% w.r.t tot.]')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                elif(key == 'lost'):
                    unitfact_spi_0_ax1 = 1
                    unitfact_spi_0_ax2 = 1e9
                    if(  opt == 'BF'):
                        ax1[spi, 0].set_ylabel('Lost macrop\n(BF)')
                        ax2[spi][0].set_ylabel('Lost intensity\n[$10^{9}$ ppb]')
                    elif(opt == 'BS'):
                        ax1[spi, 0].set_ylabel('Lost macrop\n(BS)')
                        ax2[spi][0].set_ylabel('Lost satellites\n[$10^{9}$ ppb]')
                    elif(opt == 'BG'):
                        ax1[spi, 0].set_ylabel('Lost macrop\n(BG)')
                        ax2[spi][0].set_ylabel('Lost ghosts\n[$10^{9}$ ppb]')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    if(  opt == 'BF'):
                        ax1[spi, 1].set_ylabel('Lost macrop\ngrowth (BF) [%]')
                        ax2[spi][1].set_ylabel('Lost intensity\ngrowth [%]')
                    elif(opt == 'BS'):
                        ax1[spi, 1].set_ylabel('Lost macrop\ngrowth (BS)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Lost satellites\ngrowth [% w.r.t tot.]')
                    elif(opt == 'BG'):
                        ax1[spi, 1].set_ylabel('Lost macrop\ngrowth (BG)\n[% wrt total]')
                        ax2[spi][1].set_ylabel('Lost ghosts\ngrowth [% w.r.t tot.]')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                elif(key == 'bunchPosition'): # eq. mean_dt
                    unitfact_spi_0_ax1 = 1e-9
                    unitfact_spi_0_ax2 = 1e3
                    ax1[spi, 0].set_ylabel('Position\n('+opt+') [ns]')
                    ax2[spi][0].set_ylabel('Position\n('+opt+') [$10^3$ deg]')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    #
                    unitfact_spi_1_ax1 = 1e-12
                    unitfact_spi_1_ax2 = 1.
                    ax1[spi, 1].set_ylabel('Position diff\n('+opt+') [ps]')
                    ax2[spi][1].set_ylabel('Position diff\n('+opt+') [deg]')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                elif('bunchPositionOff' in key):
                    unitfact_spi_0_ax1 = 1e-12
                    unitfact_spi_0_ax2 = 1.
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                    unitfact_spi_1_ax1 = 1e-12
                    unitfact_spi_1_ax2 = 1.
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    if('ctr' in key): # eq. mean_dtOff_ctr
                        ax1[spi, 0].set_ylabel('Pos off wrt ctr\n('+opt+') [ps]')
                        ax2[spi][0].set_ylabel('Pos off wrt ctr\n('+opt+') [deg]')
                        ax1[spi, 1].set_ylabel('Pos off wrt ctr\ndiff ('+opt+') [ps]')
                        ax2[spi][1].set_ylabel('Pos off wrt ctr\ndiff ('+opt+') [deg]')
                    elif('fit' in key): # eq. mean_dtOff_fit
                        ax1[spi, 0].set_ylabel('Pos off wrt fit\n('+opt+') [ps]')
                        ax2[spi][0].set_ylabel('Pos off wrt fit\n('+opt+') [deg]')
                        ax1[spi, 1].set_ylabel('Pos off wrt fit\ndiff ('+opt+') [ps]')
                        ax2[spi][1].set_ylabel('Pos off wrt fit\ndiff ('+opt+') [deg]')
                    else: # i.e. elif('brf' in key), no equiv in beam
                        ax1[spi, 0].set_ylabel('Pos off wrt brf\n[ps]')
                        ax2[spi][0].set_ylabel('Pos off wrt brf\n[deg]')
                        ax1[spi, 1].set_ylabel('Pos off wrt brf\ndiff [ps]')
                        ax2[spi][1].set_ylabel('Pos off wrt brf\ndiff [deg]')
                    #
                elif(key == 'bunchLength'): # eq. sigma_dt
                    unitfact_spi_0_ax1 = 1e-9
                    unitfact_spi_0_ax2 = 1e-9
                    ax1[spi, 0].set_ylabel('Length\n('+opt+') [ns]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Length growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                elif(key == 'bunchEnergySpread'): # eq. sigma_dE
                    unitfact_spi_0_ax1 = 1e6
                    unitfact_spi_0_ax2 = 1e6
                    ax1[spi, 0].set_ylabel('Energy spread\n('+opt+') [MeV]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Energy spread growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                elif(key == 'bunchEmittance'): # eq. epsnrmsl
                    unitfact_spi_0_ax1 = 1.
                    unitfact_spi_0_ax2 = 1.
                    ax1[spi, 0].set_ylabel('Emittance\n('+opt+') [eV$\cdot$s]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Emittance growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    #
                elif(key == 'bunchFormFactor'): # no equiv in beam
                    unitfact_spi_0_ax1 = 1.
                    unitfact_spi_0_ax2 = 1.
                    ax1[spi, 0].set_ylabel('Form fact\n[eV$\cdot$s]')
                    ax2[spi][0].set_ylabel('')
                    ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #
                    unitfact_spi_1_ax1 = 1./100.
                    unitfact_spi_1_ax2 = 1./100.
                    ax1[spi, 1].set_ylabel('Form fact growth\n('+opt+') [%]')
                    ax2[spi][1].set_ylabel('')
                    ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                # Bunch ave, minmax
                data_Profile_bunch_opt_key_ave_nb = np.array([])
                data_Profile_bunch_opt_key_min_nb = np.array([])
                data_Profile_bunch_opt_key_max_nb = np.array([])
                for nb in range(nbf):
                    data_Profile_bunch_opt_key_ave_nb = np.append(data_Profile_bunch_opt_key_ave_nb, np.array(data_Profile_bunch_opt_key_ave)[-1,nb])
                    data_Profile_bunch_opt_key_min_nb = np.append(data_Profile_bunch_opt_key_min_nb, np.array(data_Profile_bunch_opt_key_min)[-1,nb])
                    data_Profile_bunch_opt_key_max_nb = np.append(data_Profile_bunch_opt_key_max_nb, np.array(data_Profile_bunch_opt_key_max)[-1,nb])
                data_Profile_bunch_opt_key_minmax_nb = np.array([data_Profile_bunch_opt_key_ave_nb-data_Profile_bunch_opt_key_min_nb, data_Profile_bunch_opt_key_max_nb-data_Profile_bunch_opt_key_ave_nb])
                if(opt == 'BF' and key == 'alive'): # needed for plotting 'lost' in percent
                    data_Profile_bunch_BF_alv_ave_nb = np.copy(data_Profile_bunch_opt_key_ave_nb)
                    data_Profile_bunch_BF_alv_min_nb = np.copy(data_Profile_bunch_opt_key_min_nb)
                    data_Profile_bunch_BF_alv_max_nb = np.copy(data_Profile_bunch_opt_key_max_nb)
                    #data_Profile_bunch_BF_alv_minmax_nb = np.copy(data_Profile_bunch_opt_key_minmax_nb)

                # BUNCH:

                ax1[spi, 0].errorbar(range(nbf), data_Profile_bunch_opt_key_ave_nb/unitfact_spi_0_ax1, color='#888888', yerr=data_Profile_bunch_opt_key_minmax_nb/unitfact_spi_0_ax1, ls='') #, alpha=0.33)
                ax1[spi, 0].scatter( range(nbf), data_Profile_bunch_opt_key_ave_nb/unitfact_spi_0_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                ax1[spi, 0].plot(    range(nbf), data_Profile_bunch_opt_key_fst   /unitfact_spi_0_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                ax1[spi, 0].scatter( range(nbf), data_Profile_bunch_opt_key_fst   /unitfact_spi_0_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                ax1[spi, 0].plot(    range(nbf), data_Profile_bunch_opt_key[-1]   /unitfact_spi_0_ax1, color='#b40426', ls=':', alpha=0.33)
                ax1[spi, 0].scatter( range(nbf), data_Profile_bunch_opt_key[-1]   /unitfact_spi_0_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)')

                if(attr == 'nmacrop'):
                    if(opt == 'BF'):
                        if(key == 'alive'):
                            # Percent growth w.r.t. its own value before tracking (turn -1)
                            ax1[spi, 1].errorbar(range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Profile_bunch_opt_key_minmax_nb/data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33)
                            ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                            ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                            ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                            ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                            ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)')
                        else:
                            # Percent growth w.r.t. "bunch-alive" value before tracking (turn -1)
                            ax1[spi, 1].errorbar(range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Profile_bunch_opt_key_minmax_nb/data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33) # np.array([data_Profile_bunch_opt_key_max_nb-data_Profile_bunch_opt_key_ave_nb, data_Profile_bunch_opt_key_ave_nb-data_Profile_bunch_opt_key_min_nb])/data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1
                            ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                            ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                            ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                            ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                            ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)')
                    else:
                        # Percent growth w.r.t. "beam-alive" value before tracking (turn -1)
                        ax1[spi, 1].errorbar(range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Profile_bunch_opt_key_minmax_nb/data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33) # np.array([data_Profile_bunch_opt_key_max_nb-data_Profile_bunch_opt_key_ave_nb, data_Profile_bunch_opt_key_ave_nb-data_Profile_bunch_opt_key_min_nb])/data_Profile_bunch_BF_alv_fst/unitfact_spi_1_ax1
                        ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                        ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                        ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)')
                else:
                    if('Position' in key):
                        # Difference growth w.r.t. its own value before tracking (turn -1)
                        ax1[spi, 1].errorbar(range(nbf), (data_Profile_bunch_opt_key_ave_nb-data_Profile_bunch_opt_key_fst)/unitfact_spi_1_ax1, color='#888888', yerr=data_Profile_bunch_opt_key_minmax_nb/unitfact_spi_1_ax1, ls='') #, alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), (data_Profile_bunch_opt_key_ave_nb-data_Profile_bunch_opt_key_fst)/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                        ax1[spi, 1].plot(    range(nbf), (data_Profile_bunch_opt_key_fst   -data_Profile_bunch_opt_key_fst)/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), (data_Profile_bunch_opt_key_fst   -data_Profile_bunch_opt_key_fst)/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                        ax1[spi, 1].plot(    range(nbf), (data_Profile_bunch_opt_key[-1]   -data_Profile_bunch_opt_key_fst)/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), (data_Profile_bunch_opt_key[-1]   -data_Profile_bunch_opt_key_fst)/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)')
                    else:
                        # Percent growth w.r.t. its own value before tracking (turn -1)
                        ax1[spi, 1].errorbar(range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Profile_bunch_opt_key_minmax_nb/data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_ave_nb/data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                        ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key_fst   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                        ax1[spi, 1].plot(    range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                        ax1[spi, 1].scatter( range(nbf), data_Profile_bunch_opt_key[-1]   /data_Profile_bunch_opt_key_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)')


                # BEAM:

                if(attr == 'nmacrop'):
                    if(opt == 'BF'):
                        ax1[spi, 0].axhline((data_Profile_beam_opt_key_ave[-1]/nbf)/unitfact_spi_0_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                        ax1[spi, 0].axhline((data_Profile_beam_opt_key_fst    /nbf)/unitfact_spi_0_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam bef. track')
                        ax1[spi, 0].axhline((data_Profile_beam_opt_key[-1]    /nbf)/unitfact_spi_0_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam at turn {lastturn} (last)')
                        if(key == 'alive'):
                            ax1[spi, 1].axhline((data_Profile_beam_opt_key_ave[-1]/nbf)/(data_Profile_beam_opt_key_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                            ax1[spi, 1].axhline((data_Profile_beam_opt_key_fst    /nbf)/(data_Profile_beam_opt_key_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                            ax1[spi, 1].axhline((data_Profile_beam_opt_key[-1]    /nbf)/(data_Profile_beam_opt_key_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. at turn {lastturn} (last)')
                        else:
                            ax1[spi, 1].axhline((data_Profile_beam_opt_key_ave[-1]/nbf)/(data_Profile_beam_BF_alv_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                            ax1[spi, 1].axhline((data_Profile_beam_opt_key_fst    /nbf)/(data_Profile_beam_BF_alv_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                            ax1[spi, 1].axhline((data_Profile_beam_opt_key[-1]    /nbf)/(data_Profile_beam_BF_alv_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. at turn {lastturn} (last)')
                    else:
                        ax1[spi, 0].axhline( data_Profile_beam_opt_key_ave[-1]     /unitfact_spi_0_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                        ax1[spi, 0].axhline( data_Profile_beam_opt_key_fst         /unitfact_spi_0_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam bef. track')
                        ax1[spi, 0].axhline( data_Profile_beam_opt_key[-1]         /unitfact_spi_0_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam at turn {lastturn} (last)')
                        #
                        ax1[spi, 1].axhline(data_Profile_beam_opt_key_ave[-1]/data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                        ax1[spi, 1].axhline(data_Profile_beam_opt_key_fst    /data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                        ax1[spi, 1].axhline(data_Profile_beam_opt_key[-1]    /data_Profile_beam_BF_alv_fst/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. at turn {lastturn} (last)')
                else:
                    ax1[spi, 0].axhline( data_Profile_beam_opt_key_ave[-1]     /unitfact_spi_0_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                    ax1[spi, 0].axhline( data_Profile_beam_opt_key_fst         /unitfact_spi_0_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam bef. track')
                    ax1[spi, 0].axhline( data_Profile_beam_opt_key[-1]         /unitfact_spi_0_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam at turn {lastturn} (last)')
                    if('Position' in key):
                        ax1[spi, 1].axhline((data_Profile_beam_opt_key_ave[-1]-data_Profile_beam_opt_key_fst)/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                        ax1[spi, 1].axhline((data_Profile_beam_opt_key_fst    -data_Profile_beam_opt_key_fst)/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                        ax1[spi, 1].axhline((data_Profile_beam_opt_key[-1]    -data_Profile_beam_opt_key_fst)/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. at turn {lastturn} (last)
                    else:
                        ax1[spi, 1].axhline(data_Profile_beam_opt_key_ave[-1]/data_Profile_beam_opt_key_fst/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                        ax1[spi, 1].axhline(data_Profile_beam_opt_key_fst    /data_Profile_beam_opt_key_fst/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                        ax1[spi, 1].axhline(data_Profile_beam_opt_key[-1]    /data_Profile_beam_opt_key_fst/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. at turn {lastturn} (last)')')


                # SPECIAL CASES:

                if(key == 'bunchLength' and opt == 'FWHM' and blmeas is not None):
                    ax1[spi, 0].axhline(blmeas/unitfact_spi_0_ax1, ls='-.', color='#00cc00', alpha=0.50)
                   #ax1[spi, 1].axhline(100.,                      ls='--', color='#00cc00', alpha=0.50)

                casename = os.path.split(outdir)[1]

                if('LHC-' not in casename):
                    if('benchmarkTheo' in casename): # or 'test' in casename):
                        if(key == 'bunchPositionOff_fit'):
                            #data_model = np.load(f'/afs/cern.ch/work/l/lmedinam/BLonD_simulations/Simulations/2020-05_SPS-LHC-losses/analysis/TheoThesis/data_TheoThesis-model.npy')
                            #data_meas  = np.load(f'/afs/cern.ch/work/l/lmedinam/BLonD_simulations/Simulations/2020-05_SPS-LHC-losses/analysis/TheoThesis/data_TheoThesis-meas.npy')
                            data_model = np.load(f'{dirinpbench}/TheoThesis/data_TheoThesis-model.npy')
                            data_meas  = np.load(f'{dirinpbench}/TheoThesis/data_TheoThesis-meas.npy')
                            ax1[spi, 0].plot(range(len(data_meas)),  data_meas /unitfact_spi_0_ax1, ':k', label='CERN-THESIS-2015-421 (meas.)') # 2010 measurements
                            ax1[spi, 0].plot(range(len(data_model)), data_model/unitfact_spi_0_ax1, '-k', label='CERN-THESIS-2015-421 (model)') # Static FB+FF model in T.Argyopoulos PhD Thesis (2015) to reproduce 2010 measurements

                    #if('BBBBenchmark' in casename):
                    if('6805' in casename or '6808' in casename or '6809' in casename or '7137' in casename):
                        if(attr == 'stats'):
                            #
                            # LHC tomo profile stats: taking the stats (Luis) over the bunch distributions reconstructed from LHC first-turn profiles via tomography (Theo)
                            if('6805' in casename or '7137' in casename):
                                if(  '6805' in casename): h5Filename_dummy = f'{dirinpbench}/tomo-distr/PROFILE_B1_b25540_20180616020329/LHCTomo_stats.h5'
                                elif('7137' in casename): h5Filename_dummy = f'{dirinpbench}/tomo-distr/PROFILE_B2_b30660_20180908022914/LHCTomo_stats.h5'
                                # To also show the black measurements in plot of offset w.r.t. fit
                                if key == 'bunchPositionOff_fit' or key == 'bunchPositionOff':
                                    if opt == 'BRF':
                                        optdummy = 'RMS'
                                    else:
                                        optdummy = opt
                                    keydummy = 'bunchPositionOff_ctr'
                                else:
                                    optdummy = opt
                                    keydummy = key
                                with hp.File(h5Filename_dummy, 'r') as h5File_dummy:
                                    data_LHCTomo_Profile_bunch_optdummy_keydummy_fst = h5File_dummy[f'/Profile/bunch/{optdummy}/{keydummy}'][0]
                                    data_LHCTomo_Profile_beam_optdummy_keydummy_fst  = h5File_dummy[f'/Profile/beam/{optdummy}/{keydummy}' ][0]
                                try: h5File_dummy.close()
                                except: pass
                                labelmeas = 'LHC tomo.'
                                correct_Offshift = True
                                if(correct_Offshift and (keydummy == 'bunchPositionOff_fit')):
                                    Offshift = data_LHCTomo_Profile_beam_optdummy_keydummy_fst
                                    labelmeas += f' (shift by {-1*Offshift/unitfact_spi_1_ax1:+.1f})'
                                else:
                                    Offshift = 0
                                ax1[spi, 0].plot(range(len(data_LHCTomo_Profile_bunch_optdummy_keydummy_fst)), (data_LHCTomo_Profile_bunch_optdummy_keydummy_fst-Offshift)/unitfact_spi_0_ax1, '.:k', label=labelmeas)
                                ax1[spi, 0].axhline(                                                      (data_LHCTomo_Profile_beam_optdummy_keydummy_fst -Offshift)/unitfact_spi_0_ax1, ls='--', color='k', alpha=0.50)
                            #
                            # LHC bunch position offsets: taking the difference of the mean bunch position from LHC first-turn profiles w.r.t. fit (Theo)
                            if('6808' in casename or '6809' in casename or '7137' in casename):
                                if(key == 'bunchPositionOff_fit'):
                                    if(  '6808' in casename): data_meas = np.load(f'{dirinpbench}/profile/PROFILE_B1_b25540_20180616043102_BunchPositionVariations.npy')
                                    elif('6809' in casename): data_meas = np.load(f'{dirinpbench}/profile/PROFILE_B1_b25540_20180616060554_BunchPositionVariations.npy')
                                    elif('7137' in casename): data_meas = np.load(f'{dirinpbench}/profile/PROFILE_B2_b30660_20180908022914_BunchPositionVariations.npy')
                                    ax1[spi, 0].plot(range(nbf), data_meas[:nbf]*1e-12/unitfact_spi_0_ax1, '--', color='#888888', label='LHC profile')
                                    ax1[spi, 0].axhline(np.average(data_meas[:nbf])*1e-12/unitfact_spi_0_ax1, ls='--', color='888888', alpha=0.50)
                            #
                            # SPS bunch position offsets: taking the difference of mean bunch position from SPS raw profiles w.r.t. fit (Luis)
                            if('6805' in casename or '6808' in casename or '6809' in casename or '7137' in casename):
                                if(key == 'bunchPositionOff_fit'):
                                    if(  '6805' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_6805_First48b_LastBatch_B1.npy')
                                    elif('6808' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_6808_First48b_LastBatch_B1.npy')
                                    elif('6809' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_6809_First48b_LastBatch_B1.npy')
                                    elif('7137' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_7137_First48b_PenultimateBatch_B2.npy')
                                    ax1[spi, 0].plot(range(nbf), data_meas[:nbf]/unitfact_spi_1_ax1, '--c',label='SPS profile')
                                    ax1[spi, 0].axhline(np.average(data_meas[:nbf])/unitfact_spi_1_ax1, ls='--', color='c', alpha=0.50)
                        #
#                        # YAML input: the casename is the concatenation of all the stages (i.e SPS or SPS-LHC or SPS-LHC-LHCramp), but we only want the case name of the last stage:
#                        allfnames = os.listdir(f'{outdir}')
#                        casename_stage = False; idxfname = 0
#                        while(idxfname < len(allfnames) and not casename_stage):
#                            if(allfnames[idxfname][:-4] in casename): casename_stage = allfnames[idxfname][:-4]
#                            idxfname += 1
                        casename_stage = casename
                        with open(f'{outdir}/{casename_stage}.pkl', 'rb') as pklFile_dummy:
                            MACupd = pickle.load(pklFile_dummy)
                            if(key == 'alive'):
                                key_dummy = 'Nb_list'
                                unit_dummy = 1.
                                ax2[spi][0].plot(range(len(MACupd[key_dummy])), MACupd[key_dummy]*unit_dummy/unitfact_spi_0_ax2, '-', color='#00cc00', label='input') #SPS meas. (inp)')
                            elif(key == 'bunchLength' and opt == 'FWHM'):
                                key_dummy = 'bl_list'
                                unit_dummy = 1.
                                ax1[spi, 0].plot(range(len(MACupd[key_dummy])), MACupd[key_dummy]*unit_dummy/unitfact_spi_0_ax1, '-', color='#00cc00', label='input') #SPS meas. (inp)')
                            try: pklFile_dummy.close()
                            except: pass

                # SECONDARY AXIS:

                ax1_spi_0_yticks = ax1[spi, 0].get_yticks()
                ax1[spi][0].set_ylim( ax1_spi_0_yticks[0], ax1_spi_0_yticks[-1] )
                if(unitfact_spi_0_ax1 == unitfact_spi_0_ax2): scalefactor_tmp = 1
                else:                                         scalefactor_tmp = scalefactor
                ax2_spi_0_yticks = ax1_spi_0_yticks*unitfact_spi_0_ax1 * scalefactor_tmp /unitfact_spi_0_ax2
                ax2[spi][0].set_ylim( ax2_spi_0_yticks[0], ax2_spi_0_yticks[-1] )
                ax2[spi][0].set_yticks(ax2_spi_0_yticks)
                #print(f'key: {key}: {ax1_spi_0_yticks} (unit: {unitfact_spi_0_ax1}) -> (scale: {scalefactor_tmp}) -> {ax2_spi_0_yticks} (unit: {unitfact_spi_0_ax2})')
                #
                ax1_spi_1_yticks = ax1[spi, 1].get_yticks()
                ax1[spi][1].set_ylim( ax1_spi_1_yticks[0], ax1_spi_1_yticks[-1] )
                if(unitfact_spi_1_ax1 == unitfact_spi_1_ax2): scalefactor_tmp = 1
                else:                                         scalefactor_tmp = scalefactor
                ax2_spi_1_yticks = ax1_spi_1_yticks*unitfact_spi_1_ax1 * scalefactor_tmp /unitfact_spi_1_ax2
                ax2[spi][1].set_ylim( ax2_spi_1_yticks[0], ax2_spi_1_yticks[-1] )
                ax2[spi][1].set_yticks(ax2_spi_1_yticks)
                #print(f'key: {key}: {ax1_spi_1_yticks} (unit: {unitfact_spi_1_ax1}) -> (scale: {scalefactor_tmp}) -> {ax2_spi_1_yticks} (unit: {unitfact_spi_1_ax2})')

                # LEGEND:

                if(attr == 'nmacrop'):
                    if(spi == 0):
                        ax1[spi, 0].legend(loc=3, ncol=2)
                else:
                    if(  spi == 0): ax1[spi, 0].legend(loc=2, ncol=2)
                    elif(spi == 1): ax1[spi, 0].legend(loc=1, ncol=2)
                    elif(spi == 2): ax1[spi, 0].legend(loc=4, ncol=2)


                spi += 1
                gc.collect()

        fig.tight_layout()
        fig.savefig(f'{outdir}/plot_profile_{attr}_vs_bn.png')
        plt.cla()
        plt.close(fig)

    try:    h5File.close()
    except: pass


###############################################################################

# Profiles (superposed bunches, at a given turn) ------------------------------

def plot_profile_superposed_at_turn(outdir, turn_i, profilepattern_obj, bunchPosition_i=None):
    """
    Based on 'plot_beam_profile' from BLonD's plots module
    """

    nbf = len(profilepattern_obj.bucket_centres)
    Ns = profilepattern_obj.Ns

    with hp.File(f'{outdir}/monitor_profile.h5', 'r') as h5File:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_Profile_turns = h5File['/Profile/turns'][:] # 'int64'
        #print(f'data_Profile_turns = {data_Profile_turns}')
        #print(f'turn_i = {turn_i}')
        idxturni = np.argmin(data_Profile_turns < turn_i)
        #print(f'idxturni = {idxturni}')
        data_Profile_turns_i = h5File['/Profile/turns'][idxturni]
        #print(f'data_Profile_turns_i = {data_Profile_turns_i}')

        # data_Profile_bin_centers = h5File[f'/Profile/bin_centers'][:]
        # data_Profile_n_macroparticles = h5File[f'/Profile/n_macroparticles'][:]
        # print(f'data_Profile_bin_centers = {data_Profile_bin_centers}')
        # print(f'data_Profile_n_macroparticles = {data_Profile_n_macroparticles}')

        data_Profile_bin_centers_i = h5File[f'/Profile/bin_centers'][idxturni]
        data_Profile_n_macroparticles_i = h5File[f'/Profile/n_macroparticles'][idxturni]
        # print(f'data_Profile_bin_centers_i = {data_Profile_bin_centers_i}')
        #print(f'data_Profile_n_macroparticles_i = {data_Profile_n_macroparticles_i}')

        maxbktl_0 = 5.0e-9 #2*bunchPosition_i[0]
        if(bunchPosition_i is None):
            with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5FileFull:
                dataFull_Profile_turns = h5FileFull['/Profile/turns'][:] # 'int64'
                idxFullturni = np.argmin(dataFull_Profile_turns < turn_i) #+1
                data_Profile_bunch_FWHM_bunchPosition_i = h5FileFull[f'/Profile/bunch/FWHM/bunchPosition'][idxFullturni]
                data_Profile_bunch_FWHM_bunchPosition_0 = h5FileFull[f'/Profile/bunch/FWHM/bunchPosition'][0]
                #maxbktl_0 = (data_Profile_bunch_FWHM_bunchPosition_0[1] - data_Profile_bunch_FWHM_bunchPosition_0[0])/nbs
                #maxbktl_0 = 2*data_Profile_bunch_FWHM_bunchPosition_0[0]
            try:    h5FileFull.close()
            except: pass
        #print(f'data_Profile_bin_centers_i = {data_Profile_bin_centers_i}, shape = {data_Profile_bin_centers_i.shape}')
        #print(f'data_Profile_bunch_FWHM_bunchPosition_i = {data_Profile_bunch_FWHM_bunchPosition_i}, shape = {data_Profile_bunch_FWHM_bunchPosition_i.shape}')

        #Ns = int(len(data_Profile_bin_centers_i)/nbt)
        #nbf = int((nbt-1-2+(nbs-1))/nbs)

        #print(f'Ns = {Ns}')
        #print(f'nbf = {nbf}')
        #print(f'maxbktl_0 = {maxbktl_0}')

        gc.collect()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        fig, ax = plt.subplots()
        fig.set_size_inches(8.0,6.0)



        ax.set_xlabel(r"$\Delta t$ [ns]")
        ax.set_ylabel('Profile [macroparticles]')
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        mycm = plt.get_cmap('coolwarm')
        if(nbf > 1): color_cycle_nb = [mycm(nb/(nbf-1.)) for nb in range(nbf)]
        else:        color_cycle_nb = [mycm(0.)]

        ax.set_prop_cycle(cycler('color', color_cycle_nb))
        ax.grid()

        # if(nbt-3 == 1): # Single bunch with 1 empty bucket at the beggining, 2 at the end
        #      ax.plot(data_Profile_bin_centers_i-data_Profile_bunch_FWHM_bunchPosition_i, data_Profile_n_macroparticles_i, '-')
        # else:

        #     ibf=1
        #     for nb in range(1, nbt):
        #         idx_0 = int( (nbm+nb)   *Ns)
        #         idx_f = int(((nbm+nb)+1)*Ns)-1
        #         #print(nb, idx_0, idx_f, end='')
        #         if((nb-1-nbm) % nbs == 0):
        #             #print('->', ibf)
        #             # print(ibf)
        #             # print(f'data_Profile_bin_centers_i[idx_0:idx_f] = {data_Profile_bin_centers_i[idx_0:idx_f]}')
        #             # print(f'data_Profile_bunch_FWHM_bunchPosition_i[ibf-1] = {data_Profile_bunch_FWHM_bunchPosition_i[ibf-1]}')
        #             #print(f'data_Profile_n_macroparticles_i[idx_0:idx_f] = {data_Profile_n_macroparticles_i[idx_0:idx_f]}')
        #             ax.plot( (data_Profile_bin_centers_i[idx_0:idx_f]-data_Profile_bunch_FWHM_bunchPosition_i[ibf-1])/1e-9, data_Profile_n_macroparticles_i[idx_0:idx_f], '-')
        #             ibf += 1
        #             if(ibf > nbf): # strictly '>', as at '=' corresponds to the last filled bucket
        #                 break
        #         #else:
        #             #print('')

        for j in range(nbf):

            if(j == 0 or j == nbf-1): label_nb = f'Bunch {j}'
            else:                     label_nb = None

            idx_BF_bunch_j = profilepattern_obj.idxP['BF']['bunch'][j]
            #print(j, idx_BF_bunch_j)
            ax.plot( (data_Profile_bin_centers_i[idx_BF_bunch_j]-data_Profile_bunch_FWHM_bunchPosition_i[j])/1e-9, data_Profile_n_macroparticles_i[idx_BF_bunch_j], '-', label=label_nb)

        ax.set_xlim(-maxbktl_0/1e-9, maxbktl_0/1e-9)
        #plt.figtext(0.95, 0.95, '%d turns' %turn_i, fontsize=12, ha='right', va='center')
        ax.set_title(f'Turn {turn_i:d}', ha='right', va='center')
        ax.legend(loc=1)

        fig.tight_layout()
        fig.savefig(f'{outdir}/plot_profile_superposed_at_turn_{turn_i:d}.png')
        plt.cla()
        #plt.clf()
        plt.close(fig)
        gc.collect()

    try:    h5File.close()
    except: pass


# Profiles (highlighting corresponding BF/BS sections) vs time (2D plot) ------

def plot_profile_BF_BS_vs_turn(outdir, idxbf_beam, idxbs_beam, bucket_centres, maxbktl): #allprofiles_dict, idxbinfb, idxbinsb, Nt_rmv, outdir):

    data_Profile_turns        = mynpload(outdir+'/Profile/turns.npy') # 'int64'

    #data_profiles_edges      = mynpload(outdir+'/profiles_edges.npy')
    data_profiles_bin_centers = mynpload(outdir+'/profiles_bin_centers.npy')
    data_profiles_nmacrop     = mynpload(outdir+'/profiles_nmacrop.npy')# 'int64'

    ##print('data_Profile_turns =', data_Profile_turns)
    ###print('data_profiles_edges =', data_profiles_edges)
    ##print('data_profiles_bin_centers =', data_profiles_bin_centers)
    ##print('data_profiles_nmacrop =', data_profiles_nmacrop)

    fig = plt.figure(1)
    fig.set_size_inches(64.0,8.0)
    ax = plt.axes([0.015, 0.125, 0.935, 0.825])

    if(  idxbs_beam[-1] <  5000): ax.xaxis.set_major_locator(MultipleLocator( 5.0))   # meant for < ~12, it might not be the best for > ~12
    elif(idxbs_beam[-1] < 10000): ax.xaxis.set_major_locator(MultipleLocator(10.0))
    elif(idxbs_beam[-1] < 15000): ax.xaxis.set_major_locator(MultipleLocator(15.0))
    else:                         ax.xaxis.set_major_locator(MultipleLocator(20.0))

    mycm_inbf = plt.get_cmap('Blues')
    mycm_inbs = plt.get_cmap('Reds')

    gc.collect()

    data_profiles_nmacrop_inbf = []
    data_profiles_nmacrop_inbs = []

    for k in range(len(data_Profile_turns)):

        data_profiles_nmacrop_inbf_k = np.copy(data_profiles_nmacrop[k]).astype('float32')
        data_profiles_nmacrop_inbf_k[idxbs_beam] = np.NaN

        data_profiles_nmacrop_inbs_k = np.copy(data_profiles_nmacrop[k]).astype('float32')
        data_profiles_nmacrop_inbs_k[idxbf_beam] = np.NaN

        data_profiles_nmacrop_inbf.append(data_profiles_nmacrop_inbf_k)
        data_profiles_nmacrop_inbs.append(data_profiles_nmacrop_inbs_k)

    data_profiles_nmacrop_inbf = np.array(data_profiles_nmacrop_inbf)
    data_profiles_nmacrop_inbs = np.array(data_profiles_nmacrop_inbs)

    ##print('data_profiles_nmacrop_inbf =', data_profiles_nmacrop_inbf)
    ##print('data_profiles_nmacrop_inbs =', data_profiles_nmacrop_inbs)

    image_inbf = ax.pcolormesh(data_profiles_bin_centers/1e-9, data_Profile_turns, data_profiles_nmacrop_inbf, cmap=mycm_inbf, shading='gouraud')
    image_inbs = ax.pcolormesh(data_profiles_bin_centers/1e-9, data_Profile_turns, data_profiles_nmacrop_inbs, cmap=mycm_inbs, shading='gouraud')

    #plt.subplots_adjust(right=0.800)
    gc.collect()

    mycbar_ax_inbf = fig.add_axes([0.955, 0.125, 0.005, 0.825])
    mycbar_ax_inbs = fig.add_axes([0.976, 0.125, 0.005, 0.825])
    mycolorbar_inbf = fig.colorbar(image_inbf, ax=ax, cax=mycbar_ax_inbf) #, format=fmtij)
    mycolorbar_inbs = fig.colorbar(image_inbs, ax=ax, cax=mycbar_ax_inbs) #, format=fmtij)

    for nb in range(len(bucket_centres[0])):
        ax.plot(np.array(bucket_centres)[ data_Profile_turns ,nb]/1e-9 - 0.5*maxbktl/1e-9, data_Profile_turns, ':k')
        ax.plot(np.array(bucket_centres)[ data_Profile_turns ,nb]/1e-9 + 0.5*maxbktl/1e-9, data_Profile_turns, ':k')

    ax.set_xlabel(r"$\Delta t$ [ns]") # "Bin centers"
    ax.set_ylabel('Turns')
    mycolorbar_inbf.ax.set_ylabel('Profile (in filled buckets) [arb. units]') #nmacrop inbf
    mycolorbar_inbs.ax.set_ylabel('Profile (in separation buckets) [arb. units]') #nmacrop inbf

    # Save plot
    fign = outdir+'/plot_profile_BF_BS_vs_turn.png'
    #print('Saving '+outdir+'/plot_profiles_nmacrop_inbf_inbs.png')
    #fig.tight_layout()
    plt.savefig(fign)
    plt.clf()
    gc.collect()

###############################################################################

