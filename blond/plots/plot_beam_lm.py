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

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoLocator #  MultipleLocator, FormatStrFormatter,
from cycler import cycler

# BLonD-based tools
from blond.beam.beam_tools import BeamTools

#-------------

import pathlib

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

dirbase = 'BLonD_simulations/sps_lhc_losses'
dirin   = f'{dirhome}/{dirbase}/sps_lhc_losses'
dirinp  = f'{dirin}/inp'
dirinpbench  = f'{dirinp}/benchmark'

#-------------

###############################################################################

# Beam macroparticles OR stats vs turn ----------------------------------------

def plot_beam_vs_turn(attr, outdir, scalefactor=None):  # Formally, we should pass NbperNp_list_0 instead of NbperNp_tot_0

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_Beam_turns = h5File['/Beam/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_Beam_turns)
        data_Beam_turns = h5File['/Beam/turns'][:idxlastturn+1]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if(  attr == 'nmacrop'): key_list = ['alive', 'lost'] # Order is important, alive should come before lost!
        elif(attr == 'stats'):   key_list = ['mean_dt', 'mean_dtOff_ctr', 'mean_dtOff_fit', 'sigma_dt', 'mean_dE', 'sigma_dE', 'epsnrmsl']
        n_key_list = len(key_list)

        fig, ax1 = plt.subplots(n_key_list, 2, sharex=True)
        #ax2 = ax1.twinx() # [ax1[spi, 0].twinx() for spi in range(n_key_list)]
        ax2 = [[ax1[spi,spj].twinx() for spj in range(2)] for spi in range(n_key_list)]
        fig.set_size_inches(2*8.0,n_key_list*2.0)

        gc.collect()

        spi = 0
        for key in key_list:

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            data_Beam_bunch_key_fst = h5File[f'/Beam/bunch/{key}'][0]
            data_Beam_beam_key_fst  = h5File[f'/Beam/beam/{key}' ][0]

            data_Beam_bunch_key = h5File[f'/Beam/bunch/{key}'][:idxlastturn+1]
            data_Beam_beam_key  = h5File[f'/Beam/beam/{key}' ][:idxlastturn+1]

            nbf = len(data_Beam_bunch_key[0])

            # - - - - - - - - - - - - - - - - -

            # if(attr == 'nmacrop'): # nmacrop (alive and lost) for beam will be plotted as the average over all bunches
            #     data_Beam_beam_key_fst = data_Beam_beam_key_fst.astype(float)
            #     data_Beam_beam_key_fst = data_Beam_beam_key_fst/nbf
            #     data_Beam_beam_key = data_Beam_beam_key.astype(float)
            #     data_Beam_beam_key = data_Beam_beam_key/nbf

            if(key == 'alive'): # Needed for plotting 'lost' in percent:
                data_Beam_bunch_alv_fst = np.copy(data_Beam_bunch_key_fst)
                data_Beam_beam_alv_fst  = np.copy(data_Beam_beam_key_fst)
                # data_Beam_bunch_alv = np.copy(data_Beam_bunch_key)
                # data_Beam_beam_alv  = np.copy(data_Beam_beam_key)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            mycm = plt.get_cmap('coolwarm')
            if(nbf > 1): color_cycle_nb = [mycm(nb/(nbf-1.)) for nb in range(nbf)]
            else:        color_cycle_nb = [mycm(0.)]

            if(len(data_Beam_turns) == 2): mymarker = '.' # Turns -1 and 0
            else:                          mymarker = '-'

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
                ax1[spi, 0].set_ylabel('Alive macrop\n')
                ax2[spi][0].set_ylabel('Intensity\n[$10^{11}$ ppb]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('Alive macrop\ngrowth [%]')
                ax2[spi][1].set_ylabel('Intensity\ngrowth [%]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
            elif(key == 'lost'):
                unitfact_spi_0_ax1 = 1
                unitfact_spi_0_ax2 = 1e9
                ax1[spi, 0].set_ylabel('Lost macrop\n')
                ax2[spi][0].set_ylabel('Lost intensity\n[$10^{9}$ ppb]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('Lost macrop\ngrowth [%]')
                ax2[spi][1].set_ylabel('Lost intensity\ngrowth [%]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
            elif(key == 'mean_dt'): # eq. bunchPosition
                unitfact_spi_0_ax1 = 1e-9
                unitfact_spi_0_ax2 = 1e3
                ax1[spi, 0].set_ylabel('mean($\Delta t$) [ns]')
                ax2[spi][0].set_ylabel('mean($\Delta t$) [$10^3$ deg]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                #
                unitfact_spi_1_ax1 = 1e-12
                unitfact_spi_1_ax2 = 1.
                ax1[spi, 1].set_ylabel('mean($\Delta t$)\ndiff [ps]')
                ax2[spi][1].set_ylabel('mean($\Delta t$)\ndiff [deg]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                #
            elif(key == 'mean_dtOff_ctr'): # eq. bunchPositionOff_ctr
                unitfact_spi_0_ax1 = 1e-12
                unitfact_spi_0_ax2 = 1.
                ax1[spi, 0].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) [ps]')
                ax2[spi][0].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) [deg]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1e-12
                unitfact_spi_1_ax2 = 1.
                ax1[spi, 1].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) diff [ps]')
                ax2[spi][1].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) diff [deg]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'mean_dtOff_fit'): # eq. bunchPositionOff_fit
                unitfact_spi_0_ax1 = 1e-12
                unitfact_spi_0_ax2 = 1.
                ax1[spi, 0].set_ylabel('mean($\Delta t$ off)\n(wrt fit) [ps]')
                ax2[spi][0].set_ylabel('mean($\Delta t$ off)\n(wrt fit) [deg]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1e-12
                unitfact_spi_1_ax2 = 1.
                ax1[spi, 1].set_ylabel('mean($\Delta t$ off)\n(wrt fit) diff [ps]')
                ax2[spi][1].set_ylabel('mean($\Delta t$ off)\n(wrt fit) diff [deg]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'mean_dE'):
                unitfact_spi_0_ax1 = 1e6
                unitfact_spi_0_ax2 = 1e6
                ax1[spi, 0].set_ylabel('mean($\Delta E$) [MeV]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1e6
                unitfact_spi_1_ax2 = 1e6
                ax1[spi, 1].set_ylabel('mean($\Delta E$)\ndiff [MeV]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'sigma_dt'): # eq. bunchLength
                unitfact_spi_0_ax1 = 1e-9*(1./4)
                unitfact_spi_0_ax2 = 1e-9*(1./4)
                ax1[spi, 0].set_ylabel('4$\cdot$rms($\Delta t$) [ns]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('4*rms($\Delta t$)\ngrowth [%]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'sigma_dE'):
                unitfact_spi_0_ax1 = 1e6
                unitfact_spi_0_ax2 = 1e6
                ax1[spi, 0].set_ylabel('rms($\Delta E$) [MeV]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('rms($\Delta E$)\ngrowth [%]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'epsnrmsl'):
                unitfact_spi_0_ax1 = 1.*(1./4)
                unitfact_spi_0_ax2 = 1.*(1./4)
                ax1[spi, 0].set_ylabel('4$\pi\cdot$rms($\Delta t$)$\cdot$rms($\Delta E$)\n[eV$\cdot$s]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('4$\pi\cdot$rms($\Delta t$)$\cdot$rms($\Delta E$)\ngrowth [%]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # BUNCH:

            for nb in range(nbf):
                if(nb == 0 or nb == nbf-1): label_nb = f'Bunch {nb}'
                else:                       label_nb = None
                ax1[spi, 0].plot(data_Beam_turns, data_Beam_bunch_key[:,nb]/unitfact_spi_0_ax1, mymarker, alpha=0.20, label=label_nb)
                if(attr == 'nmacrop'):
                    if(key == 'alive'):
                        # Percent growth w.r.t. its own value before tracking (turn -1)
                        ax1[spi, 1].plot(data_Beam_turns, data_Beam_bunch_key[:,nb]/data_Beam_bunch_key_fst[nb]/unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)
                    else:
                        # Percent growth w.r.t. "alive" value before tracking (turn -1)
                        ax1[spi, 1].plot(data_Beam_turns, data_Beam_bunch_key[:,nb]/data_Beam_bunch_alv_fst[nb]/unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)
                else:
                    if('mean' in key):
                        # Difference growth w.r.t. its own value before tracking (turn -1)
                        ax1[spi, 1].plot(data_Beam_turns, (data_Beam_bunch_key[:,nb]-data_Beam_bunch_key_fst[nb])/unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)
                    else:
                        # Percent growth w.r.t. its own value before tracking (turn -1)
                        ax1[spi, 1].plot(data_Beam_turns,  data_Beam_bunch_key[:,nb]/data_Beam_bunch_key_fst[nb] /unitfact_spi_1_ax1, mymarker, alpha=0.20, label=label_nb)

            # BEAM:

            if(attr == 'nmacrop'):
                ax1[spi, 0].plot(data_Beam_turns, (data_Beam_beam_key/nbf)/unitfact_spi_0_ax1, mymarker+'-k', label='Bunch. ave.')
                if(key == 'alive'): ax1[spi, 1].plot(data_Beam_turns, (data_Beam_beam_key/nbf)/(data_Beam_beam_key_fst/nbf)/unitfact_spi_1_ax1, mymarker+'-k', label='Bunch. ave.')
                else:               ax1[spi, 1].plot(data_Beam_turns, (data_Beam_beam_key/nbf)/(data_Beam_beam_alv_fst/nbf)/unitfact_spi_1_ax1, mymarker+'-k', label='Bunch. ave.')
            else:
                ax1[spi, 0].plot(data_Beam_turns,  data_Beam_beam_key     /unitfact_spi_0_ax1, mymarker+'-k', label='Bunch. ave.')
                if('mean' in key):
                    ax1[spi, 1].plot(data_Beam_turns, (data_Beam_beam_key-data_Beam_beam_key_fst)/unitfact_spi_1_ax1, mymarker+'-k', label='Bunch. ave.')
                else:
                    ax1[spi, 1].plot(data_Beam_turns,  data_Beam_beam_key/data_Beam_beam_key_fst /unitfact_spi_1_ax1, mymarker+'-k', label='Bunch. ave.')

            # SPECIAL CASES:

            # if(key == 'mean_dt' and len(data_Beam_turns) > 2 and bucket_centres_i is not None):
            #     for nb in range(nbf):
            #         ax1[spi].plot(data_Beam_turns, np.ones(len(data_Beam_turns))*bucket_centres_i[nb]/unitfact_spi_0_ax1, ':', color='#888888')

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

            spi += 1
            gc.collect()

        fig.tight_layout()
        fig.savefig(f'{outdir}/plot_beam_{attr}_vs_turn.png')
        plt.cla()
        plt.close(fig)
        gc.collect()

    try:    h5File.close()
    except: pass


# Beam macroparticles OR stats vs bunch no. -----------------------------------

def plot_beam_vs_bn(attr, outdir, turn_bn=None, scalefactor=None):  # Formally, we should pass NbperNp_list_0 instead of NbperNp_tot_0

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_Beam_turns = h5File['/Beam/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_Beam_turns)
        lastturn = data_Beam_turns[idxlastturn]
        idxzeroturn = 0
        zeroturn = data_Beam_turns[idxzeroturn]
        data_Beam_turns = h5File['/Beam/turns'][:idxlastturn+1]

        if(turn_bn is not None):
            # if(lastturn > 5000): turn_bn = 5000
            # idxfirstturn = np.argmax(data_Beam_turns >= turn_bn)
            # #print(f'data_Beam_turns = {data_Beam_turns}')
            # #print(f'idxfirstturn = {idxfirstturn}')
            if(lastturn > turn_bn):
                idxfirstturn = np.argmax(data_Beam_turns >= turn_bn)
            else:
                idxfirstturn = 0
        else:
            idxfirstturn = 0
        firstturn = data_Beam_turns[idxfirstturn]
        data_Beam_turns = h5File['/Beam/turns'][firstturn:idxlastturn+1]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if(  attr == 'nmacrop'): key_list = ['alive', 'lost'] # Order is important, lave should come before lost!
        elif(attr == 'stats'):   key_list = ['mean_dt', 'mean_dtOff_ctr', 'mean_dtOff_fit', 'sigma_dt', 'mean_dE', 'sigma_dE', 'epsnrmsl']
        n_key_list = len(key_list)

        fig, ax1 = plt.subplots(n_key_list, 2, sharex=True)
        #ax2 = ax1.twinx() # [ax1[spi, 0].twinx() for spi in range(n_key_list)]
        ax2 = [[ax1[spi,spj].twinx() for spj in range(2)] for spi in range(n_key_list)]
        fig.set_size_inches(2*8.0,n_key_list*2.0)

        gc.collect()

        spi = 0
        for key in key_list:

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            data_Beam_bunch_key_fst = h5File[f'/Beam/bunch/{key}'][0]
            data_Beam_beam_key_fst  = h5File[f'/Beam/beam/{key}' ][0]

            data_Beam_bunch_key = h5File[f'/Beam/bunch/{key}'][idxfirstturn:idxlastturn+1]
            data_Beam_beam_key  = h5File[f'/Beam/beam/{key}' ][idxfirstturn:idxlastturn+1]

            nmntturns = len(data_Beam_bunch_key)
            nbf       = len(data_Beam_bunch_key[0])

            # - - - - - - - - - - - - - - - - -

            data_Beam_bunch_key_min = np.minimum.accumulate(data_Beam_bunch_key)
            data_Beam_bunch_key_max = np.maximum.accumulate(data_Beam_bunch_key)
            data_Beam_bunch_key_ave = np.add.accumulate(data_Beam_bunch_key)/ np.arange(1,nmntturns+1).reshape((nmntturns,1)) # the divisor equivalent to np.array([ np.ones(nbf)*i for i in np.arange((1,nmntturns+1) ])
          ##data_Beam_bunch_key_min = np.array([ np.array([    min(data_Beam_bunch_key[:i+1,j]) for j in range(nbf)]) for i in range(nmntturns)])
          ##data_Beam_bunch_key_ave = np.array([ np.array([np.mean(data_Beam_bunch_key[:i+1,j]) for j in range(nbf)]) for i in range(nmntturns)])
          ##data_Beam_bunch_key_max = np.array([ np.array([    max(data_Beam_bunch_key[:i+1,j]) for j in range(nbf)]) for i in range(nmntturns)])

           #data_Beam_beam_key_min = np.minimum.accumulate(data_Beam_beam_key)
           #data_Beam_beam_key_max = np.maximum.accumulate(data_Beam_beam_key)
            data_Beam_beam_key_ave = np.add.accumulate(data_Beam_beam_key) / np.arange(1,nmntturns+1)
          ##data_Beam_beam_key_min = np.array([     min(data_Beam_beam_key[:i+1]) for i in range(nmntturns)])
          ##data_Beam_beam_key_ave = np.array([ np.mean(data_Beam_beam_key[:i+1]) for i in range(nmntturns)])
          ## data_Beam_beam_key_max = np.array([     max(data_Beam_beam_key[:i+1]) for i in range(nmntturns)])

            # - - - - - - - - - - - - - - - - -

            # if(attr == 'nmacrop'): # nmacrop (alive and lost) for beam will be plotted as the average over all bunches
            #     data_Beam_beam_key_fst = data_Beam_beam_key_fst.astype(float)
            #     data_Beam_beam_key_fst = data_Beam_beam_key_fst/nbf
            #     data_Beam_beam_key = data_Beam_beam_key.astype(float)
            #     data_Beam_beam_key = data_Beam_beam_key/nbf
            #     data_Beam_beam_key_ave = data_Beam_beam_key_ave.astype(float)
            #     data_Beam_beam_key_ave = data_Beam_beam_key_ave/nbf

            if(key == 'alive'): # Needed for plotting 'lost' in percent:
                data_Beam_bunch_alv_fst = np.copy(data_Beam_bunch_key_fst)
                data_Beam_beam_alv_fst  = np.copy(data_Beam_beam_key_fst)
                # data_Beam_bunch_alv = np.copy(data_Beam_bunch_key)
                # data_Beam_beam_alv  = np.copy(data_Beam_beam_key)
                # data_Beam_bunch_alv_min = np.copy(data_Beam_bunch_key_min)
                # data_Beam_bunch_alv_ave = np.copy(data_Beam_bunch_key_ave)
                # data_Beam_bunch_alv_max = np.copy(data_Beam_bunch_key_max)
                # data_Beam_beam_alv_min = np.copy(data_Beam_beam_key_min)
                # data_Beam_beam_alv_ave = np.copy(data_Beam_beam_key_ave)
                # data_Beam_beam_alv_max = np.copy(data_Beam_beam_key_max)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            ax1[spi, 0].grid()
            ax1[spi, 1].grid()

            if(spi == n_key_list-1):
                ax1[spi, 0].set_xlabel('Bunch no.')
                ax1[spi, 1].set_xlabel('Bunch no.')

            if(  key == 'alive'):
                unitfact_spi_0_ax1 = 1
                unitfact_spi_0_ax2 = 1e11
                ax1[spi, 0].set_ylabel('Alive macrop\n')
                ax2[spi][0].set_ylabel('Intensity\n[$10^{11}$ ppb]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('Alive macrop\ngrowth [%]')
                ax2[spi][1].set_ylabel('Intensity\ngrowth [%]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
            elif(key == 'lost'):
                unitfact_spi_0_ax1 = 1
                unitfact_spi_0_ax2 = 1e9
                ax1[spi, 0].set_ylabel('Lost macrop\n')
                ax2[spi][0].set_ylabel('Lost intensity\n[$10^{9}$ ppb]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('Lost macrop\ngrowth [%]')
                ax2[spi][1].set_ylabel('Lost intensity\ngrowth [%]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
            elif(key == 'mean_dt'): # eq. bunchPosition
                unitfact_spi_0_ax1 = 1e-9
                unitfact_spi_0_ax2 = 1e3
                ax1[spi, 0].set_ylabel('mean($\Delta t$) [ns]')
                ax2[spi][0].set_ylabel('mean($\Delta t$) [$10^3$ deg]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                #
                unitfact_spi_1_ax1 = 1e-12
                unitfact_spi_1_ax2 = 1.
                ax1[spi, 1].set_ylabel('mean($\Delta t$)\ndiff [ps]')
                ax2[spi][1].set_ylabel('mean($\Delta t$)\ndiff [deg]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                #
            elif(key == 'mean_dtOff_ctr'): # eq. bunchPositionOff_ctr
                unitfact_spi_0_ax1 = 1e-12
                unitfact_spi_0_ax2 = 1.
                ax1[spi, 0].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) [ps]')
                ax2[spi][0].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) [deg]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1e-12
                unitfact_spi_1_ax2 = 1.
                ax1[spi, 1].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) diff [ps]')
                ax2[spi][1].set_ylabel('mean($\Delta t$ off)\n(wrt ctr) diff [deg]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'mean_dtOff_fit'): # eq. bunchPositionOff_fit
                unitfact_spi_0_ax1 = 1e-12
                unitfact_spi_0_ax2 = 1.
                ax1[spi, 0].set_ylabel('mean($\Delta t$ off)\n(wrt fit) [ps]')
                ax2[spi][0].set_ylabel('mean($\Delta t$ off)\n(wrt fit) [deg]')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1e-12
                unitfact_spi_1_ax2 = 1.
                ax1[spi, 1].set_ylabel('mean($\Delta t$ off)\n(wrt fit) diff [ps]')
                ax2[spi][1].set_ylabel('mean($\Delta t$ off)\n(wrt fit) diff [deg]')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'mean_dE'):
                unitfact_spi_0_ax1 = 1e6
                unitfact_spi_0_ax2 = 1e6
                ax1[spi, 0].set_ylabel('mean($\Delta E$) [MeV]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1e6
                unitfact_spi_1_ax2 = 1e6
                ax1[spi, 1].set_ylabel('mean($\Delta E$)\ndiff [MeV]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'sigma_dt'): # eq. bunchLength
                unitfact_spi_0_ax1 = 1e-9*(1./4)
                unitfact_spi_0_ax2 = 1e-9*(1./4)
                ax1[spi, 0].set_ylabel('4*rms($\Delta t$) [ns]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('4$\cdot$rms($\Delta t$)\ngrowth [%]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'sigma_dE'):
                unitfact_spi_0_ax1 = 1e6
                unitfact_spi_0_ax2 = 1e6
                ax1[spi, 0].set_ylabel('rms($\Delta E$) [MeV]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('rms($\Delta E$)\ngrowth [%]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                #
            elif(key == 'epsnrmsl'):
                unitfact_spi_0_ax1 = 1.*(1./4)
                unitfact_spi_0_ax2 = 1.*(1./4)
                ax1[spi, 0].set_ylabel('4$\pi\cdot$rms($\Delta t$)$\cdot$rms($\Delta E$)\n[eV$\cdot$s]')
                ax2[spi][0].set_ylabel('')
                ax1[spi, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax2[spi][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                #
                unitfact_spi_1_ax1 = 1./100.
                unitfact_spi_1_ax2 = 1./100.
                ax1[spi, 1].set_ylabel('4$\pi\cdot$rms($\Delta t$)$\cdot$rms($\Delta E$)\ngrowth [%]')
                ax2[spi][1].set_ylabel('')
                ax1[spi, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax2[spi][1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Bunch ave, minmax
            data_Beam_bunch_key_ave_nb = np.array([])
            data_Beam_bunch_key_min_nb = np.array([])
            data_Beam_bunch_key_max_nb = np.array([])
            for nb in range(nbf):
                data_Beam_bunch_key_ave_nb = np.append(data_Beam_bunch_key_ave_nb, np.array(data_Beam_bunch_key_ave)[-1,nb])
                data_Beam_bunch_key_min_nb = np.append(data_Beam_bunch_key_min_nb, np.array(data_Beam_bunch_key_min)[-1,nb])
                data_Beam_bunch_key_max_nb = np.append(data_Beam_bunch_key_max_nb, np.array(data_Beam_bunch_key_max)[-1,nb])
            data_Beam_bunch_key_minmax_nb = np.array([data_Beam_bunch_key_ave_nb-data_Beam_bunch_key_min_nb, data_Beam_bunch_key_max_nb-data_Beam_bunch_key_ave_nb])
            if(key == 'alive'): # needed for plotting 'lost' in percent
                data_Beam_bunch_alv_ave_nb = np.copy(data_Beam_bunch_key_ave_nb)
                data_Beam_bunch_alv_min_nb = np.copy(data_Beam_bunch_key_min_nb)
                data_Beam_bunch_alv_max_nb = np.copy(data_Beam_bunch_key_max_nb)
                #data_Beam_bunch_alv_minmax_nb = np.copy(data_Beam_bunch_key_minmax_nb)

            # BUNCH:

            ax1[spi, 0].errorbar(range(nbf), data_Beam_bunch_key_ave_nb/unitfact_spi_0_ax1, color='#888888', yerr=data_Beam_bunch_key_minmax_nb/unitfact_spi_0_ax1, ls='') #, alpha=0.33)
            ax1[spi, 0].scatter( range(nbf), data_Beam_bunch_key_ave_nb/unitfact_spi_0_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
            ax1[spi, 0].plot(    range(nbf), data_Beam_bunch_key_fst   /unitfact_spi_0_ax1, color='#3b4cc0', ls=':', alpha=0.33)
            ax1[spi, 0].scatter( range(nbf), data_Beam_bunch_key_fst   /unitfact_spi_0_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
            ax1[spi, 0].plot(    range(nbf), data_Beam_bunch_key[-1]   /unitfact_spi_0_ax1, color='#b40426', ls=':', alpha=0.33)
            ax1[spi, 0].scatter( range(nbf), data_Beam_bunch_key[-1]   /unitfact_spi_0_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)') #last turn') #at turn {lastturn} (last)')

            if(attr == 'nmacrop'):
                if(key == 'alive'):
                    # Percent growth w.r.t. its own value before tracking (turn -1)
                    ax1[spi, 1].errorbar(range(nbf), data_Beam_bunch_key_ave_nb/data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Beam_bunch_key_minmax_nb/data_Beam_bunch_key_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key_ave_nb/data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].plot(    range(nbf), data_Beam_bunch_key_fst   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key_fst   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                    ax1[spi, 1].plot(    range(nbf), data_Beam_bunch_key[-1]   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key[-1]   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)') #last turn') #at turn {lastturn} (last)')
                else:
                    # Percent growth w.r.t. "alive" value before tracking (turn -1)
                    ax1[spi, 1].errorbar(range(nbf), data_Beam_bunch_key_ave_nb/data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Beam_bunch_key_minmax_nb/data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33) # np.array([data_Beam_bunch_key_max_nb-data_Beam_bunch_key_ave_nb, data_Beam_bunch_key_ave_nb-data_Beam_bunch_key_min_nb])/data_Beam_bunch_alv_fst/unitfact_spi_1_ax1
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key_ave_nb/data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].plot(    range(nbf), data_Beam_bunch_key_fst   /data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key_fst   /data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                    ax1[spi, 1].plot(    range(nbf), data_Beam_bunch_key[-1]   /data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key[-1]   /data_Beam_bunch_alv_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)') #last turn') #at turn {lastturn} (last)')
            else:
                if('mean' in key):
                    # # Difference growth w.r.t. its own value before tracking (turn -1)
                    # print(f'key: {key}')
                    # print(f'data_Beam_bunch_key_fst = {data_Beam_bunch_key_fst}')
                    # print(f'data_Beam_bunch_key[-1] = {data_Beam_bunch_key[-1]}')
                    # print(f'data_Beam_bunch_key_ave_nb = {data_Beam_bunch_key_ave_nb}')
                    # print(f'data_Beam_bunch_key_minmax_nb = {data_Beam_bunch_key_minmax_nb}')
                    # print(f'data_Beam_bunch_key_ave_nb-data_Beam_bunch_key_fst = {data_Beam_bunch_key_ave_nb-data_Beam_bunch_key_fst}')
                    # print(f'data_Beam_bunch_key_minmax_nb-data_Beam_bunch_key_fst = {data_Beam_bunch_key_minmax_nb-data_Beam_bunch_key_fst}')
                    # print('')
                    ax1[spi, 1].errorbar(range(nbf), (data_Beam_bunch_key_ave_nb-data_Beam_bunch_key_fst)/unitfact_spi_1_ax1, color='#888888', yerr=data_Beam_bunch_key_minmax_nb/unitfact_spi_1_ax1, ls='') #, alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), (data_Beam_bunch_key_ave_nb-data_Beam_bunch_key_fst)/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].plot(    range(nbf), (data_Beam_bunch_key_fst   -data_Beam_bunch_key_fst)/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), (data_Beam_bunch_key_fst   -data_Beam_bunch_key_fst)/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                    ax1[spi, 1].plot(    range(nbf), (data_Beam_bunch_key[-1]   -data_Beam_bunch_key_fst)/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), (data_Beam_bunch_key[-1]   -data_Beam_bunch_key_fst)/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)') #last turn') #at turn {lastturn} (last)')
                else:
                    # Percent growth w.r.t. its own value before tracking (turn -1)
                    ax1[spi, 1].errorbar(range(nbf), data_Beam_bunch_key_ave_nb/data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#888888', yerr=data_Beam_bunch_key_minmax_nb/data_Beam_bunch_key_fst/unitfact_spi_1_ax1, ls='') #, alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key_ave_nb/data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#888888', marker='.', label=f'ave. turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].plot(    range(nbf), data_Beam_bunch_key_fst   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key_fst   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#3b4cc0', marker='.', label=f'bef. track') #at turn {zeroturn} (bef. track)')
                    ax1[spi, 1].plot(    range(nbf), data_Beam_bunch_key[-1]   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#b40426', ls=':', alpha=0.33)
                    ax1[spi, 1].scatter( range(nbf), data_Beam_bunch_key[-1]   /data_Beam_bunch_key_fst/unitfact_spi_1_ax1, color='#b40426', marker='.', label=f'at turn {lastturn} (last)') #last turn') #at turn {lastturn} (last)')


            # BEAM:

            if(attr == 'nmacrop'):
                ax1[spi, 0].axhline((data_Beam_beam_key_ave[-1]/nbf)/unitfact_spi_0_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                ax1[spi, 0].axhline((data_Beam_beam_key_fst    /nbf)/unitfact_spi_0_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                ax1[spi, 0].axhline((data_Beam_beam_key[-1]    /nbf)/unitfact_spi_0_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. last turn') #last turn') #at turn {lastturn} (last)')
                if(key == 'alive'):
                    ax1[spi, 1].axhline((data_Beam_beam_key_ave[-1]/nbf)/(data_Beam_beam_key_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].axhline((data_Beam_beam_key_fst    /nbf)/(data_Beam_beam_key_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                    ax1[spi, 1].axhline((data_Beam_beam_key[-1]    /nbf)/(data_Beam_beam_key_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. last turn') #last turn') #at turn {lastturn} (last)')
                else:
                    ax1[spi, 1].axhline((data_Beam_beam_key_ave[-1]/nbf)/(data_Beam_beam_alv_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].axhline((data_Beam_beam_key_fst    /nbf)/(data_Beam_beam_alv_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                    ax1[spi, 1].axhline((data_Beam_beam_key[-1]    /nbf)/(data_Beam_beam_alv_fst/nbf)/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. last turn') #last turn') #at turn {lastturn} (last)')
            else:
                ax1[spi, 0].axhline( data_Beam_beam_key_ave[-1]     /unitfact_spi_0_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                ax1[spi, 0].axhline( data_Beam_beam_key_fst         /unitfact_spi_0_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                ax1[spi, 0].axhline( data_Beam_beam_key[-1]         /unitfact_spi_0_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. last turn') #last turn') #at turn {lastturn} (last)')
                if('mean' in key):
                    ax1[spi, 1].axhline((data_Beam_beam_key_ave[-1]-data_Beam_beam_key_fst)/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].axhline((data_Beam_beam_key_fst    -data_Beam_beam_key_fst)/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                    ax1[spi, 1].axhline((data_Beam_beam_key[-1]    -data_Beam_beam_key_fst)/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. last turn') #last turn') #at turn {lastturn} (last)')
                else:
                    ax1[spi, 1].axhline(data_Beam_beam_key_ave[-1]/data_Beam_beam_key_fst/unitfact_spi_1_ax1, ls='--', color='#888888', alpha=0.50) #, label=f'beam ave. over turns [{firstturn}:{lastturn}]')
                    ax1[spi, 1].axhline(data_Beam_beam_key_fst    /data_Beam_beam_key_fst/unitfact_spi_1_ax1, ls='--', color='#3b4cc0', alpha=0.50) #, label=f'beam ave. bef. track')
                    ax1[spi, 1].axhline(data_Beam_beam_key[-1]    /data_Beam_beam_key_fst/unitfact_spi_1_ax1, ls='--', color='#b40426', alpha=0.50) #, label=f'beam ave. last turn') #last turn') #at turn {lastturn} (last)')


            # SPECIAL CASES:

            casename = os.path.split(outdir)[1]

            if('LHC-' not in casename):
                if('benchmarkTheo' in casename): # or 'test' in casename):
                    if(key == 'mean_dtOff_fit'):
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
                        # LHC tomo beam stats: taking the stats (Luis) over the bunch distributions reconstructed from LHC first-turn profiles via tomography (Theo)
                        if('6805' in casename or '7137' in casename):
                            if(  '6805' in casename): h5Filename_dummy = f'{dirinpbench}/tomo-distr/PROFILE_B1_b25540_20180616020329/LHCTomo_stats.h5'
                            elif('7137' in casename): h5Filename_dummy = f'{dirinpbench}/tomo-distr/PROFILE_B2_b30660_20180908022914/LHCTomo_stats.h5'
                            keydummy = 'mean_dtOff_ctr' if key == 'mean_dtOff_fit' else key  # To also show the black measurements in plot of offset w.r.t. fit
                            with hp.File(h5Filename_dummy, 'r') as h5File_dummy:
                                data_LHCTomo_Beam_bunch_keydummy_fst = h5File_dummy[f'/Beam/bunch/{keydummy}'][0]
                                data_LHCTomo_Beam_beam_keydummy_fst  = h5File_dummy[f'/Beam/beam/{keydummy}' ][0]
                            try: h5File_dummy.close()
                            except: pass
                            labelmeas = 'LHC tomo.'
                            correct_Offshift = True
                            if(correct_Offshift and ('mean_dtOff' in keydummy or keydummy == 'mean_dE')):
                                Offshift = data_LHCTomo_Beam_beam_keydummy_fst
                                labelmeas += f' (shift by {-1*Offshift/unitfact_spi_0_ax1:+.1f})'
                            else:
                                Offshift = 0
                            ax1[spi, 0].plot(range(len(data_LHCTomo_Beam_bunch_keydummy_fst)), (data_LHCTomo_Beam_bunch_keydummy_fst-Offshift)/unitfact_spi_0_ax1, '.:k', label=labelmeas)
                            ax1[spi, 0].axhline(                                               (data_LHCTomo_Beam_beam_keydummy_fst -Offshift)/unitfact_spi_0_ax1, ls='--', color='k', alpha=0.50)
                        #
                        # LHC bunch position offsets: taking the difference of the mean bunch position from LHC first-turn profiles w.r.t. fit (Theo)
                        if('6808' in casename or '6809' in casename or '7137' in casename):
                            if(key == 'mean_dtOff_fit'):
                                if(  '6808' in casename): data_meas = np.load(f'{dirinpbench}/profile/PROFILE_B1_b25540_20180616043102_BunchPositionVariations.npy')
                                elif('6809' in casename): data_meas = np.load(f'{dirinpbench}/profile/PROFILE_B1_b25540_20180616060554_BunchPositionVariations.npy')
                                elif('7137' in casename): data_meas = np.load(f'{dirinpbench}/profile/PROFILE_B2_b30660_20180908022914_BunchPositionVariations.npy')
                                ax1[spi, 0].plot(range(nbf), data_meas[:nbf]*1e-12/unitfact_spi_0_ax1, '--', color='#888888', label='LHC profile')
                                ax1[spi, 0].axhline(np.average(data_meas[:nbf])*1e-12/unitfact_spi_0_ax1, ls='--', color='888888', alpha=0.50)
                        #
                        # SPS bunch position offsets: taking the difference of mean bunch position from SPS raw profiles w.r.t. fit (Luis)
                        if('6805' in casename or '6808' in casename or '6809' in casename or '7137' in casename):
                            if(key == 'mean_dtOff_fit'):
                                if(  '6805' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_6805_First48b_LastBatch_B1.npy')
                                elif('6808' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_6808_First48b_LastBatch_B1.npy')
                                elif('6809' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_6809_First48b_LastBatch_B1.npy')
                                elif('7137' in casename): data_meas = np.load(f'{dirinpbench}/profile/profile_7137_First48b_PenultimateBatch_B2.npy')
                                ax1[spi, 0].plot(range(nbf), data_meas[:nbf]/unitfact_spi_0_ax1, '--c',label='SPS profile')
                                ax1[spi, 0].axhline(np.average(data_meas[:nbf])/unitfact_spi_0_ax1, ls='--', color='c', alpha=0.50)
                    #
#                    # YAML input: the casename is the concatenation of all the stages (i.e SPS or SPS-LHC or SPS-LHC-LHCramp), but we only want the case name of the last stage:
#                    allfnames = os.listdir(f'{outdir}')
#                    casename_stage = False; idxfname = 0
#                    while(idxfname < len(allfnames) and not casename_stage):
#                        if(allfnames[idxfname][:-4] in casename): casename_stage = allfnames[idxfname][:-4]
#                        idxfname += 1
                    casename_stage = casename
                    with open(f'{outdir}/{casename_stage}.pkl', 'rb') as pklFile_dummy:
                        MACupd = pickle.load(pklFile_dummy)
                        if(key == 'alive'):
                            key_dummy = 'Nb_list'
                            unit_dummy = 1.
                            ax2[spi][0].plot(range(len(MACupd[key_dummy])), MACupd[key_dummy]*unit_dummy/unitfact_spi_0_ax2, '-', color='#cc00cc', label='input') #SPS BQM (inp)')
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

            spi += 1
            gc.collect()

        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')

        fig.tight_layout()
        fig.savefig(f'{outdir}/plot_beam_{attr}_vs_bn.png')
        plt.cla()
        plt.close(fig)
        gc.collect()

    try:    h5File.close()
    except: pass


###############################################################################

# Beam phase space ------------------------------------------------------------

def plot_beam_phase_space(beam_obj,
                          profile_obj,
                          beampattern_obj, #profilepattern_obj,
                          turn_i,
                          ymin, ymax,
                          histres,             # Choose histres=Ns, that is, the the profile resolution (per bunch) to match the horizontal (time) histogram with it. The same resolution (per bunch) will be used for the vertical (energy) histogram
                          sampling=1,          # [cont. from histres] Alternatively, choose 40 S/bucket for the SPS raw profile resolution, or 100 S/bucket for the resolution of the saved LHC profiles
                          outdir='.',
                          whichbunch='first',  #'first',  # first, last, all
                          separatrices=None):  # None, or a list: [ sep0 = None or (xarray, yarray),        <- w/o int. eff
                                               #                    sep1 = None or (xarray, yarray),        <- w/  int. eff. (if available); e.g. inner separatrix 0 or OUTMOST
                                               #                    sepi = None or (xarray, yarray), ...],  <- w/  int. eff. (if available); e.f. inner separatrix i

                          # ring_obj and rfstation_obj, used to be needed, as the separatrix would be computed here internally


    '''
    Based on 'plot_long_phase_space' from BLonD's plots module.
    The associated time array of TotalInducedVoltage has to be equal to profile.bin_centers
    '''

    # Definition of some parameters from input  - - - - - - - - - - - - - - - -
    # Note that it is important to have the profile.cut_[left|right] options
    # updated, if there the dphi_rf offset is non-zero

    nbf = beampattern_obj.nbf
    nbm = beampattern_obj.nbm
    Np_list = np.array([len(idxBi) for idxBi in beampattern_obj.idxB])
    maxNp = max(Np_list)
    #Ns = profilepattern_obj.Ns
    #print(nbf, maxNp)

    if  (whichbunch == 'first' or whichbunch == 0):
        whichbunch = 0
        scattertext = f'First bunch ({whichbunch}), '
        #xmin = profile_obj.cut_left  + 0*beampattern_obj.maxbktl_0 # there is one emtpy bucket at the start as margin
        #xmax = profile_obj.cut_left  + 3*beampattern_obj.maxbktl_0
        idx0f = np.copy(beampattern_obj.idxB[0])
        #print('first', whichbunch, idx0f)
        xmin = beampattern_obj.bucket_centres[whichbunch] - 1.5*beampattern_obj.maxbktl_0
        xmax = beampattern_obj.bucket_centres[whichbunch] + 1.5*beampattern_obj.maxbktl_0
        nplotb = 1
    elif(whichbunch == 'last' or whichbunch == nbf-1):
        whichbunch = nbf-1
        scattertext = f'Last bunch ({whichbunch}), '
        # xmin = profile_obj.cut_right - 4*beampattern_obj.maxbktl_0 # there are two empty buckets at the end as margin
        # xmax = profile_obj.cut_right - 1*beampattern_obj.maxbktl_0
        idx0f = np.copy(beampattern_obj.idxB[-1])
        #print('last', whichbunch, idx0f)
        xmin = beampattern_obj.bucket_centres[whichbunch] - 1.5*beampattern_obj.maxbktl_0
        xmax = beampattern_obj.bucket_centres[whichbunch] + 1.5*beampattern_obj.maxbktl_0
        nplotb = 1
    elif(whichbunch == 'all'):
        whichbunch = np.arange(nbf)
        scattertext = ''
        idx0f = np.arange(beampattern_obj.idxB[0][0], beampattern_obj.idxB[-1][-1]+1) #np.array([j for idxBi in beampattern_obj.idxB for j in idxBi])
        #print('all', whichbunch, idx0f)
        xmin = profile_obj.cut_left
        xmax = profile_obj.cut_right
        nplotb = 'all' # nplotb is used to compute the bin width of the histograms. In the cases above, it can be different to the profile resolution by
                       # the user's request; when plotting all bunches, the only available option is the same profile resolution.
    else:
        sys.error(f'[!] ERROR: No valid whichbunch = {whichbunch} option!')

    #print(whichbunch, beampattern_obj.bucket_centres, beampattern_obj.bucket_centres[whichbunch], xmin, xmax)

    scattertext += f'turn {turn_i:d}'



    # Parameters for size/positioning of the axes - - - - - - - - - - - - - - -

    #          wa  X1    W1      wb  X2  W2    wb
    #  1.05-+------+----------+------+------+------+
    #    ^  |      |          |      |      |      |
    #  0.10 |      |          |      |      |      | hb
    #    v  |      |          |      |      |      |
    #  0.95-+------+----------+------+------+------+
    #    ^  |      |XXXXXXXXXX|      |      |      |
    #  0.20 |      |XXXXXXXXXX|      |      |      | H2
    #    v  |      |XXXXXXXXXX|      |      |      |
    #  0.75-+------+----------+------+------+------+-Y2
    #    ^  |      |          |      |      |      |
    #  0.10 |      |          |      |      |      | hb
    #    v  |      |          |      |      |      |
    #  0.65-+------+----------+------+------+------+
    #    ^  |      |XXXXXXXXXX|      |XXXXXX|      |
    #       |      |XXXXXXXXXX|      |XXXXXX|      |
    #  0.50 |      |XXXXXXXXXX|      |XXXXXX|      | H1
    #       |      |XXXXXXXXXX|      |XXXXXX|      |
    #    v  |      |XXXXXXXXXX|      |XXXXXX|      |
    #  0.15-+------+----------+------+------+------+-Y1
    #    ^  |      |          |      |      |      |
    #  0.15 |      |          |      |      |      | ha
    #    v  |      |          |      |      |      |
    #  0.00-+------+----------+------+------+------+
    #       |<0.10>|<  0.70  >|<0.05>|<0.15>|<0.05>|
    #     0.00   0.10       0.80   0.85   1.00   1.05


    W1 = 0.70
    W2 = 0.12
    H1 = 0.60
    H2 = 0.20
    X1 = 0.10
    Y1 = 0.10
    wa = X1
    ha = Y1
    wb = 0.5*(1.00-W1-W2-wa)
    hb = 0.5*(1.00-H1-H2-ha)
    X2 = wa+W1+wb
    Y2 = ha+H1+hb

    rect_scatter = [X1, Y1, W1, H1]
    rect_histx   = [X1, Y2, W1, H2]
    rect_histy   = [X2, Y1, W2, H1]

    fig = plt.figure(1)
    fig.set_size_inches(8.0,6.0)
    axScatter = plt.axes(rect_scatter)
    axHistx   = plt.axes(rect_histx)
    axHisty   = plt.axes(rect_histy)

    axScatter.xaxis.set_major_locator(AutoLocator())
    axScatter.xaxis.set_minor_locator(AutoLocator())
    #axScatter.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #axScatter.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Separatrices  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if separatrices is not None:
        for i in range(len(separatrices)):
            if(i == 0):
                myc  = '#000000'
                mylw = 0.10
            else:
                myc  = '#00ff00'
                mylw = 0.25
            if(separatrices[i] is not None):
                for j in [-1, 1]: # To plot lower and upper sides
                    axScatter.plot(separatrices[i][0]/1e-9,  j*separatrices[i][1]/1e6, '-', color=myc, lw=mylw)

    gc.collect()

    # Bunches - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    indlost  = np.where( beam_obj.id[ idx0f[0]:idx0f[-1]+1:sampling ] == 0 )[0] + idx0f[0] # particles lost
    indalive = np.where( beam_obj.id[ idx0f[0]:idx0f[-1]+1:sampling ] != 0 )[0] + idx0f[0] # particles transmitted
    #print(f'indlost = {indlost}')
    #print(f'indalive = {indalive}')

    if(maxNp < 1e3): myalpha = 1.
    else:            myalpha = np.exp( np.log(0.02)/(1e6-1e3) *(maxNp-1e3)) # 1.00 below 1k macrop, 0.02 at 1M macrop
    #myalpha  = np.interp(, [1e3, 1e6], [1.0, 0.001])
    #print(f'myalpha = {myalpha} -> {str(1.-myalpha)}')

    if(maxNp < 1e3): mys = 0.3
    else:            mys = 0.1

    axScatter.scatter(    beam_obj.dt[indalive]/1e-9, beam_obj.dE[indalive]/1e6, s=mys, edgecolor='none', color='#0000ff', alpha=myalpha)
    if len(indlost) > 0:
        axScatter.scatter(beam_obj.dt[indlost] /1e-9, beam_obj.dE[indlost] /1e6, s=mys, edgecolor='none', color='#ff0000', alpha=1.00)

    # Bucket centres -- ideal bucket centres based on the lowest harmonic; they can displace due to dphi_rf offset if requested in bucket_centres.
    # Also, two vertical lines (one at 0, the other at the end of the last idel buckets) for reference
    zeros = np.zeros(nbf)
    axScatter.axvline((beampattern_obj.bucket_centres_0[0    ] - 0.5*beampattern_obj.maxbktl_0)/1e-9, color='#000000', ls='-', lw=0.10) # at zero (exact)
    axScatter.axvline((beampattern_obj.bucket_centres_0[nbf-1] + 0.5*beampattern_obj.maxbktl_0)/1e-9, color='#000000', ls='-', lw=0.10)
    axScatter.plot(beampattern_obj.bucket_centres_0[whichbunch]/1e-9, zeros[whichbunch], 'P', mfc='#aaaaaa', mec='#000000', ms=7, mew=0.4, label='Bucket centre (no int. eff., start)')
    axScatter.plot(beampattern_obj.bucket_centres[  whichbunch]/1e-9, zeros[whichbunch], 'P', mfc='#ffffff', mec='#000000', ms=7, mew=0.4, label='Bucket centre (no int. eff)')

    # Bunch centres from stats
    beamstats = BeamTools.stats_bunch(beam_obj, beampattern_obj.idxB, beampattern_obj.bucket_centres)
    axScatter.plot(beamstats['mean_dt'][whichbunch]/1e-9, beamstats['mean_dE'][whichbunch]/1e6, 'X', mfc='#66ccff', mec='#000000', ms=7, mew=0.4, label='Bunch centroid')

    #

    axScatter.legend(loc='lower left', ncol=1, fontsize='small', frameon=False)

    axScatter.set_xlabel(r'$\Delta t$ [ns]')
    axScatter.set_ylabel(r'$\Delta E$ [MeV]')
    axScatter.yaxis.labelpad = 1

    axScatter.set_xlim(xmin/1e-9, xmax/1e-9)
    axScatter.set_ylim(ymin/1e6,  ymax/1e6)

    axScatter.text(0.50, 0.95, scattertext, transform=axScatter.transAxes, ha='center', va='center')

    gc.collect()

    # Phase and momentum histograms - - - - - - - - - - - - - - - - - - - - - -

    if isinstance(whichbunch, int): # A single bunch
        xbin = (xmax - xmin)/(histres*(nplotb+2))
        xh   = np.arange(xmin, xmax, xbin)
        # print(f'xbin = {xbin}')
    else: # All bunches
        xh = profile_obj.edges
    ybin = 5e6 #  Changed to a fixed resoultion of 5 MeV instead of (ymax - ymin)/ histres
    yh   = np.arange(ymin, ymax, ybin)
    # print(f'xh = {xh}, shape = {xh.shape}')
    # print(f'yh = {yh}, shape = {yh.shape}')

    axHistx.hist(beam_obj.dt[ idx0f[0]:idx0f[-1]:sampling ]/1e-9,  bins=xh/1e-9, histtype='step', color='#0000ff')
    axHisty.hist(beam_obj.dE[ idx0f[0]:idx0f[-1]:sampling ]/1e6,   bins=yh/1e6,  histtype='step', color='#0000ff', orientation='horizontal')

    axHistx.ticklabel_format(style='sci') #, axis='y', scilimits=(0,0))
    axHisty.ticklabel_format(style='sci') #, axis='x', scilimits=(0,0))

    axHistx.axes.get_xaxis().set_visible(False)
    axHisty.axes.get_yaxis().set_visible(False)

    axHistx.set_xlim(xmin/1e-9, xmax/1e-9)
    axHisty.set_ylim(ymin/1e6, ymax/1e6)

    labels = axHisty.get_xticklabels()
    for label in labels:
        label.set_rotation(-90)

    gc.collect()

    # Save plot - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if isinstance(whichbunch, int):
        #print(whichbunch, 'float')
        pass
    else:
        #print(whichbunch, 'Not float')
        whichbunch = f'{whichbunch[0]}to{whichbunch[-1]}'

    #fig.tight_layout()
    fig.savefig(f'{outdir}/plot_beam_phase_space_{whichbunch}_{turn_i:d}.png')
    plt.cla()
    fig.clf()
    plt.close(fig)
    gc.collect()

###############################################################################






###############################################################################
# # Retrieve home path and re-assign if running in afs or batch
# ###############################################################################
# print('')
# home = str(pathlib.Path.home())
# print('home =', home)

# if(  home.startswith('/home') ): myenv = 'homeUbu'
# elif(home.startswith('/Users')): myenv = 'homeMac'
# elif(home.startswith('/afs')  ): myenv = 'afs'
# elif(home.startswith('/pool') ): myenv = 'batch'
# print('myenv =', myenv)

# if(myenv == 'afs'):   home = '/afs/cern.ch/work/l/lmedinam'
# if(myenv == 'batch'): home = '/afs/cern.ch/work/l/lmedinam'
# print('home =', home)
# print('')
# ###############################################################################

# # My custom functions for BLonD

# sys.path.insert(0, home+'/BLonD_simulations/BLonDBasedTools/blond_lm')
# #from beam.beam_tools import BeamTools
# #from profile.profile_tools import ProfileTools
# #from input_output.input_output_tools import print_object_attributes
# #from input_output.input_output_tools import mynpsave, mynpload # myh5save, myh5load
