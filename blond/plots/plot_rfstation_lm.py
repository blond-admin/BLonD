#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import gc
import numpy as np
import h5py as hp
import pathlib

import matplotlib.pyplot as plt

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

if myenv in ['afs', 'batch', 'hpc']:
    import matplotlib as mpl
    mpl.use('Agg')

dirbase = 'BLonD_simulations/sps_lhc_losses'
dirin   = f'{dirhome}/{dirbase}/sps_lhc_losses'
dirinp  = f'{dirin}/inp'
dirinpbench  = f'{dirinp}/benchmark'

###############################################################################

# Momentum and voltage vs turn ------------------------------------------------

def plot_rfstation_vs_turn(outdir, rfstation_obj):

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    turns = range(len(rfstation_obj.momentum))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    key_list = ['momentum', 'voltage']

    fig, ax = plt.subplots(len(key_list), sharex=True)
    fig.set_size_inches(8.0,2.0*len(key_list))

    gc.collect()

    spi = 0
    for key in key_list:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        rfstation_key = getattr(rfstation_obj, key)

        if(key in ['voltage']):
            nrf = len(rfstation_key)
            #print(f'nrf ={nrf}')
        if(key == 'momentum'):
            val_key_0 = rfstation_key[0]
        elif(key == 'voltage'):
            val_key_0 = np.array([rfstation_key[i,0] for i in range(nrf)])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ax[spi].grid()

        if(spi == len(key_list)-1): ax[spi].set_xlabel('Turns')

        if(key == 'momentum'):
            unit_factor = 1e6
            #ax[spi].set_ylabel(r'Momentum - $450$ GeV [MeV]') # LHCramp['ps'] = 451.15
            ax[spi].set_ylabel(r'$\Delta p_s$ [MeV]') # LHCramp['ps'] = 451.15
            ax[spi].plot(turns, (rfstation_key-val_key_0)/unit_factor, '-k', alpha=1.00)
        elif(key == 'voltage'):
            unit_factor = 1e3
            ax[spi].set_ylabel(r'$\Delta V_{RF}$ [kV]')
            for irf in range(nrf):
                ax[spi].plot(turns, (rfstation_key[irf]-val_key_0[irf])/unit_factor, '-', alpha=1.00)

        spi += 1
        gc.collect()

    fig.tight_layout()
    fig.savefig(f'{outdir}/plot_rfstation_vs_turn.png')
    plt.cla()
    plt.close(fig)
    gc.collect()

###############################################################################

