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

def plot_wake_vs_time(time, wake, titlestr='', outdir='.'):

    fig = plt.figure()
    fig.set_size_inches(8.0,6.0)
    ax = plt.axes([0.15, 0.1, 0.8, 0.8])

    ax.grid()

    ax.plot(time/1e-9, wake/1e0, '-')

    ax.set_xlim(0.0, 5.0) #e9)
    #ax.set_ylim(-0.2, 5.0)

    ax.set_xlabel('Wake length [ns]')   #, fontsize=16)
    ax.set_ylabel(r'Wake [$\Omega/s$]') #, fontsize=16)

    ax.set_title(titlestr)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(f'{outdir}/plot_wake_vs_time.png')
    plt.cla()
    plt.close(fig)


def plot_impedance_vs_freq(freq, impedance, titlestr='', outdir='.', ZZn='Z'):

    fig, ax = plt.subplots(2, sharex=True, sharey=True) #len(beamstats_keys), sharex=True)
    fig.set_size_inches(8.0,6.0)

    ax[0].grid()
    ax[1].grid()

    # xmin = 0.3999 # GHz
    # xmax = 0.4001 # GHz
    # ymin= -0.000 # MOhm
    # ymax = 0.015 # MOhm

    xmin = 0.00 # GHz
    xmax = 3.00 # GHz
    ymin= -0.05 # MOhm
    ymax = 0.50 # MOhm

   #  # LHC
   #  #modelDir = '/afs/cern.ch/work/l/lmedinam/BLonD_simulations/ImpedanceRepositories/LHC-web'
   #  #imp = 'Zlong_Allthemachine_450GeV_B1_LHC_fill6650_2018-05-07_07-41-00'
   # ##imp = 'Zlong_Allthemachine_450GeV_B1_LHC_inj_450GeV_B1'
   #  # HL-LHC
   #  modelDir = '/afs/cern.ch/work/l/lmedinam/BLonD_simulations/ImpedanceRepositories/HLLHC-github'
   #  imp = 'Zlong_Allthemachine_450GeV_B1_injection_v1p4_crab_fully_updated_tapers_exact_weld_DRF_BB_option1_5GHz'
   #  #
   #  impedancetable_freq_full, impedancetable_ReZlong_full, impedancetable_ImZlong_full = np.loadtxt(f'{modelDir}/scenarios/{imp}.dat', delimiter="\t", skiprows=1, unpack=True)
   #  print(impedancetable_freq_full, impedancetable_ReZlong_full, impedancetable_ImZlong_full)

   #  idxmin = np.argmax(impedancetable_freq_full > xmin*1e9)
   #  idxmax = np.argmax(impedancetable_freq_full > xmax*1e9)
   #  print(idxmin, impedancetable_freq_full[idxmin])
   #  print(idxmax, impedancetable_freq_full[idxmax])

    # impedancetable_freq_full = impedancetable_freq_full[idxmin+1:idxmax]
    # impedancetable_full = impedancetable_ReZlong_full[idxmin+1:idxmax] + 1j*impedancetable_ImZlong_full[idxmin+1:idxmax]

    if(ZZn == 'Z'):

        ax[0].set_ylabel(r'Re(Z), Im(Z) [M$\Omega$]') # from Profile)
        ax[1].set_ylabel(r'|Z| [M$\Omega$]') # from Profile)
        ax[1].set_xlabel('Frequency [GHz]')

        # ax[1].plot(impedancetable_freq_full/1e9, abs(impedancetable_full)/1e6, '+g', label='Total full')

        ax[0].plot(freq/1e9, np.real(impedance)/1e6, '-', label='Re')
        ax[0].plot(freq/1e9, np.imag(impedance)/1e6, '-', label='Im')
        ax[1].plot(freq/1e9,     abs(impedance)/1e6, '-k', label='Total')


    elif(ZZn == 'Zn'):

        ax[0].set_ylabel(r'Re(Z/ (2$\pi$f/$\omega_{rev}$)), Im(Z/ (2$\pi$f/$\omega_{rev}$)) [$\Omega$]') # from Profile)
        ax[1].set_ylabel(r'| Z / (2$\pi$f/$\omega_{rev}$) | [$\Omega$]') # from Profile)
        ax[1].set_xlabel('Frequency [GHz]')

        ax[0].plot(freq/1e9, np.real(impedance), '-', label='Re')
        ax[0].plot(freq/1e9, np.imag(impedance), '-', label='Im')
        ax[1].plot(freq/1e9,     abs(impedance), '-k', label='Total')


    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)

    #ax[0].set_ylim(-0.05, 0.5)
    #ax[0].set_xlim(0.0, 3.0)

    ax[0].legend(loc=2)
    ax[1].legend(loc=2)

    fig.suptitle(titlestr)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(f'{outdir}/plot_impedance_vs_freq.png')
    plt.cla()
    plt.close(fig)


###############################################################################

def plot_induced_voltage_at_turn(outdir, time_array, induced_voltage, turn_i, tmax=None):

    fig, ax = plt.subplots()
    fig.set_size_inches(8.0,6.0)

    ax.grid()
    ax.set_xlabel('Time [$\mu$s]')
    ax.set_ylabel('Induced voltage [MV]')
    unit_factor = 1e6

    ax.plot(time_array/1e-6, induced_voltage/unit_factor, '-k', alpha=1.00, label=f'Turn {turn_i:d}')

    ax.legend(loc=4)
    ax.set_title(f'Turn {turn_i:d}', ha='right', va='center')

    if tmax is not None:
        #if tmax < time_array[-1]:
        #    idxtmax = np.argmax(time_array > tmax)
        #    tmax =
        ax.set_xlim(0, tmax/1e-6)

    fig.tight_layout()
    fig.savefig(f'{outdir}/plot_induced_voltage_at_turn_{turn_i:d}.png')
    #print('Saving', f'{outdir}/plot_induced_voltage.png'+'...')
    plt.cla()
    plt.close(fig)

#############

def plot_impedances_induced_voltage_abs_vs_turn(outdir):

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_turns = h5File['/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_turns)
        data_turns = h5File['/turns'][:idxlastturn+1]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        #induced_voltage_minabs = h5File['/TotalInducedVoltage/induced_voltage_minabs'][:idxlastturn+1]
        #induced_voltage_aveabs = h5File['/TotalInducedVoltage/induced_voltage_aveabs'][:idxlastturn+1]
        induced_voltage_maxabs = h5File['/TotalInducedVoltage/induced_voltage_maxabs'][:idxlastturn+1]

        gc.collect()

        fig, ax = plt.subplots()
        fig.set_size_inches(8.0,6.0)

        if(len(data_turns) == 2): mymarker = '.' # Turns -1 and 0
        else:                     mymarker = '-'

        ax.grid()

        unit_factor = 1e6
        ax.set_xlabel('Turns')
        ax.set_ylabel(r'max($|V_\mathsf{ind}|$) [MV]')

        #ax.fill_between(data_turns, induced_voltage_maxabs/unit_factor, induced_voltage_minabs/unit_factor, alpha=0.2)
        #ax.plot(        data_turns, induced_voltage_aveabs/unit_factor, mymarker+'k', alpha=1.00)
        ax.plot(         data_turns, induced_voltage_maxabs/unit_factor, mymarker+'k', alpha=1.00)

        fig.tight_layout()
        fig.savefig(f'{outdir}/plot_impedances_induced_voltage_abs_vs_turn.png')
        plt.cla()
        plt.close(fig)
        gc.collect()

    try:    h5File.close()
    except: pass

###############################################################################
