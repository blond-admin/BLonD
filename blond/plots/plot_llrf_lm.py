#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import numpy as np
import h5py as hp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler

# CavityFeedback parameters all turns -----------------------------------------

def plot_cavityfeedback_allturns(outdir, t0R, t1L, bin_size=None, vrf_phirf_d=None, vrf_phirf_i=None, sampling=1): # If bin_size is not None, plot is vs time; if None, plot is vs. bin no.

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        idxfirstturn = 0 # 0 if we want to include turn '-1', or 1 if we want from turn '0'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_turns = h5File['/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_turns)
        data_turns = h5File['/turns'][idxfirstturn:idxlastturn+1]

        #print(f'data_turns = {data_turns}, shape = {data_turns.shape}')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        key_list = ['V_sum', 'V_corr', 'phi_corr']

        nVsumextraplots = 3 # From V_sum, we plot Re, Im, abs, angle
        
        fig, ax = plt.subplots(len(key_list)+nVsumextraplots, 2, sharex='col', sharey='row')
        fig.set_size_inches(8.0,2.0*(len(key_list)+nVsumextraplots))
        #plt.subplots_adjust(wspace=0.1)

        t0R /= 1e-6
        t1L /= 1e-6
        #t0R *= bin_size/1e-6
        #t1L *= bin_size/1e-6
        #print(f't0R = {t0R}')
        #print(f't1L = {t1L}')

        spi = 0
        for key in key_list:

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            data_CavityFeedback_key = h5File[f'/CavityFeedback/{key}'][idxfirstturn:idxlastturn+1]
            #print(f'{key}, data_CavityFeedback_key = {data_CavityFeedback_key}, shape = {data_CavityFeedback_key.shape}')

            nturns  = len(data_CavityFeedback_key)
            nslices = len(data_CavityFeedback_key[0])
            time_array = np.arange(nslices).astype(float)
            if(bin_size is not None): time_array *= bin_size/1e-6

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            mycm = plt.get_cmap('coolwarm')
            color_cycle_nturns = [mycm(i/(nturns-1.)) for i in range(nturns)] # For Re and Im

            for i in range(nturns):
                if(i == 0 or i == nturns-1): label_turni = str(data_turns[i])
                else:                        label_turni = None

#            if(len(data_turns) == 2-idxfirstturn): mymarker = '.-' # Turns -1 and 0
#            else:                                  mymarker = '-'

            for spj in range(2):
                ax[spi,spj].set_prop_cycle(cycler('color', color_cycle_nturns))
                ax[spi,spj].grid()

            if(spi == len(key_list)+nVsumextraplots-1):
                for spj in range(2):
                    if(bin_size is not None): ax[spi,spj].set_xlabel('Time [$\mu$s]')
                    else:                     ax[spi,spj].set_xlabel('Slice index [1]')

            # hide the spines between ax and ax2
            ax[spi,0].spines['right'].set_visible(False)
            ax[spi,1].spines['left' ].set_visible(False)
            ax[spi,0].yaxis.tick_left()
            ax[spi,1].yaxis.tick_right()
            ax[spi,0].tick_params(labelright=False)
            ax[spi,1].tick_params(labelleft=False)
            ax[spi,0].tick_params(right=False)
            ax[spi,1].tick_params(left=False)

            if(  key == 'V_sum'):
                unit_factor_1 = 1e6
                unit_factor_2 = 1
                ax[spi  ,0].set_ylabel('Re($V_\mathsf{sum}$) [MV]')
                ax[spi+1,0].set_ylabel('Im($V_\mathsf{sum}$) [MV]')
                ax[spi+2,0].set_ylabel('|$V_\mathsf{sum}$| [MV]')
                ax[spi+3,0].set_ylabel(r'$\frac{\pi}{2}$ - $\mathsf{tan}^{-1}$($V_\mathsf{sum}$) [rad]')
                ax[spi  ,spj].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                for ii in range(1,nVsumextraplots+1):
                    for spj in range(2):
                        ax[spi+ii,spj].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                        ax[spi+ii,spj].set_prop_cycle(cycler('color', color_cycle_nturns))
                        ax[spi+ii,spj].grid()
                    ax[spi+ii,0].spines['right'].set_visible(False)
                    ax[spi+ii,1].spines['left' ].set_visible(False)
                    ax[spi+ii,0].yaxis.tick_left()
                    ax[spi+ii,1].yaxis.tick_right()
                    ax[spi+ii,0].tick_params(labelright=False)
                    ax[spi+ii,1].tick_params(labelleft=False)
                    ax[spi+ii,0].tick_params(right=False)
                    ax[spi+ii,1].tick_params(left=False)
                #print('data_CavityFeedback_Vsum =', data_CavityFeedback_key[nturns-1][::sampling])
                #print('angle(data_CavityFeedback_Vsum =',np.angle(data_CavityFeedback_key[nturns-1][::sampling])/unit_factor_2)
                #print('pi/2-angle(data_CavityFeedback_Vsum) =',np.pi/2. - np.angle(data_CavityFeedback_key[nturns-1][::sampling])/unit_factor_2)
                ##self.V_corr, alpha_sum = cartesian_to_polar(self.V_sum)
                for i in range(nturns-1):
                    for spj in range(2):
                        if(i == 0 and spj == 1): label_turni = f'Turn {data_turns[i]}'
                        else:                    label_turni = None
                        ax[spi  ,spj].plot(time_array[::sampling],             np.real(data_CavityFeedback_key[i][::sampling])/unit_factor_1,      alpha=1.00, label=label_turni)
                        ax[spi+1,spj].plot(time_array[::sampling],             np.imag(data_CavityFeedback_key[i][::sampling])/unit_factor_1,      alpha=1.00, label=label_turni)
                        ax[spi+2,spj].plot(time_array[::sampling],              np.abs(data_CavityFeedback_key[i][::sampling])/unit_factor_1,      alpha=1.00, label=label_turni)
                        ax[spi+3,spj].plot(time_array[::sampling], np.pi/2. - np.angle(data_CavityFeedback_key[i][::sampling])/unit_factor_2,      alpha=1.00, label=label_turni)
                if(nturns >= 2):
                    i = nturns-1
                    for spj in range(2):
                        if(spj == 1): label_turni = f'Turn {data_turns[i]}'
                        else:         label_turni = None
                        ax[spi  ,spj].plot(time_array[::sampling],             np.real(data_CavityFeedback_key[i][::sampling])/unit_factor_1, 'k', alpha=1.00, label=label_turni)
                        ax[spi+1,spj].plot(time_array[::sampling],             np.imag(data_CavityFeedback_key[i][::sampling])/unit_factor_1, 'k', alpha=1.00, label=label_turni)
                        ax[spi+2,spj].plot(time_array[::sampling],              np.abs(data_CavityFeedback_key[i][::sampling])/unit_factor_1, 'k', alpha=1.00, label=label_turni)
                        ax[spi+3,spj].plot(time_array[::sampling], np.pi/2. - np.angle(data_CavityFeedback_key[i][::sampling])/unit_factor_2, 'k', alpha=1.00, label=label_turni)
                if(vrf_phirf_d is not None):  # vrf and phir_rf at first turn
                    #print(f'vrf_phirf_d = {vrf_phirf_d}')
                    for spj in range(2):
                        ax[spi+2,spj].axhline(vrf_phirf_d[0]/unit_factor_1, ls=':',  color='#000000', alpha=1.0)
                        ax[spi+3,spj].axhline(vrf_phirf_d[1]/unit_factor_2, ls=':',  color='#000000', alpha=1.0)
                if(vrf_phirf_i is not None):  # vrf and phir_rf at current turn
                    #print(f'vrf_phirf_i = {vrf_phirf_i}')
                    for spj in range(2):
                        ax[spi+2,spj].axhline(vrf_phirf_i[0]/unit_factor_1, ls='--', color='#000000', alpha=1.0)
                        ax[spi+3,spj].axhline(vrf_phirf_i[1]/unit_factor_2, ls='--', color='#000000', alpha=1.0)
                          
                ax[spi+2,1].legend(loc=1)
                ax[spi,  0].set_xlim(0., t0R)
                ax[spi,  1].set_xlim(t1L, time_array[-1])
                for ii in range(1,nVsumextraplots+1):
                    ax[spi+ii,0].set_xlim(0., t0R)
                    ax[spi+ii,1].set_xlim(t1L, time_array[-1])
                spi += nVsumextraplots

            elif(key == 'V_corr'):
                unit_factor_1 = 1
                ax[spi,0].set_ylabel('$V_\mathsf{corr}$ [$V_\mathsf{RF0}$]')
                ax[spi,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                for i in range(nturns-1):
                    for spj in range(2):
                        if(i == 0 and spj == 1): label_turni = f'Turn {data_turns[i]}'
                        else:                    label_turni = None
                        ax[spi,spj].plot(time_array[::sampling], data_CavityFeedback_key[i][::sampling]/unit_factor_1,      alpha=1.00, label=label_turni)
                if(nturns >= 2):
                    i = nturns-1
                    for spj in range(2):
                        if(spj == 1): label_turni = f'Turn {data_turns[i]}'
                        else:         label_turni = None
                        ax[spi,spj].plot(time_array[::sampling], data_CavityFeedback_key[i][::sampling]/unit_factor_1, 'k', alpha=1.00, label=label_turni)
                for spj in range(2):
                    ax[spi,spj].axhline(1., ls='-.', color='#000000', alpha=1.0)
                ax[spi,1].legend(loc=1)
                ax[spi,0].set_xlim(0., t0R)
                ax[spi,1].set_xlim(t1L, time_array[-1])

            elif(key == 'phi_corr'):
                unit_factor_1 = 1
                ax[spi,0].set_ylabel('$\phi_\mathsf{corr}$ [rad]')
                ax[spi,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                for i in range(nturns-1):
                    for spj in range(2):
                        if(i == 0  and spj == 1): label_turni = f'Turn {data_turns[i]}'
                        else:                     label_turni = None
                        ax[spi,spj].plot(time_array[::sampling], data_CavityFeedback_key[i][::sampling]/unit_factor_1,      alpha=1.00, label=label_turni)
                if(nturns >= 2):
                    i = nturns-1
                    for spj in range(2):
                        if(spj == 1): label_turni = f'Turn {data_turns[i]}'
                        else:         label_turni = None
                        ax[spi,spj].plot(time_array[::sampling], data_CavityFeedback_key[i][::sampling]/unit_factor_1, 'k', alpha=1.00, label=label_turni)
                for spj in range(2):
                    ax[spi,spj].axhline(0., ls='-.', color='#000000', alpha=1.0)
                ax[spi,0].set_xlim(0., t0R)
                ax[spi,1].set_xlim(t1L, time_array[-1])

            spi += 1
            #print('')

        fig.tight_layout() #w_pad=1.0) # fig.tight_layout(pad=5.0, h_pad=0.0)
        fig.savefig(f'{outdir}/plot_cavityfeedback_allturns.png')
        plt.cla()
        plt.close(fig)

        # Polar:
        
        # radar green, solid grid lines
        #plt.rc('grid', color='#aaaaaa', linewidth=1, linestyle='-')
        #plt.rc('xtick', labelsize=15)
        #plt.rc('ytick', labelsize=15)

        fig = plt.figure()
        fig.set_size_inches(6.0, 6.0)
        
        ax = fig.add_axes([0.075, 0.050, 0.875, 0.850], projection='polar')

        key = 'V_sum'
        
        data_CavityFeedback_key = h5File[f'/CavityFeedback/{key}'][idxfirstturn:idxlastturn+1]
        #print(f'{key}, data_CavityFeedback_key = {data_CavityFeedback_key}, shape = {data_CavityFeedback_key.shape}')

        nturns  = len(data_CavityFeedback_key)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        mycm = plt.get_cmap('coolwarm')
        color_cycle_nturns = [mycm(i/(nturns-1.)) for i in range(nturns)] # For Re and Im

        for i in range(nturns):
            if(i == 0 or i == nturns-1): label_turni = str(data_turns[i])
            else:                        label_turni = None

        #ax.gca().set_prop_cycle(cycler('color', color_cycle_nturns))
        ax.set_thetagrids( angles=np.arange(0., 360., 15.) ) # deg
        ax.set_rlabel_position(2.5) # deg
        ax.grid( color='#aaaaaa', linewidth=1, linestyle='-' )

        unit_factor_1 = 1e6
        unit_factor_2 = 1

        for i in range(nturns-1):
            if(i == 0): label_turni = f'Turn {data_turns[i]}'
            else:       label_turni = None
            ax.plot(np.pi/2. - np.angle(data_CavityFeedback_key[i][::sampling])/unit_factor_2, np.abs(data_CavityFeedback_key[i][::sampling])/unit_factor_1,  c=color_cycle_nturns[i], alpha=1.00, label=label_turni)
        if(nturns >= 2):
            i = nturns-1
            label_turni = f'Turn {data_turns[i]}'
            ax.plot(np.pi/2. - np.angle(data_CavityFeedback_key[i][::sampling])/unit_factor_2, np.abs(data_CavityFeedback_key[i][::sampling])/unit_factor_1, 'k',                      alpha=1.00, label=label_turni)
        if(vrf_phirf_d is not None):  # vrf and phir_rf at first turn
            ax.plot(vrf_phirf_d[1]/unit_factor_2, vrf_phirf_d[0]/unit_factor_1, 'o', markerfacecolor='#ffffff', markeredgecolor='#888888', markeredgewidth=1.0, alpha=1.0, label='SP: bef. track')
        if(vrf_phirf_i is not None):  # vrf and phir_rf at current turn
            ax.plot(vrf_phirf_i[1]/unit_factor_2, vrf_phirf_i[0]/unit_factor_1, 'o', markerfacecolor='#000000', markeredgecolor='#888888', markeredgewidth=1.0, alpha=1.0, label=f'SP: turn {data_turns[nturns-1]}')

        #plt.gca().set_prop_cycle(cycler('color', color_cycle_nturns))
        
        ax.set_ylim(0.5*vrf_phirf_d[0]/unit_factor_1, 1.5*vrf_phirf_d[0]/unit_factor_1)
        
        ax.set_title(r'$V_\mathsf{sum}$ = ( |$V_\mathsf{sum}$| [MV],  $\frac{\pi}{2}$ - $\mathsf{tan}^{-1}$($V_\mathsf{sum}$) [deg] ) ')
        ax.legend(loc=3, framealpha=1.0)

        #fig.tight_layout() #w_pad=1.0) # fig.tight_layout(pad=5.0, h_pad=0.0)
        fig.savefig(f'{outdir}/plot_cavityfeedback_polar_allturns.png')
        plt.cla()
        plt.close(fig)
        

    try:    h5File.close()
    except: pass

# CavityFeedback parameters vs turn -------------------------------------------

def plot_cavityfeedback_vs_turn(outdir):

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        idxfirstturn = 0 # 0 if we want to include turn '-1', or 1 if we want from turn '0'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_turns = h5File['/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_turns)
        data_turns = h5File['/turns'][idxfirstturn:idxlastturn+1]

        #print(f'data_turns = {data_turns}, shape = {data_turns.shape}')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        key_list = ['V_sum', 'V_corr', 'phi_corr']

        fig, ax = plt.subplots(len(key_list), sharex=True)
        fig.set_size_inches(8.0,2.0*len(key_list))

        spi = 0
        for key in key_list:

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            data_CavityFeedback_key = h5File[f'/CavityFeedback/{key}'][idxfirstturn:idxlastturn+1]
            #print(f'{key}, data_CavityFeedback_key = {data_CavityFeedback_key}, shape = {data_CavityFeedback_key.shape}')

            #nturns  = len(data_CavityFeedback_key)
            #nslices = len(data_CavityFeedback_key[0])

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            mycm = plt.get_cmap('coolwarm')
            color_cycle_nb = [mycm(ii/(2-1.)) for ii in range(2)] # For Re and Im
            color_cycle_nb.reverse()

#            for nb in range(nrf):
#                if(nb == 0 or nb == nrf-1): label_nb = str(nb+1)
#                else:                       label_nb = None

#            if(len(data_turns) == 2-idxfirstturn): mymarker = '.-' # Turns -1 and 0
#            else:                                  mymarker = '-'

            ax[spi].set_prop_cycle(cycler('color', color_cycle_nb))
            ax[spi].grid()

            if(spi == len(key_list)-1): ax[spi].set_xlabel('Turns')

        #ax.set_ylabel(r'max($|V_\mathsf{ind}|$) [MV]')

            if(  key == 'V_sum'):
                unit_factor = 1e6
                #ax[spi].set_ylabel('max(Re|$V_\mathsf{sum}$|),\nmax(Im|$V_\mathsf{sum}$|),\nmax(|$V_\mathsf{sum}$|) [MV]')
                ax[spi].set_ylabel('Re($V_\mathsf{sum}$), Im($V_\mathsf{sum}$),\n|$V_\mathsf{sum}$| [MV]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #ax[spi].plot(data_turns, np.max(abs(np.real(data_CavityFeedback_key)), axis=1)/unit_factor, mymarker,            alpha=1.00, label='Real')
                #ax[spi].plot(data_turns, np.max(abs(np.imag(data_CavityFeedback_key)), axis=1)/unit_factor, mymarker,            alpha=1.00, label='Imag')
                #ax[spi].plot(data_turns, np.max(    np.abs( data_CavityFeedback_key) , axis=1)/unit_factor, mymarker, color='k', alpha=1.00, label='Abs')
                ax[spi].fill_between(data_turns,  np.min(np.real(data_CavityFeedback_key), axis=1)/unit_factor, np.max(np.real(data_CavityFeedback_key), axis=1)/unit_factor,            alpha=0.25, label='Real')
                ax[spi].fill_between(data_turns,  np.min(np.imag(data_CavityFeedback_key), axis=1)/unit_factor, np.max(np.imag(data_CavityFeedback_key), axis=1)/unit_factor,            alpha=0.25, label='Imag')
                ax[spi].fill_between(data_turns,  np.min( np.abs(data_CavityFeedback_key), axis=1)/unit_factor, np.max( np.abs(data_CavityFeedback_key), axis=1)/unit_factor, color='k', alpha=0.25, label='Abs')
                #ax[spi].axhline(0., ls=':', color='#888888', alpha=1.0)
                ax[spi].legend()

            elif(key == 'V_corr'):
                unit_factor = 1
                ax[spi].set_ylabel('max|$V_\mathsf{corr}$| [$V_\mathsf{RF0}$]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #ax[spi].plot(data_turns, np.max(abs(data_CavityFeedback_key), axis=1)/unit_factor, mymarker, color='k', alpha=1.00) #, label='Abs')
                ax[spi].fill_between(data_turns, np.min(abs(data_CavityFeedback_key), axis=1)/unit_factor, np.max(abs(data_CavityFeedback_key), axis=1)/unit_factor, color='k', alpha=0.25)
                #ax[spi].axhline(0., ls=':', color='#888888', alpha=1.0)
                #ax[spi].legend()

            elif(key == 'phi_corr'):
                unit_factor = 1
                ax[spi].set_ylabel('max|$\phi_\mathsf{corr}$| [rad]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #ax[spi].plot(data_turns, np.max(data_CavityFeedback_key, axis=1)/unit_factor, mymarker, color='k', alpha=1.00) #, label='Abs')
                ax[spi].fill_between(data_turns, np.min(abs(data_CavityFeedback_key), axis=1)/unit_factor, np.max(abs(data_CavityFeedback_key), axis=1)/unit_factor, color='k', alpha=0.25)
                #ax[spi].axhline(0., ls=':', color='#888888', alpha=1.0)
                #ax[spi].legend()

            spi += 1
            #print('')

        fig.tight_layout() # fig.tight_layout(pad=5.0, h_pad=0.0)
        fig.savefig(f'{outdir}/plot_cavityfeedback_vs_turn.png')
        plt.cla()
        plt.close(fig)

    try:    h5File.close()
    except: pass

# BeamFeedback parameters vs turn ---------------------------------------------

def plot_beamfeedback_vs_turn(outdir):

    with hp.File(f'{outdir}/monitor_full.h5', 'r') as h5File:

        idxfirstturn = 1 # 0 if we want to include turn '-1', or 1 if we want from turn '0'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data_turns = h5File['/turns'][:] # 'int64'
        idxlastturn = np.argmax(data_turns)
        data_turns = h5File['/turns'][idxfirstturn:idxlastturn+1]

        #print(f'data_turns = {data_turns}, shape = {data_turns.shape}')

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # key_list = ['PL/omega_rf',  # RF
        #             'PL/phi_rf',    # RF
        #             'PL/phi_beam',
        #             'PL/dphi',
        #             'PL/domega_rf',
        #             'SL/dphi_rf',   # RF
        #             'RL/drho']
        # Preferred order:
        key_list = ['PL/phi_rf',      # RF
                    'PL/phi_beam',
                    'PL/dphi',
                    'PL/domega_rf',
                    'PL/omega_rf',    # RF
                    'SL/dphi_rf',     # RF
                    'RL/drho']

        fig, ax = plt.subplots(len(key_list), sharex=True)
        fig.set_size_inches(8.0,2.0*len(key_list))

        spi = 0
        for key in key_list:

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            data_BeamFeedback_key = h5File[f'/BeamFeedback/{key}'][idxfirstturn:idxlastturn+1]
            #print(f'{key}, data_BeamFeedback_key = {data_BeamFeedback_key}, shape = {data_BeamFeedback_key.shape}')

            if(key in ['PL/omega_rf', 'PL/phi_rf', 'SL/dphi_rf']):
                nrf = len(data_BeamFeedback_key[0])
                #print(nrf)
            if(key == 'PL/omega_rf'):
                omega_rf_d_0 = data_BeamFeedback_key[1-idxfirstturn,:] # The value of omega_rf at index 1 (in the full array) corresponds to the design omega_rf (the first element is not, it is equal to zero); for all RF ssytems
                #print(f'omega_rf_d_0 = {omega_rf_d_0}')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            mycm = plt.get_cmap('coolwarm')
            if(nrf > 1): color_cycle_nb = [mycm(irf/(nrf-1.)) for irf in range(nrf)]
            else:        color_cycle_nb = [mycm(0.)]

            # for nb in range(nrf):
            #     if(nb == 0 or nb == nrf-1): label_nb = str(nb+1)
            #     else:                       label_nb = None

            if(len(data_turns) == 2-idxfirstturn): mymarker = '.-' # Turns -1 and 0
            else:                                  mymarker = '-'

            ax[spi].set_prop_cycle(cycler('color', color_cycle_nb))
            ax[spi].grid()

            if(spi == len(key_list)-1): ax[spi].set_xlabel('Turns')

            if(  key == 'PL/phi_rf'):
                unit_factor = 1
                ax[spi].set_ylabel('RF phase\n$\phi_{\mathsf{RF}}$ (RF,PL)\n[rad]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                for irf in range(nrf):
                    ax[spi].plot(data_turns, data_BeamFeedback_key[:,irf]/unit_factor, mymarker, alpha=1.00, label='RF'+str(irf+1))
                ax[spi].axhline(0., ls=':', color='#888888', alpha=1.0)
                if(nrf == 2):
                    ax[spi].axhline(np.pi, ls=':', color='#888888', alpha=1.0)
                ax[spi].legend()

            elif(key == 'PL/phi_beam'):
                unit_factor = 1
                ax[spi].set_ylabel('Beam phase\n$\phi_{\mathsf{beam}}$ (PL)\n[rad]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax[spi].plot(data_turns, data_BeamFeedback_key/unit_factor, mymarker, color='k', alpha=1.00,)

            elif(key == 'PL/dphi'):
                unit_factor = 1
                ax[spi].set_ylabel('Phase diff.\n$\Delta \phi = \phi_{\mathsf{beam}} - \phi_s$ (PL)\n[rad]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax[spi].plot(data_turns, data_BeamFeedback_key/unit_factor, mymarker, color='k', alpha=1.00,)

            elif(key == 'PL/domega_rf'): # = \Delta omega_{\delta phi} + (\Delta omega_{\delta f} [for PL] OR \delta omega_{\delta rho} [RL])
                unit_factor = 1
                ax[spi].set_ylabel('RF freq. corr.\n$\Delta \omega_{\mathsf{RF}}$ (PL)\n[1/s]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax[spi].plot(data_turns, data_BeamFeedback_key/unit_factor, mymarker, alpha=1.00, color='k', label='RF'+str(irf+1))

            elif(key == 'PL/omega_rf'):  # New freq after adding domega_rf (scaled for each system according to hRF)
                # Plot omega_rf directly
                unit_factor = 1e9
                ax[spi].set_ylabel('RF freq.\n$\omega_{\mathsf{RF}}$ (RF,PL)\n[$10^9$ 1/s]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                for irf in range(nrf):
                    ax[spi].plot(data_turns, data_BeamFeedback_key[:,irf]/unit_factor, mymarker, alpha=1.00, label='RF'+str(irf+1))
                    ax[spi].axhline(omega_rf_d_0[irf]/unit_factor, ls=':', color='#888888', alpha=1.0)
                ax[spi].legend()
                # # OR plot difference of omega_rf w.r.t. omega_rf_d
                # unit_factor = 1e9
                # ax[spi].set_ylabel('RF freq. diff. w.r.t. design\n$\omega_{\mathsf{RF}} - \omega_{\mathsf{RF},d}$ (RF,PL)\n[$10^9$ 1/s]')
                # ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # for irf in range(nrf):
                #     ax[spi].plot(data_turns, (data_BeamFeedback_key[:,irf]-omega_rf_d_0[irf])/unit_factor, mymarker, alpha=1.00, label='RF'+str(irf+1))
                #     ax[spi].axhline(0., ls=':', color='#888888', alpha=1.0)
                # ax[spi].legend()

            elif(key == 'SL/dphi_rf'):
                unit_factor = 1
                ax[spi].set_ylabel('RF phase corr.\n$\Delta \phi_{\mathsf{RF}}$ (SL)\n[rad]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                for irf in range(nrf):
                    ax[spi].plot(data_turns, data_BeamFeedback_key[:,irf]/unit_factor, mymarker, alpha=1.00, label='RF'+str(irf+1))
                ax[spi].legend()

            elif(key == 'RL/drho'):
                unit_factor = 1
                ax[spi].set_ylabel('Radial diff.\n'+r'$\Delta \rho$ (RL)'+'\n[m]')
                ax[spi].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax[spi].plot(data_turns, np.array(data_BeamFeedback_key)/unit_factor, mymarker, color='k', alpha=1.00)


            #ax[spi].set_xlim(left=0) # Omit turn -1 to better observe the variation of the parameters vs turn, since they are all zero at turn -1 and this needlessly streches the yrange 
            
            spi += 1
            #print('')

        fig.tight_layout() # fig.tight_layout(pad=5.0, h_pad=0.0)
        fig.savefig(f'{outdir}/plot_beamfeedback_vs_turn.png')
        plt.cla()
        plt.close(fig)

    try:    h5File.close()
    except: pass


###############################################################################
