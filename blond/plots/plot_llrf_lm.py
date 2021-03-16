#!/afs/cern.ch/work/l/lmedinam/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:24:59 2020

@author: medinamluis
"""

import sys
import numpy as np
import h5py as hp
import pathlib

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

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

dirbase = 'BLonD_simulations/sps_lhc_losses'
dirin   = f'{dirhome}/{dirbase}/sps_lhc_losses'
dirinp  = f'{dirin}/inp'
dirinpbench  = f'{dirinp}/benchmark'


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


###############################################################################


# def mnpangle(complex_data):
#     # real = complex_data.real
#     # imag = complex_data.imag
#     # angle = np.angle(complex_data)
#     # if isinstance(complex_data, np.ndarray):
#     #     angle[ np.intersect1d( np.where( real < 1e-15 )[0], np.where( imag < 1e-15 )[0] ) ] = 0.
#     # else:
#     #     if real < 1e-6 and imag < 1e-6:
#     #         angle = 0.
#     # return angle
#     return np.angle(complex_data)


# CavityFeedback (MonitorOTFB) parameters all turns ---------------------------

def plot_cavityfeedback_allparams(outdir, monitorotfb, cavityfeedback, profile, t_rev_0, total_induced_voltage_extra=None, profile_bucket_centres=None):

    #print('plot', i)
    #print_object_attributes(monitorotfb, ok=noMPi_or_MPImaster, onlyattr=False)

    #n_set_toplot = sets_of_turns_to_saveplot_flatten.index(i)//4

    # Plot with overlapping consecutive turns:

    ##if i == sets_of_turns_to_saveplot[n_set_toplot][-1]: # At the end of each set to plot
    #if MAC['Nt_trk'] <= monitorotfb.i0:
    # if True:
    turns_to_plot = monitorotfb.turns[:]
    # print(f'turns_to_plot = monitorotfb.turns[:] = {turns_to_plot}, len = {len(turns_to_plot)}') # e.g. turns 0, 2, 4, etc
    # print(f'idx_turns_to_plot = np.arange(len(turns_to_plot)) = {np.arange(len(turns_to_plot))}')
    # print(f'0.5/(len(turns_to_plot)+1)**2 = {0.5/(len(turns_to_plot)+1)**2}')
    # print(f'0.5/(len(turns_to_plot)+1) = {0.5/(len(turns_to_plot)+1)}')

    i = monitorotfb.turns[-1]

    #turn0_seti = sets_of_turns_to_saveplot[n_set_toplot][0]
    #turnf_seti = sets_of_turns_to_saveplot[n_set_toplot][-1]
    #print(f'Plotting (x2) for turns-{turn0_seti}-{turnf_seti}')

    #idx_turns_toplot = list(range(int(4*n_set_toplot), int(4*n_set_toplot)+4))
    #print(f'idx_turns_toplot = {idx_turns_toplot}') # and index 0 to 4 for the turn no.. in the flatten turn list
    #print(f'sets_of_turns_to_saveplot_flatten[idx_turns_toplot] = {sets_of_turns_to_saveplot_flatten[idx_turns_toplot]}') # the actual turn no.

    # if True: #False:
    # PLOT: cavityfeedback_allparams

    fname = f'{outdir}/plot_cavityfeedback_allparams.pdf'
    #fname = f'{outdir}/plot_cavityfeedback_allparams_turns-{turn0_seti}-{turnf_seti}.pdf'
    with PdfPages(fname) as pdf:

        # If_gen_coarse in OTFB_2 is always the longest array:
        tarraymax = monitorotfb.OTFB_2_t_coarse_long[-1,-1]
        #print(f'tarraymax = {tarraymax}')

        # twindow = 1.5e-6 #  1.5 us to see the full batch (48b)
        # twindow = 2.5e-6 #  2.5 us to see the full batch (72b)
        #twindow = 3.0e-6 #  More when you have nbm
        #twindow = 10e-9 #30e-9 # 30 ns to see the first 2 bunches, 55 ns for 3 ns, 80 ns for 4 bunches
        #twindow = 650e-9 # 500. ns to see the batch reaching steady state
        #print( ( 0/unit_t, twindow/unit_t ) )
        #print( ( (tarraymax - twindow)/unit_t, tarraymax/unit_t ) )

        twindow = profile.bin_centers[ monitorotfb.indices_beamH_fine[-1] ] \
            + 1.00*(profile.bin_centers[ monitorotfb.indices_beamH_fine[-1] ] - profile.bin_centers[ monitorotfb.indices_beamH_fine[0] ])

        if monitorotfb.n_samples_beam_coarse > 1 and int(monitorotfb.n_samples_beam_coarse * (monitorotfb.indices_beamF_coarse[1]-monitorotfb.indices_beamF_coarse[0])) ==  monitorotfb.n_coarse_1:
            # The crazy case of a full machine
            twindow = 0.5*profile.cut_right

        #plotfuncs = (np.real, np.imag, np.abs, np.angle)

        for param in monitorotfb.list_params:

            ot = '1' if '1' in param else '2'
            if   'fine'     in param: fc = 'fine'
            elif 'coarseFF' in param: fc = 'coarseFF'
            else:                     fc = 'coarse'

            nrows = 4 #2
            ncols = 4 #6

            fig = plt.figure(constrained_layout=True) #,sharex=True) #,sharey='row')
            fig.set_size_inches(ncols*5.00, nrows*2.50)

            gs = fig.add_gridspec(nrows, ncols)
            ax0a = fig.add_subplot(gs[0,0])
            ax0b = fig.add_subplot(gs[0,1])
            ax1a = fig.add_subplot(gs[1,0])
            ax1b = fig.add_subplot(gs[1,1])
            ax2a = fig.add_subplot(gs[2,0])
            ax2b = fig.add_subplot(gs[2,1])
            ax3a = fig.add_subplot(gs[3,0])
            ax3b = fig.add_subplot(gs[3,1])
            ax4 = fig.add_subplot(gs[:,2:], projection='polar')

            ax0a2 = ax0a.twinx()
            ax0b2 = ax0b.twinx()
            ax1a2 = ax1a.twinx()
            ax1b2 = ax1b.twinx()
            ax2a2 = ax2a.twinx()
            ax2b2 = ax2b.twinx()
            ax3a2 = ax3a.twinx()
            ax3b2 = ax3b.twinx()

            # if not True:
            #     mycm = plt.get_cmap('coolwarm')
            #     #color_cycle_nb = [mycm(ii/(len(turns_to_plot)-1.)) for ii in turns_to_plot]
            #     color_cycle_nb = [mycm(ii/(len(idx_turns_toplot)-1.)) for ii in idx_turns_toplot]
            #     color_cycle_nb[-1] = 'k' # Last (current) turn always black
            # else:
            #    #turns_to_plot = np.array([0, i-1]) # Only first and last (current) turn
            turns_to_plot = monitorotfb.turns[:] # e.g. turns 0, 2, 4, etc
            idx_turns_to_plot = np.arange(len(turns_to_plot)) # would correspond to 0, 1, 2, etc.
            if len(idx_turns_to_plot) == 1:
                idx_turns_to_plot = np.array([0])
                color_cycle_nb = ['k'] # Last (current) turn always black
            else:
                mycm = plt.get_cmap('coolwarm')
                color_cycle_nb = [mycm(ii/(len(turns_to_plot)-1.)) for ii in idx_turns_to_plot]
                color_cycle_nb[-1] = 'k' # Last (current) turn always black

            if 'coarse' in param and twindow <= 551e-9:
                mysty = '.-'
                myms  = 5
            else:
                mysty = '-'
                myms = None

            unit_t = 1e-6;           symb_t = '$\\mu$s'
            unit_Q = 1e-9;           symb_Q = 'nC'
            unit_I = 1;              symb_I = 'A'
            unit_V = 1e6;            symb_V = 'MV'
            unit_P = 1e6;            symb_P = 'MW'
            unit_angle = np.pi/180.; symb_angle = 'deg' # rad to deg

            if   'Q' in param: # Q_gen, Q_beam
                unit_param = unit_Q
                symb_param = symb_Q
            elif 'I' in param: # I_gen_coarse, I_beam
                unit_param = unit_I
                symb_param = symb_I
            elif 'V' in param: # Vind, Vtot, Vgen
                unit_param = unit_V
                symb_param = symb_V
            elif 'P' in param: # Pgen, Pgen2
                unit_param = unit_P
                symb_param = symb_P

            for axi in [ax0a, ax0b, ax1a, ax1b, ax2a, ax2b, ax3a, ax3b, ax4]:
                axi.set_prop_cycle(cycler('color', color_cycle_nb))

            #

            for ii in idx_turns_to_plot:

                turn_ii = turns_to_plot[ii]

                color_ii = color_cycle_nb[ii] # To use the color map

                param_ii = getattr(monitorotfb, param)[ii]
                bucket_centres_shift = getattr(cavityfeedback, f'OTFB_{ot}').TWC.tau if 'long' in monitorotfb.time_arrays_dict[param] else 0.0
                time_ii  = getattr(monitorotfb, monitorotfb.time_arrays_dict[param])[ii] - bucket_centres_shift
               #print(param, param_ii.shape, monitorotfb.time_arrays_dict[param], time_ii.shape)

                if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                    param_ii_max_beam   = getattr(monitorotfb, f'{param}_max_beam')[ii]
                    t_ii_max_beam  = time_ii[ getattr(monitorotfb,  f'index_max_beam_{param}')[ii] ] # Doesnt need the correction, the index for max for Q_gen and P_gen (long params) was already saved in the proper array
                else:
                    param_ii_max_beam = np.NaN
                    t_ii_max_beam     = np.NaN
                if monitorotfb.track_ave:
                    param_ii_ave_beamM  = getattr(monitorotfb, f'{param}_ave_beamM' )[ii]
                    param_ii_ave_beamH  = getattr(monitorotfb, f'{param}_ave_beamH' )[ii] # np.average(param_ii[ getattr(monitorotfb, f'indices_beamH_{fc}') ])
                    param_ii_ave_nobeam = getattr(monitorotfb, f'{param}_ave_nobeam')[ii] # np.average(param_ii[ getattr(monitorotfb, f'indices_nobeam_{fc}')])
                    t_ii_ave_beamM  = time_ii[ int(np.average( getattr(monitorotfb,  f'indices_beamM_{fc}') + (getattr(monitorotfb, f'n_mov_av_{fc}_{ot}') if 'long' in monitorotfb.time_arrays_dict[param] else 0) ))  ]
                    t_ii_ave_beamH  = time_ii[ int(np.average( getattr(monitorotfb,  f'indices_beamH_{fc}') + (getattr(monitorotfb, f'n_mov_av_{fc}_{ot}') if 'long' in monitorotfb.time_arrays_dict[param] else 0) ))  ]
                   #t_ii_ave_nobeam = time_ii[ int(np.average( getattr(monitorotfb, f'indices_nobeam_{fc}'))) ]
                    if not np.isnan(param_ii_ave_nobeam):
                        t_ii_ave_nobeam  = (tarraymax - 0.95*twindow)

                if monitorotfb.profile is not None:
                    # Profile stored at the corresponding turn in monitorotfb
                    profile_bin_centers_ii      = monitorotfb.profile_bin_centers[ii]
                    profile_n_macroparticles_ii = monitorotfb.profile_n_macroparticles[ii]
                else:
                    # Profile at the current turn only: should not be used with PL
                    if ii == idx_turns_to_plot[-1]:
                        profile_bin_centers_ii      = profile.bin_centers[:]
                        profile_n_macroparticles_ii = profile.n_macroparticles[:]
                    else:
                        profile_bin_centers_ii      = None
                        profile_n_macroparticles_ii = None


                # if   len(param_ii) == profile.n_slices:
                #     # Vind_tot_sum_fine falls here too
                #     tarray = profile.bin_centers
                # else:
                #     if   '1' in param: ot = '1'
                #     elif '2' in param: ot = '2'
                #     if len(param_ii) <= profile.n_slices:
                #         tarray = profile.cut_left + getattr(cavityfeedback, f'OTFB_{ot}').rf_centers
                #         if len(param_ii) > getattr(monitorotfb, f'n_coarse_{ot}'):
                #             # Extended coarse array
                #             tarray_extra = tarray[-1] + np.arange(1, getattr(monitorotfb, f'n_mov_av_coarse_{ot}')+1) * getattr(cavityfeedback, f'OTFB_{ot}').T_s_coarse # T_s_coarse = maxbktl = 4.990118014819246e-09, 4.990118014819246e-09
                #             tarray = np.concatenate( (tarray, tarray_extra) )
                #     else:
                #         tarray = np.concatenate( (profile.bin_centers, profile.bin_centers[-1] + np.arange(1, getattr(monitorotfb, f"OTFB_{ot}_I_gen_fine").shape[1] - profile.n_slices + 1) * profile.bin_size) )
                # print(f'tarray = {tarray}, shape = {tarray.shape}')

                # SPECIAL CASE: totalinducedvoltage + Vind_tot_fine, so it is at the background
                if total_induced_voltage_extra is not None and param in ['OTFB_sum_Vind_tot_fine'] and ii == idx_turns_to_plot[-1]:
                #if param in ['OTFB_sum_Vind_tot_fine', 'tracker_Vrf_fine'] and ii == turns_to_plot[-1]:
                #if param in ['OTFB_sum_Vind_tot_fine', 'tracker_Vrf_fine'] and ii == len(idx_turns_toplot)-1: # Only for last (current) turn
                    # Cartesian
                    for axiab in [[ax0a, ax1a, ax2a, ax3a], [ax0b, ax1b, ax2b, ax3b]]:
                        for ifunc in range(4):
                            unitparam_i = unit_param if ifunc < 3 else unit_angle
                            #val = np.array(list(map(func, param_ii + total_induced_voltage_extra)))/unitparam_i
                            if   ifunc == 0: val = np.real( param_ii + total_induced_voltage_extra)/unitparam_i
                            elif ifunc == 1: val = np.imag( param_ii + total_induced_voltage_extra)/unitparam_i
                            elif ifunc == 2: val = np.abs(  param_ii + total_induced_voltage_extra)/unitparam_i
                            elif ifunc == 3: val = np.angle(param_ii + total_induced_voltage_extra)/unitparam_i
                            #if ifunc == 3: val[ np.where(np.abs(val) > 0.999*np.pi)[0] ] = 0. # to make the +pi/-pi jumps go to zero
                            axiab[ifunc].plot(time_ii/unit_t, val, mysty, markersize=myms, color='green', alpha=0.5)
                    # Polar: no need to add unit_angle to 1st array, as polar plots already are shown in deg
                    ax4.plot(np.angle(param_ii), np.abs(param_ii + total_induced_voltage_extra)/unit_param, color='green', alpha=0.5)

                # Show the bunch profile during the testing to compare positions w.r.t. current/coltage/etc signals
                # It also shows window spanned by the beam segment
                if profile_bin_centers_ii is not None and profile_bucket_centres is not None: #True: #twindow < 100e-9: # Show profiles when window include only 4 bunches:
                    mymark = '|' if twindow < 100e-9 else None
                    for axiab2 in [[ax0a2, ax1a2, ax2a2, ax3a2], [ax0b2, ax1b2, ax2b2, ax3b2]]:
                        for ifunc in range(4):
                            # profile_n_macroparticles_ii_toplot = profile_n_macroparticles_ii - np.min(profile_n_macroparticles_ii) # Rebaseline to zero
                            # profile_n_macroparticles_ii_toplot /= MAC['Np_ave_0'] # Normalize max to ave Np_ave_0
                            # #
                            # #axiab2[ifunc].plot(profile_bin_centers_ii/unit_t,                      profile_n_macroparticles_ii_toplot,                                     color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1))
                            # #axiab2[ifunc].plot(profile_bin_centers_ii[::profilepattern.Ns]/unit_t, profile_n_macroparticles_ii_toplot[::profilepattern.Ns], marker=mymark, color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1))
                            # #
                            # #axiab2[ifunc].plot(profile_bin_centers_ii/unit_t, profile_n_macroparticles_ii_toplot, color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1))
                            # #
                            # #axiab2[ifunc].plot(profile_bin_centers_ii[profile_bucket_centres]/unit_t, profile_n_macroparticles_ii_toplot[profile_bucket_centres], '.', color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1))
                            #
                            # For a parameter in a long array, either shit the batch, or the xlim
                            # bucket_centres_shift = getattr(cavityfeedback, f'OTFB_{ot}').TWC.tau if 'long' in monitorotfb.time_arrays_dict[param] else 0.0
                            # # Window:
                            # #axiab2[ifunc].vlines(profile_bin_centers_ii[profile_bucket_centres]/unit_t, ymin=0, ymax=1, color=color_ii, lw=1.0, alpha=max(1./(turns_to_plot+1), 1./(monitorotfb.i0+1))) #, transform=trans)
                            # axiab2[ifunc].axvspan( (profile_bin_centers_ii[profile_bucket_centres[0]]  + bucket_centres_shift)/unit_t, (profile_bin_centers_ii[profile_bucket_centres[-1]] + bucket_centres_shift)/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1)**2, zorder=0) #alpha=max(0.5/(len(turns_to_plot)+1), 0.5/(monitorotfb.i0+1)), zorder=0)
                            axiab2[ifunc].axvspan( profile_bin_centers_ii[profile_bucket_centres[0]] /unit_t, profile_bin_centers_ii[profile_bucket_centres[-1]]/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1)**2 if monitorotfb.profile is not None else 0.125, zorder=0) #alpha=max(0.5/(len(turns_to_plot)+1), 0.5/(monitorotfb.i0+1)), zorder=0)
                            # # Edges:
                            # axiab2[ifunc].axvline( (profile_bin_centers_ii[profile_bucket_centres[0]]  + bucket_centres_shift)/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1), zorder=1)
                            # axiab2[ifunc].axvline( (profile_bin_centers_ii[profile_bucket_centres[-1]] + bucket_centres_shift)/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1), zorder=1)
                            axiab2[ifunc].axvline( profile_bin_centers_ii[profile_bucket_centres[0]] /unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1) if monitorotfb.profile is not None else 0.125, zorder=1)
                            axiab2[ifunc].axvline( profile_bin_centers_ii[profile_bucket_centres[-1]]/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1) if monitorotfb.profile is not None else 0.125, zorder=1)


                # ALL
                # Cartesian
                for axiab in [[ax0a, ax1a, ax2a, ax3a], [ax0b, ax1b, ax2b, ax3b]]:
                    for ifunc in range(4):
                        unitparam_i = unit_param if ifunc < 3 else unit_angle
                        #val = np.array(list(map(func, param_ii)))/unitparam_i
                        if   ifunc == 0: val = np.real( param_ii)/unitparam_i
                        elif ifunc == 1: val = np.imag( param_ii)/unitparam_i
                        elif ifunc == 2: val = np.abs(  param_ii)/unitparam_i
                        elif ifunc == 3: val = np.angle(param_ii)/unitparam_i
                        #if ifunc == 3: val[ np.where(np.abs(val) > 0.999*np.pi)[0] ] = 0. # to make the +pi/-pi jumps go to zero
                        labelturn = None
                        if ii == idx_turns_to_plot[-1] and ifunc == 3: labelturn = f'Turn {turns_to_plot[ii]}'
                        axiab[ifunc].plot(time_ii/unit_t, val, mysty, markersize=myms, color=color_ii, label=labelturn)
                    #
                    # Markers of the average of beam and no-beam segments
                    if ii == idx_turns_to_plot[0] or ii == idx_turns_to_plot[-1]:
                        ms_ii = 10.0 if ii == turns_to_plot[0] else 7.5 # 5.0
                        for ifunc in range(4):
                            unitparam_i = unit_param if ifunc < 3 else unit_angle
                            # To map this floats we need to make the iterable (i.e. a list), then just simply take the single 0-th element
                            if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                                #val_max   = list(map(func, [param_ii_max_beam]  ))[0]/unitparam_i
                                if   ifunc == 0: val_max  = np.real( param_ii_max_beam)/unitparam_i
                                elif ifunc == 1: val_max  = np.imag( param_ii_max_beam)/unitparam_i
                                elif ifunc == 2: val_max  = np.abs(  param_ii_max_beam)/unitparam_i
                                elif ifunc == 3: val_max  = np.angle(param_ii_max_beam)/unitparam_i
                            if monitorotfb.track_ave:
                                #val_beamM   = list(map(func, [param_ii_ave_beamM]  ))[0]/unitparam_i
                                if   ifunc == 0: val_beamM  = np.real( param_ii_ave_beamM)/unitparam_i
                                elif ifunc == 1: val_beamM  = np.imag( param_ii_ave_beamM)/unitparam_i
                                elif ifunc == 2: val_beamM  = np.abs(  param_ii_ave_beamM)/unitparam_i
                                elif ifunc == 3: val_beamM  = np.angle(param_ii_ave_beamM)/unitparam_i
                                #val_beamH   = list(map(func, [param_ii_ave_beamH]  ))[0]/unitparam_i
                                if   ifunc == 0: val_beamH  = np.real( param_ii_ave_beamH)/unitparam_i
                                elif ifunc == 1: val_beamH  = np.imag( param_ii_ave_beamH)/unitparam_i
                                elif ifunc == 2: val_beamH  = np.abs(  param_ii_ave_beamH)/unitparam_i
                                elif ifunc == 3: val_beamH  = np.angle(param_ii_ave_beamH)/unitparam_i
                                #if ifunc == 3 and np.abs(val_beamH) > 0.999*np.pi: val_beamH = 0. # to make the +pi/-pi jumps go to zero
                                if not np.isnan(param_ii_ave_nobeam):
                                    #val_nobeam = list(map(func, [param_ii_ave_nobeam]))[0]/unitparam_i
                                    if   ifunc == 0: val_nobeam = np.real( param_ii_ave_nobeam)/unitparam_i
                                    elif ifunc == 1: val_nobeam = np.imag( param_ii_ave_nobeam)/unitparam_i
                                    elif ifunc == 2: val_nobeam = np.abs(  param_ii_ave_nobeam)/unitparam_i
                                    elif ifunc == 3: val_nobeam = np.angle(param_ii_ave_nobeam)/unitparam_i
                                    #if ifunc == 3 and np.abs(val_nobeam) > 0.999*np.pi: val_nobeam = 0. # to make the +pi/-pi jumps go to zero
                            if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                                axiab[ifunc].axhline(val_max,  color=color_ii, ls='--', alpha=0.75)
                                axiab[ifunc].plot( t_ii_max_beam /unit_t, val_max,  '^', markersize=ms_ii, markerfacecolor=color_ii, markeredgecolor='white')
                            if monitorotfb.track_ave:
                                axiab[ifunc].axhline(val_beamM,  color=color_ii, ls=':',  alpha=0.50)
                                axiab[ifunc].axhline(val_beamH,  color=color_ii, ls='--', alpha=0.25)
                                if not np.isnan(param_ii_ave_nobeam):
                                    axiab[ifunc].axhline(val_nobeam, color=color_ii, ls=':',  alpha=0.50)
                                axiab[ifunc].plot( t_ii_ave_beamM /unit_t, val_beamM,  'D', markersize=ms_ii, markerfacecolor=color_ii, markeredgecolor='white')
                                axiab[ifunc].plot( t_ii_ave_beamH /unit_t, val_beamH,  's', markersize=ms_ii, markerfacecolor=color_ii, markeredgecolor='white')
                                if not np.isnan(param_ii_ave_nobeam):
                                    axiab[ifunc].plot( t_ii_ave_nobeam/unit_t, val_nobeam, 'o', markersize=ms_ii, markerfacecolor=color_ii, markeredgecolor='white')

                # Polar: no need to add unit_angle to 1st array, as polar plots already are shown in deg
                ms_ii = 7.5 #5.0
                labelmax    = None
                labelbeamM  = None
                labelbeamH  = None
                labelnobeam = None
                if ii == idx_turns_to_plot[0]:
                    ms_ii = 10.0
                    if i == 0:
                        if monitorotfb.track_ave:
                            labelmax    = 'Beam (max)'
                        if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                            labelbeamM  = 'Beam (mid)'
                            labelbeamH  = 'Beam (ave)'
                            labelnobeam = 'No beam'
                elif ii == idx_turns_to_plot[-1]:
                    if i != 0:
                        if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                            labelmax    = 'Beam (max)'
                        if monitorotfb.track_ave:
                            labelbeamM  = 'Beam (mid)'
                            labelbeamH  = 'Beam (ave)'
                            labelnobeam = 'No beam'

                ax4.plot(np.angle(param_ii),            np.abs(param_ii)/unit_param,            color=color_ii, alpha=0.5) #, label=labelturn)
                ax4.plot(np.angle(param_ii_max_beam),   np.abs(param_ii_max_beam) /unit_param,  '^', markerfacecolor=color_ii, markeredgecolor='white', markersize=ms_ii, label=labelmax)
                ax4.plot(np.angle(param_ii_ave_beamM),  np.abs(param_ii_ave_beamM)/unit_param,  'D', markerfacecolor=color_ii, markeredgecolor='white', markersize=ms_ii, label=labelbeamM)
                ax4.plot(np.angle(param_ii_ave_beamH),  np.abs(param_ii_ave_beamH)/unit_param,  's', markerfacecolor=color_ii, markeredgecolor='white', markersize=ms_ii, label=labelbeamH)
                if not np.isnan(param_ii_ave_nobeam):
                    ax4.plot(np.angle(param_ii_ave_nobeam), np.abs(param_ii_ave_nobeam)/unit_param, 'o', markerfacecolor=color_ii, markeredgecolor='white', markersize=ms_ii, label=labelnobeam)

                # Vertical line at the end of the turn: t_rev
                for axi in [ax0b, ax1b, ax2b, ax3b]:
                    axi.axvline(t_rev_0/unit_t, ymax=10.5, color='grey', ls='-.', alpha=0.5)

            axia_ylim_list = []
            for axia in [ax0a, ax1a, ax2a, ax3a]:
                axia.set_xlim( (profile.cut_left-bucket_centres_shift)/unit_t, (profile.cut_left+twindow-bucket_centres_shift)/unit_t )
                axia_ylim_list.append(axia.get_ylim())
                axia.spines['right'].set_visible(False)
                axia.yaxis.tick_left()
                axia.tick_params(labelright=False)
                axia.tick_params(right=False)
                axia.yaxis.set_major_formatter(FormatStrFormatter('%5.4f'))
            for ib, axib in enumerate([ax0b, ax1b, ax2b, ax3b]):
                if twindow >= monitorotfb.n_mov_av_coarse_2 * cavityfeedback.OTFB_2.T_s_coarse: # i.e. > 620 ns = 0.62 us
                    axib.set_xlim( (tarraymax - twindow - bucket_centres_shift)/unit_t, (tarraymax-bucket_centres_shift)/unit_t )
                else:
                    # In this case take t_rf as the margin (we will lose the extra part of I_gen_coarse, but we wouldnt be able to the end of the other parameters as the window is very small (larger than n_mov_av)
                    axib.set_xlim( (t_rev_0 - twindow - bucket_centres_shift)/unit_t, (t_rev_0 - bucket_centres_shift)/unit_t )
                axib.set_ylim( axia_ylim_list[ib] )
                axib.spines['left'].set_visible(False)
                axib.tick_params(labelleft=False)
                axib.tick_params(left=False)
                axib.yaxis.set_major_formatter(FormatStrFormatter('%5.4f'))

            axia2_ylim_list = []
            for axia2 in [ax0a2, ax1a2, ax2a2, ax3a2]:
                axia2_ylim_list.append(axia2.get_ylim())
                axia2.spines['right'].set_visible(False)
                axia2.tick_params(labelright=False)
                axia2.tick_params(right=False)
            for ib, axib2 in enumerate([ax0b2, ax1b2, ax2b2, ax3b2]):
                axib2.set_ylim( axia2_ylim_list[ib] )
                axib2.spines['left'].set_visible(False)
                axib2.spines['right'].set_visible(False)
                #axib2.yaxis.tick_right()
                axib2.tick_params(labelleft=False)
                axib2.tick_params(labelright=False)
                axib2.tick_params(left=False)
                axib2.tick_params(right=False)

            ax4.set_ylim(0.0, 1.1*ax4.get_ylim()[-1])

            ax3a.set_xlabel(f'Time [{symb_t}]')

            ax0a.set_ylabel(f'Real [{symb_param}]')
            ax1a.set_ylabel(f'Imaginary [{symb_param}]')
            ax2a.set_ylabel(f'Magnitude [{symb_param}]')
            ax3a.set_ylabel(f'Angle [{symb_angle}]')

            ax3b.legend(loc=2, framealpha=1.0)
            ax4.legend(loc=3, framealpha=1.0)

            figtitle = f'{param} '
            if 'sum' not in param and 'tracker' not in param:
                if   '1' in param: ot = '1'
                elif '2' in param: ot = '2'
                cavtype_ot = getattr(getattr(cavityfeedback, f'OTFB_{ot}'), 'cavtype')
                figtitle = figtitle + f'(1x{cavtype_ot[2:]})' if 'P' in param else figtitle + f'({cavtype_ot})'

            fig.suptitle(figtitle)
            #fig.tight_layout()
            #fname = f'{outdir}/plot_cavityfeedback_{param}.png'
            #print(f'Saving {fname} ...')
            #fig.savefig(fname)
            pdf.savefig()
            #plt.cla()
            plt.close() #fig)

    print(f'Saving {fname} ...')

    #gc.collect()


# CavityFeedback (MonitorOTFB) parameters all turns (continuous) --------------

def plot_cavityfeedback_allparams_continuous(outdir, monitorotfb, cavityfeedback, profile, t_rev_0, total_induced_voltage_extra=None, profile_bucket_centres=None):

    # same condition for the 2nd plot...
    fname = f'{outdir}/plot_cavityfeedback_allparams_continuous.pdf'
    #fname = f'{outdir}/plot_cavityfeedback_allparams_continuous_turns-{turn0_seti}-{turnf_seti}.pdf'
    with PdfPages(fname) as pdf:

        for param in monitorotfb.list_params:

            ot = '1' if '1' in param else '2'
            if   'fine'     in param: fc = 'fine'
            elif 'coarseFF' in param: fc = 'coarseFF'
            else:                     fc = 'coarse'

            nrows = 4
            ncols = 1

            fig, ax = plt.subplots(nrows, ncols, sharex=True)
            fig.set_size_inches(ncols*20.00, nrows*2.50)

            ax0a = ax[0]
            ax1a = ax[1]
            ax2a = ax[2]
            ax3a = ax[3]

            ax0a2 = ax[0].twinx()
            ax1a2 = ax[1].twinx()
            ax2a2 = ax[2].twinx()
            ax3a2 = ax[3].twinx()

            turns_to_plot = monitorotfb.turns[:] # e.g. turns 0, 2, 4, etc
            idx_turns_to_plot = np.arange(len(turns_to_plot)) # would correspond to 0, 1, 2, etc.
            if len(turns_to_plot) == 1:
                idx_turns_to_plot = np.array([0])
                color_cycle_nb = ['k'] # Last (current) turn always black
            else:
                mycm = plt.get_cmap('coolwarm')
                color_cycle_nb = [mycm(ii/(len(turns_to_plot)-1.)) for ii in idx_turns_to_plot]
                color_cycle_nb[-1] = 'k' # Last (current) turn always black

            if 'coarse' in param: #and twindow <= 551e-9:
                mysty = '.-'
                myms  = 5
            else:
                mysty = '-'
                myms = None

            unit_t = 1e-6;           symb_t = '$\\mu$s'
            unit_Q = 1e-9;           symb_Q = 'nC'
            unit_I = 1;              symb_I = 'A'
            unit_V = 1e6;            symb_V = 'MV'
            unit_P = 1e6;            symb_P = 'MW'
            unit_angle = np.pi/180.; symb_angle = 'deg' # rad to deg

            if   'Q' in param: # Q_gen, Q_beam
                unit_param = unit_Q
                symb_param = symb_Q
            elif 'I' in param: # I_gen_coarse, I_beam
                unit_param = unit_I
                symb_param = symb_I
            elif 'V' in param: # Vind, Vtot, Vgen
                unit_param = unit_V
                symb_param = symb_V
            elif 'P' in param: # Pgen
                unit_param = unit_P
                symb_param = symb_P

            for ii in idx_turns_to_plot:

                turn_ii = turns_to_plot[ii]

                color_ii = color_cycle_nb[ii] # To use the color map

                param_ii = getattr(monitorotfb, param)[ii]
                bucket_centres_shift = getattr(cavityfeedback, f'OTFB_{ot}').TWC.tau if 'long' in monitorotfb.time_arrays_dict[param] else 0.0
                time_ii  = getattr(monitorotfb, monitorotfb.time_arrays_dict[param])[ii] - bucket_centres_shift
               #print(param, param_ii.shape, monitorotfb.time_arrays_dict[param], time_ii.shape)

                if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                    param_ii_max_beam   = getattr(monitorotfb, f'{param}_max_beam')[ii]
                    t_ii_max_beam  = turn_ii*t_rev_0/unit_t + time_ii[ getattr(monitorotfb, f'index_max_beam_{param}')[ii] ] # Doesnt need the correction, the index for max for Q_gen and P_gen (long params) was already saved in the proper array
                else:
                    param_ii_max_beam = np.NaN
                    t_ii_max_beam     = np.NaN
                if monitorotfb.track_ave:
                    param_ii_ave_beamM  = getattr(monitorotfb, f'{param}_ave_beamM' )[ii]
                    param_ii_ave_beamH  = getattr(monitorotfb, f'{param}_ave_beamH' )[ii] # np.average(param_ii[ getattr(monitorotfb, f'indices_beamH_{fc}') ])
                    param_ii_ave_nobeam = getattr(monitorotfb, f'{param}_ave_nobeam')[ii] # np.average(param_ii[ getattr(monitorotfb, f'indices_nobeam_{fc}')])
                    t_ii_ave_beamM  = turn_ii*t_rev_0/unit_t + time_ii[ int(np.average( getattr(monitorotfb, f'indices_beamM_{fc}')  + (getattr(monitorotfb, f'n_mov_av_{fc}_{ot}') if 'long' in monitorotfb.time_arrays_dict[param] else 0) )) ]
                    t_ii_ave_beamH  = turn_ii*t_rev_0/unit_t + time_ii[ int(np.average( getattr(monitorotfb, f'indices_beamH_{fc}')  + (getattr(monitorotfb, f'n_mov_av_{fc}_{ot}') if 'long' in monitorotfb.time_arrays_dict[param] else 0) )) ]
                    if not np.isnan(param_ii_ave_nobeam):
                        t_ii_ave_nobeam = turn_ii*t_rev_0/unit_t + time_ii[ int(np.average( getattr(monitorotfb, f'indices_nobeam_{fc}') + (getattr(monitorotfb, f'n_mov_av_{fc}_{ot}') if 'long' in monitorotfb.time_arrays_dict[param] else 0) )) ]

                if monitorotfb.profile is not None:
                    # Profile stored at the corresponding turn in monitorotfb
                    profile_bin_centers_ii      = monitorotfb.profile_bin_centers[ii]
                    profile_n_macroparticles_ii = monitorotfb.profile_n_macroparticles[ii]
                else:
                    # Profile at the current turn only: should not be used with PL
                    if ii == idx_turns_to_plot[-1]:
                        profile_bin_centers_ii      = profile.bin_centers[:]
                        profile_n_macroparticles_ii = profile.n_macroparticles[:]
                    else:
                        profile_bin_centers_ii      = None
                        profile_n_macroparticles_ii = None

                # SPECIAL CASE: totalinducedvoltage + Vind_tot_fine, so it is at the background
                if total_induced_voltage_extra is not None and param in ['OTFB_sum_Vind_tot_fine'] and ii == idx_turns_to_plot[-1]: # Only for last (current) turn
               #if param in ['OTFB_sum_Vind_tot_fine', 'tracker_Vrf_fine'] and ii == turns_to_plot[-1]: # Only for last (current) turn
                    # Cartesian
                    for axiab in [[ax0a, ax1a, ax2a, ax3a]]: #, [ax0b, ax1b, ax2b, ax3b]]:
                        for ifunc in range(4):
                            unitparam_i = unit_param if ifunc < 3 else unit_angle
                            #val = np.array(list(map(func, param_ii + total_induced_voltage_extra)))/unitparam_i
                            if   ifunc == 0: val = np.real( param_ii + total_induced_voltage_extra)/unitparam_i
                            elif ifunc == 1: val = np.imag( param_ii + total_induced_voltage_extra)/unitparam_i
                            elif ifunc == 2: val = np.abs(  param_ii + total_induced_voltage_extra)/unitparam_i
                            elif ifunc == 3: val = np.angle(param_ii + total_induced_voltage_extra)/unitparam_i
                            #if ifunc == 3: val[ np.where(np.abs(val) > 0.999*np.pi)[0] ] = 0. # to make the +pi/-pi jumps go to zero
                            # Shift the corresponding time array by one t_rev
                            axiab[ifunc].plot(turn_ii*t_rev_0/unit_t + time_ii/unit_t, val, color='green', alpha=0.5)

                # Show the bunch profile during the testing to compare positions w.r.t. current/coltage/etc signals
                # It also shows window spanned by the beam segment
                if profile_bin_centers_ii is not None and profile_bucket_centres is not None: #True: #twindow < 100e-9: # Show profiles when window include only 4 bunches:
                    mymark = None #'|' if twindow < 100e-9 else None
                    for axia2 in [[ax0a2, ax1a2, ax2a2, ax3a2]]:
                        for ifunc in range(4):
                            # profile_n_macroparticles_ii_toplot = profile_n_macroparticles_ii - np.min(profile_n_macroparticles_ii) # Rebaseline to zero
                            # profile_n_macroparticles_ii_toplot /= MAC['Np_ave_0'] # Normalize max to ave Np_ave_0
                            # #
                            # #axia2[ifunc].plot(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[int(0.5*profilepattern.Ns)::profilepattern.Ns]/unit_t, profile_n_macroparticles_ii_toplot[int(0.5*profilepattern.Ns)::profilepattern.Ns],                color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1)) # Just plot the middle point of the profile of each bucket for lightness
                            # #axia2[ifunc].plot(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[::profilepattern.Ns]/unit_t,                           profile_n_macroparticles_ii_toplot[::profilepattern.Ns],                           marker=mymark, color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1))
                            # #
                            # #axia2[ifunc].plot(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres]/unit_t, profile_n_macroparticles_ii_toplot[profile_bucket_centres], color=color_ii, lw=1.0, alpha=1./(turns_to_plot+1))
                            # #
                            # #axia2[ifunc].plot(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres]/unit_t, profile_n_macroparticles_ii_toplot[profile_bucket_centres], '.', color=color_ii, lw=1.0)
                            #
                            #
                            # For a parameter in a longa array, either shit the batch, or the xlim
                            # bucket_centres_shift = getattr(cavityfeedback, f'OTFB_{ot}').TWC.tau if 'long' in monitorotfb.time_arrays_dict[param] else 0.0
                            # # Window:
                            # #axia2[ifunc].vlines(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres]/unit_t, ymin=0, ymax=1, color=color_ii, lw=1.0, alpha=max(1./(len(turns_to_plot)+1), 1./(monitorotfb.i0+1)))
                            # axia2[ifunc].axvspan(turn_ii*t_rev_0/unit_t + (profile_bin_centers_ii[profile_bucket_centres[0]]  + bucket_centres_shift)/unit_t, ii*t_rev_0/unit_t + (profile_bin_centers_ii[profile_bucket_centres[-1]] + bucket_centres_shift)/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1)**2, zorder=0) # max(1.0/(len(turns_to_plot)+1), 1.0/(monitorotfb.i0+1)), zorder=0)
                            axia2[ifunc].axvspan(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres[0]] /unit_t, ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres[-1]]/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1)**2 if monitorotfb.profile is not None else 0.125, zorder=0) # max(1.0/(len(turns_to_plot)+1), 1.0/(monitorotfb.i0+1)), zorder=0)
                            # #
                            # # Edges
                            # axia2[ifunc].axvline(turn_ii*t_rev_0/unit_t + (profile_bin_centers_ii[profile_bucket_centres[0]]  + bucket_centres_shift)/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1), zorder=1) # max(1.0/(len(turns_to_plot)+1), 1.0/(monitorotfb.i0+1)), zorder=0)
                            # axia2[ifunc].axvline(turn_ii*t_rev_0/unit_t + (profile_bin_centers_ii[profile_bucket_centres[-1]] + bucket_centres_shift)/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1), zorder=1) # max(1.0/(len(turns_to_plot)+1), 1.0/(monitorotfb.i0+1)), zorder=0)
                            axia2[ifunc].axvline(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres[0]] /unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1) if monitorotfb.profile is not None else 0.125, zorder=1) # max(1.0/(len(turns_to_plot)+1), 1.0/(monitorotfb.i0+1)), zorder=0)
                            axia2[ifunc].axvline(turn_ii*t_rev_0/unit_t + profile_bin_centers_ii[profile_bucket_centres[-1]]/unit_t, color=color_ii, alpha=0.5/(len(turns_to_plot)+1) if monitorotfb.profile is not None else 0.125, zorder=1) # max(1.0/(len(turns_to_plot)+1), 1.0/(monitorotfb.i0+1)), zorder=0)

                # ALL
                # Cartesian
                for axiab in [[ax0a, ax1a, ax2a, ax3a]]: #, [ax0b, ax1b, ax2b, ax3b]]:
                    for ifunc in range(4):
                        unitparam_i = unit_param if ifunc < 3 else unit_angle
                        #val = np.array(list(map(func, param_ii)))/unitparam_i
                        if   ifunc == 0: val = np.real( param_ii)/unitparam_i
                        elif ifunc == 1: val = np.imag( param_ii)/unitparam_i
                        elif ifunc == 2: val = np.abs(  param_ii)/unitparam_i
                        elif ifunc == 3: val = np.angle(param_ii)/unitparam_i
                        #if ifunc == 3: val[ np.where(np.abs(val) > 0.999*np.pi)[0] ] = 0. # to make the +pi/-pi jumps go to zero
                        # Shift the corresponding time array by one t_rev
                        axiab[ifunc].plot(turn_ii*t_rev_0/unit_t + time_ii/unit_t, val, color=color_ii)
                    #
                    # Markers of the average of beam and no-beam segments (only for voltage-parameters)
                    #if ii == idx_turns_to_plot[0] or ii == idx_turns_to_plot[-1]:
                    #ms_ii = 10.0 if ii == idx_turns_to_plot[0] else 7.5 # 5.0
                    for ifunc in range(4):
                        unitparam_i = unit_param if ifunc < 3 else unit_angle
                        # To map this floats we need to make the iterable (i.e. a list), then just simply take the single 0-th element
                        if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                            #val_max   = list(map(func, [param_ii_max_beam]  ))[0]/unitparam_i
                            if   ifunc == 0: val_max   = np.real( param_ii_max_beam)/unitparam_i
                            elif ifunc == 1: val_max   = np.imag( param_ii_max_beam)/unitparam_i
                            elif ifunc == 2: val_max   = np.abs(  param_ii_max_beam)/unitparam_i
                            elif ifunc == 3: val_max   = np.angle(param_ii_max_beam)/unitparam_i
                        if monitorotfb.track_ave:
                            #val_beamM   = list(map(func, [param_ii_ave_beamM]  ))[0]/unitparam_i
                            if   ifunc == 0: val_beamM   = np.real( param_ii_ave_beamM)/unitparam_i
                            elif ifunc == 1: val_beamM   = np.imag( param_ii_ave_beamM)/unitparam_i
                            elif ifunc == 2: val_beamM   = np.abs(  param_ii_ave_beamM)/unitparam_i
                            elif ifunc == 3: val_beamM   = np.angle(param_ii_ave_beamM)/unitparam_i
                            #val_beamH   = list(map(func, [param_ii_ave_beamH]  ))[0]/unitparam_i
                            if   ifunc == 0: val_beamH   = np.real( param_ii_ave_beamH)/unitparam_i
                            elif ifunc == 1: val_beamH   = np.imag( param_ii_ave_beamH)/unitparam_i
                            elif ifunc == 2: val_beamH   = np.abs(  param_ii_ave_beamH)/unitparam_i
                            elif ifunc == 3: val_beamH   = np.angle(param_ii_ave_beamH)/unitparam_i
                            #if ifunc == 3 and np.abs(val_beamH) > 0.999*np.pi: val_beamH = 0. # to make the +pi/-pi jumps go to zero
                            if not np.isnan(param_ii_ave_nobeam):
                                #val_nobeam = list(map(func, [param_ii_ave_nobeam]))[0]/unitparam_i
                                if   ifunc == 0: val_nobeam = np.real( param_ii_ave_nobeam)/unitparam_i
                                elif ifunc == 1: val_nobeam = np.imag( param_ii_ave_nobeam)/unitparam_i
                                elif ifunc == 2: val_nobeam = np.abs(  param_ii_ave_nobeam)/unitparam_i
                                elif ifunc == 3: val_nobeam = np.angle(param_ii_ave_nobeam)/unitparam_i
                                #if ifunc == 3 and np.abs(val_nobeam) > 0.999*np.pi: val_nobeam = 0. # to make the +pi/-pi jumps go to zero
                        if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                            # The lines cannot be axhline any more, but extend only over the turn (except for 1st and last, for reference)
                            if ii == idx_turns_to_plot[0] or ii == idx_turns_to_plot[-1]:
                                axiab[ifunc].axhline(val_max,  color=color_ii, ls='--', alpha=0.75)
                            axiab[ifunc].plot( t_ii_max_beam/unit_t, val_max,  '^', markersize=10.0, markerfacecolor=color_ii, markeredgecolor='white')
                        if monitorotfb.track_ave:
                            # The lines cannot be axhline any more, but extend only over the turn (except for 1st and last, for reference)
                            if ii == idx_turns_to_plot[0] or ii == idx_turns_to_plot[-1]:
                                axiab[ifunc].axhline(val_beamM,  color=color_ii, ls=':',  alpha=0.50)
                                axiab[ifunc].axhline(val_beamH,  color=color_ii, ls='--', alpha=0.25)
                                if not np.isnan(param_ii_ave_nobeam):
                                    axiab[ifunc].axhline(val_nobeam, color=color_ii, ls=':',  alpha=0.50)
                            axiab[ifunc].plot( t_ii_ave_beamM /unit_t, val_beamM,  'D', markersize=10.0, markerfacecolor=color_ii, markeredgecolor='white')
                            axiab[ifunc].plot( t_ii_ave_beamH /unit_t, val_beamH,  's', markersize=10.0, markerfacecolor=color_ii, markeredgecolor='white')
                            if not np.isnan(param_ii_ave_nobeam):
                                axiab[ifunc].plot( t_ii_ave_nobeam/unit_t, val_nobeam, 'o', markersize= 7.5, markerfacecolor=color_ii, markeredgecolor='white')

                    # Vertical line at the end of each turn: t_rev
                    for axi in [ax0a, ax1a, ax2a, ax3a]: #[ax0b, ax1b, ax2b, ax3b]:
                        axi.axvline(turn_ii*t_rev_0/unit_t, color='grey', ls='-.', alpha=0.5)

            for axia in [ax0a, ax1a, ax2a, ax3a]:
                axia.yaxis.set_major_formatter(FormatStrFormatter('%5.4f'))

            for axia2 in [ax0a2, ax1a2, ax2a2, ax3a2]:
                axia2.tick_params(labelright=False)
                axia2.tick_params(right=False)


            ax3a.set_xlabel(f'Time [{symb_t}]')

            ax0a.set_ylabel(f'Real [{symb_param}]')
            ax1a.set_ylabel(f'Imaginary [{symb_param}]')
            ax2a.set_ylabel(f'Magnitude [{symb_param}]')
            ax3a.set_ylabel(f'Angle [{symb_angle}]')

            # ax4.legend(loc=3, framealpha=1.0)

            figtitle = f'{param} '
            if 'sum' not in param and 'tracker' not in param:
                if   '1' in param: ot = '1'
                elif '2' in param: ot = '2'
                cavtype_ot = getattr(getattr(cavityfeedback, f'OTFB_{ot}'), 'cavtype')
                figtitle = figtitle + f'(1x{cavtype_ot[2:]})' if 'P' in param else figtitle + f'({cavtype_ot})'

            ax0a.set_title(figtitle)
            fig.tight_layout()
            #fname = f'{outdir}/plot_cavityfeedback_{param}.png'
            #print(f'Saving {fname} ...')
            #fig.savefig(fname)
            pdf.savefig()
            #plt.cla()
            plt.close() #fig)

    print(f'Saving {fname} ...')

    # gc.collect()


# CavityFeedback (MonitorOTFB) ave parameters all turns (continuous) ----------

def plot_cavityfeedback_allparams_ave_continuous_turns(outdir, monitorotfb, cavityfeedback):

    # We can only look at the ave beam data in the continuos-turns plot:

    fname = f'{outdir}/plot_cavityfeedback_allparams_ave_continuous_turns.pdf'
    #fname = f'{outdir}/plot_cavityfeedback_allparams_continuous_turns-{turn0_seti}-{turnf_seti}.pdf'
    with PdfPages(fname) as pdf:

        for param in monitorotfb.list_params:

            ot = '1' if '1' in param else '2'
            if   'fine'     in param: fc = 'fine'
            elif 'coarseFF' in param: fc = 'coarseFF'
            else:                     fc = 'coarse'

            nrows = 4
            ncols = 2

            fig = plt.figure(constrained_layout=True) # fig, ax = plt.subplots(nrows, ncols, sharex=True)
            fig.set_size_inches(ncols*10.00, nrows*2.50)

            gs = fig.add_gridspec(nrows, ncols)
            ax0a = fig.add_subplot(gs[0,0])
            ax1a = fig.add_subplot(gs[1,0])
            ax2a = fig.add_subplot(gs[2,0])
            ax3a = fig.add_subplot(gs[3,0])
            ax4 = fig.add_subplot(gs[:,1], projection='polar')

            turns_to_plot = monitorotfb.turns[:] # e.g. turns 0, 2, 4, etc
            idx_turns_to_plot = np.arange(len(turns_to_plot)) # would correspond to 0, 1, 2, etc.
            if len(turns_to_plot) == 1:
                idx_turns_to_plot = np.array([0])
                color_cycle_nb = ['k'] # Last (current) turn always black
            else:
                mycm = plt.get_cmap('coolwarm')
                color_cycle_nb = [mycm(ii/(len(idx_turns_to_plot)-1.)) for ii in idx_turns_to_plot]
                color_cycle_nb[-1] = 'k' # Last (current) turn always black

            unit_t = 1;              symb_t = 'turns'
            unit_Q = 1e-9;           symb_Q = 'nC'
            unit_I = 1;              symb_I = 'A'
            unit_V = 1e6;            symb_V = 'MV'
            unit_P = 1e6;            symb_P = 'MW'
            unit_angle = np.pi/180.; symb_angle = 'deg' # rad to deg

            if   'Q' in param: # Q_gen, Q_beam
                unit_param = unit_Q
                symb_param = symb_Q
            elif 'I' in param: # I_gen, I_beam
                unit_param = unit_I
                symb_param = symb_I
            elif 'V' in param: # Vind, Vtot, Vgen
                unit_param = unit_V
                symb_param = symb_V
            elif 'P' in param: # Pgen
                unit_param = unit_P
                symb_param = symb_P

            #for axi in [ax0a, ax1a, ax2a, ax3a]:
            #    axi.set_prop_cycle(cycler('color', color_cycle_nb))

            #print(i, param)
            turns_all = turns_to_plot[idx_turns_to_plot]
            #turns_all = turns_all.reshape(turns_all.shape[0],1)
            #print(f'turns_all = {turns_all}, shape = {turns_all.shape}')

            if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                param_all_max_beam  = getattr(monitorotfb, f'{param}_max_beam' )[idx_turns_to_plot]
            else:
                param_all_max_beam = np.empty(len(idx_turns_to_plot)) * np.NaN
            if monitorotfb.track_ave:
                param_all_ave_beamM  = getattr(monitorotfb, f'{param}_ave_beamM' )[idx_turns_to_plot]
                param_all_ave_beamH  = getattr(monitorotfb, f'{param}_ave_beamH' )[idx_turns_to_plot]
                param_all_ave_nobeam = getattr(monitorotfb, f'{param}_ave_nobeam')[idx_turns_to_plot]
                #param_all_ave_beamH  = param_all_ave_beamH.reshape(  param_all_ave_beamH.shape[0],1)
                #param_all_ave_nobeam = param_all_ave_nobeam.reshape(param_all_ave_nobeam.shape[0],1)
                #print(f'param_all_ave_beamH = {param_all_ave_beamH}, shape = {param_all_ave_beamH.shape}')

            # ALL
            # Cartesian
            for axiab in [[ax0a, ax1a, ax2a, ax3a]]:
                for ifunc in range(4):
                    unitparam_i = unit_param if ifunc < 3 else unit_angle
                    # To map this floats we need to make the iterable (i.e. a list), then just simply take the single 0-th element
                    if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                        #val_max   = list(map(func, [param_all_max_beam]  ))[0]/unitparam_i
                        if   ifunc == 0: val_max   = np.real( param_all_max_beam)/unitparam_i
                        elif ifunc == 1: val_max   = np.imag( param_all_max_beam)/unitparam_i
                        elif ifunc == 2: val_max   = np.abs(  param_all_max_beam)/unitparam_i
                        elif ifunc == 3: val_max   = np.angle(param_all_max_beam)/unitparam_i
                        labelmax  = 'Beam (max)' if ifunc == 3 else None
                    if monitorotfb.track_ave:
                        #val_beamM   = list(map(func, [param_all_ave_beamM]  ))[0]/unitparam_i
                        if   ifunc == 0: val_beamM   = np.real( param_all_ave_beamM)/unitparam_i
                        elif ifunc == 1: val_beamM   = np.imag( param_all_ave_beamM)/unitparam_i
                        elif ifunc == 2: val_beamM   = np.abs(  param_all_ave_beamM)/unitparam_i
                        elif ifunc == 3: val_beamM   = np.angle(param_all_ave_beamM)/unitparam_i
                        #val_beamH   = list(map(func, [param_all_ave_beamH]  ))[0]/unitparam_i
                        if   ifunc == 0: val_beamH   = np.real( param_all_ave_beamH)/unitparam_i
                        elif ifunc == 1: val_beamH   = np.imag( param_all_ave_beamH)/unitparam_i
                        elif ifunc == 2: val_beamH   = np.abs(  param_all_ave_beamH)/unitparam_i
                        elif ifunc == 3: val_beamH   = np.angle(param_all_ave_beamH)/unitparam_i
                        #if ifunc == 3 and np.abs(val_beamH) > 0.999*np.pi: val_beamH = 0. # to make the +pi/-pi jumps go to zero
                        #if not np.isnan(param_ii_ave_nobeam):
                        #val_nobeam = list(map(func, [param_all_ave_nobeam]))[0]/unitparam_i
                        if   ifunc == 0: val_nobeam = np.real( param_all_ave_nobeam)/unitparam_i
                        elif ifunc == 1: val_nobeam = np.imag( param_all_ave_nobeam)/unitparam_i
                        elif ifunc == 2: val_nobeam = np.abs(  param_all_ave_nobeam)/unitparam_i
                        elif ifunc == 3: val_nobeam = np.angle(param_all_ave_nobeam)/unitparam_i
                        #if ifunc == 3 and np.abs(val_nobeam) > 0.999*np.pi: val_nobeam = 0. # to make the +pi/-pi jumps go to zero
                        labelbeamM  = 'Beam (mid)' if ifunc == 3 else None
                        labelbeamH  = 'Beam (ave)' if ifunc == 3 else None
                        labelnobeam = 'No beam'    if ifunc == 3 else None
                    #
                    if monitorotfb.track_max and param in monitorotfb.list_params_max_beam:
                        axiab[ifunc].plot(turns_all/unit_t, val_max,    '^-',  label=labelmax,    alpha=0.75, color='#0022dd')   # color='k', alpha=0.50,
                    if monitorotfb.track_ave:
                        axiab[ifunc].plot(turns_all/unit_t, val_beamM,  'D:',  label=labelbeamM,  alpha=0.50, color='#0033cc')  # color='k', alpha=0.50,
                        axiab[ifunc].plot(turns_all/unit_t, val_beamH,  's--', label=labelbeamH,  alpha=0.25, color='#0044bb')  # color='k', alpha=0.50,
                        # if not np.isnan(param_all_ave_nobeam):
                        axiab[ifunc].plot(turns_all/unit_t, val_nobeam, 'o:',  label=labelnobeam, alpha=0.50, color='#dd0000') # color='k', alpha=0.25,
                        #for ii in range(len(val_beamH)):
                        #    # The nobeam segement is displaced by 0.1 to avoid overlap and bettter see the points in the plot
                        #    axiab[ifunc].plot(turns_all[ii]/unit_t,     val_beamH[ii],   's', color=color_cycle_nb[ii], alpha=1.00, markersize=5.0, label=labelbeamH)
                        #    axiab[ifunc].plot(turns_all[ii]/unit_t+0.1, val_nobeam[ii], 'o', color=color_cycle_nb[ii], alpha=0.50, markersize=3.3, label=labelnobeam)

                    ####
                    # points_beam   = np.array([turns_all/unit_t,     param_all_ave_beamH  ]).T.reshape(-1,1,2)
                    # points_nobeam = np.array([turns_all/unit_t+0.5, param_all_ave_nobeam]).T.reshape(-1,1,2)
                    # segments_beam = np.concatenate([points_beam[:-1],points_beam[1:]], axis=1)
                    # segments_nobeam = np.concatenate([points_nobeam[:-1],points_nobeam[1:]], axis=1)

                    # #cmap = LinearSegmentedColormap.from_list("", color_cycle_nb)
                    # cmap = LinearSegmentedColormap.from_list("", [(1, 0, 0), (0, 0, 1)])

                    # lc_beam   = LineCollection(segments_beam,   cmap=cmap) #, linewidth=10)
                    # lc_nobeam = LineCollection(segments_nobeam, cmap=cmap) #, linewidth=10)
                    # lc_beam.set_array(turns_all/unit_t)
                    # lc_nobeam.set_array(turns_all/unit_t+0.5)

                    # axiab[ifunc].add_collection(lc_beam)
                    # #axiab[ifunc].autoscale()
                    # #plt.show()

            # Polar: no need to add unit_angle to 1st array, as polar plots already are shown in deg
            ms_ii = 7.5 #5.0

            ax4.plot(np.angle(param_all_max_beam),   np.abs(param_all_max_beam) /unit_param,  '^-',  alpha=0.75, color='#0022dd', markerfacecolor='#0022dd', markeredgecolor='white', markersize=ms_ii) #, label=labelmax)
            ax4.plot(np.angle(param_all_ave_beamM),  np.abs(param_all_ave_beamM)/unit_param,  'D:',  alpha=0.50, color='#0033cc', markerfacecolor='#0033cc', markeredgecolor='white', markersize=ms_ii) #, label=labelbeamM)
            ax4.plot(np.angle(param_all_ave_beamH),  np.abs(param_all_ave_beamH)/unit_param,  's--', alpha=0.25, color='#0044bb', markerfacecolor='#0044bb', markeredgecolor='white', markersize=ms_ii) #, label=labelbeamH)
            # if not np.isnan(param_ii_ave_nobeam):
            ax4.plot(np.angle(param_all_ave_nobeam), np.abs(param_all_ave_nobeam)/unit_param, 'o:',  alpha=0.50, color='#dd0000', markerfacecolor='#dd0000', markeredgecolor='white', markersize=ms_ii) #, label=labelnobeam)

            ax4.set_ylim(0.0, 1.1*ax4.get_ylim()[-1])

            ax3a.set_xlabel('Turns')

            ax0a.set_ylabel(f'Real [{symb_param}]')
            ax1a.set_ylabel(f'Imaginary [{symb_param}]')
            ax2a.set_ylabel(f'Magnitude [{symb_param}]')
            ax3a.set_ylabel(f'Angle [{symb_angle}]')

            ax3a.legend(loc=1)

            # ax4.legend(loc=3, framealpha=1.0)

            figtitle = f'{param} '
            if 'sum' not in param and 'tracker' not in param:
                if   '1' in param: ot = '1'
                elif '2' in param: ot = '2'
                cavtype_ot = getattr(getattr(cavityfeedback, f'OTFB_{ot}'), 'cavtype')
                figtitle = figtitle + f'(1x{cavtype_ot[2:]})' if 'P' in param else figtitle + f'({cavtype_ot})'

            fig.suptitle(figtitle)
            # fig.tight_layout()
            #fname = f'{outdir}/plot_cavityfeedback_{param}.png'
            #print(f'Saving {fname} ...')
            #fig.savefig(fname)
            pdf.savefig()
            #plt.cla()
            plt.close() #fig)

    print(f'Saving {fname} ...')

    # gc.collect()


# Some older plots from which the allparams plots above were built:

# ####

# if False:

#      #currentfact = 1e-6 # for uC
#      currentfact = 1e3 # for kA
#      currentfact = 1. # for A

#      nrows = 10
#      fig, ax = plt.subplots(nrows,int(3*2)) #,sharex=True) #,sharey='row')
#      fig.set_size_inches(3*(4.00*2), nrows*2.5)

#      mycm = plt.get_cmap('coolwarm')
#      color_cycle_nb = [mycm(ii/(MAC['Nt_trk']-1-1.)) for ii in range(MAC['Nt_trk']-1)]

#      # ncoarse1 = len(OTFB1_Vgen_coarse[0])
#      # ncoarse2 = len(OTFB2_Vgen_coarse[0])
#      # fact_VgenIgen_1 = cavityfeedback.OTFB_1.G_tx/cavityfeedback.OTFB_1.TWC.R_gen*cavityfeedback.OTFB_1.T_s_coarse
#      # fact_VgenIgen_2 = cavityfeedback.OTFB_2.G_tx/cavityfeedback.OTFB_2.TWC.R_gen*cavityfeedback.OTFB_2.T_s_coarse

#      for jext in [0,1]:

#          for spi in range(nrows):
#              for spj in [0,2,4]:
#                  ax[spi,spj+jext].set_prop_cycle(cycler('color', color_cycle_nb))
#                  ax[spi,spj+jext].axvline( (ncoarse1+0.5)*cavityfeedback.OTFB_1.rf.t_rf[0,0]/1e-6, color='grey', ls=':') # We are using OTFB_1 are the same for both

#          # Remainder of induced voltage from impedance model only for the current turn:
#         #ax[9,4+jext].plot(totalinducedvoltage.time_array/1e-6, total_induced_voltage_extra/1e6, '-', c='green', label='|V_imp|')
#          ax[9,4+jext].plot(totalinducedvoltage.time_array/1e-6, total_induced_voltage_extra/1e6 + np.absolute(cavityfeedback.OTFB_1.V_tot_fine + cavityfeedback.OTFB_2.V_tot_fine)/1e6, '-', c='orange', label='|V_ind| + |V_imp|')

#          #for ii in [0,-1]: # plot first and current turn only
#          for ii in range(len(OTFB1_I_gen_coarse)): # plot first and current turn only

#              mycolor = 'k' if ii == len(OTFB1_I_gen_coarse)-1 else None

#              # Generator voltage
#              ax[0,0+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vgen_coarse[ii])/1e6, '-',  c=mycolor, label='|Vgen_1|')
#              ax[0,2+jext].plot(cavityfeedback.OTFB_2.rf_centers/1e-6, np.absolute(OTFB2_Vgen_coarse[ii])/1e6,  '-',  c=mycolor, label='|Vgen_2|')
#              #
#              # Generator current
#              ax[1,0+jext].plot((np.arange(ncoarse1+nmovave1)+0.5)*cavityfeedback.OTFB_1.rf.t_rf[0,0]/1e-6, np.absolute(OTFB1_I_gen_coarse[ii])/currentfact, '-', c=mycolor, label='|Igen_1|')
#              ax[1,2+jext].plot((np.arange(ncoarse2+nmovave2)+0.5)*cavityfeedback.OTFB_2.rf.t_rf[0,0]/1e-6, np.absolute(OTFB2_I_gen_coarse[ii])/currentfact, '-', c=mycolor, label='|Igen_2|')
#              #
#              # Generator induced voltage
#              ax[2,0+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vgenind_coarse[ii])/1e6, '-', c=mycolor, label='|Vgen_ind_1|')
#              ax[2,2+jext].plot(cavityfeedback.OTFB_2.rf_centers/1e-6, np.absolute(OTFB2_Vgenind_coarse[ii])/1e6, '-', c=mycolor, label='|Vgen_ind_2|')
#              ax[2,4+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vgenind_coarse[ii] + OTFB2_Vgenind_coarse[ii])/1e6, '-', c=mycolor, label='|Vgen_ind| = |Vgen_ind_1 + Vgen_ind_2|')
#              #
#              # Generator power
#              ax[3,0+jext].plot((np.arange(ncoarse1+nmovave1)+0.5)*cavityfeedback.OTFB_1.rf.t_rf[0,0]/1e-6, OTFB1_P_gen_0_coarse[ii]/1e6, '-', c=mycolor, label='|Igen_1|')
#              ax[3,2+jext].plot((np.arange(ncoarse2+nmovave2)+0.5)*cavityfeedback.OTFB_2.rf.t_rf[0,0]/1e-6, OTFB2_P_gen_0_coarse[ii]/1e6, '-', c=mycolor, label='|Igen_2|')
#              #
#              #
#              # Beam current
#              ax[4,0+jext].plot((np.arange(ncoarse1)+0.5)*cavityfeedback.OTFB_1.rf.t_rf[0,0]/1e-6, np.absolute(OTFB1_Ibeam_coarse[ii])/currentfact, '-', c=mycolor, label='|Ibeam_1|')
#              ax[4,2+jext].plot((np.arange(ncoarse2)+0.5)*cavityfeedback.OTFB_2.rf.t_rf[0,0]/1e-6, np.absolute(OTFB2_Ibeam_coarse[ii])/currentfact, '-', c=mycolor, label='|Ibeam_2|')
#              ax[5,0+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB1_Ibeam_fine[ii])/currentfact, '-', c=mycolor, label='|Ibeam_1|')
#              ax[5,2+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB2_Ibeam_fine[ii])/currentfact, '-', c=mycolor, label='|Ibeam_2|')
#              #
#              # Beam induced voltage
#              ax[6,0+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vbeamind_coarse[ii])/1e6, '-', c=mycolor, label='|Vbeam_ind_1|')
#              ax[6,2+jext].plot(cavityfeedback.OTFB_2.rf_centers/1e-6, np.absolute(OTFB2_Vbeamind_coarse[ii])/1e6, '-', c=mycolor, label='|Vbeam_ind_2|')
#              ax[6,4+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vbeamind_coarse[ii] + OTFB2_Vbeamind_coarse[ii])/1e6, '-', c=mycolor, label='|Vbeam_ind| = |Vbeam_ind_1 + Vbeam_ind_2|')
#              ax[7,0+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB1_Vbeamind_fine[ii])/1e6, '-', c=mycolor, label='|Vbeam_ind_1|')
#              ax[7,2+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB2_Vbeamind_fine[ii])/1e6, '-', c=mycolor, label='|Vbeam_ind_2|')
#              ax[7,4+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB1_Vbeamind_fine[ii] + OTFB2_Vbeamind_fine[ii])/1e6, '-', c=mycolor, label='|Vbeam_ind| = |Vbeam_ind_1 + Vbeam_ind_2|')
#              #
#              # Total (generator + beam) induced voltage
#              ax[8,0+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vtotind_coarse[ii])/1e6, '-', c=mycolor, label='|Vtot_ind_1|')
#              ax[8,2+jext].plot(cavityfeedback.OTFB_2.rf_centers/1e-6, np.absolute(OTFB2_Vtotind_coarse[ii])/1e6, '-', c=mycolor, label='|Vtot_ind_2|')
#              ax[8,4+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, np.absolute(OTFB1_Vtotind_coarse[ii] + OTFB2_Vtotind_coarse[ii])/1e6, '-', c=mycolor, label='|Vtot_ind| = |Vtot_ind_1 + Vtot_ind_2|')
#              ax[9,0+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB1_Vtotind_fine[ii])/1e6, '-', c=mycolor, label='|Vtot_ind_1|')
#              ax[9,2+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB2_Vtotind_fine[ii])/1e6, '-', c=mycolor, label='|Vtot_ind_2|')
#              ax[9,4+jext].plot(profile.bin_centers/1e-6, np.absolute(OTFB1_Vtotind_fine[ii] + OTFB2_Vtotind_fine[ii])/1e6, '-', c=mycolor, label='|Vtot_ind| = |Vtot_ind_1 + Vtot_ind_2|') # This is the same than cavityfeedback.V_sum (times the Vrf0)
#              #
#              # Total voltage from impedance model (i.e. all costributions except the main TWCs):
#              # Taking just the value at the center of each bunch:
#              #ax[9,2+jext].plot(totalinducedvoltage.time_array[profile_bucket_centres]/1e-6,   total_induced_voltage_extra[profile_bucket_centres]/1e6, '-', color='green')
#              # Interpolating to the same cavityfeedback coarse beam:
#              #totalinducedvoltage_induced_voltage_i = np.interp(cavityfeedback.OTFB_1.rf_centers, totalinducedvoltage.time_array, total_induced_voltage_extra)
#              #ax[8,2+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, totalinducedvoltage_induced_voltage_i/1e6, '-', color='green', label='|V_ind| (all except main TWCs)')
#              #ax[8,2+jext].plot(cavityfeedback.OTFB_1.rf_centers/1e-6, totalinducedvoltage_induced_voltage_i/1e6 + np.absolute(OTFB1_Vgenind_coarse[ii] + OTFB2_Vgenind_coarse[ii])/1e6, '-', color='orange', label='|V_ind| + |V_gen_coarse|')
#              # As is, that is, at the the full profile time array, which is the same that the fine grids of cavityfeedback:
#             #ax[9,2+jext].plot(totalinducedvoltage.time_array/1e-6, total_induced_voltage_extra/1e6, '-', color='green', label='|V_imp| (all except main TWCs)')
#              #
#              # Just to see the -v^- of the induced voltage (impedance model contribution, shape centred around each bunch, with peaks (+/-) growing along the batch until a steady-state, with ripples in the separation buckets):
#              #ax[0,4+jext].plot(totalinducedvoltage.time_array[:int(64*6)]/1e-6*1e3, total_induced_voltage_extra[:int(64*6)]/1e6, '-', color='green')

#      for spi in range(nrows):
#          for spj in [0, 2, 4]:

#              twindow = 0.035 # um: 3.5 to see the full batch, 0.035 to see the first 2 bunches
#              ax[spi,spj  ].set_xlim(0,twindow)
#              ax[spi,spj+1].set_xlim( (ncoarse2+nmovave2+0.5)*cavityfeedback.OTFB_2.rf.t_rf[0,0]/1e-6 - twindow,(ncoarse2+nmovave2+0.5)*cavityfeedback.OTFB_2.rf.t_rf[0,0]/1e-6)

#              ax_spi_spj_ylim = ax[spi,spj].get_ylim()
#              ax[spi,spj+1].set_ylim(ax_spi_spj_ylim)

#              ax[spi,spj  ].spines['right'].set_visible(False)
#              ax[spi,spj+1].spines['left' ].set_visible(False)
#              ax[spi,spj  ].yaxis.tick_left()
#              ax[spi,spj+1].yaxis.tick_right()
#              ax[spi,spj  ].tick_params(labelright=False)
#              ax[spi,spj+1].tick_params(labelleft=False)
#              ax[spi,spj  ].tick_params(right=False)
#              ax[spi,spj+1].tick_params(left=False)

#      jext=0
#      #ax[0,0].legend(loc=8)
#      #ax[2,5].legend(loc=4)
#      ax[3,5].legend(loc=1)
#      #ax[4,5].legend(loc=1)
#      #
#      ax[0,0].set_ylabel('|Vgen|\n(coarse) [MV]')
#      #
#      ax[1,0].set_ylabel('|Igen|\n(coarse) [A]')
#     #ax[1,0].set_ylabel('|Igen|\n(coarse) [kA]')
#     #ax[1,0].set_ylabel('|Igen|\n(coarse) [$\mu$C]')
#     #
#      ax[2,0].set_ylabel('|Vgen_ind|\n(coarse) [MV]')
#      #
#     #ax[3,0].set_ylabel('Pgen/Zext\n[MW / circuit $\Omega$]')
#      ax[3,0].set_ylabel('Pgen (1 cav)\n(coarse) [MW]')
#      #
#      ax[4,0].set_ylabel('|Ibeam|\n(coarse) [A]')
#     #ax[4,0].set_ylabel('|Ibeam|\n(coarse) [kA]')
#     #ax[4,0].set_ylabel('|Ibeam|\n(coarse) [$\mu$C]')
#      ax[5,0].set_ylabel('|Ibeam|\n(fine) [A]')
#     #ax[5,0].set_ylabel('|Ibeam|\n(fine) [kA]')
#     #ax[5,0].set_ylabel('|Ibeam|\n(fine) [$\mu$C]')
#      #
#      ax[6,0].set_ylabel('|Vbeam_ind|\n(coarse) [MV]')
#      ax[7,0].set_ylabel('|Vbeam_ind|\n(fine) [MV]')
#      #
#      ax[8,0].set_ylabel('|Vtot_ind|\n(coarse) [MV]')
#      ax[9,0].set_ylabel('|Vtot_ind|\n(fine) [MV]')
#      #
#      #
#     #  ax[0,4].set_ylabel('|V_ind|,\n|V_ind+imp|\n(fine) [MV]')
#     # #ax[0,4].set_ylabel('|V_ind|n\n|V_imp|,\n|V_ind+imp| [MV]')
#     #  ax[1,4].set_ylabel('N/A')
#     #  ax[2,4].set_ylabel('|Vgen_ind|\n(coarse) [MV]')
#     #  ax[3,4].set_ylabel('N/A')
#     #  ax[5,4].set_ylabel('N/A')
#     #  ax[4,4].set_ylabel('N/A')
#     #  ax[6,4].set_ylabel('|Vbeam_ind|\n(coarse) [MV]')
#     #  ax[7,4].set_ylabel('|Vbeam_ind|\n(fine) [MV]')
#      #
#      ax[nrows-1,0].set_xlabel(r'Time [$\mu$s]')
#      ax[nrows-1,2].set_xlabel(r'Time [$\mu$s]')
#      ax[nrows-1,4].set_xlabel(r'Time [$\mu$s]')
#      #
#      ax[0,0].set_title(f'{cavityfeedback.OTFB_1.cavtype}')
#      ax[0,2].set_title(f'{cavityfeedback.OTFB_2.cavtype}')
#      ax[0,4].set_title('Full RF')
#      #
#      # To see the beam segment in detail. Remove to see the full
#      # turn (useful to see the last points in the turn, where the
#      # voltage is preparing for the upcoming beam segment in the
#      # next turn):
#      #
#      fig.tight_layout()
#      fname = f'{outdir}/plot_gen-beam.png'
#      print(f'Saving {fname} ...')
#      fig.savefig(fname)
#      plt.cla()
#      plt.close(fig)

## Voltage by feed-forwdward plot:

# if(i % MAC['Nt_plt'] == 0 or i == MAC['Nt_trk']-2):

#     fig, ax = plt.subplots(3, 2, sharey=True)
#     fig.set_size_inches(6.0*2, 3.0*3)
#     #
#     Vff_t   = cavityfeedback.OTFB_2.rf_centers[::5]
#     #
#     ax[0,0].set_ylabel('OTFB1 [MV]')
#     ncav1         = cavityfeedback.OTFB_1.n_cavities
#     coeffff1      = cavityfeedback.OTFB_1.coeff_FF
#     coeffff1_norm = coeffff1/np.sum(coeffff1)*ncav1
#     ncoefff1      = cavityfeedback.OTFB_1.n_FF # = len(coeffff1)
#     coeffff1_t    = np.arange(ncoefff1)*(beampattern.nbs*beampattern.maxbktl)
#     Vff1corr = cavityfeedback.OTFB_1.V_ff_corr
#     dVff1    = cavityfeedback.OTFB_1.dV_ff
#     for jj in range(2):
#         ax[0,jj].plot(coeffff1_t/1e-6, coeffff1_norm,        '.-', label=f'coeff_norm')
#         ax[0,jj].plot(Vff_t     /1e-6, np.abs(Vff1corr)/1e6, '--', label=f'|V_ff_corr|')
#         ax[0,jj].plot(Vff_t     /1e-6, np.abs(dVff1)   /1e6, '-',  label=f'|dV_ff|')
#         ax[0,jj].grid()
#     #
#     ax[1,0].set_ylabel('OTFB2 [MV]')
#     ncav2         = cavityfeedback.OTFB_2.n_cavities
#     coeffff2      = cavityfeedback.OTFB_2.coeff_FF
#     coeffff2_norm = coeffff2/np.sum(coeffff2)*ncav2
#     ncoefff2      = cavityfeedback.OTFB_2.n_FF # = len(coeffff2)
#     coeffff2_t    = np.arange(ncoefff2)*(beampattern.nbs*beampattern.maxbktl)
#     Vff2corr = cavityfeedback.OTFB_2.V_ff_corr
#     dVff2    = cavityfeedback.OTFB_2.dV_ff
#     for jj in range(2):
#         ax[1,jj].plot(coeffff2_t/1e-6, coeffff2_norm,        '.-', label=f'coeff_norm')
#         ax[1,jj].plot(Vff_t     /1e-6, np.abs(Vff2corr)/1e6, '--', label=f'|V_ff_corr|')
#         ax[1,jj].plot(Vff_t     /1e-6, np.abs(dVff2)   /1e6, '-',  label=f'|dV_ff|')
#         ax[1,jj].grid()
#     #
#     ax[2,0].set_ylabel('OTFB1+2 [MV]')
#     #print(f'coeffff1_t = {coeffff1_t}, shape = {coeffff1_t.shape}')
#     #print(f'coeffff2_t = {coeffff2_t}, shape = {coeffff2_t.shape}')
#     if(ncoefff1 > ncoefff2):
#         coeffff_t = np.copy(coeffff1_t)
#         coeffff2_norm_tmp = np.hstack( (coeffff2_norm, np.zeros(ncoefff1-ncoefff2)) )
#         coeffff_norm = coeffff1_norm + coeffff2_norm_tmp
#     else:
#         coeffff_t = np.copy(coeffff2_t)
#         coeffff1_norm_tmp = np.hstack( (coeffff1_norm, np.zeros(ncoefff2-ncoefff1)) )
#         coeffff_norm = coeffff1_norm_tmp + coeffff2_norm
#     for jj in range(2):
#         ax[2,jj].plot(coeffff2_t/1e-6, coeffff_norm,                    '.-', label=f'coeff_norm')
#         ax[2,jj].plot(Vff_t     /1e-6, np.abs(Vff1corr + Vff2corr)/1e6, '--', label=f'|V_ff_corr|')
#         ax[2,jj].plot(Vff_t     /1e-6, np.abs(dVff1    + dVff2   )/1e6, '-',  label=f'|dV_ff|')
#         ax[2,jj].set_xlabel('Time [$\mu$s]')
#         ax[2,jj].grid()
#     ax[2,1].legend(loc=2)
#     #
#     t0R =                     beampattern.bucket_centres[-1]+40*beampattern.nbs*beampattern.maxbktl #
#     1.5 * profilepattern.Ns * beampattern.fillpattern[-1]
#     t1L = profile.cut_right - beampattern.bucket_centres[-1]-40*beampattern.nbs*beampattern.maxbktl #
#     profile.n_slices - 1.5 * profilepattern.Ns * beampattern.fillpattern[-1] # We can also use
#     ring.t_rev since = profile.cut_right for SPS-FT with OTFB
#     for ii in range(3):
#         ax[ii,0].set_xlim(0.,       t0R/1e-6              )
#         ax[ii,1].set_xlim(t1L/1e-6, profile.cut_right/1e-6)
#     #
#     fig.suptitle(f'Turn {i}')
#     fig.tight_layout() #w_pad=1.0) # fig.tight_layout(pad=5.0, h_pad=0.0)
#     fig.savefig(f'{outdir}/plot_Vff_dVff_tmp.png')
#     plt.cla()
#     plt.close(fig)
