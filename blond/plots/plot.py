
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Class to choose plots and customize plot layout**

:Authors: **Helga Timko**
'''

import os

import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
from ..plots.plot_beams import (plot_long_phase_space, plot_bunch_length_evol,
                                plot_bunch_length_evol_gaussian, plot_position_evol,
                                plot_energy_evol, plot_transmitted_particles)
from ..plots.plot_llrf import (plot_PL_bunch_phase,
                               plot_PL_RF_phase, plot_PL_phase_corr, plot_PL_RF_freq,
                               plot_PL_freq_corr, plot_RF_phase_error, plot_RL_radial_error,
                               plot_COM_motion, plot_LHCNoiseFB, plot_LHCNoiseFB_FWHM,
                               plot_LHCNoiseFB_FWHM_bbb)
from ..plots.plot_slices import (plot_beam_profile, plot_beam_spectrum)

def fig_folder(dirname):
    '''
    Create folder where plots will be stored.
    '''

    # Try to create directory
    try:
        os.makedirs(dirname)
    # Check whether already exists/creation failed
    except OSError:
        if os.path.exists(dirname):
            pass
        else:
            raise


class Plot:

    def __init__(self, Ring, RFStation, Beam, dt_plot,
                 dt_bckp, xmin, xmax, ymin, ymax, xunit = 's', sampling = 1,
                 show_plots = False,
                 separatrix_plot = False, histograms_plot = True, 
                 Profile = None, h5file = None, output_frequency = 1, 
                 PhaseLoop = None, LHCNoiseFB = None, format_options = None):
        '''
        Define what plots should be plotted during the simulation. Passing only
        basic objects, only phase space plot will be produced. Passing optional
        objects, plots related to those objects will be produced as well.
        For plots at a certain turn: use 'dt_plot' to set the plotting frequency 
        in units of time steps. 
        For plots as a function of time: use 'dt_bckp' to set plotting frequency
        in units of time steps.
        '''


        #: | *Import Ring*
        self.general_params = Ring

        #: | *Import RFStation*
        self.rf_params = RFStation

        #: | *Import actual time step RFStation*
        self.tstep = RFStation.counter

        #: | *Import Beam*
        self.beam = Beam

        #: | *Defining whether the plots should be saved or directly shown*
        self.show_plt = show_plots

        #: | *Plotting frequency in units of time steps*
        self.dt_plot = dt_plot
        self.dt_bckp = dt_bckp

        #: | *Plot limit (where applicable) minimum on x-axis [xunit]*
        self.xmin = xmin

        #: | *Plot limit (where applicable) maximum on x-axis [xunit]*
        self.xmax = xmax

        #: | *Choice of x-axis unit (where applicable) 's' or 'rad'*
        self.xunit = xunit

        #: | *Plot limit (where applicable) minimum on y-axis [eV]*
        self.ymin = ymin

        #: | *Plot limit (where applicable) maximum on y-axis [eV]*
        self.ymax = ymax

        #: | *Sampling of large arrays (where applicable)*
        self.sampling = sampling

        #: | *Separatrix in phase space plot 'True' or 'False'*
        self.separatix = separatrix_plot

        #: | *Histogram in phase space plot 'True' or 'False'*
        self.histogram = histograms_plot

        #: | *Optional import of Profile*
        self.profile = Profile

        #: | *Optional import of Monitor file*
        self.h5file = h5file

        #: | *Optional sampling of monitored quantities*
        self.dt_mon = output_frequency

        #: | *Optional import of PhaseLoop*
        self.PL = PhaseLoop

        #: | *Optional import of LHCNoiseFB*
        self.noiseFB = LHCNoiseFB

        # Set plotting format
        self.set_format(format_options)

        # Track at initialisation
        self.track()

    def set_format(self, format_options):
        '''
        Initialize plot folder and custom plot formatting. For more options, see

        http://matplotlib.org/1.3.1/users/customizing.html
        '''

        if format_options is None:
            format_options = {'dummy': 0}

        if 'dirname' not in format_options:
            self.dirname = 'fig'
        else:
            self.dirname = format_options['dirname']

        if 'linewidth' not in format_options:
            self.lwidth = 2
        else:
            self.lwidth = format_options['linewidth']

        if 'linestyle' not in format_options:
            self.lstyle = '-'
        else:
            self.lstyle = format_options['linestyle']

        if 'markersize' not in format_options:
            self.msize = 6
        else:
            self.msize = format_options['markersize']

        if 'alpha' not in format_options:
            self.alpha = alpha=10**(-np.log10(self.beam.n_macroparticles)/6)
        else:
            self.alpha = format_options['alpha']

        if 'labelsize' not in format_options:
            self.lsize = 18
        else:
            self.lsize = format_options['labelsize']

        if 'fontfamily' not in format_options:
            self.ffamily = 'sans-serif'
        else:
            self.ffamily = format_options['fontfamily']

        if 'fontweight' not in format_options:
            self.fweight = 'normal'
        else:
            self.fweight = format_options['fontweight']

        if 'dpi' not in format_options:
            self.dpi = 100
        else:
            self.dpi = format_options['dpi']

        # Directory where longitudinal_plots will be stored
        fig_folder(self.dirname)

        # Ticksize
        self.tsize = self.lsize - 2

        # Set size of x- and y-grid numbers
        plt.rc('xtick', labelsize=self.tsize)
        plt.rc('ytick', labelsize=self.tsize)

        # Set x- and y-grid labelsize and weight
        plt.rc('axes', labelsize=self.lsize)
        plt.rc('axes', labelweight=self.fweight)

        # Set linewidth for continuous, markersize for discrete plotting
        plt.rc('lines', linewidth=self.lwidth, markersize=self.msize)

        # Set figure resolution, font
        plt.rc('figure', dpi=self.dpi)
        plt.rc('figure', autolayout = True)
        plt.rc('savefig', dpi=self.dpi)
        plt.rc('savefig', bbox='tight')
        plt.rc('savefig', pad_inches = 0.1)
        plt.rc('font', family=self.ffamily)

    def track(self):
        '''
        Plot in certain time steps and depending on imported objects
        '''

        # Snapshot-type plots
        if (self.tstep[0] % self.dt_plot) == 0:
            
            plot_long_phase_space(self.general_params, self.rf_params, 
                                  self.beam, self.xmin, self.xmax, self.ymin, 
                                  self.ymax, self.xunit, 
                                  sampling = self.sampling, 
                                  separatrix_plot = self.separatix, 
                                  histograms_plot = self.histogram, 
                                  dirname = self.dirname, show_plot=self.show_plt,
                                  alpha = self.alpha)
            
            if self.profile:
                plot_beam_profile(self.profile, self.tstep[0], 
                                  style = self.lstyle, dirname = self.dirname, show_plot = self.show_plt)

                self.profile.beam_spectrum_freq_generation(
                    self.profile.n_slices)
                self.profile.beam_spectrum_generation(self.profile.n_slices)
                plot_beam_spectrum(self.profile, self.tstep[0], 
                                   style = self.lstyle, dirname = self.dirname, show_plot = self.show_plt)
        
        # Plots as a function of time        
        if (self.tstep[0] % self.dt_bckp) == 0 and self.h5file:

            h5data = hp.File(self.h5file + '.h5', 'r')
            plot_bunch_length_evol(self.rf_params, h5data, 
                                   output_freq = self.dt_mon, 
                                   dirname = self.dirname, show_plot=self.show_plt)
            
            if self.profile and self.profile.fit_option == 'gaussian':
                    plot_bunch_length_evol_gaussian(self.rf_params, self.profile,
                                                    h5data, 
                                                    output_freq = self.dt_mon, 
                                                    dirname = self.dirname, show_plot=self.show_plt)
            plot_position_evol(self.rf_params, h5data, 
                               output_freq = self.dt_mon, 
                               style = self.lstyle, dirname = self.dirname, show_plot=self.show_plt)
            plot_energy_evol(self.rf_params, h5data, output_freq = self.dt_mon, 
                             style = self.lstyle, dirname = self.dirname,show_plot=self.show_plt)
            plot_COM_motion(self.general_params, self.rf_params, h5data,
                            output_freq = self.dt_mon, dirname = self.dirname, show_plot=self.show_plt)
            plot_transmitted_particles(self.rf_params, h5data, 
                                       output_freq = self.dt_mon, 
                                       style = self.lstyle, 
                                       dirname = self.dirname, show_plot=self.show_plt)
                    
            if self.PL:
                plot_PL_RF_freq(self.rf_params, h5data, 
                                output_freq = self.dt_mon,
                                dirname = self.dirname, show_plot=self.show_plt)
                plot_PL_RF_phase(self.rf_params, h5data, 
                                 output_freq = self.dt_mon,
                                 dirname = self.dirname, show_plot=self.show_plt)
                plot_PL_bunch_phase(self.rf_params, h5data, 
                                    output_freq = self.dt_mon, 
                                    dirname = self.dirname, show_plot=self.show_plt)
                plot_PL_phase_corr(self.rf_params, h5data, 
                                   output_freq = self.dt_mon, 
                                   dirname = self.dirname,show_plot=self.show_plt)
                plot_PL_freq_corr(self.rf_params, h5data, 
                                  output_freq = self.dt_mon, 
                                  dirname = self.dirname,show_plot=self.show_plt)
                plot_RF_phase_error(self.rf_params, h5data, 
                                    output_freq = self.dt_mon, 
                                    dirname = self.dirname,show_plot=self.show_plt)
                plot_RL_radial_error(self.rf_params, h5data, 
                                     output_freq = self.dt_mon, 
                                     dirname = self.dirname,show_plot=self.show_plt)
            
            if self.noiseFB:
                plot_LHCNoiseFB(self.rf_params, self.noiseFB, h5data, 
                                output_freq = self.dt_mon, 
                                dirname = self.dirname, show_plot=self.show_plt)
                plot_LHCNoiseFB_FWHM(self.rf_params, self.noiseFB, h5data, 
                                     output_freq = self.dt_mon, 
                                     dirname = self.dirname, show_plot=self.show_plt)
            
                if self.noiseFB.bl_meas_bbb != None:
                    plot_LHCNoiseFB_FWHM_bbb(self.rf_params, self.noiseFB, 
                                             h5data, output_freq = self.dt_mon, 
                                             dirname = self.dirname, show_plot=self.show_plt)

            h5data.close()

    def reset_frame(self, xmin, xmax, ymin, ymax):

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
