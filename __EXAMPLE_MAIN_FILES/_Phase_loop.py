# Example input for longitudinal simulation with phase loop (LHC)
# No intensity effects


import time 
import numpy as np

from llrf.RF_noise import *
from input_parameters.general_parameters import *
from input_parameters.rf_parameters import *
from trackers.longitudinal_tracker import *
from llrf.feedbacks import *
from beams.beams import *
from beams.longitudinal_distributions import *
from beams.slices import *
from monitors.monitors import *
from longitudinal_plots.plot_settings import *
from longitudinal_plots.plot_beams import *
from longitudinal_plots.plot_llrf import *
from longitudinal_plots.plot_slices import *



# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 10001          # Macro-particles
tau_0 = 0.4          # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_s = 450.e9         # Synchronous momentum [eV]
h = 35640            # Harmonic number
V = 6.e6             # RF voltage [eV]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 1001           # Number of turns to track
dt_plt = 200         # Time steps between plots
dt_mon = 1           # Time steps between monitoring


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""

# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, p_s, 'proton')

# Define RF station parameters, phase loop, and corresponding tracker
RF_params = RFSectionParameters(general_params, 1, h, V, dphi)
PL_gain = 1./(5.*general_params.t_rev[0])
print "PL gain is %.4e 1/s, Trev = %.4e s" %(PL_gain, general_params.t_rev[0])
PL = PhaseLoop(general_params, RF_params, PL_gain, sampling_frequency = 1, 
               machine = 'LHC')
long_tracker = RingAndRFSection(RF_params, PhaseLoop=PL)

print "General and RF parameters set..."


# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
# Generate new distribution
longitudinal_bigaussian(general_params, RF_params, beam, tau_0/4, seed=1234,
                        xunit = 'ns', reinsertion = 'on')
print "Beam set and distribution generated..."


# Need slices for the Gaussian fit; slice for the first plot
slice_beam = Slices(beam, 100, slicing_coord = 'theta')#, #fit_option = 'gaussian', 
                    #slice_immediately = 'on')

# Define what to save in file
bunchmonitor = BunchMonitor('output_data', N_t+1, "Longitudinal", PhaseLoop=PL)#slice_beam, PL)

print "Statistics set..."

# Initialize plots
PlotSettings().set_plot_format()

# Accelerator map
map_ = [slice_beam] + [bunchmonitor] +[long_tracker] 
print "Map set"
print ""


# Initial injection kick/error
beam.theta += 1.e-5
#beam.dE *= -1.
bunchmonitor.track(beam)
slice_beam.track(beam)


# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    t0 = time.clock()
    
   
    # Plot has to be done before tracking (at least for cases with separatrix)
    if (i % dt_plt) == 0:
        print "Outputting at time step %d..." %i
        print "   Beam momentum %.6e eV" %beam.momentum
        print "   Beam gamma %3.3f" %beam.gamma_r
        print "   Beam beta %3.3f" %beam.beta_r
        print "   Beam energy %.6e eV" %beam.energy
        print "   Four-times r.m.s. bunch length %.4e rad" %(4.*beam.sigma_theta)
        #print "   Gaussian bunch length %.4e rad" %beam.bl_gauss
        print ""
        # In plots, you can choose following units: rad, ns, m  
        plot_long_phase_space(beam, general_params, RF_params, 0, 0.0001763, 
                              -450, 450, separatrix_plot = True)
        plot_beam_profile(i, general_params, slice_beam)
        plot_COM_motion(beam, general_params, RF_params, 'output_data', 
                        0.8e-4, 1.1e-4, -75, 75, separatrix_plot = False)


    # Track
    slice_beam.track(beam)
    #slice_beam.mean_theta = beam.theta[0] # For single particle
    long_tracker.track(beam)
    bunchmonitor.track(beam)
        
    # These plots have to be done after the tracking
    if (i % dt_plt) == 0 and i > dt_mon:
        plot_bunch_length_evol(beam, 'output_data', general_params, i, 
                               output_freq=dt_mon, unit='ns')
        #plot_bunch_length_evol_gaussian(beam, 'output_data', general_params, 
        #                                slice_beam, i, output_freq=dt_mon, unit='ns')
        plot_position_evol(beam, 'output_data', general_params, i,
                           output_freq=dt_mon, unit = None, style = '.') 
        plot_PL_phase_corr(PL, 'output_data', i, output_freq=dt_mon)
        plot_PL_freq_corr(PL, 'output_data', i, output_freq=dt_mon)

    
    # Define losses according to separatrix and/or longitudinal position
    #beam.losses_separatrix(general_params, rf_params)
    #beam.losses_longitudinal_cut(0.28e-4, 0.75e-4)


bunchmonitor.h5file.close()
print "Done!"
print ""

