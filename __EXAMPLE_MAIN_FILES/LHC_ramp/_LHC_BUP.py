# Example input for longitudinal simulation with phase loop and phase noise
# For LHC controlled emittance blow-up during the acceleration ramp
# No intensity effects
#
# Run first:
# Preprocess_ramp.py
# Preprocess_LHC_noise.py
#
# H. Timko


import time 
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys

from input_parameters.general_parameters import *
from input_parameters.preprocess import *
from input_parameters.rf_parameters import *
from trackers.tracker import *
from llrf.feedbacks import *
from llrf.RF_noise import *
from beams.beams import *
from beams.distributions import *
from beams.slices import *
from monitors.monitors import *
from plots.plot_settings import *
from plots.plot_beams import *
from plots.plot_llrf import *
from plots.plot_slices import *



# Simulation parameters --------------------------------------------------------
# Bunch parameters
N_b = 1.2e9          # Intensity
N_p = 50001          # Macro-particles
tau_0 = 1.2          # Initial bunch length, 4 sigma [ns]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
h = 35640            # Harmonic number
dphi = 0.            # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 8700001        # Number of turns to track; full ramp: 8700001
dt_plt = 43500       # Time steps between plots
dt_mon = 1           # Time steps between monitoring
dt_save = 500000     # Time steps between saving coordinates

# RF noise injection
noise_inj = 'PL'     # Through CCL or PL
bl_target = 1.2e-9   # 4 sigma r.m.s. target bunch length in [s]


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Set up plot formatting
PlotSettings().set_plot_format()

# Import pre-processed momentum and voltage for the acceleration ramp
ps = np.loadtxt('LHC_momentum_programme.dat', unpack=True)
V = np.loadtxt('LHC_voltage_programme.dat', unpack=True)
print "Momentum and voltage loaded..."

# Import pre-processed phase noise
dphi = np.loadtxt('LHC_noise_programme.dat', unpack=True)
print "Phase noise of %.4e rad (%.4e deg) r.m.s loaded..." \
    %(np.std(dphi), np.std(dphi)*180/np.pi)

# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, ps[0:N_t+1], 'proton')
print "General parameters set..."

# Define emittance BUP feedback
noiseFB = LHCNoiseFB(general_params, bl_target, gain=1.5e9, factor=0.8)


if noise_inj == 'PL':
    
    # Define RF station parameters, phase loop, and corresponding tracker
    PL_gain = 1./(5.*general_params.t_rev[0])
    print "   PL gain is %.4e 1/s for initial turn T0 = %.4e s" %(PL_gain, 
        general_params.t_rev[0])

    # No noise in the cavity, but injected directly in the PL
    RF_params = RFSectionParameters(general_params, 1, h, V[0:N_t+1], 0.)

    LHCnoise = LHCFlatSpectrum(general_params, RF_params, 1000000)
    LHCnoise.dphi = dphi[0:N_t+1]

    PL = PhaseLoop(general_params, RF_params, PL_gain, machine = 'LHC', 
                   RFnoise=LHCnoise, FB=noiseFB)
    long_tracker = RingAndRFSection(RF_params, PhaseLoop=PL)
    
elif noise_inj == 'CCL':
    
    # Injecting noise in the cavity, PL off (or otherwise need symmetric fill)
    # Note: Noise needs to be injected on an f_rev side-band for multi-bunch!
    RF_params = RFSectionParameters(general_params, 1, h, V[0:N_t+1], 0.)

    LHCnoise = LHCFlatSpectrum(general_params, RF_params, 1000000)
    LHCnoise.dphi = dphi[0:N_t+1]
    
    RF_params = RFSectionParameters(general_params, 1, h, V[0:N_t+1], 
                                    LHCnoise.dphi[0:N_t+1])
    long_tracker = RingAndRFSection(RF_params)
    
else:
    
    raise RuntimeError('ERROR: RF phase noise injection scheme not recognised!')

print "General and RF parameters set..."

 
# Define beam and distribution: Load matched, filamented distribution
beam = Beam(general_params, N_p, N_b)
beam.theta, beam.dE = np.loadtxt('initial_long_distr.dat', unpack=True)


# Slices required for statistics (needed in PL) and for the Gaussian fit
slice_beam = Slices(beam, 100, cut_left = 0., cut_right = 0.0001763, 
                    cuts_coord = 'theta', slicing_coord='theta', 
                    fit_option='gaussian')
 
# Define what to save in file
if noise_inj == 'PL':
    bunchmonitor = BunchMonitor('output_data', N_t+1,
                                slice_beam, PL, noiseFB)
elif noise_inj == 'CCL':
    bunchmonitor = BunchMonitor('output_data', N_t+1, 
                                slices=slice_beam, LHCNoiseFB=noiseFB)

# Initial losses, slicing, statistics
beam.losses_separatrix(general_params, RF_params)
beam.losses_cut(0, 0.0001763)      
slice_beam.track(beam)
bunchmonitor.track(beam)
slice_beam.track(beam)
print "Statistics set..."

# Plot initial distribution
plot_long_phase_space(beam, general_params, RF_params, 0, 0.0001763, 
                      -450, 450, separatrix_plot = True)
plot_beam_profile(0, general_params, slice_beam)


# Accelerator map
map_ = [slice_beam] + [long_tracker] + [bunchmonitor]
print "Map set"

print "Initial mean bunch position %.4e rad" %(beam.mean_theta)
print "Initial four-times r.m.s. bunch length %.4e rad" %(4.*beam.sigma_theta)
print "Initial Gaussian bunch length %.4e rad" %beam.bl_gauss

print "Ready for tracking!"
print ""


# Tracking ---------------------------------------------------------------------
for i in range(N_t):
    t0 = time.clock()

    # Remove lost particles to obtain a correct r.m.s. value
    if (i % 10) == 0: # reduce computational costs
        beam.losses_separatrix(general_params, RF_params)
        beam.losses_cut(0, 0.0001763)

    # Track
    for m in map_:
        m.track(beam)

    # Feedback on BUP every 3 seconds
    if (i % 33740) == 0:
        print "Adjusting noise amplitude in time step %d" %i
        if noise_inj == 'PL':
            
            print "    Before FB, first phase noise point was %.4e" \
                %(noiseFB.x*LHCnoise.dphi[0])
            noiseFB.FB(RF_params, beam, LHCnoise, slices=slice_beam)
            print "    After FB, first phase noise point is %.4e" \
                %(noiseFB.x*LHCnoise.dphi[0])
                
        elif noise_inj == 'CCL':
            
            print "    Before FB, first phase noise point was %.4e" \
                %long_tracker.phi_offset[0,0]
            noiseFB.FB(RF_params, beam, LHCnoise, slices=slice_beam, CCL=True)
            print "    After FB, first phase noise point is %.4e" \
                %long_tracker.phi_offset[0,0]
            
    # Plots and outputting
    if (i % dt_plt) == 0:
        
        print "Outputting at time step %d, tracking time %.4e s..." %(i, t0)
        print "RF tracker counter is %d" %RF_params.counter[0]
        print "   Beam momentum %.6e eV" %beam.momentum
        print "   Beam gamma %3.3f" %beam.gamma_r
        print "   Beam beta %3.3f" %beam.beta_r
        print "   Beam energy %.6e eV" %beam.energy
        print "   Mean bunch position %.4e rad" %(beam.mean_theta)
        print "   Four-times r.m.s. bunch length %.4e rad" %(4.*beam.sigma_theta)
        print "   Gaussian bunch length %.4e rad" %beam.bl_gauss
        print ""
        sys.stdout.flush()

        plot_long_phase_space(beam, general_params, RF_params, 0, 0.0001763, 
                              -450, 450, separatrix_plot = True)
        plot_beam_profile(i, general_params, slice_beam)

 
    # Save phase space data
    if (i % dt_save) == 0:
        np.savetxt('coords_' "%d" %RF_params.counter[0] + '.dat', 
                   np.c_[beam.theta, beam.dE, beam.id], fmt='%.10e')

plot_COM_motion(general_params, RF_params, 'output_data', 
                0, 0.0001763, -100, 100, separatrix_plot = False)

plot_bunch_length_evol('output_data', general_params, i, 
                       output_freq=dt_mon, unit='ns')
plot_bunch_length_evol_gaussian('output_data', general_params, 
                                slice_beam, i, output_freq=dt_mon, unit='ns')
plot_position_evol('output_data', general_params, i,
                   output_freq=dt_mon, unit = None, style = '.') 
if noise_inj == 'PL':
    plot_PL_phase_corr(PL, 'output_data', i, output_freq=dt_mon)
    plot_PL_freq_corr(PL, 'output_data', i, output_freq=dt_mon)
plot_LHCNoiseFB(noiseFB, 'output_data', i, output_freq=dt_mon)
 
 
bunchmonitor.h5file.close()
print "Done!"
print ""

