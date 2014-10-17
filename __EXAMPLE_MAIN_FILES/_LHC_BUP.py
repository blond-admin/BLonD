# Example input for longitudinal simulation with phase loop and phase noise
# For LHC controlled emittance blow-up during the acceleration ramp
# No intensity effects
# H. Timko


import time 
import numpy as np

from input_parameters.general_parameters import *
from input_parameters.preprocess import *
from input_parameters.rf_parameters import *
from trackers.longitudinal_tracker import *
from llrf.feedbacks import *
from llrf.RF_noise import *
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
N_t = 10001         # Number of turns to track; full ramp: 8700001
dt_plt = 1000        # Time steps between plots
dt_mon = 1           # Time steps between monitoring
dt_save = 10000      # Time steps between saving coordinates

# RF noise injection
noise_inj = 'CCL'     # Through CCL or PL
bl_target = 1.3      # 4 sigma r.m.s. target bunch length in [ns]


# Simulation setup -------------------------------------------------------------
print "Setting up the simulation..."
print ""


# Set up plot formatting
PlotSettings().set_plot_format()

# Import pre-processed momentum and voltage for the acceleration ramp
ps = np.loadtxt('LHC_momentum_programme.dat', unpack=True)
V = np.loadtxt('LHC_voltage_programme.dat', unpack=True)
print "Momentum and voltage loaded..."

# Generate phase noise
f = np.arange(0, 5.6227612455e+03, 1.12455000e-02)
spectrum = np.concatenate((1.11100000e-07 * np.ones(4980), np.zeros(495021)))
RFnoise = PhaseNoise(f, spectrum, seed1=1234, seed2=7564)
RFnoise.spectrum_to_phase_noise()
print "RF phase noise generated"
plot_noise_spectrum(f, spectrum, sampling=100)
plot_phase_noise(RFnoise.t, RFnoise.dphi, sampling=100)
print "   Sigma of RF noise is %.4e" %np.std(RFnoise.dphi)
print "   Time step of RF noise is %.4e" %RFnoise.t[1]
print ""


# Define general parameters
general_params = GeneralParameters(N_t, C, alpha, ps[0:N_t+1], 'proton')
print "General parameters set..."

# Define RF station parameters, phase loop, and corresponding tracker
PL_gain = 1./(5.*general_params.t_rev[0])
print "   PL gain is %.4e 1/s for initial turn T0 = %.4e s" %(PL_gain, 
    general_params.t_rev[0])

if noise_inj == 'PL':
    
    # No noise in the cavity, but injected directly in the PL
    RF_params = RFSectionParameters(general_params, 1, h, V[0:N_t+1], dphi)
    PL = PhaseLoop(general_params, RF_params, PL_gain, machine = 'LHC', 
                   RF_noise=RFnoise)
    long_tracker = RingAndRFSection(RF_params, PhaseLoop=PL)
    
elif noise_inj == 'CCL':
    
    # Injecting noise in the cavity, PL only corrects
    # Note: Noise needs to be injected on an f_rev side-band!
    RF_params = RFSectionParameters(general_params, 1, h, V[0:N_t+1], RFnoise.dphi[0:N_t+1])
    PL = PhaseLoop(general_params, RF_params, PL_gain, machine = 'LHC')
    long_tracker = RingAndRFSection(RF_params, PhaseLoop=PL)
    
else:
    
    raise RuntimeError('ERROR: RF phase noise injection scheme not recognised!')

print "General and RF parameters set..."

# Define emittance BUP feedback
noiseFB = LHCNoiseFB(general_params, bl_target, gain=0.1, 
                     factor=0.64, sampling_frequency=100)

 
# Define beam and distribution: Load matched, filamented distribution
beam = Beam(general_params, N_p, N_b)
beam.theta, beam.dE = np.loadtxt('initial_long_distr.dat', unpack=True)


# Slices required for statistics (needed in PL) and for the Gaussian fit
slice_beam = Slices(beam, 100, cut_left = 0., cut_right = 0.0001763, 
                    cuts_coord = 'theta', slicing_coord='theta', 
                    mode='const_space_hist',
                    fit_option='gaussian')
 
# Define what to save in file
bunchmonitor = BunchMonitor('output_data', N_t+1, "Longitudinal",
                            slices=slice_beam, PhaseLoop=PL, LHCNoiseFB=noiseFB)

# Initial losses, slicing, statistics
beam.losses_separatrix(general_params, RF_params)
beam.losses_longitudinal_cut(0, 0.0001763)      
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
        beam.losses_longitudinal_cut(0, 0.0001763)

    # Track
    for m in map_:
        m.track(beam)

    # Feedback on BUP
    noiseFB.FB(RF_params, beam, RFnoise)
        
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
        print beam.id
        print ""


        plot_long_phase_space(beam, general_params, RF_params, 0, 0.0001763, 
                              -450, 450, separatrix_plot = True)
        plot_beam_profile(i, general_params, slice_beam)
        plot_COM_motion(beam, general_params, RF_params, 'output_data', 
                        0, 0.0001763, -100, 100, separatrix_plot = False)

        plot_bunch_length_evol(beam, 'output_data', general_params, i, 
                               output_freq=dt_mon, unit='ns')
        plot_bunch_length_evol_gaussian(beam, 'output_data', general_params, 
                                        slice_beam, i, output_freq=dt_mon, unit='ns')
        plot_position_evol(beam, 'output_data', general_params, i,
                           output_freq=dt_mon, unit = None, style = '.') 
        plot_PL_phase_corr(PL, 'output_data', i, output_freq=dt_mon)
        plot_PL_freq_corr(PL, 'output_data', i, output_freq=dt_mon)
        plot_LHCNoiseFB(noiseFB, 'output_data', i, output_freq=dt_mon)
 
    # Save phase space data
    if (i % dt_save) == 0:
        np.savetxt('coords_' "%d" %RF_params.counter[0] + '.dat', 
                   np.c_[beam.theta, beam.dE, beam.id], fmt='%.10e')

 
 
bunchmonitor.h5file.close()
print "Done!"
print ""

