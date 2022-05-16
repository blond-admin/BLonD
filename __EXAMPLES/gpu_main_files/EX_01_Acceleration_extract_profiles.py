
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Example input for simulation of acceleration
No intensity effects

:Authors: **Helga Timko**
'''
#  General Imports
from __future__ import division, print_function
from builtins import range
import numpy as np
import os
import time

try:
    from pyprof import timing
    from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing
#  BLonD Imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import CutOptions, FitOptions, Profile
from blond.monitors.monitors import MultiBunchMonitor, BunchMonitor
from blond.plots.plot import Plot
from blond.utils.input_parser import parse
from blond.utils import bmath as bm
from blond.utils.mpi_config import worker, mpiprint

this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'

try:
    os.mkdir(this_directory + '../output_files')
except:
    pass
try:
    os.mkdir(this_directory + '../output_files/EX_01_fig')
except:
    pass

# Simulation parameters -------------------------------------------------------
# Bunch parameters
N_b = 1e9           # Intensity
N_p = 50000         # Macro-particles
tau_0 = 0.4e-9          # Initial bunch length, 4 sigma [s]

# Machine and RF parameters
C = 26658.883        # Machine circumference [m]
p_i = 450e9         # Synchronous momentum [eV/c]
p_f = 460.005e9      # Synchronous momentum, final
h = 35640            # Harmonic number
V = 6e6                # RF voltage [V]
dphi = 0             # Phase modulation/offset
gamma_t = 55.759505  # Transition gamma
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor

# Tracking details
N_t = 2000           # Number of turns to track
n_iterations = N_t
dt_plt = 200         # Time steps between plots
n_slices = 100
n_turns_reduce = 1
seed = 0

args = parse()

n_iterations = n_iterations if args['turns'] == None else args['turns']
N_p = N_p if args['particles'] == None else args['particles']
n_bunches = 1
n_turns_reduce = n_turns_reduce if args['reduce'] == None else args['reduce']
seed = seed if args['seed'] == None else args['seed']
approx = args['approx']
timing.mode = args['time']
os.environ['OMP_NUM_THREADS'] = str(args['omp'])
withtp = bool(args['withtp'])
precision = args['precision']

if args['monitor']:
    split_args = args['monitor'].split(',')
    monitor_interval = split_args[0]
    monitor_interval = int(monitor_interval) if monitor_interval else 0
    assert (monitor_interval >= 0)

    if len(split_args) > 1:
        monitor_firstturn = int(split_args[1])
        monitor_firstturn = monitor_firstturn if monitor_firstturn >= 0 else n_iterations+monitor_firstturn
    else:
        monitor_firstturn = 0
    if len(split_args) > 2:
        monitor_lastturn = int(split_args[2])
        monitor_lastturn = monitor_lastturn if monitor_lastturn >= 0 else n_iterations+monitor_lastturn+1
    else:
        monitor_lastturn = n_iterations

bm.use_precision(precision)
bm.use_mpi()
bm.use_fftw()

worker.assignGPUs(num_gpus=args['gpu'])


worker.greet()
if worker.isMaster:
    worker.print_version()
    # os.system("gcc --version")


mpiprint(args)
# Simulation setup ------------------------------------------------------------
mpiprint("Setting up the simulation...")
mpiprint("")


# Define general parameters
ring = Ring(C, alpha, np.linspace(p_i, p_f, N_t+1), Proton(), N_t)

# Define beam and distribution
beam = Beam(ring, N_p, N_b)


# Define RF station parameters and corresponding tracker
rf = RFStation(ring, [h], [V], [dphi])
long_tracker = RingAndRFTracker(rf, beam)


bigaussian(ring, rf, beam, tau_0/4, reinsertion=True, seed=seed)


# Need slices for the Gaussian fit
profile = Profile(beam, CutOptions(n_slices=n_slices),
                  FitOptions(fit_option='gaussian'))

# Define what to save in file
bunchmonitor = BunchMonitor(ring, rf, beam,
                            this_directory + '../output_files/EX_01_output_data', Profile=profile)

format_options = {'dirname': this_directory + '../output_files/EX_01_fig'}
plots = Plot(ring, rf, beam, dt_plt, N_t, 0, 0.0001763*h,
             -400e6, 400e6, xunit='rad', separatrix_plot=True,
             Profile=profile, h5file=this_directory + '../output_files/EX_01_output_data',
             format_options=format_options)

# For testing purposes
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'mean_dE', 'std_dE', 'mean_dt', 'std_dt')
test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))


# For the GPU version we disable bunchmonitor and plots, since they
# make the simulation much slower

# Accelerator map
map_ = [long_tracker] + [profile]
# + [bunchmonitor] + [plots]
mpiprint("Map set")
mpiprint("")

# This is the way to enable the GPU
# GPU = 1
# if GPU:
if worker.hasGPU:
    bm.use_gpu(gpu_id=worker.gpu_id)    
    beam.use_gpu()
    long_tracker.use_gpu()
    profile.use_gpu()
    if args['gpucache'] == 1:
        bm.enable_gpucache()
print(f'Glob rank: [{worker.rank}], Node rank: [{worker.noderank}], Intra rank: [{worker.intrarank}], GPU rank: [{worker.gpucommrank}], hasGPU: {worker.hasGPU}')


if args['monitor'] and monitor_interval > 0 and worker.isMaster:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'monitorfiles/ex01-t{}-p{}-b{}-sl{}-approx{}-prec{}-r{}-m{}-se{}-w{}'.format(
            n_iterations, N_p, n_bunches, n_slices, approx, args['precision'],
            n_turns_reduce, args['monitor'], seed, worker.workers)
    multiBunchMonitor = MultiBunchMonitor(filename=filename,
                                          n_turns=np.ceil(
                                              (monitor_lastturn-monitor_firstturn) /
                                              monitor_interval),
                                      profile=profile,
                                      rf=rf,
                                      Nbunches=n_bunches)


mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('dt mean: ', np.mean(beam.dE))
mpiprint('dt std: ', np.std(beam.dE))
beam.split(random=True)

worker.sync()
timing.reset()
start_t = time.time()
# Tracking --------------------------------------------------------------------
for turn in range(n_iterations):

    # Track
    # for m in map_:
        # m.track()
    # Update profile
    if (approx == 0):
        profile._slice()
        # worker.sync()
        profile.reduce_histo()
    elif (approx == 1) and (turn % n_turns_reduce == 0):
        profile._slice()
        # worker.sync()
        profile.reduce_histo()
    elif (approx == 2):
        profile._slice()
        profile.scale_histo()

    long_tracker.track()

    if (args['monitor'] and monitor_interval > 0) and \
            (turn >= monitor_firstturn and turn < monitor_lastturn) and \
            (turn % monitor_interval == 0):
        beam.statistics()
        beam.gather_statistics()
        profile.fwhm_multibunch(n_bunches, 1, rf.t_rf[0, turn], bucket_tolerance=0,
                                shiftX=0)
        if worker.isMaster:
            multiBunchMonitor.track(turn)

    # Define losses according to separatrix and/or longitudinal position
    # beam.losses_separatrix(ring, rf)
    # beam.losses_longitudinal_cut(0., 2.5e-9)

# For testing purposes
# test_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
#     np.mean(beam.dE), np.std(beam.dE), np.mean(beam.dt), np.std(beam.dt))
# with open(this_directory + '../output_files/EX_01_test_data.txt', 'w') as f:
#     f.write(test_string)

beam.gather()
end_t = time.time()

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(worker.rank))

worker.finalize()

if args['monitor'] and monitor_interval > 0:
    multiBunchMonitor.close()

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('dt mean: ', np.mean(beam.dt))
mpiprint('dt std: ', np.std(beam.dt))

# mpiprint('dt mean, 1st bunch: ', np.mean(beam.dt[:n_particles]))
# mpiprint('shift ', rf.phi_rf[0, turn]/rf.omega_rf[0, turn])

mpiprint('profile sum: ', np.sum(profile.n_macroparticles))
mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))

mpiprint('Done!')