# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:11:42 2018

@author: schwarz, kiliakis
"""

import numpy as np
import os
try:
    from pyprof import timing
    # from pyprof import mpiprof
except ImportError:
    from blond.utils import profile_mock as timing
    mpiprof = timing

import time
from scipy.constants import c

from SPSimpedanceModel.impedance_scenario import scenario, impedance2blond
from impedance_reduction_dir.impedance_reduction import ImpedanceReduction

# BLonD imports
from blond.beam.distributions import matched_from_distribution_function
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.impedance_sources import TravelingWaveCavity
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.llrf.beam_feedback import BeamFeedback
from blond.utils.input_parser import parse
from blond.monitors.monitors import MultiBunchMonitor
from blond.utils.mpi_config import worker, mpiprint
from blond.utils import bmath as bm


this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'


# --- Simulation parameters -------------------------------------

# where and at what turns to save, only relevant if SAVE_DATA is True
save_folder = this_directory + '/../output_files/lossInSimulation_scanSimParam/'
save_turn_fine = 5
save_turn_coarse = 1

# bunch parameters
BUNCHLENGTH_MODULATION = False
if BUNCHLENGTH_MODULATION is False:
    PS_case = 'rms13.0ns_full15ns'
else:
    PS_case = 'blMod'

optics = 'Q22'
intensity_pb = 1.7e11
V1 = 2.0e6
n_bunches = 72
bunch_shift = 0  # how many degrees to displace the bunch [deg]

INTENSITY_MODULATION = False
# number of turns to simulate in the SPS
n_turns = 43348+1  # 1s

# impedance & LLRF parameters
SPS_IMPEDANCE = True
impedance_model_str = 'present'  # present or future
cavities = impedance_model_str

SPS_PHASELOOP = True
PLrange = 5*12
PL_2ndLoop = 'F_Loop'

FB_strength = 'present'

# simulation parameters
n_particles = int(4e6)  # 4M macroparticles per bunch
n_bins_rf = 256  # number of slices per RF-bucket
nFrev = 2  # multiples of f_rev for frequency resolution

n_iterations = n_turns
n_turns_reduce = 1
seed = 0
args = parse()


n_iterations = n_iterations if args['turns'] == None else args['turns']
n_particles = n_particles if args['particles'] == None else args['particles']
n_bunches = n_bunches if args['bunches'] == None else args['bunches']
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
#bm.use_fftw()

worker.assignGPUs(num_gpus=args['gpu'])

worker.greet()
if worker.isMaster:
    worker.print_version()

worker.initLog(bool(args['log']), args['logdir'])
#worker.initTrace(bool(args['trace']), args['tracefile'])
worker.taskparallelism = withtp

mpiprint(args)


# initialize simulation

np.random.seed(seed)

SAVE_DATA = False

FB_model = 'ImpRed'
reduction_model = 'filter'  # reduce via a 'filter' or R_S of the 200 MHz'cavity
filter_type = 'fb_reduction'  # apply 'fb_reduction' or 'none' filter

save_folder += optics+'/'+FB_model+'/'+PS_case+'/'

if n_turns == 43348+1:
    save_folder += '1s_sim/'
else:
    save_folder += str(n_turns)+'turns/'

case = 'int'+str(intensity_pb/1e11)+'_'+'V' + \
    str(V1/1e6)+'_'+str(n_bunches)+'Nbun'

if bunch_shift != 0:
    case += '_'+str(bunch_shift)+'deg'

if SPS_IMPEDANCE is True:
    case += '_'+impedance_model_str+'Imp'

if SPS_PHASELOOP is True:
    case += '_PL2'+str(int(PLrange/5))+'b'+PL_2ndLoop[0]

case += '_'+FB_strength+'FBstr'

case += '_seed'+str(seed) + '_'+str(n_particles/1e6)+'Mmppb'
case += '_'+str(n_bins_rf)+'binRF_' + str(nFrev)+'fRes'
mpiprint('simulating case: '+case)
mpiprint('saving in: '+save_folder)
save_file_name = case + '_data'

if INTENSITY_MODULATION:
    save_folder += 'intMod/'
if BUNCHLENGTH_MODULATION:
    save_folder += 'blMod/'

# SPS --- Ring Parameters -------------------------------------------

bunch_spacing = 5   # how many SPS RF buckets between bunches in the SPS
# 5*t_rf_SPS spacing = 5*5ns = 25ns

intensity = n_bunches * intensity_pb     # total intensity SPS

# Ring parameters SPS
circumference = 6911.5038  # Machine circumference [m]
sync_momentum = 25.92e9  # SPS momentum at injection [eV/c]

if optics == 'Q20':
    gamma_transition = 17.95142852  # Q20 Transition gamma
elif optics == 'Q22':
    gamma_transition = 20.071  # Q22 Transition gamma
else:
    raise RuntimeError('No gamma_transition specified')
momentum_compaction = 1./gamma_transition**2  # Momentum compaction array

ring = Ring(circumference, momentum_compaction, sync_momentum, Proton(),
            n_turns=n_turns)
tRev = ring.t_rev[0]

# RF parameters SPS
n_rf_systems = 1    # Number of rf systems

if n_rf_systems == 2:
    V2_ratio = 1.00e-01     # voltage 800 MHz  [V]
    harmonic_numbers = [4620, 18480]             # Harmonic numbers
    voltage = [V1, V1*V2_ratio]
    phi_offsets = [0, np.pi]
elif n_rf_systems == 1:
    harmonic_numbers = 4620  # Harmonic numbers
    voltage = V1
    phi_offsets = 0

rf_station = RFStation(ring, harmonic_numbers, voltage, phi_offsets,
                       n_rf=n_rf_systems)
t_rf = rf_station.t_rf[0, 0]

# calculate fs in case of two RF systems
if n_rf_systems == 2:
    h = rf_station.harmonic[0, 0]
    omega0 = ring.omega_rev[0]
    phiS = rf_station.phi_s[0]
    eta = rf_station.eta_0[0]
    V0 = rf_station.voltage[0, 0]
    beta0 = ring.beta[0, 0]
    E0 = ring.energy[0, 0]
    omegaS0 = np.sqrt(-h*omega0**2*eta*np.cos(phiS)*V0/(2*np.pi*beta0**2*E0))

    nh = rf_station.harmonic[1, 0] / rf_station.harmonic[0, 0]
    V2 = rf_station.voltage[1, 0]
    phi2 = rf_station.phi_offset[1, 0]
    omegaS = omegaS0 * \
        np.sqrt(1 + nh*V2*np.cos(nh*phiS+phi2) / (V0*np.cos(phiS)))
    fs = omegaS/(2*np.pi)
elif n_rf_systems == 1:
    fs = rf_station.omega_s0[0]/(2*np.pi)


# --- PS beam --------------------------------------------------------
n_macroparticles = n_bunches * n_particles
beam = Beam(ring, n_macroparticles, intensity)

# PS_n_bunches = 1

n_shift = 0  # how many rf-buckets to shift beam

# PS_folder = this_directory +'/../input_files/'


# SPS --- Profile -------------------------------------------
mpiprint('Setting up profile')

profile_margin = 20 * t_rf

t_batch_begin = n_shift * t_rf
t_batch_end = t_rf * (bunch_spacing * (n_bunches-1) + 1+n_shift)

cut_left = t_batch_begin - profile_margin
cut_right = t_batch_end + profile_margin

# number of rf-buckets of the beam
# + rf-buckets before the beam + rf-buckets after the beam
n_slices = n_bins_rf * (bunch_spacing * (n_bunches-1) + 1
                        + int(np.round((t_batch_begin - cut_left)/t_rf))
                        + int(np.round((cut_right - t_batch_end)/t_rf)))

profile = Profile(beam, CutOptions=CutOptions(cut_left=cut_left,
                                              cut_right=cut_right, n_slices=n_slices))


mpiprint('Profile set!')


# SPS --- Impedance and induced voltage ------------------------------
mpiprint('Setting up impedance')

frequency_step = nFrev*ring.f_rev[0]

if SPS_IMPEDANCE == True:

    if impedance_model_str == 'present':
        number_vvsa = 28
        number_vvsb = 36
        shield_vvsa = False
        shield_vvsb = False
        HOM_630_factor = 1
        UPP_factor = 25/25
        new_MKE = True
        BPH_shield = False
        BPH_factor = 1
    elif impedance_model_str == 'future':
        number_vvsa = 28
        number_vvsb = 36
        shield_vvsa = False
        shield_vvsb = False
        HOM_630_factor = 1/3
        UPP_factor = 10/25
        new_MKE = True
        BPH_shield = False
        BPH_factor = 0

    # The main 200MHz impedance is effectively 0.0
    impedance_scenario = scenario(MODEL=impedance_model_str,
                                  Flange_VVSA_R_factor=number_vvsa/31,
                                  Flange_VVSB_R_factor=number_vvsb/33,
                                  HOM_630_R_factor=HOM_630_factor,
                                  HOM_630_Q_factor=HOM_630_factor,
                                  UPP_R_factor=UPP_factor,
                                  Flange_BPHQF_R_factor=BPH_factor,
                                  FB_attenuation=-1000)

    impedance_scenario.importImpedanceSPS(VVSA_shielded=shield_vvsa,
                                          VVSB_shielded=shield_vvsb,
                                          # kickerMario=new_MKE, noMKP=False,
                                          # BPH_shield=BPH_shield
                                          )

    # Convert to formats known to BLonD

    impedance_model = impedance2blond(impedance_scenario.table_impedance)

    # Induced voltage calculated by the 'frequency' method

    SPS_freq = InducedVoltageFreq(beam, profile,
                                  impedance_model.impedanceListToPlot,
                                  frequency_step)

    # # The main 200MHz impedance is effectively 0.0
    # impedance_scenario = scenario(MODEL=impedance_model_str,
    #                               FB_attenuation=-1000)


#    induced_voltage = TotalInducedVoltage(beam, profile, [SPS_freq])

mpiprint('SPS impedance model set!')

R2 = 27.1e3  # series impedance [kOhm/m^2]
vg = 0.0946*c  # group velocity [m/s]
fr = 200.222e6  # resonant frequency [Hz]

if cavities == 'present':

    L_long = 54*0.374  # interaction length [m]
    R_shunt_long = L_long**2*R2/8  # shunt impedance [Ohm]
    damping_time_long = 2*np.pi*L_long/vg*(1+0.0946)
    n_cav_long = 2  # factor 2 because of two cavities are used for tracking

    L_short = 43*0.374  # interaction length [m]
    R_shunt_short = L_short**2*R2/8  # shunt impedance [Ohm]
    damping_time_short = 2*np.pi*L_short/vg*(1+0.0946)
    n_cav_short = 2  # factor 2 because of two cavities are used for tracking
elif cavities == 'future':

    L_long = 43*0.374  # interaction length [m]
    R_shunt_long = L_long**2*R2/8  # shunt impedance [Ohm]
    damping_time_long = 2*np.pi*L_long/vg*(1+0.0946)
    n_cav_long = 2  # factor 2 because of two cavities are used for tracking

    L_short = 32*0.374  # interaction length [m]
    R_shunt_short = L_short**2*R2/8  # shunt impedance [Ohm]
    damping_time_short = 2*np.pi*L_short/vg*(1+0.0946)
    n_cav_short = 4  # factor 4 because of four cavities are used for tracking

longCavity = TravelingWaveCavity(n_cav_long*R_shunt_long, fr,
                                 damping_time_long)
longCavityFreq = InducedVoltageFreq(beam, profile, [longCavity],
                                    frequency_step)
longCavityIntensity = TotalInducedVoltage(beam, profile, [longCavityFreq])

shortCavity = TravelingWaveCavity(n_cav_short*R_shunt_short, fr,
                                  damping_time_short)
shortCavityFreq = InducedVoltageFreq(beam, profile, [shortCavity],
                                     frequency_step)
shortCavityIntensity = TotalInducedVoltage(beam, profile, [shortCavityFreq])

# FB parameters
if FB_strength == 'present':
    FBstrengthLong = 1.05
    FBstrengthShort = 0.73
elif FB_strength == 'future':
    # -26dB
    FBstrengthLong = 1.8
    FBstrengthShort = FBstrengthLong

filter_center_frequency = 200.222e6  # center frequency of filter [Hz]
filter_bandwidth = 2e6  # filter bandwidth [Hz]

# bandwidth should be 3MHz, but then 'reduction' is greater 1...
longCavityImpedanceReduction = ImpedanceReduction(ring, rf_station, longCavityFreq,
                                                  filter_type, filter_center_frequency,
                                                  filter_bandwidth,
                                                  2*tRev, L_long, start_time=2*tRev,
                                                  FB_strength=FBstrengthLong)

shortCavityImpedanceReduction = ImpedanceReduction(ring, rf_station, shortCavityFreq,
                                                   filter_type, filter_center_frequency,
                                                   filter_bandwidth,
                                                   2*tRev, L_short, start_time=2*tRev,
                                                   FB_strength=FBstrengthShort)

if SPS_IMPEDANCE:
    inducedVoltage = TotalInducedVoltage(beam, profile,
                                         [longCavityFreq, shortCavityFreq, SPS_freq])
else:
    inducedVoltage = TotalInducedVoltage(beam, profile,
                                         [longCavityFreq, shortCavityFreq])


# SPS --- Phase Loop Setup -------------------------------------

if SPS_PHASELOOP is True:

    mpiprint('Setting up phase-loop')
    PLgain = 5e3  # [1/s]
    try:
        PLalpha = -1/PLrange / t_rf
        PLoffset = n_shift * t_rf
    except ZeroDivisionError:
        PLalpha = 0.0
        PLoffset = None
    PLdict = {'time_offset': PLoffset, 'PL_gain': PLgain,
              'window_coefficient': PLalpha}
    PL_save_turns = 50
    if PL_2ndLoop == 'R_Loop':
        gain2nd = 5e9
        PLdict['machine'] = 'SPS_RL'
        PLdict['RL_gain'] = gain2nd
    elif PL_2ndLoop == 'F_Loop':
        gain2nd = 0.9e-1
        PLdict['machine'] = 'SPS_F'
        PLdict['FL_gain'] = gain2nd
    phaseLoop = BeamFeedback(ring, rf_station, profile, PLdict)
    beamPosPrev = t_batch_begin + 0.5*t_rf


# SPS --- Tracker Setup ----------------------------------------

mpiprint('Setting up tracker')
tracker = RingAndRFTracker(rf_station, beam, Profile=profile,
                           TotalInducedVoltage=inducedVoltage,
                           interpolation=True)
fulltracker = FullRingAndRF([tracker])


mpiprint('Creating SPS bunch from PS bunch')
# create 72 bunches from PS bunch

beginIndex = 0
endIndex = 0
PS_beam = Beam(ring, n_particles, 0.0)
PS_n_bunches = 1
for copy in range(n_bunches):
    # create binomial distribution;
    # use different seed for different bunches to avoid cloned bunches
    matched_from_distribution_function(PS_beam, fulltracker, seed=seed+copy,
                                       distribution_type='binomial',
                                       distribution_exponent=0.7,
                                       emittance=0.35)

    endIndex = beginIndex + n_particles

    # now place PS bunch at correct position
    beam.dt[beginIndex:endIndex] \
        = PS_beam.dt + copy * t_rf * PS_n_bunches * bunch_spacing

    beam.dE[beginIndex:endIndex] = PS_beam.dE
    beginIndex = endIndex

i = 0
min_dt = (bunch_spacing * i + 0.5) * \
    rf_station.t_rf[0, 0] - 0.9 * rf_station.t_rf[0, 0]
max_dt = (bunch_spacing * i + 0.5) * \
    rf_station.t_rf[0, 0] + 0.9 * rf_station.t_rf[0, 0]

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
# profile.track()
# profile.reduce_histo()
# mpiprint('profile sum: ', np.sum(profile.n_macroparticles))


beam.split(random=True)

# do profile on inital beam


# SPS --- Tracking -------------------------------------
# to save computation time, compute the reduction only for times < 8*FBtime
FBtime = max(longCavityImpedanceReduction.FB_time,
             shortCavityImpedanceReduction.FB_time)/tRev


mpiprint("Ready for tracking!\n")

if args['monitor'] and monitor_interval > 0 and worker.isMaster:
    if args.get('monitorfile', None):
        filename = args['monitorfile']
    else:
        filename = 'monitorfiles/sps-t{}-p{}-b{}-sl{}-approx{}-prec{}-r{}-m{}-se{}-w{}'.format(
            n_iterations, n_particles, n_bunches, n_slices, approx, args['precision'],
            n_turns_reduce, args['monitor'], seed, worker.workers)
    slicesMonitor = MultiBunchMonitor(filename=filename,
                                      n_turns=np.ceil(
                                              (monitor_lastturn-monitor_firstturn) /
                                              monitor_interval),
                                      profile=profile,
                                      rf=rf_station,
                                      Nbunches=n_bunches)


if args['gpu'] > 0:
    bm.use_gpu(gpu_id=worker.gpu_id)
    profile.use_gpu()
    tracker.use_gpu()
    phaseLoop.use_gpu()

print(f'Glob rank: [{worker.rank}], Node rank: [{worker.noderank}], GPU rank: [{worker.gpucommrank}], hasGPU: {worker.hasGPU}')


#worker.initDLB(args['loadbalance'], n_iterations)

delta = 0
worker.sync()
timing.reset()
start_t = time.time()
# import cuprof.cuprof as cp

# cp.enable()

# with cp.region_timer('main_loop'):
for turn in range(n_iterations):

    if ring.n_turns <= 450 and turn % 10 == 0:
        mpiprint('turn: '+str(turn))
    elif turn % 1000 == 0:
        mpiprint('turn: '+str(turn))
    
    profile.track()
    if (turn < 8*int(FBtime)):
        longCavityImpedanceReduction.track()
        shortCavityImpedanceReduction.track()
    
    inducedVoltage.induced_voltage_sum()
    tracker.track() 
    
    if SPS_PHASELOOP is True:
        if turn % PL_save_turns == 0 and turn > 0:

            # present beam position
            beamPosFromPhase = (phaseLoop.phi_beam - rf_station.phi_rf[0, turn])\
                / rf_station.omega_rf[0, turn] + t_batch_begin
            # how much to shift the bin_centers
            delta = beamPosPrev - beamPosFromPhase
            beamPosPrev = beamPosFromPhase

            profile.bin_centers -= delta
            profile.cut_left -= delta
            profile.cut_right -= delta
            profile.edges -= delta

            # shift time_offset of phase loop as well, so that it starts at correct
            # bin_center corresponding to time_offset
            if phaseLoop.alpha != 0:
                phaseLoop.time_offset -= delta

            min_dt -= delta
            max_dt -= delta

    if (args['monitor'] and monitor_interval > 0) and \
            (turn >= monitor_firstturn and turn < monitor_lastturn) and \
            (turn % monitor_interval == 0):
        beam.losses_longitudinal_cut(min_dt, max_dt)
        beam.statistics()
        beam.gather_statistics()
        profile.fwhm_multibunch(n_bunches, bunch_spacing,
                                rf_station.t_rf[0, turn], bucket_tolerance=0)
        # shiftX=rf_station.phi_rf[0, turn]/rf_station.omega_rf[0, turn])

        if worker.isMaster:
            # profile.fwhm()
            slicesMonitor.track(turn)

    #worker.DLB(turn, beam)

# cp.report()
#if False:
beam.gather()

end_t = time.time()
mpiprint('Total time: ', end_t - start_t)

timing.report(total_time=1e3*(end_t-start_t),
              out_dir=args['timedir'],
              out_file='worker-{}.csv'.format(worker.rank))

worker.finalize()

if args['monitor'] and monitor_interval > 0:
    slicesMonitor.close()

mpiprint('dE mean: ', np.mean(beam.dE))
mpiprint('dE std: ', np.std(beam.dE))
mpiprint('profile sum: ', np.sum(profile.n_macroparticles))
mpiprint('profile mean: ', np.mean(profile.n_macroparticles))
mpiprint('profile std: ', np.std(profile.n_macroparticles))
# --- Saving results ----------------------------------------------------

mpiprint('Done!')

