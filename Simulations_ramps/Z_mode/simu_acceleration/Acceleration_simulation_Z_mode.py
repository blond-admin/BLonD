from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from blond.beam.beam import Beam, Electron
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.distributions import bigaussian
from ramp_modules.Ramp_optimiser_functions import HEBee_Eramp_parameters
from ring_parameters.generate_rings import generate_HEB_ring
from scipy.constants import c
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.distributions import matched_from_distribution_function
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from Simulations_ramps.Z_mode.simu_acceleration.plots_theory import plot_hamiltonian, get_hamiltonian, animated_plot_tracking
tracking_outdated = False
test_mode = False
optimise = False
verbose  = False
test_beams = False
tracking = True
ani_fig = False

particle_type = Electron()
n_particles = int(1.7e11)
n_macroparticles = int(1e5)

dt = 1e-9
dE = 1e9
        # Number of turns to track

with open("/Users/lvalle/cernbox/FCC-ee/Voltage_program/ramps_14_04_2025_14_27_48.pickle", "rb") as file:
    data_opt = pkl.load(file)
directory = 'output_figs_new_distribution'
voltage_ramp = data_opt['turn']['voltage_ramp_V']
energy_ramp = data_opt['turn']['energy_ramp_eV']
phi_s = data_opt['turn']['phi_s']
Nturns = len(energy_ramp)-1
tracking_parameters = HEBee_Eramp_parameters(op_mode='Z', dec_mode = True)
ring_HEB = generate_HEB_ring(op_mode='Z', Nturns=Nturns, momentum=energy_ramp)

beam = Beam(ring_HEB, n_macroparticles, n_particles)
beam.dt = np.load('../../../damped_distribution_dt_4mm.npy')
beam.dE = np.load('../../../damped_distribution_dE_4mm.npy')

rfcav = RFStation(ring_HEB, tracking_parameters.harmonic, voltage_ramp, phi_rf_d= np.pi)
long_tracker = RingAndRFTracker(rfcav, beam)
full_tracker = FullRingAndRF([long_tracker])


SR = [SynchrotronRadiation(ring_HEB, rfcav, beam, quantum_excitation=True, python=True, shift_beam=False)]
SR[0].print_SR_params()
plot_hamiltonian(ring_HEB, rfcav, beam, 1e-9, ring_HEB.energy[0][0]/20, k = 0, n_lines = 0, separatrix = True, option = 'test')
map_ = [long_tracker] + SR #+ [slice_beam]
#for hamiltonian
n_points = 1001
bl=[]
eml=[]
position = []
sE = []
pos = []
beam.statistics()
bl.append(beam.sigma_dt * c * 1e3)
sE.append(beam.sigma_dE/beam.energy*100)
eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
pos.append(phi_s[0]/rfcav.omega_rf[0,0]*1e9)

if ani_fig:
    animated_plot_tracking(ring_HEB, rfcav, beam, map_, 1e-9, ring_HEB.energy[0][0]/20, n_points=1001,
                           saving_file='animated_tracking_Z_mode_electrons', option='')
if tracking:
    for i in range(1, Nturns+1):
        # Track
        for m in map_:
            m.track()
        beam.statistics()
        beam.losses_separatrix(ring_HEB, rfcav)
        beam.eliminate_lost_particles()
        bl.append(beam.sigma_dt * c * 1e3)
        sE.append(beam.sigma_dE/beam.energy * 100)
        eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
        pos.append(phi_s[i]/rfcav.omega_rf[0,i]*1e9)
        position.append(beam.mean_dt*1e9)
        #print("   Longitudinal emittance (rms) %.4e eVs" % (np.pi * 4 * beam.sigma_dt * beam.sigma_dE))
        if (i % 50) == 0:
            plot_hamiltonian(ring_HEB, rfcav, beam, 1e-9, ring_HEB.energy[0][0] / 10, k=i, n_lines=0, separatrix=True,
                             directory=directory, option='test')

    fig, ax = plt.subplots()
    ax.plot(position, label = 'from tracking')
    ax.plot(pos, label = 'expected')
    ax.set_title('Average bunch position [ns]')
    ax.set(xlabel='turn', ylabel = 'Bunch position [ns]')
    ax.legend()
    plt.savefig(directory+'/bunch_position')
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(bl, label = 'from tracking')
    ax.plot(data_opt['turn']['rms_bunch_length']*1e3, label = 'expected')
    ax.legend()
    ax.set(xlabel='turn', ylabel = 'Bunch length [mm]')
    ax.set_title('RMS bunch length [mm]')
    plt.savefig(directory+'/bunch_length')
    plt.close()
    
    fig, ax = plt.subplots()
    ax.plot(sE, label = 'from tracking')
    ax.plot(data_opt['turn']['energy_spread']*100, label = 'expected')
    ax.legend()
    ax.set(xlabel='turn', ylabel = 'Energy spread [%]')
    ax.set_title('RMS energy spread [%]')
    plt.savefig(directory+'/energy_spread')
    plt.close()


if tracking_outdated:
    plt.ion()
    fig, axes = plt.subplots()
    dt_array = np.linspace(-dt, dt, n_points)
    dE_array = np.linspace(-dE, dE, n_points)
    X, Y = np.meshgrid(dt_array, dE_array)
    Z, hamiltonian_energy = get_hamiltonian(ring_HEB, rfcav, beam, X, Y, k=0)
    C = [axes.contour(X * 1e9, Y / 1e9, Z, [hamiltonian_energy], colors=['red'])]
    x, y = [], []
    scat = axes.scatter(x, y)
    axes.set_title('Acceleration simulation Z mode')
    bl=[]
    eml=[]
    beam.statistics()
    bl.append(4. * beam.sigma_dt * c * 1e3)
    eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
    print("   Longitudinal emittance (rms) %.4e eVs" % (np.pi * 4 * beam.sigma_dt * beam.sigma_dE))

    def animate(i):
        for m in map_:
            m.track()
        beam.statistics()
        bl.append(4. * beam.sigma_dt * c * 1e3)
        eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
        print("   Longitudinal emittance (rms) %.4e eVs" % (np.pi * 4 * beam.sigma_dt * beam.sigma_dE))
        x.append(beam.dt*1e9)
        y.append(beam.dE/1e9)
        for coll in C[0].axes.collections:
            coll.remove()
        Z, hamiltonian_energy = get_hamiltonian(ring_HEB, rfcav, beam, X, Y, k=i)
        C[0] = axes.contour(X,Y,Z, [hamiltonian_energy], colors = 'red')
        scat.set_offsets(np.c_[x, y])
        plt.draw()

    for i in range(1, Nturns + 1):
        # Track
        for m in map_:
            m.track()
        beam.statistics()
        bl.append(4. * beam.sigma_dt * c * 1e3)
        eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
        print("   Longitudinal emittance (rms) %.4e eVs" % (np.pi * 4 * beam.sigma_dt * beam.sigma_dE))
        x.append(beam.dt*1e9)
        y.append(beam.dE/1e9)
        for coll in C[0].axes.collections:
            coll.remove()
        Z, hamiltonian_energy = get_hamiltonian(ring_HEB, rfcav, beam, X, Y, k=i)
        C[0] = axes.contour(X,Y,Z, [hamiltonian_energy], colors='red')
        scat.set_array([beam.dt*1e9, beam.dE/1e9])
        fig.canvas.draw_idle()
        plt.pause(0.1)

    #plt.waitforbuttonpress()
    #ani = animation.FuncAnimation(fig, animate, interval=10, save_count=200)
    plt.show()
    #ani.save('animated_simulation_ramp_final_gain.gif')


if test_beams:
    beam1 = Beam(ring_HEB, n_macroparticles, n_particles)
    beam2 = Beam(ring_HEB, n_macroparticles, n_particles)
    beam3 = Beam(ring_HEB, n_macroparticles, n_particles)
    beam3.dt = np.load('../../../beam_phase.npy')
    beam3.dE = np.load('../../../beam_energy.npy')
    rfcav = RFStation(ring_HEB, tracking_parameters.harmonic, voltage_ramp, phi_rf_d= np.pi)
    bigaussian(ring_HEB, rfcav, beam1, tracking_parameters.sigmaz_0 / c / 4, sigma_dE = tracking_parameters.sigmaE_0* tracking_parameters.E_flat_bottom, reinsertion=True, seed=1)
    number_slices = 500
    long_tracker = RingAndRFTracker(rfcav, beam2)
    full_tracker = FullRingAndRF([long_tracker])
    matched_from_distribution_function(beam2, full_tracker, emittance=0.0008,
                                      distribution_type='gaussian',
                                      distribution_variable='Action',
                                      seed=1000)

    plot_hamiltonian(ring_HEB, rfcav, beam1, 1e-9, ring_HEB.energy[0][0]/20, n_lines = 100, separatrix = True)
    plt.scatter(beam1.dt * 1e9, beam1.dE / 1e9, label='bigaussian')
    plt.scatter(beam2.dt*1e9, beam2.dE/1e9, label='matched_from_distribution')
    #plt.scatter(beam3.dt * 1e9, beam3.dE / 1e9, label='damped_right_emittance')
    #plt.scatter(-beam3.dt * 1e9, beam3.dE / 1e9, label='minus t damped_right_emittance')
    plt.legend()
    plt.show()
