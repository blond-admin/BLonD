from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import imageio
from blond.beam.beam import Beam, Positron
from blond.input_parameters.rf_parameters import RFStation

from ramp_modules.Ramp_optimiser_functions import HEBee_Eramp_parameters
from ring_parameters.generate_rings import generate_HEB_ring
from scipy.constants import c
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from Simulations_ramps.Z_mode.simu_acceleration_positron.plots_theory_positron import plot_hamiltonian, get_hamiltonian
from Simulations_ramps.Z_mode.simu_acceleration.functions_wiggler import wiggler, update_rad_int, update_SRtracker_and_track

test_mode = False
optimise = False
verbose  = False
tracking = False
jitter = True
get_data_animation = True

particle_type = Positron()
n_particles = int(1.7e11)
n_macroparticles = int(1e5)

dt = 1e-9
dE = 1e9
        # Number of turns to track

with open("/Users/lvalle/cernbox/FCC-ee/Voltage_program/ramps_optimised_overshoot_two_wigglers_espread_max_100GeV_s_FCC_week_2025.pickle", "rb") as file:
    data_opt = pkl.load(file)
directory = 'output_figs_jitter_wiggler_disabled'
voltage_ramp = data_opt['turn']['voltage_ramp_V']
energy_ramp = data_opt['turn']['energy_ramp_eV']
time_ramp = data_opt['turn']['time_ramp_s']
phi_s = data_opt['turn']['phi_s']
Nturns = len(energy_ramp)-1
tracking_parameters = HEBee_Eramp_parameters(op_mode='Z', dec_mode = True)
ring_HEB = generate_HEB_ring(op_mode='Z', particle=particle_type,Nturns=Nturns, momentum=energy_ramp)
wiggler_HEB = wiggler()

beam = Beam(ring_HEB, n_macroparticles, n_particles)
beam.dt = np.load('../../../beam_phase.npy')
beam.dE = np.load('../../../beam_energy.npy')
###### Shift the injected beam ##############
Delta_E = 3e-3 * ring_HEB.energy[0][0] #eV # max. 3e-3 relative energy error (from transfer line)
Delta_t = 50e-12 #s max 50ps max time jitter

if jitter:
    beam.dt += -Delta_t
    beam.dE += Delta_E
rfcav = RFStation(ring_HEB, tracking_parameters.harmonic, voltage_ramp, phi_rf_d= 0)
long_tracker = RingAndRFTracker(rfcav, beam)
full_tracker = FullRingAndRF([long_tracker])

SR = [SynchrotronRadiation(ring_HEB, rfcav, beam, rad_int = update_rad_int(ring_HEB, wiggler_HEB, E=20e9), quantum_excitation=True, python=True, shift_beam=False)]
SR[0].print_SR_params()

plot_hamiltonian(ring_HEB, rfcav, beam, 1e-9, ring_HEB.energy[0][0]/20, k = 0, n_lines = 0, directory=directory, separatrix_flag = True, option = 'test')

t_disable = 0.204
map_ = [long_tracker] + SR

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
folder_paths = [
    '/Users/lvalle/PycharmProjects/BLonD/Simulations_ramps/Z_mode/data_figs/',
]
gif_paths = [
    '/Users/lvalle/PycharmProjects/BLonD/Simulations_ramps/Z_mode/simu_acceleration_positron/gif_path/animated_ramp_Z_mode_jitter_wiggler_disabled.gif',
]

n = 0
get_images = True
# Folder to save frames
opmode = 'Z'
folder_name = opmode + '_mode/frames_jitter_wiggler'
os.makedirs("frames_jitter_wiggler", exist_ok=True)
filenames = []
for i in range(1, Nturns + 1):
    # Track
    #for m in map_:
    #    m.track()
    long_tracker.track()
    if time_ramp[i] > t_disable > time_ramp[i - 1]:
        wiggler_HEB.DI2_woE = 0
        wiggler_HEB.DI3_woE = 0
        wiggler_HEB.DI4_woE = 0
        wiggler_HEB.DI5_woE = 0
    SR = [SynchrotronRadiation(ring_HEB, rfcav, beam, rad_int=update_rad_int(ring_HEB, wiggler_HEB, E=energy_ramp[i]),
                                   quantum_excitation=True, python=True, shift_beam=False)]
    SR[0].print_SR_params()
    SR[0].track()
    beam.statistics()
    bl.append(beam.sigma_dt * c * 1e3)
    sE.append(beam.sigma_dE / beam.energy * 100)
    eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
    pos.append(phi_s[i] / rfcav.omega_rf[0, i] * 1e9)
    position.append(beam.mean_dt * 1e9)
    if get_data_animation:
        if i < 50:
            frame_path = f"frames_jitter_wiggler/plot_{i:03d}.png"
            plot_hamiltonian(ring_HEB, rfcav, beam, 1.25e-9, ring_HEB.energy[0][0] / 10, k=i, n_lines=0,
                             separatrix_flag=True,
                             directory=directory, option='test', get_data_animation=get_data_animation,
                             frame_path=frame_path)
            filenames.append(frame_path)
            print(f"Iteration {i} done")
        elif (i % 10) == 0:
            frame_path = f"frames_jitter_wiggler/plot_{i:03d}.png"
            plot_hamiltonian(ring_HEB, rfcav, beam, 1.25e-9, ring_HEB.energy[0][0] / 10, k=i, n_lines=0,
                             separatrix_flag=True,
                             directory=directory, option='test', get_data_animation=get_data_animation,
                             frame_path=frame_path)
            filenames.append(frame_path)
            print(f"Iteration {i} done")
    else:
        if (i % 50) == 0:
            plot_hamiltonian(ring_HEB, rfcav, beam, 1.25e-9, ring_HEB.energy[0][0] / 10, k=i, n_lines=0,
                             separatrix_flag=True,
                             directory=directory, option='test', get_data_animation=get_data_animation)

if filenames:
    with imageio.get_writer(gif_paths[n], mode='I', duration=0.05) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            print(f"Added {filename} to GIF")
    print(f"GIF saved as {filename}.gif.")
else:
    print("No frames to create a GIF.")

# # Clean up frames
for filename in filenames:
    if os.path.exists(filename):
        os.remove(filename)
if os.path.exists("frames_jitter_wiggler") and not os.listdir("frames_jitter_wiggler"):
    os.rmdir("frames_jitter_wiggler")
    print(f"Saved frame {i:03d} at {frame_path}")

print("Temporary frames deleted.")
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


if tracking:
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
