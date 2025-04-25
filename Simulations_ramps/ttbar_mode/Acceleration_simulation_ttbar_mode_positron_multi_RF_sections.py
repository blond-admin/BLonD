import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from blond.beam.beam import Beam, Positron
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.distributions import bigaussian
from ramp_modules.Ramp_optimiser_functions import HEBee_Eramp_parameters
from ring_parameters.generate_rings import generate_HEB_ring
from scipy.constants import c
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from plots_theory_positron import plot_hamiltonian, get_hamiltonian

test_mode = False
optimise = False
verbose  = False
test_beams = True
tracking = False

particle_type = Positron()
n_particles = int(1.7e11)
n_macroparticles = int(1e5)

dt = 1e-9
dE = 1e9

# Number of RF sections
n_sections = 4
option_summary = f'n_Sections_{n_sections}_elle_input'
op_mode = 'ttbar'
with open("/Users/lvalle/cernbox/FCC-ee/Voltage_program/ttbar/ramps_ramp22_04_2025_16_32_18ttbar.pickle", "rb") as file:
    data_opt = pkl.load(file)
directory = 'output_figs_multi_sections'
voltage_ramp = data_opt['turn']['voltage_ramp_V']
energy_ramp = data_opt['turn']['energy_ramp_eV']
phi_s = data_opt['turn']['phi_s']
Nturns = len(energy_ramp)-1
tracking_parameters = HEBee_Eramp_parameters(op_mode=op_mode, dec_mode = True)
ring_HEB = generate_HEB_ring(op_mode=op_mode, particle=particle_type, n_sections= n_sections, Nturns=Nturns, momentum=energy_ramp)

beam = Beam(ring_HEB, n_macroparticles, n_particles)
beam.dt = np.load('../../damped_distribution_dt_4mm.npy')
beam.dE = np.load('../../damped_distribution_dE_4mm.npy')

rfcavs = []
long_tracker = []
SR = []
map_ = []

for i in range(n_sections):
    rfcavs.append(RFStation(ring_HEB, tracking_parameters.harmonic, voltage_ramp/n_sections, phi_rf_d= 0, section_index=i+1))
    long_tracker.append(RingAndRFTracker(rfcavs[i], beam))
    SR.append(SynchrotronRadiation(ring_HEB, rfcavs[i], beam, quantum_excitation=True, python=True, shift_beam=False))
    map_ += [SR[i]] + [long_tracker[i]]

full_tracker = FullRingAndRF(long_tracker)
SR[0].print_SR_params()

rfcav = RFStation(ring_HEB, tracking_parameters.harmonic, voltage_ramp, phi_rf_d= 0)
plot_hamiltonian(ring_HEB, rfcav, beam, 1e-9, ring_HEB.energy[0][0]/20, k = 0, n_lines = 0, directory=directory,separatrix = True, option = 'test')
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

for i in range(1, Nturns+1):
    # Track
    for m in map_:
        m.track()
    beam.statistics()
    bl.append(beam.sigma_dt * c * 1e3)
    sE.append(beam.sigma_dE/beam.energy * 100)
    eml.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
    pos.append(phi_s[i]/rfcav.omega_rf[0,i]*1e9)
    position.append(beam.mean_dt*1e9)
    #print("   Longitudinal emittance (rms) %.4e eVs" % (np.pi * 4 * beam.sigma_dt * beam.sigma_dE))
    if (i % 50) == 0:
        plot_hamiltonian(ring_HEB, rfcav, beam, 1e-9, ring_HEB.energy[0][0] /2, k=i, n_lines=0, separatrix=True,
                         directory=directory, option='test')

fig, ax = plt.subplots()
ax.plot(position, label = 'from tracking')
ax.plot(pos, label = 'expected')
ax.set_title(f'Average bunch position [ns], n_sections = {n_sections}')
ax.set(xlabel='turn', ylabel = 'Bunch position [ns]')
ax.legend()
plt.savefig(directory+'/bunch_position'+option_summary)
plt.close()

fig, ax = plt.subplots()
ax.plot(bl, label = 'from tracking')
ax.plot(data_opt['turn']['rms_bunch_length']*1e3, label = 'expected')
ax.legend()
ax.set(xlabel='turn', ylabel = 'Bunch length [mm]')
ax.set_title(f'RMS bunch length [mm], n_sections = {n_sections}')
plt.savefig(directory+'/bunch_length'+option_summary)
plt.close()

fig, ax = plt.subplots()
ax.plot(sE, label = 'from tracking')
ax.plot(data_opt['turn']['energy_spread']*100, label = 'expected')
ax.legend()
ax.set(xlabel='turn', ylabel = 'Energy spread [%]')
ax.set_title(f'RMS energy spread [%], n_sections = {n_sections}')
plt.savefig(directory+'/energy_spread'+option_summary)
plt.close()


dictresults = {}
dictresults.update({'energy_ramp_eV': energy_ramp,
             'voltage_ramp_V': voltage_ramp,
             'phi_s': phi_s,
             'tracking_sigmaE_perc': sE,
             'expected_sigmaE_perc': data_opt['turn']['energy_spread']*100,
             'tracking_bl_mm': bl,
             'expected_bl_mm': data_opt['turn']['rms_bunch_length']*1e3,
             'tracking_bpos_ns': position,
             'expected_bpos_ns': pos,
             'n_sections': n_sections,
             'op_mode':op_mode,
            })
filename = f'summary_characteristics_{n_sections}_rf_sections_{op_mode}_mode'

pkl.dump(dictresults, open(filename, "wb"))




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
    axes.set_title('Acceleration simulation ttbar mode')
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