'''
PSB simulation of a single particle with a ramp.

:Author: **Thom Arnoldus van Rijswijk**
'''

# Imports
import os
import numpy as np
from scipy.constants import c, e, m_p
import matplotlib.pyplot as plt

# Xsuite imports
import xpart as xp
import xtrack as xt
import xplt

# BLonD objects
from blond.beam.beam import Beam, Proton
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.trackers.tracker import RingAndRFTracker

# Interface
from blond.interfaces.xsuite import (BlondElement, EnergyUpdate,
                                     blond_beam_to_xsuite_coords,
                                     xsuite_coords_to_blond_coords)

# Monitor objects
from blond.monitors.monitors import BunchMonitor
from blond.plots.plot import Plot

# Directory to save files
this_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
results_directory = this_directory + '../output_files/EX_27'
os.makedirs(results_directory, exist_ok=True)

# Parameters ----------------------------------------------------------------------------------------------------------
# Accelerator parameters
C = 2 * np.pi * 25.000057945260654              # Machine circumference, radius 25 [m]
gamma_t = 4.11635447373496                      # Transition gamma [-]
alpha = 1./gamma_t/gamma_t                      # First order mom. comp. factor [-]
h = 1                                           # Harmonic number [-]
V = 8e3                                         # RF voltage [V]
dphi = np.pi                                    # Phase modulation/offset [rad]

# Beam parameters
N_m = 1                                         # Number of macroparticles [-]
N_p = 1e11                                      # Intensity / number of particles
sigma_dt = 180e-9 / 4                           # [s]        
blen = sigma_dt * 4                             # Initial bunch length, 4 sigma [s]
kin_beam_energy = 1.4e9                         # Kinetic energy [eV]


# Simulation parameters
N_t = 6000                                      # Number of (tracked) turns [-]
N_buckets = 1                                   # Number of buckets [-]
dt_plt = 500                                    # Timestep between plots [-]
input_dt = 5.720357923415153e-07 - sigma_dt     # Input particles dt [s]
input_dE = 0.0                                  # Input particles dE [eV]

# Derived parameters
E_0 = m_p * c**2 / e                                  # [eV]
tot_beam_energy = E_0 + kin_beam_energy               # [eV]
sync_momentum = np.sqrt(tot_beam_energy**2 - E_0**2)  # [eV / c]
momentum_increase = 0.001e9                           # [eV / c]

# Preparing BLonD items -----------------------------------------------------------------------------------------------
print(f"\nPreparing BLonD items...")
# --- BLonD objects ---
ring = Ring(C, alpha, np.linspace(sync_momentum, sync_momentum + momentum_increase, N_t + 1), Proton(), n_turns=N_t)
rfstation = RFStation(ring, [h], [V], [dphi]) 
beam = Beam(ring, N_m, N_p)

# --- Insert particle ---
beam.dt = np.array([input_dt])
beam.dE = np.array([input_dE])

# --- RF tracker object ---
blond_tracker = RingAndRFTracker(rfstation, beam, Profile=None, interpolation=False,
                                 CavityFeedback=None, BeamFeedback=None,
                                 TotalInducedVoltage=None, 
                                 with_xsuite=True)

# --- Bunch monitor ---
# Print parameters for plotting
print(f"\n Calculating plotting parameters...")

t_u = np.pi / rfstation.omega_rf[0, 0]
Y_phi_s = np.sqrt(abs(-np.cos(rfstation.phi_s[0]) + (np.pi - 2 * rfstation.phi_s[0]) / 2 * np.sin(rfstation.phi_s[0])))
dE_sep = np.sqrt(2 * abs(rfstation.charge) * rfstation.voltage[0, 0] * rfstation.beta[0] * rfstation.beta[0]
                 * rfstation.energy[0] / (np.pi * rfstation.harmonic[0, 0] * abs(ring.eta_0[0, 0]))) * Y_phi_s

print('t_u: ' + str(2 * t_u))
print('dE_sep,m: ' + str(dE_sep))

# Make bunchmonitor
bunchmonitor = BunchMonitor(ring, rfstation, beam,
                            results_directory + '/EX_27_output_data', Profile=None)

format_options = {'dirname': results_directory + '/EX_27_fig'}
plots = Plot(ring, rfstation, beam, dt_plt, N_t, 0, 2 * t_u,
             -1.05 * dE_sep, 1.05 * dE_sep, xunit='s', separatrix_plot=True,
             Profile=None, h5file=results_directory + '/EX_27_output_data',
             format_options=format_options)

# --- Creating interface elements ---
# Blond tracker
cavity = BlondElement(blond_tracker, beam)
# BLonD monitor
beam_monitor = BlondElement(bunchmonitor, beam)
beam_plots = BlondElement(plots, beam)

# Preparing Xsuite items -----------------------------------------------------------------------------------------------
print(f"\nPreparing Xsuite items...")
# --- Setup matrix ---
# Make First order matrix map (takes care of drift in Xsuite)
matrix = xt.LineSegmentMap(
    longitudinal_mode='nonlinear',
    qx=1.1, qy=1.2,
    betx=1., 
    bety=1., 
    voltage_rf=0,
    frequency_rf=0,
    lag_rf=0,
    momentum_compaction_factor=alpha,
    length=C)

# Create line
line = xt.Line(elements=[matrix], element_names={'matrix'})
line['matrix'].length = C

# Insert the BLonD elements
line.insert_element(index=0, element=cavity, name='blond_cavity')
line.append_element(element=beam_monitor, name='blond_monitor')
line.append_element(element=beam_plots, name='blond_plots')

# Insert energy ramp
energy_update = EnergyUpdate(ring.momentum[0, :])
line.insert_element(index=0, element=energy_update, name='energy_update')

# Add particles to line and build tracker 
line.particle_ref = xp.Particles(p0c=sync_momentum, mass0=xp.PROTON_MASS_EV, q0=1.)
line.build_tracker()

# Show table
line.get_table().show()
# prints:

# name               s element_type   isthick isreplica parent_name iscollective
# energy_update      0 EnergyUpdate     False     False None                True
# blond_cavity       0 BlondElement     False     False None                True
# matrix             0 LineSegmentMap    True     False None               False
# blond_monitor 157.08 BlondObserver    False     False None                True
# blond_plots   157.08 BlondElement     False     False None                True
# _end_point    157.08                  False     False None               False

# Simulating ----------------------------------------------------------------------------------------------------------
print(f"\nSetting up simulation...")

# --- Convert the initial BLonD distribution to xsuite coordinates ---
zeta, ptau = blond_beam_to_xsuite_coords(beam, 
                                         line.particle_ref.beta0[0],
                                         line.particle_ref.energy0[0],
                                         phi_s=rfstation.phi_s[0] - rfstation.phi_rf[0, 0],  # Correct for RF phase
                                         omega_rf=rfstation.omega_rf[0, 0])

# --- Track matrix ---
particles = line.build_particles(x=0, y=0, px=0, py=0, zeta=np.copy(zeta), ptau=np.copy(ptau))
line.track(particles, num_turns=N_t, turn_by_turn_monitor=True, with_progress=True)
mon = line.record_last_track

# Saving turn-by-turn particle coordinates
test_string = ''
test_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'dE', 'dt', 'momentum', 'gamma', 'beta', 'energy', 'x', 'px', 'y', 'py')
test_string += ('{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}'
                '\t{:+10.10e}\t{:+10.10e}\n').format(beam.dE[0], beam.dt[0], rfstation.momentum[0],
                                                     rfstation.gamma[0], rfstation.beta[0], rfstation.energy[0],
                                                     mon.x[0, 0].T, mon.px[0, 0].T, mon.y[0, 0].T, mon.py[0, 0].T)

# Saving turn-by-turn Xsuite particle coordinates
xsuite_string = ''
xsuite_string += '{:<17}\t{:<17}\t{:<17}\t{:<17}\t{:<17}\n'.format(
    'ptau', 'zeta', 'momentum', 'gamma', 'beta')
xsuite_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
    mon.ptau[0, 0].T, mon.zeta[0, 0].T, np.float64(mon.p0c[0, 0]), np.float64(mon.gamma0[0, 0]),
    np.float64(mon.beta0[0, 0])
)

# Convert the xsuite particle coordinates back to BLonD
for i in range(N_t):
    dt, dE = xsuite_coords_to_blond_coords(mon.zeta[:, i].T, mon.ptau[:, i].T,
                                           rfstation.beta[i],
                                           rfstation.energy[i],
                                           phi_s=rfstation.phi_s[i] - rfstation.phi_rf[0, 0],  # Correct for RF phase
                                           omega_rf=rfstation.omega_rf[0, i])

    # Statistics
    test_string += ('{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}'
                    '\t{:+10.10e}\t{:+10.10e}\n').format(dE[0], dt[0], rfstation.momentum[i], rfstation.gamma[i],
                                                         rfstation.beta[i], rfstation.energy[i], mon.x[0, i].T,
                                                         mon.px[0, i].T, mon.y[0, i].T, mon.py[0, i].T)

    xsuite_string += '{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\t{:+10.10e}\n'.format(
        mon.ptau[0, i].T, mon.zeta[0, i].T, np.float64(mon.p0c[0, i]), np.float64(mon.gamma0[0, i]),
        np.float64(mon.beta0[0, i])
    )

with open(results_directory + '/EX_27_output_data.txt', 'w') as f:
    f.write(test_string)
with open(results_directory + '/EX_27_output_data_xsuite.txt', 'w') as f:
    f.write(xsuite_string)

# Results --------------------------------------------------------------------------------------------------------------
print(f"\nPlotting result...")

# Plot Phasespace

plt.figure()
plt.plot(mon.zeta.T,mon.ptau.T)
plt.scatter(np.mean(mon.zeta.T), np.mean(mon.ptau.T), label='mon')
plt.grid()
plt.title('Phasespace')
plt.xlabel(r'$\zeta$')
plt.ylabel(r'$p_{\tau}$')
plt.savefig(results_directory + '/figure_Phasespace.png')

# Use Xplt to plot Phasespace

plot = xplt.PhaseSpacePlot(
    mon
)
plot.fig.suptitle("Particle distribution for a single turn")
plot.fig.savefig(results_directory + '/figure_ps_mon.png')

# Show plots
plt.show()
