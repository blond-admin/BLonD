import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import copy

from scipy.constants import c

from blond.input_parameters.rf_parameters import RFStation
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation
from blond.trackers.tracker import RingAndRFTracker
from blond3 import BiGaussian
from blond3.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import gather_longitudinal_synchrotron_radiation_parameters

from blond3.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond3._core.beam.beams import Beam
from blond3._core.beam.particle_types import electron
from blond3._core.simulation.simulation import Simulation
from blond3.physics.cavities import SingleHarmonicCavity
from blond3.physics.drifts import DriftSimple

def plot_hamiltonian(ring, rfstation, beam, dt, dE, k = int(0), hamiltonian_energy = None, n_points = 1001, n_lines = 100, separatrix = True, directory = 'output_figs_wigglers',
                     frame_path = '',get_data_animation = False,  option = ''):
    dt_array = np.linspace(-dt, dt, n_points)
    dE_array = np.linspace(-dE, dE, n_points)
    plt.figure()
    hE_t = lambda DE: c * np.pi/ (ring.ring_circumference * beam.beta * beam.energy) * \
                      (
                        ring.eta_0[0, k] * DE ** 2 + ring.eta_1[0, k] * DE ** 3 + ring.eta_2[0, k] * DE ** 4)

    hdelta_phi = lambda delta:   1 / 2 * rfstation.harmonic * ring.eta_0[0, k] * delta ** 2 + \
                                 1 / 3 * rfstation.harmonic * ring.eta_1[0, k] * delta ** 3 + \
                                 1 / 4 * rfstation.harmonic * ring.eta_2[0, k] * delta ** 4

    hphi_delta = lambda phi: ring.Particle.charge * rfstation.voltage[k] / (2 * np.pi * ring.energy[0, k]) * (np.cos(phi) - np.cos(rfstation.phi_s[k]) + (phi - rfstation.phi_s[k])* np.sin(rfstation.phi_s[k]))

    hphi_DE = lambda phi: c * beam.beta * ring.Particle.charge * rfstation.voltage[0, k] / (rfstation.harmonic[0, k] * ring.ring_circumference) * \
                        (np.cos(phi) - np.cos(rfstation.phi_s[k]) + (phi - rfstation.phi_s[k]) * np.sin(rfstation.phi_s[k]))
    X, Y = np.meshgrid(dt_array, dE_array)
    Xt = rfstation.omega_rf[0,k] * X + rfstation.phi_rf_d[0,k]
    Z = hphi_DE(Xt) + hE_t(Y)

    if n_lines != 0:
        plt.contour(X*1e9, Y/1e9, Z, n_lines)
    if separatrix:
        plt.contour(X*1e9, Y/1e9, Z, [hphi_DE(np.pi - rfstation.phi_s[k])], colors=['red'])
        if get_data_animation:
            plt.figure(figsize=(6, 5))
            plt.title(f'ttbar mode\n Turn: {k}')
            plt.xlabel('t [ns]')
            plt.ylabel('DE [GeV]')
            plt.xlim([0, 1.25])

            dt = beam.dt[0:10000] * 1e9
            dE = beam.dE[0:10000] * 1e-9

            if max(dE)<=1.5:
                plt.ylim([-2, 2])
            else:
                plt.ylim([max(dE)*-1.1, max(dE)*1.1])

            plt.contour(X * 1e9, Y / 1e9, Z, [hphi_DE(np.pi - rfstation.phi_s[k])], colors=['red'])
            plt.scatter(dt, dE, s = 0.2)
            plt.savefig(frame_path)
            plt.close()
            plt.close()
    if hamiltonian_energy is not None:
        for energy in hamiltonian_energy:
            plt.contour(X*1e9, Y/1e9, Z, [energy])
    plt.xlabel('t [ns]')
    plt.xlim([0,1.25])
    plt.ylabel('DE [GeV]')
    plt.scatter(beam.dt*1e9, beam.dE/1e9, s = 0.2)

    plt.title(f'ttbar \n turn #{k}')
    text = directory + '/plot_' + str(k) + option
    plt.savefig(text)
    plt.close()
class SynchrotronRadiationSimulation:
    def __init__(self):

        self.synchrotron_radiation_integrals = np.array([0.646747216157, 0.0005936549319, 5.6814536525e-08, 5.92870407301e-09 , 1.71368060083-11])
        cavity1 = SingleHarmonicCavity()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        N_TURNS = int(10)
        energy_cycle = MagneticCyclePerTurn(
            value_init=182.5e9,
            values_after_turn=np.linspace(182.5e9, 182.5e9, N_TURNS),
            reference_particle=electron,
            in_unit="momentum",
        )

        drift1 = DriftSimple(
            orbit_length=90.65874532 * 1e3,
        )
        drift1.transition_gamma = (55.759505,)

        # beam = Beam(n_particles=1e9, particle_type=electron)
        # dt = beam.write_partial_dt()
        # dE = beam.write_partial_dE()
        #
        # self.beam = beam
        #
        # simulation = Simulation.from_locals(locals())
        # simulation.print_one_turn_execution_order()

        # simulation.on_prepare_beam(
        #      beam=beam,
        #      preparation_routine=BiGaussian(
        #          sigma_dt=0.4e-9 / 4,
        #          sigma_dE=1e9 / 4,
        #          reinsertion=False,
        #          seed=1,
        #          n_macroparticles=10,
        #      ),
        #      turn_i=1,
        #  )
        # dt[:] += np.load(
        #     'Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
        #     '/damped_distribution_dt_4mm'
        #     '.npy')
        # dE[:] += np.load(
        #     'Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
        #     '/damped_distribution_dE_4mm'
        #     '.npy')
        # self.simulation = simulation
        # self.simulation.run_simulation()


    def compare_kicks(self):
        # particles_energy = self.beam.reference_total_energy + self.beam.read_partial_dE()
        # U0, tau_z, sigma0 = gather_longitudinal_synchrotron_radiation_parameters(
        #      particle_type=self.beam.particle_type,
        #     energy=particles_energy,
        #     synchrotron_radiation_integrals = self.synchrotron_radiation_integrals,
        #  )
        #
        # U0_mono, tau_z_mono, sigma0_mono = gather_longitudinal_synchrotron_radiation_parameters(
        #      particle_type=self.beam.particle_type,
        #     energy=self.beam.reference_total_energy,
        #     synchrotron_radiation_integrals = self.synchrotron_radiation_integrals,
        #  )

        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111)
        # sb.distplot(U0)
        # plt.title('Energy lost per particle')
        #
        # plt.show()

        # return (
        #  -
        #  - 2.0 / tau_z * self.beam.read_partial_dE()
        #  - 2.0
        #  * sigma0
        #  / np.sqrt(tau_z)
        #  * self.beam.reference_total_energy
        #  * np.random.normal(size=len(self.beam.n_macroparticles_partial()))
        # )
        pass


def prepare_ring_rf_beam(n_turns = 100, n_sections= 1):
    from blond.beam.beam import Beam, Positron
    from blond.input_parameters.ring import Ring

    n_particles = int(2.725e10)
    n_macroparticles = int(1e5)
    C = 90.65874532 * 1e3
    alpha_0 = 7.120435962 * 1e-6
    rho = C / 2 * np.pi
    momentum = 20e9
    ring_HEB = Ring(C, alpha_0, momentum * np.ones(n_turns+1), Positron(),
                    n_turns,
                    bending_radius=rho,
                    synchronous_data_type='total energy')
    SRS = SynchrotronRadiationSimulation()
    U0_mono, tau_z_mono, sigma0_mono = (
        gather_longitudinal_synchrotron_radiation_parameters(
            energy=ring_HEB.energy[0][0],
            synchrotron_radiation_integrals=SRS.synchrotron_radiation_integrals,
        ))

    voltagee = U0_mono * ring_HEB.f_rev[0]
    rfcav = RFStation(ring_HEB, 242400, voltagee* np.ones(n_turns+1), phi_rf_d=0)
    beam = Beam(ring_HEB, n_macroparticles, n_particles)
    beam.dt = np.load('/Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
                              '/damped_distribution_dt_4mm'
                              '.npy')
    beam.dE = np.load('/Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
                              '/damped_distribution_dE_4mm'
                              '.npy')

    return ring_HEB, rfcav, beam

def compare_energy_kicks():
    ring_HEB, rfcav, beam = prepare_ring_rf_beam()

    SRS = SynchrotronRadiationSimulation()
    SR = [SynchrotronRadiation(ring_HEB, rfcav, beam,
                               bending_radius = ring_HEB.bending_radius,
                               quantum_excitation=True,
                               python=True, shift_beam=False)]

    U0, tau_z, sigma0 = gather_longitudinal_synchrotron_radiation_parameters(
         energy=ring_HEB.energy[0][0] + beam.dE,
         synchrotron_radiation_integrals = SRS.synchrotron_radiation_integrals,
     )
    U0_mono, tau_z_mono, sigma0_mono = (
        gather_longitudinal_synchrotron_radiation_parameters(
         energy=ring_HEB.energy[0][0],
         synchrotron_radiation_integrals = SRS.synchrotron_radiation_integrals,
     ))
    ### Validate calculations
    SR[0].I1 = SRS.synchrotron_radiation_integrals[0]
    SR[0].I2 = SRS.synchrotron_radiation_integrals[1]
    SR[0].I3 = SRS.synchrotron_radiation_integrals[2]
    SR[0].I4 = SRS.synchrotron_radiation_integrals[3]
    SR[0].I5 = SRS.synchrotron_radiation_integrals[4]
    SR[0].calculate_SR_params()
    SR[0].print_SR_params()

    print(f'BLonD3 Energy loss per turn: {U0_mono/1e9} GeV per turn')
    print(f'BLonD3 Damping time: {tau_z_mono} turns')
    print(f'BLonD3 Energy spread: {sigma0_mono} ')

    energy_kick_per_particle = (- U0 + sigma0_mono / np.sqrt(tau_z_mono)
                                *beam.energy
                                *np.random.normal(size=beam.n_macroparticles))

    energy_kick_conventional = - U0_mono - 2.0 / tau_z_mono * beam.dE
    (- 2.0 * sigma0_mono / np.sqrt(tau_z_mono) *
     beam.energy  *np.random.normal(size=beam.n_macroparticles))

    return energy_kick_conventional, energy_kick_per_particle

def track_with_SR_kicks(directory = 'plots', n_turns = 2000, n_sections = 1):
    ring_HEB, rfcav, beam = prepare_ring_rf_beam(n_turns = n_turns+1,
                                                 n_sections= n_sections)
    beam_particles = copy.deepcopy(beam)
    rfcav_particles = copy.deepcopy(rfcav)



    RFtracker = RingAndRFTracker(rfcav, beam)
    RFtracker_particles = RingAndRFTracker(rfcav_particles, beam_particles)
    bl_conventional = []
    eml_conventional = []
    sE_conventional = []
    pos_conventional = []
    beam.statistics()
    bl_conventional.append(beam.sigma_dt * c * 1e3)
    sE_conventional.append(beam.sigma_dE / beam.energy * 100)
    eml_conventional.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
    pos_conventional.append(beam.mean_dt * 1e9)
    plot_hamiltonian(ring_HEB, rfcav, beam, 1.25e-9, ring_HEB.energy[0][0],
                     k=0, n_lines=0, directory='plots', separatrix=True,
                     option='test')

    bl_particles = []
    eml_particles = []
    sE_particles = []
    pos_particles = []
    beam_particles.statistics()
    bl_particles.append(beam_particles.sigma_dt * c * 1e3)
    sE_particles.append(beam_particles.sigma_dE / beam_particles.energy * 100)
    eml_particles.append(np.pi * 4 * beam_particles.sigma_dt * beam_particles.sigma_dE)
    pos_particles.append(beam_particles.mean_dt * 1e9)

    U0 = []
    U0_particles = []

    for i in range(n_turns):

        energy_kick_conventional, energy_kick_per_particle = compare_energy_kicks()
        U0.append(np.average(energy_kick_conventional))
        U0_particles.append(np.average(energy_kick_per_particle))

        rfcav.voltage = energy_kick_conventional * ring_HEB.f_rev[0]
        rfcav_particles.voltage = (np.average(energy_kick_per_particle) *
                                   ring_HEB.f_rev[0])
        RFtracker = RingAndRFTracker(rfcav, beam)
        RFtracker_particles = RingAndRFTracker(rfcav_particles, beam_particles)
        RFtracker.track()
        RFtracker_particles.track()

        beam.dE += energy_kick_conventional
        beam_particles.dE += energy_kick_per_particle

        beam.statistics()
        bl_conventional.append(beam.sigma_dt * c * 1e3)
        sE_conventional.append(beam.sigma_dE/beam.energy * 100)
        eml_conventional.append(np.pi * 4 * beam.sigma_dt * beam.sigma_dE)
        pos_conventional.append(beam.mean_dt*1e9)

        beam_particles.statistics()
        bl_particles.append(beam_particles.sigma_dt * c * 1e3)
        sE_particles.append(beam_particles.sigma_dE / beam_particles.energy * 100)
        eml_particles.append(np.pi * 4 * beam_particles.sigma_dt * beam_particles.sigma_dE)
        pos_particles.append(beam_particles.mean_dt * 1e9)

    fig, ax = plt.subplots()
    ax.plot(pos_conventional, label = 'Energy lost per turn')
    ax.plot(pos_particles, ls = '--', label = ('Energy lost per '
                                                      'particle'))
    ax.set_title(f'Average bunch position [ns], n_sections = {n_sections}')
    ax.set(xlabel='turn', ylabel = 'Bunch position [ns]')
    ax.legend()
    plt.savefig(f'bunch_position_n_turns_{n_turns}')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(bl_conventional, label = 'Energy lost per turn')
    ax.plot(bl_particles, ls = '--', label='Energy lost per particle')
    ax.legend()
    ax.set(xlabel='turn', ylabel = 'Bunch length [mm]')
    ax.set_title(f'RMS bunch length [mm], n_sections = {n_sections}')
    plt.savefig(f'bunch_length_n_turns_{n_turns}')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(sE_conventional, label = 'Energy lost per turn')
    ax.plot(sE_particles, ls = '--', label = 'Energy lost per particle')
    ax.legend()
    ax.set(xlabel='turn', ylabel = 'Energy spread [%]')
    ax.set_title(f'RMS energy spread [%], n_sections = {n_sections}')
    plt.savefig(f'energy_spread_n_turns_{n_turns}')
    plt.close()

    fig, ax = plt.subplots()

    ax.plot(U0_particles, ls='--', label='Average energy lost per '
                                               'particle')
    ax.plot(np.average(U0_particles)*np.ones(len(U0_particles)), ls='--',
            color =
    'red',
            label=(
      'average value of the latter'))
    ax.plot(U0, label='Energy lost per turn')
    ax.legend()
    ax.set(xlabel='turn', ylabel='Energy spread [%]')
    ax.set_title(f'Energy lost per turn')
    plt.savefig(f'energy_lost_per_turn_{n_turns}')
    plt.close()

track_with_SR_kicks(n_turns=5000)