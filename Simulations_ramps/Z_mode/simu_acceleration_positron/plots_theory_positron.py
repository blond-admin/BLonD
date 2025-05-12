import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from blond.trackers.utilities import hamiltonian, separatrix



def get_hamiltonian(ring, rfstation, beam, X, Y, k = int(0)):
    hE_t = lambda DE: c * np.pi / (ring.ring_circumference * beam.beta * beam.energy) * \
                      (1 / 2 * ring.eta_0[0, k] * DE ** 2 + 1 / 3 * ring.eta_1[0, k] * DE ** 3 + 1 / 4 *
                              ring.eta_2[0, k] * DE ** 4)

    hphi_DE = lambda phi: c * beam.beta * ring.Particle.charge * rfstation.voltage[0, k] / (
                rfstation.harmonic[0, k] * ring.ring_circumference) * \
                          (np.cos(phi) - np.cos(rfstation.phi_s[k]) + (phi - rfstation.phi_s[k]) * np.sin(
                              rfstation.phi_s[k]))

    Xt = rfstation.omega_rf[0, k] * X + rfstation.phi_rf_d[0, k]
    Z = hphi_DE(Xt) + hE_t(Y)

    return Z, hphi_DE(np.pi - rfstation.phi_s[k])


def plot_hamiltonian(ring, rfstation, beam, dt, dE, k = int(0), hamiltonian_energy = None, n_points = 1001, n_lines = 100, separatrix_flag = True, get_data_animation = False, frame_path = '',
                     directory = '', option = ''):
    dt_array = np.linspace(-dt, dt, n_points)
    dE_array = np.linspace(-dE, dE, n_points)
    plt.figure()
    hE_t = lambda DE: c * np.pi/ (ring.ring_circumference * beam.beta * beam.energy) * \
                      (ring.eta_0[0, k] * DE ** 2 + ring.eta_1[0, k] * DE ** 3 + ring.eta_2[0, k] * DE ** 4)

    hdelta_phi = lambda delta:   1 / 2 * rfstation.harmonic * ring.eta_0[0, k] * delta ** 2 + \
                                 1 / 3 * rfstation.harmonic * ring.eta_1[0, k] * delta ** 3 + \
                                 1 / 4 * rfstation.harmonic * ring.eta_2[0, k] * delta ** 4

    hphi_delta = lambda phi: ring.Particle.charge * rfstation.voltage[k] / (2 * np.pi * ring.energy[0, k]) * (np.cos(phi) - np.cos(rfstation.phi_s[k]) + (phi - rfstation.phi_s[k])* np.sin(rfstation.phi_s[k]))

    hphi_DE = lambda phi: c * beam.beta * ring.Particle.charge * rfstation.voltage[0, k] / (rfstation.harmonic[0, k] * ring.ring_circumference) * \
                        (np.cos(phi) - np.cos(rfstation.phi_s[k]) + (phi - rfstation.phi_s[k]) * np.sin(rfstation.phi_s[k]))
    X, Y = np.meshgrid(dt_array, dE_array)
    Xt = rfstation.omega_rf[0,k] * X + rfstation.phi_rf_d[0,k]
    Z = hphi_DE(Xt) + hE_t(Y)

    #h_blond = separatrix(ring, rfstation, dt_array)
    #plt.plot(dt_array * 1e9, h_blond / 1e9, 'g--')
    if n_lines != 0:
        plt.contour(X*1e9, Y/1e9, Z, n_lines)
    if separatrix_flag:
        plt.contour(X*1e9, Y/1e9, Z, [hphi_DE(np.pi - rfstation.phi_s[k])], colors=['red'])
        if get_data_animation:
            plt.figure(figsize=(4, 4))
            plt.title(f'Z mode\n Turn: {k}')
            plt.xlabel('t [ns]')
            plt.ylabel('DE [GeV]')
            plt.xlim([0, 1.25])

            dt = beam.dt[0:10000] * 1e9
            dE = beam.dE[0:10000] * 1e-9
            plt.contour(X * 1e9, Y / 1e9, Z, [hphi_DE(np.pi - rfstation.phi_s[k])], colors=['red'])
            plt.scatter(dt, dE, s = 0.2)
            plt.savefig(frame_path)
            plt.close()
            plt.close()
    if hamiltonian_energy is not None:
        for energy in hamiltonian_energy:
            plt.contour(X*1e9, Y/1e9, Z, [energy])
    plt.xlabel('t [ns]')
    #plt.xlim([0, (2 * np.pi - rfstation.phi_rf_d[0,k])/rfstation.omega_rf[0,k]])
    plt.ylabel('DE [GeV]')
    plt.scatter(beam.dt*1e9, beam.dE/1e9, s = 0.2)
    plt.title(f'Turns {k}')
    text = directory + '/plot_' + str(k) + option
    plt.savefig(text)
    plt.close()