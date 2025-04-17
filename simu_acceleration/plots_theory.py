import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e
from blond.trackers.utilities import hamiltonian
import matplotlib.animation as ani
import matplotlib.pyplot as plt


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


def plot_hamiltonian(ring, rfstation, beam, dt, dE, k = int(0), hamiltonian_energy = None, n_points = 1001, n_lines = 100, separatrix = True, directory = 'output_figs_wigglers', option = ''):
    dt_array = np.linspace(-dt, dt, n_points)
    dE_array = np.linspace(-dE, dE, n_points)
    plt.figure()
    #plt.xlim([-1, 1])
    hE_t = lambda DE: c * np.pi/ (ring.ring_circumference * beam.beta * beam.energy) * \
                      (
                        1/2 * ring.eta_0[0, k] * DE ** 2 + 1/3 * ring.eta_1[0, k] * DE ** 3 + 1/4 * ring.eta_2[0, k] * DE ** 4)

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
    if hamiltonian_energy is not None:
        for energy in hamiltonian_energy:
            plt.contour(X*1e9, Y/1e9, Z, [energy])
    plt.xlabel('t [ns]')
    #plt.xlim([0, (2 * np.pi - rfstation.phi_rf_d[0,k])/rfstation.omega_rf[0,k]])
    plt.ylabel('DE [GeV]')
    plt.scatter(-beam.dt*1e9, beam.dE/1e9, s = 0.2)

    plt.title(f'Turns {k}')
    text = directory + '/plot_' + str(k) + option
    plt.savefig(text)
    plt.close()



def animated_plot_tracking(ring, rfcav, beam, map_, dt, dE, n_points = 1001, saving_file='animated_tracking_Z_mode_electrons', option = ''):
    global C, scat
    fig, ax = plt.subplots()
    ax.set_xlim([-dt*1e9, dt*1e9/2])
    ax.set_ylim([-2.0, 2.0])
    scat = ax.scatter(-beam.dt*1e9,beam.dE/1e9, s = 0.2, color = 'blue')
    dt_array = np.linspace(-dt, dt, n_points)
    dE_array = np.linspace(-dE, dE, n_points)
    X, Y = np.meshgrid(dt_array, dE_array)
    Z, hamiltonian_energy = get_hamiltonian(ring, rfcav, beam, X, Y, k=0)
    C = ax.contour(X*1e9, Y/1e9, Z, [hamiltonian_energy], colors='red')

    def animate(i, ax):
        for m in map_:
            m.track()
        beam.statistics()
        if (i%10)==0:
            for coll in ax.collections:
                coll.remove()
            Z, hamiltonian_energy = get_hamiltonian(ring, rfcav, beam, X, Y, k=i)
            ax.scatter(-beam.dt * 1e9, beam.dE / 1e9, s=0.2, color='cyan')
            ax.contour(X*1e9, Y/1e9, Z, [hamiltonian_energy], colors='red')
            #scat.set_offsets(np.column_stack([-beam.dt*1e9, beam.dE/1e9]))
            #ax.scatter(-beam.dt * 1e9, beam.dE / 1e9, s=0.2, color = 'blue')
            fig.suptitle(f'Turn: {i}')
        return ax

    animated_fig = ani.FuncAnimation(fig, animate, fargs = {ax},repeat=True, frames = ring.n_turns+1, interval = ring.t_rev[0]*1e3/10)
    plt.show()
    writer = ani.PillowWriter(fps=15,
                                     metadata=dict(artist='Me'),
                                     bitrate=1800)
    #animated_fig.to_html5_video()
    #animated_fig.to_jshtml()
    animated_fig.save(saving_file+option+'.gif', writer=writer)



