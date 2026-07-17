
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Numerical diffusion model based on Ivanov (1992). Stationary single-harmonic
RF bucket is considered.**

:Authors: **Helga Timko**
'''

from __future__ import division

from builtins import range

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as int
from pylab import cm
from scipy.special import ellipk

from .action import x2, action_from_phase_amplitude


def phase_noise_diffusion(Ring, RFStation, spectrum, distribution,
                          distributionBins, Ngrids=200, M=1,
                          iterations=100000, figdir=None):
    '''
    Calculate diffusion in action space according to a given double-sided phase
    noise spectrum, on a uniform grid in oscillation amplitude.
    The spectrum is defined on the grid points (Ngrids + 1 points) for all M.
    The particle distribution in action is defined on the grids (Ngrids points). 
    Returns the diffused action distribution.
    Optional: define number of side-bands (M) to be taken into account, default
    is M = 1; N.B. this will only give impair modes for phase noise. 
    Optional: number of iterations to track.
    Optional: save figures into directory 'figdir'.
    '''

    # Input check
    N = Ngrids
    if spectrum.shape != (M, N + 1):
        # NoiseDiffusionError
        raise RuntimeError("In phase_noise_diffusion(): spectrum has to have shape (M, Ngrids+1)!")
    if len(distribution) != N:
        # NoiseDiffusionError
        raise RuntimeError("In phase_noise_diffusion(): distribution has to be an array of Ngrids elements!")

    # Some constants
    T0 = Ring.t_rev[0]
    omega_s0 = RFStation.omega_s0[0]
    h = RFStation.harmonic[0, 0]
    Jsep = 8. * omega_s0 / (np.pi * h**2)  # Action at the separatrix

    # Settings for plots
    plt.rc('axes', labelsize=16, labelweight='normal')
    plt.rc('lines', linewidth=1.5, markersize=6)
    plt.rc('font', family='sans-serif')
    plt.rc('legend', fontsize=12)

    # Construct action grid
    phimax = np.linspace(0., np.pi, N + 1, endpoint=True)
    xx = x2(phimax)
    J = action_from_phase_amplitude(xx)
    dJ = J[1:] - J[:-1]  # Differential on grid
    Jav = 0.5 * (J[1:] + J[:-1])  # Average on grid

    # Interpolate distribution
    distributionInterp = np.interp(Jav, distributionBins, distribution)

    # Normalise distribution
    distributionInterp /= int.simps(distributionInterp, Jav)

    # Construct weighting function
    Wm = np.zeros((M, N + 1))
    for k in range(0, M):
        m = 2 * k + 1
        Wm[k][:] = (np.pi * m / ellipk(xx))**4 / \
                   (4. * np.cosh(0.5 * np.pi * m * ellipk(1 - xx) / ellipk(xx))**2)

    # Diffusion coefficient for stationary bucket, according to Ivanov
    # Twice the sum over positive frequencies for double-sided spectrum
    D = (omega_s0 / h)**4 * np.sum(Wm * spectrum, axis=0)
    Dav = 0.5 * (D[1:] + D[:-1])  # Average on grid

    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    ax.plot(J, D)
    ax.set_xlabel(r"Relative action (J/J$_{\mathrm{sep}}$)")
    ax.set_ylabel(r"Diffusion coefficient [rad$^2$/s$^3$]")
    if figdir:
        plt.savefig(figdir + "D_vs_J.png")
        plt.clf()
    else:
        plt.show()

    ax = plt.axes([0.15, 0.1, 0.8, 0.8])
    for k in range(0, M):
        ax.plot(J, Wm[k], color=cm.get_cmap('jet')(k / M),
                label="m=%d mode" % (2 * k + 1))
    ax.set_xlabel(r"Relative action (J/J$_{\mathrm{sep}}$)")
    ax.set_ylabel(r"Weight function W$_m$ [1]")
    plt.legend(loc=0)
    if figdir:
        plt.savefig(figdir + "Wm_vs_J.png")
        plt.clf()
    else:
        plt.show()

    # Discretised diffusion equation ------------------------------------------
    A = np.zeros((N, N), float)
    for i in range(1, N - 1):
        A[i, i - 1] = dJ[i - 1]
        A[i, i] = 2. * (dJ[i - 1] + dJ[i])
        A[i, i + 1] = dJ[i]
    A[0, 0] = 2. * dJ[0]
    A[0, 1] = dJ[0]
    A[N - 1, N - 2] = dJ[N - 2]
    A[N - 1, N - 1] = 2. * (dJ[N - 2] + dJ[N - 1])
    A *= Jsep / 6.

    B = np.zeros((N, N), float)
    for i in range(1, N - 1):
        B[i, i - 1] = -Dav[i - 1] / dJ[i - 1]
        B[i, i] = Dav[i - 1] / dJ[i - 1] + Dav[i] / dJ[i]
        B[i, i + 1] = -Dav[i] / dJ[i]
    B[0, 0] = Dav[0] / dJ[0]
    B[0, 1] = -Dav[0] / dJ[0]
    B[N - 1, N - 2] = -Dav[N - 2] / dJ[N - 2]
    B[N - 1, N - 1] = Dav[N - 2] / dJ[N - 2] + Dav[N - 1] / dJ[N - 1]
    B /= Jsep

    # Time evolution ----------------------------------------------------------
    M1 = A - T0 * B / 2.
    M2 = A + T0 * B / 2.
    M1 = np.matrix(M1)
    M2 = np.matrix(M2)
    F = np.matrix(distributionInterp)
    Mtot = np.dot(M2.I, M1)
    Fold = F.T

    for i in range(0, iterations):

        Fnew = np.dot(Mtot, Fold)
        Fold = Fnew

    # Back to array for post-processing
    Fnew = Fnew.T
    F = np.array(F, order=0)
    Fnew = np.array(Fnew, order=0)

    # Plot --------------------------------------------------------------------
    # Distributions along action J
    norm_J_i = int.simps(F[0], Jav)
    norm_J_f = int.simps(Fnew[0], Jav)
    J_av_i = int.simps(F[0] * Jav, Jav) / norm_J_i
    J_av_f = int.simps(Fnew[0] * Jav, Jav) / norm_J_f

    # Conversion in SHORT-BUNCH APPROXIMATION!!!
    sigma_phi_i = np.sqrt(8. / np.pi * J_av_i)
    sigma_phi_f = np.sqrt(8. / np.pi * J_av_f)
    tau_i = sigma_phi_i * 2 * T0 / h / np.pi * 1.e9
    tau_f = sigma_phi_f * 2 * T0 / h / np.pi * 1.e9

    ax = plt.axes([0.12, 0.1, 0.78, 0.8])
    ax.plot(Jav, F[0], "b", label="Initial distribution")
    ax.plot(Jav, Fnew[0], "r", label="Final distribution")
    ax.set_xlabel(r"Relative action (J/J$_{\mathrm{sep}}$)")
    ax.set_ylabel("Particle distribution [1]")
    plt.legend(loc=1)
    ax2 = plt.twinx(ax)
    for k in range(0, M):
        ax2.plot(J, spectrum[k], color=cm.get_cmap('jet')(k / M), alpha=0.5)
        ax2.fill_between(J, 0, spectrum[k], color=cm.get_cmap('jet')(k / M),
                         alpha=0.2)
    ax2.set_ylabel(r"Double-sided spectral density [rad$^2$/Hz]")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.figtext(0.6, 0.7, r'$\int_0^{J_{sep}}{F_i(J)}dJ=$ %.3f' % norm_J_i,
                fontsize=14, ha='left', va='center')
    plt.figtext(0.6, 0.625, r'$\int_0^{J_{sep}}{F_f(J)}dJ=$ %.3f' % norm_J_f,
                fontsize=14, ha='left', va='center')
    plt.figtext(0.6, 0.55, r'$<J>_i=$ %.4e s' % J_av_i, fontsize=14, ha='left',
                va='center')
    plt.figtext(0.6, 0.5, r'$<J>_f=$ %.4e s' % J_av_f, fontsize=14, ha='left',
                va='center')
    plt.figtext(0.4, 0.4, "Converting in short-bunch approximation...",
                fontsize=12, ha='left', va='center')
    plt.figtext(0.6, 0.35, r'$\sigma_{\varphi}^{(i)}=$ %.4f rad' % sigma_phi_i,
                fontsize=12, ha='left', va='center')
    plt.figtext(0.6, 0.3, r'$\sigma_{\varphi}^{(f)}=$ %.4f rad' % sigma_phi_f,
                fontsize=12, ha='left', va='center')
    plt.figtext(0.6, 0.25, r'$\tau_{4\sigma}^{(i)}=$ %.4f ns' % tau_i, fontsize=12,
                ha='left', va='center')
    plt.figtext(0.6, 0.2, r'$\tau_{4\sigma}^{(f)}=$ %.4f ns' % tau_f, fontsize=12,
                ha='left', va='center')
    if figdir:
        plt.savefig(figdir + "F_vs_J.png")
        plt.clf()
    else:
        plt.show()

    return Jav, distributionInterp, Fnew[0]
