# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Filters and methods for control loops**

:Authors: **Birk Emil Karlsen-Bæck**, **Helga Timko**
"""

from __future__ import annotations

# Set up logging
import logging

import numpy as np
from scipy.constants import c
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


def cavity_response_sparse_matrix(
    I_beam,
    I_gen,
    n_samples,
    V_ant_init,
    I_gen_init,
    samples_per_rf,
    R_over_Q,
    Q_L,
    detuning,
):
    """Solving the ACS cavity response model as a sparse matrix problem
    for a given set of initial conditions, resonator parameters and
    generator and RF beam currents.

    Parameters
    ----------
    I_beam : complex array
        RF beam current
    I_gen : complex array
        Generator current
    n_samples : int
        Number of samples of the result array - 1
    V_ant_init : complex float
        Initial condition for the antenna voltage
    I_gen_init : complex float
        Initial condition of the generator current, i.e. one sample before the I_gen array
    samples_per_rf : int
        Number of samples per RF period
    R_over_Q : float
        The R over Q of the cavity
    Q_L : float
        The loaded quality factor of the cavity
    detuning : float
        The detuning of the cavity in frequency divided by the rf frequency

    Returns
    -------
    complex array
        The antenna voltage evaluated for the same period as I_beam and I_gen of length n_samples + 1

    """

    # Add a zero at the start of RF beam current
    if len(I_beam) != n_samples + 1:
        I_beam = np.concatenate((np.zeros(1, dtype=complex), I_beam))

    # Check length of the generator current array
    if len(I_gen) != n_samples + 1:
        I_gen = np.concatenate((I_gen_init * np.ones(1, dtype=complex), I_gen))

    # Compute matrix elements
    A = 0.5 * R_over_Q * samples_per_rf
    B = 1 - 0.5 * samples_per_rf / Q_L + 1j * detuning * samples_per_rf

    # Initialize the two sparse matrices needed to find antenna voltage
    B_matrix = diags(
        [-B, 1], [-1, 0], (n_samples + 1, n_samples + 1), dtype=complex, format="csc"
    )
    I_matrix = diags([A], [-1], (n_samples + 1, n_samples + 1), dtype=complex)

    # Find vector on the "current" side of the equation
    b = I_matrix.dot(2 * I_gen - I_beam)
    b[0] = V_ant_init

    # Solve the sparse linear system of equations and return
    return spsolve(B_matrix, b)


def rectangle(t, tau):
    r"""Rectangular function of time

    .. math:: \mathsf{rect} \left( \frac{t}{\tau} \right) =
        \begin{cases}
            1 \, , \, t \in (-\tau/2, \tau/2) \\
            0.5 \, , \, t = \pm \tau/2 \\
            0 \, , \, \textsf{otherwise}
        \end{cases}

    Parameters
    ----------
    t : float array
        Time array
    tau : float
        Time window of rectangular function

    Returns
    -------
    float array
        Rectangular function for given time array

    """

    dt = t[1] - t[0]
    llimit = np.where(np.fabs(t + tau / 2) < dt / 2)[0]
    ulimit = np.where(np.fabs(t - tau / 2) < dt / 2)[0]
    if len(llimit) != 1:
        # ImpulseError
        raise RuntimeError(
            "ERROR in impulse_response.rectangle(): time"
            + " array doesn't start at rising edge!"
        )
    if len(ulimit) not in [0, 1]:
        # ImpulseError
        raise RuntimeError(
            "ERROR in impulse_response.rectangle(): time"
            + " array has multiple falling edges!"
        )
    logger.debug("In rectangle(), index of rising edge is %d" % llimit[0])
    y = np.zeros(len(t))
    y[llimit[0]] = 0.5
    if len(ulimit) == 1:
        y[llimit[0] + 1 : ulimit[0]] = np.ones(ulimit[0] - llimit[0] - 1)
        y[ulimit[0]] = 0.5
    else:
        y[llimit[0] + 1 :] = 1

    return y


def triangle(t, tau):
    r"""Triangular function of time

    .. math:: \mathsf{tri} \left( \frac{t}{\tau} \right) =
        \begin{cases}
            1 - \frac{t}{\tau}\, , \, t \in (0, \tau) \\
            0.5 \, , \, t = 0 \\
            0 \, , \, \textsf{otherwise}
        \end{cases}

    Parameters
    ----------
    t : float array
        Time array
    tau : float
        Time window of rectangular function

    Returns
    -------
    float array
        Triangular function for given time array

    """

    dt = t[1] - t[0]
    llimit = np.where(np.fabs(t) < dt / 2)[0]
    logger.debug("In triangle(), index of rising edge is %d" % llimit[0])
    if len(llimit) != 1:
        # ImpulseError
        raise RuntimeError(
            "ERROR in impulse_response.triangle(): time"
            + " array doesn't start at rising edge!"
        )
    y = np.zeros(len(t))
    y[llimit[0]] = 0.5
    y[llimit[0] + 1 :] = 1 - t[llimit[0] + 1 :] / tau
    y[np.where(y < 0)[0]] = 0

    return y


class TravellingWaveCavity:
    r"""Impulse responses of a travelling wave cavity. The induced voltage
    :math:`V(t)` from the impulse response :math:`h(t)` and the I,Q (cavity or
    generator) current :math:`I(t)` can be written in matrix form,

    .. math::
        \left( \begin{matrix} V_I(t) \\
        V_Q(t) \end{matrix} \right)
        = \left( \begin{matrix} h_s(t) & - h_c(t) \\
        h_c(t) & h_s(t) \end{matrix} \right)
        * \left( \begin{matrix} I_I(t) \\
        I_Q(t) \end{matrix} \right) \, ,

    where :math:`*` denotes convolution,
    :math:`h(t)*x(t) = \int d\tau h(\tau)x(t-\tau)`.

    For the **cavity-to-beam induced voltage**, we define

    .. math::
        R_b \equiv \frac{\rho l^2}{8} \,

    where :math:`\rho` is the series impedance, :math:`l` the accelerating
    length, :math:`\tau` the filling time. The cavity-to-beam wake is

    .. math::
        W_b(t) = \frac{4 R_b}{\tau} \mathsf{tri}\left(\frac{t}{\tau}\right)
         \cos(\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \mathsf{tri}\left(\frac{t}{\tau}\right)
         \cos((\omega_c - \omega_r)t) \, , \\
        h_{c,b}(t) &= \frac{2 R_b}{\tau} \mathsf{tri}\left(\frac{t}{\tau}\right)
        \sin((\omega_c - \omega_r)t) \, ,

    where :math:`\mathsf{tri}(x)` is the triangular function, :math:`\omega_r`
    is the central revolution frequency of the cavity, and :math:`\omega_c` is
    the carrier revolution frequency of the I,Q demodulated current signal. On
    the carrier frequency, :math:`\omega_c = \omega_r`,

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \mathsf{tri}\left(\frac{t}{\tau}\right) \\
        h_{c,b}(t) &= 0 \, .

    For the **cavity-to-generator induced voltage**, we define

    .. math::
        R_g \equiv l \sqrt{\frac{\rho Z_0}{2}} \,

    where :math:`Z_0` is the shunt impedance when measuring the generator
    current; assumed to be 50 :math:`\Omega`. The cavity-to-generator wake is

    .. math::
        W_g(t) = \frac{2 R_g}{\tau} \mathsf{rect}\left(\frac{t}{\tau}\right)
        \cos(\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \mathsf{rect}\left(\frac{t}{\tau}\right)
        \cos((\omega_c - \omega_r)t) \, , \\
        h_{c,g}(t) &= \frac{R_g}{\tau} \mathsf{rect}\left(\frac{t}{\tau}\right)
        \sin((\omega_c - \omega_r)t) \, ,

    where :math:`\mathsf{rect}(x)` is the rectangular function. On the carrier
    frequency, :math:`\omega_c = \omega_r`,

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \mathsf{rect}\left(\frac{t}{\tau}\right) \\
        h_{c,g}(t) &= 0 \, .

    Parameters
    ----------
    l_cell : float
        Cavity cell length [m]
    N_cells : int
        Number of accelerating (interacting) cells in a cavity
    rho : float
        Series impedance [Ohms/m^2] of the cavity
    v_g : float
        Group velocity [c] in units of the speed of light
    omega_r : flaot
        Central (resonance) revolution frequency [1/s] of the cavity

    Attributes
    ----------
    Z_0 : float
        Shunt impedance of generator current measurement; assumed to be 50 Ohms
    R_beam : float
        :math:`R_b` [\Omega] as defined above
    R_gen : float
        :math:`R_g` [\Omega] as defined above
    l_cav : float
        Length [m] of the interaction region
    tau : float
        Cavity filling time [s]

    """

    def __init__(
        self,
        l_cell: float,
        N_cells: int,
        rho: float,
        v_g: float,
        omega_r: float,
        df: float = 0,
    ):
        self.l_cell = float(l_cell)
        self.N_cells = int(N_cells)
        self.rho = float(rho)
        if v_g > 0 and v_g < 1:
            self.v_g = float(v_g)
        else:
            # ImpulseError
            raise RuntimeError(
                "ERROR in TravellingWaveCavity: group"
                + " velocity out of limits (0,1)!"
            )
        self.omega_r = float(omega_r) + 2 * np.pi * float(df)

        # Calculated
        self.l_cav = float(self.l_cell * self.N_cells)
        # v_g opposite to wave!
        self.tau = self.l_cav / (self.v_g * c) * (1 + self.v_g)

        # Assumed impedance for measurement of generator current
        self.Z_0 = 50
        # Shunt impedances towards beam and generator
        self.R_beam = 0.125 * self.rho * self.l_cav**2
        self.R_gen = self.l_cav * np.sqrt(0.5 * self.rho * self.Z_0)

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")
        self.logger.debug("Filling time %.4e s", self.tau)

    def impulse_response_gen(self, omega_c: float, time_coarse: float):
        r"""Impulse response from the cavity towards the
        generator. For a signal that is I,Q demodulated at a given carrier
        frequency :math:`\omega_c`. The formulae assume that the carrier
        frequency is be close to the central frequency
        :math:`\omega_c/\omega_r \ll 1` and that the signal is low-pass
        filtered (i.e.\ high-frequency components can be neglected).

        Parameters
        ----------
        omega_c : float
            Carrier revolution frequency [1/s]
        time_coarse : float
            Time array of the LLRF to act on

        Attributes
        ----------
        d_omega : float
            :math:`\omega_c - \omega_r` [1/s]
        t_gen : float array
            time array for generator wake and impulse response; starts from
            :math:`- \tau/2`
        h_gen : complex array
            :math:`h_{s,b}(t) + i*h_{c,b}(t)` [\Omega/s] as defined above
        """

        self.omega_c = float(omega_c)
        self.d_omega = self.omega_c - self.omega_r
        if np.fabs((self.d_omega) / self.omega_r) > 0.1:
            # ImpulseError
            raise RuntimeError(
                "ERROR in TravellingWaveCavity"
                + " impulse_response(): carrier frequency"
                + " should be close to central frequency of the"
                + " cavity!"
            )

        # Move starting point of impulse response to correct value
        t_gen = time_coarse - time_coarse[0]

        # Impulse response if on carrier frequency
        self.h_gen = (
            self.R_gen / self.tau * rectangle(t_gen - 0.5 * self.tau, self.tau)
        ).astype(np.complex128)

        # Impulse response if not on carrier frequency
        if np.fabs((self.d_omega) / self.omega_r) > 1e-12:
            self.h_gen = self.h_gen.real * (
                np.cos(self.d_omega * t_gen) - 1j * np.sin(self.d_omega * t_gen)
            )

    def impulse_response_beam(
        self, omega_c: float, time_fine: float, time_coarse: float = None
    ):
        r"""Impulse response from the cavity towards the beam. For a signal
        that is I,Q demodulated at a given carrier
        frequency :math:`\omega_c`. The formulae assume that the carrier
        frequency is be close to the central frequency
        :math:`\omega_c/\omega_r \ll 1` and that the signal is low-pass
        filtered (i.e.\ high-frequency components can be neglected).

        Parameters
        ----------
        omega_c : float
            Carrier revolution frequency [1/s]
        time_fine : float
            Time array of the beam profile to act on
        time_coarse : float
            Time array of the LLRF to act on; default is None

        Attributes
        ----------
        d_omega : float
            :math:`\omega_c - \omega_r` [1/s]
        t_beam : float array
            time array for beam wake and impulse response; starts from zero
        h_beam : complex array
            :math:`h_{s,b}(t) + i*h_{c,b}(t)` [\Omega/s] as defined above
        h_beam_coarse : complex array
            Impulse response evaluated on the coarse grid
        """

        self.omega_c = float(omega_c)
        self.d_omega = self.omega_c - self.omega_r
        if np.fabs((self.d_omega) / self.omega_r) > 0.1:
            raise RuntimeError(
                "ERROR in TravellingWaveCavity"
                + " impulse_response(): carrier frequency"
                + " should be close to central frequency of the"
                + " cavity!"
            )

        # Move starting point of impulse response to correct value
        t_beam = time_fine - time_fine[0]

        # Impulse response if on carrier frequency
        self.h_beam = (-2 * self.R_beam / self.tau * triangle(t_beam, self.tau)).astype(
            np.complex128
        )

        # Impulse response if not on carrier frequency
        if np.fabs((self.d_omega) / self.omega_r) > 1e-12:
            self.h_beam = self.h_beam.real * (
                np.cos(self.d_omega * t_beam) - 1j * np.sin(self.d_omega * t_beam)
            )

        if time_coarse is not None:
            # Move starting point of impulse response to correct value
            t_beam = time_coarse - time_coarse[0]

            # Impulse response if on carrier frequency
            self.h_beam_coarse = (
                -2 * self.R_beam / self.tau * triangle(t_beam, self.tau)
            ).astype(np.complex128)

            # Impulse response if not on carrier frequency
            if np.fabs((self.d_omega) / self.omega_r) > 1e-12:
                self.h_beam_coarse = self.h_beam_coarse.real * (
                    np.cos(self.d_omega * t_beam) - 1j * np.sin(self.d_omega * t_beam)
                )

    def compute_wakes(self, time: float):
        r"""Computes the wake fields towards the beam and generator on the
        central cavity frequency.

        Parameters
        ----------
        time_beam : float
            Time array of the beam to act on
        time_gen : float
            Time array of the generator to act on

        Attributes
        ----------
        W_beam : float array
            :math:`W_b(t)` [\Omega/s] as defined above
        W_gen : float array
            :math:`W_g(t)` [\Omega/s] as defined above

        """

        t_beam = time - time[0]
        t_gen = time - time[0] - 0.5 * self.tau

        # Wake fields towards beam and generator
        self.W_beam = 2 * self.h_beam.real * np.cos(self.omega_r * t_beam)
        self.W_gen = 2 * self.h_gen.real * np.cos(self.omega_r * t_gen)


class SPS3Section200MHzTWC(TravellingWaveCavity):
    def __init__(self, df: float = 0):
        TravellingWaveCavity.__init__(
            self, 0.374, 32, 2.71e4, 0.0946, 2 * np.pi * 200.03766667e6, df=df
        )


class SPS4Section200MHzTWC(TravellingWaveCavity):
    def __init__(self, df: float = 0):
        TravellingWaveCavity.__init__(
            self, 0.374, 43, 2.71e4, 0.0946, 2 * np.pi * 199.9945e6, df=df
        )


class SPS5Section200MHzTWC(TravellingWaveCavity):
    def __init__(self, df: float = 0):
        TravellingWaveCavity.__init__(
            self, 0.374, 54, 2.71e4, 0.0946, 2 * np.pi * 200.1e6, df=df
        )
