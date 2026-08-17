# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
Filters and methods for control loops.

Notes
-----
Authors:
Birk Emil Karlsen-Bæck
Helga Timko
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

logger = logging.getLogger(__name__)


def rectangle(t: NumpyArray, tau: float) -> NumpyArray:
    r"""
    Rectangular function of time.

    .. math:: \\mathsf{rect} \\left( \frac{t}{\tau} \right) =
        \begin{cases}
            1 \\, , \\, t \\in (-\tau/2, \tau/2) \\
            0.5 \\, , \\, t = \\pm \tau/2 \\
            0 \\, , \\, \textsf{otherwise}
        \\end{cases}

    Parameters
    ----------
    t
        Time array.
    tau
        Time window of rectangular function.

    Returns
    -------
    float array
        Rectangular function for given time array.
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
    logger.debug(f"In rectangle(), index of rising edge is {llimit[0]}")
    y = np.zeros(len(t))
    y[llimit[0]] = 0.5
    if len(ulimit) == 1:
        y[llimit[0] + 1 : ulimit[0]] = np.ones(ulimit[0] - llimit[0] - 1)
        y[ulimit[0]] = 0.5
    else:
        y[llimit[0] + 1 :] = 1

    return y


def triangle(t: NumpyArray, tau: float) -> NumpyArray:
    r"""
    Triangular function of time.

    .. math:: \\mathsf{tri} \\left( \frac{t}{\tau} \right) =
        \begin{cases}
            1 - \frac{t}{\tau}\\, , \\, t \\in (0, \tau) \\
            0.5 \\, , \\, t = 0 \\
            0 \\, , \\, \textsf{otherwise}
        \\end{cases}

    Parameters
    ----------
    t
        Time array.
    tau
        Time window of rectangular function.

    Returns
    -------
    float array
        Triangular function for given time array.
    """
    dt = t[1] - t[0]
    llimit = np.where(np.fabs(t) < dt / 2)[0]
    logger.debug(f"In triangle(), index of rising edge is {llimit[0]}")
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
    r"""
    Impulse responses of a travelling wave cavity.

    The induced voltage :math:`V(t)` from the impulse response :math:`h(t)`
    and the I,Q (cavity or generator) current :math:`I(t)` can be written
    in matrix form,

    .. math::
        \\left( \begin{matrix} V_I(t) \\
        V_Q(t) \\end{matrix} \right)
        = \\left( \begin{matrix} h_s(t) & - h_c(t) \\
        h_c(t) & h_s(t) \\end{matrix} \right)
        * \\left( \begin{matrix} I_I(t) \\
        I_Q(t) \\end{matrix} \right) \\, ,

    where :math:`*` denotes convolution,
    :math:`h(t)*x(t) = \\int d\tau h(\tau)x(t-\tau)`.

    For the **cavity-to-beam induced voltage**, we define

    .. math::
        R_b \\equiv \frac{\rho l^2}{8} \\,

    where :math:`\rho` is the series impedance, :math:`l` the accelerating
    length, :math:`\tau` the filling time. The cavity-to-beam wake is

    .. math::
        W_b(t) = \frac{4 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{tri}(x)` is the triangular function, :math:`\\omega_r`
    is the central revolution frequency of the cavity, and :math:`\\omega_c` is
    the carrier revolution frequency of the I,Q demodulated current signal. On
    the carrier frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right) \\
        h_{c,b}(t) &= 0 \\, .

    For the **cavity-to-generator induced voltage**, we define

    .. math::
        R_g \\equiv l \\sqrt{\frac{\rho Z_0}{2}} \\,

    where :math:`Z_0` is the shunt impedance when measuring the generator
    current; assumed to be 50 :math:`\\Omega`. The cavity-to-generator wake is

    .. math::
        W_g(t) = \frac{2 R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{rect}(x)` is the rectangular function. On the carrier
    frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right) \\
        h_{c,g}(t) &= 0 \\, .

    Parameters
    ----------
    l_cell
        Cavity cell length [m].
    n_cells
        Number of accelerating (interacting) cells in a cavity.
    rho
        Series impedance [Ohms/m^2] of the cavity.
    v_g
        Group velocity [c] in units of the speed of light.
    omega_r
        Central (resonance) revolution frequency [1/s] of the cavity.
    df
        Option to add a shift to the central frequency.
    """

    def __init__(
        self,
        l_cell: float,
        n_cells: int,
        rho: float,
        v_g: float,
        omega_r: float,
        df: float = 0,
    ):
        self.l_cell = float(l_cell)
        self.N_cells = int(n_cells)
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
        self.logger.debug(f"Filling time {self.tau:.4e}%.4e s")

    def impulse_response_gen(self, omega_c: float, time_coarse: NumpyArray):
        r"""
        Impulse response from the cavity towards the generator.

        For a signal that is I,Q demodulated at a given carrier
        frequency :math:`\omega_c`. The formulae assume that the carrier
        frequency is close to the central frequency
        :math:`\omega_c/\omega_r \ll 1` and that the signal is low-pass
        filtered (i.e.\ high-frequency components can be neglected).

        Parameters
        ----------
        omega_c
            Carrier revolution frequency [1/s].
        time_coarse
            Time array of the LLRF to act on.
        """
        h_thres = 1e-12
        freq_thres = 0.1

        self.omega_c = float(omega_c)
        self.d_omega = self.omega_c - self.omega_r
        if np.fabs((self.d_omega) / self.omega_r) > freq_thres:
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
        if np.fabs((self.d_omega) / self.omega_r) > h_thres:
            self.h_gen = self.h_gen.real * (
                np.cos(self.d_omega * t_gen)
                - 1j * np.sin(self.d_omega * t_gen)
            )

    def impulse_response_beam(
        self,
        omega_c: float,
        time_fine: NumpyArray,
        time_coarse: NumpyArray | None = None,
    ):
        r"""
        Impulse response from the cavity towards the beam.

        For a signal that is I,Q demodulated at a given carrier
        frequency :math:`\omega_c`. The formulae assume that the carrier
        frequency is be close to the central frequency
        :math:`\omega_c/\omega_r \ll 1` and that the signal is low-pass
        filtered (i.e.\ high-frequency components can be neglected).

        Parameters
        ----------
        omega_c : float
            Carrier revolution frequency [1/s].
        time_fine : NumpyArray
            Time array of the beam profile to act on.
        time_coarse : NumpyArray
            Time array of the LLRF to act on; default is None.
        """
        h_thres = 1e-12
        freq_thres = 0.1

        self.omega_c = float(omega_c)
        self.d_omega = self.omega_c - self.omega_r
        if np.fabs((self.d_omega) / self.omega_r) > freq_thres:
            raise RuntimeError(
                "ERROR in TravellingWaveCavity"
                + " impulse_response(): carrier frequency"
                + " should be close to central frequency of the"
                + " cavity!"
            )

        # Move starting point of impulse response to correct value
        t_beam = time_fine - time_fine[0]

        # Impulse response if on carrier frequency
        self.h_beam = (
            -2 * self.R_beam / self.tau * triangle(t_beam, self.tau)
        ).astype(np.complex128)

        # Impulse response if not on carrier frequency
        if np.fabs((self.d_omega) / self.omega_r) > h_thres:
            self.h_beam = self.h_beam.real * (
                np.cos(self.d_omega * t_beam)
                - 1j * np.sin(self.d_omega * t_beam)
            )

        if time_coarse is not None:
            # Move starting point of impulse response to correct value
            t_beam = time_coarse - time_coarse[0]

            # Impulse response if on carrier frequency
            self.h_beam_coarse = (
                -2 * self.R_beam / self.tau * triangle(t_beam, self.tau)
            ).astype(np.complex128)

            # Impulse response if not on carrier frequency
            if np.fabs((self.d_omega) / self.omega_r) > h_thres:
                self.h_beam_coarse = self.h_beam_coarse.real * (
                    np.cos(self.d_omega * t_beam)
                    - 1j * np.sin(self.d_omega * t_beam)
                )

    def compute_wakes(self, time: NumpyArray):
        """
        Compute the wake fields towards the beam and generator on the central cavity frequency.

        Parameters
        ----------
        time
            Values in time [s] to compute the wake on.
        """
        t_beam = time - time[0]
        t_gen = time - time[0] - 0.5 * self.tau

        # Wake fields towards beam and generator
        self.W_beam = 2 * self.h_beam.real * np.cos(self.omega_r * t_beam)
        self.W_gen = 2 * self.h_gen.real * np.cos(self.omega_r * t_gen)


class SPS3Section200MHzTWC(TravellingWaveCavity):
    r"""
    Impulse responses of a 3-section 200 MHz SPS travelling wave cavity.

    The induced voltage :math:`V(t)` from the impulse response :math:`h(t)`
    and the I,Q (cavity or generator) current :math:`I(t)` can be written
    in matrix form,

    .. math::
        \\left( \begin{matrix} V_I(t) \\
        V_Q(t) \\end{matrix} \right)
        = \\left( \begin{matrix} h_s(t) & - h_c(t) \\
        h_c(t) & h_s(t) \\end{matrix} \right)
        * \\left( \begin{matrix} I_I(t) \\
        I_Q(t) \\end{matrix} \right) \\, ,

    where :math:`*` denotes convolution,
    :math:`h(t)*x(t) = \\int d\tau h(\tau)x(t-\tau)`.

    For the **cavity-to-beam induced voltage**, we define

    .. math::
        R_b \\equiv \frac{\rho l^2}{8} \\,

    where :math:`\rho` is the series impedance, :math:`l` the accelerating
    length, :math:`\tau` the filling time. The cavity-to-beam wake is

    .. math::
        W_b(t) = \frac{4 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{tri}(x)` is the triangular function, :math:`\\omega_r`
    is the central revolution frequency of the cavity, and :math:`\\omega_c` is
    the carrier revolution frequency of the I,Q demodulated current signal. On
    the carrier frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right) \\
        h_{c,b}(t) &= 0 \\, .

    For the **cavity-to-generator induced voltage**, we define

    .. math::
        R_g \\equiv l \\sqrt{\frac{\rho Z_0}{2}} \\,

    where :math:`Z_0` is the shunt impedance when measuring the generator
    current; assumed to be 50 :math:`\\Omega`. The cavity-to-generator wake is

    .. math::
        W_g(t) = \frac{2 R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{rect}(x)` is the rectangular function. On the carrier
    frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right) \\
        h_{c,g}(t) &= 0 \\, .

    Parameters
    ----------
    df
        Option to add a shift to the central frequency.
    """

    def __init__(self, df: float = 0):
        super().__init__(
            0.374, 32, 2.71e4, 0.0946, 2 * np.pi * 200.03766667e6, df=df
        )


class SPS4Section200MHzTWC(TravellingWaveCavity):
    r"""
    Impulse responses of a 4-section 200 MHz SPS travelling wave cavity.

    The induced voltage :math:`V(t)` from the impulse response :math:`h(t)`
    and the I,Q (cavity or generator) current :math:`I(t)` can be written
    in matrix form,

    .. math::
        \\left( \begin{matrix} V_I(t) \\
        V_Q(t) \\end{matrix} \right)
        = \\left( \begin{matrix} h_s(t) & - h_c(t) \\
        h_c(t) & h_s(t) \\end{matrix} \right)
        * \\left( \begin{matrix} I_I(t) \\
        I_Q(t) \\end{matrix} \right) \\, ,

    where :math:`*` denotes convolution,
    :math:`h(t)*x(t) = \\int d\tau h(\tau)x(t-\tau)`.

    For the **cavity-to-beam induced voltage**, we define

    .. math::
        R_b \\equiv \frac{\rho l^2}{8} \\,

    where :math:`\rho` is the series impedance, :math:`l` the accelerating
    length, :math:`\tau` the filling time. The cavity-to-beam wake is

    .. math::
        W_b(t) = \frac{4 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{tri}(x)` is the triangular function, :math:`\\omega_r`
    is the central revolution frequency of the cavity, and :math:`\\omega_c` is
    the carrier revolution frequency of the I,Q demodulated current signal. On
    the carrier frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right) \\
        h_{c,b}(t) &= 0 \\, .

    For the **cavity-to-generator induced voltage**, we define

    .. math::
        R_g \\equiv l \\sqrt{\frac{\rho Z_0}{2}} \\,

    where :math:`Z_0` is the shunt impedance when measuring the generator
    current; assumed to be 50 :math:`\\Omega`. The cavity-to-generator wake is

    .. math::
        W_g(t) = \frac{2 R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{rect}(x)` is the rectangular function. On the carrier
    frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right) \\
        h_{c,g}(t) &= 0 \\, .

    Parameters
    ----------
    df
        Option to add a shift to the central frequency.
    """

    def __init__(self, df: float = 0):
        super().__init__(
            0.374, 43, 2.71e4, 0.0946, 2 * np.pi * 199.9945e6, df=df
        )


class SPS5Section200MHzTWC(TravellingWaveCavity):
    r"""
    Impulse responses of a 5-section 200 MHz SPS travelling wave cavity.

    The induced voltage :math:`V(t)` from the impulse response :math:`h(t)`
    and the I,Q (cavity or generator) current :math:`I(t)` can be written
    in matrix form,

    .. math::
        \\left( \begin{matrix} V_I(t) \\
        V_Q(t) \\end{matrix} \right)
        = \\left( \begin{matrix} h_s(t) & - h_c(t) \\
        h_c(t) & h_s(t) \\end{matrix} \right)
        * \\left( \begin{matrix} I_I(t) \\
        I_Q(t) \\end{matrix} \right) \\, ,

    where :math:`*` denotes convolution,
    :math:`h(t)*x(t) = \\int d\tau h(\tau)x(t-\tau)`.

    For the **cavity-to-beam induced voltage**, we define

    .. math::
        R_b \\equiv \frac{\rho l^2}{8} \\,

    where :math:`\rho` is the series impedance, :math:`l` the accelerating
    length, :math:`\tau` the filling time. The cavity-to-beam wake is

    .. math::
        W_b(t) = \frac{4 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
         \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{tri}(x)` is the triangular function, :math:`\\omega_r`
    is the central revolution frequency of the cavity, and :math:`\\omega_c` is
    the carrier revolution frequency of the I,Q demodulated current signal. On
    the carrier frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,b}(t) &= \frac{2 R_b}{\tau} \\mathsf{tri}\\left(\frac{t}{\tau}\right) \\
        h_{c,b}(t) &= 0 \\, .

    For the **cavity-to-generator induced voltage**, we define

    .. math::
        R_g \\equiv l \\sqrt{\frac{\rho Z_0}{2}} \\,

    where :math:`Z_0` is the shunt impedance when measuring the generator
    current; assumed to be 50 :math:`\\Omega`. The cavity-to-generator wake is

    .. math::
        W_g(t) = \frac{2 R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos(\\omega_r t)

    and the impulse response components are

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\cos((\\omega_c - \\omega_r)t) \\, , \\
        h_{c,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right)
        \\sin((\\omega_c - \\omega_r)t) \\, ,

    where :math:`\\mathsf{rect}(x)` is the rectangular function. On the carrier
    frequency, :math:`\\omega_c = \\omega_r`,

    .. math::
        h_{s,g}(t) &= \frac{R_g}{\tau} \\mathsf{rect}\\left(\frac{t}{\tau}\right) \\
        h_{c,g}(t) &= 0 \\, .

    Parameters
    ----------
    df
        Option to add a shift to the central frequency.
    """

    def __init__(self, df: float = 0):
        super().__init__(0.374, 54, 2.71e4, 0.0946, 2 * np.pi * 200.1e6, df=df)
