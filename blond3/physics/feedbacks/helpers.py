from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scipy
from numpy._typing import NDArray as NumpyArray
from scipy.constants import e

from ..._core.beam.base import BeamBaseClass

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional
    from typing import Optional

    from numpy.typing import NDArray as NumpyArray

    from numpy.typing import NDArray as NumpyArray

    from ...physics.profiles import StaticProfile

logger = logging.getLogger(__name__)


def low_pass_filter(signal: NumpyArray, cutoff_frequency: float = 0.5) -> NumpyArray:
    """
    Low-pass filter based on Butterworth 5th order digital filter

    Notes
    -----
    See `scipy`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    Parameters
    ----------
    signal : float array
        Signal to be filtered
    cutoff_frequency : float
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5

    Returns
    -------
    float array
        Low-pass filtered signal

    """

    b, a = scipy.signal.butter(5, cutoff_frequency, "low", analog=False)

    return scipy.signal.filtfilt(b, a, signal)


def rf_beam_current(
    beam: BeamBaseClass,
    profile: StaticProfile,
    omega_c: float,
    T_rev: float,
    use_lowpass_filter: bool = True,
    downsample: Optional[dict] = None,
    external_reference: bool = True,
    dT: float = 0,
) -> NumpyArray | tuple[NumpyArray, NumpyArray]:
    r"""Calculates the beam charge at the (RF) frequency slice by slice

    Function calculating the beam charge at the (RF) frequency, slice by
    slice. The charge distribution [C] of the beam is determined from the beam
    profile :math:`\lambda_i`, the particle charge :math:`q_p` and the real vs.
    macro-particle ratio :math:`N_{\mathsf{real}}/N_{\mathsf{macro}}`

    .. math::
        Q_i = \frac{N_{\mathsf{real}}}{N_{\mathsf{macro}}} q_p \lambda_i

    The total charge [C] in the beam is then

    .. math::
        Q_{\mathsf{tot}} = \sum_i{Q_i}

    The DC beam current [A] is the total number of charges per turn :math:`T_0`

    .. math:: I_{\mathsf{DC}} = \frac{Q_{\mathsf{tot}}}{T_0}

    The RF beam charge distribution [C] at a revolution frequency
    :math:`\omega_c` is the complex quantity

    .. math::
        \left( \begin{matrix} I_{rf,i} \\
        Q_{rf,i} \end{matrix} \right)
        = 2 Q_i \left( \begin{matrix} \cos(\omega_c t_i) \\
        \sin(\omega_c t_i)\end{matrix} \right) \, ,

    where :math:`t_i` are the time coordinates of the beam profile.
    After demodulation, a low-pass filter at 20 MHz is applied.

    For multi-bunch cases, make sure that the real beam intensity is the total
    number of charges in the ring.

    Parameters
    ----------
    profile : class
        A Profile type class
    omega_c : float
        Revolution frequency [1/s] at which the current should be calculated
    T_rev : float
        Revolution period [s] of the machine
    use_lowpass_filter : bool
        Apply low-pass filter; default is True
    downsample : dict
        Dictionary containing float value for 'Ts' sampling time and int value
        for 'points'. Will downsample the RF beam charge onto a coarse time
        grid with 'Ts' sampling time and 'points' points.
    external_reference : bool
        Option to include the changing external reference of the time-grid
    dT : float
        The shift in time due to shifting reference frames

    Returns
    -------
    complex array
        RF beam charge array [C] at 'frequency' omega_c, with the sampling time
        of the Profile object. To obtain current, divide by the sampling time
    (complex array)
        If time_coarse is specified, returns also the RF beam charge array [C]
        on the coarse time grid

    """

    # Convert from dimensionless to Coulomb/Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio
    charges = (
        beam.ratio # FIXME add to beam
        * beam.particle_type.charge
        * e
        * np.copy(profile.hist_y)
    )
    logger.debug(
        "Sum of particles: %d, total charge: %.4e C",
        np.sum(profile.hist_y),
        np.sum(charges),
    )
    logger.debug("DC current is %.4e A", np.sum(charges) / T_rev)

    # Mix with frequency of interest; remember factor 2 demodulation
    I_f = 2.0 * charges * np.cos(omega_c * profile.hist_x)
    Q_f = -2.0 * charges * np.sin(omega_c * profile.hist_x)

    # Pass through a low-pass filter
    if use_lowpass_filter is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f)) / T_rev)

    charges_fine = I_f + 1j * Q_f
    if external_reference:
        # slippage in phase due to a non-integer harmonic number
        dphi = dT * omega_c
        # Total phase correction
        phase = dphi
        charges_fine = charges_fine * np.exp(1j * phase)

    if downsample:
        try:
            T_s = float(downsample["Ts"])
            n_points = int(downsample["points"])
        except Exception:
            raise RuntimeError("Downsampling input erroneous in rf_beam_current")

        # Find which index in fine grid matches index in coarse grid
        ind_fine = np.round((profile.hist_x + dT - np.pi / omega_c) / T_s)
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]

        # Pick total current within one coarse grid
        charges_coarse = np.zeros(n_points, dtype=complex)
        charges_coarse[ind_fine[0]] = np.sum(charges_fine[np.arange(indices[0])])
        for i in range(1, len(indices)):
            charges_coarse[i + ind_fine[0]] = np.sum(
                charges_fine[np.arange(indices[i - 1], indices[i])]
            )

        return charges_fine, charges_coarse

    else:
        return charges_fine


def cartesian_to_polar(
    IQ_vector: NumpyArray,
) -> tuple[NumpyArray, NumpyArray]:
    """Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components

    Returns
    -------
    amplitude
        Amplitude of signal
    phase
        Phase of signal, in [rad]

    """
    return np.absolute(IQ_vector), np.angle(IQ_vector)


def polar_to_cartesian(
    amplitude: float | NumpyArray,
    phase: float | NumpyArray,
) -> NumpyArray | complex:
    """Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude
        Amplitude of signal
    phase
        Phase of signal, in [rad]

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components
    """

    return amplitude * (np.cos(phase) + 1j * np.sin(phase))
