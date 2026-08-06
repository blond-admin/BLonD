# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
RF beam-current demodulation for the cavity feedbacks.

``rf_beam_current`` turns a beam profile into the complex IQ beam-current
envelope at the carrier frequency, optionally re-binned onto the coarse
grid; ``low_pass_filter`` is the low-pass filter it uses.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.constants import elementary_charge

from blond.core.beam.base import BeamBaseClass
from blond.generals.cupy.no_cupy_import import copy_to_cpu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.physics.profiles import StaticProfile


def low_pass_filter(
    signal: NumpyArray, cutoff_frequency: float = 0.5
) -> NumpyArray:
    """
    Low-pass filter based on Butterworth 5th order digital filter.

    Parameters
    ----------
    signal : float array
        Signal to be filtered.
    cutoff_frequency : float
        Cutoff frequency [1] corresponding to a 3 dB gain drop, relative to the
        Nyquist frequency of 1; default is 0.5.

    Returns
    -------
    float array
        Low-pass filtered signal.

    Notes
    -----
    See `scipy`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """
    b, a = scipy.signal.butter(5, cutoff_frequency, "low", analog=False)

    return scipy.signal.filtfilt(b, a, signal)


def rf_beam_current(
    beam: BeamBaseClass,
    profile: StaticProfile,
    omega_c: float,
    *,
    sampling_time: float | None = None,
    n_points: int | None = None,
    use_lowpass_filter: bool = False,
    dT: float = 0.0,
    carrier_phase_offset: float = 0.0,
    forbid_charge_in_first_coarse_cell: bool = False,
) -> NumpyArray | tuple[NumpyArray, NumpyArray]:
    r"""
    Turn the beam profile into the complex IQ beam-current envelope.

    The beam oscillates at the RF carrier angular frequency
    :math:`\omega_c` (the carrier is the fast RF frequency every signal
    rides on). This routine *demodulates* the beam's charge onto that
    carrier -- it recovers the slowly-varying complex amplitude of the
    beam's RF-frequency current, the **IQ envelope** (in-phase
    :math:`+ i` quadrature) -- and optionally re-bins it onto the coarse
    grid. See the "Concepts and notation" section of
    :ref:`mucol_cavity_feedback_overview` for the IQ/demodulation picture.

    The charge distribution [C] of the beam is determined from the beam
    profile :math:`\lambda_i`, the particle charge :math:`q_p` and the real
    vs. macro-particle ratio :math:`N_{\mathsf{real}}/N_{\mathsf{macro}}`

    .. math::
        Q_i = \frac{N_{\mathsf{real}}}{N_{\mathsf{macro}}} q_p \lambda_i

    The RF beam charge distribution [C] at a carrier frequency
    :math:`\omega_c` is the complex quantity

    .. math::
        \left( \begin{matrix} I_{rf,i} \\
        Q_{rf,i} \end{matrix} \right)
        = 2 Q_i \left( \begin{matrix} \cos(\omega_c t_i) \\
        \sin(\omega_c t_i)\end{matrix} \right) \, ,

    where :math:`t_i` are the time coordinates of the beam profile. The
    factor 2 is the single-sideband demodulation convention: it recovers
    the fundamental amplitude from the projection onto
    :math:`\cos(\omega_c t)` and :math:`\sin(\omega_c t)`.

    The demodulated envelope is then rotated by
    ``exp(1j * (dT * omega_c + pi/2 + carrier_phase_offset))`` into the
    antenna-voltage IQ frame: the ``dT``-derived RF-clock slip, the fixed
    ``+pi/2`` axis alignment and an optional extra carrier phase (see the
    parameter descriptions below).

    For multi-bunch cases, make sure that the real beam intensity is the
    total number of charges in the ring.

    Parameters
    ----------
    beam : BeamBaseClass
        Beam to calculate the current on.
    profile : StaticProfile
        A Profile type class.
    omega_c : float
        Carrier angular frequency [rad/s] at which the current should be
        calculated.
    sampling_time : float, optional
        Coarse-grid sampling time ``T_s`` [s]. When given, the fine-grid
        RF beam charge is additionally downsampled onto a coarse time
        grid of ``n_points`` cells of ``sampling_time`` each, with the
        fine-to-coarse mapping centred on half a coarse cell
        (``sampling_time / 2``) and shifted by ``dT``.
    n_points : int, optional
        Number of coarse-grid points to downsample onto. Required when
        ``sampling_time`` is given.
    use_lowpass_filter : bool
        Apply a low-pass filter at 20 MHz after the demodulation; default
        is False.
    dT : float
        The shift in time [s] due to shifting reference frames. Rotates
        the demodulation carrier by ``dT * omega_c`` (the turn-to-turn
        slip of the RF clock from a non-integer harmonic / detuned
        reference) and shifts the fine-to-coarse binning by ``dT``.
    carrier_phase_offset : float
        Additional demodulation-carrier phase [rad], on top of the
        ``dT``-derived rotation. Used by the timing class to anchor the
        carrier to the accumulated actual RF phase
        (``- int delta_omega_rf dt`` since simulation start) when the
        parent station runs with an RF-frequency offset. Without one it
        is exactly ``0.0`` (bit-identical demodulation). A pure phase: it
        must not (and does not) move the fine-to-coarse binning, which
        follows the physical sample times.
    forbid_charge_in_first_coarse_cell : bool
        If True, raise a ``ValueError`` when the downsampling assigns
        beam charge to the first coarse-grid cell. Callers that take the
        fine-grid initial antenna voltage from that cell (e.g.
        ``IQCavityFeedbackTimingClass``) must keep it charge-free, since
        a populated first cell would double-count its kick.

    Returns
    -------
    charges_fine : complex array
        RF beam charge array [C] at the carrier ``omega_c``, on the
        sampling-time grid of the Profile object. Divide by the sampling
        time to obtain the current.
    charges_coarse : complex array
        RF beam charge array [C] downsampled onto the ``n_points`` coarse
        grid. Only returned when ``sampling_time`` is given.

    Raises
    ------
    TypeError
        If ``sampling_time`` is given without ``n_points``.
    ValueError
        If ``forbid_charge_in_first_coarse_cell`` is True and the
        downsampling assigns beam charge to the first coarse-grid cell.
    """
    # The cavity-feedback signal processing runs on the host (the cavity
    # response solvers downstream use scipy, which is host-only). Bring the
    # profile arrays to the host once here so this function works on any
    # backend, including a GPU/cupy backend.
    hist_x = copy_to_cpu(profile.hist_x)
    hist_y = copy_to_cpu(profile.hist_y)

    # Warn before anything else if the profile does not capture the whole
    # beam (particle loss or particles outside the profile window): the
    # missing charge is invisible to the feedback and will not be treated.
    if profile.hist_y_to_density_factor is not None:
        captured_fraction = float(
            np.sum(hist_y) * profile.hist_y_to_density_factor
        )
        if not np.isclose(captured_fraction, 1.0, rtol=0, atol=1e-6):
            warnings.warn(
                f"Only {captured_fraction:.6f} of the beam's macroparticles "
                "are inside the profile window (particle loss or particles "
                "outside the window). Their charge is invisible to the "
                "feedback and will not be treated correctly.",
                stacklevel=2,
            )

    # Convert from dimensionless to Coulomb/Ampères
    # Take into account macro-particle charge with real-to-macro-particle ratio.
    # The direction-signed charge makes the gap current of a counter-rotating
    # beam come out with the correct sign (opposite charge x opposite direction
    # = same current as the co-rotating beam); for co-rotating beams it is
    # simply the plain particle charge.
    charges = (
        -elementary_charge
        * beam.signed_charge_with_direction()
        * beam.intensity
        * hist_y
        * profile.hist_y_to_density_factor
    )

    logger.debug(
        "Sum of particles: %d, total charge: %.4e C",
        np.sum(hist_y),
        np.sum(charges),
    )

    # Demodulate the (real) beam charge to the complex baseband envelope at the
    # carrier omega_c: I/Q are the in-phase (cos) and quadrature (-sin)
    # projections, and the factor 2 recovers the fundamental amplitude from the
    # single-sideband mixing (charges_fine = 2 * charges * exp(-i omega_c t)).
    I_f = 2.0 * charges * np.cos(omega_c * hist_x)
    Q_f = -2.0 * charges * np.sin(omega_c * hist_x)

    # Pass through a low-pass filter
    if use_lowpass_filter:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)

    # Rotate the beam current into the same I/Q frame as the antenna
    # voltage / generator current with three contributions:
    #   * dphi = dT * omega_c: the turn-to-turn slip of the RF clock from a
    #     non-integer harmonic / detuned reference (dT == 0 leaves it out).
    #   * +pi/2: aligns the beam-current demodulation axis (in-phase = cos)
    #     with the antenna-voltage lab-frame convention
    #     V_lab = -Im[V_ant exp(i omega_c t)] (in-phase = -sin), which differ
    #     by 90 deg. Verified against the independent resonator / multi-turn
    #     wake models: removing the pi/2 rotates the beam-induced voltage by
    #     90 deg (breaks the energy-gain and phase tests) and flipping its
    #     sign flips the beam-loading sign.
    #   * carrier_phase_offset: the caller's extra demodulation-carrier
    #     phase (exactly 0.0 when unused: bit-identical demodulation).
    charges_fine = I_f + 1j * Q_f
    dphi = dT * omega_c
    charges_fine = charges_fine * np.exp(
        1j * (dphi + np.pi / 2 + carrier_phase_offset)
    )

    if sampling_time is None:
        return charges_fine

    if (
        n_points is None
    ):  # TODO: this should be checked at the beginning of the function
        raise TypeError(
            "rf_beam_current: n_points is required when sampling_time "
            "is given."
        )  # TODO: attribute Error

    # Downsample onto the coarse grid. The mapping is centred on half a
    # coarse cell (the coarse bin centre) and shifted by dT.
    coarse_center_offset = sampling_time / 2
    ind_fine = np.round((hist_x + dT - coarse_center_offset) / sampling_time)
    ind_fine = np.array(ind_fine, dtype=int)
    indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]
    if np.any(ind_fine < 0):
        warnings.warn(
            "part of the beam is located before turn time 0, "
            "this will cause problems, please shift the beam",
            stacklevel=2,
        )

    charges_coarse = np.zeros(n_points, dtype=complex)
    if len(indices) == 0:
        # single bucket in ind_fine --> all ind_fine identical
        charges_coarse[ind_fine[0]] = np.sum(charges_fine)
    else:
        # Pick total current within one coarse grid
        charges_coarse[ind_fine[0]] = np.sum(charges_fine[: indices[0]])
        for i in range(1, len(indices)):
            # The write index is kept in range by the % n_points wrap
            # (periodic coarse grid), so no bounds guard is needed.
            charges_coarse[(i + ind_fine[0]) % n_points] = np.sum(
                charges_fine[indices[i - 1] : indices[i]]
            )
        # Remainder after the last cell boundary. Dropping it would lose
        # all charge of the profile past that boundary (up to ~half the
        # bunch), which corrupts the coarse-grid beam loading and
        # everything propagated from it across turns.
        charges_coarse[(len(indices) + ind_fine[0]) % n_points] = np.sum(
            charges_fine[indices[-1] :]
        )

    if forbid_charge_in_first_coarse_cell:
        # The fine-grid initial antenna voltage is taken from the first
        # coarse cell (see circuit_track), so it must stay charge-free,
        # otherwise its beam kick is double-counted by the fine grid.
        # Relative threshold: far Gaussian tails are non-zero in float
        # arithmetic (~1e-100) without being physically populated.
        total_charge = np.sum(np.abs(charges_fine))
        if np.abs(charges_coarse[0]) > 1e-9 * total_charge:
            raise ValueError(
                "Beam charge was downsampled into the first coarse-grid "
                "cell. The fine-grid initial antenna voltage is taken "
                "from this cell, so its beam kick would be "
                "double-counted by the fine grid. Shift the profile "
                "window (cut_left) or the bunch so that no charge lies "
                "in the first coarse cell."
            )

    return charges_fine, charges_coarse
