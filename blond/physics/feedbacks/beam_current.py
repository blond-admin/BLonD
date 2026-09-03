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
from blond.generals.cupy_.no_cupy_import import copy_to_cpu

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


def _check_coarse_index_bounds(
    ind_fine: NumpyArray,
    charges_fine: NumpyArray,
    n_points: int,
) -> None:
    """
    Reject fine-to-coarse write indices that fall outside the grid.

    The coarse grid is not periodic, so an index out of either end is
    silently wrong rather than merely inaccurate: past the last cell it
    overwrites an earlier cell (or raises a bare ``IndexError``), and
    below the first cell NumPy's negative indexing deposits the charge
    into the *last* cells instead -- about one forward-segment span late,
    and out of reach of the ``forbid_charge_in_first_coarse_cell`` guard,
    which only inspects cell 0. Both ends are therefore bounded here.

    The lower bound uses the same relative threshold idiom as the
    first-coarse-cell guard: far Gaussian tails are non-zero in float
    arithmetic (~1e-100) without being physically populated, so a
    charge-free tail sticking out below the grid start only warns, while
    bins that really carry charge raise.

    Parameters
    ----------
    ind_fine : int array
        Coarse-grid index each fine bin is downsampled into.
    charges_fine : complex array
        Demodulated fine-grid charge [C], used to weigh how much charge
        an out-of-range index actually carries.
    n_points : int
        Number of coarse-grid cells.

    Raises
    ------
    ValueError
        If the profile maps past the last coarse cell, or if bins
        carrying non-negligible charge map before the first one.

    Warns
    -----
    UserWarning
        If bins map before the first coarse cell but carry negligible
        charge.
    """
    negative_bins = ind_fine < 0
    if np.any(negative_bins):
        total_charge = np.sum(np.abs(charges_fine))
        underflow_charge = np.sum(np.abs(charges_fine[negative_bins]))
        if underflow_charge > 1e-9 * total_charge:
            raise ValueError(
                f"Beam charge maps onto coarse-grid index "
                f"{int(np.min(ind_fine))}, before the start of the coarse "
                f"grid: {underflow_charge / total_charge:.3e} of the "
                "demodulated charge lies before the grid start, where it "
                "would wrap onto the last coarse cells instead. Shift the "
                "profile window (cut_left), or the beam, so that it lies "
                "inside the grid."
            )
        warnings.warn(
            "part of the beam is located before turn time 0, "
            "this will cause problems, please shift the beam",
            stacklevel=3,
        )
    if len(ind_fine) and ind_fine[-1] >= n_points:
        raise ValueError(
            f"The profile maps onto coarse-grid index {int(ind_fine[-1])}, "
            f"past the last cell ({n_points - 1}): the profile window lies "
            "(partly) after the end of the coarse grid. Shift the profile "
            "window (cut_left), or the beam, so that it lies inside the "
            "grid."
        )


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
        downsampling assigns beam charge to the first coarse-grid cell;
        if the profile window is longer than the coarse grid it is
        downsampled onto, or maps past the last coarse cell, or carries
        charge before the start of it; or if the profile binning
        (``hist_step``) is coarser than ``sampling_time``.
    """
    # The cavity-feedback signal processing runs on the host (the cavity
    # response solvers downstream use scipy, which is host-only). Bring the
    # profile arrays to the host once here so this function works on any
    # backend, including a GPU/cupy backend.
    hist_x = copy_to_cpu(profile.hist_x)
    hist_y = copy_to_cpu(profile.hist_y)

    # Whether the profile captured the whole beam is NOT checked here.
    # That is a property of the profile's own window, not of this
    # consumer, and it is owned by
    # ``ProfileBaseClass._warn_if_beam_not_captured``, which warns once
    # at fill time -- before this function is ever reached, and for every
    # consumer of a profile rather than for the feedback alone. Do not
    # re-add a copy here: it would fire per passage against that
    # once-per-profile latch.

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
    #
    # WHICH FRAME. The +pi/2 is only meaningful together with the frame
    # the coarse-grid caller establishes, so state it here rather than
    # leaving it to be re-derived:
    #   * dT is the tail left by the PRECEDING coarse segment, normally
    #     t_rf / 2, so ``dphi = omega_c * dT`` is pi (mod 2 pi);
    #   * carrier_phase_offset is ``-(phi_rf + carrier_slip_gap)``, i.e.
    #     minus the total the station's kick and the readout's
    #     phase_correction add back on top of ``angle(V_ant)``;
    #   * the readout itself is POLAR and referenced to the station
    #     setpoint (``cartesian_to_polar`` -> ``phase_correction =
    #     alpha_sum - mean(angle(station_voltage_coarse_grid)) +
    #     carrier_slip_gap``), applied as ``sin(omega_rf t + phi_rf +
    #     phase_offsets)``.
    # Do NOT re-derive the sign from a lab-frame identity such as
    # ``V_lab = -Im[V_ant e^{i omega_c t}]``: that form is used nowhere
    # in this chain, and reasoning from it yields -pi/2 here, which
    # inverts the beam loading. The check that matters is the end-to-end
    # one -- a bunch must LOSE energy to its own wake.
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

    # General invariant, owned by the Profile (see
    # ProfileBaseClass.check_fits_in_span), not by the feedback: a window
    # longer than the span the coarse grid covers cannot be re-binned onto
    # it. The grid here covers only the FORWARD segment (the drift to the
    # next RF station, 1 / n_sections of a turn) -- it is NOT a full turn
    # and NOT periodic, so a wrapped group would overwrite an earlier cell
    # instead of accumulating into it. That forward span IS the interval
    # between two consecutive passages of this station, so it is the same
    # quantity the per-passage wake solvers pass to the same guard.
    # Independent of the profile's own incomplete-capture warning: the
    # fold destroys charge even when the window captures the whole beam.
    profile.check_fits_in_span(
        n_points * sampling_time,
        span_description=(
            f"the coarse grid it is downsampled onto (n_points="
            f"{n_points} x sampling_time={sampling_time} s)"
        ),
    )

    # The downsampling loop below derives each coarse cell from
    # CONSECUTIVE-index steps in ind_fine: its running group counter is the
    # coarse index, which only holds while ind_fine advances by at most 1
    # per fine bin. A fine grid coarser than the coarse grid lets the
    # rounded index jump by 2 or more, the counter falls behind, and the
    # charge is placed at the WRONG TIME while the total stays conserved --
    # silent corruption. Reachable from a legitimate-looking setup, since
    # sub-stepping (n_rf_periods_per_coarse_grid < 1) shrinks
    # sampling_time.
    hist_step = float(profile.hist_step)
    if hist_step > sampling_time:
        raise ValueError(
            f"The profile binning (hist_step={hist_step} s) is coarser "
            f"than the coarse grid it is downsampled onto "
            f"(sampling_time={sampling_time} s). The downsampling assumes "
            "at least one fine bin per coarse cell; a coarser profile "
            "makes the coarse index jump, which places charge at the "
            "wrong time while conserving the total. Bin the profile more "
            "finely, or use a larger n_rf_periods_per_coarse_grid."
        )

    # Downsample onto the coarse grid. The mapping is centred on half a
    # coarse cell (the coarse bin centre) and shifted by dT.
    coarse_center_offset = sampling_time / 2
    ind_fine = np.round((hist_x + dT - coarse_center_offset) / sampling_time)
    ind_fine = np.array(ind_fine, dtype=int)
    indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]
    _check_coarse_index_bounds(ind_fine, charges_fine, n_points)

    charges_coarse = np.zeros(n_points, dtype=complex)
    if len(indices) == 0:
        # single bucket in ind_fine --> all ind_fine identical
        charges_coarse[ind_fine[0]] = np.sum(charges_fine)
    else:
        # Pick total current within one coarse grid
        charges_coarse[ind_fine[0]] = np.sum(charges_fine[: indices[0]])
        for i in range(1, len(indices)):
            # Every write index is inside the grid: the guards above
            # bound it on both sides -- check_fits_in_span and the
            # ind_fine[-1] bound reject anything past the last cell, the
            # negative-index guard anything before the first. Both
            # matter, because the coarse grid is not periodic: an index
            # out of either end wraps or overwrites instead of
            # accumulating.
            charges_coarse[i + ind_fine[0]] = np.sum(
                charges_fine[indices[i - 1] : indices[i]]
            )
        # Remainder after the last cell boundary. Dropping it would lose
        # all charge of the profile past that boundary (up to ~half the
        # bunch), which corrupts the coarse-grid beam loading and
        # everything propagated from it across turns.
        charges_coarse[len(indices) + ind_fine[0]] = np.sum(
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
