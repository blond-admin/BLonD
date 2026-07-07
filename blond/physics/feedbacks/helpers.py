# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Convenience function used in cavity and beam feedback modules."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy
from scipy.constants import elementary_charge
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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


# TODO: split this function into helpers and remove the PLR0912 noqa
def rf_beam_current(  # noqa: PLR0912
    beam: BeamBaseClass,
    profile: StaticProfile,
    omega_c: float,
    T_rev: float,
    use_lowpass_filter: bool = True,
    downsample: dict | None = None,
    external_reference: bool = True,
    dT: float = 0,
    forbid_charge_in_first_coarse_cell: bool = False,
    dT_index_sign: float = 1.0,
    coarse_center_offset: float | None = None,
) -> NumpyArray | tuple[NumpyArray, NumpyArray]:
    r"""
    Calculate the beam charge at the carrier frequency slice by slice.

    Function calculating the beam charge at the carrier frequency, slice by
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

    The RF beam charge distribution [C] at a carrier frequency
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
    beam : class
        Beam to calculate the current on.
    profile : class
        A Profile type class.
    omega_c : float
        Carrier angular frequency [rad/s] at which the current should be calculated.
    T_rev : float
        Revolution period [s] of the machine.
    use_lowpass_filter : bool
        Apply low-pass filter; default is True.
    downsample : dict
        Dictionary containing float value for 'Ts' sampling time and int value
        for 'points'. Will downsample the RF beam charge onto a coarse time
        grid with 'Ts' sampling time and 'points' points.
    external_reference : bool
        Option to include the changing external reference of the time-grid.
    dT : float
        The shift in time due to shifting reference frames.
    forbid_charge_in_first_coarse_cell : bool
        If True, raise a ``ValueError`` when the downsampling assigns beam
        charge to the first coarse-grid cell. Callers that take the
        fine-grid initial antenna voltage from that cell (e.g.
        ``IQCavityFeedbackTimingClass``) must keep it charge-free, since a
        populated first cell would double-count its kick.
    dT_index_sign : float
        Sign of ``dT`` in the fine-to-coarse index mapping ``ind_fine``.
        ``+1`` (default) is the mucol/reworked convention; ``-1`` matches the
        blond2 reference (``legacy/blond2/llrf/signal_processing.py``) and is
        used by the LHC comparison path. Only affects the downsampling
        binning when ``dT != 0``; the phase correction always uses ``+dT``.
    coarse_center_offset : float | None
        Bin-centre offset [s] of the fine-to-coarse index mapping. Default
        ``None`` uses ``pi / omega_c`` (half a period of the actual carrier,
        the blond2/LHC convention). Callers whose coarse cell is not one
        carrier period (e.g. the sub-stepped timing class) must pass their
        own half-cell offset, ``T_s / 2``.

    Returns
    -------
    complex array
        RF beam charge array [C] at 'frequency' omega_c, with the sampling time
        of the Profile object. To obtain current, divide by the sampling time (complex array)
        If time_coarse is specified, returns also the RF beam charge array [C]
        on the coarse time grid.
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
    # Take into account macro-particle charge with real-to-macro-particle ratio
    charges = (
        -elementary_charge
        * beam.particle_type.charge
        * beam.intensity
        * hist_y
        * profile.hist_y_to_density_factor
    )

    logger.debug(
        "Sum of particles: %d, total charge: %.4e C",
        np.sum(hist_y),
        np.sum(charges),
    )
    logger.debug("DC current is %.4e A", np.sum(charges) / T_rev)

    # Demodulate the (real) beam charge to the complex baseband envelope at the
    # carrier omega_c: I/Q are the in-phase (cos) and quadrature (-sin)
    # projections, and the factor 2 recovers the fundamental amplitude from the
    # single-sideband mixing (charges_fine = 2 * charges * exp(-i omega_c t)).
    I_f = 2.0 * charges * np.cos(omega_c * hist_x)
    Q_f = -2.0 * charges * np.sin(omega_c * hist_x)

    # Pass through a low-pass filter
    if use_lowpass_filter is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f)) / T_rev)

    charges_fine = I_f + 1j * Q_f
    if external_reference:
        # Rotate the beam current into the same I/Q frame as the antenna
        # voltage / generator current with two contributions:
        #   * dphi = dT * omega_c: the turn-to-turn slip of the RF clock from a
        #     non-integer harmonic / detuned reference (dT == 0 leaves it out).
        #   * +pi/2: aligns the beam-current demodulation axis (in-phase = cos)
        #     with the antenna-voltage lab-frame convention
        #     V_lab = -Im[V_ant exp(i omega_c t)] (in-phase = -sin), which differ
        #     by 90 deg. Verified against the independent resonator / multi-turn
        #     wake models: removing the pi/2 rotates the beam-induced voltage by
        #     90 deg (breaks the energy-gain and phase tests) and flipping its
        #     sign flips the beam-loading sign.
        dphi = dT * omega_c
        charges_fine = charges_fine * np.exp(1j * (dphi + np.pi / 2))

    if downsample:
        try:
            T_s = float(downsample["Ts"])
            n_points = int(downsample["points"])
        except Exception as e:
            raise RuntimeError(
                "Downsampling input erroneous in rf_beam_current."
            ) from e

        # Find which index in fine grid matches index in coarse grid.
        # dT enters with sign `dT_index_sign`: +1 (mucol/reworked convention,
        # default) or -1 (blond2 reference, used by the LHC comparison path).
        # The centering offset places the mapping on the coarse bin centre.
        # The default, pi / omega_c (half a period of the *actual* carrier,
        # including any RF-frequency offset), is the blond2 convention the
        # LHC comparisons rely on. Callers with a coarse cell that is not
        # one carrier period (e.g. the sub-stepped timing class, where the
        # default would equal a *whole* cell and shift all charge one cell
        # early -- a sign-flipped induced voltage) must pass their own
        # half-cell offset via ``coarse_center_offset``.
        if coarse_center_offset is None:
            coarse_center_offset = np.pi / omega_c
        ind_fine = np.round(
            (hist_x + dT_index_sign * dT - coarse_center_offset) / T_s
        )
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]
        if np.any(ind_fine < 0):
            warnings.warn(
                "part of the beam is located before turn time 0, "
                "this will cause problems, please shift the beam",
                stacklevel=2,
            )

        charges_coarse = np.zeros(n_points, dtype=complex)

        if (
            len(indices) == 0
        ):  # only a single bucket in ind_fine  --> all ind_fine identical
            charges_coarse[ind_fine[0]] = np.sum(charges_fine)
        else:
            # Pick total current within one coarse grid
            charges_coarse[ind_fine[0]] = np.sum(charges_fine[: indices[0]])

            for i in range(
                1, len(indices)
            ):  # TODO: not good for sparse profiles !!!!
                # The write index is kept in range by the % n_points wrap
                # below (periodic coarse grid), so no bounds guard is needed.
                charges_coarse[(i + ind_fine[0]) % n_points] = np.sum(
                    charges_fine[
                        indices[i - 1] : indices[i]
                    ]  # TODO: +1 for edges?
                )
            # Remainder after the last cell boundary. Dropping it loses all
            # charge of the profile past that boundary (up to ~half the
            # bunch), which corrupts the coarse-grid beam loading and
            # everything propagated from it across turns.
            charges_coarse[(len(indices) + ind_fine[0]) % n_points] = np.sum(
                charges_fine[indices[-1] :]
            )
        if forbid_charge_in_first_coarse_cell:
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

    else:
        return charges_fine


def rf_beam_current_partial(
    beam: BeamBaseClass,
    profile: StaticProfile,
    omega_c: float,
    T_rev: float,
    sampling_time: float,
    n_points: int,
    dT: float,
    use_lowpass_filter: bool = False,
) -> tuple[NumpyArray, NumpyArray]:
    r"""
    RF beam current on the sub-stepped coarse grid (mucol timing class only).

    Dedicated, fully-separated counterpart of :func:`rf_beam_current` for the
    :class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
    forward pass. It hard-codes the conventions that path always uses -- the
    external reference frame is on, the fine-to-coarse index sign is ``+1``, the
    mapping is centred on half a *coarse cell* (``sampling_time / 2``, not the
    half-carrier-period ``pi / omega_c`` of the LHC path), and the first coarse
    cell must stay charge-free -- so those no longer leak into the LHC-shared
    :func:`rf_beam_current` as optional flags. It is kept a separate function
    (not a thin wrapper) so the two paths can evolve independently:
    :func:`rf_beam_current` must stay bit-identical for the LHC comparison
    tests, which drive it through the (experimental) LHC feedback with
    ``dT_index_sign = -1``.

    See :func:`rf_beam_current` for the physics of the demodulation, the
    ``+pi/2`` rotation and the downsampling.

    Parameters
    ----------
    beam : BeamBaseClass
        Beam to calculate the current on.
    profile : StaticProfile
        A Profile type class.
    omega_c : float
        Carrier angular frequency [rad/s] of the forward-tracking RF.
    T_rev : float
        Forward-tracking period [s] (used only for the DC-current logging).
    sampling_time : float
        Coarse-grid sampling time ``T_s`` [s] of the forward segment.
    n_points : int
        Number of coarse-grid points to downsample onto.
    dT : float
        Demodulation time shift [s] that rotates the beam current into the
        frame of the coarse-grid envelope recursion.
    use_lowpass_filter : bool
        Apply the 20 MHz low-pass filter; default False.

    Returns
    -------
    charges_fine : complex array
        RF beam charge [C] on the fine (profile) grid.
    charges_coarse : complex array
        RF beam charge [C] downsampled onto the ``n_points`` coarse grid.
    """
    # The cavity-feedback signal processing runs on the host; bring the profile
    # arrays to the host once so this works on any backend (see rf_beam_current).
    hist_x = copy_to_cpu(profile.hist_x)
    hist_y = copy_to_cpu(profile.hist_y)

    # Warn if the profile does not capture the whole beam: the missing charge is
    # invisible to the feedback and will not be treated.
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

    # Convert from dimensionless to Coulomb, including the real-to-macro ratio.
    charges = (
        -elementary_charge
        * beam.particle_type.charge
        * beam.intensity
        * hist_y
        * profile.hist_y_to_density_factor
    )

    logger.debug(
        "Sum of particles: %d, total charge: %.4e C",
        np.sum(hist_y),
        np.sum(charges),
    )
    logger.debug("DC current is %.4e A", np.sum(charges) / T_rev)

    # Demodulate the (real) beam charge to the complex baseband envelope at the
    # carrier omega_c; the factor 2 recovers the fundamental amplitude.
    I_f = 2.0 * charges * np.cos(omega_c * hist_x)
    Q_f = -2.0 * charges * np.sin(omega_c * hist_x)

    if use_lowpass_filter is True:
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f)) / T_rev)

    # External reference frame is always on for the timing class: rotate by the
    # RF-clock slip dphi = dT * omega_c and the +pi/2 that aligns the beam-
    # current I/Q axis with the antenna-voltage lab-frame convention.
    charges_fine = I_f + 1j * Q_f
    dphi = dT * omega_c
    charges_fine = charges_fine * np.exp(1j * (dphi + np.pi / 2))

    # Downsample onto the coarse grid. The mapping is centred on half a coarse
    # cell and dT enters with +1 sign (the mucol convention).
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
        charges_coarse[ind_fine[0]] = np.sum(charges_fine[: indices[0]])
        for i in range(1, len(indices)):
            # The write index is kept in range by the % n_points wrap
            # (periodic coarse grid), so no bounds guard is needed.
            charges_coarse[(i + ind_fine[0]) % n_points] = np.sum(
                charges_fine[indices[i - 1] : indices[i]]
            )
        # Remainder after the last cell boundary; dropping it would lose all
        # charge past that boundary (up to ~half the bunch).
        charges_coarse[(len(indices) + ind_fine[0]) % n_points] = np.sum(
            charges_fine[indices[-1] :]
        )

    # The fine-grid initial antenna voltage is taken from the first coarse cell
    # (see circuit_track), so it must stay charge-free, otherwise its beam kick
    # is double-counted by the fine grid. Relative threshold: far Gaussian tails
    # are non-zero in float arithmetic without being physically populated.
    total_charge = np.sum(np.abs(charges_fine))
    if np.abs(charges_coarse[0]) > 1e-9 * total_charge:
        raise ValueError(
            "Beam charge was downsampled into the first coarse-grid cell. "
            "The fine-grid initial antenna voltage is taken from this cell, "
            "so its beam kick would be double-counted by the fine grid. Shift "
            "the profile window (cut_left) or the bunch so that no charge lies "
            "in the first coarse cell."
        )

    return charges_fine, charges_coarse


def cartesian_to_polar(
    IQ_vector: NumpyArray,
) -> tuple[NumpyArray, NumpyArray]:
    """
    Convert data from Cartesian (I,Q) to polar coordinates.

    Parameters
    ----------
    IQ_vector : complex array
        Signal with in-phase and quadrature (I,Q) components.

    Returns
    -------
    amplitude
        Amplitude of signal.
    phase
        Phase of signal, in [rad].
    """
    return np.absolute(IQ_vector), np.angle(IQ_vector)


def polar_to_cartesian(
    amplitude: float | NumpyArray,
    phase: float | NumpyArray,
) -> NumpyArray | complex:
    """
    Convert data from polar to cartesian (I,Q) coordinates.

    Parameters
    ----------
    amplitude
        Amplitude of signal.
    phase
        Phase of signal, in [rad].

    Returns
    -------
    complex array
        Signal with in-phase and quadrature (I,Q) components.
    """
    return amplitude * (np.cos(phase) + 1j * np.sin(phase))


def cavity_response_sparse_matrix(
    I_beam: NumpyArray,
    I_gen: NumpyArray,
    V_ant_init: float,
    I_gen_init: float,
    samples_per_rf: float,
    R_over_Q: float,
    Q_L: float,
    relative_detuning: float,
):
    """
    Solver for the ACS cavity response model as a sparse matrix problem.

    Solving the ACS cavity response model as a sparse matrix problem
    for a given set of initial conditions, resonator parameters and
    generator and RF beam currents. The input arrays are extended by
    one entry (I_gen_init and V_ant_init respectively) to take
    respect the fact that the first matrix entry is not part of the solution
    domain.

    Parameters
    ----------
    I_beam : complex array
        RF beam current.
    I_gen : complex array
        Generator current.
    V_ant_init : complex float
        Initial condition for the antenna voltage.
    I_gen_init : complex float
        Initial condition for the generator current.
    samples_per_rf : float
        Number of samples per RF period == samping time * actual rf frequency.
    R_over_Q : float
        The R over Q of the cavity.
    Q_L : float
        The loaded quality factor of the cavity.
    relative_detuning : float
        The detuning of the cavity in frequency divided by the rf frequency.

    Returns
    -------
    complex array
        The antenna voltage evaluated for the same period as I_beam and I_gen of length len(I_gen).
    """
    assert len(I_beam) == len(I_gen), (
        "length of beam and generator currents need to match"
    )

    # Extend arrays to take initial values into account
    internal_I_gen = np.concatenate(([I_gen_init], I_gen))
    internal_I_beam = np.concatenate(([0j], I_beam))

    n_samples = len(internal_I_gen)

    # Compute matrix elements
    A = 0.5 * R_over_Q * samples_per_rf
    B = (
        1
        - 0.5 * samples_per_rf / Q_L
        + 1j * relative_detuning * samples_per_rf
    )

    # Initialize the two sparse matrices needed to find antenna voltage
    B_matrix = diags(
        [-B, 1],
        [-1, 0],
        (n_samples, n_samples),
        dtype=complex,
        format="csc",
    )
    I_matrix = diags([A], [-1], (n_samples, n_samples), dtype=complex)

    # Find vector on the "current" side of the equation
    b = I_matrix.dot(2 * internal_I_gen - internal_I_beam)
    b[0] = V_ant_init

    # Solve the sparse linear system of equations and return
    return spsolve(B_matrix, b)[1:]
    # first value is intial condition


def cavity_response_sparse_matrix_second_order(
    I_beam: NumpyArray,
    I_gen: NumpyArray,
    V_ant_init: float,
    I_gen_init: float,
    samples_per_rf: float,
    R_over_Q: float,
    Q_L: float,
    relative_detuning: float,
):
    r"""
    Second-order (trapezoidal / Crank-Nicolson) ACS cavity response solver.

    Drop-in alternative to :func:`cavity_response_sparse_matrix`. It solves
    the same cavity-envelope ODE

    .. math::
        \frac{\mathrm{d}V}{\mathrm{d}t}
        = \Big(-\frac{\omega}{2 Q_L} + i\,\Delta\omega\Big) V
          + \frac{R/Q\,\omega}{2}\,(2 I_{\mathrm{gen}} - I_{\mathrm{beam}}),

    but integrates it with the trapezoidal rule (averaging the homogeneous
    term *and* the current drive over each step) instead of the forward-Euler
    (left-endpoint) step used by :func:`cavity_response_sparse_matrix`. The
    truncation error is therefore :math:`O(\Delta t^2)` rather than
    :math:`O(\Delta t)`, which matters most at coarse binning (large
    ``samples_per_rf``).

    With ``lam = -0.5 * samples_per_rf / Q_L + 1j * relative_detuning *
    samples_per_rf`` (so ``B = 1 + lam`` of the first-order solver) and the
    per-step drive ``s[i] = A * (2 I_gen[i] - I_beam[i])``, the recursion is

    .. math::
        (1 - \mathrm{lam}/2)\,V_i
        = (1 + \mathrm{lam}/2)\,V_{i-1} + \tfrac12 (s_{i-1} + s_i).

    Parameters
    ----------
    I_beam : complex array
        RF beam current.
    I_gen : complex array
        Generator current.
    V_ant_init : complex float
        Initial condition for the antenna voltage.
    I_gen_init : complex float
        Initial condition for the generator current.
    samples_per_rf : float
        Number of samples per RF period == sampling time * actual rf frequency.
    R_over_Q : float
        The R over Q of the cavity.
    Q_L : float
        The loaded quality factor of the cavity.
    relative_detuning : float
        The detuning of the cavity in frequency divided by the rf frequency.

    Returns
    -------
    complex array
        The antenna voltage evaluated for the same period as I_beam and I_gen
        of length len(I_gen).
    """
    assert len(I_beam) == len(I_gen), (
        "length of beam and generator currents need to match"
    )

    # Extend arrays to take initial values into account
    internal_I_gen = np.concatenate(([I_gen_init], I_gen))
    internal_I_beam = np.concatenate(([0j], I_beam))

    n_samples = len(internal_I_gen)

    A = 0.5 * R_over_Q * samples_per_rf
    # lam == B - 1 of the first-order solver, i.e. (step size) * (decay/detuning)
    lam = -0.5 * samples_per_rf / Q_L + 1j * relative_detuning * samples_per_rf

    # Per-step current drive, identical to the first-order solver's source term
    s = A * (2 * internal_I_gen - internal_I_beam)

    # Bidiagonal trapezoidal system. Row 0 pins the initial condition
    # (diagonal 1), all later rows use the Crank-Nicolson coefficients.
    diagonal = np.full(n_samples, 1 - 0.5 * lam, dtype=complex)
    diagonal[0] = 1.0
    sub_diagonal = np.full(n_samples - 1, -(1 + 0.5 * lam), dtype=complex)
    cn_matrix = diags(
        [sub_diagonal, diagonal],
        [-1, 0],
        (n_samples, n_samples),
        dtype=complex,
        format="csc",
    )

    b = np.empty(n_samples, dtype=complex)
    b[0] = V_ant_init
    b[1:] = 0.5 * (s[:-1] + s[1:])

    return spsolve(cn_matrix, b)[1:]
    # first value is the initial condition


def pretrack_fill_voltage(
    r_over_q: float,
    q_l: float,
    omega: float,
    delta_omega: float,
    generator_current: complex,
    n_pretrack: int,
    t_rev: float,
    injection_voltage: float | None = None,
) -> complex:
    r"""
    Seed antenna voltage from a feedforward (constant-current) cavity fill.

    The no-beam cavity envelope driven by a constant generator current
    :math:`I_\mathsf{gen}` obeys

    .. math::
        \frac{\mathrm{d}V}{\mathrm{d}t} = \lambda V
            + \frac{R}{Q}\,\omega\,I_\mathsf{gen},
        \qquad \lambda = -\frac{\omega}{2 Q_L} + i\,\Delta\omega,

    which fills from a cold cavity (:math:`V(0) = 0`) as

    .. math::
        V(t) = V_\mathsf{ss}\,(1 - e^{\lambda t}),
        \qquad V_\mathsf{ss} = -\frac{(R/Q)\,\omega\,I_\mathsf{gen}}{\lambda}.

    On resonance (:math:`\Delta\omega = 0`) this reduces to
    :math:`V_\mathsf{ss} = 2 (R/Q) Q_L I_\mathsf{gen}`.

    Without ``injection_voltage`` the seed is :math:`V(n_\mathsf{pretrack} T_0)`
    (the fill after ``n_pretrack`` turns, which approaches :math:`V_\mathsf{ss}`).
    With ``injection_voltage`` the seed is :math:`V(t^\star)` at the first
    :math:`t^\star \in [0, n_\mathsf{pretrack} T_0]` where :math:`|V(t)|` reaches
    ``injection_voltage`` -- i.e. the beam is injected part-way through the fill.

    Parameters
    ----------
    r_over_q
        Geometric shunt impedance of the cavity [Ohm].
    q_l
        Loaded quality factor of the cavity.
    omega
        RF angular frequency [rad/s].
    delta_omega
        Cavity resonance detuning [rad/s].
    generator_current
        Constant (feedforward) generator current [A].
    n_pretrack
        Cavity fill budget in turns.
    t_rev
        Revolution period [s].
    injection_voltage
        If given, seed from the fill transient when ``|V_ant|`` first reaches
        this magnitude [V]; otherwise seed from the fill after ``n_pretrack``
        turns.

    Returns
    -------
    complex
        Seed antenna voltage [V].
    """
    lam = -omega / (2.0 * q_l) + 1j * delta_omega
    v_ss = -(r_over_q * omega) * generator_current / lam

    fill_time = n_pretrack * t_rev
    if injection_voltage is None:
        return v_ss * (1.0 - np.exp(lam * fill_time))

    # Scan the fill transient for the first time |V(t)| reaches the injection
    # target. Resolve the fill time constant tau = 2 Q_L / omega with ~200
    # points (well past the crossing, which sits on the initial rise), capped
    # so an over-long budget stays affordable.
    tau = 2.0 * q_l / omega
    n_points = int(np.clip(200.0 * fill_time / tau, 2000, 2_000_000))
    t = np.linspace(0.0, fill_time, n_points)
    voltage = v_ss * (1.0 - np.exp(lam * t))
    magnitude = np.abs(voltage)

    if magnitude.max() < injection_voltage:
        raise ValueError(
            f"injection_voltage ({injection_voltage:.3g} V) is not reached "
            f"within {n_pretrack} pre-fill turns; the fill only reaches "
            f"{magnitude.max():.3g} V. Increase the generator current, the "
            "detuning, or n_pretrack, or lower injection_voltage."
        )

    # First grid point at/above the target, then linearly interpolate the
    # crossing time between it and the previous point for sub-grid accuracy.
    idx = int(np.argmax(magnitude >= injection_voltage))
    step = magnitude[idx] - magnitude[idx - 1]
    frac = (
        (injection_voltage - magnitude[idx - 1]) / step if step != 0 else 0.0
    )
    t_cross = t[idx - 1] + frac * (t[idx] - t[idx - 1])
    return v_ss * (1.0 - np.exp(lam * t_cross))
