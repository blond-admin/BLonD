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
from numpy._typing import NDArray as NumpyArray
from scipy.constants.constants import elementary_charge
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
    phi_s: float = 0,
    forbid_charge_in_first_coarse_cell: bool = False,
    dT_index_sign: float = 1.0,
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
    phi_s : float
        Dummy.
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
    # charges = (
    #     beam.ratio  # TODO: remove this
    #     * beam.particle_type.charge
    #     * elementary_charge
    #     * profile.hist_y
    # )

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

    # Mix with frequency of interest; remember factor 2 demodulation
    # TODO: where do we have to apply the demodulation?
    I_f = 2.0 * charges * np.cos(omega_c * hist_x)  # TODO: flipperydoo?
    Q_f = -2.0 * charges * np.sin(omega_c * hist_x)

    # Pass through a low-pass filter
    if use_lowpass_filter is True:
        # Nyquist frequency 0.5*f_slices; cutoff at 20 MHz
        cutoff = 20.0e6 * 2.0 * profile.hist_step
        I_f = low_pass_filter(I_f, cutoff_frequency=cutoff)
        Q_f = low_pass_filter(Q_f, cutoff_frequency=cutoff)
    logger.debug("RF total current is %.4e A", np.fabs(np.sum(I_f)) / T_rev)

    charges_fine = I_f + 1j * Q_f
    # dT = 0
    if external_reference:
        # slippage in phase due to a non-integer harmonic number
        dphi = dT * omega_c
        # Total phase correction
        # dphi = 0
        # phase = dphi
        charges_fine = charges_fine * np.exp(
            1j * (dphi + np.pi / 2)  # TODO: why?
        )  # TODO: +1j or -1j?

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
        # np.pi / omega_c --> center of T_s (bin_center)
        ind_fine = np.round(
            (hist_x + dT_index_sign * dT - np.pi / omega_c) / T_s
        )
        ind_fine = np.array(ind_fine, dtype=int)
        indices = np.where((ind_fine[1:] - ind_fine[:-1]) == 1)[0]
        if any(indices < 0):
            raise RuntimeError("yorak")
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
                if i + ind_fine[0] > n_points:
                    raise RuntimeError("yorak")
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
        # print(np.angle(charges_coarse[1], deg=True))
        return charges_fine, charges_coarse

    else:
        return charges_fine


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
