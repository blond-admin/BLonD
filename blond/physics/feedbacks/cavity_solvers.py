# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
Muon-collider cavity-response solvers.

These are used only by the muon-collider timing-class feedback
(:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`).
:func:`cavity_response_sparse_matrix` is the first-order (forward-Euler)
fine-grid solver the timing class uses by default;
:func:`cavity_response_sparse_matrix_second_order` is its second-order
(trapezoidal / Crank-Nicolson) twin.

The coarse-grid recursion the timing class runs on top of those solvers is
the exact exponential propagator of the cavity envelope, built from
:func:`coarse_step_exponent` and the propagator weights
:func:`exponential_voltage_multiplier` and :func:`exponential_drive_weight`.
They live here so that the per-cell (reference) and vectorised
(numba-kernel) coarse paths in
:mod:`~blond.physics.feedbacks.cavity_feedback` spell the step arithmetic
once. The derivation of that step -- and why the forward-Euler coarse step it
replaced (removed 2026-09-11) was only its first-order truncation -- is in
the Notes of ``IQCavityFeedbackTimingClass._advance_coarse_voltage``.
:func:`propagate_beam_free_voltage` applies the same closed form to the
beam-free seed propagation.

On naming: the step size these solvers take is spelled ``omega_times_dt``
everywhere in the muon-collider feedback -- the RF phase advanced in one step
[rad], a literal transcription of the formula symbol ``omega * dt``. Two
earlier spellings were retired in favour of it. ``samples_per_rf`` asserted
the *reciprocal* of what it held (callers pass ``omega * dt``, ~0.06 rad, not
a sample count), so a reader taking the name at face value inverted the
quantity. ``omega_times_T_s`` collided with the domain convention that
``T_s`` is the *synchrotron* period -- a bad ambiguity in files full of
synchrotron motion. One quantity, one name; do not reintroduce a synonym.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def propagate_beam_free_voltage(
    initial_voltage: complex,
    generator_current: NumpyArray,
    rf_centers: NumpyArray,
    end_time: float,
    omega: float,
    R_over_Q: float,
    Q_L: float,
    delta_omega: float,
) -> complex:
    """
    Evolve a coarse seed to a new time without depositing beam charge.

    Parameters
    ----------
    initial_voltage
        Per-cavity envelope at ``rf_centers[0]`` [V].
    generator_current
        Applied, limited generator commands [A], in the voltage's IQ frame.
        Command ``i`` is held from centre ``i`` to centre ``i + 1``.
    rf_centers
        Increasing command timestamps [s], in the passage-local frame.
    end_time
        Target time [s]. Outside the grid, the nearest command is held.
        Backward evolution is only appropriate for a charge-free window;
        the caller must reject a charged window preceding its seed.
    omega
        Carrier angular frequency of this segment [rad/s].
    R_over_Q
        Cavity R over Q [ohm].
    Q_L
        Loaded quality factor.
    delta_omega
        Cavity detuning [rad/s].

    Returns
    -------
    voltage
        Per-cavity envelope at ``end_time`` [V]. No controller is stepped.
    """
    voltage = initial_voltage
    start_time = float(rf_centers[0])
    stop_index = np.searchsorted(rf_centers, end_time, side="left")
    endpoints = np.append(rf_centers[1:stop_index], end_time)
    for index, next_time in enumerate(endpoints):
        time_step = next_time - start_time
        step_exponent = (-omega / (2 * Q_L) + 1j * delta_omega) * time_step
        voltage = np.exp(
            step_exponent
        ) * voltage + R_over_Q * omega * time_step * generator_current[
            index
        ] * exponential_drive_weight(step_exponent)
        start_time = next_time
    return voltage


def cavity_response_sparse_matrix(
    I_beam: NumpyArray,
    I_gen: NumpyArray,
    V_ant_init: float,
    I_gen_init: float,
    omega_times_dt: float,
    R_over_Q: float,
    Q_L: float,
    relative_detuning: float,
    *,
    initial_at_bin_edge: bool = False,
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
    omega_times_dt : float
        RF phase advanced in one sampling step [rad], i.e.
        ``omega_rf * sampling_time``; callers pass
        ``omega_input * profile.hist_step``.
    R_over_Q : float
        The R over Q of the cavity.
    Q_L : float
        The loaded quality factor of the cavity.
    relative_detuning : float
        The detuning of the cavity in frequency divided by the rf frequency.
    initial_at_bin_edge
        If True, the initial state is at the first histogram bin's left
        edge and output samples are at bin centres. The first step is a
        half step. Beam current is a bin density: later steps integrate
        half of each adjacent bin. Voltage and generator drive retain
        forward-Euler stepping. False preserves the uniform-step API.

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
    A = 0.5 * R_over_Q * omega_times_dt
    B = (
        1
        - 0.5 * omega_times_dt / Q_L
        + 1j * relative_detuning * omega_times_dt
    )

    # Initialize the two sparse matrices needed to find antenna voltage
    sub_diagonal = np.full(n_samples - 1, -B, dtype=complex)
    if initial_at_bin_edge and n_samples > 1:
        sub_diagonal[0] = -(1 + 0.5 * (B - 1))
    B_matrix = diags(
        [sub_diagonal, 1],
        [-1, 0],
        (n_samples, n_samples),
        dtype=complex,
        format="csc",
    )
    I_matrix = diags([A], [-1], (n_samples, n_samples), dtype=complex)

    # Find vector on the "current" side of the equation
    b = I_matrix.dot(2 * internal_I_gen - internal_I_beam)
    b[0] = V_ant_init
    if initial_at_bin_edge and n_samples > 1:
        b[1] = 0.5 * A * (2 * I_gen_init - I_beam[0])
        b[2:] = A * (2 * I_gen[:-1] - 0.5 * (I_beam[:-1] + I_beam[1:]))

    # Solve the sparse linear system of equations and return
    return spsolve(B_matrix, b)[1:]
    # first value is intial condition


def cavity_response_sparse_matrix_second_order(
    I_beam: NumpyArray,
    I_gen: NumpyArray,
    V_ant_init: float,
    I_gen_init: float,
    omega_times_dt: float,
    R_over_Q: float,
    Q_L: float,
    relative_detuning: float,
    *,
    initial_at_bin_edge: bool = False,
):
    r"""
    Second-order (trapezoidal / Crank-Nicolson) ACS cavity response solver.

    Drop-in alternative to
    :func:`cavity_response_sparse_matrix`.
    It solves the same cavity-envelope ODE

    .. math::
        \frac{\mathrm{d}V}{\mathrm{d}t}
        = \Big(-\frac{\omega}{2 Q_L} + i\,\Delta\omega\Big) V
          + \frac{R/Q\,\omega}{2}\,(2 I_{\mathrm{gen}} - I_{\mathrm{beam}}),

    but integrates it with the trapezoidal rule (averaging the homogeneous
    term *and* the current drive over each step) instead of the forward-Euler
    (left-endpoint) step used by
    :func:`cavity_response_sparse_matrix`.
    The truncation error is therefore :math:`O(\Delta t^2)` rather than
    :math:`O(\Delta t)`, which matters most at coarse binning (large
    ``omega_times_dt``).

    With ``lam = -0.5 * omega_times_dt / Q_L + 1j * relative_detuning *
    omega_times_dt`` (so ``B = 1 + lam`` of the first-order solver) and the
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
    omega_times_dt : float
        RF phase advanced in one sampling step [rad], i.e.
        ``omega_rf * sampling_time``; callers pass
        ``omega_input * profile.hist_step``.
    R_over_Q : float
        The R over Q of the cavity.
    Q_L : float
        The loaded quality factor of the cavity.
    relative_detuning : float
        The detuning of the cavity in frequency divided by the rf frequency.
    initial_at_bin_edge
        If True, the initial state is at the first histogram bin's left
        edge. The first step reaches its centre with half a step of decay
        and generator drive, and half a bin of beam charge. Subsequent
        steps span adjacent centres. False preserves the uniform-step API.

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

    A = 0.5 * R_over_Q * omega_times_dt
    # lam == B - 1 of the first-order solver, i.e. (step size) * (decay/detuning)
    lam = -0.5 * omega_times_dt / Q_L + 1j * relative_detuning * omega_times_dt

    # Per-step current drive, identical to the first-order solver's source term
    s = A * (2 * internal_I_gen - internal_I_beam)

    # Bidiagonal trapezoidal system. Row 0 pins the initial condition
    # (diagonal 1), all later rows use the Crank-Nicolson coefficients.
    diagonal = np.full(n_samples, 1 - 0.5 * lam, dtype=complex)
    diagonal[0] = 1.0
    sub_diagonal = np.full(n_samples - 1, -(1 + 0.5 * lam), dtype=complex)
    if initial_at_bin_edge and n_samples > 1:
        diagonal[1] = 1 - 0.25 * lam
        sub_diagonal[0] = -(1 + 0.25 * lam)
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
    if initial_at_bin_edge and n_samples > 1:
        b[1] = 0.5 * A * (I_gen_init + I_gen[0] - I_beam[0])

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


def coarse_step_exponent(
    omega_times_dt: NumpyArray | float,
    Q_L: float,
    relative_detuning: float,
) -> NumpyArray | complex:
    r"""
    Dimensionless growth exponent of one coarse-grid step.

    The coarse recursion advances the cavity envelope over a step of length
    :math:`\Delta t` with :math:`L = \lambda\,\Delta t`, where
    :math:`\lambda = -\omega / (2 Q_L) + i\,\Delta\omega` is the cavity pole.
    Expressed through ``omega * dt`` and the detuning normalised to the step
    frequency, that is ``L = -0.5 * omega_times_dt / Q_L + 1j *
    relative_detuning * omega_times_dt``.

    Scalar and array inputs are put through the identical expression, so the
    per-cell reference recursion and its vectorised (kernel) twin agree
    bit-for-bit.

    Parameters
    ----------
    omega_times_dt
        RF phase advanced in one step [rad], i.e. ``omega * dt``; a scalar
        for the per-cell path, a per-cell array for the vectorised one.
    Q_L
        The loaded quality factor of the cavity.
    relative_detuning
        The detuning of the cavity normalised to the step frequency
        (``delta_omega / omega``).

    Returns
    -------
    complex or complex array
        The growth exponent ``L``, shaped like ``omega_times_dt``.
    """
    return (
        -0.5 * omega_times_dt / Q_L + 1j * relative_detuning * omega_times_dt
    )


def exponential_voltage_multiplier(
    step_exponent: NumpyArray | complex,
) -> NumpyArray | complex:
    """
    Exact voltage multiplier ``B = e^L`` of one coarse step.

    The homogeneous part of the exponential propagator, which integrates the
    cavity decay and the detuning rotation exactly, so ``|B| <= 1`` for every
    step. Derivation, and how the retired forward-Euler factor ``1 + L``
    truncates it: Notes of
    ``IQCavityFeedbackTimingClass._advance_coarse_voltage``.

    Parameters
    ----------
    step_exponent
        Growth exponent ``L`` of the step, from
        :func:`coarse_step_exponent`.

    Returns
    -------
    complex or complex array
        The voltage multiplier ``B``, shaped like ``step_exponent``.
    """
    return np.exp(step_exponent)


def exponential_drive_weight(
    step_exponent: NumpyArray | complex,
) -> NumpyArray | complex | float:
    """
    Drive weight ``W = (e^L - 1) / L`` of the exponential propagator.

    Weight of the piecewise-constant per-step drive in the exact propagator
    ``V_next = e^L V + src * W`` (derived in the Notes of
    ``IQCavityFeedbackTimingClass._advance_coarse_voltage``; the retired
    forward-Euler step used ``W = 1``, its zeroth-order truncation).
    ``np.expm1`` keeps it accurate (``-> 1``) as ``L -> 0``.

    The removable singularity at ``L == 0`` is guarded for scalar input only.
    The per-cell path can legitimately be handed a zero step (a caller may
    advance the recursion by ``omega * dt == 0``), whereas the vectorised
    path never sees one: its caller filters coincident coarse points out of
    the whole segment and defers them to the per-cell reference loop, which
    warns and duplicates the previous cell into them rather than advancing
    the recursion at all. Making the guard elementwise would therefore add a
    full extra pass over every cell of the hot recursion for a branch that is
    unreachable there, so the rank test is deliberate rather than an
    oversight.

    Parameters
    ----------
    step_exponent
        Growth exponent ``L`` of the step, from
        :func:`coarse_step_exponent`.

    Returns
    -------
    complex or complex array or float
        The drive weight ``W``, shaped like ``step_exponent``; ``1.0`` for a
        scalar zero exponent.
    """
    if np.ndim(step_exponent) == 0 and step_exponent == 0:
        return 1.0
    return np.expm1(step_exponent) / step_exponent
