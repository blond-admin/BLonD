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
built from :func:`coarse_step_exponent` and the propagator weights
:func:`euler_voltage_multiplier`, :func:`exponential_voltage_multiplier` and
:func:`exponential_drive_weight`. They live here so that the per-cell
(reference) and vectorised (numba-kernel) coarse paths in
:mod:`~blond.physics.feedbacks.cavity_feedback` spell the step arithmetic
once, and so that :class:`ForwardEulerValidityGuard` sits beside the very
formula it caps.

:class:`ForwardEulerValidityGuard` collects the tripwires that decide whether
the forward-Euler discretisation is admissible at all -- the per-step decay,
detuning phase and beam kick. It lives here, beside the solvers it certifies,
because it is pure numerics: the feedback owns one instance and passes the
cavity parameters per call.

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

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def cavity_response_sparse_matrix(
    I_beam: NumpyArray,
    I_gen: NumpyArray,
    V_ant_init: float,
    I_gen_init: float,
    omega_times_dt: float,
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
    omega_times_dt: float,
    R_over_Q: float,
    Q_L: float,
    relative_detuning: float,
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


def euler_voltage_multiplier(
    step_exponent: NumpyArray | complex,
) -> NumpyArray | complex:
    """
    Forward-Euler voltage multiplier ``B = 1 + L`` of one coarse step.

    The left-endpoint (forward-Euler) discretisation of the cavity envelope
    ODE multiplies the previous antenna voltage by ``1 + L``, the first two
    terms of the exact propagator ``e^L`` -- which is why the step must stay
    small, and what :class:`ForwardEulerValidityGuard` caps.

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
    return 1.0 + step_exponent


def exponential_voltage_multiplier(
    step_exponent: NumpyArray | complex,
) -> NumpyArray | complex:
    """
    Exact voltage multiplier ``B = e^L`` of one coarse step.

    The homogeneous part of the exponential propagator, which integrates the
    cavity decay and the detuning rotation exactly and is unconditionally
    stable. It replaces :func:`euler_voltage_multiplier` when
    ``exponential_coarse_solver_enable`` is set, and is not subject to the
    forward-Euler step-size caps.

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
    ``V_next = e^L V + src * W``. ``np.expm1`` keeps it accurate (``-> 1``)
    as ``L -> 0``.

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


class ForwardEulerValidityGuard:
    """
    Validity tripwires for the forward-Euler cavity discretisation.

    The coarse recursion advances the antenna voltage by the forward-Euler
    factor :func:`euler_voltage_multiplier` of the step exponent
    :func:`coarse_step_exponent`, i.e. ``(1 - 0.5 * omega * dt / Q_L + 1j *
    delta_omega * dt)``, plus the beam-loading increment
    ``-I_beam * 0.5 * R_over_Q * omega * dt``. Each is only an accurate
    discretisation of the underlying ODE while it represents a small change
    per step, so this guard tests exactly those three magnitudes: the
    per-step decay, the per-step detuning phase, and the per-step beam kick
    relative to the voltage it acts on.

    It lives beside the step arithmetic it caps -- and the solvers it
    certifies -- rather than on the feedback, because it is pure numerics: it
    reads no grid, no RF station and no beam, and the only state it owns is
    the once-only beam-kick warning flag. All cavity parameters are passed
    per call, so a caller that mutates them between calls is measured against
    their current values.

    Set ``enabled=False`` for the exact exponential propagator
    (``exponential_coarse_solver_enable=True``, i.e.
    :func:`exponential_voltage_multiplier` and
    :func:`exponential_drive_weight` in place of the Euler factor): it
    integrates the decay, the detuning and the piecewise-constant drive --
    beam included -- exactly, so none of these are discretisation errors
    there. That is also precisely the low-``Q_L`` / large-detuning regime the
    option exists to enable, so applying the caps would defeat its documented
    purpose.

    Parameters
    ----------
    enabled
        Whether the tripwires apply. ``False`` makes every check a no-op.
    """

    # Heuristic thresholds for forward-Euler validity [rad, and relative].
    #: Soft (warning) threshold for the per-step decay and detuning phase.
    max_step_angle = 0.1
    # Hard cap on the per-step decay ``d = decay_per_step``. The forward-Euler
    # decay factor (real part, ignoring detuning) is ``1 - d``:
    #   * d < 1:      factor in (0, 1)  -- ordinary contraction.
    #   * 1 < d < 2:  factor in (-1, 0) -- the sign flips every step
    #                 (unphysical), but |factor| < 1 so the voltage still
    #                 contracts (oscillatory decay).
    #   * d > 2:      |factor| > 1      -- the magnitude grows every step: a
    #                 genuinely divergent discretization.
    # The exact factor ``exp(-omega dt / (2 Q_L))`` is positive for every step
    # size, so the whole band d > 1 misrepresents the physics -- the cap
    # deliberately sits at the sign-flip boundary d = 1, not at the divergence
    # boundary d = 2: contracting while inverting every step is still wrong,
    # just not explosively so.
    #: Hard (raising) threshold for the per-step decay and detuning phase.
    max_step_angle_hard = 1.0
    #: Soft (warning) threshold for the relative per-step beam kick.
    max_relative_kick = 0.1
    # Beyond this, the single-step beam kick exceeds the antenna voltage it is
    # being subtracted from/added to: the discretized update can flip the sign
    # of the antenna voltage within one coarse-grid step purely due to the
    # beam, which the underlying continuous cavity equation cannot do -- a
    # divergent/unphysical discretization rather than just an inaccurate one.
    #: Hard (raising) threshold for the relative per-step beam kick.
    max_relative_kick_hard = 1.0

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._beam_kick_warning_issued = False

    def check_step_sizes(
        self,
        omega_rf: float,
        sampling_time: float,
        Q_L: float,
        delta_omega: float,
    ) -> None:
        """
        Check that the per-step decay and detuning phase are not too large.

        Parameters
        ----------
        omega_rf
            Angular RF frequency the coarse step is advanced with [rad/s].
            NB: this is the frequency ``cavity_response`` uses as
            ``omega_input``, not the carrier frequency ``omega_rf / n``.
            Using the carrier would cancel the
            ``n_rf_periods_per_coarse_grid`` dependence to a constant
            ``2 * pi`` and misjudge the stability of the discretization.
        sampling_time
            Coarse-grid sampling time (the per-step ``dt``) [s].
        Q_L
            Loaded quality factor of the cavity.
        delta_omega
            Cavity detuning (resonance minus RF) [rad/s].

        Raises
        ------
        ValueError
            If the per-step decay or detuning phase exceeds the hard cap.
        """
        if not self.enabled:
            return

        omega_times_dt = omega_rf * sampling_time
        decay_per_step = 0.5 * omega_times_dt / Q_L
        detuning_phase_per_step = delta_omega * sampling_time

        if decay_per_step > self.max_step_angle_hard:
            raise ValueError(
                f"{decay_per_step=:.3g} > {self.max_step_angle_hard}: the "
                "forward-Euler decay factor (1 - decay_per_step) used in "
                "cavity_response() is negative, so the discretized cavity "
                "voltage inverts its sign every step -- an unphysical "
                "per-step sign inversion, since the exact decay factor "
                "exp(-omega * dt / (2 * Q_L)) is always positive. (It stays "
                "contracting, |factor| < 1, until decay_per_step exceeds 2, "
                "beyond which the response also diverges.) "
                "Increase Q_L or decrease n_rf_periods_per_coarse_grid; for "
                "steps that must stay this large, set "
                "exponential_coarse_solver_enable=True to use the exact "
                "propagator, which integrates the decay exactly and is not "
                "subject to this check."
            )
        if decay_per_step > self.max_step_angle:
            warnings.warn(
                f"{decay_per_step=:.3g} is not << 1: the forward-Euler "
                "approximation of the cavity decay "
                "(1 - 0.5 * omega * dt / Q_L) used in cavity_response() "
                "may be inaccurate; consider increasing Q_L or decreasing "
                "n_rf_periods_per_coarse_grid.",
                stacklevel=3,
            )
        if abs(detuning_phase_per_step) > self.max_step_angle_hard:
            raise ValueError(
                f"{detuning_phase_per_step=:.3g} > "
                f"{self.max_step_angle_hard}: "
                "the forward-Euler approximation of the detuning-induced "
                "phase rotation (1 + 1j * delta_omega * dt) used in "
                "cavity_response() rotates the antenna voltage by more "
                "than one step's worth of angle per coarse-grid sample, "
                "i.e. the discretization can no longer track the cavity "
                "phase. Decrease delta_omega or "
                "n_rf_periods_per_coarse_grid."
            )
        if abs(detuning_phase_per_step) > self.max_step_angle:
            warnings.warn(
                f"{detuning_phase_per_step=:.3g} is not << 1: the "
                "forward-Euler approximation of the detuning-induced phase "
                "rotation (1 + 1j * delta_omega * dt) used in "
                "cavity_response() may be inaccurate; consider decreasing "
                "delta_omega or n_rf_periods_per_coarse_grid.",
                stacklevel=3,
            )

    def check_beam_kick_magnitude(
        self,
        beam_current: complex | float | int,
        omega_times_dt: float | int,
        previous_voltage: complex | float | int,
        R_over_Q: float,
    ) -> None:
        """
        Warn (once) if the beam-induced voltage kick is too large.

        Warns if the beam-induced voltage kick within a single coarse-grid
        step is not small compared to the antenna voltage it is added to.

        Parameters
        ----------
        beam_current
            Beam current sample used for this step [A].
        omega_times_dt
            RF phase advanced in this step [rad], i.e. ``omega * dt``.
        previous_voltage
            Antenna voltage of the previous coarse-grid step, which the
            kick is added to/subtracted from.
        R_over_Q
            The R over Q of the cavity [Ohm].

        Raises
        ------
        ValueError
            If the relative beam kick exceeds the hard cap.
        """
        if not self.enabled:
            return
        if beam_current == 0:
            return

        beam_kick = beam_current * 0.5 * R_over_Q * omega_times_dt
        previous_voltage_abs = np.abs(previous_voltage)
        if previous_voltage_abs == 0:
            return

        relative_kick = np.abs(beam_kick) / previous_voltage_abs
        if relative_kick > self.max_relative_kick_hard:
            raise ValueError(
                f"{relative_kick=:.3g} > {self.max_relative_kick_hard}: the "
                "beam-induced voltage kick per coarse-grid step "
                "(beam_current * 0.5 * R_over_Q * omega * dt) exceeds the "
                "antenna voltage it acts on, i.e. the forward-Euler update "
                "in cavity_response() can flip the sign of the antenna "
                "voltage within a single step -- unphysical for the "
                "underlying cavity ODE. Decrease "
                "n_rf_periods_per_coarse_grid or check whether the beam "
                "current/intensity is physically reasonable for this "
                "cavity."
            )
        if self._beam_kick_warning_issued:
            return
        if relative_kick > self.max_relative_kick:
            self._beam_kick_warning_issued = True
            warnings.warn(
                f"{relative_kick=:.3g} is not << 1: the beam-induced "
                "voltage kick per coarse-grid step "
                "(beam_current * 0.5 * R_over_Q * omega * dt) is large "
                "compared to the antenna voltage. The forward-Euler update "
                "in cavity_response() may be inaccurate; consider "
                "decreasing n_rf_periods_per_coarse_grid or checking "
                "whether the beam current/intensity is physically "
                "reasonable for this cavity.",
                stacklevel=3,
            )

    def check_beam_kicks(
        self,
        beam_current: NumpyArray,
        omega_times_dt: NumpyArray,
        voltage_init: complex,
        voltage_out: NumpyArray,
        skip_first: bool,
        R_over_Q: float,
    ) -> None:
        """
        Vectorised beam-kick tripwire for a forward kernel segment.

        Reproduces the per-cell :meth:`check_beam_kick_magnitude` sweep of the
        reference path: it locates the offending cells with vectorised numpy
        and then delegates to :meth:`check_beam_kick_magnitude` for the actual
        raise/warn, so the exception and warning messages (and the once-only
        warning flag) are identical. A merely-large kick that precedes any
        hard violation warns first, matching the reference ordering.

        Parameters
        ----------
        beam_current
            Per-cell beam current of the segment.
        omega_times_dt
            Per-cell RF phase advanced in one step [rad], i.e.
            ``omega * dt``.
        voltage_init
            Antenna voltage preceding the first cell.
        voltage_out
            Per-cell antenna voltage just computed for the segment.
        skip_first
            Whether to skip the first cell (the carried ``rf_centers`` index 0,
            which the reference never checks).
        R_over_Q
            The R over Q of the cavity [Ohm].
        """
        if not self.enabled:
            return
        previous_voltage = np.empty_like(voltage_out)
        previous_voltage[0] = voltage_init
        previous_voltage[1:] = voltage_out[:-1]
        beam_kick = beam_current * 0.5 * R_over_Q * omega_times_dt
        previous_voltage_abs = np.abs(previous_voltage)
        valid = (beam_current != 0) & (previous_voltage_abs != 0)
        if skip_first:
            valid[0] = False
        if not valid.any():
            return
        relative_kick = np.zeros(beam_current.shape[0], dtype=np.float64)
        relative_kick[valid] = (
            np.abs(beam_kick[valid]) / previous_voltage_abs[valid]
        )
        hard = valid & (relative_kick > self.max_relative_kick_hard)
        soft = valid & (relative_kick > self.max_relative_kick)
        first_hard = int(np.argmax(hard)) if hard.any() else -1
        first_soft = int(np.argmax(soft)) if soft.any() else -1

        # Warn once if a merely-large kick precedes any hard violation, so the
        # observable warn-then-raise ordering matches the per-cell loop.
        if first_soft >= 0 and (first_hard < 0 or first_soft < first_hard):
            self.check_beam_kick_magnitude(
                beam_current[first_soft],
                omega_times_dt[first_soft],
                previous_voltage[first_soft],
                R_over_Q,
            )
        if first_hard >= 0:
            self.check_beam_kick_magnitude(
                beam_current[first_hard],
                omega_times_dt[first_hard],
                previous_voltage[first_hard],
                R_over_Q,
            )
