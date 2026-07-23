# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


"""
Muon-collider cavity-response solvers.

Split out of ``helpers.py``: these are used only by the muon-collider
timing-class feedback (:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`).
The first-order ``cavity_response_sparse_matrix`` stays in ``helpers.py``
because the (experimental) LHC cavity feedback uses it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


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

    Drop-in alternative to
    :func:`~blond.physics.feedbacks.helpers.cavity_response_sparse_matrix`.
    It solves the same cavity-envelope ODE

    .. math::
        \frac{\mathrm{d}V}{\mathrm{d}t}
        = \Big(-\frac{\omega}{2 Q_L} + i\,\Delta\omega\Big) V
          + \frac{R/Q\,\omega}{2}\,(2 I_{\mathrm{gen}} - I_{\mathrm{beam}}),

    but integrates it with the trapezoidal rule (averaging the homogeneous
    term *and* the current drive over each step) instead of the forward-Euler
    (left-endpoint) step used by
    :func:`~blond.physics.feedbacks.helpers.cavity_response_sparse_matrix`.
    The truncation error is therefore :math:`O(\Delta t^2)` rather than
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
