# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Compiled single-turn kernel of the LHC ACS cavity loop.

The cavity loop is a sample-by-sample recursion: every coarse sample
depends on the preceding one, so the turn cannot be vectorised. Running it
as Python method calls costs tens of millions of interpreter calls per
turn, which dominates the run time. This module runs the same recursion
compiled.

Each block of the loop is one function here, mirroring the method of the
same name on :class:`LHCCavityFeedback`, so that the kernel can be read
next to the Python reference implementation it must reproduce
(``LHCCavityFeedback.track_one_turn_reference``). The blocks are declared
``inline="always"``, so the compiler sees one flat loop regardless of the
split: measured against a single monolithic function, the modular form
costs nothing.

All signals travel as **one** array, ``signals``, of shape
``(N_SIGNALS, 2 * n_coarse)``: row ``V_ANT`` holds the antenna voltage,
and so on, with :data:`SIGNAL_NAMES` giving the order. Within a row the
previous turn is followed by the current turn, so the index
``n_coarse + i`` of sample ``i`` reproduces the reach-back semantics of
:class:`~blond.physics.feedbacks.buffers.TwoTurnArray`: a negative offset
simply walks into the previous turn. The scalars travel as one
:class:`Settings` tuple. Passing the signals as a block rather than as one
argument per buffer is what keeps the signatures short; measured against
one argument per buffer it costs nothing, whereas holding the arrays in a
tuple instead is about three times slower and cannot be cached.

:data:`SIGNAL_NAMES` is the single definition of the row order:
:class:`LHCCavityFeedbackCoarseBuffers` allocates the block and hands out
its rows in that order, so adding or renaming a signal is one edit here
plus its dataclass field, and no signature changes.

This kernel is CPU-only (NumPy arrays, Numba ``njit``), matching the
feedback buffers, which are allocated as NumPy arrays and never live on a
device.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from numba import njit

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

#: Row order of the ``signals`` block, and therefore the order in which
#: :class:`LHCCavityFeedbackCoarseBuffers` hands out its rows.
SIGNAL_NAMES = (
    "v_setpoint",
    "v_ant",
    "i_beam",
    "i_gen",
    "v_excitation",
    "v_feedback_in",
    "v_analog_in",
    "i_analog_out",
    "i_digital_out",
    "i_feedback_out",
    "v_otfb_ac_in",
    "v_otfb_comb",
    "v_otfb_fir_out",
    "v_otfb_out",
    "i_swap_out",
    "i_gen_test",
    "i_gen_predrive",
    "tuner_in",
    "tuner_integrated",
)

# Row indices, in the order of SIGNAL_NAMES. Numba freezes these globals
# at compile time, so they cost nothing at run time.
(
    V_SETPOINT,
    V_ANT,
    I_BEAM,
    I_GEN,
    V_EXCITATION,
    V_FEEDBACK_IN,
    V_ANALOG_IN,
    I_ANALOG_OUT,
    I_DIGITAL_OUT,
    I_FEEDBACK_OUT,
    V_OTFB_AC_IN,
    V_OTFB_COMB,
    V_OTFB_FIR_OUT,
    V_OTFB_OUT,
    I_SWAP_OUT,
    I_GEN_TEST,
    I_GEN_PREDRIVE,
    TUNER_IN,
    TUNER_INTEGRATED,
) = range(len(SIGNAL_NAMES))

#: Number of rows in the ``signals`` block.
N_SIGNALS = len(SIGNAL_NAMES)


class Settings(NamedTuple):
    """
    Scalar settings of the cavity loop for one turn.

    A tuple of scalars is free to pass into a compiled kernel, so the
    whole parameter set travels as one argument.

    Attributes
    ----------
    n_coarse
        Number of samples per turn on the coarse grid.
    n_delay
        Loop delay in coarse samples.
    n_otfb
        OTFB delay in coarse samples.
    t_s
        Sampling time [s] on the coarse grid.
    samples
        Samples per RF period, ``omega_rf * t_s``.
    r_over_q
        Cavity R/Q [Ohm].
    q_l
        Cavity loaded quality factor.
    detuning
        Relative cavity detuning.
    alpha
        Memory parameter of the OTFB comb filter.
    gain_analog
        Gain of the analog feedback branch.
    gain_digital
        Gain of the digital feedback branch.
    gain_otfb
        Gain of the one-turn delay feedback.
    gain_generator
        Overall driver chain gain.
    d_phi_ad
        Phase misalignment [rad] of the digital w.r.t. the analog branch.
    tau_a
        Analog feedback time constant [s].
    tau_d
        Digital feedback time constant [s].
    tau_o
        OTFB AC coupling time constant [s].
    i_gen_offset
        Generator current offset [A].
    i_swap_threshold
        Current [A] at which the switch-and-protect module acts.
    open_drive
        Multiplier that is zero when the drive is open.
    open_drive_inv
        Complement of ``open_drive``.
    open_loop
        Multiplier that is zero when the loop is open.
    open_otfb
        Multiplier that is zero when the OTFB is open.
    open_rffb
        Multiplier that is zero when the RF feedback is open.
    excitation_otfb
        Multiplier that is one while injecting noise at the OTFB output.
    clamping
        Whether the switch-and-protect limiter is active.
    saturation
        Whether the klystron saturation curve is applied.
    enable_klystron
        Whether the klystron bandwidth FIR filter is applied.
    """

    n_coarse: int
    n_delay: int
    n_otfb: int
    t_s: float
    samples: float
    r_over_q: float
    q_l: float
    detuning: float
    alpha: float
    gain_analog: float
    gain_digital: float
    gain_otfb: float
    gain_generator: float
    d_phi_ad: float
    tau_a: float
    tau_d: float
    tau_o: float
    i_gen_offset: float
    i_swap_threshold: float
    open_drive: float
    open_drive_inv: float
    open_loop: float
    open_otfb: float
    open_rffb: float
    excitation_otfb: float
    clamping: bool
    saturation: bool
    enable_klystron: bool


class Coefficients(NamedTuple):
    """
    Quantities that are constant over a turn, derived from the settings.

    Attributes
    ----------
    cavity_decay
        Per-sample decay and rotation of the antenna voltage.
    ac_coupling
        Per-sample decay of the AC couplers, ``1 - t_s / tau_o``.
    analog_decay
        Per-sample decay of the analog branch, ``1 - t_s / tau_a``.
    digital_decay
        Per-sample decay of the digital branch, ``1 - t_s / tau_d``.
    digital_gain
        Complex gain of the digital branch, including its phase offset.
    saturation_onset
        Current [A] beyond which the klystron gain starts to decrease.
    saturation_span
        Width [A] from the onset to the point of zero klystron gain.
    """

    cavity_decay: complex
    ac_coupling: float
    analog_decay: float
    digital_decay: float
    digital_gain: complex
    saturation_onset: float
    saturation_span: float


# The blocks below are inlined into the driver loop.
_INLINED = {"cache": True, "fastmath": False, "inline": "always"}


@njit(**_INLINED)
def _coefficients(settings: Settings) -> Coefficients:  # pragma: no cover
    # Everything that does not change from sample to sample.
    return Coefficients(
        1
        - 0.5 * settings.samples / settings.q_l
        + 1j * settings.detuning * settings.samples,
        1 - settings.t_s / settings.tau_o,
        1 - settings.t_s / settings.tau_a,
        1 - settings.t_s / settings.tau_d,
        settings.t_s
        / settings.tau_d
        * settings.gain_analog
        * settings.gain_digital
        * (math.cos(settings.d_phi_ad) + 1j * math.sin(settings.d_phi_ad)),
        0.8 * settings.i_swap_threshold,
        settings.i_swap_threshold - 0.8 * settings.i_swap_threshold,
    )


@njit(**_INLINED)
def _saturated_magnitude(  # pragma: no cover
    magnitude: float, onset: float, span: float
) -> float:
    # Scalar form of `klystron_saturation_curve`: linear up to `onset`,
    # then rolled off by a cubic reaching zero gain one `span` further.
    k = 1.0 / span**2
    sign = 0.0
    if magnitude > 0.0:
        sign = 1.0
    elif magnitude < 0.0:
        sign = -1.0
    absolute = abs(magnitude)
    overshoot = absolute - onset
    overshoot = max(overshoot, 0.0)
    inner = absolute
    inner = min(inner, onset)
    outer = overshoot - k * overshoot**3 / 3.0
    return sign * (inner + outer)


@njit(**_INLINED)
def _clipped_magnitude(  # pragma: no cover
    magnitude: float, limit: float
) -> float:
    # Scalar form of `ideal_switch_and_limit`.
    if magnitude < -limit:
        return -limit
    if magnitude > limit:
        return limit
    return magnitude


@njit(**_INLINED)
def _from_polar(magnitude: float, phase: float) -> complex:  # pragma: no cover
    # Rebuild a complex number from its magnitude and phase.
    return magnitude * (math.cos(phase) + 1j * math.sin(phase))


@njit(**_INLINED)
def _phase_of(value: complex) -> float:  # pragma: no cover
    # Phase [rad] of a complex number, as `np.angle` computes it.
    return math.atan2(value.imag, value.real)


@njit(**_INLINED)
def _cavity_response(  # pragma: no cover
    signals: NumpyArray,
    settings: Settings,
    coefficients: Coefficients,
    ind: int,
) -> None:
    # ACS cavity response model; mirrors `LHCCavityFeedback.cavity_response`.
    signals[V_ANT, ind] = (
        signals[I_GEN, ind - 1] * settings.r_over_q * settings.samples
        + signals[V_ANT, ind - 1] * coefficients.cavity_decay
        - signals[I_BEAM, ind - 1] * 0.5 * settings.r_over_q * settings.samples
    )


@njit(**_INLINED)
def _one_turn_feedback(  # pragma: no cover
    signals: NumpyArray,
    settings: Settings,
    coefficients: Coefficients,
    fir_coeff: NumpyArray,
    ind: int,
) -> None:
    # Effect of the OTFB on the analog branch; mirrors
    # `LHCCavityFeedback.one_turn_feedback`.
    signals[V_OTFB_COMB, ind] = (
        settings.alpha * signals[V_OTFB_COMB, ind - settings.n_coarse]
        + settings.gain_otfb
        * (1 - settings.alpha)
        * signals[V_OTFB_AC_IN, ind - settings.n_coarse + settings.n_otfb]
    )

    # FIR filter
    fir_sum = 0.0 + 0.0j
    for tap in range(len(fir_coeff)):
        fir_sum += fir_coeff[tap] * signals[V_OTFB_COMB, ind - tap]
    signals[V_OTFB_FIR_OUT, ind] = fir_sum

    # AC coupling at output
    signals[V_OTFB_OUT, ind] = (
        coefficients.ac_coupling * signals[V_OTFB_OUT, ind - 1]
        + signals[V_OTFB_FIR_OUT, ind]
        - signals[V_OTFB_FIR_OUT, ind - 1]
    )


@njit(**_INLINED)
def _rf_feedback(  # pragma: no cover
    signals: NumpyArray,
    settings: Settings,
    coefficients: Coefficients,
    fir_coeff: NumpyArray,
    ind: int,
) -> None:
    # Analog and digital RF feedback response; mirrors
    # `LHCCavityFeedback.rf_feedback`.

    # Voltage difference to act on
    signals[V_FEEDBACK_IN, ind] = (
        signals[V_SETPOINT, ind] - settings.open_loop * signals[V_ANT, ind]
    )

    # On the analog branch, the OTFB can contribute
    signals[V_OTFB_AC_IN, ind] = (
        coefficients.ac_coupling * signals[V_OTFB_AC_IN, ind - 1]
        + signals[V_FEEDBACK_IN, ind]
        - signals[V_FEEDBACK_IN, ind - 1]
    )
    _one_turn_feedback(signals, settings, coefficients, fir_coeff, ind)

    signals[V_ANALOG_IN, ind] = (
        signals[V_FEEDBACK_IN, ind]
        + settings.open_otfb * signals[V_OTFB_OUT, ind]
        + settings.excitation_otfb * signals[V_EXCITATION, ind]
    )

    # Output of the analog feedback (separate branch)
    signals[I_ANALOG_OUT, ind] = signals[
        I_ANALOG_OUT, ind - 1
    ] * coefficients.analog_decay + (
        settings.gain_analog
        * (signals[V_ANALOG_IN, ind] - signals[V_ANALOG_IN, ind - 1])
    )

    # Output of the digital feedback (separate branch)
    signals[I_DIGITAL_OUT, ind] = (
        signals[I_DIGITAL_OUT, ind - 1] * coefficients.digital_decay
        + coefficients.digital_gain * signals[V_FEEDBACK_IN, ind - 1]
    )

    # Total output: sum of analog and digital feedback
    signals[I_FEEDBACK_OUT, ind] = settings.open_rffb * (
        signals[I_ANALOG_OUT, ind] + signals[I_DIGITAL_OUT, ind]
    )


@njit(**_INLINED)
def _swap(  # pragma: no cover
    signals: NumpyArray, settings: Settings, ind: int
) -> None:
    # Switch-and-protect module; mirrors `LHCCavityFeedback.swap`.
    if settings.clamping:
        feedback_out = signals[I_FEEDBACK_OUT, ind]
        signals[I_SWAP_OUT, ind] = _from_polar(
            _clipped_magnitude(abs(feedback_out), settings.i_swap_threshold),
            _phase_of(feedback_out),
        )
    else:
        signals[I_SWAP_OUT, ind] = signals[I_FEEDBACK_OUT, ind]


@njit(**_INLINED)
def _generator_current(  # pragma: no cover
    signals: NumpyArray,
    settings: Settings,
    coefficients: Coefficients,
    klystron_fir: NumpyArray,
    ind: int,
) -> None:
    # Generator response; mirrors `LHCCavityFeedback.generator_current`.
    signals[I_GEN_TEST, ind] = (
        settings.gain_generator * signals[I_SWAP_OUT, ind]
    )
    signals[I_GEN_PREDRIVE, ind] = (
        settings.open_drive * signals[I_GEN_TEST, ind]
        + settings.open_drive_inv * settings.i_gen_offset
    )

    if settings.saturation:
        predrive = signals[I_GEN_PREDRIVE, ind]
        signals[I_GEN_PREDRIVE, ind] = _from_polar(
            _saturated_magnitude(
                abs(predrive),
                coefficients.saturation_onset,
                coefficients.saturation_span,
            ),
            _phase_of(predrive),
        )

    if settings.enable_klystron:
        # FIR filter modelling the klystron bandwidth
        klystron_sum = 0.0 + 0.0j
        for tap in range(len(klystron_fir)):
            klystron_sum += (
                klystron_fir[tap] * signals[I_GEN_PREDRIVE, ind - tap]
            )
        signals[I_GEN, ind] = klystron_sum
    else:
        signals[I_GEN, ind] = signals[I_GEN_PREDRIVE, ind - settings.n_delay]


@njit(**_INLINED)
def _tuner_input(signals: NumpyArray, ind: int) -> None:  # pragma: no cover
    # Data gathering for the detuning algorithm; mirrors
    # `LHCCavityFeedback.tuner_input`.
    signals[TUNER_IN, ind] = signals[I_GEN, ind] * (
        signals[V_ANT, ind].real - 1j * signals[V_ANT, ind].imag
    )

    # CIC component
    signals[TUNER_INTEGRATED, ind] = (
        (1 / 64)
        * (
            signals[TUNER_IN, ind]
            - 2 * signals[TUNER_IN, ind - 8]
            + signals[TUNER_IN, ind - 16]
        )
        + 2 * signals[TUNER_INTEGRATED, ind - 1]
        - signals[TUNER_INTEGRATED, ind - 2]
    )


@njit(cache=True, fastmath=False)
def track_one_turn_kernel(  # pragma: no cover
    signals: NumpyArray,
    settings: Settings,
    fir_coeff: NumpyArray,
    klystron_fir: NumpyArray,
) -> None:
    """
    Track the LHC cavity loop over one turn, sample by sample.

    ``signals`` is written in place: row ``V_ANT`` and friends each hold
    the previous turn followed by the current turn, so sample ``i`` of the
    current turn lives at column ``settings.n_coarse + i``.

    Parameters
    ----------
    signals
        Block of shape ``(N_SIGNALS, 2 * n_coarse)`` holding every signal
        of the loop, one per row, in the order of :data:`SIGNAL_NAMES`.
    settings
        Scalar settings of the loop for this turn.
    fir_coeff
        Coefficients of the OTFB FIR filter.
    klystron_fir
        Coefficients of the klystron bandwidth FIR filter.
    """
    coefficients = _coefficients(settings)

    for i in range(settings.n_coarse):
        ind = settings.n_coarse + i

        _cavity_response(signals, settings, coefficients, ind)
        _rf_feedback(signals, settings, coefficients, fir_coeff, ind)
        _swap(signals, settings, ind)
        _generator_current(signals, settings, coefficients, klystron_fir, ind)
        _tuner_input(signals, ind)
