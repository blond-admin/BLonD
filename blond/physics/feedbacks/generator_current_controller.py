# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Standalone generator-current controllers for a cavity feedback.

A controller is the pure signal-processing part of the feedback loop: it
maps a (complex, IQ) antenna-voltage error to a generator-current command,
independent of any cavity, profile or RF station. This makes it directly
testable with plain numbers and stubs, and lets a cavity feedback delegate
the error-to-current conversion instead of implementing it inline.

Two control laws implement the interface, and they share nothing but it:
:class:`GeneratorCurrentPIController` (proportional-integral, with a
choice of anti-windup, see :data:`ANTI_WINDUP_SCHEMES`) and
:class:`GeneratorCurrentPController` (proportional only). Each carries its
own tuning and state and names its own compiled closed-loop scan in
:mod:`~blond.physics.feedbacks.control_law_kernels`; the cavity model in
:mod:`~blond.physics.feedbacks.envelope_kernel` knows neither.

Both laws can also carry the klystron's bandwidth, as one real pole
between the clamped command and the generator current that drives the
cavity (``klystron_time_constant``, off by default; see
:func:`klystron_time_constant_from_bandwidth`).
"""

from __future__ import annotations

# Import the module, not the name: a bare ``deque`` in the module namespace is
# documented by automodule, and on Python 3.14 (the CI doc image) autodoc fails
# to format its C-level signature, which breaks the ``-W`` doc build.
import collections
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from numpy.typing import NDArray as NumpyArray

#: Anti-windup schemes of :class:`GeneratorCurrentPIController`, in the
#: order of the integer code its compiled scan takes.
#:
#: ``"conditional"`` (the default) freezes the whole complex integral on
#: every sample the klystron clamp fires. ``"directional"`` drops only the
#: part of the sample's integral increment that would push the command
#: further out and keeps the rest: the clamp limits the magnitude of the
#: command and leaves its phase free, so outward is the only direction the
#: integrator can wind up in.
ANTI_WINDUP_SCHEMES: tuple[str, ...] = ("conditional", "directional")


def current_limit_from_power(
    power: float, R_over_Q: float, Q_L: float
) -> float:
    r"""
    Convert a klystron forward-power limit to a generator-current limit.

    Uses the matched-generator relation
    :math:`I_\mathsf{max} = \sqrt{2 P / ((R/Q)\,Q_L)}`.

    Parameters
    ----------
    power
        Available klystron forward power per cavity [W].
    R_over_Q
        Geometric shunt impedance of the cavity [Ohm].
    Q_L
        Loaded quality factor of the cavity.

    Returns
    -------
    max_current
        Corresponding maximum generator-current magnitude [A].
    """
    return float(np.sqrt(2.0 * power / (R_over_Q * Q_L)))


def klystron_time_constant_from_bandwidth(
    bandwidth: float, attenuation_db: float = 1.0
) -> float:
    r"""
    Time constant of the one-pole klystron with a given bandwidth.

    A klystron's bandwidth is quoted as the full width about the carrier
    over which its gain stays within ``attenuation_db`` (1 dB, 3 dB, ...).
    On the complex envelope that is a low-pass, here one real pole with
    :math:`|H(\omega)|^2 = 1 / (1 + (\omega \tau)^2)`, which loses
    ``attenuation_db`` at half the bandwidth:

    .. math::
        \tau = \frac{\sqrt{10^{\,a / 10} - 1}}{\pi B}

    with :math:`a` the attenuation [dB] and :math:`B` the full bandwidth.

    Parameters
    ----------
    bandwidth
        Full bandwidth about the carrier [Hz], e.g. 5 MHz for +-2.5 MHz.
    attenuation_db
        Gain lost at the band edges [dB]. Default 1.

    Returns
    -------
    time_constant
        The pole's time constant [s].

    Raises
    ------
    ValueError
        If ``bandwidth`` or ``attenuation_db`` is not positive.
    """
    if not bandwidth > 0.0:
        raise ValueError(f"bandwidth={bandwidth} must be positive")
    if not attenuation_db > 0.0:
        raise ValueError(f"attenuation_db={attenuation_db} must be positive")
    return math.sqrt(10.0 ** (attenuation_db / 10.0) - 1.0) / (
        math.pi * bandwidth
    )


def _validated_klystron_time_constant(klystron_time_constant: float) -> float:
    """
    Refuse a negative (or NaN) klystron time constant.

    Parameters
    ----------
    klystron_time_constant
        Time constant handed to a controller [s].

    Returns
    -------
    klystron_time_constant
        The same value, as a float.

    Raises
    ------
    ValueError
        If it is not ``>= 0``.
    """
    if not klystron_time_constant >= 0.0:
        raise ValueError(
            f"klystron_time_constant={klystron_time_constant} must be >= 0 "
            "(0 is no pole)"
        )
    return float(klystron_time_constant)


def klystron_relax(
    previous_output: complex,
    command: complex,
    delta_t: float,
    time_constant: float,
) -> complex:
    """
    Advance the klystron's one-pole output by one coarse cell.

    The Python twin of
    :func:`~blond.physics.feedbacks.control_law_kernels.klystron_cell`:
    the output relaxes exactly towards the command held over the cell.

    Parameters
    ----------
    previous_output
        Klystron output at the end of the previous cell [A].
    command
        Command held over this cell [A].
    delta_t
        Length of this cell [s].
    time_constant
        The pole's time constant [s], positive.

    Returns
    -------
    output
        Klystron output at the end of this cell [A].
    """
    return command + (previous_output - command) * math.exp(
        -delta_t / time_constant
    )


def clamp_magnitude(
    value: complex | NumpyArray,
    max_magnitude: float | None,
) -> complex | NumpyArray:
    """
    Clamp the magnitude of a complex value or array, preserving its phase.

    This is a saturating clamp (it limits how large the output can get):
    entries whose magnitude exceeds ``max_magnitude`` are scaled down to
    it, while their phase (their direction in the IQ plane) is left alone.

    Parameters
    ----------
    value
        Complex value or array to clamp.
    max_magnitude
        Maximum allowed magnitude. If None, ``value`` is returned unchanged.

    Returns
    -------
    clamped
        ``value`` with each magnitude limited to ``max_magnitude`` and the
        phase left unchanged.
    """
    if max_magnitude is None:
        return value
    magnitude = np.abs(value)
    # The inner where() avoids a division by zero for zero entries (which are
    # below the limit and therefore left unchanged).
    scale = np.where(
        magnitude > max_magnitude,
        max_magnitude / np.where(magnitude == 0.0, 1.0, magnitude),
        1.0,
    )
    return value * scale


class GeneratorCurrentController(ABC):
    """
    Interface between a cavity feedback and its generator-current controller.

    A controller does two jobs. On each coarse-grid sample it turns the
    antenna-voltage error into a generator-current command
    (:meth:`update_generator_current`). It can also clamp a current to the
    actuator (klystron) limit on the fine grid (:meth:`limit`). It carries
    all of its own tuning and state, so the feedback holds only an instance
    of this interface and does not need to know the control law.

    Optionally a controller can also supply a *compiled* form of its law, so
    the feedback's coarse-grid recursion runs as a single compiled scan
    instead of a per-cell Python call. That is an opt-in capability: a
    controller advertises it with :attr:`supports_envelope_scan` and then owns
    the scan kernel (:meth:`envelope_scan_kernel`), the marshalling of its own
    tuning and state into the kernel's arguments
    (:meth:`envelope_scan_state`) and the write-back of the state the kernel
    returns (:meth:`absorb_envelope_scan_state`). Controllers that do not
    advertise it are driven cell-by-cell through
    :meth:`update_generator_current` instead, so implementing this interface
    never requires knowing anything about the compiled path.

    A controller may also model the klystron's bandwidth: a positive
    :attr:`klystron_time_constant` puts one real pole between the clamped
    command and the generator current the cavity is driven with, and the
    feedback then advances it on every coarse cell through
    :meth:`klystron_output`. The base class has none.
    """

    #: Whether this controller supplies a compiled scan (see class docstring).
    supports_envelope_scan: bool = False

    @property
    def klystron_time_constant(self) -> float:
        """
        Time constant of the klystron pole; the base class has none.

        Returns
        -------
        klystron_time_constant
            0, no pole [s].
        """
        return 0.0

    @abstractmethod
    def update_generator_current(
        self,
        error: complex,
        delta_t: float,
        generator_current_feedforward: complex = 0.0 + 0.0j,
    ) -> complex:
        """
        Map one antenna-voltage error sample to a generator current.

        Parameters
        ----------
        error
            Antenna-voltage error of this sample, ``V_set - V_ant`` [V].
        delta_t
            Time step of this sample [s].
        generator_current_feedforward
            Drive-side feedforward of this sample [A]: a precomputed
            addition to the bias, summed *before* any actuator limit so
            the limit acts on the total. The feedback passes it only when
            a drive table is attached, so a law that is never fed forward
            may leave the parameter out of its signature.

        Returns
        -------
        generator_current
            The generator-current command for this sample [A].
        """

    def limit(
        self, generator_current: complex | NumpyArray
    ) -> complex | NumpyArray:
        """
        Clamp a generator current to the actuator (klystron) limit.

        The base implementation applies no limit; controllers with a
        klystron current limit override this. It is used to enforce the
        limit on the fine grid, where the current is not produced by
        :meth:`update_generator_current`.

        Parameters
        ----------
        generator_current
            Generator current [A], scalar or array.

        Returns
        -------
        limited
            The input, limited to the actuator range.
        """
        return generator_current

    def klystron_output(
        self, previous_output: complex, delta_t: float
    ) -> complex:
        """
        Advance the klystron pole by one coarse cell.

        Only called when :attr:`klystron_time_constant` is positive: the
        output relaxes towards the last command this controller issued.

        Parameters
        ----------
        previous_output
            Klystron output at the end of the previous cell [A].
        delta_t
            Length of this cell [s].

        Returns
        -------
        output
            Klystron output at the end of this cell [A].

        Raises
        ------
        NotImplementedError
            When the controller models no klystron pole.
        """
        raise NotImplementedError(
            f"{type(self).__name__} models no klystron pole."
        )

    def envelope_scan_kernel(self) -> Callable:
        """
        Compiled kernel running this control law over a coarse-grid span.

        Only called when :attr:`supports_envelope_scan` is set. The kernel
        receives the cavity/grid arguments followed by whatever
        :meth:`envelope_scan_state` returned, and returns the state to hand
        back to :meth:`absorb_envelope_scan_state`.

        Returns
        -------
        kernel
            The compiled scan callable.

        Raises
        ------
        NotImplementedError
            When the controller advertises no compiled scan.
        """
        raise NotImplementedError(
            f"{type(self).__name__} supplies no compiled envelope scan; the "
            "feedback drives it through update_generator_current instead."
        )

    def envelope_scan_state(self) -> tuple:
        """
        Own tuning and live state, in the kernel's argument order.

        Only called when :attr:`supports_envelope_scan` is set. The feedback
        passes the result straight through to the kernel without inspecting
        it, so the layout is private to the controller.

        Returns
        -------
        state
            Positional arguments appended to the kernel's cavity arguments.

        Raises
        ------
        NotImplementedError
            When the controller advertises no compiled scan.
        """
        raise NotImplementedError(
            f"{type(self).__name__} supplies no compiled envelope scan; the "
            "feedback drives it through update_generator_current instead."
        )

    def absorb_envelope_scan_state(self, state: tuple) -> None:
        """
        Take back the live state the compiled scan advanced.

        Only called when :attr:`supports_envelope_scan` is set, and only for
        a span the feedback actually commits, so the controller resumes
        exactly where the scan left off.

        Parameters
        ----------
        state
            The state returned by :meth:`envelope_scan_kernel`'s callable.

        Raises
        ------
        NotImplementedError
            When the controller advertises no compiled scan.
        """
        raise NotImplementedError(
            f"{type(self).__name__} supplies no compiled envelope scan; the "
            "feedback drives it through update_generator_current instead."
        )


class GeneratorCurrentPIController(GeneratorCurrentController):
    r"""
    Saturating PI controller mapping a voltage error to a generator current.

    In plain terms: this controller reads how far the cavity's antenna
    voltage is from its target -- the error ``V_set - V_ant`` -- and
    commands a generator current that pushes that error towards zero. It
    adds a proportional term and an integral (running sum of past error)
    term, acts on the error from a few samples ago (a loop delay), and
    clamps its output so it never exceeds the klystron current limit.

    See the "Concepts and notation" section of
    :ref:`mucol_cavity_feedback_overview` for the vocabulary.

    Each :meth:`update_generator_current` converts one (complex)
    antenna-voltage error sample into the generator-current command

    .. math::
        I_\mathsf{gen} = \mathrm{clamp}\big(I_0
            + K_p\,e_\mathsf{d} + K_i \textstyle\sum e_\mathsf{d}\,\Delta t\big)

    where :math:`e_\mathsf{d}` is the error delayed by ``n_delay`` samples
    and :math:`I_0` is the generator-current bias. The clamp enforces the
    klystron current limit. By default the integrator uses conditional,
    anti-windup integration: it is frozen while the output is saturated
    (clamped at the limit), so a persistent error cannot keep inflating the
    stored integral. The clamp limits only the magnitude of the command,
    though, and leaves its phase free, so ``anti_windup="directional"``
    freezes only the outward part of each clamped sample's increment,
    :math:`e_\mathsf{d}\,\Delta t`, and keeps integrating the rest: with
    :math:`u` the unit vector of the command and :math:`r =
    \mathrm{Re}(e_\mathsf{d}\,\Delta t\,u^*)`, the increment loses
    :math:`r\,u` when :math:`K_i\,r > 0` and is kept whole otherwise.
    All state lives on the controller -- the delay line (the buffer of
    recent errors), the running integral and, with a klystron pole, the
    command the klystron is chasing -- so it can be driven and inspected
    in isolation.

    With ``klystron_time_constant`` positive the cavity is not driven by
    the command itself but by a klystron that follows it through one real
    pole: on every coarse cell (not only on controller samples) its output
    relaxes towards the last command as ``exp(-dt / tau)``. The pole sits
    after the clamp, so the output never leaves the limit circle, and the
    law does not see it: the anti-windup acts on the command, and the
    loop learns what the klystron did only through the cavity voltage.

    Parameters
    ----------
    gain_proportional
        Proportional gain :math:`K_p` [A/V].
    gain_integral
        Integral gain :math:`K_i` [A/(V s)].
    generator_current_bias
        Generator current bias :math:`I_0` [A] the PI correction is added
        on top of.
    n_delay
        Loop delay in samples; the error acted on is the one from ``n_delay``
        :meth:`update_generator_current` calls ago. Default 0. Note this
        counts coarse-grid *samples*, not time: driven by a sub-stepped
        feedback (``n_rf_periods_per_coarse_grid < 1``) the physical delay
        is ``n_delay * n_rf_periods_per_coarse_grid * t_rf``, i.e. it
        shrinks with the sub-step. Fixed at construction: the delay line is
        sized once from this value, so :attr:`n_delay` is read-only and the
        loop delay cannot be retuned afterwards. Build a new controller
        instead.
    max_output
        Maximum generator-current magnitude [A] (klystron limit). If None,
        the output is not limited and the integrator never saturates.
    anti_windup
        What the integrator does on a clamped sample, one of
        :data:`ANTI_WINDUP_SCHEMES`: ``"conditional"`` (default) freezes
        it, ``"directional"`` keeps all but the outward part of the
        increment. The two are the same law while the clamp is idle.
        Fixed at construction, like ``n_delay``.
    klystron_time_constant
        Time constant of the klystron pole [s] (see
        :func:`klystron_time_constant_from_bandwidth`); 0, the default, is
        no pole. Fixed at construction.

    Raises
    ------
    ValueError
        If ``anti_windup`` is not one of :data:`ANTI_WINDUP_SCHEMES`, or
        ``klystron_time_constant`` is negative.
    """

    def __init__(
        self,
        gain_proportional: float,
        gain_integral: float,
        generator_current_bias: complex,
        n_delay: int = 0,
        max_output: float | None = None,
        anti_windup: str = "conditional",
        klystron_time_constant: float = 0.0,
    ):
        assert n_delay >= 0, f"{n_delay=}, but must be >= 0."
        if anti_windup not in ANTI_WINDUP_SCHEMES:
            raise ValueError(
                f"anti_windup={anti_windup!r} is not one of "
                f"{ANTI_WINDUP_SCHEMES}"
            )
        self.gain_proportional = gain_proportional
        self.gain_integral = gain_integral
        self.generator_current_bias = generator_current_bias
        self._n_delay = int(n_delay)
        self.max_output = max_output
        self._anti_windup = anti_windup
        self._klystron_time_constant = _validated_klystron_time_constant(
            klystron_time_constant
        )
        # The command the klystron is chasing; it starts where the bias
        # holds the cavity, so a run starts without a transient.
        self._klystron_command: complex = complex(generator_current_bias)

        self._integral: complex = 0.0 + 0.0j
        # Zero-prefilled so the first n_delay updates act on a null error.
        #
        # Held as a circular buffer rather than a deque because the state
        # handoff to the compiled scan happens once per tracked span: a
        # deque has to be rebuilt element by element in Python on the way
        # back, which is O(n_delay) per span. That is invisible at the
        # 20-sample delay of a fast trim loop and dominates at the ~1300
        # samples a physically slow LLRF needs.
        #
        # Invariant: ``_delay_buffer[_delay_head]`` is the slot written
        # next and currently holds the oldest error -- exactly the
        # convention ``control_law_kernels.delay_line_push`` uses, so the two representations
        # need no translation.
        self._delay_buffer: NumpyArray = np.zeros(
            self._n_delay + 1, dtype=np.complex128
        )
        self._delay_head: int = 0

    @property
    def _delay_line(self) -> collections.deque[complex]:
        """
        The delay line as a deque, oldest error first.

        Returns
        -------
        delay_line
            The ``n_delay + 1`` most recent errors [V], oldest first,
            unrolled from the internal circular buffer.
        """
        return collections.deque(
            (
                complex(value)
                for value in np.roll(self._delay_buffer, -self._delay_head)
            ),
            maxlen=self._n_delay + 1,
        )

    #: The PI law has a compiled counterpart (see :meth:`envelope_scan_kernel`).
    supports_envelope_scan: bool = True

    @property
    def n_delay(self) -> int:
        """
        Loop delay in coarse-grid samples, fixed at construction.

        Read-only: the delay line is sized once, in ``__init__``, so a write
        here could never change the delay. Rejecting it makes an attempted
        retune fail loudly instead of being silently ignored; construct a
        new controller to change the loop delay.

        Returns
        -------
        n_delay
            The loop delay this controller was built with [samples].
        """
        return self._n_delay

    @property
    def anti_windup(self) -> str:
        """
        Anti-windup scheme, fixed at construction.

        Returns
        -------
        anti_windup
            One of :data:`ANTI_WINDUP_SCHEMES`.
        """
        return self._anti_windup

    @property
    def klystron_time_constant(self) -> float:
        """
        Time constant of the klystron pole, fixed at construction.

        Returns
        -------
        klystron_time_constant
            The pole's time constant [s]; 0 is no pole.
        """
        return self._klystron_time_constant

    def klystron_output(
        self, previous_output: complex, delta_t: float
    ) -> complex:
        """
        Advance the klystron pole by one coarse cell.

        Parameters
        ----------
        previous_output
            Klystron output at the end of the previous cell [A].
        delta_t
            Length of this cell [s].

        Returns
        -------
        output
            Klystron output at the end of this cell, relaxed towards the
            last command [A].
        """
        return klystron_relax(
            previous_output,
            self._klystron_command,
            delta_t,
            self._klystron_time_constant,
        )

    @property
    def integral(self) -> complex:
        """
        Committed error integral.

        Returns
        -------
        integral
            The error integral currently held by the controller [V s].
        """
        return self._integral

    def envelope_scan_kernel(self) -> Callable:
        """
        Compiled counterpart of this PI law.

        Returns
        -------
        kernel
            :func:`~blond.physics.feedbacks.control_law_kernels.envelope_pi_scan`,
            which reproduces :meth:`update_generator_current` byte-for-byte.
        """
        # Imported lazily: this keeps importing a controller free of numba,
        # which only the compiled path needs.
        from blond.physics.feedbacks.control_law_kernels import (
            envelope_pi_scan,
        )

        return envelope_pi_scan

    def envelope_scan_state(self) -> tuple:
        """
        Gains, bias, limit and live state, in the kernel's argument order.

        The delay line travels as the circular buffer it already is, so no
        reordering is needed in either direction.

        The buffer is **copied**, deliberately. The kernel advances the
        buffer it is handed in place, and the result is committed only by
        :meth:`absorb_envelope_scan_state`; handing out a copy keeps the
        live state untouched until then, so a scan that is not absorbed
        (an exception between the two calls) leaves the controller
        consistent.

        Returns
        -------
        state
            ``(gain_proportional, gain_integral, generator_current_bias,
            delay_buffer, delay_head, integral, max_output, anti_windup,
            klystron_time_constant, klystron_command)``, ``anti_windup`` as
            its index in :data:`ANTI_WINDUP_SCHEMES`.
        """
        return (
            float(self.gain_proportional),
            float(self.gain_integral),
            complex(self.generator_current_bias),
            self._delay_buffer.copy(),
            self._delay_head,
            complex(self._integral),
            np.inf if self.max_output is None else float(self.max_output),
            ANTI_WINDUP_SCHEMES.index(self._anti_windup),
            self._klystron_time_constant,
            complex(self._klystron_command),
        )

    def absorb_envelope_scan_state(self, state: tuple) -> None:
        """
        Restore the state the compiled scan advanced.

        Adopts the kernel's buffer and head as-is, so a following span or
        turn resumes exactly where the scan stopped. The buffer handed out
        by :meth:`envelope_scan_state` was already a private copy, so there
        is nothing to duplicate here and the handoff costs O(1) rather than
        rebuilding an ``n_delay``-long container per span.

        Parameters
        ----------
        state
            ``(delay_buffer, delay_head, integral, klystron_command)`` as
            returned by the kernel.
        """
        delay_buffer, delay_head, integral, klystron_command = state
        self._delay_buffer = delay_buffer
        self._delay_head = int(delay_head)
        self._integral = integral
        self._klystron_command = complex(klystron_command)

    def update_generator_current(
        self,
        error: complex,
        delta_t: float,
        generator_current_feedforward: complex = 0.0 + 0.0j,
    ) -> complex:
        """
        Advance the controller by one sample and return the current command.

        Parameters
        ----------
        error
            Antenna-voltage error of this sample, ``V_set - V_ant`` [V].
        delta_t
            Time step of this sample [s], used to integrate the error.
        generator_current_feedforward
            Drive-side feedforward of this sample [A], added to the bias.
            The clamp and the anti-windup act on the sum, exactly as in
            the compiled scan, which hands the law ``I_0 + I_ff`` as its
            bias.

        Returns
        -------
        generator_current
            The (clamped) generator-current command for this sample [A].
            While clamped, the integral takes what :attr:`anti_windup`
            keeps of this sample's increment.
        """
        # Write at the head, advance, then read the new head: the slot that
        # falls under it is the error from n_delay updates ago. Identical to
        # deque ``append`` followed by ``[0]``, and to the compiled scan.
        self._delay_buffer[self._delay_head] = error
        self._delay_head = (self._delay_head + 1) % self._delay_buffer.size
        delayed_error = complex(self._delay_buffer[self._delay_head])

        increment = delayed_error * delta_t
        candidate_integral = self._integral + increment
        # Bias and feedforward are summed first, as in the compiled scan,
        # so the two stay byte-identical.
        output = (
            (self.generator_current_bias + generator_current_feedforward)
            + self.gain_proportional * delayed_error
            + self.gain_integral * candidate_integral
        )

        # Anti-windup: commit the whole increment only while the output is
        # not saturated by the klystron current limit.
        saturated = (
            self.max_output is not None and np.abs(output) > self.max_output
        )
        if not saturated:
            self._integral = candidate_integral
        elif self._anti_windup == "directional":
            # The clamp leaves the phase free: drop only the part of the
            # increment that would push the command further out.
            unit = output / np.abs(output)
            radial = (increment * np.conj(unit)).real
            if self.gain_integral * radial > 0.0:
                increment = increment - radial * unit
            self._integral = self._integral + increment

        command = clamp_magnitude(output, self.max_output)
        self._klystron_command = command
        return command

    def limit(
        self, generator_current: complex | NumpyArray
    ) -> complex | NumpyArray:
        """
        Clamp a generator current to this controller's klystron limit.

        Parameters
        ----------
        generator_current
            Generator current [A], scalar or array.

        Returns
        -------
        limited
            The input with ``|I_gen| <= max_output`` (unchanged if no limit).
        """
        return clamp_magnitude(generator_current, self.max_output)


class GeneratorCurrentPController(GeneratorCurrentController):
    r"""
    Saturating proportional controller mapping a voltage error to a current.

    The second control law beside :class:`GeneratorCurrentPIController`,
    and deliberately not derived from it: it has no integral, so no
    anti-windup and no running state beyond the loop's delay line. Each
    :meth:`update_generator_current` returns

    .. math::
        I_\mathsf{gen} = \mathrm{clamp}\big(I_0 + K_p\,e_\mathsf{d}\big)

    with :math:`e_\mathsf{d}` the error delayed by ``n_delay`` samples,
    :math:`I_0` the generator-current bias and the clamp the klystron
    current limit.

    Why a proportional law suits a cavity loop: over the loop's own
    timescale the cavity integrates the generator current (its
    half-bandwidth time is hundreds of loop delays), so the loop is
    already of type one and tracks a constant setpoint without an
    integrator in the controller. What it gives up is the rejection of a
    constant *drive* disturbance -- a bias error, a detuning mismatch --
    which leaves a static error of the disturbance's open-loop response
    divided by ``1 + K_p Z``, ``Z`` the cavity's static impedance. What it
    gains is about 15 % more gain for the same stability margin (the
    boundary of ``gain * delay`` is ``pi / 2`` against 1.37 for a PI with
    the integral time at four delays), and no integrator to wind up while
    the klystron is saturated.

    Parameters
    ----------
    gain_proportional
        Proportional gain :math:`K_p` [A/V].
    generator_current_bias
        Generator current bias :math:`I_0` [A] the correction is added to.
    n_delay
        Loop delay in samples, fixed at construction; the error acted on
        is the one from ``n_delay`` updates ago. Default 0.
    max_output
        Maximum generator-current magnitude [A] (klystron limit). If None,
        the output is not limited.
    klystron_time_constant
        Time constant of the klystron pole [s], after the clamp, as for
        :class:`GeneratorCurrentPIController`; 0, the default, is no pole.

    Raises
    ------
    ValueError
        If ``n_delay`` or ``klystron_time_constant`` is negative.
    """

    #: The P law has a compiled counterpart (see :meth:`envelope_scan_kernel`).
    supports_envelope_scan: bool = True

    def __init__(
        self,
        gain_proportional: float,
        generator_current_bias: complex,
        n_delay: int = 0,
        max_output: float | None = None,
        klystron_time_constant: float = 0.0,
    ):
        if n_delay < 0:
            raise ValueError(f"n_delay={n_delay} must be >= 0")
        self.gain_proportional = gain_proportional
        self.generator_current_bias = generator_current_bias
        self._n_delay = int(n_delay)
        self.max_output = max_output
        self._klystron_time_constant = _validated_klystron_time_constant(
            klystron_time_constant
        )
        self._klystron_command: complex = complex(generator_current_bias)
        # Circular buffer: ``_delay_buffer[_delay_head]`` is the slot written
        # next and holds the oldest error -- the convention of
        # ``control_law_kernels.delay_line_push``, so the compiled scan
        # takes and returns it without translation.
        self._delay_buffer: NumpyArray = np.zeros(
            self._n_delay + 1, dtype=np.complex128
        )
        self._delay_head: int = 0

    @property
    def n_delay(self) -> int:
        """
        Loop delay in controller samples, fixed at construction.

        Returns
        -------
        n_delay
            The loop delay this controller was built with [samples].
        """
        return self._n_delay

    @property
    def klystron_time_constant(self) -> float:
        """
        Time constant of the klystron pole, fixed at construction.

        Returns
        -------
        klystron_time_constant
            The pole's time constant [s]; 0 is no pole.
        """
        return self._klystron_time_constant

    def klystron_output(
        self, previous_output: complex, delta_t: float
    ) -> complex:
        """
        Advance the klystron pole by one coarse cell.

        Parameters
        ----------
        previous_output
            Klystron output at the end of the previous cell [A].
        delta_t
            Length of this cell [s].

        Returns
        -------
        output
            Klystron output at the end of this cell, relaxed towards the
            last command [A].
        """
        return klystron_relax(
            previous_output,
            self._klystron_command,
            delta_t,
            self._klystron_time_constant,
        )

    @property
    def _delay_line(self) -> collections.deque[complex]:
        """
        The delay line as a deque, oldest error first.

        Returns
        -------
        delay_line
            The ``n_delay + 1`` most recent errors [V], oldest first.
        """
        return collections.deque(
            (
                complex(value)
                for value in np.roll(self._delay_buffer, -self._delay_head)
            ),
            maxlen=self._n_delay + 1,
        )

    def update_generator_current(
        self,
        error: complex,
        delta_t: float,
        generator_current_feedforward: complex = 0.0 + 0.0j,
    ) -> complex:
        """
        Advance the controller by one sample and return the current command.

        Parameters
        ----------
        error
            Antenna-voltage error of this sample, ``V_set - V_ant`` [V].
        delta_t
            Time step of this sample [s]; unused, since a proportional law
            has no time constant of its own.
        generator_current_feedforward
            Drive-side feedforward of this sample [A], added to the bias
            and clamped with it.

        Returns
        -------
        generator_current
            The (clamped) generator-current command for this sample [A].
        """
        self._delay_buffer[self._delay_head] = error
        self._delay_head = (self._delay_head + 1) % self._delay_buffer.size
        delayed_error = complex(self._delay_buffer[self._delay_head])
        output = (
            self.generator_current_bias + generator_current_feedforward
        ) + self.gain_proportional * delayed_error
        command = clamp_magnitude(output, self.max_output)
        self._klystron_command = command
        return command

    def limit(
        self, generator_current: complex | NumpyArray
    ) -> complex | NumpyArray:
        """
        Clamp a generator current to this controller's klystron limit.

        Parameters
        ----------
        generator_current
            Generator current [A], scalar or array.

        Returns
        -------
        limited
            The input with ``|I_gen| <= max_output`` (unchanged if no limit).
        """
        return clamp_magnitude(generator_current, self.max_output)

    def envelope_scan_kernel(self) -> Callable:
        """
        Compiled counterpart of this proportional law.

        Returns
        -------
        kernel
            :func:`~blond.physics.feedbacks.control_law_kernels.envelope_p_scan`.
        """
        from blond.physics.feedbacks.control_law_kernels import (
            envelope_p_scan,
        )

        return envelope_p_scan

    def envelope_scan_state(self) -> tuple:
        """
        Gain, bias, delay line, limit and pole, in the kernel's order.

        The buffer is copied: the kernel advances it in place, and only
        :meth:`absorb_envelope_scan_state` commits the result.

        Returns
        -------
        state
            ``(gain_proportional, generator_current_bias, delay_buffer,
            delay_head, max_output, klystron_time_constant,
            klystron_command)``.
        """
        return (
            float(self.gain_proportional),
            complex(self.generator_current_bias),
            self._delay_buffer.copy(),
            self._delay_head,
            np.inf if self.max_output is None else float(self.max_output),
            self._klystron_time_constant,
            complex(self._klystron_command),
        )

    def absorb_envelope_scan_state(self, state: tuple) -> None:
        """
        Take back the state the compiled scan advanced.

        Parameters
        ----------
        state
            ``(delay_buffer, delay_head, klystron_command)`` as returned
            by the kernel.
        """
        delay_buffer, delay_head, klystron_command = state
        self._delay_buffer = delay_buffer
        self._delay_head = int(delay_head)
        self._klystron_command = complex(klystron_command)
