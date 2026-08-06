# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Standalone PI controller for the generator current of a cavity feedback.

The controller is the pure signal-processing part of the feedback loop: it
maps a (complex, IQ) antenna-voltage error to a generator-current command,
independent of any cavity, profile or RF station. This makes it directly
testable with plain numbers and stubs, and lets a cavity feedback delegate
the error-to-current conversion instead of implementing it inline.
"""

from __future__ import annotations

# Import the module, not the name: a bare ``deque`` in the module namespace is
# documented by automodule, and on Python 3.14 (the CI doc image) autodoc fails
# to format its C-level signature, which breaks the ``-W`` doc build.
import collections
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from numpy.typing import NDArray as NumpyArray


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
    """

    #: Whether this controller supplies a compiled scan (see class docstring).
    supports_envelope_scan: bool = False

    @abstractmethod
    def update_generator_current(
        self, error: complex, delta_t: float
    ) -> complex:
        """
        Map one antenna-voltage error sample to a generator current.

        Parameters
        ----------
        error
            Antenna-voltage error of this sample, ``V_set - V_ant`` [V].
        delta_t
            Time step of this sample [s].

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
    klystron current limit. The integrator uses conditional, anti-windup
    integration: it is frozen while the output is saturated (clamped at the
    limit), so a persistent error cannot keep inflating the stored integral.
    All state lives on the controller -- the delay line (the buffer of
    recent errors) and the running integral -- so it can be driven and
    inspected in isolation.

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
        shrinks with the sub-step.
    max_output
        Maximum generator-current magnitude [A] (klystron limit). If None,
        the output is not limited and the integrator never saturates.
    """

    def __init__(
        self,
        gain_proportional: float,
        gain_integral: float,
        generator_current_bias: complex,
        n_delay: int = 0,
        max_output: float | None = None,
    ):
        assert n_delay >= 0, f"{n_delay=}, but must be >= 0."
        self.gain_proportional = gain_proportional
        self.gain_integral = gain_integral
        self.generator_current_bias = generator_current_bias
        self.n_delay = int(n_delay)
        self.max_output = max_output

        self._integral: complex = 0.0 + 0.0j
        # Zero-prefilled so the first n_delay updates act on a null error.
        self._delay_line: collections.deque[complex] = collections.deque(
            [0.0 + 0.0j] * (self.n_delay + 1), maxlen=self.n_delay + 1
        )

    #: The PI law has a compiled counterpart (see :meth:`envelope_scan_kernel`).
    supports_envelope_scan: bool = True

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
            :func:`~blond.physics.feedbacks.envelope_kernel.envelope_pi_scan`,
            which reproduces :meth:`update_generator_current` byte-for-byte.
        """
        # Imported lazily: this keeps importing a controller free of numba,
        # which only the compiled path needs.
        from blond.physics.feedbacks.envelope_kernel import envelope_pi_scan

        return envelope_pi_scan

    def envelope_scan_state(self) -> tuple:
        """
        Gains, bias, limit and live state, in the kernel's argument order.

        The delay line travels as a plain array; the kernel treats it as a
        circular buffer whose head starts at zero, which reproduces
        :class:`~collections.deque` ``append``/``[0]`` semantics exactly.

        Returns
        -------
        state
            ``(gain_proportional, gain_integral, generator_current_bias,
            delay_buffer, delay_head, integral, max_output)``.
        """
        return (
            float(self.gain_proportional),
            float(self.gain_integral),
            complex(self.generator_current_bias),
            np.array(self._delay_line, dtype=np.complex128),
            0,
            complex(self._integral),
            np.inf if self.max_output is None else float(self.max_output),
        )

    def absorb_envelope_scan_state(self, state: tuple) -> None:
        """
        Restore the delay line and integral the compiled scan advanced.

        Rebuilds the deque from the circular buffer oldest-first starting at
        the returned head, so a following span or turn resumes exactly where
        the scan stopped.

        Parameters
        ----------
        state
            ``(delay_buffer, delay_head, integral)`` as returned by the
            kernel.
        """
        delay_buffer, delay_head, integral = state
        buffer_len = delay_buffer.shape[0]
        self._delay_line = collections.deque(
            (
                delay_buffer[(delay_head + offset) % buffer_len]
                for offset in range(buffer_len)
            ),
            maxlen=buffer_len,
        )
        self._integral = integral

    def update_generator_current(
        self, error: complex, delta_t: float
    ) -> complex:
        """
        Advance the controller by one sample and return the current command.

        Parameters
        ----------
        error
            Antenna-voltage error of this sample, ``V_set - V_ant`` [V].
        delta_t
            Time step of this sample [s], used to integrate the error.

        Returns
        -------
        generator_current
            The (clamped) generator-current command for this sample [A].
        """
        self._delay_line.append(error)
        delayed_error = self._delay_line[0]

        candidate_integral = self._integral + delayed_error * delta_t
        output = (
            self.generator_current_bias
            + self.gain_proportional * delayed_error
            + self.gain_integral * candidate_integral
        )

        # Conditional anti-windup: only commit the integral while the output
        # is not saturated by the klystron current limit.
        saturated = (
            self.max_output is not None and np.abs(output) > self.max_output
        )
        if not saturated:
            self._integral = candidate_integral

        return clamp_magnitude(output, self.max_output)

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
