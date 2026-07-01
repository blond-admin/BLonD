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

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
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

    A controller converts, at each coarse-grid sample, the antenna-voltage
    error into a generator-current command (:meth:`update_generator_current`)
    and can clamp a current to the actuator limit on the fine grid
    (:meth:`limit`). It carries
    all of its own tuning and state, so the feedback holds only an instance of
    this interface and does not know the control law.
    """

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


class GeneratorCurrentPIController(GeneratorCurrentController):
    r"""
    Saturating PI controller mapping a voltage error to a generator current.

    Each :meth:`update_generator_current` converts one (complex)
    antenna-voltage error sample into the generator-current command

    .. math::
        I_\mathsf{gen} = \mathrm{clamp}\big(I_0
            + K_p\,e_\mathsf{d} + K_i \textstyle\sum e_\mathsf{d}\,\Delta t\big)

    where :math:`e_\mathsf{d}` is the error delayed by ``n_delay`` samples,
    :math:`I_0` the generator current bias and the clamp enforces the
    klystron current limit. The integrator uses conditional (anti-windup)
    integration: it is frozen while the output is saturated. All state (the
    delay line and the integral) lives on the controller, so it can be driven
    and inspected in isolation.

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
        :meth:`update_generator_current` calls ago. Default 0.
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
        self._delay_line: deque[complex] = deque(
            [0.0 + 0.0j] * (self.n_delay + 1), maxlen=self.n_delay + 1
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
