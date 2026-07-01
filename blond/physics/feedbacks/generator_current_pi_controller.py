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

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


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


class GeneratorCurrentPIController:
    r"""
    Saturating PI controller mapping a voltage error to a generator current.

    Each :meth:`update` converts one (complex) antenna-voltage error sample
    into the generator-current command

    .. math::
        I_\mathsf{gen} = \mathrm{clamp}\big(I_\mathsf{ff}
            + K_p\,e_\mathsf{d} + K_i \textstyle\sum e_\mathsf{d}\,\Delta t\big)

    where :math:`e_\mathsf{d}` is the error delayed by ``n_delay`` samples,
    :math:`I_\mathsf{ff}` the feedforward current and the clamp enforces the
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
    feedforward
        Constant feedforward generator current :math:`I_\mathsf{ff}` [A] the
        PI correction is added on top of.
    n_delay
        Loop delay in samples; the error acted on is the one from ``n_delay``
        :meth:`update` calls ago. Default 0.
    max_output
        Maximum generator-current magnitude [A] (klystron limit). If None,
        the output is not limited and the integrator never saturates.

    Attributes
    ----------
    integral
        Current value of the (committed) error integral [V s].
    """

    def __init__(
        self,
        gain_proportional: float,
        gain_integral: float,
        feedforward: complex,
        n_delay: int = 0,
        max_output: float | None = None,
    ):
        assert n_delay >= 0, f"{n_delay=}, but must be >= 0."
        self.gain_proportional = gain_proportional
        self.gain_integral = gain_integral
        self.feedforward = feedforward
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

    def update(self, error: complex, delta_t: float) -> complex:
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
            self.feedforward
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
