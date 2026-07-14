# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Generator-current regulation for the cavity-feedback timing class.

:class:`GeneratorRegulationMixin` groups the PI-controller-facing pieces of
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`:
the controller-active check, the per-cavity IQ setpoint, the klystron power,
and the per-coarse-step generator-current update. A *mixin* -- the methods read
host state (``_controller``, ``_voltage_setpoint``, ``n_cavities``,
``R_over_Q``, ``Q_L``, ``generator_current_coarse_grid``,
``antenna_voltage_coarse_grid``, ``_omega_input_for_pi``, ...). Extracted
verbatim from ``cavity_feedback.py``; behaviour unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.physics.feedbacks.iq import polar_to_cartesian

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class GeneratorRegulationMixin:
    """Generator-current regulation mixin (see module docstring)."""

    @property
    def _controller_active(self) -> bool:
        """
        Whether a generator-current controller is attached.

        Returns
        -------
        controller_active
            True when a controller regulates the generator current;
            otherwise it stays at the constant value and the controller
            update is skipped.
        """
        return self._controller is not None

    @property
    def pi_setpoint(self) -> complex:
        """
        Per-cavity voltage setpoint of the PI controller in the IQ frame.

        Returns
        -------
        pi_setpoint
            The explicit setpoint given at construction, or the voltage of
            the parent rf station divided by the number of cavities.
        """
        if self._voltage_setpoint is not None:
            return self._voltage_setpoint
        return polar_to_cartesian(
            self.get_voltage_from_parent_rf_station() / self.n_cavities,
            0,
        )

    def generator_power(
        self, generator_current: complex | NumpyArray | None = None
    ) -> float | NumpyArray:
        r"""
        Klystron forward power per cavity from the generator current.

        .. math::
            P = 0.5\,(R/Q)\,Q_L\,|I_\mathsf{gen}|^2

        Parameters
        ----------
        generator_current
            Generator current [A] to convert; defaults to the coarse-grid
            generator current of the current turn.

        Returns
        -------
        generator_power
            Generator forward power [W], same shape as the input.
        """
        if generator_current is None:
            generator_current = self.generator_current_coarse_grid
        return 0.5 * self.R_over_Q * self.Q_L * np.abs(generator_current) ** 2

    def _update_generator_current(
        self,
        omega_times_T_s: float,
        coarse_grid_index_to_update: int,
    ) -> None:
        """
        Update the coarse-grid generator current from the voltage error.

        Forms the antenna-voltage error and the per-step time and hands them
        to the attached controller, which returns the generator current
        written to the coarse grid.

        Parameters
        ----------
        omega_times_T_s
            Angular frequency times sampling time of this step.
        coarse_grid_index_to_update
            Coarse grid index whose generator current is written.
        """
        if self._omega_input_for_pi is None:
            raise RuntimeError(
                "cavity_response() was called before circuit_track(); the"
                " controller needs omega_input to recover the sampling time."
            )
        idx = coarse_grid_index_to_update
        error = self.pi_setpoint - self.antenna_voltage_coarse_grid[idx]
        delta_t = omega_times_T_s / self._omega_input_for_pi
        self.generator_current_coarse_grid[idx] = (
            self._controller.update_generator_current(error, delta_t)
        )
