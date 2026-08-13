# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Generator-current regulation for the cavity-feedback timing class.

:class:`GeneratorRegulationMixin` holds the parts of
:class:`~blond.physics.feedbacks.cavity_feedback.IQCavityFeedbackTimingClass`
that need nothing but the attached controller and the voltage setpoint:

- the setpoint policy -- ``_validate_voltage_setpoint`` (the constructor's
  real-and-positive rule) and ``pi_setpoint`` (the per-cavity setpoint that
  rule guards, explicit or derived from the parent rf station);
- ``_controller_active``, whether a controller is attached at all;
- ``generator_power``, the klystron forward power readout;
- ``_update_generator_current``, the error-to-current step the reference
  per-cell path takes;
- ``_limit_fine_grid_generator_current``, the actuator (klystron) clamp
  applied to the fine grid before the fine solve.

It is a *mixin*: those methods read and write host state
(``_controller``, ``_voltage_setpoint``, ``n_cavities``, ``R_over_Q``,
``Q_L``, ``generator_current_coarse_grid``, ``antenna_voltage_coarse_grid``,
``generator_current_fine_grid``, ``_omega_input_for_pi``, ...) that
``IQCavityFeedbackTimingClass`` owns.

**What this module does NOT own.** The file name promises the whole
generator regulation; it delivers only the above. Most of the
controller-facing code stays on the timing class in ``cavity_feedback.py``,
and a reader chasing "where does the controller get its state from" has to
look there:

- **The compiled envelope scan** (``_circuit_track_cells_kernel``) marshals
  the controller's own compiled kernel, tuning and carried state into one
  call and writes the returned state back through
  ``absorb_envelope_scan_state``. It is not moved because it reads both
  coarse grids and all three values carried across the turn boundary
  (``_last_val_ant_voltage``, ``_last_val_generator_current``,
  ``_last_val_beam_current``), and because it depends on ``pi_setpoint``
  staying **unevaluated** on a span the controller sits out -- that property
  may reach through to the parent rf station, which a no-beam backfill span
  must not require. Moving it would relocate that coupling, not remove it.
- **The per-cell stepping decision** -- ``_circuit_track_cells`` choosing the
  compiled or the reference path, and ``cavity_response`` gating
  ``_update_generator_current`` on "forward cells only, never a no-beam
  backfill segment". That gate is a statement about the segment structure of
  the coarse grid, which is the timing class's business.

So: this module owns the control *law's* interface to the feedback (setpoint,
error step, limits, power); the timing class owns *when and over which cells*
the controller runs, and the state it runs on.
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

    def _validate_voltage_setpoint(
        self, voltage_setpoint: complex | None
    ) -> None:
        """
        Reject an explicit voltage setpoint that is not real and positive.

        The constructor-time half of the setpoint policy whose runtime half
        is ``pi_setpoint``: it constrains the value ``pi_setpoint`` returns
        when an explicit setpoint was given. ``None`` (the parent-derived
        setpoint) is always accepted.

        Parameters
        ----------
        voltage_setpoint
            Per-cavity voltage setpoint in the IQ frame [V] as handed to the
            constructor, or None for the parent-derived one.

        Raises
        ------
        ValueError
            If ``voltage_setpoint`` has a non-zero imaginary part or a
            non-positive real part, i.e. is not at phase 0.
        """
        # The station's phase correction is formed against the parent-derived
        # station_voltage_coarse_grid, whose phase is 0 by construction; an
        # explicit setpoint with a non-zero phase would make the PI regulate
        # to a frame the phase correction does not use. Until that frame is
        # unified, only real, positive setpoints are supported.
        if voltage_setpoint is not None and (
            np.imag(voltage_setpoint) != 0.0 or np.real(voltage_setpoint) <= 0
        ):
            raise ValueError(
                f"voltage_setpoint={voltage_setpoint} must be real and "
                "positive (phase 0): the RF station's phase correction is "
                "referenced to the parent-derived setpoint at phase 0, so a "
                "rotated explicit setpoint would be regulated by the "
                "controller but not reflected in the applied kick. Rotate "
                "phi_rf on the station instead."
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
        omega_times_dt: float,
        coarse_grid_index_to_update: int,
    ) -> None:
        """
        Update the coarse-grid generator current from the voltage error.

        Forms the antenna-voltage error and the per-step time and hands them
        to the attached controller, which returns the generator current
        written to the coarse grid.

        The error is formed in the KICK frame: the demodulation-frame sum
        of this cell rotated by ``_kick_frame_rotation`` (the per-passage
        scalar ``exp(+i * carrier slip gap)``), i.e. the envelope of the
        kick the station actually applies against ``phi_rf`` -- so the
        loop regulates the applied voltage, not a bookkeeping frame. The
        rotation is exactly unity without an RF-frequency offset and
        without multi-section acceleration.

        Parameters
        ----------
        omega_times_dt
            RF phase advanced in this step [rad], i.e. ``omega * dt``.
        coarse_grid_index_to_update
            Coarse grid index whose generator current is written.
        """
        if self._omega_input_for_pi is None:
            raise RuntimeError(
                "cavity_response() was called before circuit_track(); the"
                " controller needs omega_input to recover the sampling time."
            )
        idx = coarse_grid_index_to_update
        error = self.pi_setpoint - (
            self.antenna_voltage_coarse_grid[idx] * self._kick_frame_rotation
        )
        delta_t = omega_times_dt / self._omega_input_for_pi
        self.generator_current_coarse_grid[idx] = (
            self._controller.update_generator_current(error, delta_t)
        )

    def _limit_fine_grid_generator_current(
        self, initial_generator_current_fine_grid: complex
    ) -> complex:
        """
        Clamp the fine-grid generator current to the actuator limit.

        Rewrites the host's ``generator_current_fine_grid`` with the
        controller's klystron-limited version and returns the scalar initial
        condition limited the same way; a no-op without a controller. Must
        run *before* the fine-grid solve, so the response matrix never sees a
        current above the limit.

        Parameters
        ----------
        initial_generator_current_fine_grid
            Initial condition of the generator current on the fine grid [A].

        Returns
        -------
        initial_generator_current_fine_grid
            The initial condition, limited to the actuator range.
        """
        # Enforce the controller's actuator (klystron) limit on the fine grid
        # too, so the response matrix never sees a current above the limit.
        # The coarse values are already clamped; this guards the interpolated
        # initial condition and any externally set fine-grid current.
        if self._controller is not None:
            self.generator_current_fine_grid = self._controller.limit(
                self.generator_current_fine_grid
            )
            initial_generator_current_fine_grid = self._controller.limit(
                initial_generator_current_fine_grid
            )
        return initial_generator_current_fine_grid
