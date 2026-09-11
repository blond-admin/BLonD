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
  per-cell path takes.

The klystron limit is enforced in exactly one place: by the controller, on
every coarse command it returns. The fine-grid generator current is a linear
interpolation of those commands, and a straight line between two points
inside the limit circle stays inside it, so no second clamp is needed there.

It is a *mixin*: those methods read and write host state
(``_controller``, ``_voltage_setpoint``, ``n_cavities``, ``R_over_Q``,
``Q_L``, ``generator_current_coarse_grid``, ``antenna_voltage_coarse_grid``,
``_omega_input_for_pi``, ...) that
``IQCavityFeedbackTimingClass`` owns. Every method therefore annotates its
``self`` as that host, exactly as ``rf_center_grid.py`` does: the dependency
exists either way, and stating it in the signature is what lets a reader and
a type checker resolve those attributes instead of reconstructing the
requirement from the accesses. The import stays inside ``TYPE_CHECKING`` --
the host inherits from this mixin, so a runtime import would be a cycle.

**What this module does NOT own.** The file name promises the whole
generator regulation; it delivers only the above. Most of the
controller-facing code stays on the timing class in ``cavity_feedback.py``,
and a reader chasing "where does the controller get its state from" has to
look there:

- **The compiled envelope scan** (``_circuit_track_cells_kernel``) marshals
  the controller's own compiled kernel, tuning and carried state into one
  call and writes the returned state back through
  ``absorb_envelope_scan_state``. It is not moved because it reads every
  coarse grid (the summed, generator- and beam-sourced antenna voltages
  and the generator current) and all five values carried across the turn
  boundary (``_last_val_ant_voltage``, ``_last_val_ant_voltage_gen``,
  ``_last_val_ant_voltage_beam``, ``_last_val_generator_current``,
  ``_last_val_beam_current``), and because it depends on ``pi_setpoint``
  staying **unevaluated** on a span with no controller attached -- that
  property may reach through to the parent rf station. Moving it would
  relocate that coupling, not remove it.
- **The per-cell stepping decision** -- ``_circuit_track_cells`` choosing the
  compiled or the reference path, and ``cavity_response`` stepping
  ``_update_generator_current`` on every tracked cell, the ``no_beam``
  backfill reconstruction segments included (a real LLRF regulates
  continuously; a loop confined to the forward passage is open-loop for
  ``(N - 1) / N`` of each turn). Which cells exist, and in which frame, is
  a statement about the segment structure of the coarse grid, which is the
  timing class's business.

So: this module owns the control *law's* interface to the feedback (setpoint,
error step, power); the timing class owns *when and over which cells*
the controller runs, and the state it runs on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.physics.feedbacks.iq import polar_to_cartesian

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

    from blond.physics.feedbacks.cavity_feedback import (
        IQCavityFeedbackTimingClass,
    )


class GeneratorRegulationMixin:
    """Generator-current regulation mixin (see module docstring)."""

    @property
    def _controller_active(self: IQCavityFeedbackTimingClass) -> bool:
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
    def pi_setpoint(self: IQCavityFeedbackTimingClass) -> complex:
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
        self: IQCavityFeedbackTimingClass,
        voltage_setpoint: complex | None,
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
        self: IQCavityFeedbackTimingClass,
        generator_current: complex | NumpyArray | None = None,
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

    def reflected_current(
        self: IQCavityFeedbackTimingClass,
        generator_current: complex | NumpyArray | None = None,
        antenna_voltage: complex | NumpyArray | None = None,
        generator_frame_rotation: complex | None = None,
    ) -> complex | NumpyArray:
        r"""
        Current reflected back out of the coupler, per cavity.

        .. math::
            I_\mathsf{refl} = \frac{V_\mathsf{ant}}{(R/Q)\,Q_L}
                              - r_\mathsf{gen}\,I_\mathsf{gen}

        Parameters
        ----------
        generator_current
            Generator current [A] in the design frame it is stored in on
            the grids; defaults to the coarse-grid generator current of
            the current turn.
        antenna_voltage
            Antenna voltage [V] **per cavity**, as composed on the grids;
            defaults to the coarse-grid antenna voltage of the current
            turn.
        generator_frame_rotation
            The rotation :math:`r_\mathsf{gen}` that takes the design-frame
            generator current into the frame of ``antenna_voltage``;
            defaults to the feedback's current generator frame rotation,
            which is the one the grids of this passage were composed
            with.  Pass ``1`` for arrays that are already in one frame.

        Returns
        -------
        reflected_current
            Reflected current [A], same shape as the inputs.

        Notes
        -----
        The cavity solver drives the envelope with ``2 I_gen - I_beam``,
        so the forward wave in that convention is
        :math:`V_\mathsf{for} = (R/Q) Q_L I_\mathsf{gen}` and the
        reflected wave is the usual superposition
        :math:`V_\mathsf{refl} = V_\mathsf{ant} - V_\mathsf{for}`, which
        is what the expression above states divided through by
        :math:`(R/Q) Q_L`. It is fixed by the two limits that leave no
        freedom, and it satisfies
        :math:`P_\mathsf{for} - P_\mathsf{refl} = P_\mathsf{beam}` in
        steady state exactly:

        * **no beam** -- the cavity settles at
          :math:`V^* = 2 Q_L (R/Q) I_\mathsf{gen}`, giving
          :math:`I_\mathsf{refl} = I_\mathsf{gen}`, i.e. **total**
          reflection. That is the correct answer for a superconducting
          cavity: with :math:`Q_0 \to \infty` there is nowhere for the
          power to go, so all of it comes back out of the coupler.
        * **beam-matched** -- when the beam absorbs the whole forward
          wave (:math:`I_\mathsf{beam} = I_\mathsf{gen}`, on crest, on
          resonance) the cavity sits at
          :math:`V = (R/Q) Q_L I_\mathsf{gen}` and the reflection is
          exactly zero. This is the point a linac coupler is matched to.

        Between those, what shows up here is beam loading (the passage
        drops ``V_ant`` while the generator still drives), detuning
        (``V_ant`` and ``I_gen`` acquire a relative phase, so the
        difference is non-zero even at matched magnitude -- a detuned
        cavity reflects) and transients.

        The unloaded quality factor :math:`Q_0` does not appear
        separately: for the superconducting cavities used here
        :math:`Q_0 \gg Q_L`, so :math:`Q_L \simeq Q_\mathsf{ext}` and the
        wall dissipation is negligible against the coupler outflow. That
        matters because the machine files carry ``Q_L_TESLA`` but no
        ``Q_0``.

        Both arguments are per cavity, matching
        :meth:`generator_power`; the coarse grid of
        ``IQCavityFeedbackTimingClass`` is normalised per cavity while the
        fine grid carries the station total.

        **The two grids are not in the same frame.**  The composed
        antenna voltage is ``V_beam + V_gen * r_gen``: the generator
        component is rotated by the generator frame rotation (station
        clock, kick-clock gap and registration phase), while the stored
        generator current stays in the design frame that drives
        ``V_gen``.  Subtracting it unrotated mixes frames, which on an
        accelerating multi-section ring -- where the rotation grows turn
        by turn -- leaves the cavity energy balance
        :math:`P_\mathsf{for} - P_\mathsf{refl} = P_\mathsf{beam} +
        \mathrm{d}U/\mathrm{d}t` open by a few per cent.  With the
        rotation it closes to rounding.  :meth:`generator_power` depends
        only on :math:`|I_\mathsf{gen}|` and needs no rotation.
        """
        if generator_current is None:
            generator_current = self.generator_current_coarse_grid
        if antenna_voltage is None:
            antenna_voltage = self.antenna_voltage_coarse_grid
        if generator_frame_rotation is None:
            generator_frame_rotation = self._generator_frame_rotation
        return (
            antenna_voltage / (self.R_over_Q * self.Q_L)
            - generator_frame_rotation * generator_current
        )

    def reflected_power(
        self: IQCavityFeedbackTimingClass,
        generator_current: complex | NumpyArray | None = None,
        antenna_voltage: complex | NumpyArray | None = None,
        generator_frame_rotation: complex | None = None,
    ) -> float | NumpyArray:
        r"""
        Power reflected back out of the coupler, per cavity.

        .. math::
            P_\mathsf{refl} = 0.5\,(R/Q)\,Q_L\,|I_\mathsf{refl}|^2

        The same convention as :meth:`generator_power`, applied to
        :meth:`reflected_current`, so the two are directly comparable and
        their ratio is the fraction of forward power the cavity throws
        back.

        Parameters
        ----------
        generator_current
            Generator current [A]; defaults to the coarse-grid generator
            current of the current turn.
        antenna_voltage
            Antenna voltage [V] per cavity; defaults to the coarse-grid
            antenna voltage of the current turn.
        generator_frame_rotation
            Frame rotation of the generator current, see
            :meth:`reflected_current`; defaults to the feedback's current
            one.

        Returns
        -------
        reflected_power
            Reflected power [W], same shape as the inputs.

        Notes
        -----
        This is the *instantaneous* reflected power on whichever grid it
        is evaluated, not a pulse average: during a bunch passage it rises
        sharply and it is largest exactly where the loop is fighting
        hardest. A klystron sees the peak, not the mean, so the peak is
        usually the number that sizes the circulator load.
        """
        return (
            0.5
            * self.R_over_Q
            * self.Q_L
            * np.abs(
                self.reflected_current(
                    generator_current=generator_current,
                    antenna_voltage=antenna_voltage,
                    generator_frame_rotation=generator_frame_rotation,
                )
            )
            ** 2
        )

    def _update_generator_current(
        self: IQCavityFeedbackTimingClass,
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

        It is then rotated into the ACTUATOR frame by
        ``_pi_error_frame_rotation`` (``exp(+i * delta_phi_rf)``): the
        controller returns a generator current, which drives the
        design-anchored generator component, so the loop gain would
        otherwise pick up the composition's ``exp(-i * delta_phi_rf)`` and
        rotate without bound under an RF-frequency offset. Also exactly
        unity when no offset ever acted.

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
        error = (
            self.pi_setpoint
            - (
                self.antenna_voltage_coarse_grid[idx]
                * self._kick_frame_rotation
            )
        ) * self._pi_error_frame_rotation
        delta_t = omega_times_dt / self._omega_input_for_pi
        self.generator_current_coarse_grid[idx] = (
            self._controller.update_generator_current(error, delta_t)
        )
