# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""The pole-residue wakefield solver `MultiPoleSparseSolve`."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.base import (
    SupportsVectorFittedModel,
    WakeField,
    WakeFieldSolver,
)
from blond.physics.impedances.bin_average import triple_box_average_poles
from blond.physics.impedances.multipole.checks import (
    _check_decaying_poles,
    _check_whole_bin_gaps,
)
from blond.physics.impedances.multipole.model import (
    _Carry,
    _FarField,
    _is_far_field_pole,
    _PoleModel,
)
from blond.physics.impedances.multipole.vector_fit import VectorFit
from blond.physics.profiles import StaticProfile
from blond.physics.profiles_sparse import EquidistantMultiProfile

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


# Slack on the call-gap check below, as a fraction of a bin: a profile that
# spans the whole revolution period meets the check with equality, and its
# two sides are computed along different routes.
_CALL_GAP_TOLERANCE = 1e-6
# Lag, in bin widths, past which the bin-averaged wake is a pure
# exponential (the B-spline no longer straddles the causal onset).
_CARRIED_ONSET_BINS = 1.5


def _vector_fit_of(source: SupportsVectorFittedModel) -> VectorFit:
    """
    The vector fit of a source, from either return form.

    Parameters
    ----------
    source
        Source implementing `SupportsVectorFittedModel`.

    Returns
    -------
    vector_fit
        The vector fit, with zero direct and inductive term if the source
        returns the plain ``(poles, residues, counterrotation_signs)``.

    Raises
    ------
    TypeError
        If the source does not provide a vector fit.
    """
    if not hasattr(source, "get_vectorfit"):
        raise TypeError(
            "`MultiPoleSparseSolve` needs sources implementing "
            f"`SupportsVectorFittedModel`, but got {type(source)=}."
        )
    return VectorFit(*source.get_vectorfit())


class MultiPoleSparseSolve(WakeFieldSolver):
    r"""
    Solver that uses a vector-fitted pole-residue model to calculate the induced voltage.

    The bin-averaged wake of a pole :math:`p` with residue :math:`\rho` is a
    pure exponential from a lag of :math:`\tfrac32 \Delta t` on, so it
    splits into a **far field** -- one complex state per pole, summed by
    `Specials.wake_from_pole_residue` with the residue scaled to
    :math:`\rho\,((e^{p\Delta t} - 1) / (p\Delta t))^3\, e^{p\Delta t / 2}`
    and read two bins behind the bin -- and a **near field**, the three
    taps at lags of minus one, zero and one bin, evaluated in closed form by
    :func:`~blond.physics.impedances.bin_average.triple_box_average_poles`.
    The split is an identity, not an approximation, for a profile whose
    bins lie a whole number of bins apart (checked at the first call); see
    Steps 61-71 of ``docs/models_new/pole-residue/
    bin_averaged_wake_derivation.py``.

    The direct term :math:`d` and the inductive term :math:`e` of the
    vector fit have the wakes :math:`d\,\delta(t)` and
    :math:`e\,\delta'(t)`, so they act on the binned profile directly:
    as :math:`d I` and as :math:`e\,dI/dt` by a central difference, the
    stencil `InductiveImpedance` uses for the frequency-domain solvers. A
    bin next to a gap sees the empty bins beyond it.

    Between calls the kernel hands its state over one bin before the last
    bin, and the last bin's charge is carried separately: it enters the
    state after the next call's first read-out, and its whole contribution
    to that first bin is added here in closed form at the true lag.

    A pole whose wake dies within a bin gains nothing from the recursion:
    when the sum of its far-field wake over every lag from two bins on is
    below ``far_field_tolerance`` times the peak of its near taps, it
    skips the recursion and only the near taps carry it. It keeps its
    closed-form term for the carried charge, because a call gap between
    1.5 and 2 bins reads that term where the bin-averaged wake of a fast
    pole falls off only as :math:`(p\Delta t)^{-2}`, not exponentially.

    Parameters
    ----------
    far_field_tolerance
        Largest share of a pole's near-field peak its far field may have,
        summed over all lags from two bins on, for the pole to skip the
        recursion. The default of 1e-12 drops real poles decaying by more
        than about 41 per bin; 0 keeps every pole.

    See Also
    --------
    blond.physics.impedances.base.SupportsVectorFittedModel : Interface for wakefield sources that can provide the poles and residues this solver consumes.
    """

    def __init__(self, far_field_tolerance: float = 1e-12) -> None:
        # Share of a pole's near field its far field may have and still be
        # dropped from the recursion; see the class docstring.
        self._far_field_tolerance = far_field_tolerance
        # The `WakeField` this solver belongs to; gives sources and profile.
        self._parent_wakefield: WakeField | None = None

        # Built at the first call: the model on the bin grid, the poles the
        # recursion keeps, and what each call hands to the next.
        self._model: _PoleModel | None = None
        self._far_field: _FarField | None = None
        self._carry: _Carry | None = None
        self._direct_term: float = 0.0  # in [Ohm]
        self._inductive_term: float = 0.0  # in [Ohm s]
        # Working buffers of the kernel, changed every call.
        # The far-field recursion's state, one complex number per kept
        # pole: the wake of all charge so far, ready to read at the next bin.
        self._states: NumpyArray | CupyArray | None = None
        # Kernel scratch, one voltage row per thread, summed afterwards.
        self._voltage_threaded: NumpyArray | CupyArray | None = None
        # The carried charge's wake at a seamless hand-over, per whether
        # the two calls rotate opposite ways; see `_near_field_by_kernel`.
        self._seamless_carried_wakes: dict[bool, float] = {}
        # Reference time of the previous call, in [s]; None before the
        # first call. Gives the lag to the carried charge.
        self.last_reference_time: float | None = None

    def on_wakefield_init_simulation(
        self, simulation: Simulation, parent_wakefield: WakeField
    ) -> None:
        """
        Lateinit method when WakeField is late-initialized.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        parent_wakefield
            `WakeField` that this solver affiliated to.

        Raises
        ------
        TypeError
            If the profile is neither an `EquidistantMultiProfile` nor a
            `StaticProfile`.
        """
        profile = parent_wakefield.profile
        if not isinstance(profile, (EquidistantMultiProfile, StaticProfile)):
            raise TypeError(
                "Expected `EquidistantMultiProfile` or `StaticProfile`,"
                f" but got {type(profile)=}."
            )
        self._parent_wakefield = parent_wakefield
        self._profile = profile

    def _initialise(self) -> None:
        """Collect the model and allocate the state, once."""
        poles, residues, signs = [], [], []
        assert self._parent_wakefield is not None

        for source in self._parent_wakefield.sources:
            vector_fit = _vector_fit_of(source)
            poles.extend(vector_fit.poles)
            residues.extend(vector_fit.residues)
            signs.extend(vector_fit.counterrotation_signs)
            self._direct_term += float(vector_fit.direct_term)
            self._inductive_term += float(vector_fit.inductive_term)
        poles = backend.array(poles, dtype=backend.complex)
        residues = backend.array(residues, dtype=backend.complex)
        signs = backend.array(signs, dtype=backend.float)
        _check_decaying_poles(poles)
        profile = self._parent_wakefield.profile
        hist_x = profile.hist_x
        bin_dt = float(profile.hist_step)
        _check_whole_bin_gaps(hist_x, bin_dt)

        pole_dt = poles * bin_dt
        # The inductive term's share of the near taps; see `_PoleModel.taps`.
        inductive = 0.5 * self._inductive_term / bin_dt**2
        self._model = _PoleModel(
            poles=poles,
            residues=residues,
            counterrotating_pole_signs=signs,
            pair_factors=2.0 - (poles.imag == 0),
            tail_residues=(
                residues
                * (backend.expm1(pole_dt) / pole_dt) ** 3
                * backend.exp(pole_dt / 2.0)
            ),
            taps=triple_box_average_poles(
                backend.array([bin_dt, 0.0, -bin_dt], dtype=backend.float),
                poles,
                residues,
                bin_dt,
            )
            + backend.array(
                [-inductive, self._direct_term / bin_dt, inductive],
                dtype=backend.float,
            ),
            adjacent=(hist_x[1:] - hist_x[:-1]) < 1.5 * bin_dt,
            bin_dt=bin_dt,
            span_dt=float(hist_x[-1] - hist_x[0]),
        )
        kept = backend.array(
            np.flatnonzero(
                _is_far_field_pole(self._model, self._far_field_tolerance)
            )
        )
        far_poles = poles[kept]
        far_residues = self._model.tail_residues[kept]
        far_signs = signs[kept]
        gemm_plan = None
        no_parity = backend.specials.no_parity
        if no_parity is not None:
            offsets = np.rint(copy_to_cpu(hist_x - hist_x[0]) / bin_dt)
            gemm_plan = no_parity.pole_tile_gemm_plan(
                bin_offsets=offsets.astype(np.int64),
                poles=far_poles,
                residues=far_residues,
                counterrotating_pole_signs=far_signs,
                bin_dt=bin_dt,
            )
        self._far_field = _FarField(
            poles=far_poles,
            residues=far_residues,
            pole_signs=far_signs,
            gemm_plan=gemm_plan,
        )
        self._carry = _Carry(charge=backend.zeros(1, dtype=backend.float))

        self._states = backend.zeros(
            len(self._far_field.poles), dtype=backend.complex
        )
        self._voltage_threaded = backend.zeros(
            (backend.specials.get_max_threads(), len(hist_x)),
            dtype=backend.float,
        )

    def _thread_scratch(self) -> NumpyArray | CupyArray:
        """
        The kernel's per-thread scratch, grown for the active backend.

        The scratch is sized at the first call, but the active backend may
        change later and use more threads.

        Returns
        -------
        voltage_threaded
            At least ``get_max_threads()`` rows of ``n_bins``.
        """
        n_threads = backend.specials.get_max_threads()
        if self._voltage_threaded.shape[0] < n_threads:
            self._voltage_threaded = backend.zeros(
                (n_threads, self._voltage_threaded.shape[1]),
                dtype=backend.float,
            )
        return self._voltage_threaded

    def _carried_lag_since_previous_call(self, beam: BeamBaseClass) -> float:
        """
        Time from the previous call's last bin to this call's first, in [s].

        Measured between bin centres, so two seamless calls are one bin
        width apart.

        Parameters
        ----------
        beam
            Simulation object of a particle beam, for the reference time.

        Returns
        -------
        carried_lag_dt
            Exactly one bin width (seamless) or at least 1.5. On the first
            call nothing is carried and the state is zero, so any valid lag
            gives the same result; two bin widths are reported.

        Raises
        ------
        ValueError
            If no time passed since the previous call, if the lag is
            shorter than one bin width, or longer than one but shorter
            than 1.5.
        """
        bin_dt = self._model.bin_dt
        if self.last_reference_time is None:
            return 2.0 * bin_dt
        passed_dt = float(beam.reference.time - self.last_reference_time)
        if passed_dt <= 0.0:
            # The cavity state lives on one time line. Two beams reaching
            # the solver at the same time would need their charge
            # superposed in one call, which is not implemented.
            raise ValueError(
                "MultiPoleSparseSolve: calls must come at strictly "
                "increasing reference times, but this call is "
                f"{passed_dt} s after the previous one. Two beams that "
                "reach one solver at the same reference time (e.g. a co- "
                "and a counter-rotating beam in one `run_simulation`) "
                "are not supported."
            )
        carried_lag_dt = passed_dt - self._model.span_dt
        lag_bins = carried_lag_dt / bin_dt
        # `reference.time` advances by `+=` every turn, so `passed_dt` is
        # only known to about one ulp of `reference.time`; late in a cycle
        # that is more than `_CALL_GAP_TOLERANCE` of a bin.
        tolerance = (
            _CALL_GAP_TOLERANCE
            + 4.0 * float(np.spacing(abs(beam.reference.time))) / bin_dt
        )
        # A gate on user-supplied geometry, so it raises rather than
        # asserting. Seamless calls (a profile spanning the whole
        # revolution period) are one bin apart; a longer profile would let
        # the bins of consecutive turns overlap and count the same charge
        # twice.
        if lag_bins < 1.0 - tolerance:
            raise ValueError(
                "MultiPoleSparseSolve: the profile must not be longer than "
                "the time between two calls of the solver. The profile "
                f"spans {self._model.span_dt / bin_dt + 1:g} bins of {bin_dt} s, "
                f"but only {passed_dt} s passed since the previous call, "
                f"leaving {lag_bins} bin widths between the last bin "
                "of that call and the first bin of this one (at least 1 "
                "needed)."
            )
        if lag_bins <= 1.0 + tolerance:
            # Exactly one bin can land a few ulp short; never hand the
            # kernel a negative lag.
            return bin_dt
        # Between seamless and the onset the carried charge would need the
        # near-field series per pole every call.
        if lag_bins < _CARRIED_ONSET_BINS - tolerance:
            raise ValueError(
                "MultiPoleSparseSolve: consecutive calls must be either "
                "seamless or at least half a bin width apart. The last bin "
                "of the previous call and the first bin of this one are "
                f"{lag_bins} bin widths apart (centre to centre); allowed "
                f"are exactly 1 or at least {_CARRIED_ONSET_BINS}."
            )
        return carried_lag_dt

    def _carried_charge_voltage(
        self, carried_lag_dt: float, sign: NumpyArray | CupyArray | float
    ) -> NumpyArray | CupyArray:
        """
        The carried charge's whole contribution to this call's first bin.

        Parameters
        ----------
        carried_lag_dt
            Lag from the carried bin to this call's first bin, in [s].
        sign
            Per-pole sign with which the carried charge is seen, or 1.0.

        Returns
        -------
        voltage
            Length-1 array, in [V].
        """
        return self._carry.charge * self._carried_wake(carried_lag_dt, sign)

    def _carried_wake(
        self, carried_lag_dt: float, sign: NumpyArray | CupyArray | float
    ) -> NumpyArray | CupyArray:
        r"""
        The wake of the carried bin at this call's first bin, all poles.

        Past the onset the bin-averaged wake of every pole is the pure
        exponential, :math:`\rho\,((e^{p\Delta t} - 1) / (p\Delta t))^3
        e^{p (t - \frac32\Delta t)}`, which is the tail residue advanced
        by :math:`t - 2\Delta t`; that is evaluated for all poles at once,
        those outside the recursion included. A seamless hand-over sits on
        the near tap and takes the per-pole closed form.

        Parameters
        ----------
        carried_lag_dt
            Lag from the carried bin to this call's first bin, in [s].
        sign
            Per-pole sign with which the carried charge is seen, or 1.0.

        Returns
        -------
        wake
            Length-1 array, in [V/C].
        """
        model = self._model
        bin_dt = model.bin_dt
        if carried_lag_dt == bin_dt:
            return triple_box_average_poles(
                backend.array([carried_lag_dt], dtype=backend.float),
                model.poles,
                model.residues * sign,
                bin_dt,
            )
        wake = (
            model.tail_residues
            * backend.exp(model.poles * (carried_lag_dt - 2.0 * bin_dt))
        ).real
        return backend.sum(model.pair_factors * sign * wake, keepdims=True)

    def _near_field_by_kernel(
        self,
        no_parity: Any,
        hist_y: NumpyArray | CupyArray,
        factor: float,
        carried_lag_dt: float,
        flipped: bool,
        voltage: NumpyArray | CupyArray,
    ) -> None:
        """
        The near field, the carried charge and its hand-over, in one kernel.

        `calc_induced_voltage`'s steps after the far field, by
        ``no_parity.pole_near_field``. A seamless hand-over's closed form
        is the same every call, so it is evaluated once per direction.

        Parameters
        ----------
        no_parity
            ``backend.specials.no_parity``.
        hist_y
            Histogram of the profile.
        factor
            Conversion of `hist_y` to the charge per bin.
        carried_lag_dt
            Lag of the carried charge to the first bin, in [s].
        flipped
            Whether the carried charge's call rotated the other way.
        voltage
            Induced voltage per bin, in [V]; added to.
        """
        model = self._model
        carried: dict[str, Any] = {}
        if carried_lag_dt == model.bin_dt:
            if flipped not in self._seamless_carried_wakes:
                sign = model.counterrotating_pole_signs if flipped else 1.0
                self._seamless_carried_wakes[flipped] = float(
                    copy_to_cpu(self._carried_wake(carried_lag_dt, sign))[0]
                )
            carried["carried_wake"] = self._seamless_carried_wakes[flipped]
        else:
            weights = model.pair_factors
            if flipped:
                weights = weights * model.counterrotating_pole_signs
            carried.update(
                carried_poles=model.poles,
                carried_residues=model.tail_residues,
                carried_weights=weights,
                carried_shift_dt=carried_lag_dt - 2.0 * model.bin_dt,
            )
        no_parity.pole_near_field(
            profile=hist_y,
            factor=factor,
            taps=model.taps,
            adjacent=model.adjacent,
            carried_charge=self._carry.charge,
            voltage=voltage,
            **carried,
        )

    def _far_field_voltage(
        self,
        beam: BeamBaseClass,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
        factor: float,
        carried_lag_dt: float,
    ) -> NumpyArray | CupyArray:
        """
        The far field: the recursion over the kept poles.

        The residues are scaled to the bin-averaged wake past its onset.
        Reads and advances the poles' states. Evaluated as tiled GEMMs by
        ``backend.specials.no_parity`` where its plan exists (the CUDA
        backend, with enough poles on a grid that tiles), else by
        `Specials.wake_from_pole_residue`; both give the same result.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.
        hist_x
            Bin centres of the profile, in [s].
        hist_y
            Histogram of the profile.
        factor
            Conversion of `hist_y` to the charge per bin.
        carried_lag_dt
            Lag of the carried charge to the first bin, in [s].

        Returns
        -------
        voltage
            The far-field voltage per bin, in [V].
        """
        far_field, carry = self._far_field, self._carry
        bin_dt = self._model.bin_dt
        voltage = backend.zeros(len(hist_y), dtype=backend.float)
        no_parity = backend.specials.no_parity
        if far_field.gemm_plan is not None and no_parity is not None:
            no_parity.wake_from_pole_residue_gemm(
                plan=far_field.gemm_plan,
                profile=hist_y,
                carried_charge=carry.charge,
                carried_is_counterrotating=carry.is_counter_rotating,
                state_lag_dt=carried_lag_dt - bin_dt,
                carried_lag_dt=carried_lag_dt,
                is_counterrotating_beam=beam.is_counter_rotating,
                factor=factor,
                states=self._states,
                voltage=voltage,
            )
        elif len(far_field.poles) > 0:
            backend.specials.wake_from_pole_residue(
                profile_time=hist_x,
                profile=hist_y,
                carried_charge=carry.charge,
                carried_is_counterrotating=carry.is_counter_rotating,
                state_lag_dt=carried_lag_dt - bin_dt,
                carried_lag_dt=carried_lag_dt,
                poles=far_field.poles,
                residues=far_field.residues,
                is_counterrotating_beam=beam.is_counter_rotating,
                counterrotating_pole_signs=far_field.pole_signs,
                factor=factor,
                bin_dt=bin_dt,
                states=self._states,
                voltage=voltage,
                voltage_threaded=self._thread_scratch(),
            )
        return voltage

    def calc_induced_voltage(
        self, beam: BeamBaseClass
    ) -> NumpyArray | CupyArray:
        """
        Calculate the induced voltage in this turn based on the last profiles.

        Parameters
        ----------
        beam
            Simulation object of a particle beam.

        Returns
        -------
        induced_voltage
            The induced voltage, in [V].
        """
        if self._model is None:
            self._initialise()
        model, carry = self._model, self._carry
        profile = self._parent_wakefield.profile
        hist_x, hist_y = profile.hist_x, profile.hist_y
        factor = self._hist_y_to_intensity_factor(beam=beam, profile=profile)
        carried_lag_dt = self._carried_lag_since_previous_call(beam)

        voltage = self._far_field_voltage(
            beam, hist_x, hist_y, factor, carried_lag_dt
        )

        # Near field: the previous bin, the bin itself and the next bin --
        # when adjacent; a neighbour across a gap is the recursion's. The
        # taps include the direct and inductive terms. Then
        # the carried charge's whole contribution to the first bin, at the
        # true lag; a pole's sign enters once if the two calls rotate
        # opposite ways. Last, the last bin's charge is carried on.
        flipped = beam.is_counter_rotating != carry.is_counter_rotating
        no_parity = backend.specials.no_parity
        if no_parity is not None:
            self._near_field_by_kernel(
                no_parity, hist_y, factor, carried_lag_dt, flipped, voltage
            )
        else:
            taps, adjacent = model.taps, model.adjacent
            voltage += (factor * taps[1]) * hist_y
            voltage[1:] += factor * taps[0] * adjacent * hist_y[:-1]
            voltage[:-1] += factor * taps[2] * adjacent * hist_y[1:]
            sign = model.counterrotating_pole_signs if flipped else 1.0
            voltage[:1] += self._carried_charge_voltage(carried_lag_dt, sign)
            carry.charge[:] = factor * hist_y[-1:]
        carry.is_counter_rotating = beam.is_counter_rotating
        self.last_reference_time = copy(beam.reference.time)
        return voltage
