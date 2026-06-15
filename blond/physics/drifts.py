# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of implementations to handle movement in bent synchrotron sections."""

from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING

import sympy
from scipy.constants import speed_of_light as c0

from blond.core.backends.backend import backend
from blond.core.base import (
    AltersReference,
    BeamPhysicsRelevant,
    DynamicParameter,
    HasSymbolicHamiltonian,
    Schedulable,
)
from blond.core.reference_clock.reference_clock import ReferenceCoordinates

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


class DriftBaseClass(BeamPhysicsRelevant, AltersReference, ABC):
    """
    Base class of a drift.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m].
        Length / Velocity => Time to pass the element.
    section_index
        Section index to group elements into sections.
    radiation_integrals
        Synchrotron radiation integrals.
        Use `SynchrotronRadiationMaster` to activate synchrotron radiation.
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.
    """

    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        radiation_integrals: NumpyArray | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ) -> None:
        super().__init__(
            section_index=section_index,
            **kwargs,  # for MRO of fused elements
        )

        self.orbit_length = orbit_length
        self._radiation_integrals = radiation_integrals

    @property
    def radiation_integrals(self) -> NumpyArray | None:
        """
        Radiation integrals of the drift.

        Returns
        -------
        radiation_integrals
            Synchrotron radiation integrals.
        """
        return self._radiation_integrals

    @abc.abstractmethod  # pragma: no cover
    def eta_0(self, gamma: float) -> backend.float:
        """
        Drift in arc parameter eta for one turn in synchrotron.

        Parameters
        ----------
        gamma
            Lorentz gamma factor.

        Returns
        -------
        eta_0
            Drift in arc parameter eta for one turn in synchrotron.
        """
        pass


class DriftSimple(DriftBaseClass, Schedulable, HasSymbolicHamiltonian):
    r"""
    Base class to implement beam drifts in synchrotrons.

    The arrival-time change over the drift is calculated as:

    .. math::
        \Delta dt = \frac{L}{\beta c}\,\eta_0\,\frac{dE}{\beta^2 E},
        \qquad \eta_0 = \alpha_0 - \frac{1}{\gamma^2}

    where :math:`L` is the orbit length, :math:`\beta`, :math:`\gamma` and
    :math:`E` are the reference beam quantities, :math:`dE` the energy
    deviation and :math:`\eta_0` the (first-order) phase-slip factor built
    from the momentum compaction factor :math:`\alpha_0`.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m].
    section_index
        Section index to group elements into sections.
    radiation_integrals
        Synchrotron radiation integrals.
        Use `SynchrotronRadiationMaster` to activate synchrotron radiation.
    momentum_compaction_factor
        Momentum compaction factor of this drift section. In multi-drift
        setups the ring combines per-section values into a global weighted
        average; see :attr:`blond.core.ring.ring.Ring.momentum_compaction_factor`.
    **kwargs
        Additional keyword arguments for method
        resolution order of inheriting elements.
    """

    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        radiation_integrals: NumpyArray | None = None,
        momentum_compaction_factor: float | None = None,
        **kwargs: dict[str, Any],  # for MRO of fused elements
    ) -> None:
        """
        Base class to implement beam drifts in synchrotrons.

        Parameters
        ----------
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element.
        section_index
            Section index to group elements into sections.
        radiation_integrals
            Synchrotron radiation integrals.
        momentum_compaction_factor
            Momentum compaction factor of this drift section. In multi-drift
            setups the ring combines per-section values into a global weighted
            average; see :attr:`blond.core.ring.ring.Ring.momentum_compaction_factor`.
            Use ``drift.schedule("momentum_compaction_factor", ...)`` to influence
            the parameter along the ramp.
        **kwargs
            Additional keyword arguments for method
            resolution order of inheriting elements.

        Examples
        --------
        Parameters can be scheduled along the simulation execution

        >>> from blond import DriftSimple
        >>> drift = DriftSimple(...)
        >>> drift.schedule(attribute='momentum_compaction_factor', value=np.array(...), mode="per-turn")
        """
        super().__init__(
            orbit_length=orbit_length,
            section_index=section_index,
            radiation_integrals=radiation_integrals,
            **kwargs,  # for MRO of fused elements
        )

        self._turn_counter: DynamicParameter | None = None

        self._add_intended_schedule("momentum_compaction_factor")

        self._simulation: Simulation | None = None

        self._last_eta_0: float | None = None

        self.momentum_compaction_factor: float | None = (
            momentum_compaction_factor
        )

    @staticmethod
    def headless(
        momentum_compaction_factor: NumpyArray | tuple[NumpyArray, NumpyArray],
        orbit_length: float,
        section_index: int = 0,
        turn_counter: DynamicParameter | None = None,
    ) -> DriftSimple:
        """
        Initialize object without simulation context.

        Parameters
        ----------
        momentum_compaction_factor
            Momentum compaction factor.
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element.
        section_index
            Section index to group elements into sections.
        turn_counter
            Live turn counter; accessed as ``turn_counter.value`` each track call.

        Returns
        -------
        drift_simple
            DriftSimple object without simulation context.
        """
        d = DriftSimple(
            orbit_length=orbit_length,
            section_index=section_index,
        )

        if isinstance(momentum_compaction_factor, int | float):
            d.momentum_compaction_factor = float(momentum_compaction_factor)
        else:
            d.schedule(
                "momentum_compaction_factor", momentum_compaction_factor
            )

        d.configure(turn_counter=turn_counter)

        return d

    def on_init_simulation(self, simulation: Simulation, **kwargs) -> None:
        """
        Lateinit method when `simulation.__init__` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        **kwargs
            Configure parameters collected by the MRO chain.
        """
        super().on_init_simulation(
            simulation, turn_counter=simulation.turn_counter, **kwargs
        )
        if (
            self.momentum_compaction_factor is None
        ) and "momentum_compaction_factor" not in self.schedules:
            raise ValueError(
                "You need to define `momentum_compaction_factor` via `.momentum_compaction_factor=...` "
                "or `.schedule(attribute='momentum_compaction_factor', value=...)`"
            )

    def configure(
        self, *, turn_counter: DynamicParameter | None = None, **kwargs
    ) -> None:
        """
        Store the turn counter needed for schedule application during tracking.

        Parameters
        ----------
        turn_counter
            Live turn counter; accessed as ``turn_counter.value`` each track call.
        **kwargs
            Passed to the next level in the MRO chain.
        """
        self._turn_counter = turn_counter
        super().configure(**kwargs)

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super()._track(beam=beam)

        if self.schedule_active:
            assert self._turn_counter is not None, (
                "Turn counter must be set with active scheduling."
            )
            self.apply_schedules(
                turn_i=self._turn_counter.value,
                reference_time=beam.reference.time,
            )

        dt = self.track_reference(beam.reference)
        gamma = beam.reference.gamma
        self._last_eta_0 = self.eta_0(gamma)

        if beam.common_array_size > 0:
            backend.specials.drift_simple(
                dt=beam.write_partial_dt(),
                dE=beam.read_partial_dE(),
                T=dt,
                eta_0=self._last_eta_0,
                beta=beam.reference.beta,
                energy=beam.reference.total_energy,
            )

    def track_reference(
        self, reference: ReferenceCoordinates, **kwargs
    ) -> float:
        """
        Update the coordinates of the reference coordinate system.

        Parameters
        ----------
        reference
            The object that holds the reference time [s] and total energy [eV].
        **kwargs
            Allows more arguments in the method definition outside the
            abstract class.

        Returns
        -------
        reference_time_change
            Change of reference time.
        """
        reference_time_change = self.orbit_length / reference.velocity
        reference.time += reference_time_change
        return reference_time_change

    def eta_0(self, gamma: float) -> float:
        """
        Drift in arc parameter eta for one turn in synchrotron.

        Parameters
        ----------
        gamma
            Lorentz gamma factor.

        Returns
        -------
        eta_0
            Drift in arc parameter eta for one turn in synchrotron.
        """
        return self.alpha_0 - (1 / (gamma * gamma))

    # alias of momentum_compaction_factor
    @property  # as readonly attributes
    def alpha_0(self) -> float:
        """
        Momentum compaction factor.

        Returns
        -------
        alpha_0
            Momentum compaction factor.

        See Also
        --------
        blond.core.ring.ring.Ring.momentum_compaction_factor : Orbit-length weighted average for multi-drift setups.
        """
        return self.momentum_compaction_factor

    def get_hamilton_symbolic(
        self, replace_symbols: bool = True
    ) -> sympy.Expr:
        r"""
        Return the partial Hamiltonian symbolic expression.

        The tracker (see ``DriftSimple._track``) maps
        ``dt -> dt + T * eta_0 * dE / (beta^2 E)`` with

        .. math::

            T = \frac{L}{\beta c},\qquad
            \eta_0 = \alpha_0 - \frac{1}{\gamma^2}.

        Hamilton's equation :math:`\partial H/\partial dE = T\,\eta_0\,dE /
        (\beta^2 E)` integrates to the linearized Hamiltonian

        .. math::

            H = \frac{1}{2}\,\frac{T\,\eta_0}{\beta^2 E}\,dE^2.

        Parameters
        ----------
        replace_symbols
            If ``True``, the according variables will be replaced by
            their current numeric value.
            ``False`` is intended to derive the value of an parameter
            analytically.

        Returns
        -------
        expression
            The symbolic expression.
        """
        dE, beta, gamma, E = sympy.symbols("dE beta gamma E", real=True)

        if replace_symbols:
            assert self.alpha_0 is not None
            alpha_0 = float(self.alpha_0)
        else:
            alpha_0 = sympy.Symbol("alpha_0", real=True)

        T = float(self.orbit_length) / (beta * c0)
        eta_0 = alpha_0 - 1 / gamma**2

        return sympy.Rational(1, 2) * T * eta_0 / (beta**2 * E) * dE**2


class DriftExact(DriftSimple, HasSymbolicHamiltonian):
    r"""
    Drift element using the exact drift formulation.

    This replaces the simple drift with the exact solver based on:
      - exact delta from dE
      - full alpha(delta) expansion
      - exact (1 + dE/E) / (1 + delta) factor

    The arrival-time change over the drift is calculated as:

    .. math::
        \Delta dt = \frac{L}{\beta c}\left[
            \mathrm{poly}(\delta)\,\frac{1 + dE/E}{1 + \delta} - 1
        \right]

    with

    .. math::
        \delta(dE) &= \sqrt{1 + \frac{dE^2 + 2\,dE\,E}{\beta^2 E^2}} - 1 \\
        \mathrm{poly}(\delta) &= 1 + \alpha_0\,\delta
            + \sum_k \alpha_{k+1}\,\delta^{k+2}

    where :math:`\delta` is the exact relative momentum deviation and
    :math:`\mathrm{poly}(\delta)` the full momentum-compaction expansion in
    the momentum compaction factor :math:`\alpha_0` and the higher-order
    coefficients :math:`\alpha_k`.

    Parameters
    ----------
    orbit_length
        Length of drift, in [m].
        Length / Velocity => Time to pass the element.
    section_index
        Section index to group elements into sections.
    momentum_compaction_factor
        Momentum compaction factor.
        Use ``drift.schedule("momentum_compaction_factor", ...)`` to influence
        the parameter along the ramp.
    higher_order_alpha
        Higher-order alpha array up to desired order.
        Use ``drift.schedule("higher_order_alpha", ...)`` to influence
        the parameter along the ramp.
    **kwargs
        Additional keyword arguments for MRO of fused elements.
    """

    def __init__(
        self,
        orbit_length: float,
        section_index: int = 0,
        momentum_compaction_factor: float | None = None,
        higher_order_alpha: NumpyArray | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            orbit_length=orbit_length,
            section_index=section_index,
            momentum_compaction_factor=momentum_compaction_factor,
            **kwargs,
        )

        self._add_intended_schedule("higher_order_alpha")
        self.higher_order_alpha = higher_order_alpha

    @staticmethod
    def headless(
        orbit_length: float,
        section_index: int = 0,
        momentum_compaction_factor: float | None = None,
        higher_order_alpha: NumpyArray | None = None,
        turn_counter: DynamicParameter | None = None,
    ) -> DriftExact:
        """
        `DriftExact` element using the exact drift formulation.

        This replaces the simple drift with the exact solver based on:
          - exact delta from dE
          - full alpha(delta) expansion
          - exact (1 + dE/E) / (1 + delta) factor

        Parameters
        ----------
        orbit_length
            Length of drift, in [m].
            Length / Velocity => Time to pass the element.
        section_index
            Section index to group elements into sections.
        momentum_compaction_factor
            Momentum compaction factor.
        higher_order_alpha
            Higher-order alpha array up to desired order.
        turn_counter
            Live turn counter; accessed as ``turn_counter.value`` each track call.

        Returns
        -------
        drift_exact
            ``DriftExact`` object.
        """
        drift = DriftExact(
            orbit_length=orbit_length,
            section_index=section_index,
            momentum_compaction_factor=momentum_compaction_factor,
            higher_order_alpha=higher_order_alpha,
        )
        drift.configure(turn_counter=turn_counter)
        return drift

    def get_hamilton_symbolic(
        self, replace_symbols: bool = True
    ) -> sympy.Expr:
        r"""
        Return the partial Hamiltonian symbolic expression.

        The tracker (see ``DriftExact._track``) maps
        ``dt -> dt + T * F(dE)`` with

        .. math::

            F(dE) &= \mathrm{poly}(\delta)\,\frac{1 + dE/E}{1 + \delta} - 1 \\
            \delta(dE) &= \sqrt{1 + (dE^2 + 2\,dE\,E)/(\beta^2 E^2)} - 1 \\
            \mathrm{poly}(\delta) &= 1 + \alpha_0\,\delta
                + \sum_k \alpha_{k+1}\,\delta^{k+2}

        Hamilton's equation :math:`\partial H/\partial dE = T\,F` plus the
        substitution :math:`u \to \delta` closes the integral in form:

        .. math::

            H = T\,(\beta^2 E\,P(\delta(dE)) - dE),\qquad
            P(\delta) := \int_0^\delta \mathrm{poly}(\delta')\,d\delta'

        We Taylor-expand :math:`\delta(dE)` in ``dE`` up to order
        ``len(higher_order_alpha) + 2`` — the highest order needed to fully
        represent every supplied :math:`\alpha_k` — so the result is a
        polynomial in ``dE`` and ``coeff(dE, n)`` works for downstream
        consumers like
        :class:`~blond.utilities.separatrix.symbolic_separatrix.SymbolicSeparatrixHelper`.

        Parameters
        ----------
        replace_symbols
            If ``True``, the according variables will be replaced by
            their current numeric value.
            ``False`` is intended to derive the value of an parameter
            analytically.

        Returns
        -------
        expression
            Polynomial in ``dE`` with coefficients in ``beta``, ``E``.
        """
        dE, beta, E = sympy.symbols("dE beta E", real=True)
        # Cast numeric inputs to native Python float: older sympy parses
        # numpy scalars via str(), which on NumPy 2.x yields
        # 'np.float64(...)' and fails Float.__new__.
        if replace_symbols:
            assert self.alpha_0 is not None
            assert self.higher_order_alpha is not None

            alpha_0 = float(self.alpha_0)
            higher = tuple(float(a) for a in self.higher_order_alpha)
        else:
            alpha_0 = sympy.Symbol("alpha_0", real=True)
            # Honor the configured number of higher-order alphas in
            # symbolic mode -- otherwise the Taylor truncation collapses
            # to dE**2 and every dE**k (k > 2) contribution is silently
            # dropped from the analytical Hamiltonian.
            n_higher = (
                0
                if self.higher_order_alpha is None
                else len(self.higher_order_alpha)
            )
            higher = tuple(
                sympy.Symbol(f"alpha_{k + 1}", real=True)
                for k in range(n_higher)
            )
        T = float(self.orbit_length) / (beta * c0)
        truncation = len(higher) + 2

        # Taylor-expand delta(dE) in dE and treat it as a polynomial.
        delta_exact = (
            sympy.sqrt(1 + (dE**2 + 2 * dE * E) / (beta**2 * E**2)) - 1
        )
        delta = delta_exact.series(dE, 0, truncation + 1).removeO()

        # P(delta) = integral_0^delta poly(delta') ddelta'
        P = delta + alpha_0 * delta**2 / 2
        for k, alpha_k in enumerate(higher):
            P += alpha_k * delta ** (k + 3) / (k + 3)

        # Expand and discard dE**k terms beyond the truncation order that
        # appear as artifacts of multiplying truncated polynomials.
        H = sympy.expand(T * (beta**2 * E * P - dE))
        return sum(H.coeff(dE, k) * dE**k for k in range(truncation + 1))

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine (exact drift).

        Parameters
        ----------
        beam
            Beam.
        """
        # Apply schedules if active
        if self.schedule_active:
            self.apply_schedules(
                turn_i=self._simulation.turn_counter.value,
                reference_time=beam.reference.time,
            )

        # Advance reference
        dt = self.track_reference(beam.reference)

        higher_alpha = backend.array(
            self.higher_order_alpha, dtype=backend.float
        )

        # Track macroparticles
        if beam.common_array_size > 0:
            backend.specials.drift_exact(
                dt=beam.write_partial_dt(),
                dE=beam.read_partial_dE(),
                T=dt,
                alpha_0=self.alpha_0,
                higher_alpha=higher_alpha,
                beta=beam.reference.beta,
                energy=beam.reference.total_energy,
            )
