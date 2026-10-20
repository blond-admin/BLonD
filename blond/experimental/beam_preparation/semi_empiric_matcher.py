# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

import gc
import math
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from blond.beam_preparation.base import MatchingRoutine
from blond.beam_preparation.helpers import populate_beam
from blond.core.backends.backend import (
    backend,
)
from blond.core.helpers import int_from_float_with_warning
from blond.experimental.beam_preparation.bucket_filler_functions import (
    hamilton_to_density_by_max,
)
from blond.experimental.simulation.warmup import warmup
from blond.generals.cupy_.no_cupy_import import AllowPlotting, copy_to_cpu

# Oversampling factor for potential well calculation
_POTENTIAL_WELL_OVERSAMPLING = 10

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import Any

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.beam.particle_types import ParticleType
    from blond.core.simulation.simulation import Simulation


def get_hamilton_semi_analytic(
    ts: NumpyArray | CupyArray,
    potential_well: NumpyArray | CupyArray,
    reference_total_energy: float,
    eta: float,
    beta: float,
    shape: tuple[int, int],
    energy_range: tuple[float, float] | None = None,
) -> tuple[NumpyArray, NumpyArray, NumpyArray]:
    r"""
    Compute the 2D Hamiltonian :math:`H(t, \Delta E)` of a potential well.

    The time axis is ``ts``; the energy axis is sampled uniformly over
    ``energy_range``.

    Notes
    -----
    The Hamiltonian is computed using the relation

    .. math::

        H(t, \Delta E) = \frac{1}{2} \frac{|\eta|}{\beta^2 E_0} (\Delta E)^2
                         + V(t)

    where

    - :math:`E_0` is the reference total energy [eV],
    - :math:`\eta` is the zeroth-order slippage factor,
    - :math:`V(t)` is ``potential_well`` evaluated at ``ts`` [eV].

    The absolute value :math:`|\eta|` assumes that ``potential_well`` is
    oriented such that stable fixed points are minima on both sides of
    transition, as returned by
    :meth:`~blond.core.simulation.simulation.Simulation.get_potential_well_empiric`.

    Parameters
    ----------
    ts : array_like
        Time coordinates of the potential well [s].
    potential_well : array_like
        Potential well values at ``ts`` [eV]. Its offset is arbitrary.
    reference_total_energy : float
        Reference total energy :math:`E_0` [eV].
    eta : float
        Zeroth-order slippage factor [unitless]. Only :math:`|\eta|` is used.
    beta : float
        Relativistic velocity factor :math:`\beta = v/c` [unitless].
    shape : tuple of int
        Grid shape as ``(n_time, n_energy)``. Only ``n_energy`` is used;
        the time axis always has ``len(ts)`` points.
    energy_range : tuple of float or None, optional
        Range of :math:`\Delta E` values to evaluate [eV].
        If ``None``, the range is chosen symmetric around zero such that
        :math:`H` spans the full depth of the potential well, which
        comprises the largest separatrix inside the given potential.

    Returns
    -------
    deltaE_grid : numpy.ndarray
        2D grid of energy offsets :math:`\Delta E` [eV], with shape
        ``(len(ts), n_energy)``.
    time_grid : numpy.ndarray
        2D grid of time coordinates :math:`t` [s], same shape.
    hamilton_2D : numpy.ndarray
        Hamiltonian evaluated on the grid [eV], same shape. All outputs are
        NumPy arrays, also when the inputs are CuPy arrays.
    """
    # from here only CPU code, because `populate_beam` can only
    # handle numpy arrays
    potential_well = copy_to_cpu(potential_well)
    ts = copy_to_cpu(ts)

    assert len(ts) == len(potential_well), (
        f"{len(ts)=}, but {len(potential_well)=}"
    )
    ts = copy_to_cpu(ts)
    potential_well = copy_to_cpu(potential_well)

    E0 = reference_total_energy  # [eV]

    # Compute kinetic energy term constant
    drift_term = np.abs(eta) / (np.square(beta) * E0)  # [1/eV]

    # Auto-estimate ΔE range if not provided
    if energy_range is None:
        dE_max = np.sqrt(
            (potential_well.max() - potential_well.min())
            / (0.5 * abs(drift_term))
        )
        _energy_range = (-dE_max, dE_max)
    else:
        _energy_range = energy_range

    assert _energy_range[1] > _energy_range[0], (
        f"``energy_range`` must be increasing, but got, {_energy_range=}"
    )

    # Uniformly sample energy differences ΔE
    _dE_base = np.linspace(
        _energy_range[0], _energy_range[1], shape[1]
    )  # [eV]

    # Create 2D meshgrid: time_grid is time [s], deltaE_grid is ΔE [eV]
    time_grid, deltaE_grid = np.meshgrid(ts, _dE_base, indexing="ij")
    # Expand potential V(t) to 2D grid
    V = potential_well[:, None]  # [eV]

    # Compute the Hamiltonian hamilton_2D(t, ΔE) = 0.5 * const * ΔE² + V(t)
    hamilton_2D = 0.5 * drift_term * np.square(deltaE_grid) + V  # [eV]

    return deltaE_grid, time_grid, hamilton_2D


def get_one_turn_map_empiric(
    simulation: Simulation,
    t_guess: float,
    delta_t: float,
    delta_E: float,
    particle_type: ParticleType,
    intensity: int = 0,
    until_section_index: int = -1,
    newton_steps: int = 2,
) -> tuple[float, float, NumpyArray]:
    r"""
    Fixed point and linear one-turn map, by tracking five probe particles.

    Probe particles at the centre and at :math:`\pm` ``delta_t`` and
    :math:`\pm` ``delta_E`` around it are tracked through one turn. The
    central differences give the Jacobian

    .. math::

        M = \begin{pmatrix}
            \partial t_1 / \partial t_0 & \partial t_1 / \partial \Delta E_0 \\
            \partial \Delta E_1 / \partial t_0 &
            \partial \Delta E_1 / \partial \Delta E_0
        \end{pmatrix},

    and Newton steps move the centre onto the fixed point of the one-turn
    map. Since the map is tracked, ``M`` contains whatever the ring
    contains: any drift formula, multi-harmonic RF stations, several RF
    stations per turn and the order of the elements.

    Parameters
    ----------
    simulation
        Simulation whose one-turn map is tracked. It is not modified.
    t_guess
        Start value for the time coordinate of the fixed point, in [s],
        e.g. the minimum of
        :meth:`~blond.core.simulation.simulation.Simulation.get_potential_well_empiric`.
    delta_t
        Step in time for the central differences, in [s]. Small compared
        to the bucket, so that the map is linear.
    delta_E
        Step in energy for the central differences, in [eV]. Small
        compared to the bucket height, so that the map is linear.
    particle_type
        Type of particle (e.g., proton, electron).
    intensity
        Beam intensity (number of real particles) to include collective
        effects. Default is 0 (no intensity effects).
    until_section_index
        Section index until which to run the simulation. Default is -1.
    newton_steps
        Number of Newton steps towards the fixed point. Default is 2.

    Returns
    -------
    t_fixed
        Time coordinate of the fixed point, in [s].
    dE_fixed
        Energy coordinate of the fixed point, in [eV].
    one_turn_map
        Jacobian :math:`M` of the one-turn map at the fixed point, shape
        (2, 2), units [[1, s/eV], [eV/s, 1]].

    See Also
    --------
    get_grid_correction_from_one_turn_map : Uses ``one_turn_map``.

    Notes
    -----
    The Courant-Snyder parameters of the matched ellipse at the start of
    the turn follow from :math:`M`, e.g.
    :math:`\cos\mu = \mathrm{tr}(M) / 2` and
    :math:`\alpha = (M_{11} - M_{22}) / (2 \sin\mu)`.
    """
    from blond.core.beam.beams import ProbeBeam  # prevent circular import

    centre = np.array([t_guess, 0.0])
    one_turn_map = np.eye(2)
    for _ in range(newton_steps + 1):
        dt = centre[0] + np.array([0.0, delta_t, -delta_t, 0.0, 0.0])
        dE = centre[1] + np.array([0.0, 0.0, 0.0, delta_E, -delta_E])
        probe_bunch = ProbeBeam(
            dt=dt,
            dE=dE,
            particle_type=particle_type,
            intensity=intensity,
        )
        # prevent side effect
        deepcopy(simulation).run_simulation(
            beams=(probe_bunch,),
            n_turns=1,
            show_progressbar=False,
            verbose=False,
            until_section_index=until_section_index,
        )
        dt_1 = copy_to_cpu(probe_bunch.read_partial_dt()).astype(float)
        dE_1 = copy_to_cpu(probe_bunch.read_partial_dE()).astype(float)
        one_turn_map = np.array(
            [
                [
                    (dt_1[1] - dt_1[2]) / (2 * delta_t),
                    (dt_1[3] - dt_1[4]) / (2 * delta_E),
                ],
                [
                    (dE_1[1] - dE_1[2]) / (2 * delta_t),
                    (dE_1[3] - dE_1[4]) / (2 * delta_E),
                ],
            ]
        )
        residual = np.array([dt_1[0], dE_1[0]]) - centre
        centre = centre - np.linalg.solve(one_turn_map - np.eye(2), residual)
    return float(centre[0]), float(centre[1]), one_turn_map


def get_grid_correction_from_one_turn_map(
    one_turn_map: NumpyArray,
    beta_hamiltonian: float,
) -> tuple[float, float]:
    r"""Shear and energy scale that turn Hamiltonian contours into the matched ellipse.

    The contours of the semi-analytic Hamiltonian are upright ellipses near
    the bucket centre, with the aspect ratio
    :math:`\beta_H = \Delta t / \Delta E` given by the analytic drift term.
    The particles keep the invariant ellipse of the tracked one-turn map
    :math:`M` instead, which is tilted, because the kicks and drifts happen
    one after the other, and whose aspect ratio differs from
    :math:`\beta_H` by :math:`\mathcal{O}(\mu^2)`, :math:`\mu` being the
    synchrotron phase advance per turn.

    Mapping the Hamiltonian grid with
    :math:`\Delta E \to s_E \Delta E`, then
    :math:`t \to t + s_t \Delta E`, turns the upright ellipse into the
    invariant one. Both keep the crossings of the contours with
    :math:`\Delta E = 0`, which come from the empiric potential well, and
    are the same for every bucket.

    Parameters
    ----------
    one_turn_map
        Jacobian of the one-turn map at the fixed point, shape (2, 2), see
        :func:`get_one_turn_map_empiric`.
    beta_hamiltonian
        Aspect ratio :math:`\Delta t / \Delta E` of the Hamiltonian contours
        at the bucket centre, :math:`\sqrt{a / V''}` for
        :math:`H = a \Delta E^2 / 2 + V(t)`, in [s/eV].

    Returns
    -------
    shear_dt_per_dE
        :math:`s_t = (M_{11} - M_{22}) / (2 M_{21})`, slope
        :math:`\mathrm{d}t / \mathrm{d}\Delta E` of the invariant ellipse
        where it crosses :math:`\Delta E = 0`, in [s/eV].
    deltaE_scale
        :math:`s_E = \gamma \beta_H`, with the Courant-Snyder
        :math:`\gamma = -M_{21} / \sin\mu`.

    Raises
    ------
    ValueError
        If ``one_turn_map`` is not linearly stable,
        :math:`|\mathrm{tr}(M)| \geq 2`.
    """
    cos_mu = 0.5 * float(np.trace(one_turn_map))
    if not abs(cos_mu) < 1:
        raise ValueError(
            f"The one-turn map is not linearly stable, {2 * cos_mu=}."
        )
    sin_mu = np.sign(one_turn_map[0, 1]) * np.sqrt(1 - cos_mu**2)
    gamma_twiss = -one_turn_map[1, 0] / sin_mu
    shear_dt_per_dE = (one_turn_map[0, 0] - one_turn_map[1, 1]) / (
        2 * one_turn_map[1, 0]
    )
    return float(shear_dt_per_dE), float(gamma_twiss * beta_hamiltonian)


class DebuggingEndpoints:
    """
    Helper to expose variables when debugging `SemiEmpiricMatcher`.
    """

    def __init__(self):
        self.last_potential_well = None
        self.last_density = None
        self.last_hamilton_2D = None


class SemiEmpiricMatcher(MatchingRoutine):
    r"""Match a distribution to ``potential_well_empiric`` using an analytic drift term.

    This function matches a beam distribution to the empirically determined potential well,
    including an analytic drift term. The process iteratively adjusts the distribution until
    it converges within a specified tolerance, optionally accounting for intensity effects.

    Parameters
    ----------
    time_limit : tuple of float
        Start and stop of time, in seconds (s).
        The Hamiltonian will be calculated within this time range.
    n_macroparticles : int
        Number of macroparticles to inject into the beam.
    hamilton_to_density_kwargs : dict, optional
        Keyword arguments passed to ``hamilton_to_density_function``.
        For the default function, the following keys may be used:

        - ``density_modifier`` : float
          Exponent that shapes the density distribution according to :math:`H^{\mathrm{density\_modifier}}`.
        - ``hamilton_max`` : float
          Maximum value of the Hamiltonian, in arbitrary units.
          Tip: Set `animate=True` to see the internal state of the Hamilton values.
    hamilton_to_density_function : callable
        Function that converts the 2D Hamiltonian array into a density function.
        See also :func:`hamilton_to_density_by_max`.
    internal_grid_shape : tuple of int
        Shape of the internal grid as ``(n_time, n_energy)``, used to generate
        the beam particle coordinates.
    maxiter_intensity_effects : int, optional
        Maximum number of iterations allowed for convergence with intensity effects.
    increment_intensity_effects_until_iteration_i : int, optional
        Number of iterations during which intensity effects are gradually increased
        before matching at full beam intensity.
        Useful for convergence when intensity effects are strong.
    seed : int, optional
        Random seed ensuring reproducibility.
        Runs with the same seed will produce identical distributions.
    animate : bool, default=False
        If ``True``, draws a plot using ``matplotlib.pyplot.draw()`` at each iteration.
    animate_pause_time : float, default=0.1
        Time to pause after each animation frame, in [s].
    tolerance_potential_well : float, optional
        Convergence threshold. The matching process stops when the error
        falls below this tolerance.
    stagnation_patience : int, optional
        Stop the intensity-effects iteration after this many steps without
        a new error minimum. The fixed-point iteration can settle into a
        limit cycle instead of converging below ``tolerance_potential_well``,
        and the matched beam at the stalled point is already correct.
    n_warmup_turns : int, optional
        Number of frozen-beam warmup turns (see
        :func:`blond.experimental.simulation.warmup.warmup`) executed in
        each intensity-effects iteration, before the wake-sampling turn.
        This equilibrates turn-dependent internal state of multi-turn
        wakefield solvers and feedbacks with the current bunch shape, so
        the matched potential well includes the equilibrium multi-turn
        wake instead of a cold single-pass wake. Default is 0 (disabled).
        Note that warmup turns always cover the full ring;
        ``until_section_index`` only applies to the sampling turn.
    verbose : bool, default=False
        If ``True``, prints convergence and status messages to the console.
    debug
        If ``True``, variables are saved into the `debug_helper` attribute.
    until_section_index
        Section index until which to run the simulation. Default is -1.
    correct_to_one_turn_map
        If ``True``, tilt and scale the matched distribution in energy so
        that its core is the invariant ellipse of the tracked one-turn map,
        see :func:`get_grid_correction_from_one_turn_map`. If ``False``
        (default, unchanged results for existing scripts), the bunch is
        matched to the smooth Hamiltonian only, and its length oscillates
        by a fraction of :math:`\mu`, the synchrotron phase advance per
        turn, unless the RF kicks sit in the middle of the drifts. The
        correction is linear, so it does not correct the shape of
        large-amplitude contours.

    See Also
    --------
    hamilton_to_density_by_max : Default ``hamilton_to_density_function``.
    blond.experimental.beam_preparation.bucket_filler_functions.multibunch_match_metric_to_hamilton :
        Fill several buckets, fitting e.g. the RMS emittance of each bunch.
    blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes.ProfileMatcherAddon :
        Match to a measured line density.
    blond.core.simulation.simulation.Simulation.get_potential_well_empiric :
        Potential well that the Hamiltonian is built from.

    Notes
    -----
    This routine is intended for iterative beam-matching workflows where
    the potential well and intensity effects are both considered analytically.

    Further documentation:

    - :ref:`semi_empiric_matcher`: theory, algorithm and convergence
      criterion.
    - :ref:`potential_well_empiric`: how the potential well is obtained.
    - :doc:`/examples/notebooks/tutorial_SemiEmpiricMatcher`: step-by-step
      notebook, including matching to a measured line density.
    - Example scripts ``blond/examples/scripts/EX_08_Semi_empiric_matcher.py``
      (multi-bunch filling with a custom density function and a wakefield)
      and ``blond/examples/scripts/EX_06_Acceleration_match_density.py``
      (matching to a line density during acceleration).

    **Custom density function.** ``hamilton_to_density_function`` is called
    as ``hamilton_to_density_function(time_grid=..., deltaE_grid=...,
    hamilton_2D=..., **hamilton_to_density_kwargs)``. It must therefore

    - accept the keyword arguments ``time_grid`` [s], ``deltaE_grid`` [eV]
      and ``hamilton_2D`` [eV] (spelled exactly like this), all NumPy arrays
      of shape ``internal_grid_shape``,
    - accept every key of ``hamilton_to_density_kwargs`` as keyword argument,
    - return a non-negative NumPy array of the same shape. Its normalization
      does not matter; the beam is sampled from ``density / density.sum()``.

    The Hamiltonian is approximately zero at the bottom of the potential
    well, because the matcher uses the minimum-subtracted potential from
    :meth:`~blond.core.simulation.simulation.Simulation.get_potential_well_empiric`
    (averaged between iterations). Subtracting its minimum, as in the example
    below, makes a custom function independent of that convention.

    Examples
    --------
    Fill a single LHC bucket with a density that decays exponentially in
    the Hamiltonian, :math:`\rho \propto \exp(-H / H_\mathrm{decay})`,
    cut at a fraction of the Hamiltonian at the separatrix.

    >>> import numpy as np
    >>> from blond import (
    ...     Beam,
    ...     ConstantMagneticCycle,
    ...     DriftSimple,
    ...     Ring,
    ...     Simulation,
    ...     SingleHarmonicRFStation,
    ...     momentum_compaction_factor,
    ...     proton,
    ... )
    >>> from blond.experimental.beam_preparation.semi_empiric_matcher import (
    ...     SemiEmpiricMatcher,
    ... )
    >>>
    >>> def truncated_exponential_density(
    ...     time_grid,  # required, [s]
    ...     deltaE_grid,  # required, [eV]
    ...     hamilton_2D,  # required, [eV]
    ...     decay,  # custom: 1/e decay, as fraction of the well depth
    ...     filling,  # custom: cut, as fraction of the well depth
    ... ):
    ...     # H(t, dE=0) = V(t), so the minimum along dE is the potential
    ...     potential_well = hamilton_2D.min(axis=1)
    ...     well_depth = potential_well.max() - potential_well.min()
    ...     # 0 at the bucket centre, 1 at the separatrix
    ...     hamilton_relative = (
    ...         hamilton_2D - potential_well.min()
    ...     ) / well_depth
    ...     density = np.exp(-hamilton_relative / decay)
    ...     density[hamilton_relative > filling] = 0.0
    ...     return density
    >>>
    >>> ring = Ring(circumference=26658.883)
    >>> ring.add_elements(
    ...     [
    ...         SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0),
    ...         DriftSimple(
    ...             orbit_length=26658.883,
    ...             momentum_compaction_factor=momentum_compaction_factor(
    ...                 transition_gamma=55.759505
    ...             ),
    ...         ),
    ...     ]
    ... )
    >>> sim = Simulation(
    ...     ring=ring,
    ...     magnetic_cycle=ConstantMagneticCycle(
    ...         value=450e9, reference_particle=proton
    ...     ),
    ... )
    >>> beam = Beam(intensity=1e11, particle_type=proton)
    >>> sim.prepare_beam(
    ...     beam=beam,
    ...     preparation_routine=SemiEmpiricMatcher(
    ...         time_limit=(0, 2.5e-9),  # one RF bucket
    ...         n_macroparticles=1e5,
    ...         hamilton_to_density_function=truncated_exponential_density,
    ...         hamilton_to_density_kwargs=dict(decay=0.2, filling=0.8),
    ...         maxiter_intensity_effects=0,  # no intensity effects here
    ...     ),
    ... )
    """

    def __init__(
        self,
        time_limit: tuple[float, float],
        n_macroparticles: int | float,
        hamilton_to_density_kwargs: dict[str, Any],
        hamilton_to_density_function: Callable[
            ..., NumpyArray
        ] = hamilton_to_density_by_max,
        internal_grid_shape: tuple[int, int] = (1023, 1023),
        seed: int | None = 0,
        tolerance_potential_well: float = 1e-6,
        maxiter_intensity_effects=1000,
        increment_intensity_effects_until_iteration_i: int = 0,
        stagnation_patience: int = 10,
        n_warmup_turns: int = 0,
        animate: bool = False,
        animate_pause_time: float = 0.1,
        verbose: bool = True,
        debug=False,
        until_section_index: int = -1,
        correct_to_one_turn_map: bool = False,
    ) -> None:
        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles,
            warning_stacklevel=2,
        )
        self.maxiter_intensity_effects = int_from_float_with_warning(
            maxiter_intensity_effects,
            warning_stacklevel=2,
        )
        self.increment_intensity_effects_until_iteration_i = (
            int_from_float_with_warning(
                increment_intensity_effects_until_iteration_i,
                warning_stacklevel=2,
            )
        )

        self.stagnation_patience = int_from_float_with_warning(
            stagnation_patience,
            warning_stacklevel=2,
        )
        self.n_warmup_turns = int_from_float_with_warning(
            n_warmup_turns,
            warning_stacklevel=2,
        )

        self.internal_grid_shape: tuple[int, int] = internal_grid_shape
        self.seed = (
            int_from_float_with_warning(
                seed,
                warning_stacklevel=2,
            )
            if seed is not None
            else None
        )
        self.time_limit: tuple[float, float] = time_limit
        assert callable(hamilton_to_density_function)
        self.hamilton_to_density_function = hamilton_to_density_function
        self.hamilton_to_density_kwargs = hamilton_to_density_kwargs
        self.until_section_index = until_section_index
        self.correct_to_one_turn_map = correct_to_one_turn_map
        # (shear_dt_per_dE, deltaE_scale), see
        # `get_grid_correction_from_one_turn_map`
        self._grid_correction: tuple[float, float] | None = None
        self.animate = animate
        self.animate_pause_time = animate_pause_time
        self.tolerance_potential_well = tolerance_potential_well
        self.verbose = verbose

        # For error calculation and plotting during `prepare_beam`
        self._last_potential_well: NumpyArray | CupyArray | None = None
        self._prelast_potential_well: NumpyArray | CupyArray | None = None
        self.debug_helper: DebuggingEndpoints | None = None

        # Can be potentially replaced like `hamilton_to_density_function`,
        #  for example using the `SymbolicSeparatrixHelper`.
        self._get_hamilton = self._get_hamiltonian_semianalytic

        if debug:
            self.debug_helper = DebuggingEndpoints()

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            `Simulation` context manager
        beam
            Simulation beam object
        """

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        # make sure there is not a mixed state
        simulation.intensity_effect_manager.is_active_wakefields()
        simulation.intensity_effect_manager.is_active_profiles()

        sim_tmp = deepcopy(simulation)  # prevent side effects

        ts = backend.linspace(
            self.time_limit[0], self.time_limit[1], self.internal_grid_shape[0]
        )
        # match beam without intensity effects
        sim_tmp.intensity_effect_manager.set_wakefields(active=False)
        sim_tmp.intensity_effect_manager.set_profiles(active=False)
        self._match_beam(beam, sim_tmp, ts)

        # iterate solution with intensity effects
        intensity_original = beam.intensity

        # Get decimal places from the tolerance (e.g., 1e-6 → 6)
        tolerance_decimal_places = abs(
            int(math.floor(math.log10(self.tolerance_potential_well)))
        )
        if self.animate:
            plt.figure("SemiEmpiricMatcher")
            plt.clf()
            self._plot_current_state(beam, 0, 0, ts)
            plt.draw()
            plt.pause(self.animate_pause_time)

        # Track the best (smallest) self-consistency error to detect a limit
        # cycle / stall (see `self.stagnation_patience`), together with the
        # (small, grid-sized) averaged potential well that produced it: on a
        # stall the loop must roll back to that minimum, not keep whatever
        # (worse) state the last, merely non-improving, iteration left
        # behind. The beam itself (potentially GB-sized) is never copied —
        # it is deterministically regenerated from the stored potential well
        # via `self.seed`.
        best_error = float("inf")
        stall_count = 0

        if sim_tmp.intensity_effect_manager.has_wakefields():
            for i_intensity in range(self.maxiter_intensity_effects):
                sim_tmp = deepcopy(simulation)  # prevent side effects

                # Change the strength of intensity effects to allow
                # convergence to a stable solution (if there is any?)
                if (
                    i_intensity
                    < self.increment_intensity_effects_until_iteration_i
                ):
                    scalar = (
                        i_intensity
                        / self.increment_intensity_effects_until_iteration_i
                    )  # t
                else:
                    scalar = 1.0
                beam.intensity = scalar * intensity_original

                # run simulation with beam to collect the actual profiles
                # that cause the wake-fields
                sim_tmp.intensity_effect_manager.set_profiles(active=True)
                sim_tmp.intensity_effect_manager.unfreeze_wakefields()

                # Equilibrate multi-turn wakefield solver / feedback state
                # with the current bunch shape before sampling the wake,
                # so the sampling turn below observes the equilibrium
                # multi-turn wake instead of a cold single-pass wake.
                # `warmup` restores beam coordinates, counters and
                # `beam.reference` itself.
                if self.n_warmup_turns > 0:
                    warmup(
                        simulation=sim_tmp,
                        beam=beam,
                        n_turns=self.n_warmup_turns,
                        show_progressbar=False,
                        verbose=False,
                    )

                # this might get changed by the simulation
                beam_reference_time_original = beam.reference.time
                beam_reference_total_energy_original = (
                    beam.reference.total_energy
                )
                turn_i_original = int(sim_tmp.turn_counter.value)
                section_i_original = int(sim_tmp.section_counter.value)

                sim_tmp.run_simulation(
                    beams=(beam,),
                    n_turns=1,
                    until_section_index=self.until_section_index,
                    show_progressbar=False,
                    verbose=False,
                )

                # reset to original value before simulation
                beam.reference.time = beam_reference_time_original
                beam.reference.total_energy = (
                    beam_reference_total_energy_original
                )
                sim_tmp.turn_counter.value = turn_i_original
                sim_tmp.section_counter.value = section_i_original

                # Prevent the profiles from updating.
                sim_tmp.intensity_effect_manager.set_profiles(active=False)
                sim_tmp.intensity_effect_manager.freeze_wakefields()
                # This is intended as override, so that the line density
                # inside `_match_beam` experiences the forces from the
                # previously run with the full beam
                self._match_beam(beam, sim_tmp, ts)

                if self.animate:
                    plt.figure("SemiEmpiricMatcher")
                    plt.clf()
                    self._plot_current_state(beam, i_intensity, scalar, ts)
                    plt.draw()
                    plt.pause(self.animate_pause_time)

                error_calculable = (
                    self._last_potential_well is not None
                    and self._prelast_potential_well is not None
                )
                # calculate errors on wakes (not on beam profiles)
                # because this will more stable as noise is smoothed out
                if error_calculable and i_intensity > 1:
                    # Root mean square deviation
                    obs = self._last_potential_well
                    obs = obs / obs.max()
                    obs_previous = self._prelast_potential_well
                    obs_previous = obs_previous / obs_previous.max()
                    error = float(
                        np.sqrt(backend.mean((obs - obs_previous) ** 2))
                    )
                    if self.verbose:
                        print(
                            f"Iteration: {i_intensity:<5}"
                            f" | Intensity {(scalar * 100):3.1f}%"
                            f" | Error: {error:.{tolerance_decimal_places}f}"
                        )
                    past_ramp = (
                        i_intensity
                        > self.increment_intensity_effects_until_iteration_i
                    )
                    if error < self.tolerance_potential_well and past_ramp:
                        break

                    # Limit-cycle guard: stop after `self.stagnation_patience`
                    # iterations without a new error minimum.
                    if error < best_error:
                        best_error = error
                        stall_count = 0
                        # `self._last_potential_well`/`_prelast_potential_well`
                        # were just updated by the `_match_beam` call above,
                        # so this reproduces exactly the averaged potential
                        # well that produced the beam currently in place.
                        best_avg_potential_well: NumpyArray | CupyArray = (
                            self._last_potential_well
                            + self._prelast_potential_well
                        ) / 2
                    else:
                        stall_count += 1
                    if (stall_count >= self.stagnation_patience) and past_ramp:
                        if self.verbose:
                            print(
                                "Stopping: self-consistency error stalled at"
                                f" {best_error:.{tolerance_decimal_places}f}"
                                " (limit cycle, not converging below"
                                f" tolerance {self.tolerance_potential_well})."
                            )
                        # Roll back to the beam state at the error minimum,
                        # regenerated deterministically (same `self.seed`)
                        # from the small stored potential well, instead of
                        # keeping the state left by this stalled iteration.
                        self._rematch_beam_to_potential_well(
                            beam=beam,
                            simulation=sim_tmp,
                            ts=ts,
                            avg_potential_well=best_avg_potential_well,
                        )
                        break

                # The per-iteration deepcopy of the full simulation is only
                # kept alive by reference cycles, so reassigning `sim_tmp`
                # next iteration does not free it by refcount. Drop it
                # explicitly so peak memory stays bounded over long runs
                # instead of growing until the cyclic collector happens to run.
                del sim_tmp
                gc.collect()

            beam.intensity = intensity_original

    def _match_beam(
        self,
        beam: BeamBaseClass,
        simulation: Simulation,
        ts: NumpyArray | CupyArray,
    ) -> None:
        """Matches the beam coordinates to the current potential well.

        Notes
        -----
        The potential well is overwritten by the intensity effect manager
        outside this method, so that the line distribution experiences
        the same forces than the actual bunch.

        Parameters
        ----------
        beam
            Simulation beam object
        simulation
            `Simulation` context manager
        ts
            Time coordinate, in [s] for observation of the potential well.
        """
        time_grid, deltaE_grid, hamilton_2D = self._get_hamilton(
            beam=beam, simulation=simulation, ts=ts
        )
        self._populate_beam_from_hamiltonian(
            beam=beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
        )

    def _rematch_beam_to_potential_well(
        self,
        beam: BeamBaseClass,
        simulation: Simulation,
        ts: NumpyArray | CupyArray,
        avg_potential_well: NumpyArray | CupyArray,
    ) -> None:
        """Regenerate beam coordinates from a previously observed potential well.

        Unlike :meth:`_match_beam`, this does not query
        ``simulation.get_potential_well_empiric`` again; it rebuilds the
        Hamiltonian directly from ``avg_potential_well``. Combined with the
        fixed ``self.seed``, this deterministically reproduces the beam
        state that ``avg_potential_well`` originally produced, without ever
        having to keep a copy of the (potentially GB-sized) beam coordinates
        around.

        Parameters
        ----------
        beam
            Simulation beam object.
        simulation
            `Simulation` context manager, used only for ``ring`` parameters.
        ts
            Time coordinate, in [s] for observation of the potential well.
        avg_potential_well
            Previously observed (already-averaged) potential well [V].
        """
        deltaE_grid, time_grid, hamilton_2D = get_hamilton_semi_analytic(
            ts=ts,
            potential_well=avg_potential_well,
            reference_total_energy=beam.reference.total_energy,
            beta=beam.reference.beta,
            eta=float(
                simulation.ring.calc_average_eta_0(beam.reference.gamma)
            ),
            shape=self.internal_grid_shape,
        )
        self._populate_beam_from_hamiltonian(
            beam=beam,
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
        )

    def _populate_beam_from_hamiltonian(
        self,
        beam: BeamBaseClass,
        time_grid: NumpyArray | CupyArray,
        deltaE_grid: NumpyArray | CupyArray,
        hamilton_2D: NumpyArray | CupyArray,
    ) -> None:
        """Turn a Hamiltonian grid into a density and populate ``beam`` with it.

        Parameters
        ----------
        beam
            Simulation beam object.
        time_grid
            2D grid of time coordinates :math:`t` [s].
        deltaE_grid
            2D grid of energy offset :math:`\\Delta E` [eV].
        hamilton_2D
            2D Hamiltonian evaluated on the grid [eV].
        """
        density = self.hamilton_to_density_function(
            time_grid=time_grid,
            deltaE_grid=deltaE_grid,
            hamilton_2D=hamilton_2D,
            **self.hamilton_to_density_kwargs,
        )

        if self.debug_helper is not None:
            self.debug_helper.last_density = density
            self.debug_helper.last_hamilton_2D = hamilton_2D

        if self._grid_correction is not None:
            # The density stays a function of H, but sits on the invariant
            # ellipse of the one-turn map instead of the contours of H
            shear_dt_per_dE, deltaE_scale = self._grid_correction
            deltaE_grid = deltaE_scale * deltaE_grid
            time_grid = time_grid + shear_dt_per_dE * deltaE_grid

        populate_beam(
            beam=beam,
            time_grid=time_grid.T,
            deltaE_grid=deltaE_grid.T,
            density_grid=density.T,
            n_macroparticles=self.n_macroparticles,
            seed=self.seed,
        )

    def _get_hamiltonian_semianalytic(
        self,
        beam: BeamBaseClass,
        simulation: Simulation,
        ts: NumpyArray,
    ) -> tuple[
        NumpyArray,
        NumpyArray,
        NumpyArray,
    ]:
        r"""Build the semi-analytic 2D Hamiltonian for the current beam.

        Queries the empirically determined potential well from the
        simulation, averages it with the potential well of the previous
        call to damp oscillations between successive matching iterations,
        and feeds the result into :func:`get_hamilton_semi_analytic` to
        obtain :math:`H_{2D}(t, \Delta E)`.

        The potential well is sampled on a finer grid (oversampled by
        ``_POTENTIAL_WELL_OVERSAMPLING``) and then down-sampled to the
        resolution of ``ts`` to improve the accuracy of the empiric
        estimate. Must be called on turn zero only.

        Parameters
        ----------
        beam : BeamBaseClass
            Beam whose particle type, intensity and reference parameters
            (total energy, ``beta``, ``gamma``) define the potential well
            and the analytic drift term.
        simulation : Simulation
            Simulation providing the empiric potential well via
            :meth:`Simulation.get_potential_well_empiric` and the ring used
            to compute the average slippage factor :math:`\eta_0`.
        ts : ndarray
            Time coordinates at which the Hamiltonian is evaluated [s].

        Returns
        -------
        time_grid : ndarray
            2D grid of time coordinates :math:`t` [s].
        deltaE_grid : ndarray
            2D grid of energy offset :math:`\Delta E` [eV].
        hamilton_2D : ndarray
            2D semi-analytic Hamiltonian evaluated on the grid [eV].

        """
        assert simulation.turn_counter.value == 0
        potential_well, factor, tilt_dt_per_dE = (
            simulation.get_potential_well_empiric(
                # Fine grid with dt[::_POTENTIAL_WELL_OVERSAMPLING] == ts
                # (up to rounding), so the down-sampled potential well
                # below is evaluated at ts. ts has len(ts) - 1 intervals;
                # splitting each into _POTENTIAL_WELL_OVERSAMPLING needs
                # (len(ts) - 1) * _POTENTIAL_WELL_OVERSAMPLING + 1 points.
                # len(ts) * _POTENTIAL_WELL_OVERSAMPLING points would put
                # the down-sampled points off ts, shifting the well to
                # later times by up to one grid step of ts.
                dt=np.linspace(
                    ts.min(),
                    ts.max(),
                    (len(ts) - 1) * _POTENTIAL_WELL_OVERSAMPLING + 1,
                ),
                particle_type=beam.particle_type,
                intensity=beam.intensity,
                until_section_index=self.until_section_index,
            )
        )
        # back on ts, see the construction of dt above
        potential_well = (
            potential_well[::_POTENTIAL_WELL_OVERSAMPLING] * factor
        )

        self._prelast_potential_well = self._last_potential_well
        self._last_potential_well = potential_well
        if self._prelast_potential_well is None:
            avg_pot_well = potential_well
        else:
            avg_pot_well = (potential_well + self._prelast_potential_well) / 2
        if self.debug_helper is not None:
            self.debug_helper.last_potential_well = avg_pot_well
        eta = float(simulation.ring.calc_average_eta_0(beam.reference.gamma))
        deltaE_grid, time_grid, hamilton_2D = get_hamilton_semi_analytic(
            ts=ts,
            potential_well=avg_pot_well,
            reference_total_energy=beam.reference.total_energy,
            beta=beam.reference.beta,
            eta=eta,
            shape=self.internal_grid_shape,
        )
        if self.correct_to_one_turn_map:
            self._grid_correction = self._measure_grid_correction(
                beam=beam,
                simulation=simulation,
                ts=ts,
                potential_well=avg_pot_well,
                eta=eta,
                deltaE_max=float(deltaE_grid.max()),
            )
        return time_grid, deltaE_grid, hamilton_2D

    def _measure_grid_correction(
        self,
        beam: BeamBaseClass,
        simulation: Simulation,
        ts: NumpyArray | CupyArray,
        potential_well: NumpyArray | CupyArray,
        eta: float,
        deltaE_max: float,
    ) -> tuple[float, float] | None:
        """Track the one-turn map at the bucket centre and derive the grid correction.

        Parameters
        ----------
        beam
            Simulation beam object.
        simulation
            `Simulation` context manager.
        ts
            Time coordinates of ``potential_well``, in [s].
        potential_well
            Potential well that the Hamiltonian is built from.
        eta
            Slippage factor of the analytic drift term.
        deltaE_max
            Largest energy offset of the Hamiltonian grid, in [eV].

        Returns
        -------
        grid_correction
            ``(shear_dt_per_dE, deltaE_scale)``, see
            :func:`get_grid_correction_from_one_turn_map`, or ``None`` with a
            warning if the bucket centre has no linear motion to match to,
            e.g. a flat potential well or a ``time_limit`` that misses the
            bucket centre.
        """
        ts_cpu = copy_to_cpu(ts)
        potential_well_cpu = copy_to_cpu(potential_well)
        i_min = int(np.argmin(potential_well_cpu))
        neighbours = slice(max(i_min - 3, 0), i_min + 4)
        # V''(t) at the bucket centre, by a parabola through 7 points
        curvature = (
            2
            * np.polyfit(
                ts_cpu[neighbours] - ts_cpu[i_min],
                potential_well_cpu[neighbours],
                2,
            )[0]
        )
        drift_term = abs(eta) / (
            beam.reference.beta**2 * beam.reference.total_energy
        )
        if not curvature > 0:
            warnings.warn(
                "SemiEmpiricMatcher: the potential well does not curve"
                " upwards at its minimum (it is flat, or `time_limit` misses"
                " the bucket centre), so the beam is not corrected to the"
                " one-turn map.",
                stacklevel=2,
            )
            return None
        beta_hamiltonian = float(np.sqrt(drift_term / curvature))
        # small steps compared to the bucket: linear part of the map
        delta_E = 1e-3 * deltaE_max
        t_fixed, _, one_turn_map = get_one_turn_map_empiric(
            simulation=simulation,
            t_guess=float(ts_cpu[i_min]),
            delta_t=delta_E * beta_hamiltonian,
            delta_E=delta_E,
            particle_type=beam.particle_type,
            intensity=beam.intensity,
            until_section_index=self.until_section_index,
        )
        grid_step = float(ts_cpu[1] - ts_cpu[0])
        try:
            if abs(t_fixed - ts_cpu[i_min]) > 10 * grid_step:
                raise ValueError(
                    f"no fixed point near the potential minimum, {t_fixed=}"
                )
            return get_grid_correction_from_one_turn_map(
                one_turn_map=one_turn_map,
                beta_hamiltonian=beta_hamiltonian,
            )
        except ValueError as exc:
            warnings.warn(
                "SemiEmpiricMatcher: the beam is not corrected to the"
                f" one-turn map: {exc}",
                stacklevel=2,
            )
            return None

    def _plot_current_state(
        self,
        beam: BeamBaseClass,
        i: int,
        scalar: float,
        ts: NumpyArray | CupyArray,
    ) -> None:
        """Make a plot of the current state of the matcher.

        Parameters
        ----------
        beam
            Simulation beam object
        i
            Current iteration
        scalar
            Current strength if the intensity effects
        ts
            Time coordinate, in [s] for observation of the potential well.


        """
        plt.figure("SemiEmpiricMatcher")
        with AllowPlotting():
            plt.subplot(2, 1, 1)
            plt.title(
                f"Iteration {i}, Intensity strength {(scalar * 100):3.1f}%"
            )
            plt.hist(
                copy_to_cpu(beam.read_partial_dt()),
                bins=self.internal_grid_shape[0],
                range=self.time_limit,
                density=True,
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Density (arb. unit)")

            plt.subplot(2, 1, 2)
            if self._last_potential_well is not None:
                plt.plot(ts, self._last_potential_well)
            if self._prelast_potential_well is not None:
                plt.plot(ts, self._prelast_potential_well)
            plt.xlabel("Time (s)")
            plt.ylabel("Potential (arb. unit)")
