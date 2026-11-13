.. _semi_empiric_matcher:

Semi-Empirical Beam Matching
============================

.. currentmodule:: blond.experimental.beam_preparation.semi_empiric_matcher

Overview
--------

The :class:`SemiEmpiricMatcher` class provides a **semi-empirical longitudinal
beam-matching procedure** that combines an *empirically obtained potential well*
from :meth:`~blond.core.simulation.simulation.Simulation.get_potential_well_empiric`
with an *analytic drift term* derived from the Hamiltonian of longitudinal motion.

The routine is particularly suited for cases where wakefields or high-intensity
effects modify the RF potential, as it iteratively finds a **self-consistent
equilibrium distribution**.

---

Conceptual Background
---------------------


The longitudinal dynamics of a particle in an RF system are governed by
the Hamiltonian:

.. math::

   H(t, \Delta E)
   = \frac{1}{2}
     \frac{|\eta|}{\beta^2 E_0}
     (\Delta E)^2
     + V(t)

where:

- :math:`t` is the arrival time relative to the synchronous particle [s],
- :math:`\Delta E` is the energy deviation from the synchronous energy [eV],
- :math:`E_0` is the reference total energy [eV],
- :math:`\eta` is the slippage factor (dimensionless),
- :math:`\beta` is the relativistic velocity fraction :math:`v/c` (dimensionless),
- :math:`V(t)` is the **empirical potential well** obtained from the simulation [eV].

The first term represents the kinetic energy contribution (the "drift" term),
with units of [eV], while the second term is the potential energy from the
RF system and collective effects.

The **Semi-Empiric Matcher** reconstructs a 2D grid of this Hamiltonian,
transforms it into a **density distribution** in phase space, and uses this
distribution to populate the beam macro-particles.

Through iterative refinement, the resulting beam self-consistently reproduces
the observed potential well and the associated intensity-dependent distortions.

---

Algorithmic Workflow
--------------------

The matching routine proceeds as follows:

1. **Initial potential acquisition**

   Using :meth:`~blond.core.simulation.simulation.Simulation.get_potential_well_empiric`, the empirical potential
   well is obtained by tracking a *probe beam* for one turn through the full
   RF configuration. The output potential includes all RF phase shifts and
   wakefield effects.

2. **Hamiltonian reconstruction**

   The semi-analytic Hamiltonian is computed using
   :func:`get_hamilton_semi_analytic`, combining the empirical potential
   :math:`V(t)` with an analytic drift term based on :math:`\eta`, :math:`\beta`,
   and the reference energy :math:`E_0`.

   If no energy range is explicitly provided, it is automatically estimated
   from the potential well's depth as:

   .. math::

      \Delta E_\mathrm{max} = \sqrt{\frac{V_\mathrm{max} - V_\mathrm{min}}
                                          {0.5 \cdot |\eta| / (\beta^2 E_0)}}

   This ensures that the Hamiltonian grid encompasses the largest separatrix
   within the given potential well.

3. **Density mapping**

   The mapping from a Hamiltonian to a density distribution in phase space
   can be customized via the parameter ``hamilton_to_density_function``.
   By default, the library uses :func:`hamilton_to_density_by_max`, which
   generates a density map according to

   .. math::

      \rho(t, \Delta E)
      = \left( 1 - \frac{H(t, \Delta E)}{H_\mathrm{max}} \right)^{n}

   where ``n`` = ``density_modifier`` controls how sharply the density
   decreases toward the separatrix. Hamiltonian values above ``H_\mathrm{max}``
   are truncated before computing the density, ensuring that all density
   values remain in the range [0, 1].

   This density map determines how macro-particles are distributed in phase space,
   with higher density in regions of lower Hamiltonian.

   Users can supply a custom function in place of :func:`hamilton_to_density_by_max`
   by providing it to ``hamilton_to_density_function``. The custom function must
   accept the keyword arguments

   - ``time_grid`` (:class:`numpy.ndarray`): 2D grid of time coordinates [s]
   - ``deltaE_grid`` (:class:`numpy.ndarray`): 2D grid of energy offsets [eV]
   - ``hamilton_2D`` (:class:`numpy.ndarray`): the 2D Hamiltonian [eV]

   plus every key of ``hamilton_to_density_kwargs`` (``density_modifier`` and
   ``hamilton_max`` are only the parameters of the default function), and
   return a non-negative 2D array of the same shape representing the density
   distribution, see also the Notes of :class:`SemiEmpiricMatcher` below.
   Usually, the bunch length or the emittance is known rather than a cut in
   :math:`H`, see the next section.


4. **Particle population**

   The beam’s macro-particles are populated in :math:`(t, \Delta E)` coordinates
   according to the computed density map via
   :func:`~blond.experimental.beam_preparation.semi_empiric_matcher.populate_beam`.

   **Correction to the one-turn map** (``correct_to_one_turn_map=True``,
   opt-in; the default ``False`` keeps the results of existing scripts). The contours of :math:`H` are upright ellipses near the bucket
   centre, but the tracked particles keep the invariant ellipse of the
   *discrete* one-turn map: it is tilted, because kicks and drifts happen one
   after the other, and its energy spread differs by
   :math:`\mathcal{O}(\mu^2)`, :math:`\mu` being the synchrotron phase advance
   per turn. Without correction, the bunch length oscillates by a fraction of
   :math:`\mu` unless the RF kicks sit in the middle of the drifts.

   Five probe particles around the bucket centre are tracked for one turn by
   :func:`get_one_turn_map_empiric`,
   which returns the fixed point and the Jacobian :math:`M` of the one-turn
   map. Any drift formula, multi-harmonic RF and several RF stations per turn
   are therefore included. The grid on which the density is sampled is then
   mapped by :func:`get_grid_correction_from_one_turn_map`,

   .. math::

      \Delta E \to s_E\,\Delta E, \qquad t \to t + s_t\,\Delta E,
      \qquad s_t = \frac{M_{11} - M_{22}}{2 M_{21}},
      \qquad s_E = \gamma\,\beta_H,

   with the Courant-Snyder :math:`\gamma = -M_{21} / \sin\mu` and the aspect
   ratio :math:`\beta_H = \sqrt{a / V''}` of the contours of
   :math:`H = a \Delta E^2 / 2 + V(t)`. Both keep the bunch length at
   :math:`\Delta E = 0`, which comes from the empirical potential, and are the
   same for every bucket. The correction is linear: large-amplitude contours
   keep a residual mismatch.

5. **Iterative matching with intensity effects**

   - Initially, all **intensity effects** (wakefields, profiles) are turned off.
   - The beam is matched in this unperturbed potential.
   - Then, the full **intensity effects** are gradually enabled over several iterations.
     At each iteration:

       a. The beam intensity is scaled by a factor :math:`s_i`:

          .. math::

             s_i = \begin{cases}
                   i / N_\mathrm{ramp} & \text{if } i < N_\mathrm{ramp} \\
                   1.0 & \text{otherwise}
                   \end{cases}

          where :math:`N_\mathrm{ramp}` is ``increment_intensity_effects_until_iteration_i``.

       b. The simulation runs for **one turn** with the current beam to generate
          updated beam profiles that drive wakefield effects.

       c. The profiles are then frozen, and the potential well is re-extracted
          empirically.

       d. To reduce noise, the potential well is averaged with the previous iteration:

          .. math::

             V_\mathrm{avg}(t) = \frac{V_i(t) + V_{i-1}(t)}{2}

       e. A new beam distribution is generated using the averaged potential.

       f. Convergence is tested using the normalized RMS error:

          .. math::

             \epsilon = \sqrt{\frac{1}{N} \sum_j \left(
                        \frac{V_i(t_j)}{\max V_i} - \frac{V_{i-1}(t_j)}{\max V_{i-1}}
                        \right)^2}

   The iteration stops when :math:`\epsilon < \mathrm{tolerance}` **and** the
   intensity has reached full strength (:math:`i > N_\mathrm{ramp}`), or
   when :math:`\epsilon` has not reached a new minimum for
   ``stagnation_patience`` iterations (the beam of the best iteration is
   restored). If ``maxiter_intensity_effects`` ends the iteration first, the
   matcher warns that it did not converge.

---

Bunch Length and Emittance Targets
----------------------------------

:func:`~blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_bunch_length`
and
:func:`~blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_emittance`
fill the bucket with the binomial density

.. math::

   \rho = \left( 1 - \frac{H - H_\mathrm{bottom}}{H_\mathrm{cut}}
   \right)^\mathrm{exponent}

inside the separatrix (``distribution="binomial"``, the default), or
:math:`\rho = \exp(-(H - H_\mathrm{bottom}) / H_0)` cut at the separatrix
(``distribution="gaussian"``, Gaussian in time and energy in a linear
bucket), and solve for :math:`H_\mathrm{cut}` or :math:`H_0` (with
:func:`scipy.optimize.brentq`) such that the bunch has the target
``bunch_length_4sigma`` (four times the RMS length of the line density, in
[s]) or ``emittance_4pi_rms`` (:math:`4 \pi` times the RMS emittance, in
[eVs]):

.. code-block:: python

   from blond.experimental import hamilton_to_density_by_bunch_length

   SemiEmpiricMatcher(
       time_limit=(0, 2.5e-9),
       n_macroparticles=1e6,
       hamilton_to_density_function=hamilton_to_density_by_bunch_length,
       hamilton_to_density_kwargs=dict(bunch_length_4sigma=1e-9, exponent=2),
   )

The bucket is the one around the minimum of the potential well, up to the
lower of the highest barriers on its two sides; a single ``time_limit``
should hold one bucket, several buckets need one window each (next
section). With windows, the targets can differ per bucket, see
``blond/examples/scripts/EX_08_Semi_empiric_matcher.py`` (emittances) and
``blond/examples/scripts/EX_31_Multibunch_semi_empiric_matcher.py`` (bunch
lengths). Any other metric that grows with the bunch, e.g. the RMS energy
spread, can be fitted with
:func:`~blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_metric`.

**Intensity effects.** The density function is called in every iteration,
on the potential well of that iteration, so the target holds for the
matched beam, not only for the bucket without intensity effects.

**Infeasible targets.** A target larger than a full bucket holds gives the
full bucket, cut at the separatrix: the longest bunch, or the largest
emittance, possible with this ``exponent`` (a smaller exponent gives more),
with a warning that names the value reached. With intensity effects, this
is the full bucket of the matched potential well. A target too small for the
grid gives the smallest bunch the grid resolves, with a warning to increase
``internal_grid_shape``. While the intensity is ramped up, a target may not
fit in some iterations and fit in others; the matcher therefore reports the
warnings of the density function only for the final beam.

---

Filling Selected Buckets (Windows)
----------------------------------

With intensity effects, every bucket moves and changes its depth while the
matcher iterates: beam loading shifts the synchronous phase, and in a train
each bunch sees a different induced voltage. A single ``time_limit`` that
spans exactly one bucket then cuts the moved bucket, and a ``time_limit``
over many buckets lets the absolute ``hamilton_max`` fill the buckets
depending on their offsets. Instead, ``time_limit`` can be a list with one
window per bucket to fill. This also allows sparse filling, e.g. a few
bunches of the LHC, without computing the potential well of the whole
ring.

Each window is centred on the nominal centre of its bucket and spans
:math:`\pm T_\mathrm{rf}` around it, :math:`T_\mathrm{rf}` being the period
of the fundamental RF harmonic:

.. code-block:: python

   t_rf = t_rev / harmonic
   buckets = [4, 5, 6, 7]  # bucket k is centred at (k + 0.5) * t_rf here
   time_limit = [((k - 0.5) * t_rf, (k + 1.5) * t_rf) for k in buckets]

A complete example with beam loading, per-bucket weights and per-bucket
density arguments is ``blond/examples/scripts/EX_31_Multibunch_semi_empiric_matcher.py``,
shown in the ``Examples`` section of :class:`SemiEmpiricMatcher`.

In every iteration, each window is treated on its own:

1. Its potential well is obtained as for a single ``time_limit`` and
   minimum-subtracted, so :math:`H = 0` at the bottom of every bucket and
   ``hamilton_max`` means the same depth everywhere. Offsets between
   windows have no meaning.
2. :func:`~blond.experimental.beam_preparation.bucket_windows.select_bucket` picks the bucket to fill. Among the buckets found
   by :class:`~blond.acc_math.empiric.potential_well.PotentialWellHelper`,
   it drops the partial ones at the window edges and the ones nested in
   another (the inner wells of e.g. a double-harmonic bucket, which is
   filled as a whole), and takes the one nearest to the window centre. The
   bucket reaches from its lower barrier to where the potential is as high
   again, which is the separatrix also for an accelerating bucket.
3. The Hamiltonian grid is cut to that bucket, its energy range follows
   from the bucket depth, and the density from ``hamilton_to_density_function``
   is set to zero above the separatrix (with a warning if more than 1 %
   is removed).
4. The bucket is populated with its share ``bucket_weights`` of
   ``n_macroparticles``.

The potential wells of all windows are concatenated for the convergence
error, the averaging and the roll-back, which work as for a single
``time_limit``. ``hamilton_to_density_kwargs`` can be one dict per window,
e.g. to fill the buckets differently.

A bucket may move by up to half a bucket. Beyond that, it cannot be told
apart from its neighbour (a shift of :math:`+0.55` buckets looks like
:math:`-0.45`). The matcher therefore raises a
:class:`~blond.generals.exceptions_.SemiEmpiricMatcherError` if a bucket is
more than half a bucket away from its window centre, if it jumped between
two iterations (:func:`~blond.experimental.beam_preparation.bucket_windows.check_bucket_continuity`), if two windows select
overlapping buckets, or if a window holds no complete bucket. Typically,
this means that there is no static equilibrium: e.g. the beam loading needs
more voltage than the RF can restore.

The potential well is what the tracked probe particles see. Where a window
reaches beyond the time range of a wakefield's profile, there is no induced
voltage, exactly as for particles that are tracked there.

---

Practical Notes
---------------

- The process accounts for both **RF nonlinearities** and **collective effects**.
- Smooth convergence is often achieved by ramping up intensity effects slowly
  (`increment_intensity_effects_until_iteration_i`).
- The beam shape (bunch length, energy spread) is automatically adjusted to the
  matched potential.
- **Matched is not stable.** The matcher finds a static equilibrium of the
  beam with its own intensity effects. Whether it is dynamically stable,
  e.g. against Robinson or coupled-bunch instabilities, is not checked.
  Close to such limits, track the matched beam: an instability grows out
  of the macroparticle noise at the same rate for every seed and number of
  macroparticles, while a residual mismatch keeps its initial amplitude.

---

API Reference
-------------

.. autofunction:: hamilton_to_density_by_max

.. autofunction:: blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_bunch_length

.. autofunction:: blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_emittance

.. autofunction:: blond.experimental.beam_preparation.bucket_filler_functions.hamilton_to_density_by_metric

.. autoclass:: blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.classes.ProfileMatcherAddon
   :members: hamilton_to_density_function

.. autofunction:: get_hamilton_semi_analytic

.. autofunction:: get_one_turn_map_empiric

.. autofunction:: get_grid_correction_from_one_turn_map

.. autofunction:: blond.experimental.beam_preparation.bucket_windows.select_bucket

.. autofunction:: blond.experimental.beam_preparation.bucket_windows.check_bucket_continuity

.. autoclass:: blond.experimental.beam_preparation.bucket_windows.BucketWindows
   :members:

.. autoclass:: DebuggingEndpoints

.. autoclass:: SemiEmpiricMatcher
   :members:
   :undoc-members:
   :show-inheritance:

---

Example
-------

**Basic Usage**

.. code-block:: python

   import numpy as np

   from blond import Beam, Simulation, proton
   from blond.experimental.beam_preparation.semi_empiric_matcher import (
       SemiEmpiricMatcher,
   )
   from blond.generals.cupy_.no_cupy_import import copy_to_cpu

   # Assume you have already initialized a simulation with RF systems,
   # impedances, and a ring definition
   sim = Simulation(...)
   beam = Beam(intensity=1e11, particle_type=proton)

   time_limit = (0, 2.5e-9)  # Time window spanning one RF bucket [s]

   # The Hamiltonian is zero at the bottom of the potential well and, when
   # ``time_limit`` spans one bucket, equals the well depth at the
   # separatrix. ``hamilton_max`` is an absolute value in eV, so derive it
   # from the potential well instead of guessing it.
   potential_well, factor, _ = sim.get_potential_well_empiric(
       dt=np.linspace(*time_limit, 1000),
       particle_type=beam.particle_type,
   )
   hamilton_separatrix = float(potential_well.max()) * factor  # [eV]

   sim.prepare_beam(
       beam=beam,
       preparation_routine=SemiEmpiricMatcher(
           time_limit=time_limit,
           n_macroparticles=100_000,          # Number of macro-particles
           hamilton_to_density_kwargs=dict(
               density_modifier=2.0,          # Controls density profile sharpness
               hamilton_max=0.9 * hamilton_separatrix,  # Hamiltonian cutoff [eV]
           ),
           internal_grid_shape=(1023, 1023),  # Resolution of phase space grid
           tolerance_potential_well=1e-6,     # Convergence threshold
           maxiter_intensity_effects=100,     # Max iterations with wakefields
           increment_intensity_effects_until_iteration_i=10,  # Intensity ramp-up steps
           seed=42,                           # For reproducibility
           verbose=True,                      # Print convergence info
           animate=False,                     # Set True for live plotting
       ),
   )

   # After convergence, the beam's dt and dE are distributed
   # according to the self-consistent potential well
   dt = copy_to_cpu(beam.read_partial_dt())
   dE = copy_to_cpu(beam.read_partial_dE())
   print(f"Bunch length (RMS): {dt.std():.3e} s")
   print(f"Energy spread (RMS): {dE.std():.3e} eV")

``hamilton_separatrix`` is computed without intensity effects. Strong
intensity effects deform the potential well during the iterations, so leave
some margin to the separatrix.


**Advanced Usage with Custom Density Function**

.. code-block:: python

   def custom_density_function(
       time_grid, deltaE_grid, hamilton_2D,  # required arguments
       decay, hamilton_max,  # custom arguments
   ):
       """Density decaying exponentially in H, zero beyond ``hamilton_max``."""
       density = np.exp(-hamilton_2D / (decay * hamilton_max))
       density[hamilton_2D > hamilton_max] = 0.0
       return density

   sim.prepare_beam(
       beam=beam,
       preparation_routine=SemiEmpiricMatcher(
           time_limit=time_limit,
           n_macroparticles=100_000,
           hamilton_to_density_function=custom_density_function,
           hamilton_to_density_kwargs=dict(
               decay=0.2,
               hamilton_max=0.9 * hamilton_separatrix,
           ),
           internal_grid_shape=(1023, 1023),
           tolerance_potential_well=1e-6,
           verbose=True,
       ),
   )

---

Interpretation of Results
-------------------------

After convergence:

- The beam’s **line density** matches the potential well’s curvature.
- The **Hamiltonian contours** correspond to the observed bunch boundaries.
- The routine’s iterative refinement ensures that **space-charge distortions** and
  **wakefield effects** are naturally included in the final matched state.

This makes :class:`SemiEmpiricMatcher` a robust tool for preparing realistic,
self-consistent longitudinal distributions for high-intensity beam simulations.
