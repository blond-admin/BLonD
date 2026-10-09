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
   distribution. A complete, runnable example is in the ``Examples`` section
   of :class:`SemiEmpiricMatcher` below.


4. **Particle population**

   The beam’s macro-particles are populated in :math:`(t, \Delta E)` coordinates
   according to the computed density map via
   :func:`~blond.experimental.beam_preparation.semi_empiric_matcher.populate_beam`.

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
   intensity has reached full strength (:math:`i > N_\mathrm{ramp}`).

---

Practical Notes
---------------

- The process accounts for both **RF nonlinearities** and **collective effects**.
- Smooth convergence is often achieved by ramping up intensity effects slowly
  (`increment_intensity_effects_until_iteration_i`).
- The beam shape (bunch length, energy spread) is automatically adjusted to the
  matched potential.

---

API Reference
-------------

.. autofunction:: hamilton_to_density_by_max

.. autofunction:: get_hamilton_semi_analytic

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
