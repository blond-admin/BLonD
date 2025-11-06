.. _semi_empiric_matcher:

Semi-Empirical Beam Matching
============================

.. currentmodule:: blond.experimental.beam_preparation.semi_empiric_matcher

Overview
--------

The :class:`SemiEmpiricMatcher` class provides a **semi-empirical longitudinal
beam-matching procedure** that combines an *empirically obtained potential well*
from :func:`~blond._core.simulation.simulation.Simulation.get_potential_well_empiric`
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

   H(t, \Delta E) = \frac{1}{2} \frac{\eta E_0}{\beta^2 c^2} (\Delta E)^2 + V(t)

where:

- :math:`t` is the arrival time relative to the synchronous particle,
- :math:`\Delta E` is the energy deviation from the synchronous energy,
- :math:`E_0` is the reference total energy,
- :math:`\eta` is the slippage factor,
- :math:`\beta` is the relativistic velocity fraction,
- :math:`V(t)` is the **empirical potential well** obtained from the simulation.

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

   Using :func:`Simulation.get_potential_well_empiric`, the empirical potential
   well is obtained by tracking a *probe beam* for one turn through the full
   RF configuration. The output potential includes all RF phase shifts and
   wakefield effects.

2. **Hamiltonian reconstruction**

   The semi-analytic Hamiltonian is computed using
   :func:`get_hamilton_semi_analytic`, combining the empirical potential
   :math:`V(t)` with an analytic drift term based on :math:`\eta`, :math:`\beta`,
   and the reference energy :math:`E_0`.

3. **Density mapping**

   The function :func:`hamilton_to_density_by_max` transforms the Hamiltonian
   into a density map:

   .. math::

      \rho(t, \Delta E)
      = \left( 1 - \frac{H(t, \Delta E)}{H_\text{max}} \right)^{n}

   where ``n`` = ``density_modifier`` controls how sharply the density
   decreases toward the separatrix. This density defines how macro-particles
   are distributed in phase space.

4. **Particle population**

   The beam’s macro-particles are populated in :math:`(t, \Delta E)` coordinates
   according to the computed density map via
   :func:`~blond.experimental.beam_preparation.semi_empiric_matcher.populate_beam`.

5. **Iterative matching with intensity effects**

   - Initially, all **intensity effects** (wakefields, profiles) are turned off.
   - The beam is matched in this unperturbed potential.
   - Then, the full **intensity effects** are gradually enabled over several iterations.
     At each iteration:
       - The potential well is re-measured.
       - A new beam distribution is reconstructed.
       - Convergence is tested based on RMS error between successive potential wells.

   The iteration stops once the potential stabilizes within the specified tolerance.

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

.. code-block:: python

   from blond._core.simulation.simulation import Simulation
   from blond.experimental.beam_preparation.semi_empiric_matcher import SemiEmpiricMatcher
   from blond._core.particles import ParticleType

   # Initialize simulation and beam
   sim = Simulation(...)
   beam = sim.beams[0]

   # Define matching routine
   matcher = SemiEmpiricMatcher(
       time_limit=(-2e-9, 2e-9),
       n_macroparticles=100_000,
       hamilton_to_density_kwargs=dict(density_modifier=2.0, hamilton_max=1.0),
       internal_grid_shape=(1023, 1023),
       tolerance=1e-6,
       verbose=True,
       animate=True,  # optional live visualization
   )

   # Perform beam matching
   matcher.prepare_beam(sim, beam)

   # After convergence, beam.dt and beam.dE are distributed
   # according to the self-consistent potential well.
   print("Matching complete. Beam intensity:", beam.intensity)

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
