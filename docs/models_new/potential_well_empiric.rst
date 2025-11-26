.. _potential_well_empiric:

Empirical Potential Well
========================

.. currentmodule:: blond._core.simulation.simulation

Overview
--------

The *empirical potential well* represents the effective longitudinal voltage landscape
experienced by a charged particle beam during one full revolution of the accelerator.
It is derived directly from a **numerical single-turn tracking** of a *probe beam*
in the :class:`Simulation` environment, including all active RF systems and other
beam relevant effects comprised in the simulation pipeline.

Unlike purely analytic models, this method captures **realistic distortions**
introduced by multiple RF stations, synchrotron phase shifts, and drifts between stations.

The potential well is obtained using the method
:func:`Simulation.get_potential_well_empiric`, which integrates the *energy gain and loss*
experienced by a set of probe particles over one complete turn.

---

Physical Background
-------------------

For a single turn, a set of probe particles is defined with initial time offset :math:`\Delta t_0`
and zero energy deviation :math:`\Delta E_0 = 0`.

As the probe beam drifts between RF stations, its *arrival time* changes.
When it reaches the next RF station, the local phase has shifted accordingly.
This causes apparent **distortion and phase shifts** in the reconstructed potential
because each station sees a slightly different particle phase advance.

At the end of one turn, the difference between the final and initial energy
for each initial time coordinate encodes the total work done on the particle:

.. math::

   \Delta E_\text{out}(\Delta t_0)
   = q \int_0^{T_\text{rev}} V_\text{eff}(t + \Delta t_0) \, dt

The potential well :math:`V(\Delta t)` is then reconstructed by integrating the negative
of the observed energy changes over the time coordinate.

---

Implementation Details
----------------------

Internally, the method performs the following steps:

1. **Probe initialization**

   A :class:`~blond._core.beam.beams.ProbeBeam` is created using the provided
   particle type and intensity. Its macroparticles are placed at specified
   time offsets ``dt`` and have zero energy deviation ``dE = 0``.

2. **Tracking for one turn**

   The :class:`~blond._core.simulation.simulation.Simulation` instance runs
   for exactly one revolution (``n_turns = 1``) with wakefields and drifts enabled.
   Each probe particle passes through all RF stations and drifts, accumulating
   a total energy change :math:`\Delta E_\text{out}` and a small time shift
   due to dispersion.

3. **Computation of revolution parameters**

   - The revolution time ``t_rev`` is determined from the reference time before
     and after the turn.
   - A scaling factor ``factor = (dt[-1] - dt[0]) / t_rev`` relates the
     probed time span to one revolution period.
   - The **phase-space shear**, or ``tilt_dt_per_dE``, is computed as the ratio
     of the time change to the energy change at the point of maximum displacement.

4. **Potential reconstruction**

   The potential well is obtained by integrating the energy change along
   the time coordinate:

   .. math::

      V(\Delta t) = -\int \Delta E_\text{out}(\Delta t) \, d(\Delta t)

   yielding a continuous effective potential that governs the longitudinal motion.

5. **Normalization and polarity**

   - If the ring is below transition energy, the potential is inverted
     (since the phase focusing reverses).
   - Optionally, the minimum potential value is set to zero
     if ``subtract_min=True``.

---

Returned Quantities
-------------------

The function returns a 3-tuple:

``(potential_well, factor, tilt_dt_per_dE)``

+------------------------+--------------------------------------------------------+
| Quantity               | Description                                            |
+========================+========================================================+
| ``potential_well``     | 1D array of the effective longitudinal potential in V  |
+------------------------+--------------------------------------------------------+
| ``factor``             | Time span ratio ``(dt[-1] - dt[0]) / t_rev``          |
+------------------------+--------------------------------------------------------+
| ``tilt_dt_per_dE``     | Phase-space shear: change in time per change in energy |
+------------------------+--------------------------------------------------------+

---

API Reference
-------------

:func:`Simulation.get_potential_well_empiric`

---

Example Usage
-------------

.. code-block:: python

   from blond._core.simulation.simulation import Simulation
   from blond._core.particles import ParticleType
   import numpy as np
   import matplotlib.pyplot as plt

   sim = Simulation(...)
   proton = ParticleType(name="proton")

   dt = np.linspace(-2e-9, 2e-9, 800)
   potential_well, factor, tilt = sim.get_potential_well_empiric(
       dt=dt,
       particle_type=proton,
       subtract_min=True,
       intensity=1e10,
   )

   plt.plot(dt, potential_well)
   plt.xlabel("Time offset [s]")
   plt.ylabel("Potential well [V]")
   plt.title("Empirical Longitudinal Potential")
   plt.grid(True)
   plt.show()

---

Interpretation
--------------

A flat or shallow potential corresponds to weak longitudinal focusing,
while deeper minima indicate stronger RF confinement.

Because this potential includes the cumulative effect of
**all active RF stations and drifts**, it represents the *true potential landscape*
seen by a beam in the simulation — essential for realistic matching
and for understanding bunch shape distortions.
