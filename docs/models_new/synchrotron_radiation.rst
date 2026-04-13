.. _synchrotron_radiation:

Synchrotron Radiation
=====================

.. currentmodule:: blond.physics.synchrotron_radiation.synchrotron_radiation_master

Overview
--------

The :class:`SynchrotronRadiationMaster` class provides a framework for including
**synchrotron radiation damping** and **quantum excitation** effects whilst
simulating longitudinal beam dynamics in synchrotrons.

Bending forces applied on relativistic charged particles triggers the emission
 of photons, called **synchrotron radiation**. This emission causes:

1. **Energy loss per turn** -- particles radially lose energy,
2. **Radiation damping** -- oscillation amplitudes decrease exponentially.

Additionally, discrete photon emission can happen randomly during the charged
particle trajectory. This **quantum excitation** provokes stochastic energy
fluctuations of the beam particles.

The interplay between damping and excitation leads to the natural beam sizes,
characterized by the **natural energy spread** and **natural
bunch length** in the longitudinal plane.
---

Conceptual Background
---------------------

Energy Loss Per Turn
^^^^^^^^^^^^^^^^^^^^
For highly relativistic particles, the power is emitted radially [2]. The
instantaneous radiated power is derived in [2]:
.. math::

    P_{\gamma} = \frac{c \cdot C_{\gamma}}{2 \pi} \frac{E^4}{\rho *2},

where :math:`C_\gamma = \frac{4 \pi}{3} \frac{r_c}{(m  c^2)^3}` is the Sands
radiation constant (particle-dependent), :math: 'r_c' the classical radius,
:math: 'c' the speed of light, :math: 'E' the particle energy, :math: '\rho'
the bending radius.

Integrating the instantaneously radiated power along the synchrotron's
circumference, we obtain the energy loss per turn due to synchrotron radiation,
to be compensated for beam storage:

.. math::

   U_0 = \frac{C_\gamma}{2\pi} E^4 \oint \frac{1}{\rho^2} \, ds \\
   or \\
   U_0 = \frac{C_\gamma}{2\pi} E^4 I_2, ds \\

where :math:`C_\gamma` is the Sands radiation constant [m/(eV)^3],
:math:`E` is the beam energy [eV], :math:`I_2` is the second radiation
integral.

Synchrotron Radiation Integrals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The properties of the electron beam in synchrotrons are summarized in the
**synchrotron radiation integrals**[1-3]:

.. math::

   I_1 &= \oint \frac{D_x}{\rho} \, ds, related to the momentum compaction factor, \\
   I_2 &= \oint \frac{1}{\rho^2} \, ds, related to the energy loss per turn, \\
   I_3 &= \oint \frac{1}{|\rho|^3} \, ds, related to the natural energy spread, \\
   I_4 &= \oint \frac{D_x}{\rho} \left(\frac{1}{rho^2} + 2 * K\right) \, ds, required for the damping times, \\
   I_5 &= \oint \frac{\mathcal{H}}{|\rho|^3} \, ds, required for the natural horizontal emittance \\

where:

- :math:`\rho` is the bending radius [m],
- :math:`D_x` is the horizontal dispersion function [m],
- :math:`K` is the focusing strength [m\ :sup:`-2`],
- :math:`\mathcal{H} = \beta_x D_x'^2 + 2\alpha_x D_x D_x' + \gamma_x D_x^2`
  is the :math:`\mathcal{H}'-function [m].

For an **isomagnetic ring** (uniform bending radius :math:`\rho_0`), the
integrals simplify to:

.. math::

   I_1 &= \alpha_c \cdot C, \\
   I_2 &= \frac{2\pi}{\rho_0}, \
   I_3 &= \frac{2\pi}{\rho_0^2}, \\
   I_4 &= \frac{\alpha_c \cdot C}{\rho_0^2},\\
   I_5 &= 0, for lack of information on :math: <\mathcal{H}>,\\

where :math:`\alpha_c` is the momentum compaction factor and :math:`C` is the
circumference.

Synchrotron motion damping and damping times
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For highly relativistic charged particles, synchrotron radiation is emitted
along its direction of motion, which recoil induces small perturbation of
the betatron and synchrotron motion. This effect damps the beam
amplitudes, with typical damping times.
The radiation loss around the synchronous particle can be derived as :math: 'U
= U_0 + \frac{dU}{dE}_{E = E_0)} \cdot
\Delta E', where :math: 'E_0' and 'U_0' are the energy and energy lost of the
synchronous particle.
A particle with an energy 'E_0 + \Delta E' circulates on a different orbit
than the synchronous particle, resulting in a different path length :math: 'C
+ \Delta C' after a turn, characterised by the momentum compaction factor:
.. :math:

    \frac{\Delta C}{C} = \alpha_C \frac{\Delta E }{E_0}

This variation in the path length translates into an arrival time difference
:math: '\Delta tau =
\alpha_C
\cdot T_0 \codt \frac{\Delta E}{E_0}'.
where :math: 'T_0' is the revolution period.

The full synchrotron motion including damping is described by a harmonic
oscillator:
 .. :math:
    \frac{d^2 \tau}{dt^2} + \frac{2}{\tau_z} \frac{d \tau}{dt} + \omega_s^2
    \tau = 0

with the **longitudinal damping time**  in seconds is:
 .. math::

   \tau_z = \frac{2 E}{j_z \cdot U_0} T_0

Practically, the synchrotron radiation damping times of all planes are
proportional to the
inverse of :math: 'U_0 / (2 T_0 E)'. The proportionality
coefficient are the **damping partition numbers**, which represent how damping
is distributed amongst the planes:
.. math::

   j_x &= 1 - \frac{I_4}{I_2} \\
   j_y &= 1 \\
   j_z &= 2 + \frac{I_4}{I_2}'

assuming no vertical dispersion in the synchrotron. The Robinson damping
theorem requires :math:`j_x + j_y + j_z = 4`/

Quantum Excitation and Natural Energy Spread
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spontaneous emission of a quanta yields to an immediate energy change and
random small energy oscillations which tend to blow the beam sizes. **Quantum
excitation** and synchrotron **radiation damping** combined define a natural
equilibrium state of transverse and longitudinal beam emittances.
The **natural energy spread** is:
.. math::

   \sigma_E = \sqrt{C_q \left(\frac{E}{m_0 c^2}\right)^2 \frac{I_3}{j_z I_2}} \cdot E

where :math:`C_q` is the quantum radiation constant and :math:`m_0` is the
particle rest mass.

Tracking with synchrotron radiation and quantum excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
At each tracking step, the energy deviation :math:`\Delta E` of each particle
is updated taking into account:

1. the energy lost by the synchronous particle along the turn :math: 'U_0',
2. the effect of radiation damping with :math: 'tau_z' the damping time,
3. the effect of quantum excitation, with :math: '\sigma_E' the natural energy
spread.

The effective energy kick provided by the synchrotron radiation tracker is:
.. math::

   \Delta E \rightarrow \Delta E \\
   - U_0 (energy loss per turn)\\
   - \frac{2}{\tau_z} \Delta E (radiation damping)\\
   + \sqrt{\frac{2 \sigma_E \E_0}{\tau_z}} \cdot \mathcal{N}(0,1) (quantum
   excitation), \\

where :math:`\mathcal{N}(0,1)` represents a standard normal random distribution.
---

Module Structure
----------------

The synchrotron radiation framework consists of:

**Base Class**

- :class: 'SynchrotronRadiationBaseClass': abstract class holding basic
properties for tracking with synchrotron radiation, computes the energy kick
given to the beam and updates the beam energy accordingly during the
simulation.

**Master Class**

- :class:`SynchrotronRadiationMaster`:  this object creates and inserts
synchrotron radiation trackers into the ring.

- :class: '_SynchrotronRadiationTracker': internal tracker called by the
:class: 'SynchrotronRadiationMaster'. Trackers are inserted before drift
elements and after RF cavities.

Algorithmic Workflow
--------------------

The :meth:`SynchrotronRadiationMaster.prepare_ring_for_synchrotron_radiation_tracking`
method performs the following steps:

1. **Set the synchrotron radiation integrals**

   The radiation integrals are obtained from one of:

   - The ``Ring`` object (if pre-defined),
   - Computed for an isomagnetic ring using ``bending_radius``.

For consistency, the radiation integrals obtained outside the ring will be set
as a property of the ring.

2. **Identify tracking locations**

   Based on ``track_before_element_type``, the method locates either:

   - All ``DriftBaseClass`` elements, or
   - All ``RFStationBaseClass`` elements

3. **Calculate local radiation shares**

   For each element, the share of radiation integrals is computed
   proportionally to :
   - if all drifts hold this property, use the provided radiation integrals
   of each drift,
   - its orbit length relative to the circumference for drift elements,
   - the section length between each RF cavities relative to the
   circumference,

   .. math::

      I_k^{(\text{element})} = \frac{L_{\text{element}}}{C} \cdot I_k

4. **Create and insert trackers**

Then, ``_SynchrotronRadiationTracker`` elements are inserted:

   - **before** each drift,
   - **after** each cavity.

5. **Runtime tracking**

   During the simulation, each tracker's :meth:`track` method:

   a. Computes current synchrotron radiation parameters from current beam
   energy, namely the estimated energy lost per turn, longitudinal damping time
    and natural energy spread,
   b. Calculates the energy kick (as described above), including radiation
   damping and quantum excitation,
   c. Updates the bean relative energy array accordingly.

---

Example
-------

**Example with Radiation Integrals**

.. code-block:: python

   import numpy as np

   from blond import Ring, SynchrotronRadiationMaster
   from blond.physics.drifts import DriftBaseClass

   # Define synchrotron radiation integrals for the ring
   # [I1, I2, I3, I4, I5] - obtained from the lattice optics.
   radiation_integrals = np.array([
       0.646747216157,
       0.0005936549319,
       5.6814536525e-08,
       5.92870407301e-09,
       1.71368060083e-11,
   ])

   # Create ring with radiation integrals
   ring = Ring(
       circumference=90.65874532 * 1e3,
       synchrotron_radiation_integrals=radiation_integrals,
   )
   # Creates an RF station
    cavity = SingleHarmonicRFStation()
    cavity.harmonic = 242400
    cavity.voltage = 50.1e6
    cavity.phi_rf_design = 0
    ring.add_element(cavity)

   # Add drift spaces
   ring.add_drifts(n_drifts_per_section=10, n_sections=1)

   # Initialize synchrotron radiation master
   sr_master = SynchrotronRadiationMaster(
       track_before_element_type=[DriftBaseClass],
   )

   # Prepare ring for tracking with synchrotron radiation
   sr_master.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)

   print(f"Created {sr_master.number_of_generated_synchrotron_radiation_classes} "
         f"SR tracker elements")

**Isomagnetic Ring Approximation**

.. code-block:: python

   from blond import Ring, SynchrotronRadiationMaster
   from blond.physics.drifts import DriftBaseClass

   # Create ring without explicit radiation integrals
   ring = Ring(
       circumference=90.65874532 * 1e3,
       momentum_compaction_factor=1.78e-4,
   )
   # Creates an RF station
    cavity = SingleHarmonicRFStation()
    cavity.harmonic = 242400
    cavity.voltage = 50.1e6
    cavity.phi_rf = 0
    ring.add_element(cavity)

   ring.add_drifts(n_drifts_per_section=10, n_sections=1)

   # Initialize SR master
   sr_master = SynchrotronRadiationMaster()

   # Use isomagnetic approximation with bending radius
   sr_master.prepare_ring_for_synchrotron_radiation_tracking(
       ring=ring,
       bending_radius=25.0,  # Average bending radius [m]
   )


**Computing Synchrotron Radiation Parameters**

.. code-block:: python

   from blond import Ring, Beam
   from blond.physics.synchrotron_radiation import SynchrotronRadiationMaster
   from blond.core.beam.particle_types import electron

   ring = Ring(
       circumference=844.0,
       synchrotron_radiation_integrals=radiation_integrals,
   )

   beam = Beam(
       particle_type=electron,
       n_macroparticles=10000,
       intensity=1e10,
       energy=3e9,  # 3 GeV
   )

   sr_master = SynchrotronRadiationMaster()
   sr_master.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)

   # Compute and display SR parameters for current beam energy
   sr_master.compute_synchrotron_radiation_parameters(beam=beam, ring=ring)

   print(f"Energy loss per turn: {sr_master.energy_loss_per_turn:.3e} eV")
   print(f"Longitudinal damping time: {sr_master.longitudinal_damping_time:.1f} turns")

---

Interpretation of Results
-------------------------

After running a simulation with synchrotron radiation:

- The **beam energy spread** will converge to the natural energy
spread :math:`\sigma_E` over a timescale of :math:`\tau_z` turns.

- The **bunch length** will adjust according to the relationship between
  the energy spread and the RF bucket parameters.

- The **synchronous phase** :math: '\phi_s' shifts to account for the mean
energy loss per turn, according to :math: '\sin(\phi_s) = \frac{U_0}{e \cdot
V}'.

- For ultra-relativistic electrons and positrons, radiation effects are
perceivable; From [2], the ratio between radiated power between electrons and
 protons is :math: '1836^4' (estimating proton's classical radius as the
 electron;s.). Therefore, for protons and heavier ions, the effect of synchrotron radiation
is negligible at most energies but becomes relevant in machines like the LHC at
 top energy.

---

Nota Bene
---------

- **Isomagnetic approximation**: For simple rings with uniform bending, use
  the ``bending_radius`` parameter to compute the isomagnetic radiation integrals.

- **Drift-based vs. section-based tracking**: Drift-based tracking
  (default) provides finer granularity, applying radiation effects before each
  drift. Section-based tracking applies cumulative effects after each RF
  station.

- **Quantum excitation**: Can be disabled via ``disable_quantum_excitation=True``
  to study the effect of radiation damping only.


- **Energy dependence**: All synchrotron radiation parameters
  (:math:`U_0`, :math:`\tau_z`, :math:`\sigma_E`) are recomputed at each
  tracking step using the current beam reference energy, enabling correct
  behavior during the simulation.

---
References
----------

[1] S.Y. Lee, *Accelerator Physics*, World Scientific, Third Edition, 2012
[2] H. Wiedemann, *Synchrotron Radiation*, Springer, 2003
[3] A. Wolski, *Introduction to Beam Dynamics in High-Energy Electron Storage
  Rings*, Morgan & Claypool Publishers, 2018
