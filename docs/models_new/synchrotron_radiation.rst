.. _synchrotron_radiation:

Synchrotron Radiation
=====================

.. currentmodule:: blond.physics.synchrotron_radiation.synchrotron_radiation

Overview
--------

The :class:`SynchrotronRadiationMaster` class provides a framework for including
**synchrotron radiation damping** and **quantum excitation** effects in
longitudinal beam dynamics simulations in synchrotrons.

Bending forces applied on relativistic charged particles yields to the
spontaneous emission of photons, called **synchrotron radiation**. This
emission causes:

1. **Energy loss per turn** -- particles radially lose energy,
2. **Radiation damping** -- oscillation amplitudes decrease exponentially

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

   I_1 &= \oint \frac{D_x}{\rho} \, ds \\
   I_2 &= \oint \frac{1}{\rho^2} \, ds \\
   I_3 &= \oint \frac{1}{|\rho|^3} \, ds \\
   I_4 &= \oint \frac{D_x}{\rho} \left(\frac{1}{rho^2} + 2 * K\right) \, ds \\
   I_5 &= \oint \frac{\mathcal{H}}{|\rho|^3} \, ds \\

where:

- :math:`\rho` is the bending radius [m],
- :math:`D_x` is the horizontal dispersion function [m],
- :math:`K` is the focusing strength [m\ :sup:`-2`],
- :math:`\mathcal{H} = \beta_x D_x'^2 + 2\alpha_x D_x D_x' + \gamma_x D_x^2`
  is the H-function [m].

For an **isomagnetic ring** (uniform bending radius :math:`\rho_0`), the
integrals simplify to:

.. math::

   I_1 &= \alpha_c \cdot C, related to the momentum compaction factor, \\
   I_2 &= \frac{2\pi}{\rho_0}, related to the energy loss per turn, \\
   I_3 &= \frac{2\pi}{\rho_0^2}, related to the natural energy spread, \\
   I_4 &= \frac{\alpha_c \cdot C}{\rho_0^2}, required for the damping times, \\
   I_5 &= 0, for lack of information on :math: <\mathcal{H}>, required for the natural horizontal emittance \\

where :math:`\alpha_c` is the momentum compaction factor and :math:`C` is the
circumference.

Synchrotron motion damping and damping times
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For highly relativistic charged particles, synchrotron radiation is emitted
along its direction of motion, which recoil induces small perturbation of
the betatron and synchrotron motion. This effect damps the beam
amplitudes, with typical damping times.
The radiation loss around the synchronous partilcle can be derived as :math: 'U
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
coefficient are the **damping partition numbers**:
.. math::

   j_x &= 1 - \frac{I_4}{I_2} \\
   j_y &= 1 \\
   j_z &= 2 + \frac{I_4}{I_2}'

assuming no vertical dispersion in the synchrotron. The Robinson damping
theorem requires :math:`j_x + j_y + j_z = 4`.


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
   - U_0 \\
   - \frac{2}{\tau_z} \Delta E \\
   + \sqrt{\frac{2 \sigma_E \E_0}{\tau_z}} \cdot \mathcal{N}(0,1)

where :math:`\mathcal{N}(0,1)` represents a standard normal random distribution.
---

Module Structure
----------------

The synchrotron radiation framework consists of:

**Base Class**

- :class: 'SynchrotronRadiationBaseClass': it holds basic properties
 for tracking with synchrotron radiation, computes the energy kick given to the
  beam and updates the beam energy accordingly druing the simulation.

**Master Class**

- :class:`SynchrotronRadiationMaster`:  Orchestrates the creation and
  insertion of synchrotron radiation trackers into the ring

**Tracker Elements**

- :class:`~blond.physics.synchrotron_radiation.synchrotron_radiation_elements.SynchrotronRadiationBaseClass`
  -- Abstract base class for all synchrotron radiation elements
- ``_SynchrotronRadiationTracker`` -- Tracker inserted before drift elements
and after RF cavities

**Special Elements**

- :class:`~blond.physics.synchrotron_radiation.synchrotron_radiation_elements.WigglerMagnet`
  -- Models damping wigglers that enhance synchrotron radiation

---

Algorithmic Workflow
--------------------

The :meth:`SynchrotronRadiationMaster.prepare_ring_for_synchrotron_radiation_tracking`
method performs the following steps:

1. **Set synchrotron radiation integrals**

   The radiation integrals are obtained from one of:

   - The ``Ring`` object (if pre-defined)
   - User-provided array via ``radiation_integrals`` parameter
   - Computed for an isomagnetic ring using ``bending_radius``

2. **Identify tracking locations**

   Based on ``track_before_element_type``, the method locates either:

   - All ``DriftBaseClass`` elements, or
   - All ``RFStationBaseClass`` elements

3. **Calculate local radiation shares**

   For each element, the share of radiation integrals is computed
   proportionally to the element's orbit length relative to the circumference:

   .. math::

      I_k^{(\text{element})} = \frac{L_{\text{element}}}{C} \cdot I_k

4. **Create and insert trackers**

   - For drifts: ``_SynchrotronRadiationTracker`` elements are inserted
     **before** each drift
   - For RF cavities: ``_SynchrotronRadiationTracker`` elements are inserted
     **after** each cavity

5. **Runtime tracking**

   During simulation, each tracker's :meth:`track` method:

   a. Computes local synchrotron radiation parameters from current beam energy
   b. Calculates energy kicks including damping and quantum excitation
   c. Updates particle energy deviations

---

Practical Notes
---------------

- **Isomagnetic approximation**: For simple rings with uniform bending, use
  the ``bending_radius`` parameter instead of providing all five integrals.

- **Drift-based vs. section-based tracking**: Drift-based tracking
  (default) provides finer granularity, applying radiation effects at each
  drift. Section-based tracking applies cumulative effects after each RF
  station.

- **Quantum excitation**: Can be disabled via ``disable_quantum_excitation=True``
  to study pure radiation damping.

- **Wiggler magnets**: Use :class:`WigglerMagnet` to add damping wigglers
  that modify the effective radiation integrals in an energy-dependent manner.

- **Energy dependence**: All synchrotron radiation parameters
  (:math:`U_0`, :math:`\tau_z`, :math:`\sigma_E`) are recomputed at each
  tracking step using the current beam reference energy, enabling correct
  behavior during acceleration.

---

API Reference
-------------

Master Class
^^^^^^^^^^^^

.. autoclass:: SynchrotronRadiationMaster
   :members:
   :undoc-members:
   :show-inheritance:

Base Element Class
^^^^^^^^^^^^^^^^^^

.. autoclass:: blond.physics.synchrotron_radiation.synchrotron_radiation_elements.SynchrotronRadiationBaseClass
   :members:
   :undoc-members:
   :show-inheritance:

Wiggler Magnet
^^^^^^^^^^^^^^

.. autoclass:: blond.physics.synchrotron_radiation.synchrotron_radiation_elements.WigglerMagnet
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: blond.acc_math.analytic.synchrotron_radiation.utilities.gather_longitudinal_synchrotron_radiation_parameters

.. autofunction:: blond.acc_math.analytic.synchrotron_radiation.utilities.calculate_isomagnetic_radiation_integrals

.. autofunction:: blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths.calculate_energy_loss_per_turn

.. autofunction:: blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths.calculate_longitudinal_damping_time_in_turns

.. autofunction:: blond.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths.calculate_natural_energy_spread

---

Example
-------

**Basic Usage with Explicit Radiation Integrals**

.. code-block:: python

   import numpy as np

   from blond import Ring, SynchrotronRadiationMaster
   from blond.physics.drifts import DriftBaseClass

   # Define synchrotron radiation integrals for the ring
   # [I1, I2, I3, I4, I5] - typically obtained from lattice calculations
   radiation_integrals = np.array([
       0.646747216157,      # I1: related to momentum compaction
       0.0005936549319,     # I2: related to energy loss per turn
       5.6814536525e-08,    # I3: related to natural energy spread
       5.92870407301e-09,   # I4: required for damping times
       1.71368060083e-11,   # I5: required for natural emittance
   ])

   # Create ring with radiation integrals
   ring = Ring(
       circumference=844.0,
       synchrotron_radiation_integrals=radiation_integrals,
   )

   # Add drift spaces (required for drift-based SR tracking)
   ring.add_drifts(n_drifts_per_section=10, n_sections=1)

   # Initialize synchrotron radiation master
   sr_master = SynchrotronRadiationMaster(
       track_before_element_type=[DriftBaseClass],
       disable_quantum_excitation=False,
   )

   # Prepare ring for SR tracking (inserts tracker elements)
   sr_master.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)

   print(f"Created {sr_master.number_of_generated_synchrotron_radiation_classes} "
         f"SR tracker elements")


**Isomagnetic Ring Approximation**

.. code-block:: python

   from blond import Ring, SynchrotronRadiationMaster
   from blond.physics.drifts import DriftBaseClass

   # Create ring without explicit radiation integrals
   ring = Ring(
       circumference=844.0,
       momentum_compaction_factor=1.78e-4,
   )
   ring.add_drifts(n_drifts_per_section=10, n_sections=1)

   # Initialize SR master
   sr_master = SynchrotronRadiationMaster()

   # Use isomagnetic approximation with bending radius
   sr_master.prepare_ring_for_synchrotron_radiation_tracking(
       ring=ring,
       bending_radius=25.0,  # Average bending radius [m]
   )


**Adding Damping Wigglers**

.. code-block:: python

   from blond import Ring
   from blond.physics.synchrotron_radiation import WigglerMagnet

   ring = Ring(circumference=844.0)

   # Add a damping wiggler to the ring
   wiggler = WigglerMagnet(
       name="DampingWiggler_1",
       section_index=0,
       wiggler_type="sinusoidal",
       number_of_wigglers=2,      # Two identical wigglers
       peak_field=1.8,            # Peak magnetic field [T]
       pole_length=0.095,         # Pole length [m]
       number_of_poles=43,        # Poles per wiggler
   )

   ring.add_element(wiggler)

   # The wiggler will automatically update radiation integrals
   # based on beam energy during tracking


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

- The **beam energy spread** will converge to the natural energy spread
  :math:`\sigma_E` over a timescale of :math:`\tau_z` turns.

- The **bunch length** will adjust according to the relationship between
  energy spread and the RF bucket parameters.

- Without RF voltage compensation, the **synchronous phase** will shift to
  account for the mean energy loss per turn.

- For ultra-relativistic electrons, radiation effects are typically dominant;
  for protons and heavier ions, synchrotron radiation is negligible at most
  energies but becomes relevant in machines like the LHC at top energy.

---

References
----------

[1] S.Y. Lee, *Accelerator Physics*, World Scientific, Third Edition, 2012
[2] H. Wiedemann, *Synchrotron Radiation*, Springer, 2003
[3] A. Wolski, *Introduction to Beam Dynamics in High-Energy Electron Storage
  Rings*, Morgan & Claypool Publishers, 2018
