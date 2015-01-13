trackers package
================

Submodules
----------

trackers.tracker module
-----------------------

.. automodule:: trackers.tracker
    :members:
    :undoc-members:
    :show-inheritance:

Equations of Motion
^^^^^^^^^^^^^^^^^^^
:Authors: **Helga Timko**

.. image:: ring_and_RFstation.png
        :align: right
        :width: 350
        :height: 350

Below, we shall derive the equations of motion (EOMs) for an energy kick given to the particle by the RF caviti(es) of a given RF station and the subsequent drift of the particle during one turn, see Figure. In the case of multiple RF stations, every :math:`2 \pi` factor is to be replaced by the fraction of the ring between the given RF station and the next one :math:`2 \pi \frac{L_i}{C}`, where :math:`C = \sum_{i} L_i` is the machine circumference.

Just like in the real machine, we demand the user to define beforehand the momentum programme, i.e. the ``design (synchronous) momentum`` at every time step :math:`n` and RF station :math:`(i)`, :math:`\left\{ p_{d,(i)}^n \right\}`. This will define the design total energy :math:`E_d` through following relations:

.. math:: E_d = \gamma_d m c^2 ,
   :label: E
.. math:: p_d = \beta_d \gamma_d m c ,
   :label: p
.. math:: E_d^2 = p_d^2 c^2 + m^2 c^4 .
   :label: E-p

First, we update the absolute energy :math:`E^n` of the particle with the total energy kick received from the various RF systems :math:`k` in a given RF station:

.. math:: E^{n+1} = E^{n} + \sum_k eV_k^n \sin \left( \frac{h_k^n}{h_0^n} \varphi^n + \phi_{\mathsf{offset},k}^n \right) .
   :label: 1-1

.. warning:: Here :math:`\varphi \equiv h_0 \vartheta` is defined as the **RF phase of the main RF system when the particle passes the cavity**. The main RF system is the one with index 0 in the ``RFSectionParameters.harmonic`` (:math:`h_0`), ``RFSectionParameters.voltage`` (:math:`V_0`), and ``RFSectionParameters.phi_offset`` (:math:`\phi_{\mathsf{offset},0}`) arrays. The phases of all other RF systems are defined w.r.t. the main RF system. These relative phases and other phase shifts acting on a given RF system can be added through :math:`\phi_{\mathsf{offset},k}`.

.. note:: Eq. :eq:`1-1` is intrinsically discrete; no approximation has been done.

Rather than the absolute energy, we are actually interested in the energy offset of a given particle w.r.t. the design energy :math:`\Delta E^n \equiv E^n - E_d^n, \, \forall n`. Substracting :math:`E_d^{n+1}` from both sides of Eq. :eq:`1-1`,

.. math:: \Delta E^{n+1} = \Delta E^{n} + \sum_k eV_k^n \sin \left( \frac{h_k^n}{h_0^n} \varphi^n + \phi_{\mathsf{offset},k}^n \right) - (E_d^{n+1} - E_d^n) ,
   :label: 1-2

or, in terms of :math:`\vartheta`:

.. math:: \Delta E^{n+1} = \Delta E^{n} + \sum_k eV_k^n \sin \left( h_k^n \vartheta^n + \phi_{\mathsf{offset},k}^n \right) - (E_d^{n+1} - E_d^n) .
   :label: 1-3

Given that :math:`\frac{E_d}{p_d} = \frac{c}{\beta_d}` (cf. Eqs. :eq:`E` and :eq:`p`) and the differential :math:`\delta E_d = \frac{p_d c^2}{E_d} \delta p_d` (Eq. :eq:`E-p`), we finally obtain:

.. math:: \Delta E^{n+1} = \Delta E^{n} + \sum_k eV_k^n \sin \left( h_k^n \vartheta^n + \phi_{\mathsf{offset},k}^n \right) - \beta_d c (p_d^{n+1} - p_d^n) .
   :label: 1-4

In the code, the second term on the r.h.s. is added in the ``kick`` function, and the third term in the ``kick_acceleration`` function.

.. warning:: By definition, **the synchronous particle never has an energy offset** (:math:`\Delta E^n = 0, \, \forall n`), and hence, no phase slippage. As a consequence, in the absence of intensity effects the synchronous phase(s) :math:`\varphi_s` that follow the design energy fulfil 

   .. math:: \sum_k eV_k^n \sin \left( \frac{h_k^n}{h_0^n} \varphi_s^n + \phi_{\mathsf{offset},k}^n \right) = E_d^{n+1} - E_d^n .
      :label: sphase

   In general, this equation has :math:`k` solutions.

.. note:: The design energy can change turn by turn, and the coordinate system will change with it. In general, due to acceleration, the coordinate system is a non-intertial.

To construct the drift equation, we have to determine the RF phase at the moment the particle crosses the accelerating cavity. At each turn, the RF phase has to be determined w.r.t. the RF frequency seen by the bunch. Acceleration will increase the RF frequency, and various LLRF loops can change both the RF frequency and/or RF phase. In reality, however, we don't know when and how exactly the frequency has changed; we only know the RF phase and frequency at discrete times (at the passage of the bunch).

Let's collect what we know about this complicated system. We'll denote revolution angular frequencies with lowercase :math:`\omega` and RF angular frequencies with uppercase :math:`\Omega`.

* The synchro loop keeps the design frequency. The design synchronous particle(s) leaving the cavity with exactly :math:`E_d^n` at turn :math:`n` will return to the cavity after a period of

   .. math:: T_d^n \equiv \frac{2 \pi}{\omega_d^n} \equiv \frac{2 \pi h_0^n}{\Omega_d^n} = \frac{2 \pi R_d^n}{\beta_d^n c},
      :label: period

   where :math:`R_d` is the radius of the design orbit.

   .. warning:: By definition, the design RF frequency is **exactly** :math:`h_0` times the design revolution frequency. Hence, a deviation of the RF frequency from the design value will result in both in a phase and frequency shift.

* The frequency slippage of an off-momentum particle during one turn is defined w.r.t. the **design synchronous particle**,

   .. math:: \frac{\Delta \omega}{\omega_d} = \frac{\omega - \omega_d}{\omega_d} \equiv - \eta(\delta) \delta \approx - (\eta_0 + \eta_1 \delta + \eta_2 \delta^2 + ...) \delta,
      :label: fslippage

   where :math:`\omega` is the revolution angular frequency of the off-momentum particle, :math:`\eta_i` are the slippage factors 

   .. math:: \eta_0 = \alpha_0 - \frac{1}{\gamma_d^2} \\
      \eta_1 = \frac{3 \beta_d^2}{2 \gamma_d^2} + \alpha_1 - \alpha_0 \eta_0 \\
      \eta_2 = - \frac{\beta_d^2(5 \beta_d^2 - 1)}{2 \gamma_d^2} + \alpha_2 - 2 \alpha_0 \alpha_1 + \frac{\alpha_1}{\gamma_0^2} + \alpha_0^2 \eta_0 - \frac{3 \beta_d^2 \alpha_0}{2 \gamma_d^2},
      :label: etas      


   :math:`\alpha_i` the momentum compaction factors, and :math:`\delta \equiv \frac{\Delta p}{p_d} = \frac{\Delta E}{\beta_d^2 E_d}` the relative momentum offset. 

 After one turn :math:`T^n`, the off-momentum particle (:math:`R \approx R_d + \Delta R`, :math:`\beta \approx \beta_d + \Delta \beta`) will return to the cavity with a time difference of

   .. math:: \Delta t^n = T^n - T_d^n = \frac{2 \pi R^n}{\beta^n c} - \frac{2 \pi R_d^n}{\beta_d^n c} = \frac{2 \pi}{\omega^n} - \frac{2 \pi}{\omega_d^n} = \left( \frac{1}{1 + \frac{\Delta \omega^n}{\omega_d^n}} - 1 \right) T_d^n = \left( \frac{1}{1 - (\eta \delta)^n} - 1 \right) T_d^n ,
      :label: tslippage

   which in first order becomes 

   .. math:: \Delta t^n \approx + (\eta \delta)^n T_d^n .
      :label: tslippage-approx

* ?? At any turn, the RF phase seen by the particle is defined w.r.t. the (actual) RF frequency at that turn,

   .. math:: \varphi^n \equiv \Omega^n t^n, \, \forall n.

   Furthermore, we only care about the time within one turn (i.e. we can always extract :math:`\sum_{m=1}^n T_d^m`), and hence, phases are defined :math:`\varphi \bmod 2 \pi h_0` and :math:`\vartheta \bmod 2 \pi`.

* ?? Since we chose the RF wave to be sinusoidal, the initial condition for the total RF phase is :math:`\varphi(t = 0) = 0`, including phase offsets. Let's call this phase the *'reference phase'*. 

   .. warning:: From Eq. :eq:`period` it follows that as long as the design RF frequency is kept, the reference phase is always :math:`\varphi(T_d^n) = 0, \, \forall n`. On the other hand, from Eq. :eq:`sphase` it follows that for non-constant acceleration rate, i.e. :math:`\ddot{E} \neq 0`, the design synchronous phase changes, even though the synchronous particle always arrives on time with :math:`t \equiv 0 \, (\bmod \, T_d^n)`. This may seem contradictory, but in fact, it is a consequence of our choice of coordinate system.

First, we construct the drift equation assuming no RF frequency or phase manipulations, i.e. the RF frequency remains the designed one. Assume an off-momentum particle with a time delay and energy offset of :math:`(\Delta t^n, \Delta E^n)` at turn :math:`n`, w.r.t. the design synchronous particle. After receiving an energy kick according to :eq:`1-2`, the particle will drift along the machine with energy :math:`\Delta E^{n+1}`. As a result, in the next turn, the arrival to the cavity is delayed further according to Eq. :eq:`tslippage`:

   .. math:: \Delta t^{n+1} = \Delta t^n + \left( \frac{1}{1 - (\eta \delta)^n} - 1 \right) T_d^n.
      :label: 2-1

This time delay can be converted into a phase offset w.r.t. the synchronous particle, taking into account that the design RF frequency can meanwhile change:

   .. math:: \Delta \varphi^{n+1} = \frac{\Omega_d^{n+1}}{\Omega_d^n} \Delta \varphi^n + 2 \pi h_0^n \left( \frac{1}{1 - (\eta \delta)^n} - 1 \right)
      :label: 2-2

   .. warning:: :math:`\Delta \varphi^n \equiv \Omega_d^n \Delta t^n` is defined w.r.t. :math:`\Omega_d^n` and :math:`\Delta \varphi^{n+1} \equiv \Omega_d^{n+1} \Delta t^{n+1}` w.r.t. :math:`\Omega_d^{n+1}`. The two cannot be directly compared. When forming the 'time derivative', one has to compare phases w.r.t. the same frequency: :math:`\Delta \dot{\varphi} \approx \Delta \varphi^{n+1} - \frac{\Omega_d^{n+1}}{\Omega_d^n} \Delta \varphi^n`.
 
In terms of 'absolute' RF phase: 

   .. math:: \varphi^{n+1} = \frac{\Omega_d^{n+1}}{\Omega_d^n} \varphi^n + 2 \pi h_0^n \left( \frac{1}{1 - (\eta \delta)^n} - 1 \right) + \varphi_d^{n+1} - \frac{\Omega_d^{n+1}}{\Omega_d^n} \varphi_d^n
      :label: 2-3


trackers.utilities module
-------------------------

.. automodule:: trackers.utilities
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: trackers
    :members:
    :undoc-members:
    :show-inheritance:
