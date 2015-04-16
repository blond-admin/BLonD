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

Below, we shall derive the equations of motion (EOMs) for an energy kick given to the particle by the RF caviti(es) of a given RF station and the subsequent drift of the particle during one turn, see Figure. In the case of multiple RF stations, the drift equation should be scaled by :math:`\frac{L_i}{C}`, where :math:`C = \sum_{i} L_i` is the machine circumference.

Definitions
"""""""""""

**Code-internal units** are SI-units unless otherwise specified in the below table.

:Code-internal units:

   Energy :math:`E` = [MeV] 

   Momentum :math:`p` = [MeV] 

   Mass :math:`m` = [MeV]

   Time :math:`t` = [ns] 
   
   Frequency :math:`f` = [GHz]

   Revolution frequency :math:`\omega` = [1/ns]

Just like in the real machine, we demand the user to define beforehand the momentum programme, i.e. the **synchronous (design) momentum** at every time step :math:`n` and RF station :math:`(i)`, :math:`\left\{ p_{s,(i)}^n \right\}`. This will define the design total energy :math:`E_s` through following relations:

.. math:: E_s = \gamma_s m ,
   :label: E
.. math:: p_s = \beta_s \gamma_s m ,
   :label: p
.. math:: E_s^2 = p_s^2 + m^2 ,
   :label: E-p

where :math:`m` is the mass of the particle, and energy, momentum, and mass are given in [MeV].

For a given **synchronous orbit with an average radius** :math:`R_s`, the revolution period will be :math:`\omega_0 \equiv \frac{\beta_s c}{R_s}` and one turn will take :math:`T_0 \equiv \frac{2 \pi}{\omega_0}` at the synchronous energy. The magnetic field programme is assumed to be synchronised with the design energy turn by turn. Hence, a particle leaving the RF station with the synchronous energy will always be on the synchronous orbit and return to the RF station after exactly one period, unless the actual magnetic field differs from the designed one.

We define then an external **reference (clock) time**

.. math:: t_{\mathsf{ref}}^n \equiv \sum_{i=1}^n T_0^i ,
   :label: t_ref

and as initial condition we choose the sinusoidal RF wave of the **main RF system** (harmonic :math:`h_0` and RF frequency :math:`\omega_{\mathsf{rf},0}`) to be at phase zero at time zero:

.. math:: V(t = 0) = V_0 \sin(\omega_{\mathsf{rf},0} \, t) = 0 .
   :label: IC

**All phase offsets of all RF systems are defined w.r.t. this initial condition and w.r.t. to the main RF system.** Phase offsets can be programmed through the ``phi_offset`` parameter. For :math:`n_{\mathsf{rf}}` RF systems in the RF station the total voltage [MeV] becomes:

.. math:: V_{\mathsf{tot}}(t) = \sum_{k=0}^{n_{\mathsf{rf}}-1} V_k \sin(\omega_{\mathsf{rf},k} \, t + \phi_{\mathsf{offset},k}^n) .
   :label: Vtot


We define the **arrival time of an arbitrary particle** to the RF station relative to the reference time in that turn,

.. math:: \Delta t^n \equiv t^n - t_{\mathsf{ref}}^n .
   :label: dt


Energy kick
"""""""""""

During the passage through an RF station, the energy :math:`E^n` of an arbitrary particle is changed by the total energy kick received from the various RF systems in the station. The energy gain due to the induced electric fields in the magnets is negligible and beam-induced voltage is taken into account in the ``impedace`` module. Whatever the RF phase :math:`\varphi_{\mathsf{rf}}` and revolution frequency :math:`\omega_{\mathsf{rf}}` in the different RF systems may be, the energy kick for a particle arriving at time :math:`t_n` is:

.. math:: E^{n+1} = E^{n} + \sum_{k=0}^{n_{\mathsf{rf}}-1} V_k^n \sin \left( \omega_{\mathsf{rf},k}^n \, t^n + \phi_{\mathsf{offset},k}^n \right) = E^{n} + \sum_{k=0}^{n_{\mathsf{rf}}-1} V_k^n \sin \left( \omega_{\mathsf{rf},k}^n \, \Delta t^n + \varphi_{\mathsf{rf},k}^n \right) ,
   :label: 1-1

where the RF phase of system k in turn n :math:`\varphi_{\mathsf{rf},k}` is defined as

.. math:: \varphi_{\mathsf{rf},k}^n \equiv \phi_{\mathsf{offset},k}^n + \sum_{k=0}^{n_{\mathsf{rf}}-1} \omega_{\mathsf{rf},k}^n T_0^i
   :label: 1-a

As multiples of :math:`2 \pi` can be neglected, an equivalent form is:

.. math:: \varphi_{\mathsf{rf},k}^n \equiv \phi_{\mathsf{offset},k}^n + \sum_{k=0}^{n_{\mathsf{rf}}-1} \frac{\omega_{\mathsf{rf},k}^n - h_k^n \omega_0^n}{\omega_0^n} T_0^i
   :label: 1-b


.. note:: Eq. :eq:`1-1` is intrinsically discrete; no approximation has been done.

.. note:: The RF phase (Eq. :eq:`1-b`) differs from the phase offset only if the RF frequency differs from the design RF frequency :math:`\omega_{\mathsf{rf},k}^n \neq \Omega_{\mathsf{rf},k}^n \equiv h_k^n \omega_0^n`, i.e. when feedback loops are active.

Rather than the absolute energy, we are actually interested in the energy offset of a given particle w.r.t. the synchronous energy :math:`\Delta E^n \equiv E^n - E_s^n, \, \forall n`. So we choose our coordinate system to be centred around :math:`E_s^n \, \forall n`. Substracting :math:`E_s^{n+1}` from both sides of Eq. :eq:`1-1`, we arrive at

.. math:: \Delta E^{n+1} = \Delta E^{n} + \sum_{k=1}^K V_k^n \sin \left( \omega_{\mathsf{rf},k}^n \Delta t^n + \varphi_{\mathsf{rf},k}^n \right) - (E_s^{n+1} - E_s^n) .
   :label: 1-2

.. warning:: As a consequence, during acceleration the coordinate system is non-inertial and a coordinate transform is done turn by turn.


Arrival time drift
""""""""""""""""""

.. sidebar:: Frequency slippage

   The frequency slippage of an off-momentum particle during one turn is defined w.r.t. the **design synchronous particle**,

   .. math:: \frac{\Delta \omega}{\omega_0} = \frac{\omega - \omega_0}{\omega_0} \equiv - \eta(\delta) \delta \approx - (\eta_0 + \eta_1 \delta + \eta_2 \delta^2 + ...) \delta,
      :label: slippage

   where :math:`\delta \equiv \frac{\Delta p}{p_s} = \frac{\Delta E}{\beta_s^2 E_s}` is the relative off-momentum of the particle and :math:`\eta_i` are the slippage factors 

   .. math:: \eta_0 = \alpha_0 - \frac{1}{\gamma_s^2} \\
      \eta_1 = \frac{3 \beta_s^2}{2 \gamma_s^2} + \alpha_1 - \alpha_0 \eta_0 \\
      \eta_2 = - \frac{\beta_s^2(5 \beta_s^2 - 1)}{2 \gamma_s^2} + \alpha_2 - 2 \alpha_0 \alpha_1 + \frac{\alpha_1}{\gamma_s^2} + \alpha_0^2 \eta_0 - \frac{3 \beta_s^2 \alpha_0}{2 \gamma_s^2},
      :label: etas      

   which are defined via the momentum compaction factors :math:`\alpha_i`. 

The absolute arrival time of an arbitrary particle can be expressed as a recursion

.. math:: t^{n+1} = t^n + \frac{2 \pi}{\omega^{n+1}} ,
   :label: 2-1

with initial condition :math:`t^0 = t_0` and where the revolution frequency of the particle :math:`\omega^n \equiv \frac{\beta^n c}{R^n}` can differ from :math:`\omega_0^n` due to energy and orbit deviations from the synchronous particle.

.. note:: Eq. :eq:`2-1` contains :math:`\omega^{n+1}` as we chose to perform the energy kick first and the subsequent time drift happens according to the already updated energy.

Using Eq. :eq:`dt`, the recursion on the particle arrival time relative to the clock becomes

.. math:: \Delta t^{n+1} = \Delta t^n + \frac{2 \pi}{\omega^{n+1}} - \frac{2 \pi}{\omega_0^{n+1}} =
   \Delta t^n + T_0 \left( \frac{1}{\frac{\omega^{n+1}}{\omega_0^{n+1}}} - 1 \right)
   :label: 2-2

Using definition :eq:`slippage`, the arrival time drift can be calculated as

.. math:: \Delta t^{n+1} = \Delta t^n + T_0 \left( \frac{1}{1 - \eta(\delta^{n+1})\delta^{n+1}} - 1 \right) .
   :label: 2-3

If a zeroth order slippage is used, :math:`\eta(\delta) \approx \eta_0`, the option ``solver = 'simple'`` can be used to approximate the above equation as

.. math:: \Delta t^{n+1} = \Delta t^n + \frac{\eta_0 T_0}{\beta_s^2 E_s} \Delta E^{n+1} .
   :label: 2-3-simple

The synchronous particle
""""""""""""""""""""""""

A particle is **synchronous in turn n** if it enters and leaves the RF station with zero energy offset, :math:`\Delta E^n = \Delta E^{n+1} = 0`, and thus gains exactly the designed energy gain :math:`E_s^{n+1} - E_s^n`. As a consequence, in the absence of induced voltage the synchronous particle will fulfil:

.. math:: \sum_{k=0}^{n_{\mathsf{rf}}-1} V_k^n \sin \left( \omega_{\mathsf{rf},k}^n \Delta t_s^n + \varphi_{\mathsf{rf},k}^n \right) \equiv \sum_{k=0}^{n_{\mathsf{rf}}-1} V_k^n \sin {\varphi_{s,k}^n} = (E_s^{n+1} - E_s^n) ,
   :label: E_s

and in the presence of intensity effects, the induced voltage from the particles in front should be added on the left-hand side: 

.. math:: \sum_{k=0}^{n_{\mathsf{rf}}-1} V_k^n \sin \left( \omega_{\mathsf{rf},k}^n \Delta t_s^n + \varphi_{\mathsf{rf},k}^n \right) - e \int_{\Delta t_{\mathsf{min}}}^{\Delta t_s^n} d\tau \lambda(\tau) W(\Delta t_s^n - \tau) = (E_s^{n+1} - E_s^n) ,
   :label: E_s-2

where :math:`\lambda(t)` is the beam/bunch profile and :math:`W(t)` the wake potential.

.. seealso:: http://dquartul.github.io/BLonD/impedances.html

.. warning:: In general, these equations have :math:`n_{\mathsf{rf}}` solutions. If the synchronous energy gain :math:`E_s^{n+1} - E_s^n` changes from one turn to another, also the synchronous particle changes with it. 

.. note:: **Synchronous particle arrival time**

   As a consequence, the arrival time of the synchronous particle :math:`\Delta t_s` is not necessarily constant, but can change turn by turn. This might be counter-intuitive, as the synchronous particle drifts with exactly :math:`\omega_0` along the ring. To see this effect, consider two subsequent turns with different synchronous energy gains :math:`E_s^{n+1} - E_s^{n} \ne E_s^{n+2} - E_s^{n+1}` in a single-RF system. Let particle 1 be synchronous in turn n and particle 2 be synchronous in turn n+1: 

   .. math:: \Delta E_1^n = \Delta E_1^{n+1} = 0 \,\,\, \ne \Delta E_1^{n+2} \\
      \Delta t_1^{n-1} = \Delta t_1^n = \Delta t_1^{n+1} \,\,\, \ne \Delta t_1^{n+2}
      :label: synch-1

   .. math:: \Delta E_2^n \ne \,\,\, \Delta E_2^{n+1} = \Delta E_2^{n+2} = 0  \\
      \Delta t_2^{n-1} \ne \,\,\, \Delta t_2^n = \Delta t_2^{n+1} = \Delta t_2^{n+2}
      :label: synch-2

   The arrival time of the synchronous particles in this case will be:

   .. math:: \Delta t_s^n \equiv \Delta t_1^n \\
      \Delta t_s^{n+1} \equiv \Delta t_2^{n+1}
      :label: synch-3

   Thus, because the synchronous particle can be a different particle each turn, the recursion on the synchronous arrival time becomes in general 

   .. math:: \Delta t_s^{n+1} = \Delta t_s^n + (\Delta t_2^{n+1} - \Delta t_1^n) = \Delta t_s^n + (\Delta t_2^n - \Delta t_1^n) .

   The difference in arrival time of the two particles in turn n can be determined from the energy equations

   .. math:: V^n \sin \left( \omega_{\mathsf{rf}}^n \Delta t_1^n + \varphi_{\mathsf{rf}}^n \right)= (E_s^{n+1} - E_s^n) \\
      V^{n+1} \sin \left( \omega_{\mathsf{rf}}^{n+1} \Delta t_2^{n+1} + \varphi_{\mathsf{rf}}^{n+1} \right) =  V^{n+1} \sin \left( \omega_{\mathsf{rf}}^{n+1} \Delta t_2^n + \varphi_{\mathsf{rf}}^{n+1} \right) = (E_s^{n+2} - E_s^{n+1}) , 

   which in first-order approximation (see *Small-amplitude oscillations*) gives

   .. math:: \sin{\varphi_s^{n+1}} - \sin{\varphi_s^n} \approx ( \omega_{\mathsf{rf}}^{n+1} \Delta t_2^n + \varphi_{\mathsf{rf}}^{n+1} - \omega_{\mathsf{rf}}^n \Delta t_1^n - \varphi_{\mathsf{rf}}^n) \cos{\varphi_s^n} = \frac{E_s^{n+2} - E_s^{n+1}}{V^{n+1}} - \frac{E_s^{n+1} - E_s^{n}}{V^{n}} .



Small-amplitude oscillations
""""""""""""""""""""""""""""

Assuming a single-RF station and a simple solver (Eq. :eq:`2-3-simple`), the EOMs in continous time can be written as

.. math:: \Delta \dot{E} = \frac{V}{T_0} \left( \sin(\omega_{\mathsf{rf}} \Delta t + \varphi_{\mathsf{rf}}) - \sin{\varphi_s} \right),
   :label: HO-1

.. math:: \Delta \dot{t} = \frac{\eta_0}{\beta_s^2 E_s} \Delta E .
   :label: HO-2

Assuming further a constant synchronous phase :math:`\sin{\varphi_s} \equiv \frac{\dot{E_s}}{V}` and expanding the RF wave around it 

.. math:: \omega_{\mathsf{rf}} \Delta t + \varphi_{\mathsf{rf}} = \varphi_s + (\omega_{\mathsf{rf}} \Delta t + \varphi_{\mathsf{rf}} - \varphi_s) \equiv \varphi_s + \varepsilon , \,\,\, \varepsilon \ll 1 , 

we obtain for the sinusoidal term in first order

.. math:: \sin (\omega_{\mathsf{rf}} \Delta t + \varphi_{\mathsf{rf}}) = \sin {\varphi_s} \cos{\varepsilon} + \cos{\varphi_s} \sin{\varepsilon} \approx \sin{\varphi_s} (1 + \mathcal{O}(\varepsilon^2)) +  \cos{\varphi_s} (\varepsilon + \mathcal{O}(\varepsilon^3)) .

Derivating Eq. :eq:`HO-1` a second time, and using Eq. :eq:`HO-2`

.. math:: \Delta \ddot{E} = \frac{V \cos{\varphi_s} \omega_{\mathsf{rf}}}{T_0} \Delta \dot{t} = \frac{V \eta_0 \cos{\varphi_s} \omega_{\mathsf{rf}} \omega_0 }{2 \pi \beta_s^2 E_s} \Delta E .
   :label: HO-3

Vice versa, derivating :eq:`HO-2` another time, and substituting Eq. :eq:`HO-1`, an equivalent equation can be found for the arrival time w.r.t. to the arrival of the synchronous particle :math:`\Delta t_s`:

.. math:: \Delta \ddot{t} - \Delta \ddot{t_s} = \frac{V \eta_0 \cos{\varphi_s} \omega_{\mathsf{rf}} \omega_0 }{2 \pi \beta_s^2 E_s} (\Delta t - \Delta t_s) .
   :label: HO-4

Equations :eq:`HO-3` and :eq:`HO-4` describe an oscillating motion in phase space if :math:`\eta_0 \cos{\varphi_s} < 0`, which for :math:`\omega_{\mathsf{rf}} = h \omega_0` has the synchrotron frequency

.. math:: \omega_{s,0} = \sqrt{\frac{- h V \eta_0 \cos{\varphi_s}}{2 \pi \beta_s^2 E_s}} \omega_0 .
   :label: omega_s

.. note:: that energy and time are conjugate variables, whereas energy and phase are not. When forming time derivatives in phase, one should take into account the frequency correction from one turn to another: :math:`\dot{\varphi} \approx \varphi^{n+1} - \frac{\omega_{\mathsf{rf}}^{n+1}}{\omega_{\mathsf{rf}}^n} \varphi^n`.



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

