llrf Package
============

:mod:`notch_filter` Module
--------------------------

.. automodule:: blond.llrf.notch_filter
    :members:
    :undoc-members:
    :show-inheritance:

llrf.beam_feedback Module
-------------------------

**Various beam phase loops (PL) with optional synchronisation (SL), frequency 
(FL), or radial loops (RL) for the CERN machines**

:Authors: **Helga Timko**


Machine-dependent Beam Phase Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: llrf.phase_loop.PhaseLoop(object).__init__(GeneralParameters, RFSectionParameters, Slices, gain, gain2 = 0, machine = 'LHC', period = None, window_coefficient = 0, coefficients = None, PhaseNoise = None, LHCNoiseFB = None)

   One-turn PL for different machines with different hardware. The beam phase is
   calculated as the convolution of the beam profile with the RF wave of the 
   main harmonic system (corresponding to a band-pass filter). The PL acts 
   directly on the RF frequency and phase of all harmonics. 
   
   Some machine-dependent features:
   
   * PSB: use ``sampling_frequency`` for a PL that is active only at certain 
     turns.
   
   * SPS: use ``window coefficient`` to sample beam phase over a suitable
     amount of bunches (``window_coefficient = 0`` results in single-bunch 
     acquisition as in the LHC)
     
   * LHC_F: PL with optional FL (use ``gain2`` to activate)
   
   * LHC: PL with optional SL (use ``gain2`` to activate; note that gain is 
     frequency dependent)

   :param GeneralParameters: 
      :py:class:`input_parameters.general_parameters.GeneralParameters`
   :param RFSectionParameters: 
      :py:class:`input_parameters.rf_parameters.RFSectionParameters`
   :param Slices: :py:class:`beams.slices.Slices`
   :param double gain: phase loop gain [1/ns], typically :math:`\sim 1/(10 T_0)`
   :param double gain2: FL gain [turns] or SL gain [1/ns], depending on machine;
      typically ~10 times weaker than PL
   :param str machine: machine name, determines PL choice
   :param double period: optional for PSB: period of PL being active
   :param double window_coefficient: window coefficient for band-pass filter
      determining beam phase; use 0 for single-bunch acquisition
   :param array coefficients: optional for PSB: PL transfer function 
      coefficients  
   :param PhaseNoise: optional: phase-noise type class for noise injection 
      through the PL, 
      :py:class:`llrf.rf_noise.PhaseNoise` or 
      :py:class:`llrf.rf_noise.LHCFlatSpectrum`
   :param LHCNoiseFB: optional: bunch-length feedback class for phase noise
      :py:class:`llrf.rf_noise.LHCNoiseFB`      

         
   .. py:method:: track()
      
      Calculates the PL correction on main RF frequency depending on machine.
      Updates the RF phase and frequency of the next turn for all RF systems.
      
      Let :math:`\Delta \omega_{\mathsf{TOT}}` be the total frequency correction 
      (calculation depends on the machine, see below). The RF frequency of a 
      given RF system :math:`i` is then shifted by
      
      .. math:: \Delta \omega_{\mathsf{rf},i} = \frac{h_i}{h_0} \Delta 
         \omega_{\mathsf{TOT}} ,
      
      with a corresponding RF phase shift of
      
      .. math:: \Delta \varphi_{\mathsf{rf},i} = 2 \pi h_i 
         \frac{\omega_{\mathsf{rf},i}}{\Omega_{\mathsf{rf},i}} ,
      
      where :math:`\Omega_{\mathsf{rf},i} = h_i \omega_0` is the design
      frequency and :math:`\omega_{\mathsf{rf},i}` the actual RF frequency
      applied.

         
   .. py:method:: precalculate_time(GeneralParameters)
         
      For PSB, where the PL acts only with a given periodicity, pre-calculate on
      which turns to act.   
         
      :param GeneralParameters: 
         :py:class:`input_parameters.general_parameters.GeneralParameters`
       
       
   .. py:method:: beam_phase()
   
      Beam phase measured at the main RF frequency and phase. The beam is 
      convolved with the window function of the band-pass filter of the machine.
      The coefficients of sine and cosine components determine the beam phase, 
      projected to the range -Pi/2 to 3/2 Pi. 
      
      .. note:: that this beam phase is already determined w.r.t. the 
         instantaneous RF phase.
       
      The band-pass filter modelled assumes a window function of the form
      
      .. math:: W(t) = e^{-\alpha t} 
         \cos(\omega_{\mathsf{rf}} t - \varphi_{\mathsf{rf}}) ,
      
      where :math:`\alpha` is the ``window_coefficient`` that determines how 
      many bunches are taken into account. 
      
      The convolution of :math:`W(t)` with the bunch profile :math:`\lambda(t)`
      results in two components,
      
      .. math:: f(t) = \int_{\lambda_{\mathsf{min}}}^{\lambda_{\mathsf{max}}}
         {e^{-\alpha (t-\tau)} \cos(\omega_{\mathsf{rf}} (t-\tau) - 
         \varphi_{\mathsf{rf}}) \lambda(\tau) d\tau} 
         = e^{-\alpha t} \cos(\omega_{\mathsf{rf}} t)  
         \int_{\lambda_{\mathsf{min}}}^{\lambda_{\mathsf{max}}}
         {e^{\alpha \tau} \cos(\omega_{\mathsf{rf}} \tau + 
         \varphi_{\mathsf{rf}}) \lambda(\tau) d\tau} 
         + e^{-\alpha t} \sin(\omega_{\mathsf{rf}} t)  
         \int_{\lambda_{\mathsf{min}}}^{\lambda_{\mathsf{max}}}
         {e^{\alpha \tau} \sin(\omega_{\mathsf{rf}} \tau + 
         \varphi_{\mathsf{rf}}) \lambda(\tau) d\tau} .
         
      The beam phase is determined from the coefficients of the sine and cosine
      components, i.e.
      
      .. math:: \varphi_b \equiv \arctan \left( 
         \frac{\int_{\lambda_{\mathsf{min}}}^{\lambda_{\mathsf{max}}}
         {e^{\alpha \tau} \sin(\omega_{\mathsf{rf}} \tau + 
         \varphi_{\mathsf{rf}}) \lambda(\tau) d\tau}}
         {\int_{\lambda_{\mathsf{min}}}^{\lambda_{\mathsf{max}}}
         {e^{\alpha \tau} \cos(\omega_{\mathsf{rf}} \tau + 
         \varphi_{\mathsf{rf}}) \lambda(\tau) d\tau}} \right) .
   
      This projects the beam phase to the interval 
      :math:`\left( -\frac{\pi}{2} , \frac{\pi}{2}\right)`, however, the RF 
      phase is defined on the interval 
      :math:`\left( -\frac{\pi}{2} , \frac{3 \pi}{2}\right)`. In order to get a 
      correct measurement of the beam phase, we thus add :math:`\pi` if the 
      cosine coefficient is negative (meaning normally the beam energy is above 
      transition).

      
   .. py:method:: phase_difference()               

      Phase difference between beam and RF phase of the main RF system.
      Optional: add RF phase noise through dphi directly.
      
      As the actual RF phase is taken into account already in the beam phase
      calculation, only the synchronous phase needs to be substracted and thus
      the phase difference seen by the PL becomes
      
      .. math:: \Delta \varphi_{\mathsf{PL}} = \varphi_b - \varphi_s .
      
      If phase noise is injected through the PL, it is added directly as an 
      offset to this measurement, optionally with the feedback scaling factor
      :math:`x`.
      
      .. math:: \Delta \varphi_{\mathsf{PL}} = \varphi_b - \varphi_s 
         + (x) \phi_N .
          

   .. py:method:: LHC_F():
        
      Calculates the RF frequency correction :math:`\Delta \omega_{\mathsf{PL}}`
      from the phase difference between beam and RF 
      :math:`\Delta \varphi_{\mathsf{PL}}` for the LHC. The transfer function is
        
      .. math:: \Delta \omega_{\mathsf{PL}} = - g_{\mathsf{PL}} 
         \Delta\varphi_{\mathsf{PL}} , 
            
      Using 'gain2', the frequency loop can be activated in addition to remove
      long-term frequency drifts:
        
      .. math:: \Delta \omega_{\mathsf{FL}} = - g_{\mathsf{FL}} 
         (\omega_{\mathsf{rf}} - h \omega_{0}) .

     
   .. py:method:: LHC()
   
      Calculates the RF frequency correction :math:`\Delta \omega_{\mathsf{PL}}`
      from the phase difference between beam and RF 
      :math:`\Delta \varphi_{\mathsf{PL}}` for the LHC. The transfer function is
        
      .. math:: \Delta \omega_{\mathsf{PL}} = - g_{\mathsf{PL}} 
         \Delta \varphi_{\mathsf{PL}} , 
            
      Using 'gain2', a synchro loop can be activated in addition to remove
      long-term frequency and phase drifts:     
        
      .. math:: \Delta \omega_{\mathsf{SL}} = - g_{\mathsf{SL}} 
         (y + a \, \Delta \varphi_{\mathsf{rf}}) ,
            
      where :math:`\Delta \varphi_{\mathsf{rf}}` is the accumulated RF phase
      deviation from the design value and :math:`y` is is obtained through the
      recursion (:math:`y_0 = 0`)
        
      .. math:: y_{n+1} = (1 - \tau) y_n + (1 - a) \tau 
         \Delta \varphi_{\mathsf{rf}} .
            
      The variables :math:`a` and :math:`\tau` are being defined through the 
      (single-harmonic, central) synchrotron frequency :math:`f_s` and the 
      corresponding synchrotron tune :math:`Q_s` as
        
      .. math:: a (f_s) \equiv 5.25 - \frac{f_s}{\pi 40~\text{Hz}} ,
            
      .. math:: \tau(f_s) \equiv 2 \pi Q_s \sqrt{ \frac{a}{1 + 
         \frac{g_{\mathsf{PL}}}{g_{\mathsf{SL}}} \sqrt{\frac{1 + 1/a}{1 + a}} }} .
         
         
   .. py:method:: PSB():
      
      Phase loop:
      
      The transfer function of the system is
        
      .. math:: H(z) = g \frac{b_{0}+b_{1} z^{-1}}{1 +a_{1} z^{-1}}  
      
      where g is the gain and :math:`b_{0} = 0.99901903`, :math:`b_{1} = -0.99901003`,  
      :math:`a_{1} = -0.99803799`.
      
      Let :math:`\Delta \phi_{PL}` and :math:`\Delta \omega_{PL}` be the
      phase difference and the phase loop correction on the frequency
      respectively; since these two quantities are the input and output of our 
      system, then from the transfer function we
      have in time domain (see https://en.wikipedia.org/wiki/Z-transform):
      
      .. math:: \Delta \omega_{PL}^{n+1} = - a_{1} \Delta \omega_{PL}^{n} + 
            g(b_{0} \Delta \phi_{PL}^{n+1} + b_{1} \Delta \phi_{PL}^{n})
      
      In fact the phase and radial loops act every 10 :math:`\mu s` and as a 
      consequence :math:`\Delta \phi_{PL}` is an average on all the values 
      between two trigger times.
      
      Radial loop:
      
      We estimate
      the difference of the radii of the actual trajectory and the desired trajectory
      using one of the four known differential relations with :math:`\Delta B = 0`:
      
      .. math:: \frac{\Delta R}{R} = \frac{\Delta \omega_{RF}}{\omega_{RF}} 
        \frac{\gamma^2}{\gamma_{T}^2-\gamma^2}
      
      In reality the error :math:`\Delta R` is filtered with a PI (Proportional-
      Integrator) corrector. This means that
      
      .. math:: \Delta \omega_{RL}^{n+1} = K_{P} \left(\frac{\Delta R}{R}\right)^{n} 
            + K_{I} \int_0^n \! \frac{\Delta R}{R} (t) \, \mathrm{d}t. 
      
      Writing the same equation for :math:`\Delta \omega_{RL}^{n}` and
      subtracting side by side we have
      
      .. math:: \Delta \omega_{RL}^{n+1} = \Delta \omega_{RL}^{n} + 
            K_{P} \left[ \left(\frac{\Delta R}{R}\right)^{n} - 
            \left(\frac{\Delta R}{R}\right)^{n-1} \right] + K_{I}^{'} 
            \left(\frac{\Delta R}{R}\right)^{n} 
      
      here :math:`K_{I}^{'} = K_{I} 10 \mu s` and we approximated the integral
      with a simple product.
      
      The total correction is then
      
      .. math:: \Delta \omega_{RF}^{n+1} = \Delta \omega_{PL}^{n+1} + \Delta \omega_{RL}^{n+1}
    

:mod:`cavity_feedback` Module
-----------------------------

.. automodule:: blond.llrf.cavity_feedback
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`impulse_response` Module
------------------------------

.. automodule:: blond.llrf.impulse_response
    :members:
    :undoc-members:
    :show-inheritance:

:mod:`rf_modulation` Module
---------------------------

.. automodule:: blond.llrf.rf_modulation
    :members:
    :undoc-members:
    :show-inheritance:

llrf.rf_noise Module
--------------------

**Methods to generate RF phase noise from noise spectrum and feedback noise
amplitude as a function of bunch length**

:Authors: **Helga Timko**

RF phase noise generation
^^^^^^^^^^^^^^^^^^^^^^^^^
    
.. py:class:: llrf.rf_noise.PhaseNoise(object).__init__(frequency_array, real_part_of_spectrum, seed1=None, seed2=None)    

   Contains the spectrum of RF phase noise and the actual phase noise randomly
   generated from it. Generation done via mixing with white noise.
    
   :param numpy.array frequency_array: input frequency range
   :param numpy.array real_part_of_spectrum: input spectrum, real part only, 
      same length as ``frequency_array``
   :param int seed1: seed for random number generator
   :param int seed2: seed for random number generator

   .. warning:: The spectrum has to be input as double-sided spectrum, in units
      of [:math:`\text{rad}^2/\text{Hz}`]. 
   
   Both hermitian to real and complex to complex FFTs are available. Use seeds
   to fix a certain random number sequence; with ``seed=None`` a random sequence
   will be initialized.  	
 
   .. py:method:: spectrum_to_phase_noise(transform=None)	
	
      Transforms a noise spectrum to phase noise data. 
		
      :param transform: FFT transform kind
      :type transform: choice
      :return: time and phase noise arrays
	
      .. note:: Use ``transform=None`` or ``'r'`` to transform hermitian 
         spectrum to real phase. In this case, input only the positive part of 
         the double-sided spectrum. Use ``transform='c'`` to transform complex 
         spectrum to complex phase. In this case, input first the zero and 
         positive frequency components, then the decreasingly negative frequency
         components of the double-sided spectrum. Returns only the real part of
         the phase noise. E.g. the following two ways of usage are equivalent:
			
         .. image:: RF_noise.png
            :align: center
            :width: 1000
            :height: 250       
       

      **The transformation in steps**

      **Step 1:** Set the resolution in time domain. To transform a hermitian
      spectrum to real phase noise, 
     
      .. math:: n_t = 2 (n_f - 1) \text{\,\,and\,\,} \Delta t = 1/(2 f_{\text{max}}) , 

      and to transform a complex spectrum to complex phase noise,

      .. math:: n_t = n_f \text{\,\,and\,\,} \Delta t = 1/f_{\text{max}} ,

      where ``fmax`` is the maximum frequency in the input in both cases.         

      **Step 2:** Generate white (carrier) noise in time domain
        
      .. math:: 
         w_k(t) = \cos(2 \pi r_k^{(1)}) \sqrt{-2 \ln(r_k^{(2)})} \text{\,\,\,case `r'},
    
         w_k(t) = \exp(2 \pi i r_k^{(1)}) \sqrt{-2 \ln(r_k^{(2)})} \text{\,\,\,case `c'},           
        
      **Step 3:** Transform the generated white noise to frequency domain
            
      .. math:: W_l(f) = \sum_{k=1}^N w_k(t) e^{-2 \pi i \frac{k l}{N}} .

      **Step 4:** In frequency domain, colour the white noise with the desired
      noise probability density (unit: radians). The noise probability density
      derived from the double-sided spectrum is

      .. math:: s_l(f) = \sqrt{A S_l^{\text{DB}} f_{\text{max}}} ,  

      where :math:`A=2` for ``transform = 'r'`` and :math:`A=1` for 
      ``transform = 'c'``. The coloured noise is obtained by multiplication in 
      frequency domain

      .. math:: \Phi_l(f) = s_l(f) W_l(f) .

      **Step 5:** Transform back the coloured spectrum to time domain to obtain
      the final phase shift array (we use only the real part).
        
        
LHC-type phase noise generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: llrf.rf_noise.LHCFlatSpectrum(object).__init__(GeneralParameters, RFSectionParameters, time_points, corr_time = 10000, fmin = 0.8571, fmax = 1.1, initial_amplitude = 1.e-6, seed1 = 1234, seed2 = 7564)    

   Generates LHC-type phase noise from a band-limited spectrum. Input frequency 
   band using ``fmin`` and ``fmax`` w.r.t. the synchrotron frequency. Input 
   double-sided spectrum amplitude [:math:`\text{rad}^2/\text{Hz}`] using 
   ``initial_amplitude``. Fix seeds to obtain reproducible phase noise. Select
   ``time_points`` suitably to resolve the spectrum in frequency domain. After
   ``corr_time`` turns, the seed is changed (reproducibly) to cut numerical 
   correlated sequences of the random number generator.
    
   :param GeneralParameters: 
      :py:class:`input_parameters.general_parameters.GeneralParameters`
   :param RFSectionParameters: 
      :py:class:`input_parameters.rf_parameters.RFSectionParameters`
   :param int time_points: number of phase noise points of a sample in time 
      domain
   :param int corr_time: number of turns after which seed is changed
   :param double fmin: spectrum lower limit in units of synchrotron frequency
   :param double fmax: spectrum upper limit in units of synchrotron frequency
   :param double initial_amplitude: initial double sided spectral density 
      [:math:`\text{rad}^2/\text{Hz}`]
   :param int seed1: seed for random number generator
   :param int seed2: seed for random number generator

   .. warning:: ``time_points`` should be chosen large enough to resolve the 
      desired frequency step :math:`\Delta f =` 
      :py:attr:`GeneralParameters.f_rev`/:py:attr:`LHCFlatSpectrum.time_points`
      in frequency domain.

   .. py:method:: generate()

      Generates LHC-type phase noise array (length: 
      :py:attr:`GeneralParameters.n_turns` + 1). Stored in the variable 
      :py:attr:`LHCFlatSpectrum.dphi`.


Bunch-length based feedback on noise amplitude
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
   
.. py:class:: llrf.rf_noise.LHCNoiseFB(object).__init__(bl_target, gain = 1.5, factor = 0.8)
   
   Feedback on phase noise amplitude for LHC controlled longitudinal emittance
   blow-up using noise injection through cavity controller or phase loop.
   The feedback compares the FWHM bunch length of the bunch to a target value 
   and scales the phase noise to keep the targeted value.
   
   :param bl_target: Targeted 4-sigma-equivalent FWHM bunch length [ns]
   :param gain: feedback gain [1/ns]
   :param factor: feedback recursion scaling factor [1]
   
   .. warning:: Note that the FWMH bunch length is scaled by 
      :math:`\sqrt{2/\ln{2}}` in order to obtain a 4-sigma equivalent value.

   .. py:method:: FB(RFSectionParameters, Beam, PhaseNoise, Slices, CC = False)
   
      Calculates the bunch-length based feedback scaling factor as a function
      of measured FWHM bunch length. For phase noise injected through the 
      cavity RF voltage, the feedback scaling can be directly applied on the
      :py:attr:`RFSectionParameters.phi_noise` variable by setting 
      ``CC = True``. For phase noise injected through the :py:class:`PhaseLoop`
      class, the correction can be applied inside the phase loop, via passing 
      :py:class:`LHCNoiseFB` as an argument in :py:class:`PhaseLoop`.
      
      :param RFSectionParameters: 
         :py:class:`input_parameters.rf_parameters.RFSectionParameters`
      :param Beam: :py:class:`beams.beams.Beam`
      :param PhaseNoise: phase-noise type class, 
         :py:class:`llrf.rf_noise.PhaseNoise` or 
         :py:class:`llrf.rf_noise.LHCFlatSpectrum`
      :param Slices: :py:class:`beams.slices.Slices`
      :param bool CC: cavity controller option
      
      
.. py:classmethod:: fwhm(Slices)

   Fast FWHM bunch length calculation with slice width precision.
   
   :param Slices: :py:class:`beams.slices.Slices`
   :return: 4-sigma-equivalent FWHM bunch length [ns]


:mod:`signal_processing` Module
-------------------------------

.. automodule:: blond.llrf.signal_processing
    :members:
    :undoc-members:
    :show-inheritance:

