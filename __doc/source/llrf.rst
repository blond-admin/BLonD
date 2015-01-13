llrf package
============

Submodules
----------

llrf.RF_noise module
--------------------

**Methods to generate RF phase noise from noise spectrum**

:Authors: **Helga Timko**
    
.. class:: LLRF.RF_noise.PhaseNoise(object).__init__(self, frequency_array, real_part_of_spectrum, seed1=None, seed2=None)    

   The **PhaseNoise** class contains the spectrum of RF phase noise and the 
   actual phase noise randomly generated from this spectrum (via mixing with
   white noise).
    
   :param numpy.array frequency_array: input frequency range
   :param numpy.array real_part_of_spectrum: input spectrum, real part only, same length as ``frequency_array``
   :param int seeds: seeds for random number generator
   :return: time and phase noise arrays

.. warning:: The spectrum has to be input as double-sided spectrum, in units of
   radian-square per hertz. 
   
Both hermitian to real and complex to complex FFTs are available. Use seeds to 
fix a certain random number sequence; with ``seed=None`` a random sequence will
be initialized.  	
 
   .. function:: spectrum_to_phase_noise(self, transform=None)	
	
      Transforms a noise spectrum to phase noise data. 
		
      :param transform: FFT transform kind
      :type transform: choice
	
   .. note:: Use ``transform=None`` or ``'r'`` to transform hermitian spectrum 
      to real phase. In this case, input only the positive part of the 
      double-sided spectrum. Use ``transform='c'`` to transform complex spectrum
      to complex phase. In this case, input first the zero and positive 
      frequency components, then the decreasingly negative frequency components
      of the double-sided spectrum. Returns only the real part of the phase 
      noise. E.g. the following two ways of usage are equivalent:
			
      .. image:: RF_noise.png
         :align: center
         :width: 1000
         :height: 250       
       

The transformation in steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Step 1:** Set the resolution in time domain. To transform a hermitian spectrum to 
real phase noise, 
     
.. math:: n_t = 2 (n_f - 1) \text{and} \Delta t = 1/(2 f_{\text{max}}) , 

and to transform a complex spectrum to complex phase noise,

.. math:: n_t = n_f \text{and} \Delta t = 1/f_{\text{max}} ,

where ``fmax`` is the maximum frequency in the input in both cases.         

**Step 2:** Generate white (carrier) noise in time domain
        
.. math:: 
    w_k(t) = \cos(2 \pi r_k^{(1)}) \sqrt{-2 \ln(r_k^{(2)})} \text{case 'r'},
    
    w_k(t) = \exp(2 \pi i r_k^{(1)}) \sqrt{-2 \ln(r_k^{(2)})} \text{case 'c'},           
        
**Step 3:** Transform the generated white noise to frequency domain
            
.. math:: W_l(f) = \sum_{k=1}^N w_k(t) e^{-2 \pi i \frac{k l}{N}} .

**Step 4:** In frequency domain, colour the white noise with the desired noise 
probability density (unit: radians). The noise probability density derived from 
the double-sided spectrum is

.. math:: s_l(f) = \sqrt{A S_l^{\text{DB}} f_{\text{max}}} ,  

where :math:`A=2` for ``transform = 'r'`` and A=1 for ``transform = 'c'``. The 
coloured noise is obtained by multiplication in frequency domain

.. math:: \Phi_l(f) = s_l(f) W_l(f) .

**Step 5:** Transform back the coloured spectrum to time domain to obtain the final 
phase shift array (we use only the real part).
        


Module contents
---------------

.. automodule:: llrf
   :members:
   :undoc-members:
   :show-inheritance:
