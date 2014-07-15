'''
**Utilities to calculate Hamiltonian, separatrix, total voltage for the full ring.**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**
'''


from __future__ import division
from warnings import filterwarnings
import numpy as np
from scipy.constants import c



def total_voltage(RFsection_list, harmonic = 'first'):
    """
    Total voltage from all the RF stations and systems in the ring.
    To be generalized.
    """
    
    n_sections = len(RFsection_list)
    
    #: *Sums up only the voltage of the first harmonic RF, 
    #: taking into account relative phases*
    if harmonic == 'first':
        Vtot = RFsection_list[0].voltage_program_list[0]
        if n_sections > 1:
            for i in range(1, n_sections):
                Vtot += RFsection_list[i].voltage_program_list[0] \
                * np.sin(RFsection_list[i].phi_offset_list[0] - RFsection_list[0].phi_offset_list[0])
        return Vtot
    
    #: *To be implemented*
    elif harmonic == "all":
        return 0

    else:
        print 'WARNING: In total_voltage, harmonic choice not recognize!'
    


def hamiltonian(GeneralParameters, RingAndRFSection, theta, dE, delta, 
                total_voltage = None):
    """Single RF sinusoidal Hamiltonian.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
   
    if GeneralParameters.n_sections > 1:
        print 'WARNING : The hamiltonian is not yet properly computed for several sections !!!'
    if RingAndRFSection.n_rf > 1:
        print 'WARNING: The Hamiltonian will be calculated for the first harmonic only!!!'
    filterwarnings('ignore')
         
    counter = RingAndRFSection.counter
    h0 = RingAndRFSection.harmonic_list[counter]
    if total_voltage == None:
        V0 = RingAndRFSection.voltage_list[counter]
    else: 
        V0 = total_voltage[counter]
    
    c1 = RingAndRFSection.eta_tracking(delta) * c * np.pi / (GeneralParameters.ring_circumference * 
         RingAndRFSection.beta_r[counter] * RingAndRFSection.energy[counter] )
    c2 = c * RingAndRFSection.beta_r[counter] * V0 / (h0 * GeneralParameters.ring_circumference)
     
    phi_s = RingAndRFSection.phi_s[counter]  
 
    filterwarnings('default')
    
    return c1 * dE**2 + c2 * (np.cos(h0 * theta) - np.cos(phi_s) + 
                               (h0 * theta - phi_s) * np.sin(phi_s))
         
 
 
def separatrix(GeneralParameters, RingAndRFSection, theta, total_voltage = None):
    """Single RF sinusoidal separatrix.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
 
     
    if GeneralParameters.n_sections > 1:
        print 'WARNING : The hamiltonian is not yet properly computed for several sections !!!'
    if RingAndRFSection.n_rf > 1:
        print 'WARNING: The Hamiltonian will be calculated for the first harmonic only!!!'    
    filterwarnings('ignore')
     
     
    counter = RingAndRFSection.counter
    h0 = RingAndRFSection.harmonic_list[counter]
    if total_voltage == None:
        V0 = RingAndRFSection.voltage_list[counter]
    else: 
        V0 = total_voltage[counter]
 
    phi_s = RingAndRFSection.phi_s[counter]  
      
     
    beta_average = RingAndRFSection.beta_av[counter]
     
    energy_average = (RingAndRFSection.energy[counter + 1] + RingAndRFSection.energy[counter]) / 2
     
    eta0_average = (RingAndRFSection.eta_0[counter + 1] + RingAndRFSection.eta_0[counter])/2
      
    separatrix_array = np.sqrt(beta_average**2 * energy_average *
                    V0 / (np.pi * eta0_average * h0) * 
                    (-np.cos(h0 * theta) - np.cos(phi_s) + 
                    (np.pi - phi_s - h0 * theta) * np.sin(phi_s)))
      
    filterwarnings('default')
         
    return separatrix_array
 
 
 
def is_in_separatrix(GeneralParameters, RingAndRFSection, theta, dE, delta, total_voltage = None):
    """Condition for being inside the separatrix.
    For the time being, for single RF section only or from total voltage.
    Single RF sinusoidal.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
    if GeneralParameters.n_sections > 1:
        print 'WARNING : The hamiltonian is not yet properly computed for several sections !!!'
    if RingAndRFSection.n_rf > 1:
        print 'WARNING: The Hamiltonian will be calculated for the first harmonic only!!!'
    filterwarnings('ignore')
    
         
    counter = RingAndRFSection.counter
    h0 = RingAndRFSection.harmonic_list[counter]        
    phi_s = RingAndRFSection.phi_s[counter] 
     
    Hsep = hamiltonian(GeneralParameters, RingAndRFSection, (np.pi - phi_s) / h0, 0, 0, total_voltage = None) 
    isin = np.fabs(hamiltonian(GeneralParameters, RingAndRFSection, theta, dE, delta, total_voltage = None)) < np.fabs(Hsep)
 
    return isin
        
        
