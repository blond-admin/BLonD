'''
**Utilities to calculate Hamiltonian, separatrix, total voltage for the full ring.**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**
'''


from __future__ import division
import warnings
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
        Vcos = RFsection_list[0].voltage[0]*np.cos(RFsection_list[0].phi_offset[0])
        Vsin = RFsection_list[0].voltage[0]*np.sin(RFsection_list[0].phi_offset[0])
        if n_sections > 1:
            for i in range(1, n_sections):
                print RFsection_list[i].voltage[0]
                Vcos += RFsection_list[i].voltage[0]*np.cos(RFsection_list[i].phi_offset[0])
                Vsin += RFsection_list[i].voltage[0]*np.sin(RFsection_list[i].phi_offset[0])
        Vtot = np.sqrt(Vcos**2 + Vsin**2)
        return Vtot
    
    #: *To be implemented*
    elif harmonic == "all":
        return 0

    else:
        warnings.filterwarnings("once")
        warnings.warn("WARNING: In total_voltage, harmonic choice not recognize!")
    


def hamiltonian(GeneralParameters, RFSectionParameters, theta, dE, delta, 
                total_voltage = None):
    """Single RF sinusoidal Hamiltonian.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
   
    warnings.filterwarnings("once")
    
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: The Hamiltonian is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("WARNING: The Hamiltonian will be calculated for the first harmonic only!")

         
    counter = RFSectionParameters.counter[0]
    h0 = RFSectionParameters.harmonic[0,counter]
    if total_voltage == None:
        V0 = RFSectionParameters.voltage[0,counter]
    else: 
        V0 = total_voltage[counter]
    
    c1 = RFSectionParameters.eta_tracking(delta) * c * np.pi / (GeneralParameters.ring_circumference * 
         RFSectionParameters.beta_r[counter] * RFSectionParameters.energy[counter] )
    c2 = c * RFSectionParameters.beta_r[counter] * V0 / (h0 * GeneralParameters.ring_circumference)
     
    phi_s = RFSectionParameters.phi_s[counter-1]  
    
    return c1 * dE**2 + c2 * (np.cos(h0 * theta) - np.cos(phi_s) + 
                               (h0 * theta - phi_s) * np.sin(phi_s))
         
 
 
def separatrix(GeneralParameters, RFSectionParameters, theta, total_voltage = None):
    """Single RF sinusoidal separatrix.
    For the time being, for single RF section only or from total voltage.
    Uses beta, energy averaged over the turn.
    To be generalized."""
 
    warnings.filterwarnings("once")
     
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: The separatrix is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("WARNING: The separatrix will be calculated for the first harmonic only!")    

     
     
    counter = RFSectionParameters.counter[0]
    h0 = RFSectionParameters.harmonic[0,counter]

    if total_voltage == None:
        V0 = RFSectionParameters.voltage[0,counter]
    else: 
        V0 = total_voltage[counter]
 
    phi_s = RFSectionParameters.phi_s[counter]  
      
     
    beta_average = RFSectionParameters.beta_av[counter]
     
    energy_average = (RFSectionParameters.energy[counter + 1] + RFSectionParameters.energy[counter]) / 2
     
    eta0_average = (RFSectionParameters.eta_0[counter + 1] + RFSectionParameters.eta_0[counter])/2
      
    separatrix_array = np.sqrt(beta_average**2 * energy_average *
                    V0 / (np.pi * eta0_average * h0) * 
                    (-np.cos(h0 * theta) - np.cos(phi_s) + 
                    (np.pi - phi_s - h0 * theta) * np.sin(phi_s)))
         
    return separatrix_array
 
 
 
def is_in_separatrix(GeneralParameters, RFSectionParameters, theta, dE, delta, total_voltage = None):
    """Condition for being inside the separatrix.
    For the time being, for single RF section only or from total voltage.
    Single RF sinusoidal.
    Uses beta, energy averaged over the turn.
    To be generalized."""
     
    warnings.filterwarnings("once")
    
    if GeneralParameters.n_sections > 1:
        warnings.warn("WARNING: is_in_separatrix is not yet properly computed for several sections!")
    if RFSectionParameters.n_rf > 1:
        warnings.warn("WARNING: is_in_separatrix will be calculated for the first harmonic only!")
    
         
    counter = RFSectionParameters.counter[0]
    h0 = RFSectionParameters.harmonic[0,counter]        
    phi_s = RFSectionParameters.phi_s[counter-1] 
     
    Hsep = hamiltonian(GeneralParameters, RFSectionParameters, (np.pi - phi_s) / h0, 0, 0, total_voltage = None) 
    isin = np.fabs(hamiltonian(GeneralParameters, RFSectionParameters, theta, dE, delta, total_voltage = None)) < np.fabs(Hsep)
     
    return isin
        
        
