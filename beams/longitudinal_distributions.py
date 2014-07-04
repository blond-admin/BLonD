'''
Created on 12.06.2014

@author: Danilo Quartullo, Helga Timko, Alexandre Lasheen
'''

import numpy as np
from scipy.constants import c
from trackers.longitudinal_tracker import is_in_separatrix

# def stationary_exponential(H, Hmax, H0, bunch):
# 
#     def psi(dz, dp):
#         result = np.exp(H(dz, dp, bunch) / H0) - np.exp(Hmax / H0)
#         return result
# 
#     return psi

# def _match_simple_gaussian_longitudinal(self, beta_z, sigma_z=None, epsn_z=None):
# 
#         if sigma_z and epsn_z:
#             sigma_delta = epsn_z / (4 * np.pi * sigma_z) * e / self.p0
#             if sigma_z / sigma_delta != beta_z:
#                 print '*** WARNING: beam mismatched in bucket. Set synchrotron tune as to obtain beta_z = ', sigma_z / sigma_delta
#         elif not sigma_z and epsn_z:
#             sigma_z = np.sqrt(beta_z * epsn_z / (4 * np.pi) * e / self.p0)
#             sigma_delta = sigma_z / beta_z
#         else:
#             sigma_delta = sigma_z / beta_z
# 
#         self.z *= sigma_z
#         self.delta *= sigma_delta
        
        
def longitudinal_bigaussian(GeneralParameters, RingAndRFSection, beam, sigma_x,
                             sigma_y, xunit=None, yunit=None):
    
    if RingAndRFSection.drift.drift_length != GeneralParameters.ring_circumference:
        raise RuntimeError('WARNING : The longitudinal_gaussian_matched is not\
         yet properly computed for several sections !!!')
        
    if RingAndRFSection.kick.n_rf_systems != 1:
        raise RuntimeError('longitudinal_gaussian_matched for multiple RF is \
        not implemeted yet')
    
    counter = GeneralParameters.counter[0]
    harmonic = RingAndRFSection.kick.harmonic_number_list[0][counter]
    energy = GeneralParameters.energy_program[0][counter]
    beta = GeneralParameters.beta_rel_program[0][counter]
    
    if xunit == None or xunit == 'rad':
        sigma_theta = sigma_x
    elif xunit == 'm':
        sigma_theta = sigma_x / (- beam.ring_radius * harmonic) 
    elif xunit == 'ns':       
        sigma_theta = sigma_x * beta * c * 1.e-9 / beam.ring_radius
        
    if yunit == None or yunit == 'eV':
        sigma_dE = sigma_y
    elif yunit == '1':    
        sigma_dE = sigma_y * beta**2 * energy
        
    
    beam.sigma_theta = sigma_theta
    beam.sigma_dE = sigma_dE
    phi_s = RingAndRFSection.phi_s[counter]

    beam.theta = sigma_theta * np.random.randn(beam.n_macroparticles) \
                        + phi_s/harmonic
    beam.dE = sigma_dE * np.random.randn(beam.n_macroparticles)
    
    itemindex = np.where(is_in_separatrix(GeneralParameters, RingAndRFSection,
                                 beam.theta, beam.dE, beam.delta) == False)[0]
    
    while itemindex.size != 0:
    
        beam.theta[itemindex] = sigma_theta * np.random.randn(itemindex.size) \
                + phi_s/harmonic
        beam.dE[itemindex] = sigma_dE * np.random.randn(itemindex.size)
        itemindex = np.where(is_in_separatrix(GeneralParameters, 
                            RingAndRFSection, beam.theta, beam.dE, beam.delta) 
                             == False)[0]

  

def longitudinal_gaussian_matched(GeneralParameters, RingAndRFSection, beam, 
                                  four_sigma_bunch_length, unit=None):
    
    
    if RingAndRFSection.drift.drift_length != GeneralParameters.ring_circumference:
        raise RuntimeError('WARNING : The longitudinal_gaussian_matched is not\
         yet properly computed for several sections !!!')
        
    if RingAndRFSection.kick.n_rf_systems != 1:
        raise RuntimeError('longitudinal_gaussian_matched for multiple RF is \
        not implemeted yet')
    
    counter = GeneralParameters.counter[0]
    harmonic = RingAndRFSection.kick.harmonic_number_list[0][counter]
    voltage = RingAndRFSection.kick.voltage_program_list[0][counter]
    energy = GeneralParameters.energy_program[0][counter]
    beta = GeneralParameters.beta_rel_program[0][counter]
    eta0 = GeneralParameters.eta0[0][counter]
            
    if unit == None or unit == 'rad':
        sigma_theta = four_sigma_bunch_length / 4
    elif unit == 'm':
        sigma_theta = four_sigma_bunch_length / (-4 * GeneralParameters.ring_radius) 
    elif unit == 'ns':       
        sigma_theta = four_sigma_bunch_length * beta * c * \
        0.25e-9 / GeneralParameters.ring_radius
    
    phi_s = RingAndRFSection.phi_s[counter]
  
    phi_b = harmonic*sigma_theta + phi_s
    
    sigma_dE = np.sqrt( voltage * energy * beta**2  
             * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s)) 
             / (np.pi * harmonic * eta0) )
        
    beam.sigma_theta = sigma_theta
    beam.sigma_dE = sigma_dE
    
    beam.theta = sigma_theta * np.random.randn(beam.n_macroparticles) \
                        + phi_s/harmonic
    beam.dE = sigma_dE * np.random.randn(beam.n_macroparticles)
    
    itemindex = np.where(is_in_separatrix(GeneralParameters, RingAndRFSection,
                                 beam.theta, beam.dE, beam.delta) == False)[0]
    
    while itemindex.size != 0:
    
        beam.theta[itemindex] = sigma_theta * np.random.randn(itemindex.size) \
                + phi_s/harmonic
        beam.dE[itemindex] = sigma_dE * np.random.randn(itemindex.size)
        itemindex = np.where(is_in_separatrix(GeneralParameters, 
                            RingAndRFSection, beam.theta, beam.dE, beam.delta) 
                             == False)[0]
    
    


