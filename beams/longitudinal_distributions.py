'''
Created on 12.06.2014

@author: Danilo Quartullo, Helga Timko
'''

import numpy as np
from scipy.constants import c, e

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
        
        
def longitudinal_bigaussian(ring, beam, sigma_x, sigma_y, xunit=None, yunit=None):
    
    if xunit == None or xunit == 'rad':
        sigma_theta = sigma_x
    elif xunit == 'm':
        sigma_theta = sigma_x / (- beam.radius * beam.harmonic[0]) 
    elif xunit == 'ns':       
        sigma_theta = sigma_x * ring.beta_f(beam) * c * 1.e-9 / beam.radius
        
    if yunit == None or yunit == 'eV':
        sigma_dE = sigma_y
    elif yunit == '1':    
        sigma_dE = sigma_y * ring.beta_i(beam)**2 * ring.energy_i(beam)
        
    
    beam.sigma_theta = sigma_theta
    beam.sigma_dE = sigma_dE
    phi_s = ring.calc_phi_s(beam, ring.voltage)

    for i in xrange(beam.n_macroparticles):
        beam.theta[i] = sigma_theta * np.random.randn() \
                        + phi_s/ring.harmonic[0]
        beam.dE[i] = sigma_dE * np.random.randn()
        
        if not ring.is_in_separatrix(beam, beam.theta[i], beam.dE[i], beam.delta[i]): 
            while not ring.is_in_separatrix(beam, beam.theta[i], beam.dE[i], beam.delta[i]):
                beam.theta[i] = sigma_theta * np.random.randn() \
                        + phi_s/ring.harmonic[0]
                beam.dE[i] = sigma_dE * np.random.randn()

  

def longitudinal_gaussian_matched(ring, beam, four_sigma_bunch_length, unit=None):
    
    if unit == None or unit == 'rad':
        sigma_theta = four_sigma_bunch_length / 4
    elif unit == 'm':
        sigma_theta = four_sigma_bunch_length / (-4 * beam.radius * beam.harmonic[0]) 
    elif unit == 'ns':       
        sigma_theta = four_sigma_bunch_length * ring.beta_f(beam) * c * 4.e-9 / beam.radius  
        
    phi_s = ring.calc_phi_s(beam, ring.voltage)
    phi_b = ring.harmonic[0]*sigma_theta + phi_s
    print phi_s
    print phi_b   
    print np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s)
    print ""
    print ring.voltage[0]
    print ring.energy_i(beam)
    print ring.beta_i(beam)
    print ring._eta0(beam, ring.alpha_array)
    print ""
    
    sigma_dE = np.sqrt( ring.voltage[0] * ring.energy_i(beam) * ring.beta_i(beam)**2  
             * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s)) 
             / (np.pi * ring.harmonic[0] * ring._eta0(beam, ring.alpha_array)) )
    
    beam.sigma_theta = sigma_theta
    beam.sigma_dE = sigma_dE
    print sigma_dE

    for i in xrange(beam.n_macroparticles):
        beam.theta[i] = sigma_theta * np.random.randn() \
                        + phi_s/ring.harmonic[0]
        beam.dE[i] = sigma_dE * np.random.randn()
        
        if not ring.is_in_separatrix(beam, beam.theta[i], beam.dE[i], beam.delta[i]): 
            while not ring.is_in_separatrix(beam, beam.theta[i], beam.dE[i], beam.delta[i]):
                beam.theta[i] = sigma_theta * np.random.randn() \
                        + phi_s/ring.harmonic[0]
                beam.dE[i] = sigma_dE * np.random.randn()
    
    


