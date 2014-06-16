'''
Created on 06.01.2014

@author: Kevin Li
'''
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

def _create_empty(beam):

    beam.x = np.zeros(beam.n_macroparticles)
    beam.xp = np.zeros(beam.n_macroparticles)
    beam.y = np.zeros(beam.n_macroparticles)
    beam.yp = np.zeros(beam.n_macroparticles)
    beam.z = np.zeros(beam.n_macroparticles)
    beam.delta = np.zeros(beam.n_macroparticles)

def _create_gauss(beam):

    beam.x = np.random.randn(beam.n_macroparticles)
    beam.xp = np.random.randn(beam.n_macroparticles)
    beam.y = np.random.randn(beam.n_macroparticles)
    beam.yp = np.random.randn(beam.n_macroparticles)
    beam.z = np.random.randn(beam.n_macroparticles)
    beam.delta = np.random.randn(beam.n_macroparticles)
        
def _create_uniform(beam):

    beam.x = 2 * np.random.rand(beam.n_macroparticles) - 1
    beam.xp = 2 * np.random.rand(beam.n_macroparticles) - 1
    beam.y = 2 * np.random.rand(beam.n_macroparticles) - 1
    beam.yp = 2 * np.random.rand(beam.n_macroparticles) - 1
    beam.z = 2 * np.random.rand(beam.n_macroparticles) - 1
    beam.delta = 2 * np.random.rand(beam.n_macroparticles) - 1
        
def as_bunch(beam, alpha_x, beta_x, epsn_x, alpha_y, beta_y, epsn_y, beta_z, sigma_z=None, epsn_z=None):

    beam.alpha_x = alpha_x
    beam.beta_x = beta_x
    beam.epsn_x = epsn_x
    beam.alpha_y = alpha_y
    beam.beta_y = beta_y
    beam.epsn_y =  epsn_y
    beam.sigma_z = sigma_z
    
    _create_gauss(beam)
    

    # Transverse
    sigma_x = np.sqrt(beta_x * epsn_x * 1e-6 / (beam.gamma_i() * beam.beta_i()))
    sigma_xp = sigma_x / beta_x
    sigma_y = np.sqrt(beta_y * epsn_y * 1e-6 / (beam.gamma_i() * beam.beta_i()))
    sigma_yp = sigma_y / beta_y

    beam.x *= sigma_x
    beam.xp *= sigma_xp
    beam.y *= sigma_y
    beam.yp *= sigma_yp

    # Longitudinal
    # Assuming a gaussian-type stationary distribution: beta_z = eta * circumference / (2 * np.pi * Qs)
    if sigma_z and epsn_z:
        sigma_delta = epsn_z / (4 * np.pi * sigma_z) * e / beam.p0_i()
        if sigma_z / sigma_delta != beta_z:
            print '*** WARNING: beam mismatched in bucket. Set synchrotron tune to obtain beta_z = ', sigma_z / sigma_delta
    elif not sigma_z and epsn_z:
        sigma_z = np.sqrt(beta_z * epsn_z / (4 * np.pi) * e / beam.p0_i())
        sigma_delta = sigma_z / beta_z
    else:
        sigma_delta = sigma_z / beta_z

    beam.z *= sigma_z
    beam.delta *= sigma_delta

    
def as_cloud(beam, density, extent_x, extent_y, extent_z):

    _create_uniform(beam)

    # General
    beam.charge = e
   
    beam.intensity = density * extent_x * extent_y * extent_z
    beam.mass = m_e

    # Transverse
    beam.x *= extent_x
    beam.xp *= 0
    beam.y *= extent_y
    beam.yp *= 0
    beam.z *= extent_z
    beam.delta *= 0

    # Initial distribution
    beam.x0 = beam.x.copy()
    beam.xp0 = beam.xp.copy()
    beam.y0 = beam.y.copy()
    beam.yp0 = beam.yp.copy()
    beam.z0 = beam.z.copy()
    beam.delta0 = beam.delta.copy()

    
def as_ghost(beam):

    _create_uniform(beam)

def _match_simple_gaussian_transverse(beam):

        sigma_x = np.sqrt(beam.beta_x * beam.epsn_x * 1e-6 / (beam.gamma_i() * beam.beta_i()))
        sigma_xp = sigma_x / beam.beta_x
        sigma_y = np.sqrt(beam.beta_y * beam.epsn_y * 1e-6 / (beam.gamma_i() * beam.beta_i()))
        sigma_yp = sigma_y / beam.beta_y

        beam.x *= sigma_x
        beam.xp *= sigma_xp
        beam.y *= sigma_y
        beam.yp *= sigma_yp

    
