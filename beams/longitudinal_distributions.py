'''
Created on 12.06.2014

@author: Danilo Quartullo, Helga Timko
'''

import numpy as np


# def stationary_exponential(H, Hmax, H0, bunch):
# 
#     def psi(dz, dp):
#         result = np.exp(H(dz, dp, bunch) / H0) - np.exp(Hmax / H0)
#         return result
# 
#     return psi

def _match_simple_gaussian_longitudinal(self, beta_z, sigma_z=None, epsn_z=None):

        if sigma_z and epsn_z:
            sigma_delta = epsn_z / (4 * np.pi * sigma_z) * e / self.p0
            if sigma_z / sigma_delta != beta_z:
                print '*** WARNING: beam mismatched in bucket. Set synchrotron tune as to obtain beta_z = ', sigma_z / sigma_delta
        elif not sigma_z and epsn_z:
            sigma_z = np.sqrt(beta_z * epsn_z / (4 * np.pi) * e / self.p0)
            sigma_delta = sigma_z / beta_z
        else:
            sigma_delta = sigma_z / beta_z

        self.z *= sigma_z
        self.delta *= sigma_delta