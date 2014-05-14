'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik
'''

import numpy as np
from random import sample
import cobra_functions.stats as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, nsigmaz = None, mode = 'const_space', z_cut_tail = "null" , z_cut_head = "null"):
        '''
        Constructor
        '''
        
        self.n_slices = n_slices
        self.nsigmaz = nsigmaz
        self.mode = mode

        self.mean_x = np.zeros(n_slices)
        self.mean_xp = np.zeros(n_slices)
        self.mean_y = np.zeros(n_slices)
        self.mean_yp = np.zeros(n_slices)
        self.mean_dz = np.zeros(n_slices)
        self.mean_dp = np.zeros(n_slices)
        self.sigma_x = np.zeros(n_slices)
        self.sigma_y = np.zeros(n_slices)
        self.sigma_dz = np.zeros(n_slices)
        self.sigma_dp = np.zeros(n_slices)
        self.epsn_x = np.zeros(n_slices)
        self.epsn_y = np.zeros(n_slices)
        self.epsn_z = np.zeros(n_slices)

        self.n_macroparticles = np.zeros(n_slices)
        self.dynamic_frame = "on"
        
        if (z_cut_tail != "null" and z_cut_head != "null" and mode == 'const_space'):
            
            self.z_cut_tail = z_cut_tail
            self.z_cut_head = z_cut_head
            self.z_bins = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1) 
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.
            self.dynamic_frame = "off"
        
        if (z_cut_tail != "null" and z_cut_head != "null" and mode == 'const_charge'):
            
            self.z_cut_tail = z_cut_tail
            self.z_cut_head = z_cut_head
            self.dynamic_frame = "off"
        
        
    def _set_longitudinal_cuts(self, bunch):

        if self.nsigmaz == None:
            z_cut_tail = bunch.dz[0]
            z_cut_head = bunch.dz[-1 - bunch.n_macroparticles_lost]
        else:
            mean_z = cp.mean(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            sigma_z = cp.std(bunch.dz[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            z_cut_tail = mean_z - self.nsigmaz * sigma_z
            z_cut_head = mean_z + self.nsigmaz * sigma_z

        return z_cut_tail, z_cut_head

 
    def slice_constant_space(self, bunch):

        self.sort_particles(bunch)

        if self.dynamic_frame == "on":
            
            self.z_cut_tail, self.z_cut_head = self._set_longitudinal_cuts(bunch)
            self.z_bins = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1) 
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        first_index_in_bin = np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_bins)
        if (self.z_bins[-1] in bunch.dz[:n_macroparticles_alive]):
            first_index_in_bin[-1] += 1 
        
        self.n_macroparticles = np.diff(first_index_in_bin)
        self.n_cut_tail = first_index_in_bin[0]
        self.n_cut_head = n_macroparticles_alive - first_index_in_bin[-1]
        

    def slice_constant_charge(self, bunch):

        
        self.sort_particles(bunch)
        
        if self.dynamic_frame == "on":
        
            self.z_cut_tail, self.z_cut_head = self._set_longitudinal_cuts(bunch)
            
        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
    
        self.n_cut_tail = np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_cut_tail)
        self.n_cut_head = n_macroparticles_alive - np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_cut_head) 
        
        q0 = n_macroparticles_alive - self.n_cut_tail - self.n_cut_head
        self.n_macroparticles[:] = q0 // self.n_slices
        x = sample(range(self.n_slices), q0 % self.n_slices)
        self.n_macroparticles[x] += 1
        
        first_index_in_bin = np.cumsum(self.n_macroparticles)
        indexes = (self.n_cut_tail + first_index_in_bin[:(self.n_slices-1)]).astype(int)
        z_bins_internal = 1/2 * (bunch.dz[indexes] + bunch.dz[indexes-1])
        
        self.z_bins = np.hstack((self.z_cut_tail, z_bins_internal, self.z_cut_head))
        self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.
        

    def update_slices(self, bunch):

        if self.mode == 'const_charge':
            self.slice_constant_charge(bunch)
        elif self.mode == 'const_space':
            self.slice_constant_space(bunch)


    def compute_statistics(self, bunch):

        index = self.n_cut_tail + np.cumsum(np.append(0, self.n_macroparticles))

        for i in xrange(self.n_slices):
            
            x = self.x[index[i]:index[i+1]]
            xp = self.xp[index[i]:index[i+1]]
            y = self.y[index[i]:index[i+1]]
            yp = self.yp[index[i]:index[i+1]]
            dz = self.dz[index[i]:index[i+1]]
            dp = self.dp[index[i]:index[i+1]]

            self.mean_x[i] = cp.mean(x)
            self.mean_xp[i] = cp.mean(xp)
            self.mean_y[i] = cp.mean(y)
            self.mean_yp[i] = cp.mean(yp)
            self.mean_dz[i] = cp.mean(dz)
            self.mean_dp[i] = cp.mean(dp)

            self.sigma_x[i] = cp.std(x)
            self.sigma_y[i] = cp.std(y)
            self.sigma_dz[i] = cp.std(dz)
            self.sigma_dp[i] = cp.std(dp)

            self.epsn_x[i] = cp.emittance(x, xp) * self.gamma * self.beta * 1e6
            self.epsn_y[i] = cp.emittance(y, yp) * self.gamma * self.beta * 1e6
            self.epsn_z[i] = 4 * np.pi \
                                  * self.slices.sigma_dz[i] * self.slices.sigma_dp[i] \
                                  * self.mass * self.gamma * self.beta * c / e
    
    def sort_particles(self, bunch):
       
        # update the number of lost particles
        bunch.n_macroparticles_lost = (bunch.n_macroparticles - np.count_nonzero(bunch.id))
    
        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        if bunch.n_macroparticles_lost:
            dz_argsorted = np.lexsort((bunch.dz, -np.sign(bunch.id))) # place lost particles at the end of the array
        else:
            dz_argsorted = np.argsort(bunch.dz)
    
        bunch.x = bunch.x.take(dz_argsorted)
        bunch.xp = bunch.xp.take(dz_argsorted)
        bunch.y = bunch.y.take(dz_argsorted)
        bunch.yp = bunch.yp.take(dz_argsorted)
        bunch.dz = bunch.dz.take(dz_argsorted)
        bunch.dp = bunch.dp.take(dz_argsorted)
        bunch.id = bunch.id.take(dz_argsorted)

   