'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik, Danilo Quartullo
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
        
        
    def _set_longitudinal_cuts(self, beam):

        if self.nsigmaz == None:
            z_cut_tail = beam.dz[0]
            z_cut_head = beam.dz[-1 - beam.n_macroparticles_lost]
        else:
            mean_z = cp.mean(beam.dz[:beam.n_macroparticles - beam.n_macroparticles_lost])
            sigma_z = cp.std(beam.dz[:beam.n_macroparticles - beam.n_macroparticles_lost])
            z_cut_tail = mean_z - self.nsigmaz * sigma_z
            z_cut_head = mean_z + self.nsigmaz * sigma_z

        return z_cut_tail, z_cut_head

 
    def slice_constant_space(self, beam):

        self.sort_particles(beam)

        if self.dynamic_frame == "on":
            
            self.z_cut_tail, self.z_cut_head = self._set_longitudinal_cuts(beam)
            self.z_bins = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1) 
            self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        n_macroparticles_alive = beam.n_macroparticles - beam.n_macroparticles_lost
        self.first_index_in_bin = np.searchsorted(beam.z[:n_macroparticles_alive], self.z_bins)
        if (self.z_bins[-1] in beam.z[:n_macroparticles_alive]):
            self.first_index_in_bin[-1] += 1 
        
        self.n_macroparticles = np.diff(self.first_index_in_bin)
        self.cumsum_macroparticles = np.cumsum(self.n_macroparticles)
        self.n_cut_tail = self.first_index_in_bin[0]
        self.n_cut_head = n_macroparticles_alive - self.first_index_in_bin[-1]
        

    def slice_constant_charge(self, beam):

        
        self.sort_particles(beam)
        
        if self.dynamic_frame == "on":
        
            self.z_cut_tail, self.z_cut_head = self._set_longitudinal_cuts(beam)
            
        n_macroparticles_alive = beam.n_macroparticles - beam.n_macroparticles_lost
    
        self.n_cut_tail = np.searchsorted(beam.z[:n_macroparticles_alive], self.z_cut_tail)
        self.n_cut_head = n_macroparticles_alive - np.searchsorted(beam.z[:n_macroparticles_alive], self.z_cut_head) 
        
        q0 = n_macroparticles_alive - self.n_cut_tail - self.n_cut_head
        self.n_macroparticles[:] = q0 // self.n_slices
        x = sample(range(self.n_slices), q0 % self.n_slices)
        self.n_macroparticles[x] += 1
        
        self.cumsum_macroparticles = np.cumsum(self.n_macroparticles)
        indexes = (self.n_cut_tail + self.first_index_in_bin[:(self.n_slices-1)]).astype(int)
        z_bins_internal = 1/2 * (beam.z[indexes] + beam.z[indexes-1])
        
        self.z_bins = np.hstack((self.z_cut_tail, z_bins_internal, self.z_cut_head))
        self.first_index_in_bin = np.searchsorted(beam.z[:n_macroparticles_alive], self.z_bins)
        self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.
        

    def track(self, beam):

        if self.mode == 'const_charge':
            self.slice_constant_charge(beam)
        elif self.mode == 'const_space':
            self.slice_constant_space(beam)


    def compute_statistics(self, beam):

        index = self.n_cut_tail + np.cumsum(np.append(0, self.n_macroparticles))

        for i in xrange(self.n_slices):
            
            x = self.x[index[i]:index[i+1]]
            xp = self.xp[index[i]:index[i+1]]
            y = self.y[index[i]:index[i+1]]
            yp = self.yp[index[i]:index[i+1]]
            z = self.z[index[i]:index[i+1]]
            dp = self.dp[index[i]:index[i+1]]

            self.mean_x[i] = cp.mean(x)
            self.mean_xp[i] = cp.mean(xp)
            self.mean_y[i] = cp.mean(y)
            self.mean_yp[i] = cp.mean(yp)
            self.mean_z[i] = cp.mean(z)
            self.mean_dp[i] = cp.mean(dp)

            self.sigma_x[i] = cp.std(x)
            self.sigma_y[i] = cp.std(y)
            self.sigma_z[i] = cp.std(z)
            self.sigma_dp[i] = cp.std(dp)

            self.epsn_x[i] = cp.emittance(x, xp) * self.gamma * self.beta * 1e6
            self.epsn_y[i] = cp.emittance(y, yp) * self.gamma * self.beta * 1e6
            self.epsn_z[i] = 4 * np.pi \
                                  * self.slices.sigma_z[i] * self.slices.sigma_dp[i] \
                                  * self.mass * self.gamma * self.beta * c / e
    
    def sort_particles(self, beam):
       
        # update the number of lost particles
        beam.n_macroparticles_lost = (beam.n_macroparticles - np.count_nonzero(beam.id))
    
        # sort particles according to dz (this is needed for correct functioning of beam.compute_statistics)
        if beam.n_macroparticles_lost:
            dz_argsorted = np.lexsort((beam.z, -np.sign(beam.id))) # place lost particles at the end of the array
        else:
            dz_argsorted = np.argsort(beam.z)
    
        beam.x = beam.x.take(dz_argsorted)
        beam.xp = beam.xp.take(dz_argsorted)
        beam.y = beam.y.take(dz_argsorted)
        beam.yp = beam.yp.take(dz_argsorted)
        beam.z = beam.z.take(dz_argsorted)
        beam.dp = beam.dp.take(dz_argsorted)
        beam.id = beam.id.take(dz_argsorted)

   