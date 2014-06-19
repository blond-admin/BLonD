'''
@authors: Hannes Bartosik,
          Kevin Li,
          Michael Schenk
@date:    06/01/2014
'''

import numpy as np
from random import sample
import cobra_functions.stats as cp
from scipy.constants import c



class Slices(object):
    '''
    Slices class that controlls longitudinal discretization of a bunch.
    '''

    def __init__(self, n_slices, n_sigma = None, cut_left = None, cut_right = None, unit = "theta", mode = 'const_space'):
        '''
        Constructor
        '''
        
        self.n_slices = n_slices
        self.n_sigma = n_sigma
        self.unit = unit
        self.mode = mode

        self.mean_x = np.zeros(n_slices)
        self.mean_xp = np.zeros(n_slices)
        self.mean_y = np.zeros(n_slices)
        self.mean_yp = np.zeros(n_slices)
        self.mean_theta = np.zeros(n_slices)
        self.mean_dE = np.zeros(n_slices)
        
        self.sigma_x = np.zeros(n_slices)
        self.sigma_y = np.zeros(n_slices)
        self.sigma_theta = np.zeros(n_slices)
        self.sigma_dE = np.zeros(n_slices)
        
        self.epsn_x = np.zeros(n_slices)
        self.epsn_y = np.zeros(n_slices)
        self.eps_rms_l = np.zeros(n_slices)
        
        if cut_left != None and cut_right != None:
            self.cut_left = cut_left
            self.cut_right = cut_right
            self.bins = np.linspace(cut_left, cut_right, self.n_slices + 1)
            self.centers = self.bins[:-1] + (self.bins[1:] - self.bins[:-1]) / 2.
        
    
    @property    
    def mean_z(self):
        return - self.mean_theta * self.ring.radius 
    @mean_z.setter
    def mean_z(self, value):
        self.mean_theta = - value / self.ring.radius 
    
    @property
    def mean_delta(self):
        return self.mean_dE / (self.ring.beta_i(self)**2 * self.ring.energy_i(self))
    @mean_delta.setter
    def mean_delta(self, value):
        self.mean_dE = value * self.ring.beta_i(self)**2 * self.ring.energy_i(self)

    @property    
    def sigma_z(self):
        return - self.sigma_theta * self.ring.radius 
    @sigma_z.setter
    def sigma_z(self, value):
        self.sigma_theta = - value / self.ring.radius 
    
    @property
    def sigma_delta(self):
        return self.sigma_dE / (self.ring.beta_i(self)**2 * self.ring.energy_i(self))
    @sigma_delta.setter
    def sigma_delta(self, value):
        self.sigma_dE = value * self.ring.beta_i(self)**2 * self.ring.energy_i(self)

    
    def _set_longitudinal_cuts(self, bunch):

        if self.n_sigma == None and self.unit == "theta":
            cut_left = bunch.theta[0]
            cut_right = bunch.theta[-1 - bunch.n_macroparticles_lost]
        elif self.n_sigma == None and self.unit == "z":
            cut_left = bunch.z[0]
            cut_right = bunch.z[-1 - bunch.n_macroparticles_lost]
        elif self.unit == "theta":
            mean_theta = cp.mean(bunch.theta[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            sigma_theta = cp.std(bunch.theta[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            cut_left = mean_theta - self.n_sigma * sigma_theta
            cut_right = mean_theta + self.n_sigma * sigma_theta
        else:
            mean_z = cp.mean(bunch.z[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            sigma_z = cp.std(bunch.z[:bunch.n_macroparticles - bunch.n_macroparticles_lost])
            cut_left = mean_z - self.n_sigma * sigma_z
            cut_right = mean_z + self.n_sigma * sigma_z

        return cut_left, cut_right


    def _slice_constant_space(self, bunch):
        
        self.sort_particles(bunch)

        try:
            cut_left, cut_right = self.cut_left, self.cut_right
        except AttributeError:
            cut_left, cut_right = self._set_longitudinal_cuts(bunch)
            self.bins = np.linspace(cut_left, cut_right, self.n_slices + 1)
            self.centers = self.bins[:-1] + (self.bins[1:] - self.bins[:-1]) / 2.
        
        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        
        if self.unit == 'z':
            
            self.n_cut_tail = np.searchsorted(bunch.z[:n_macroparticles_alive], cut_left)
            self.n_cut_head = np.searchsorted(bunch.z[:n_macroparticles_alive], cut_right)
    
            self.first_index_in_bin = np.searchsorted(bunch.z[:n_macroparticles_alive], self.bins)
            if (self.bins[-1] in bunch.z[:n_macroparticles_alive]): first_index_in_bin[-1] += 1
            
        else:
            self.n_cut_tail = np.searchsorted(bunch.theta[:n_macroparticles_alive], cut_left)
            self.n_cut_head = np.searchsorted(bunch.theta[:n_macroparticles_alive], cut_right)
    
            self.first_index_in_bin = np.searchsorted(bunch.theta[:n_macroparticles_alive], self.bins)
            if (self.bins[-1] in bunch.theta[:n_macroparticles_alive]): first_index_in_bin[-1] += 1
            
            
        self.n_macroparticles = np.diff(self.first_index_in_bin)

        self.set_slice_index_of_particle(bunch)
       

    def _slice_constant_charge(self, bunch):
        
        ################### TO BE CHECKED!!!!!!!!!
        
        
        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        self.sort_particles(bunch)


        # try:
        #     z_cut_tail, z_cut_head = self.z_cut_tail, self.z_cut_head
        # except AttributeError:
        z_cut_tail, z_cut_head = self._set_longitudinal_cuts(bunch)


        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        self.n_cut_tail = +np.searchsorted(bunch.z[:n_macroparticles_alive], z_cut_tail)
        self.n_cut_head = -np.searchsorted(bunch.z[:n_macroparticles_alive], z_cut_head) + n_macroparticles_alive


        # 1. n_macroparticles - distribute macroparticles uniformly along slices.
        # Must be integer. Distribute remaining particles randomly among slices with indices 'ix'.
        q0 = n_macroparticles_alive - self.n_cut_tail - self.n_cut_head
        ix = sample(range(self.n_slices), q0 % self.n_slices)


        self.n_macroparticles = (q0 // self.n_slices)*np.ones(self.n_slices)
        self.n_macroparticles[ix] += 1


        # 2. z-bins
        # Get indices of the particles defining the bin edges
        n_macroparticles_all = np.hstack((self.n_cut_tail, self.n_macroparticles, self.n_cut_head))
        first_index_in_bin = np.cumsum(n_macroparticles_all)
        self.z_index = first_index_in_bin[:-1]
        self.z_index = (self.z_index).astype(int)


        # print(self.z_index.shape)
        self.z_bins = (bunch.z[self.z_index - 1] + bunch.z[self.z_index]) / 2.
        self.z_bins[0], self.z_bins[-1] = z_cut_tail, z_cut_head
        self.z_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2.


        self._set_slice_index_of_particle(bunch)


        # # self.z_centers = map((lambda i: cp.mean(bunch.z[first_index_in_bin[i]:first_index_in_bin[i+1]])), np.arange(self.n_slices)


    def set_slice_index_of_particle(self, bunch):

        try:
            self.slice_index_of_particle
        except AttributeError:
            self.slice_index_of_particle = np.zeros(bunch.n_macroparticles, dtype=np.int)

        for i in range(self.n_slices):
            self.slice_index_of_particle[self.first_index_in_bin[i]:self.first_index_in_bin[i+1]] = i

    
    def track(self, bunch):
        
        bunch.beam_is_sliced = True
        if self.mode == 'const_charge':
            self._slice_constant_charge(bunch)
        elif self.mode == 'const_space':
            self._slice_constant_space(bunch)
        bunch.slicing = self


    def compute_statistics(self, bunch):

        index = self.n_cut_tail + np.cumsum(np.append(0, self.n_macroparticles))

        for i in xrange(self.n_slices):
           
            x  = bunch.x[index[i]:index[i + 1]]
            xp = bunch.xp[index[i]:index[i + 1]]
            y  = bunch.y[index[i]:index[i + 1]]
            yp = bunch.yp[index[i]:index[i + 1]]
            theta  = bunch.theta[index[i]:index[i + 1]]
            dE = bunch.dE[index[i]:index[i + 1]]


            self.mean_x[i] = cp.mean(x)
            self.mean_xp[i] = cp.mean(xp)
            self.mean_y[i] = cp.mean(y)
            self.mean_yp[i] = cp.mean(yp)
            self.mean_theta[i] = cp.mean(theta)
            self.mean_dE[i] = cp.mean(dE)


            self.sigma_x[i] = cp.std(x)
            self.sigma_y[i] = cp.std(y)
            self.sigma_theta[i] = cp.std(theta)
            self.sigma_dE[i] = cp.std(dE)


            self.epsn_x[i] = cp.emittance(x, xp) * bunch.ring.gamma_i(bunch) * bunch.ring.beta_i(bunch) * 1e6
            self.epsn_y[i] = cp.emittance(y, yp) * bunch.ring.gamma_i(bunch) * bunch.ring.beta_i(bunch) * 1e6
            self.eps_rms_l[i] = np.pi * self.sigma_dE[i] * self.sigma_theta[i] * bunch.ring.radius / (bunch.ring.beta_i(bunch) * c)

    
    def sort_particles(self, bunch):
       
        bunch.n_macroparticles_lost = (bunch.n_macroparticles - np.count_nonzero(bunch.id))
    
        if self.unit == 'theta':
            if bunch.n_macroparticles_lost:
                argsorted = np.lexsort((bunch.theta, -np.sign(bunch.id))) 
            else:
                argsorted = np.argsort(bunch.theta)
        else:
            if bunch.n_macroparticles_lost:
                argsorted = np.lexsort((bunch.z, -np.sign(bunch.id))) 
            else:
                argsorted = np.argsort(bunch.z)
    
        bunch.x = bunch.x.take(argsorted)
        bunch.xp = bunch.xp.take(argsorted)
        bunch.y = bunch.y.take(argsorted)
        bunch.yp = bunch.yp.take(argsorted)
        bunch.theta = bunch.theta.take(argsorted)
        bunch.dE = bunch.dE.take(argsorted)
        bunch.id = bunch.id.take(argsorted)

   