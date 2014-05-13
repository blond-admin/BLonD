'''
Created on 06.01.2014

@author: Kevin Li, Hannes Bartosik
'''


import numpy as np
import cobra_functions.stats as cp


class Slices(object):
    '''
    classdocs
    '''

    def __init__(self, n_slices, nsigmaz=None, mode='cspace'):
        '''
        Constructor
        '''
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

        self.n_macroparticles = np.zeros(n_slices, dtype=int)
        self.z_bins = np.zeros(n_slices + 1)

    
    def n_slices(self):

        return len(self.mean_x)

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

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
        bunch.sort_particles()

        # determine the longitudinal cuts (this allows for the user defined static cuts: self.z_cut_tail, self.z_cut_head)
        try:
            self.z_cut_tail, self.z_cut_head
        except AttributeError:
            self.z_cut_tail, self.z_cut_head = self._set_longitudinal_cuts(bunch)

        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        # 1. z-bins
        self.z_bins[:] = np.linspace(self.z_cut_tail, self.z_cut_head, self.n_slices + 1) # more robust than arange, to reach z_cut_head exactly
        self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.

        # 2. n_macroparticles - equivalet to x0 <= x < x1 binning
        z_bins_all = np.hstack((bunch.dz[0], self.z_bins, bunch.dz[n_macroparticles_alive - 1]))
        first_index_in_bin = np.searchsorted(bunch.dz[:n_macroparticles_alive], z_bins_all)
        first_index_in_bin[np.where(z_bins_all == bunch.dz[-1 - bunch.n_macroparticles_lost])] += 1 # treat last bin for x0 <= x <= x1
        self.z_index = first_index_in_bin[1:-2]

        n_macroparticles = np.diff(first_index_in_bin)
        self.n_cut_tail = n_macroparticles[0]
        self.n_cut_head = n_macroparticles[-1]
        self.n_macroparticles[:] = n_macroparticles[1:-1]

       
    def slice_constant_charge(self, bunch):

        # sort particles according to dz (this is needed for correct functioning of bunch.compute_statistics)
     
        bunch.sort_particles()
        
        # determine the longitudinal cuts (this allows for the user defined static cuts: self.z_cut_tail, self.z_cut_head)
        try:
            self.z_cut_tail, self.z_cut_head
        except AttributeError:
            self.z_cut_tail, self.z_cut_head = self._set_longitudinal_cuts(bunch)
            
        n_macroparticles_alive = bunch.n_macroparticles - bunch.n_macroparticles_lost
        # 1. n_macroparticles
        self.n_cut_tail = np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_cut_tail)
        self.n_cut_head = n_macroparticles_alive - (np.searchsorted(bunch.dz[:n_macroparticles_alive], self.z_cut_head) ) # always throw last index into slices (x0 <= x <= x1)
        
        # distribute macroparticles uniformly along slices
        q0 = n_macroparticles_alive - self.n_cut_tail - self.n_cut_head
        
        self.n_macroparticles[:] = q0 // self.n_slices
        
        x = sample(range(self.n_slices), (q0 % self.n_slices))
        
        self.n_macroparticles[x] += 1
        
        # 2. z-bins
        # Get indices of the particles defining the bin edges
        n_macroparticles_all = np.hstack((self.n_cut_tail, self.n_macroparticles, self.n_cut_head))
        first_index_in_bin = np.append(0, np.cumsum(n_macroparticles_all))
        # first_index_in_bin[np.where(z_bins_all == bunch.dz[-1 - bunch.n_macroparticles_lost])] += 1 # treat last bin for x0 <= x <= x1
        self.z_index = first_index_in_bin[1:-2]
        self.z_bins = map(lambda i: bunch.dz[self.z_index[i] - 1] + (bunch.dz[self.z_index[i]] - bunch.dz[self.z_index[i] - 1]) / 2,
                          np.arange(1, self.n_slices))
        self.z_bins = np.hstack((self.z_cut_tail, self.z_bins, self.z_cut_head))
        self.z_centers = self.z_bins[:-1] + (self.z_bins[1:] - self.z_bins[:-1]) / 2.
        

    def update_slices(self, bunch):

        if self.mode == 'ccharge':
            self.slice_constant_charge(bunch)
        elif self.mode == 'cspace':
            self.slice_constant_space(bunch)


    def compute_statistics(self):

        # determine the start and end indices of each slices
        i1 = np.append(np.cumsum(self.slices.n_macroparticles[:-2]), np.cumsum(self.slices.n_macroparticles[-2:]))
        i0 = np.zeros(len(i1), dtype=np.int)
        i0[1:] = i1[:-1]
        i0[-2] = 0

        for i in xrange(self.slices.n_slices + 4):
            x = self.x[i0[i]:i1[i]]
            xp = self.xp[i0[i]:i1[i]]
            y = self.y[i0[i]:i1[i]]
            yp = self.yp[i0[i]:i1[i]]
            dz = self.dz[i0[i]:i1[i]]
            dp = self.dp[i0[i]:i1[i]]

            self.slices.mean_x[i] = cp.mean(x)
            self.slices.mean_xp[i] = cp.mean(xp)
            self.slices.mean_y[i] = cp.mean(y)
            self.slices.mean_yp[i] = cp.mean(yp)
            self.slices.mean_dz[i] = cp.mean(dz)
            self.slices.mean_dp[i] = cp.mean(dp)

            self.slices.sigma_x[i] = cp.std(x)
            self.slices.sigma_y[i] = cp.std(y)
            self.slices.sigma_dz[i] = cp.std(dz)
            self.slices.sigma_dp[i] = cp.std(dp)

            self.slices.epsn_x[i] = cp.emittance(x, xp) * self.gamma * self.beta * 1e6
            self.slices.epsn_y[i] = cp.emittance(y, yp) * self.gamma * self.beta * 1e6
            self.slices.epsn_z[i] = 4 * np.pi \
                                  * self.slices.sigma_dz[i] * self.slices.sigma_dp[i] \
                                  * self.mass * self.gamma * self.beta * c / e

   