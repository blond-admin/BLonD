# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Module containing the fundamental beam class with methods to compute beam 
statistics

:Authors: **Danilo Quartullo**, **Helga Timko**, **ALexandre Lasheen**

"""

from __future__ import division
from builtins import object
import numpy as np
from scipy.constants import m_p, m_e, e, c
from ..trackers.utilities import is_in_separatrix

class Particle(object):

    def __init__(self, user_mass, user_charge):
        
        if user_mass > 0.:
            self.mass = float(user_mass)
            self.charge = float(user_charge)
        else:
            raise RuntimeError('ERROR: Particle mass not recognized!')
        
class Proton(Particle):
    
    def __init__(self):        
        
        Particle.__init__(self, float(m_p*c**2/e), np.float(1))

class Electron(Particle):
        
    def __init__(self):        
        self.mass =  float(m_e*c**2/e)
        self.charge = float(-1)




class Beam(object):
    """Class containing the beam properties.

    This class containes the beam coordinates (dt, dE) and the beam properties.
    
    The beam coordinate 'dt' is defined as the particle arrival time to the RF 
    station w.r.t. the reference time that is the sum of turns. The beam
    coordiate 'dE' is defined as the particle energy offset w.r.t. the
    energy of the synchronous particle.

    The class creates a beam with zero dt and dE, see distributions to match
    a beam with respect to the RF and intensity effects.
    
    Parameters
    ----------
    Ring : Ring
        Used to import different quantities such as the mass and the energy.
    n_macroparticles : int
        total number of macroparticles.
    intensity : float
        total intensity of the beam (in number of charge).
  
    Attributes
    ----------
    mass : float
        mass of the particle [eV].
    charge : int
        integer charge of the particle [e].
    beta : float
        relativistic velocity factor [].
    gamma : float
        relativistic mass factor [].
    energy : float
        energy of the synchronous particle [eV].
    momentum : float
        momentum of the synchronous particle [eV].
    dt : numpy_array, float
        beam arrival times with respect to synchronous time [s].
    dE : numpy_array, float
        beam energy offset with respect to the synchronous particle [eV].
    mean_dt : float
        average beam arrival time [s].
    mean_dE : float
        average beam energy offset [eV].
    sigma_dt : float
        standard deviation of beam arrival time [s].
    sigma_dE : float
        standard deviation of beam energy offset [eV].
    intensity : float
        total intensity of the beam in number of charges [].
    n_macroparticles : int
        total number of macroparticles in the beam [].
    ratio : float
        ratio intensity per macroparticle [].
    n_macroparticles_lost : int
        number of macro-particles marked as 'lost' [].
    id : numpy_array, int
        unique macro-particle ID number; zero if particle is 'lost'.
        
    See Also
    ---------
    distributions.matched_from_line_density:
        match a beam with a given bunch profile.
    distributions.matched_from_distribution_function:
        match a beam with a given distribution function in phase space.

    Examples
    --------
    >>> from input_parameters.ring import Ring
    >>> from beam.beam import Beam
    >>>
    >>> n_turns = 10
    >>> C = 100
    >>> eta = 0.03
    >>> momentum = 26e9
    >>> ring = Ring(n_turns, C, eta, momentum, 'proton')
    >>> n_macroparticle = 1e6
    >>> intensity = 1e11
    >>>
    >>> my_beam = Beam(ring, n_macroparticle, intensity)
    """

    def __init__(self, Ring, n_macroparticles, intensity):

        self.Particle = Ring.Particle
        self.beta = Ring.beta[0][0]
        self.gamma = Ring.gamma[0][0]
        self.energy = Ring.energy[0][0]
        self.momentum = Ring.momentum[0][0]
        self.dt = np.zeros([int(n_macroparticles)])
        self.dE = np.zeros([int(n_macroparticles)])
        self.mean_dt = 0.
        self.mean_dE = 0.
        self.sigma_dt = 0.
        self.sigma_dE = 0.
        self.intensity = float(intensity) 
        self.n_macroparticles = int(n_macroparticles)
        self.ratio = self.intensity/self.n_macroparticles
        self.id = np.arange(1, self.n_macroparticles + 1, dtype=int)

    @property
    def n_macroparticles_lost(self):
        '''Number of lost macro-particles, defined as @property.
        
        Returns
        -------        
        n_macroparticles_lost : int
            number of macroparticles lost.
            
        '''
        
        return len( np.where( self.id == 0 )[0] )

    @property
    def n_macroparticles_alive(self):
        '''Number of transmitted macro-particles, defined as @property.

        Returns
        -------        
        n_macroparticles_alive : int
            number of macroparticles not lost.
            
        '''

        return self.n_macroparticles - self.n_macroparticles_lost


    def eliminate_lost_particles(self):
        """Eliminate lost particles from the beam coordinate arrays
        """
        
        indexalive = np.where( self.id == 0 )[0]
        if len(indexalive) < self.n_macroparticles:
            self.dt = np.ascontiguousarray(self.beam.dt[indexalive])
            self.dE = np.ascontiguousarray(self.beam.dE[indexalive])
            self.n_macroparticles = len(self.beam.dt)
        else:
            raise RuntimeError("ERROR in Beams: all particles lost and"+
                " eliminated!")    

        
    def statistics(self):
        '''
        Calculation of the mean and standard deviation of beam coordinates,
        as well as beam emittance using different definitions.
        Take no arguments, statistics stored in

        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        '''

        # Statistics only for particles that are not flagged as lost
        itemindex = np.where(self.id != 0)[0]
        self.mean_dt = np.mean(self.dt[itemindex])
        self.mean_dE = np.mean(self.dE[itemindex])
        self.sigma_dt = np.std(self.dt[itemindex])
        self.sigma_dE = np.std(self.dE[itemindex])

        # R.m.s. emittance in Gaussian approximation
        self.epsn_rms_l = np.pi*self.sigma_dE*self.sigma_dt # in eVs



    def losses_separatrix(self, Ring, RFStation):
        '''Beam losses based on separatrix.

        Set to 0 all the particle's id not in the separatrix anymore.
        
        Parameters
        ----------
        Ring : Ring
            Used to call the function is_in_separatrix.
        RFStation : RFStation
            Used to call the function is_in_separatrix.
        '''

        itemindex = np.where(is_in_separatrix(Ring, RFStation, self, 
            self.dt, self.dE) == False)[0]

        if itemindex.size != 0:
            self.id[itemindex] = 0


    def losses_longitudinal_cut(self, dt_min, dt_max): 
        '''Beam losses based on longitudinal cuts.

        Set to 0 all the particle's id with dt not in the interval 
        (dt_min, dt_max).
        
        Parameters
        ----------
        dt_min : float
            minimum dt.
        dt_max : float
            maximum dt.
        '''

        itemindex = np.where( (self.dt - dt_min)*(dt_max - self.dt) < 0 )[0]

        if itemindex.size != 0:
            self.id[itemindex] = 0


    def losses_energy_cut(self, dE_min, dE_max): 
        '''Beam losses based on energy cuts, e.g. on collimators.

        Set to 0 all the particle's id with dE not in the interval (dE_min, dE_max).

        Parameters
        ----------
        dE_min : float
            minimum dE.
        dE_max : float
            maximum dE.
        '''

        itemindex = np.where( (self.dE - dE_min)*(dE_max - self.dE) < 0 )[0]

        if itemindex.size != 0:          
            self.id[itemindex] = 0 


    def losses_below_energy(self, dE_min): 
        '''Beam losses based on lower energy cut.

        Set to 0 all the particle's id with dE below dE_min.

        Parameters
        ----------
        dE_min : float
            minimum dE.
        '''

        itemindex = np.where( (self.dE - dE_min) < 0 )[0]

        if itemindex.size != 0:          
            self.id[itemindex] = 0 



