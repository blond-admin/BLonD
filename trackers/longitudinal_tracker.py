from __future__ import division
import numpy as np
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e


class LongitudinalMap(object):
    
    """A longitudinal map represents a longitudinal dynamical element 
    (e.g. a kick or a drift...), i.e. an abstraction of a cavity 
    of an RF system etc.
    LongitudinalMap objects can compose a longitudinal one turn map!
    Definitions of various orders of the slippage factor eta(delta)
    for delta = (Delta p / p0) should be implemented in this class. 
    Any derived objects will access self.eta(beam).
    
    Note: the momentum compaction factors are defined by the change of radius
        \Delta R / R0 = \sum_i \\alpha_i * \delta^(i + 1)
        hence yielding expressions for the higher slippage factor orders
        \Delta w / w0 = \sum_j  \eta_j  * \delta^(i + 1)
        (for the revolution frequency w)
    """
    __metaclass__ = ABCMeta

    def __init__(self, alpha_array, momentum_program_array):
        
        """The length of the momentum compaction factor array /alpha_array/
        defines the order of the slippage factor expansion. """
        self.alpha_array = alpha_array
        self.momentum_program_array = momentum_program_array

    @abstractmethod
    def track(self, beam):
        
        pass

    def eta(self, delta, beam):
        
        """Depending on the number of entries in self.alpha_array the 
        according order of \eta = \sum_i \eta_i * \delta^i where
        \delta = \Delta p / p0 will be included in this gathering function.

        Note: Please implement higher slippage factor orders as static methods
        with name _eta<N> where <N> is the order of delta in eta(delta)
        and with signature (alpha_array, beam).
        """
        eta = 0
        for i in xrange( len(self.alpha_array) ):   # order = len - 1
            eta_i = getattr(self, '_eta' + str(i))(self.alpha_array, beam)
            eta  += eta_i * (delta ** i)
        return eta

    @staticmethod
    def _eta0(alpha_array, beam):
        
        return alpha_array[0] - beam.gamma ** -2


class Kick(LongitudinalMap):
    
    """The Kick class represents the kick by a single RF element in a ring!
    The kick (i.e. Delta dp) of the particle's dp coordinate is given by
    the (separable) Hamiltonian derived by z, i.e. the force.

    self.p_increment is the momentum step per turn of the synchronous particle,
        it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.

    self.phi_offset reflects an offset of the cavity's reference system."""

    def __init__(self, alpha_array, circumference, harmonic, voltage, phi_offset = 0):
        
        super(Kick, self).__init__(alpha_array)
        self.circumference = circumference
        self.harmonic = harmonic
        self.voltage = voltage
        self.phi_offset = phi_offset
        
    def track(self, beam):
        
        beam.dE += e * self.voltage * np.sin(self.harmonic * beam.theta + self.phi_offset)


class Kick_acceleration(LongitudinalMap):
    
    def __init__(self, voltage, p_increment = 0, beam):
        
        self.voltage = voltage
        self.p_increment = p_increment
        self.beta_old = beam.beta

    def track(self, beam):
        
        """Using the average beta during the acceleration to account for 
        the change in momentum over one turn."""
        beam.dE += - (self.beta_old + beam.beta) / 2 * c * self.p_increment
        self.beta_old = beam.beta

    def calc_phi_s(self, beam):
        """The synchronous phase calculated from the rate of momentum change.
        Below transition, for decelerating bucket: phi_s is in (-Pi/2,0)
        Below transition, for accelerating bucket: phi_s is in (0,Pi/2)
        Above transition, for accelerating bucket: phi_s is in (Pi/2,Pi)
        Above transition, for decelerating bucket: phi_s is in (Pi,3Pi/2)
        The synchronous phase is calculated at a certain moment."""
        V0 = self.voltage[0]
        phi_s = np.arcsin(beam.beta * c / (e * V0) * self.p_increment)
        if self.eta(0, beam) > 0:
            phi_s = np.pi - phi_s

        return phi_s
        

class Drift(LongitudinalMap):
    
    """the drift (i.e. Delta z) of the particle's z coordinate is given by
    the (separable) Hamiltonian derived by dp (defined by (p - p0) / p0).

    self.length is the drift length,
    self.beta_factor is the change ratio of \\beta_{n+1} / \\beta_n
    which can often be neglected (and be set to one). [Otherwise it may
    continuously be adapted by the user according to Kick.p_increment.]
    """

    def __init__(self, alpha_array, length, beam):
        
        super(Drift, self).__init__(alpha_array)
        self.length = length
        self.beta_old = beam.beta
        
    def track(self, beam):
        
        beam.theta = beam.beta / self.beta_old  * beam.theta + 2 * np.pi / (1 - self.eta(beam.delta, beam) * beam.delta)
        self.beta_old = beam.beta

class LongitudinalOneTurnMap(LongitudinalMap):
    
    """A longitudinal one turn map tracks over a complete turn.
    Any inheriting classes guarantee to provide a self.track(beam) method that 
    tracks around the whole ring!

    LongitudinalOneTurnMap classes possibly comprise several 
    LongitudinalMap objects."""

    __metaclass__ = ABCMeta

    def __init__(self, alpha_array, circumference):
        
        """LongitudinalOneTurnMap objects know their circumference: 
        this is THE ONE place to store the circumference in the simulations!"""
        super(LongitudinalOneTurnMap, self).__init__(alpha_array)
        self.circumference = circumference

    @abstractmethod
    def track(self, beam):
        
        """Contract: advances the longitudinal coordinates 
        of the beam over a full turn."""
        pass


class RFSystems(LongitudinalOneTurnMap):
    
    """
        With one RFSystems object in the ring layout (with all kicks applied 
        at the same longitudinal position), the longitudinal separatrix function 
        is exact and makes a valid local statement about stability!
    """

    def __init__(self, circumference, harmonic_list, voltage_list, phi_offset_list, alpha_array, momentum_program_array):
        
        """The first entry in harmonic_list, voltage_list and phi_offset_list
        defines the parameters for the one accelerating Kick object 
        (i.e. the accelerating RF system).

        The length of the momentum compaction factor array alpha_array
        defines the order of the slippage factor expansion. 
        See the LongitudinalMap class for further details.

        self.p_increment is the momentum step per turn of the synchronous 
        particle, it can be continuously adjusted to reflect different slopes 
        in the dipole magnet strength ramp.
        See the Kick class for further details.
        self.kicks
        self.elements
        self.fundamental_kick
        self.accelerating_kick"""

        self.harmonic_list = harmonic_list
        self.voltage_list = voltage_list
        self.circumference = circumference

        super(RFSystems, self).__init__(alpha_array, circumference)

        if not len(harmonic_list) == len(voltage_list) == len(phi_offset_list):
            print ("Warning: parameter lists for RFSystems do not have the same length!")


        """Separating the kicks from the RF and the magnets.
        kick can contain multiple contributions
        kick_acceleration is only used once per time step"""
        self.kicks = []
        for h, V, dphi in zip(harmonic_list, voltage_list, phi_offset_list):
            kick = Kick(alpha_array, self.circumference, h, V, dphi)
            self.kicks.append(kick)
        self.kick_acceleration = Kick_acceleration()
        self.elements = self.kicks + [self.kick_acceleration] + [Drift(alpha_array, self.circumference)]
        self.turn_number = 0
        
    def track(self, beam):
        
        self.kick_acceleration.p_increment = self.momentum_program_array(self.turn_number+1) - self.momentum_program_array(self.turn_number)
        beam.p0 = self.momentum_program_array(self.turn_number+1)
        for longMap in self.elements:
            longMap.track(beam)
        self.turn_number += 1
        
    def _shrink_transverse_emittance(self,beam, geo_emittance_factor):
        
        """accounts for the transverse geometrical emittance shrinking"""
        beam.x *= geo_emittance_factor
        beam.xp *= geo_emittance_factor
        beam.y *= geo_emittance_factor
        beam.yp *= geo_emittance_factor

#    def potential(self, z, beam):
        
#        """the potential well of the rf system"""
#        phi_0 = self.accelerating_kick.calc_phi_0(beam)
#        h1 = self.accelerating_kick.harmonic
#        def fetch_potential(kick):
#            phi_0_i = kick.harmonic / h1 * phi_0
#            return kick.potential(z, beam, phi_0_i)
#        potential_list = map(fetch_potential, self.kicks)
#        return sum(potential_list)

#    def hamiltonian(self, z, dp, beam):
        
#        """the full separable Hamiltonian of the RF system.
#        Its zero value is located at the fundamental separatrix
#        (between bound and unbound motion)."""
#        kinetic = -0.5 * self.eta(dp, beam) * beam.beta * c * dp ** 2
#        return kinetic + self.potential(z, beam)

#    def separatrix(self, z, beam):
        
#        """Returns the separatrix delta_sep = (p - p0) / p0 for the synchronous 
#        particle (since eta depends on delta, inverting the separatrix equation 
#        0 = H(z_sep, dp_sep) becomes inexplicit in general)."""
#        return np.sqrt(2 / (beam.beta * c * self.eta(0, beam)) * self.potential(z, beam))

    def hamiltonian(self, theta, dE, delta, beam):
        """Single RF sinusoidal Hamiltonian.
        To be generalized."""
        h0 = self.harmonic_list[0]
        V0 = self.voltage_list[0]
        c1 = self.eta(delta, beam) * c * np.pi / (self.circumference * 
                                                  beam.beta * beam.energy )
        c2 = c * e * V0 / (h0 * self.circumference)
        phi_s = self.calc_phi_s(beam)

        return c1 * dE**2 + c2 * (np.cos(h0 * theta) - np.cos(phi_s) + 
                                  (h0 * theta) * np.sin(phi_s))

    def separatrix(self, theta, beam):
        """Single RF sinusoidal separatrix.
        To be generalized."""
        h0 = self.harmonic_list[0]
        V0 = self.voltage_list[0]
        phi_s = self.calc_phi_s(beam)
        return np.sqrt(beam.beta**2 * beam.energy * e * V0 / 
                       (np.pi * self.eta(0, beam) * h0) * 
                       (-np.cos(h0 * theta) - np.cos(phi_s) + 
                         (np.pi - phi_s - h0 * theta) * np.sin(phi_s)))

    def is_in_separatrix(self, theta, dE, delta, beam):
        """Condition for being inside the separatrix.
        Single RF sinusoidal.
        To be generalized."""
        h0 = self.harmonic_list[0]
        phi_s = self.calc_phi_s(beam)
        Hsep = self.hamiltonian((np.pi - phi_s) / h0, 0, 0, beam) 
        isin = np.fabs(self.hamiltonian(theta, dE, delta, beam)) < np.fabs(Hsep)

        return isin


class LinearMap(LongitudinalOneTurnMap):
    
    '''
    Linear Map represented by a Courant-Snyder transportation matrix.
    self.alpha is the linear momentum compaction factor.
    '''

    def __init__(self, circumference, alpha, Qs):
        
        """alpha is the linear momentum compaction factor,
        Qs the synchroton tune."""
        self.circumference = circumference
        self.alpha = alpha
        self.Qs = Qs

    def track(self, beam):

        eta = self.alpha - beam.gamma ** -2

        omega_0 = 2 * np.pi * beam.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = np.cos(dQs)
        sindQs = np.sin(dQs)

        z0 = beam.z
        dp0 = beam.dp

        beam.z = z0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        beam.dp = dp0 * cosdQs + omega_s / eta / c * z0 * sindQs

