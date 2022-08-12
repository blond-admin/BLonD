# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering and processing all RF parameters used in the simulation.**

:Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

from __future__ import division, print_function
from builtins import str, range, object
import numpy as np
from scipy.constants import c
from scipy.integrate import cumtrapz
from ..beam.beam import Proton
from ..input_parameters.rf_parameters_options import RFStationOptions
from ..utils import bmath as bm


class RFStation(object):
    r""" Class containing all the RF parameters for all the RF systems in one
    ring segment or RF station.

    **How to use RF programs:**

    * For 1 RF system and constant values of V, h, or phi, input a single value
    * For 1 RF system and varying values of V, h, or phi, input an array of
      n_turns+1 values
    * For several RF systems and constant values of V, h, or phi, input lists
      of single values
    * For several RF systems and varying values of V, h, or phi, input lists
      of arrays of n_turns+1 values
    * For pre-processing, pass a list of times-voltages, times-harmonics,
      and/or times-phases for **each** RF system as a tuple
      ((time_1, voltage_1), (time_2, voltage_2), ...)

    Optional: RF frequency other than the design frequency. In this case, need
    to use a beam phase loop for correct RF phase!

    Optional: empty RFStation (e.g. for machines with synchrotron radiation);
    input voltage as 0.

    The index :math:`n` denotes time steps, :math:`l` the index of the RF
    systems in the section.

    **N.B. for negative eta the RF phase has to be shifted by Pi w.r.t the time
    reference.**

    Parameters
    ----------
    Ring : class
        A Ring type class
    harmonic : float (opt: float array/matrix, tuple of float array/matrix)
        Harmonic number of the RF system, :math:`h_{l,n}` [1]. For input
        options, see above
    voltage : float (opt: float array/matrix, tuple of float array/matrix)
        RF cavity voltage as seen by the beam, :math:`V_{l,n}` [V]. For input
        options, see above
    phi_rf_d : float (opt: float array/matrix, tuple of float array/matrix)
        Programmed/designed RF cavity phase,
        :math:`\phi_{d,l,n}` [rad]. For input options, see above
    n_rf : int
        Optional, Number of harmonic rf systems in the section :math:`l`.
        Becomes mandatory for several rf systems.
    section_index : int
        Optional, In case of several sections in the Ring object, this
        specifies after which section the rf station is located (to get the
        right momentum program etc.). Value should be in the range
        1..Ring.n_sections
    omega_rf : float (opt: float array/matrix)
        Optional, Sets the rf angular frequency program that does not follow
        the harmonic condition. For input options, see above.
    phi_noise : float (opt: float array/matrix)
        Optional, programmed RF cavity phase noise, :math:`\phi_{N,l,n}` [rad].
        Added to all RF systems in the station. For input options, see above
    phi_modulation : class (opt: iterable of classes)
        A PhaseModulation type class (or iterable of classes)
    RFStationOptions : class
        Optionnal, A RFStationOptions-based class defining smoothing,
        interpolation, etc. options for harmonic, voltage, and/or
        phi_rf_d programme to be interpolated to a turn-by-turn programme

    Attributes
    ----------
    counter : int
        Counter of the current simulation time step; defined as a list in
        order to be passed by reference
    section_index : int
        Unique index :math:`k` of the RF station the present class is defined
        for. Input in the range 1..n_sections (see
        :py:class:`input_parameters.ring.Ring`).
        Inside the code, indices 0..n_sections-1 are used.
    Particle : class
        Inherited from
        :py:attr:`input_parameters.ring.Ring.Particle`
    n_turns : int
        Inherited from
        :py:attr:`input_parameters.ring.Ring.n_turns`
    ring_circumference : float
        Inherited from
        :py:attr:`input_parameters.ring.Ring.ring_circumference`
    section_length : float
        Length :math:`L_k` of the RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.ring_length`
    length_ratio : float
        Fractional RF section length :math:`L_k/C`
    t_rev : float array [n_turns+1]
        Inherited from
        :py:attr:`input_parameters.ring.Ring.t_rev`
    momentum : float array [n_turns+1]
        Momentum program of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.momentum`
    beta : float array [n_turns+1]
        Relativistic beta of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.beta`
    gamma : float array [n_turns+1]
        Relativistic gamma of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.gamma`
    energy : float array [n_turns+1]
        Total energy of the present RF section; inherited from
        :py:attr:`input_parameters.ring.Ring.energy`
    delta_E : float array [n_turns+1]
        Time derivative of total energy of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.delta_E`
    alpha_order : int
        Inherited from
        :py:attr:`input_parameters.ring.Ring.alpha_order`
    charge : int
        Inherited from
        :py:attr:`beam.Particle.charge`
    alpha_0 : float array [n_turns+1]
        Zeroth order momentum compaction factor of the present section;
        inherited from
        :py:attr:`input_parameters.ring.Ring.alpha_0`
    alpha_1 : float array [n_turns+1]
        First order momentum compaction factor of the present section;
        inherited from
        :py:attr:`input_parameters.ring.Ring.alpha_1`
    alpha_2 : float array [n_turns+1]
        Second order momentum compaction factor of the present section;
        inherited from
        :py:attr:`input_parameters.ring.Ring.alpha_2`
    eta_0 : float array [n_turns+1]
        Zeroth order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.eta_0`
    eta_1 : float array [n_turns+1]
        First order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.eta_1`
    eta_2 : float array [n_turns+1]
        Second order slippage factor of the present section; inherited from
        :py:attr:`input_parameters.ring.Ring.eta_2`
    sign_eta_0 : float array
        Sign of the eta_0 array
    harmonic : float matrix [n_rf, n_turns+1]
        Harmonic number for each rf system,
        :math:`h_{l,n}` [1]
    voltage : float matrix [n_rf, n_turns+1]
        Actual rf voltage of each harmonic system,
        :math:`V_{rf,l,n}` [V]
    empty : bool
        Flag to specify if the RFStation is empty
    phi_rf_d : float matrix [n_rf, n_turns+1]
        Designed rf cavity phase of each harmonic system,
        :math:`\phi_{d,l,n}` [rad]
    phi_rf : float matrix [n_rf, n_turns+1]
        Actual RF cavity phase of each harmonic system used for tracking,
        :math:`\phi_{rf,l,n}` [rad]. Initially the same as the designed phase.
    omega_rf_d : float matrix [n_rf, n_turns+1]
        Design RF angular frequency of the RF systems in the station
        :math:`\omega_{d,l,n} = \frac{h_{l,n} \beta_{l,n} c}{R_{s,n}}` [Hz]
    omega_rf : float matrix [n_rf, n_turns+1]
        Actual RF angular frequency of the RF systems in the station
        :math:`\omega_{rf,l,n} = \frac{h_{l,n} \beta_{l,n} c}{R_{s,n}}` [Hz].
        Initially the same as the designed angular frequency.
    phi_noise : None or float matrix [n_rf, n_turns+1]
        Programmed cavity phase noise for each RF harmonic.
    phi_modulation : None or float matrix [n_rf, n_turns+1]
        Programmed cavity phase modulation for each RF harmonic.
    dphi_rf : float matrix [n_rf]
        Accumulated RF phase error of each harmonic system
        :math:`\Delta \phi_{rf,l,n}` [rad]
    t_rf : float matrix [n_rf, n_turns+1]
        RF period :math:`\frac{2 \pi}{\omega_{rf,l,n}}` [s]
    phi_s : float array [n_turns+1]
        Synchronous phase for this section (if not empty), calculated in
        :py:func:`input_parameters.rf_parameters.calculate_phi_s`
    Q_s : float array [n_turns+1]
        Synchrotron tune for this section (if not empty), calculated in
        :py:func:`input_parameters.rf_parameters.calculate_Q_s`
    omega_s0 : float array [n_turns+1]
        Central synchronous angular frequency corresponding to Q_s (single
        harmonic, no intensity effects)
        :math:`\omega_{s,0} = Q_s \omega_{\text{rev}}` [1/s], where
        :math:`\omega_{\text{rev}}` is defined in
        :py:class:`input_parameters.ring.Ring`)
    RFStationOptions : RFStationOptions()
        The RFStationOptions is kept as an attribute of the RFStationg object
        for further usage.


    Examples
    --------
    >>> # To declare a double-harmonic RF system for protons:
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha = 3.21e-4
    >>> momentum = 450e9
    >>> ring = Ring(n_turns, C, alpha, momentum)
    >>> rf_station = RFStation(ring, [35640, 71280], [6e6, 6e5], [0, 0], 2)

    """

    def __init__(self, Ring, harmonic, voltage, phi_rf_d, n_rf=1,
                 section_index=1, omega_rf=None, phi_noise=None,
                 phi_modulation=None, RFStationOptions=RFStationOptions()):



        # Different indices
        self.counter = [int(0)]
        self.section_index = int(section_index - 1)
        if self.section_index < 0 \
                or self.section_index > Ring.n_sections - 1:
            raise RuntimeError("ERROR in RFStation: section_index out of" +
                               " allowed range!")
        self.n_rf = int(n_rf)

        # Imported from Ring
        self.Particle = Ring.Particle
        self.n_turns = Ring.n_turns
        self.ring_circumference = Ring.ring_circumference
        self.section_length = Ring.ring_length[self.section_index]
        self.length_ratio = float(self.section_length/self.ring_circumference)
        self.t_rev = Ring.t_rev
        self.momentum = Ring.momentum[self.section_index]
        self.beta = Ring.beta[self.section_index]
        self.gamma = Ring.gamma[self.section_index]
        self.energy = Ring.energy[self.section_index]
        self.delta_E = Ring.delta_E[self.section_index]
        self.alpha_order = Ring.alpha_order
        self.charge = self.Particle.charge



        # The order alpha_order used here can be replaced by Ring.alpha_order
        # when the assembler can differentiate the cases 'simple' and 'exact'
        # for the drift
        alpha_order = 2
        for i in range(alpha_order+1):
            dummy = getattr(Ring, 'eta_' + str(i))
            setattr(self, "eta_%s" % i, dummy[self.section_index])
            dummy = getattr(Ring, 'alpha_' + str(i))
            setattr(self, "alpha_%s" % i, dummy[self.section_index])
        self.sign_eta_0 = np.sign(self.eta_0)

        # Reshape input rf programs
        # Reshape design harmonic
        self.harmonic = RFStationOptions.reshape_data(harmonic,
                                                      self.n_turns,
                                                      self.n_rf,
                                                      Ring.cycle_time,
                                                      Ring.RingOptions.t_start)
        self.harmonic = self.harmonic.astype(bm.precision.real_t, order='C', copy=False)

        # Reshape design voltage
        self.voltage = RFStationOptions.reshape_data(voltage,
                                                     self.n_turns,
                                                     self.n_rf,
                                                     Ring.cycle_time,
                                                     Ring.RingOptions.t_start)
        self.voltage = self.voltage.astype(bm.precision.real_t, order='C', copy=False)
        # Checking if the RFStation is empty
        if np.sum(self.voltage) == 0:
            self.empty = True
        else:
            self.empty = False

        # Reshape design phase
        self.phi_rf_d = RFStationOptions.reshape_data(phi_rf_d,
                                                      self.n_turns,
                                                      self.n_rf,
                                                      Ring.cycle_time,
                                                      Ring.RingOptions.t_start)

        # Calculating design rf angular frequency
        if omega_rf is None:
            self.omega_rf_d = 2.*np.pi*self.beta*c*self.harmonic / \
                (self.ring_circumference)
        else:
            self.omega_rf_d = RFStationOptions.reshape_data(
                omega_rf,
                self.n_turns,
                self.n_rf,
                Ring.cycle_time,
                Ring.RingOptions.t_start)
        self.omega_rf_d = self.omega_rf_d.astype(bm.precision.real_t, order='C', copy=False)

        # Reshape phase noise
        if phi_noise is not None:
            self.phi_noise = RFStationOptions.reshape_data(
                phi_noise,
                self.n_turns,
                self.n_rf,
                Ring.cycle_time,
                Ring.RingOptions.t_start)
            self.phi_noise = self.phi_noise.astype(bm.precision.real_t, order='C', copy=False)

            
        else:
            self.phi_noise = None
            
        if phi_modulation is not None:
            
            try:
                iter(phi_modulation)
            except TypeError:
                phi_modulation = [phi_modulation]
            
            dPhi = np.zeros([self.n_rf, self.n_turns+1], dtype=bm.precision.real_t)
            dOmega = np.zeros([self.n_rf, self.n_turns+1], dtype=bm.precision.real_t)
            for pMod in phi_modulation:
                system = np.where(self.harmonic[:,0] == pMod.harmonic)[0]
                if len(system) == 0:
                    raise ValueError("No matching harmonic in phi_modulation")
                elif len(system) > 1:
                    raise RuntimeError("""Phase modulation not yet 
                                       implemented with multiple systems 
                                       at the same harmonic.""")
                else:
                    system = system[0]
                    
                pMod.calc_modulation()
                pMod.calc_delta_omega((Ring.cycle_time, self.omega_rf_d[system]))
                dPhiInput, dOmegaInput =  pMod.extend_to_n_rf(self.harmonic[:,0])
                dPhi += RFStationOptions.reshape_data(dPhiInput,
                                                     self.n_turns,
                                                     self.n_rf,
                                                     Ring.cycle_time,
                                                     Ring.RingOptions.t_start)
                dOmega += RFStationOptions.reshape_data(dOmegaInput,
                                                       self.n_turns,
                                                       self.n_rf,
                                                       Ring.cycle_time,
                                                       Ring.RingOptions.t_start)
                
                
            
            self.phi_modulation = (dPhi, dOmega)
        else:
            
            self.phi_modulation = None


        # Copy of the desing rf programs in the one used for tracking
        # and that can be changed by feedbacks
        self.phi_rf = np.array(self.phi_rf_d).astype(bm.precision.real_t)
        self.dphi_rf = np.zeros(self.n_rf).astype(bm.precision.real_t)
        self.omega_rf = np.array(self.omega_rf_d).astype(bm.precision.real_t)
        self.t_rf = 2*np.pi / self.omega_rf

        # From helper functions
        if not self.empty:
            self.phi_s = calculate_phi_s(self, self.Particle)
            self.Q_s = calculate_Q_s(self, self.Particle)
            self.omega_s0 = self.Q_s*Ring.omega_rev


    def eta_tracking(self, beam, counter, dE):
        r"""Function to calculate the slippage factor as a function of the
        energy offset :math:`\Delta E` of the particle. The slippage factor
        of the :math:`i` th order is :math:`\eta(\delta) = \sum_{i}(\eta_i \,
        \delta^i) = \sum_{i} \left(\eta_i \, \left[ \frac{\Delta E}
        {\beta_s^2 E_s} \right]^i \right)`

        """

        if self.alpha_order == 0:
            return self.eta_0[counter]
        else:
            eta = 0
            delta = dE/(beam.beta**2 * beam.energy)
            for i in range(self.alpha_order+1):
                eta_i = getattr(self, 'eta_' + str(i))[counter]
                eta += eta_i * (delta**i)
            return eta

    def to_gpu(self, recursive=True):
        '''
        Transfer all necessary arrays to the GPU
        '''
        # Check if to_gpu has been invoked already
        if hasattr(self, '_device') and self._device == 'GPU':
            return
        assert bm.device == 'GPU'
        import cupy as cp
        if self.phi_modulation is not None:
            self.phi_modulation = (cp.array(self.phi_modulation[0]),
                                   cp.array(self.phi_modulation[1]))
        if self.phi_noise is not None:
            self.phi_noise = cp.array(self.phi_noise)

        self.voltage = cp.array(self.voltage)
        self.omega_rf = cp.array(self.omega_rf)
        self.phi_rf = cp.array(self.phi_rf)
        self.omega_rf_d = cp.array(self.omega_rf_d)
        self.harmonic = cp.array(self.harmonic)
        self.dphi_rf = cp.array(self.dphi_rf)
        self.t_rf = cp.array(self.t_rf)
        self.t_rev = cp.array(self.t_rev)

        # to make sure it will not be called again
        self._device = 'GPU'

    def to_cpu(self, recursive=True):
        '''
        Transfer all necessary arrays back to the CPU
        '''
        # Check if to_cpu has been invoked already
        if hasattr(self, '_device') and self._device == 'CPU':
            return
        assert bm.device == 'CPU'
        import cupy as cp
        if self.phi_modulation is not None:
            self.phi_modulation = (cp.asnumpy(self.phi_modulation[0]),
                                   cp.asnumpy(self.phi_modulation[1]))
        if self.phi_noise is not None:
            self.phi_noise = cp.asnumpy(self.phi_noise)

        self.voltage = cp.asnumpy(self.voltage)
        self.omega_rf = cp.asnumpy(self.omega_rf)
        self.phi_rf = cp.asnumpy(self.phi_rf)
        self.omega_rf_d = cp.asnumpy(self.omega_rf_d)
        self.harmonic = cp.asnumpy(self.harmonic)
        self.dphi_rf = cp.asnumpy(self.dphi_rf)
        self.t_rf = cp.asnumpy(self.t_rf)
        self.t_rev = cp.asnumpy(self.t_rev)
        # to make sure it will not be called again
        self._device = 'CPU'

def calculate_Q_s(RFStation, Particle=Proton()):
    r""" Function calculating the turn-by-turn synchrotron tune for
    single-harmonic RF, without intensity effects.

    Parameters
    ----------
    RFStation : class
        An RFStation type class.
    Particle : class
        A Particle type class; default is Proton().

    Returns
    -------
    float
        Synchrotron tune.

    """

    return np.sqrt(RFStation.harmonic[0]*np.abs(Particle.charge) *
                   RFStation.voltage[0] *
                   np.abs(RFStation.eta_0*np.cos(RFStation.phi_s)) /
                   (2*np.pi*RFStation.beta**2*RFStation.energy))


def calculate_phi_s(RFStation, Particle=Proton(),
                    accelerating_systems='as_single'):
    r"""Function calculating the turn-by-turn synchronous phase according to
    the parameters in the RFStation object. The phase is expressed in
    the lowest RF harmonic and with respect to the RF bucket (see the equations
    of motion defined for BLonD). The returned value is given in the range [0,
    2*Pi]. Below transition, the RF wave is shifted by Pi w.r.t. the time
    reference.

    The accelerating_systems option can be set to

    * 'as_single' (default): the synchronous phase is calculated analytically
      taking into account the phase program (RFStation.phi_offset).
    * 'all': the synchronous phase is calculated numerically by finding the
      minimum of the potential well; no intensity effects included. In case of
      several minima, the deepest is taken. **WARNING:** in case of RF
      harmonics with comparable voltages, this may lead to inconsistent
      values of phi_s.
    * 'first': not yet implemented. Its purpose should be to adjust the
      RFStation.phi_offset of the higher harmonics so that only the
      main harmonic is accelerating.

    Parameters
    ----------
    RFStation : class
        An RFStation type class.
    Particle : class
        A Particle type class; default is Proton().
    accelerating_systems : str
        Choice of accelerating systems; or options, see list above.

    Returns
    -------
    float
        Synchronous phase.

    """

    eta0 = RFStation.eta_0

    if accelerating_systems == 'as_single':

        denergy = np.append(RFStation.delta_E, RFStation.delta_E[-1])
        acceleration_ratio = denergy/(Particle.charge*RFStation.voltage[0, :])
        acceleration_test = np.where((acceleration_ratio > -1) *
                                     (acceleration_ratio < 1) is False)[0]

        # Validity check on acceleration_ratio
        if acceleration_test.size > 0:
            print("WARNING in calculate_phi_s(): acceleration is not " +
                  "possible (momentum increment is too big or voltage too " +
                  "low) at index " + str(acceleration_test))

        phi_s = np.arcsin(acceleration_ratio)

        # Identify where eta swaps sign
        eta0_middle_points = (eta0[1:] + eta0[:-1])/2
        eta0_middle_points = np.append(eta0_middle_points, eta0[-1])
        index = np.where(eta0_middle_points > 0)[0]
        index_below = np.where(eta0_middle_points < 0)[0]

        # Project phi_s in correct range
        phi_s[index] = (np.heaviside(np.sign(Particle.charge),0) * np.pi - phi_s[index]) % (2*np.pi)
        phi_s[index_below] = (np.heaviside(np.sign(Particle.charge),0) * np.pi + phi_s[index_below])\
            % (2*np.pi)
        # phi_s[index] = (np.pi - phi_s[index]) % (2*np.pi)
        # phi_s[index_below] = (np.pi + phi_s[index_below]) % (2*np.pi)

        return phi_s

    elif accelerating_systems == 'all':

        phi_s = np.zeros(len(RFStation.voltage[0, 1:]))

        for indexTurn in range(len(RFStation.delta_E)):

            totalRF = 0
            if np.sign(eta0[indexTurn]) > 0:
                phase_array = np.linspace(
                    -float(RFStation.phi_rf[0, indexTurn+1]),
                    -float(RFStation.phi_rf[0, indexTurn+1]) + 2*np.pi, 1000)
            else:
                phase_array = np.linspace(
                    -float(RFStation.phi_rf[0, indexTurn+1]) - np.pi,
                    -float(RFStation.phi_rf[0, indexTurn+1]) + np.pi, 1000)

            for indexRF in range(len(RFStation.voltage[:, indexTurn+1])):
                totalRF += RFStation.voltage[indexRF, indexTurn+1] * \
                    np.sin(RFStation.harmonic[indexRF, indexTurn+1] /
                           np.min(RFStation.harmonic[:, indexTurn+1]) *
                           phase_array +
                           RFStation.phi_rf[indexRF, indexTurn+1])

            potential_well = - cumtrapz(
                np.sign(eta0[indexTurn])*(totalRF -
                                          RFStation.delta_E[indexTurn] /
                                          abs(Particle.charge)),
                dx=phase_array[1]-phase_array[0], initial=0)

            phi_s[indexTurn] = np.mean(phase_array[
                potential_well == np.min(potential_well)])

        phi_s = np.insert(phi_s, 0, phi_s[0]) + RFStation.phi_rf[0, :]
        phi_s[eta0 < 0] += np.pi
        phi_s = phi_s % (2*np.pi)

        return phi_s

    elif accelerating_systems == 'first':

        print("WARNING in calculate_phi_s(): accelerating_systems 'first'" +
              " not yet implemented")
        pass
    else:
        raise RuntimeError("ERROR in calculate_phi_s(): unrecognised" +
                           " accelerating_systems option")

