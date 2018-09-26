# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module containing all the elements to track the RF frequency, voltage, phase,
and the beam coordinates in phase space.**

:Authors:  **Helga Timko**, **Alexandre Lasheen**, **Danilo Quartullo**
"""

from __future__ import division
from builtins import range, object
import numpy as np
from scipy.integrate import cumtrapz
import ctypes
import logging

# from ..setup_cpp import libblond
from .. import libblond

from ..utils import bmath as bm


class FullRingAndRF(object):
    """
    *Definition of the full ring and RF parameters in order to be able to have
    a full turn information (used in the hamiltonian for example).*
    """

    def __init__(self, RingAndRFSection_list):

        #: *List of the total RingAndRFSection objects*
        self.RingAndRFSection_list = RingAndRFSection_list

        #: *Total potential well in [V]*
        self.potential_well = 0

        #: *Total potential well theta coordinates in [rad]*
        self.potential_well_coordinates = 0

        #: *Ring circumference in [m]*
        self.ring_circumference = 0
        for RingAndRFSectionElement in self.RingAndRFSection_list:
            self.ring_circumference += RingAndRFSectionElement.section_length

        #: *Ring radius in [m]*
        self.ring_radius = self.ring_circumference / (2*np.pi)

    def potential_well_generation(self, turn=0, n_points=1e5,
                                  main_harmonic_option='lowest_freq',
                                  dt_margin_percent=0., time_array=None):
        """Method to generate the potential well out of the RF systems. The 
        assumption made is that all the RF voltages are averaged over one turn.
        The potential well is then approximated over one turn, which is not the
        exact potential. This approximation should be fine enough to generate a
        bunch (the mismatch should be small and damped fast enough). The 
        default main harmonic is defined to be the lowest one in frequency. The
        user can change this option if it is not the case for his simulations
        (other options are: 'highest_voltage', or inputing directly the value
        of the desired main harmonic). A margin on the time array can be 
        applied in order to be able to see the min/max that might be exactly on
        the edges of the frame (by adding a % to the length of the frame, this
        is set to 0 by default. It assumes also that the slippage factor is the
        same in the whole ring.
        """

        voltages = np.array([])
        omega_rf = np.array([])
        phi_offsets = np.array([])

        for RingAndRFSectionElement in self.RingAndRFSection_list:
            charge = RingAndRFSectionElement.charge
            for rf_system in range(RingAndRFSectionElement.n_rf):
                voltages = np.append(voltages,
                                     RingAndRFSectionElement.voltage[rf_system, turn])
                omega_rf = np.append(omega_rf,
                                     RingAndRFSectionElement.omega_rf[rf_system, turn])
                phi_offsets = np.append(phi_offsets,
                                        RingAndRFSectionElement.phi_rf[rf_system, turn])

        voltages = np.array(voltages, ndmin=2)
        omega_rf = np.array(omega_rf, ndmin=2)
        phi_offsets = np.array(phi_offsets, ndmin=2)

        if main_harmonic_option is 'lowest_freq':
            main_omega_rf = np.min(omega_rf)
        elif main_harmonic_option is 'highest_voltage':
            main_omega_rf = np.min(omega_rf[voltages == np.max(voltages)])
        elif isinstance(main_harmonic_option, int) or \
                isinstance(main_harmonic_option, float):
            if omega_rf[omega_rf == main_harmonic_option].size == 0:
                raise RuntimeError("ERROR in FullRingAndRF: The desired" +
                                   " harmonic to compute the potential well does not match" +
                                   " the RF parameters...")
            main_omega_rf = np.min(omega_rf[omega_rf == main_harmonic_option])

        slippage_factor = self.RingAndRFSection_list[0].eta_0[turn]

        if time_array is None:
            time_array_margin = dt_margin_percent*2*np.pi/main_omega_rf

            first_dt = - time_array_margin/2
            last_dt = 2*np.pi/main_omega_rf + time_array_margin/2

            time_array = np.linspace(first_dt, last_dt, n_points)

        self.total_voltage = np.sum(voltages.T *
                                    np.sin(omega_rf.T*time_array + phi_offsets.T), axis=0)

        eom_factor_potential = np.sign(slippage_factor)*charge / \
            (RingAndRFSectionElement.t_rev[turn])

        potential_well = - cumtrapz(eom_factor_potential*(self.total_voltage -
                                                          (- RingAndRFSectionElement.acceleration_kick[turn])/abs(charge)),
                                    dx=time_array[1] - time_array[0], initial=0)
        potential_well = potential_well - np.min(potential_well)

        self.potential_well_coordinates = time_array
        self.potential_well = potential_well

    def track(self):
        """Function to loop over all the RingAndRFSection.track methods
        """

        for RingAndRFSectionElement in self.RingAndRFSection_list:
            RingAndRFSectionElement.track()


class RingAndRFTracker(object):
    r""" Class taking care of basic particle coordinate tracking for a given
    RF station and the part of the ring until the next station, see figure.

    .. image:: ring_and_RFstation.png
        :align: center
        :width: 600
        :height: 600

    The time step is fixed to be one turn, but the tracking can consist of 
    multiple RingAndRFTracker objects. In this case, the user should make sure 
    that the lengths of the stations sum up exactly to the circumference or use
    the FullRingAndRF object in order to let the code pre-process the 
    parameters. Each RF station may contain several RF harmonic systems which 
    are considered to be in the same location. First, the energy kick of the RF
    station is applied, and then the particle arrival time to the next station
    is updated. The change in RF phase, voltage, and frequency due to control 
    loops is tracked as well.

    Parameters
    ----------
    RFStation : class
        A RFStation type class
    counter : [int] 
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.counter`
    length_ratio : float 
        Inherited from
        :py:attr:`input_parameters.ring.Ring.length_ratio`
    section_length : float 
        Inherited from
        :py:attr:`input_parameters.ring.Ring.section_length`
    t_rev : float 
        Inherited from
        :py:attr:`input_parameters.ring.Ring.t_rev`
    n_rf : float 
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.n_rf`
    beta : float 
        Inherited from
        :py:attr:`input_parameters.ring.Ring.beta`
    charge : float 
        Inherited from
        :py:attr:`input_parameters.ring.Ring.Particle.charge`
    harmonic : float array
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.harmonic`
    voltage : float array
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.voltage`
    phi_noise : float array
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.phi_noise`
    phi_rf : float array
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.phi_rf`
    phi_s : float array
        Inherited from
        :py:attr:`input_parameters.rf_parameters.RFStation.phi_s`
    eta_0 : float array
        Inherited from
        :py:attr:`input_parameters.ring.Ring.eta_0`
    eta_1 : float array
        Inherited from
        :py:attr:`input_parameters.ring.Ring.eta_1`
    eta_2 : float array
        Inherited from
        :py:attr:`input_parameters.ring.Ring.eta_2`
    alpha_order : float array
        Inherited from
        :py:attr:`input_parameters.ring.Ring.alpha_order`
    acceleration_kick : float array
        Inherited from
        :py:attr:`input_parameters.ring.Ring.delta_E`
        and multiplied by -1
    Beam : class
        A Beam type class
    solver : str
        Type of solver used for the drift equation; use 'simple' for 1st order
        approximation and 'exact' for exact solver
    BeamFeedback : class (optional)
        A BeamFeedback type class, beam-based feedback on RF frequency;
        default is None
    NoiseFeedback : class (optional)
        A NoiseFeedback type class, bunch-length feedback on RF noise;
        default is None
    periodicity : bool (optional)
        Option to switch periodic solver on/off; default is False (off)
    interpolation : bool (optional)
        Option to use sliced and interpolated voltage for the kicker; default 
        is False

    """

    def __init__(self, RFStation, Beam, solver='simple', BeamFeedback=None,
                 NoiseFeedback=None, CavityFeedback=None, periodicity=False,
                 interpolation=False, Profile=None, TotalInducedVoltage=None):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Class initialized")

        # Imports from RF parameters
        self.rf_params = RFStation
        self.counter = RFStation.counter
        self.length_ratio = RFStation.length_ratio
        self.section_length = RFStation.section_length
        self.t_rev = RFStation.t_rev
        self.n_rf = RFStation.n_rf
        self.beta = RFStation.beta
        self.charge = RFStation.Particle.charge
        self.harmonic = RFStation.harmonic
        self.voltage = RFStation.voltage
        self.phi_noise = RFStation.phi_noise
        self.phi_rf = RFStation.phi_rf
        self.phi_s = RFStation.phi_s
        self.omega_rf = RFStation.omega_rf
        self.eta_0 = RFStation.eta_0
        self.eta_1 = RFStation.eta_1
        self.eta_2 = RFStation.eta_2
        self.alpha_order = RFStation.alpha_order
        self.acceleration_kick = - RFStation.delta_E

        # Other imports
        self.beam = Beam
        self.solver = str(solver)
        if self.solver not in ['simple', 'exact']:
            raise RuntimeError("ERROR in RingAndRFTracker: Choice of" +
                               " longitudinal solver not recognised!")
        if self.alpha_order > 1:  # Set exact solver for higher orders of eta
            self.solver = 'exact'
        self.solver = self.solver.encode(encoding='utf_8')

        # Options
        self.beamFB = BeamFeedback
        self.noiseFB = NoiseFeedback
        self.cavityFB = CavityFeedback
        try:
            self.periodicity = bool(periodicity)
        except:
            raise RuntimeError("ERROR in RingAndRFTracker: Choice of" +
                               " periodicity not recognised!")
        try:
            self.interpolation = bool(interpolation)
        except:
            raise RuntimeError("ERROR in RingAndRFTracker: Choice of" +
                               " interpolation not recognised!")
        self.profile = Profile
        self.totalInducedVoltage = TotalInducedVoltage
        if (self.interpolation is True) and (self.profile is None):
            raise RuntimeError("ERROR in RingAndRFTracker: Please specify a" +
                               " Profile object to use the interpolation option")
        if (self.cavityFB is not None) and (self.profile is None):
            raise RuntimeError("ERROR in RingAndRFTracker: Please specify a" +
                               " Profile object to use the CavityFeedback class")
        if (self.rf_params.empty is True) and (self.periodicity is True):
            raise RuntimeError("ERROR in RingAndRFTracker: Empty RFStation" +
                               " with periodicity not yet implemented!")
        if (self.cavityFB is not None) and (self.interpolation is False):
            self.interpolation = True
            self.logger.warning("Setting interpolation to TRUE")

    def kick(self, beam_dt, beam_dE, index):
        """Function updating the particle energy due to the RF kick in a given
        RF station. The kicks are summed over the different harmonic RF systems
        in the station. The cavity phase can be shifted by the user via 
        phi_offset. The main RF (harmonic[0]) has by definition phase = 0 at 
        time = 0 below transition. The phases of all other RF systems are 
        defined w.r.t.\ to the main RF. The increment in energy is given by the
        discrete equation of motion:

        .. math::
            \Delta E^{n+1} = \Delta E^n + \sum_{k=0}^{n_{\mathsf{rf}}-1}{e V_k^n \\sin{\\left(\omega_{\mathsf{rf,k}}^n \\Delta t^n + \phi_{\mathsf{rf,k}}^n \\right)}} - (E_s^{n+1} - E_s^n) 

        """

        voltage_kick = np.ascontiguousarray(self.charge*self.voltage[:, index])
        omegarf_kick = np.ascontiguousarray(self.omega_rf[:, index])
        phirf_kick = np.ascontiguousarray(self.phi_rf[:, index])

        libblond.kick(beam_dt.ctypes.data_as(ctypes.c_void_p),
                      beam_dE.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(self.n_rf),
                      voltage_kick.ctypes.data_as(ctypes.c_void_p),
                      omegarf_kick.ctypes.data_as(ctypes.c_void_p),
                      phirf_kick.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(len(beam_dt)),
                      ctypes.c_double(self.acceleration_kick[index]))

    def drift(self, beam_dt, beam_dE, index):
        """Function updating the particle arrival time to the RF station 
        (drift). If only the zeroth order slippage factor is given, 'simple' 
        and 'exact' solvers are available. The 'simple' solver is somewhat 
        faster. Otherwise, the solver is automatically 'exact' and calculates 
        the frequency slippage up to second order. The corresponding equations
        are:

        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1} \\left(\\frac{1}{1 - \\eta(\\delta^{n+1})\\delta^{n+1}} - 1\\right) \quad \\text{(exact)}

        .. math::
            \\Delta t^{n+1} = \\Delta t^{n} + \\frac{L}{C} T_0^{n+1}\\eta_0\\delta^{n+1} \quad \\text{(simple)}

        """

        libblond.drift(beam_dt.ctypes.data_as(ctypes.c_void_p),
                       beam_dE.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_char_p(self.solver),
                       ctypes.c_double(self.t_rev[index]),
                       ctypes.c_double(self.length_ratio),
                       ctypes.c_double(self.alpha_order),
                       ctypes.c_double(self.eta_0[index]),
                       ctypes.c_double(self.eta_1[index]),
                       ctypes.c_double(self.eta_2[index]),
                       ctypes.c_double(self.rf_params.beta[index]),
                       ctypes.c_double(self.rf_params.energy[index]),
                       ctypes.c_int(len(beam_dt)))

    def rf_voltage_calculation(self):
        """Function calculating the total, discretised RF voltage seen by the
        beam at a given turn. Requires a Profile object.

        """
        voltages = np.ascontiguousarray(self.voltage[:, self.counter[0]])
        omega_rf = np.ascontiguousarray(self.omega_rf[:, self.counter[0]])
        phi_rf = np.ascontiguousarray(self.phi_rf[:, self.counter[0]])
        # TODO: test with multiple harmonics, think about 800 MHz OTFB
        if self.cavityFB:
            self.rf_voltage = voltages[0] * self.cavityFB.V_corr * \
                bm.sin(omega_rf[0]*self.profile.bin_centers +
                       phi_rf[0] + self.cavityFB.phi_corr) + \
                bm.rf_volt_comp(voltages[1:], omega_rf[1:], phi_rf[1:], self)
        else:
            self.rf_voltage = bm.rf_volt_comp(voltages, omega_rf, phi_rf, self)


    def track(self):
        """Tracking method for the section. Applies first the kick, then the 
        drift. Calls also RF/beam feedbacks if applicable. Updates the counter
        of the corresponding RFStation class and the energy-related variables
        of the Beam class.

        """

        # Add phase noise directly to the cavity RF phase
        if self.phi_noise is not None:
            if self.noiseFB is not None:
                self.phi_rf[:, self.counter[0]] += \
                    self.noiseFB.x * self.phi_noise[:, self.counter[0]]
            else:
                self.phi_rf[:, self.counter[0]] += \
                    self.phi_noise[:, self.counter[0]]

        # Determine phase loop correction on RF phase and frequency
        if self.beamFB is not None and self.counter[0] >= self.beamFB.delay:
            self.beamFB.track()

        if self.periodicity:

            # Distinguish the particles inside the frame from the particles on
            # the right-hand side of the frame.
            self.indices_right_outside = \
                np.where(self.beam.dt > self.t_rev[self.counter[0] + 1])[0]
            self.indices_inside_frame = \
                np.where(self.beam.dt < self.t_rev[self.counter[0] + 1])[0]

            if len(self.indices_right_outside) > 0:
                # Change reference of all the particles on the right of the
                # current frame; these particles skip one kick and drift
                self.beam.dt[self.indices_right_outside] -= \
                    self.t_rev[self.counter[0] + 1]
                # Synchronize the bunch with the particles that are on the
                # RHS of the current frame applying kick and drift to the
                # bunch
                # After that all the particles are in the new updated frame
                self.insiders_dt = np.ascontiguousarray(
                    self.beam.dt[self.indices_inside_frame])
                self.insiders_dE = np.ascontiguousarray(
                    self.beam.dE[self.indices_inside_frame])
                self.kick(self.insiders_dt, self.insiders_dE, self.counter[0])
                self.drift(self.insiders_dt, self.insiders_dE,
                           self.counter[0]+1)
                self.beam.dt[self.indices_inside_frame] = self.insiders_dt
                self.beam.dE[self.indices_inside_frame] = self.insiders_dE
                # Check all the particles on the left of the just updated
                # frame and apply a second kick and drift to them with the
                # previous wave after having changed reference.
                self.indices_left_outside = np.where(self.beam.dt < 0)[0]

            else:
                self.kick(self.beam.dt, self.beam.dE, self.counter[0])
                self.drift(self.beam.dt, self.beam.dE, self.counter[0] + 1)
                # Check all the particles on the left of the just updated
                # frame and apply a second kick and drift to them with the
                # previous wave after having changed reference.
                self.indices_left_outside = np.where(self.beam.dt < 0)[0]

            if len(self.indices_left_outside) > 0:
                left_outsiders_dt = np.ascontiguousarray(
                    self.beam.dt[self.indices_left_outside])
                left_outsiders_dE = np.ascontiguousarray(
                    self.beam.dE[self.indices_left_outside])
                left_outsiders_dt += self.t_rev[self.counter[0]+1]
                self.kick(left_outsiders_dt, left_outsiders_dE,
                          self.counter[0])
                self.drift(left_outsiders_dt, left_outsiders_dE,
                           self.counter[0]+1)
                self.beam.dt[self.indices_left_outside] = left_outsiders_dt
                self.beam.dE[self.indices_left_outside] = left_outsiders_dE

        else:

            if self.rf_params.empty is False:
                if self.interpolation:
                    self.rf_voltage_calculation()
                    if self.totalInducedVoltage is not None:
                        self.total_voltage = self.rf_voltage \
                            + self.totalInducedVoltage.induced_voltage
                    else:
                        self.total_voltage = self.rf_voltage

                    libblond.linear_interp_kick(
                        self.beam.dt.ctypes.data_as(ctypes.c_void_p),
                        self.beam.dE.ctypes.data_as(ctypes.c_void_p),
                        self.total_voltage.ctypes.data_as(ctypes.c_void_p),
                        self.profile.bin_centers.ctypes.data_as(
                            ctypes.c_void_p),
                        ctypes.c_double(self.beam.Particle.charge),
                        ctypes.c_int(self.profile.n_slices),
                        ctypes.c_int(self.beam.n_macroparticles),
                        ctypes.c_double(
                            self.acceleration_kick[self.counter[0]]))

                else:
                    self.kick(self.beam.dt, self.beam.dE, self.counter[0])

            self.drift(self.beam.dt, self.beam.dE, self.counter[0] + 1)

        # Increment by one the turn counter
        self.counter[0] += 1

        # Updating the beam synchronous momentum etc.
        self.beam.beta = self.rf_params.beta[self.counter[0]]
        self.beam.gamma = self.rf_params.gamma[self.counter[0]]
        self.beam.energy = self.rf_params.energy[self.counter[0]]
        self.beam.momentum = self.rf_params.momentum[self.counter[0]]
