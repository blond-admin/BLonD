# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering all general input parameters used for the simulation.**
    :Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
'''

from __future__ import division
from builtins import str, range, object
import numpy as np
import warnings
from scipy.constants import c
from input_parameters.ring_options import RampOptions


class Ring(object):
    r""" Class containing the general properties of the synchrotron that are
    independent of the RF system or the beam.

    The index :math:`n` denotes time steps, :math:`k` ring segments/sections
    and :math:`i` momentum compaction orders.

    Parameters
    ----------
    n_turns : int
        Number of turns :math:`n` [1] to be simulated
    ring_length : float (opt: float array [n_sections])
        Length [m] of the n_sections ring segments of the synchrotron.
        An RF station, a synchrotron radiation kick, and/or an impedance kick
        can be included at the end of each ring section.
    alpha_0 : float (opt: float array/matrix [n_sections, n_turns+1])
        Momentum compaction factor of zeroth order :math:`\alpha_{0,k,i}` [1];
        can be input as single float or as a program of (n_turns + 1) turns
        (should be of the same size as synchronous_data).
        In case of higher order momentum compaction, check the
        documentation for the inputs: alpha_order, alpha_1, alpha_2
    synchronous_data : float (opt: float array/matrix [n_sections, n_turns+1])
        Design synchronous particle momentum (default) [eV], kinetic or
        total energy [eV] or bending field [T] on the design orbit.
        Input for each RF section :math:`p_{s,k,n}`.
        Can be input as a single constant float, or as a
        program of (n_turns + 1) turns. In case of several sections without
        acceleration, input: [[momentum_section_1], [momentum_section_2],
        etc.]. In case of several sections with acceleration, input:
        [momentum_program_section_1, momentum_program_section_2, etc.]. Can
        be input also as a tuple of time and momentum, see also
        'cycle_time' and 'PreprocessRamp'
    Particle : class
        A Particle-based class defining the primary, synchronous particle (mass
        and charge) that is reference for the momentum/energy in the ring.
    synchronous_data_type : str
        Optional: Choice of 'synchronous_data' type; can be 'momentum'
        (default), 'total energy', 'kinetic energy' or 'bending field'
        (requires bending_radius to be defined)
    bending_radius : float
        Optional: Radius [m] of the bending magnets,
        required if 'bending field' is set for the synchronous_data_type
    n_sections : int
        Optional: number of ring sections/segments; default is 1
    alpha_order : int
        Optional : Number of momentum compaction orders. The input value should
        be from 0 to 2 to consider the momentum compaction from
        zeroth order (alpha_0) to second order (alpha_2); default is 0
    alpha_1 : float (opt: float array/matrix [n_sections, n_turns+1])
        Momentum compaction factor of first order :math:`\alpha_{1,k,i}` [1];
        can be input as single float or as a program of (n_turns + 1) turns
        (should be of the same size as synchronous_data and alpha_0).
    alpha_2 : float (opt: float array/matrix [n_sections, n_turns+1])
        Momentum compaction factor of second order :math:`\alpha_{2,k,i}` [1];
        can be input as single float or as a program of (n_turns + 1) turns
        (should be of the same size as synchronous_data and alpha_0).
    RampOptions : class
        Optional : A RampOptions-based class with default options to check the
        input and initialize the momentum program for the simulation.
        This object defines the interpolation scheme, plotting options, etc.
        The options for this object can be adjusted and passed to the Ring
        object.

    Attributes
    ----------
    ring_circumference : float
        Circumference of the synchrotron. Sum of ring segment lengths,
        :math:`C = \sum_k L_k` [m]
    ring_radius : float
        Radius of the synchrotron, :math:`R = C/(2 \pi)` [m]
    bending_radius : float
        Bending radius in dipole magnets, :math:`\rho` [m]
    alpha_order : int
        Number of orders of the momentum compaction factor
    eta_0 : float matrix [n_sections, n_turns+1]
        Zeroth order slippage factor :math:`\eta_{0,k,n} = \alpha_{0,k,n} -
        \frac{1}{\gamma_{s,k,n}^2}` [1]
    eta_1 : float matrix [n_sections, n_turns+1]
        First order slippage factor :math:`\eta_{1,k,n} =
        \frac{3\beta_{s,k,n}^2}{2\gamma_{s,k,n}^2} + \alpha_{1,k,n} -
        \alpha_{0,k,n}\eta_{0,k,n}` [1]
    eta_2 : float matrix [n_sections, n_turns+1]
        Second order slippage factor :math:`\eta_{2,k,n} =
        -\frac{\beta_{s,k,n}^2\left(5\beta_{s,k,n}^2-1\right)}
        {2\gamma_{s,k,n}^2} + \alpha_{2,k,n} - 2\alpha_{0,k,n}\alpha_{1,k,n}
        + \frac{\alpha_{1,k,n}}{\gamma_{s,k,n}^2} + \alpha_{0,k}^2\eta_{0,k,n}
        - \frac{3\beta_{s,k,n}^2\alpha_{0,k,n}}{2\gamma_{s,k,n}^2}` [1]
    momentum : float matrix [n_sections, n_turns+1]
        Synchronous relativistic momentum on the design orbit :math:`p_{s,k,n}`
    beta : float matrix [n_sections, n_turns+1]
        Synchronous relativistic beta program for each segment of the
        ring :math:`\beta_{s,k}^n = \frac{1}{\sqrt{1
        + \left(\frac{m}{p_{s,k,n}}\right)^2} }` [1]
    gamma : float matrix [n_sections, n_turns+1]
        Synchronous relativistic gamma program for each segment of the ring
        :math:`\gamma_{s,k,n} = \sqrt{ 1
        + \left(\frac{p_{s,k,n}}{m}\right)^2 }` [1]
    energy : float matrix [n_sections, n_turns+1]
        Synchronous total energy program for each segment of the ring
        :math:`E_{s,k,n} = \sqrt{ p_{s,k,n}^2 + m^2 }` [eV]
    kin_energy : float matrix [n_sections, n_turns+1]
        Synchronous kinetic energy program for each segment of the ring
        :math:`E_{s,kin} = \sqrt{ p_{s,k,n}^2 + m^2 } - m` [eV]
    delta_E : float matrix [n_sections, n_turns+1]
        Derivative of synchronous total energy w.r.t. time, for all sections,
        :math:`: \quad E_{s,k,n+1}- E_{s,k,n}` [eV]
    t_rev : float array [n_turns+1]
        Revolution period turn by turn.
        :math:`T_{0,n} = \frac{C}{\beta_{s,n} c}` [s]
    f_rev : float array [n_turns+1]
        Revolution frequency :math:`f_{0,n} = \frac{1}{T_{0,n}}` [Hz]
    omega_rev : float array [n_turns+1]
        Revolution angular frequency :math:`\omega_{0,n} = 2\pi f_{0,n}` [1/s]
    cycle_time : float array [n_turns+1]
        Cumulative cycle time, turn by turn, :math:`t_n = \sum_n T_{0,n}` [s].
        Possibility to extract cycle parameters at these moments using
        'parameters_at_time'.

    Examples
    --------
    >>> # To declare a single-section synchrotron at constant energy:
    >>> # Particle type Proton
    >>> from beam.beam import Proton
    >>> from input_parameters.ring import Ring
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha_0 = 3.21e-4
    >>> momentum = 450e9
    >>> ring = Ring(n_turns, C, alpha_0, momentum, Proton())
    >>>
    >>>
    >>> # To declare a two section synchrotron at constant energy and
    >>> # higher-order momentum compaction factors; particle Electron:
    >>> from beam.beam import Electron
    >>> from input_parameters.ring import Ring
    >>>
    >>> n_turns = 10
    >>> C = [13000, 13659]
    >>> alpha_0 = [[3.21e-4, 2.e-5, 5.e-7], [2.89e-4, 1.e-5, 5.e-7]]
    >>> alpha_1 = [[3.21e-4, 2.e-5, 5.e-7], [2.89e-4, 1.e-5, 5.e-7]]
    >>> alpha_2 = [[3.21e-4, 2.e-5, 5.e-7], [2.89e-4, 1.e-5, 5.e-7]]
    >>> momentum = 450e9
    >>> ring = Ring(n_turns, C, alpha_0, momentum, Electron())

    """

    def __init__(self, n_turns, ring_length, alpha_0, synchronous_data,
                 Particle, synchronous_data_type='momentum',
                 bending_radius=None, n_sections=1, alpha_order=0,
                 alpha_1=None, alpha_2=None, RampOptions=RampOptions()):

        # Conversion of initial inputs to expected types
        self.n_sections = int(n_sections)

        # Ring length and checks
        self.ring_length = np.array(ring_length, ndmin=1, dtype=float)
        self.ring_circumference = np.sum(self.ring_length)
        self.ring_radius = self.ring_circumference/(2*np.pi)

        if bending_radius is not None:
            self.bending_radius = float(bending_radius)
        else:
            self.bending_radius = bending_radius

        if self.n_sections != len(self.ring_length):
            raise RuntimeError("ERROR in Ring: Number of sections and ring " +
                               "length size do not match!")

        # Primary particle mass and charge used for energy calculations
        self.Particle = Particle

        # Initialization of the ramp
        # If tuple, separate time and synchronous data and check data
        if isinstance(synchronous_data, tuple):
            synchronous_data_time = np.array(synchronous_data[0], ndmin=2,
                                             dtype=float)
            synchronous_data = np.array(synchronous_data[1], ndmin=2,
                                        dtype=float)

            if synchronous_data_time.shape != synchronous_data.shape:
                raise RuntimeError("ERROR in Ring: synchronous data does " +
                                   "not match the time data")

        # If string, loads the input file
        elif isinstance(synchronous_data, str):
            pass

        # If array/list or float, compares with the input number of turns and
        # if synchronous_data is a single value converts it into a (n_turns+1)
        # array
        else:
            synchronous_data_time = None
            if n_turns:
                self.n_turns = int(n_turns)
            else:
                raise RuntimeError("ERROR in Ring: need to define n_turns " +
                                   "unless using the preprocess function!")

            synchronous_data = np.array(synchronous_data, ndmin=2,
                                        dtype=float)

            if synchronous_data.shape[1] == 1:
                synchronous_data = synchronous_data*np.ones(self.n_turns+1)

            elif synchronous_data.shape[1] != (self.n_turns+1):

                raise RuntimeError("ERROR in Ring: The momentum program " +
                                   "does not match the proper length " +
                                   "(n_turns+1)")

        # Check if the input data matches the number of sections
        if (synchronous_data.shape[0] != self.n_sections):

                raise RuntimeError("ERROR in Ring: Number of sections and " +
                                   "synchronous data data do not match!")

        # Convert synchronous data to momentum
        self.momentum = self.convert_data(
            synchronous_data,
            synchronous_data_type=synchronous_data_type)

        # If synchronous_data_time is defined, the RampOptions object
        # interpolates the momentum program for every machine turn
        if synchronous_data_time is not None:
            if synchronous_data.shape[0] > 1:
                raise RuntimeError("ERROR in Ring: preprocess works just " +
                                   "for single  section, to be extended.")
            self.cycle_time, self.momentum = RampOptions.preprocess(
                self.Particle.mass, self.ring_circumference,
                synchronous_data_time[0], self.momentum[0])

            self.n_turns = len(self.cycle_time)-1
#             self.cycle_time = np.array(self.cycle_time, ndmin=2,
#                                              dtype=float)
            self.momentum = np.array(self.momentum, ndmin=2,
                                     dtype=float)

        # Derived from momentum
        self.beta = np.sqrt(1/(1 + (self.Particle.mass/self.momentum)**2))
        self.gamma = np.sqrt(1 + (self.momentum/self.Particle.mass)**2)
        self.energy = np.sqrt(self.momentum**2 + self.Particle.mass**2)
        self.kin_energy = np.sqrt(self.momentum**2 + self.Particle.mass**2) - \
            self.Particle.mass
        self.delta_E = np.diff(self.energy, axis=1)
        self.t_rev = np.dot(self.ring_length, 1/(self.beta*c))
        if synchronous_data_time is not None and RampOptions.t_start != 0:
            pass
        else:
            self.cycle_time = np.cumsum(self.t_rev)  # Always starts with zero
        self.f_rev = 1/self.t_rev
        self.omega_rev = 2*np.pi*self.f_rev

        # Momentum compaction, checks, and derived slippage factors
        self.alpha_0 = np.array(alpha_0, ndmin=1, dtype=float)
        self.alpha_1 = np.array(alpha_1, ndmin=1, dtype=float)
        self.alpha_2 = np.array(alpha_2, ndmin=1, dtype=float)
        self.alpha_order = int(alpha_order)

        if self.alpha_order > 2:
            warnings.filterwarnings("once")
            warnings.warn("WARNING in Ring: Momentum compaction factor is " +
                          "implemented only up to 2nd order. Higher orders " +
                          "are ignored.")
            self.alpha_order = 2

        if self.n_sections != self.alpha_0.shape[0]:
            raise RuntimeError("ERROR in Ring: Number of sections and size " +
                               "of momentum compaction do not match!")

        # Slippage factor derived from alpha, beta, gamma
        self.eta_generation()

    def eta_generation(self):
        """ Function to generate the slippage factors (zeroth, first, and
        second orders, see [1]_) from the momentum compaction and the
        relativistic beta and gamma program through the cycle.

        References
        ----------
        .. [1] "Accelerator Physics," S. Y. Lee, World Scientific,
                Third Edition, 2012.
        """

        for i in range(self.alpha_order+1):
            getattr(self, '_eta' + str(i))()

        # Fill unused eta arrays with zeros
        for i in range(self.alpha_order+1, 3):
            setattr(self, "eta_%s" % i, np.zeros([self.n_sections,
                                                  self.n_turns+1]))

    def _eta0(self):
        """ Function to calculate the zeroth order slippage factor eta_0 """

        self.eta_0 = np.empty([self.n_sections, self.n_turns+1])
        for i in range(0, self.n_sections):
            self.eta_0[i] = self.alpha_0[i] - self.gamma[i]**(-2.)

    def _eta1(self):
        """ Function to calculate the first order slippage factor eta_1 """

        self.eta_1 = np.empty([self.n_sections, self.n_turns+1])
        for i in range(0, self.n_sections):
            self.eta_1[i] = 3*self.beta[i]**2/(2*self.gamma[i]**2) + \
                self.alpha_1[i] - self.alpha_0[i]*self.eta_0[i]

    def _eta2(self):
        """ Function to calculate the second order slippage factor eta_2 """

        self.eta_2 = np.empty([self.n_sections, self.n_turns+1])
        for i in range(0, self.n_sections):
            self.eta_2[i] = - self.beta[i]**2*(5*self.beta[i]**2 - 1) / \
                (2*self.gamma[i]**2) + self.alpha_2[i] - 2*self.alpha_0[i] *\
                self.alpha_1[i] + self.alpha_1[i] / self.gamma[i]**2 + \
                self.alpha_0[i]**2*self.eta_0[i] - 3*self.beta[i]**2 * \
                self.alpha_0[i]/(2*self.gamma[i]**2)

    def convert_data(self, synchronous_data, synchronous_data_type='momentum'):

        if synchronous_data_type == 'momentum':
            momentum = synchronous_data
        elif synchronous_data_type == 'total energy':
            momentum = np.sqrt(synchronous_data**2 - self.Particle.mass**2)
        elif synchronous_data_type == 'kinetic energy':
            momentum = np.sqrt((synchronous_data+self.Particle.mass)**2 -
                               self.Particle.mass**2)
        elif synchronous_data_type == 'bending field':
            if self.bending_radius is None:
                raise RuntimeError("ERROR in Ring: bending_radius is not " +
                                   "defined and is required to compute " +
                                   "momentum")
            momentum = synchronous_data*self.bending_radius*c
        else:
            raise RuntimeError("ERROR in Ring: Synchronous data" +
                               " type not recognized!")

        return momentum

    def parameters_at_time(self, cycle_moments):
        """ Function to return various cycle parameters at a specific moment in
        time. The cycle time is defined to start at zero in turn zero.

        Parameters
        ----------
        cycle_moments : float array
            Moments of time at which cycle parameters are to be calculated [s].

        Attributes
        ----------
        parameters : dictionary
            Contains 'momentum', 'beta', 'gamma', 'energy', 'kin_energy',
            'f_rev', 't_rev'. 'omega_rev', 'eta_0', and 'delta_E' interpolated
            to the moments contained in the 'cycle_moments' array

        """

        parameters = {}
        parameters['momentum'] = np.interp(cycle_moments, self.cycle_time,
                                           self.momentum[0])
        parameters['beta'] = np.interp(cycle_moments, self.cycle_time,
                                       self.beta[0])
        parameters['gamma'] = np.interp(cycle_moments, self.cycle_time,
                                        self.gamma[0])
        parameters['energy'] = np.interp(cycle_moments, self.cycle_time,
                                         self.energy[0])
        parameters['kin_energy'] = np.interp(cycle_moments, self.cycle_time,
                                             self.kin_energy[0])
        parameters['f_rev'] = np.interp(cycle_moments, self.cycle_time,
                                        self.f_rev)
        parameters['t_rev'] = np.interp(cycle_moments, self.cycle_time,
                                        self.t_rev)
        parameters['omega_rev'] = np.interp(cycle_moments, self.cycle_time,
                                            self.omega_rev)
        parameters['eta_0'] = np.interp(cycle_moments, self.cycle_time,
                                        self.eta_0[0])
        parameters['delta_E'] = np.interp(cycle_moments,
                                          self.cycle_time[1:],
                                          np.diff(self.energy[0]))

        return parameters
