# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module gathering all general input parameters used for the simulation.**
    :Authors: **Alexandre Lasheen**, **Danilo Quartullo**, **Helga Timko**
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c

from .ring_options import RingOptions
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:  # only for Python type hints
    from typing import Literal, Any, Optional, Iterable

    from numpy.typing import NDArray as NumpyArray

    from ..beam.beam import Particle

    SynchronousDataTypes = Literal[
        "momentum", "total energy", "kinetic energy", "bending field"
    ]


class Ring:
    r"""Class containing the general properties of the synchrotron that are
    independent of the RF system or the beam.

    The index :math:`n` denotes time steps, :math:`k` ring segments/sections
    and :math:`i` momentum compaction orders.

    Parameters
    ----------
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
    particle : class
        A Particle-based class defining the primary, synchronous particle (mass
        and charge) that is reference for the momentum/energy in the ring.
    n_turns : int
        Optional: Number of turns :math:`n` [1] to be simulated.
        If a synchronous_data program is passed as a tuple (see below),
        the number of turns will be overwritten depending on the length in time
        of the program
    synchronous_data_type : str
        Optional: Choice of 'synchronous_data' type; can be 'momentum'
        (default), 'total energy', 'kinetic energy' or 'bending field'
        (requires bending_radius to be defined)
    bending_radius : float
        Optional: Radius [m] of the bending magnets,
        required if 'bending field' is set for the synchronous_data_type
    n_sections : int
        Optional: number of ring sections/segments; default is 1
    alpha_1 : float (opt: float array/matrix [n_sections, n_turns+1])
        Momentum compaction factor of first order
        :math:`\alpha_{1,k,i}` [1]; can be input as single float or as a
        program of (n_turns + 1) turns (should be of the same size as
        synchronous_data and alpha_0).
    alpha_2 : float (opt: float array/matrix [n_sections, n_turns+1])
        Optional : Momentum compaction factor of second order
        :math:`\alpha_{2,k,i}` [1]; can be input as single float or as a
        program of (n_turns + 1) turns (should be of the same size as
        synchronous_data and alpha_0).
    ring_options : class
        Optional : A RingOptions-based class with default options to check the
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
        Highest order of momentum compaction (as defined by the input). Can
        be 0,1,2.
    alpha_0 : float matrix [n_sections, n_turns+1]
        Zeroth order momentum compaction factor
        :math:`\alpha_{0,k,n}`
    alpha_1 : float matrix [n_sections, n_turns+1]
        First order momentum compaction factor
        :math:`\alpha_{1,k,n}`
    alpha_2 : float matrix [n_sections, n_turns+1]
        Second order momentum compaction factor
        :math:`\alpha_{2,k,n}`
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
        With more than one section, it is expected that column 0 holds the
        initial momentum for all rows.
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
    ring_options : RingOptions()
        The RingOptions is kept as an attribute of the Ring object for further
        usage.

    Examples
    --------
    >>> # To declare a single-section synchrotron at constant energy:
    >>> # Particle type Proton
    >>> from blond.beam.beam import Proton
    >>> from blond.input_parameters.ring import Ring
    >>>
    >>> n_turns = 10
    >>> C = 26659
    >>> alpha_0 = 3.21e-4
    >>> momentum = 450e9
    >>> ring = Ring(C, alpha_0, momentum, Proton(), n_turns)
    >>>
    >>>
    >>> # To declare a single-section synchrotron with a momentum program in time:
    >>> # Particle type Proton
    >>> from blond.beam.beam import Proton
    >>> from blond.input_parameters.ring import Ring
    >>>
    >>> C = 26659
    >>> alpha_0 = 3.21e-4
    >>> momentum_time = (0, 500e-3)
    >>> momentum = (450e9, 451e9)
    >>> momentum_program = (momentum_time, momentum)
    >>> ring = Ring(C, alpha_0, momentum_program, Proton())
    >>> # NB: n_turns is overwritten with the actual turns for the defined program
    >>>
    >>>
    >>> # To declare a two section synchrotron at constant energy and
    >>> # higher-order momentum compaction factors; particle Electron:
    >>> from blond.beam.beam import Electron
    >>> from blond.input_parameters.ring import Ring
    >>>
    >>> n_turns = 10
    >>> C = [13000, 13659]
    >>> alpha_0 = [[3.21e-4], [2.89e-4]]  # or [3.21e-4, 2.89e-4]
    >>> alpha_1 = [[2.e-5], [1.e-5]]  # or [2.e-5, 1.e-5]
    >>> alpha_2 = [[5.e-7], [5.e-7]]  # or [5.e-7, 5.e-7]
    >>> momentum = 450e9
    >>> ring = Ring(C, alpha_0, momentum, Electron(), n_turns,
    >>>             alpha_1=alpha_1, alpha_2=alpha_2)

    """

    @handle_legacy_kwargs
    def __init__(
        self,
        ring_length: float | list | tuple | NumpyArray,
        alpha_0: float | list | tuple | NumpyArray,
        synchronous_data: Any,  # todo type hint
        particle: Particle,
        n_turns: int = 1,
        synchronous_data_type: SynchronousDataTypes = "momentum",
        bending_radius: Optional[float] = None,
        n_sections: int = 1,
        alpha_1: None | float | list | tuple | NumpyArray = None,
        alpha_2: None | float | list | tuple | NumpyArray = None,
        ring_options: Optional[RingOptions] = None,
    ):
        if ring_options is None:
            ring_options = RingOptions()
        # Conversion of initial inputs to expected types
        self.n_turns = int(n_turns)
        self.n_sections = int(n_sections)

        # Ring length and checks
        self.ring_length: NumpyArray = np.array(
            ring_length, ndmin=1, dtype=float
        )
        self.ring_circumference: float = np.sum(self.ring_length)
        self.ring_radius: float = self.ring_circumference / (2 * np.pi)

        self.bending_radius: Optional[float] = (
            float(bending_radius) if bending_radius is not None else None
        )

        if self.n_sections != len(self.ring_length):
            # InputDataError
            raise RuntimeError(
                "ERROR in Ring: Number of sections and ring "
                + "length size do not match!"
            )

        # Primary particle mass and charge used for energy calculations
        self.particle: Particle = particle

        # Keeps RingOptions as an attribute
        self.ring_options: RingOptions = ring_options

        # Reshaping the input synchronous data to the adequate format and
        # get back the momentum program from RingOptions
        self.momentum: NumpyArray = ring_options.reshape_data(
            synchronous_data,
            self.n_turns,
            self.n_sections,
            input_to_momentum=True,
            synchronous_data_type=synchronous_data_type,
            mass=self.particle.mass,
            charge=self.particle.charge,
            circumference=self.ring_circumference,
            bending_radius=self.bending_radius,
        )

        # Updating the number of turns in case it was changed after ramp
        # interpolation
        if self.momentum.shape[1] != (self.n_turns + 1):
            self.n_turns = self.momentum.shape[1] - 1
            warnings.warn(
                "WARNING in Ring: The number of turns for the "
                "simulation was changed by passing a momentum "
                "program.",
                stacklevel=2,
            )

        # Derived from momentum
        # todo this should be attributes?
        self.beta: NumpyArray = np.sqrt(
            1 / (1 + (self.particle.mass / self.momentum) ** 2)
        )
        self.gamma: NumpyArray = np.sqrt(
            1 + (self.momentum / self.particle.mass) ** 2
        )
        self.energy: NumpyArray = np.sqrt(
            self.momentum**2 + self.particle.mass**2
        )
        self.kin_energy: NumpyArray = (
            np.sqrt(self.momentum**2 + self.particle.mass**2)
            - self.particle.mass
        )
        self.t_rev: NumpyArray = np.dot(self.ring_length, 1 / (self.beta * c))
        self.cycle_time: NumpyArray = np.cumsum(
            self.t_rev
        )  # Always starts with zero
        self.f_rev: NumpyArray = 1 / self.t_rev
        self.omega_rev: NumpyArray = 2 * np.pi * self.f_rev

        # TODO:  Revisit and improve multi-section interpolation
        if self.n_sections == 1:
            self.delta_E: NumpyArray = np.diff(self.energy, axis=1)
        else:
            # when there is more than 1 RF station, self.energy has shape (n_sections, n_turns+1)
            # where all turns have the same initial energy, the injection energy in column 0
            # Order="F" for column flattening
            self.delta_E = np.diff(self.energy.flatten(order="F"))[
                n_sections - 1 :
            ].reshape((n_sections, n_turns))
            # skipping of first n_section elements due to the same initial energy in the arrays,
            # one less is required due to the length reduction of diff

        # Momentum compaction, checks, and derived slippage factors
        if ring_options.t_start is None:
            interp_time = self.cycle_time
        else:
            interp_time = self.cycle_time + ring_options.t_start

        self.alpha_0 = ring_options.reshape_data(
            alpha_0, self.n_turns, self.n_sections, interp_time=interp_time
        )
        self.alpha_order = 0

        if alpha_1 is not None:
            self.alpha_1 = ring_options.reshape_data(
                alpha_1, self.n_turns, self.n_sections, interp_time=interp_time
            )
            self.alpha_order = 1
        else:
            # Filling alpha_1 with zeros
            # This can be removed when the BLonD assembler is in place
            # to avoid high order momentum compaction programs filled
            # with zeros (should be propagated in RFStation.__init__())
            self.alpha_1 = np.zeros(self.alpha_0.shape)

        if alpha_2 is not None:
            self.alpha_2 = ring_options.reshape_data(
                alpha_2, self.n_turns, self.n_sections, interp_time=interp_time
            )
            self.alpha_order = 2
        else:
            # Filling alpha_2 with zeros
            # This can be removed when the BLonD assembler is in place
            # to avoid high order momentum compaction programs filled
            # with zeros (should be propagated in RFStation.__init__())
            self.alpha_2 = np.zeros(self.alpha_0.shape)

        # Slippage factor derived from alpha, beta, gamma
        self.eta_generation()

    @property
    def Particle(self):
        from warnings import warn

        warn(
            "Particle is deprecated, use particle",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.particle

    @Particle.setter
    def Particle(self, val):
        from warnings import warn

        warn(
            "Particle is deprecated, use particle",
            DeprecationWarning,
            stacklevel=2,
        )
        self.particle = val

    @property
    def RingOptions(self):
        from warnings import warn

        warn(
            "RingOptions is deprecated, use ring_options",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ring_options

    @RingOptions.setter
    def RingOptions(self, val):
        from warnings import warn

        warn(
            "RingOptions is deprecated, use ring_options",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ring_options = val

    def eta_generation(self):
        """Function to generate the slippage factors (zeroth, first, and
        second orders, see [1]_) from the momentum compaction and the
        relativistic beta and gamma program through the cycle.

        References
        ----------
        .. [1] "Accelerator Physics," S. Y. Lee, World Scientific,
                Third Edition, 2012.
        """
        # TODO fix get\set methods

        for i in range(self.alpha_order + 1):
            getattr(self, f"_eta{i}")()

        # Fill unused eta arrays with zeros
        # This can be removed when the BLonD assembler is in place
        # to avoid high order momentum compaction programs filled
        # with zeros (should be propagated in RFStation.__init__())
        for i in range(self.alpha_order + 1, 3):
            setattr(
                self, f"eta_{i}", np.zeros([self.n_sections, self.n_turns + 1])
            )

    def _eta0(self):
        """Function to calculate the zeroth order slippage factor eta_0"""

        self.eta_0 = np.empty([self.n_sections, self.n_turns + 1])
        for i in range(0, self.n_sections):
            self.eta_0[i] = self.alpha_0[i] - self.gamma[i] ** (-2.0)

    def _eta1(self):
        """Function to calculate the first order slippage factor eta_1"""

        self.eta_1 = np.empty([self.n_sections, self.n_turns + 1])
        for i in range(0, self.n_sections):
            self.eta_1[i] = (
                3 * self.beta[i] ** 2 / (2 * self.gamma[i] ** 2)
                + self.alpha_1[i]
                - self.alpha_0[i] * self.eta_0[i]
            )

    def _eta2(self):
        """Function to calculate the second order slippage factor eta_2"""

        self.eta_2 = np.empty([self.n_sections, self.n_turns + 1])
        for i in range(0, self.n_sections):
            self.eta_2[i] = (
                -(self.beta[i] ** 2)
                * (5 * self.beta[i] ** 2 - 1)
                / (2 * self.gamma[i] ** 2)
                + self.alpha_2[i]
                - 2 * self.alpha_0[i] * self.alpha_1[i]
                + self.alpha_1[i] / self.gamma[i] ** 2
                + self.alpha_0[i] ** 2 * self.eta_0[i]
                - 3
                * self.beta[i] ** 2
                * self.alpha_0[i]
                / (2 * self.gamma[i] ** 2)
            )

    def parameters_at_time(self, cycle_moments: Iterable[float] | float):
        """Function to return various cycle parameters at a specific moment in
        time. The cycle time is defined to start at zero in turn zero.

        Parameters
        ----------
        cycle_moments : float array
            Moments of time at which cycle parameters are to be calculated [s].

        Returns
        -------
        parameters : dictionary
            Contains 'momentum', 'beta', 'gamma', 'energy', 'kin_energy',
            'f_rev', 't_rev'. 'omega_rev', 'eta_0', and 'delta_E' interpolated
            to the moments contained in the 'cycle_moments' array

        """

        parameters = {}
        parameters["momentum"] = np.interp(
            cycle_moments, self.cycle_time, self.momentum[0]
        )
        parameters["beta"] = np.interp(
            cycle_moments, self.cycle_time, self.beta[0]
        )
        parameters["gamma"] = np.interp(
            cycle_moments, self.cycle_time, self.gamma[0]
        )
        parameters["energy"] = np.interp(
            cycle_moments, self.cycle_time, self.energy[0]
        )
        parameters["kin_energy"] = np.interp(
            cycle_moments, self.cycle_time, self.kin_energy[0]
        )
        parameters["f_rev"] = np.interp(
            cycle_moments, self.cycle_time, self.f_rev
        )
        parameters["t_rev"] = np.interp(
            cycle_moments, self.cycle_time, self.t_rev
        )
        parameters["omega_rev"] = np.interp(
            cycle_moments, self.cycle_time, self.omega_rev
        )
        parameters["eta_0"] = np.interp(
            cycle_moments, self.cycle_time, self.eta_0[0]
        )
        parameters["delta_E"] = np.interp(
            cycle_moments, self.cycle_time[1:], np.diff(self.energy[0])
        )

        return parameters
