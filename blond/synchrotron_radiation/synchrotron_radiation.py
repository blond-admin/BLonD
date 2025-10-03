# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and
# immunities granted to it by virtue of its status as an
# Intergovernmental Organization or submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
*Class to compute synchrotron radiation damping and quantum excitation*

:Authors: **Juan F. Esteban Mueller, L. Valle**
"""

from __future__ import annotations
import warnings

import numpy as np
from typing import TYPE_CHECKING, Optional
from numpy.typing import NDArray
from ..utils.exceptions import MissingParameterError
from ..beam.beam import Beam
from ..input_parameters.ring import Ring
from ..input_parameters.rf_parameters import RFStation
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Callable, Optional

    from ..beam.beam import Beam
    from ..input_parameters.rf_parameters import RFStation
    from ..input_parameters.ring import Ring
    from ..utils.types import DeviceType

    from numpy.typing import NDArray


class SynchrotronRadiation:
    """Class to compute synchrotron radiation effects, including
    radiation damping and quantum excitation.
    For multiple RF section, instantiate one object per RF section and
    call the track() method after tracking each section.
    """

    # TO DO list:
    # - multi-turn radiation integrals handling (momentum compaction
    # factor variation, input of SR integrals TBT,
    # update of the integrals during tracking, ...),
    # - inclusion of damping wigglers,
    # - handling of lost particles during tracking,
    # - multiple RF sections

    @handle_legacy_kwargs
    def __init__(
        self,
        ring: Ring,
        rf_station: RFStation,
        beam: Beam,
        bending_radius: Optional[float] = None,
        radiation_integrals: Optional[NDArray | list] = None,
        n_kicks: Optional[int] = 1,
        quantum_excitation: Optional[bool] = True,
        python: Optional[bool] = True,
        seed: Optional[int] = None,
        shift_beam: Optional[bool] = True,
    ):
        """
        Synchrotron radiation tracker
        Calculates the energy losses per turn and longitudinal damping
        time according to the ring energy program, and implements the
        effect of synchrotron radiation damping and quantum excitation
        (if enabled) on the beam coordinates.

        Parameters
        ----------
        ring : Ring
            A Ring-type class representing the accelerator lattice.
        rf_station : RFStation
            An RFStation class containing RF parameters for the
            simulation.
        beam : Beam
            A Beam class instance representing the particle bunch or
            beam.
        bending_radius [m]: float, optional
            Bending radius used to compute the radiation integrals under
             the assumption of an isomagnetic ring.
        radiation_integrals : array_like, optional
            Radiation integrals used to calculate damping times and
            quantum excitation.
            Expected to be a list or NDArray with at least 5 elements.
        n_kicks : int, optional
            Number of discrete kicks to apply synchrotron radiation and
            quantum excitation effects. Default is 1.
        quantum_excitation : bool, optional
            If True (default), enables the quantum excitation effect
            during the simulation.
        python : bool, optional
            If True (default), uses Python implementations of the
            tracking functions.
        seed : int, optional
            Seed for random number generation when tracking in C.
        shift_beam : bool, optional
            If True, shifts the beam in phase to account for energy loss
             due to synchrotron radiation.
            (Temporary workaround until bunch generation is updated.)

        Returns
        -------
        None
        """
        self.ring = ring
        self.rf_params = rf_station
        self.beam = beam
        self.track = None
        self.beam_position_to_compensate_SR = None
        self.beam_phase_to_compensate_SR = None

        self.n_kicks = n_kicks  # To apply SR in several kicks
        np.random.seed(seed=seed)

        # Calculate static parameters
        self.c_gamma = self.ring.particle.c_gamma
        self.c_q = self.ring.particle.c_q
        # Initialize the random number array if quantum excitation
        if quantum_excitation:
            self.random_array = np.zeros(self.beam.n_macroparticles)
        # Computes the radiation integrals and initializes the SR
        # parameters
        self.assign_radiation_integrals(radiation_integrals, bending_radius)
        self.calculate_SR_params()
        self.shift_beam_function(shift_beam=shift_beam)
        self.tracker_choice(
            python=python, quantum_excitation=quantum_excitation, seed=seed
        )

    def shift_beam_function(self, shift_beam):
        """
        Displace the beam in phase to account for the energy loss due
        to synchrotron radiation (temporary until bunch generation is
        updated)
        shift_beam: bool
        """
        #
        if shift_beam and (self.rf_params.section_index == 0):
            self.beam_phase_to_compensate_SR = np.abs(
                np.arcsin(
                    self.U0
                    / (
                        self.ring.particle.charge
                        * self.rf_params.voltage[0][0]
                    )
                )
            )
            self.beam_position_to_compensate_SR = (
                self.beam_phase_to_compensate_SR
                * self.rf_params.t_rf[0, 0]
                / (2.0 * np.pi)
            )

            self.beam.dt -= self.beam_position_to_compensate_SR

    def tracker_choice(self, python, quantum_excitation, seed):
        """
        Select the right method for the tracker according to the
        selected settings
        """
        if python:
            if quantum_excitation:
                self.track = self.track_full_python
            else:
                self.track = self.track_SR_python
        else:
            if quantum_excitation:
                if seed is not None:
                    bm.set_random_seed(seed)
                self.track = self.track_full_C
            else:
                self.track = self.track_SR_C
        self.track_models: dict[str, Callable] = {
            "track_SR_python": self.track_SR_python,
            "track_full_python": self.track_full_python,
            "track_SR_C": self.track_SR_C,
            "track_full_C": self.track_full_C,
        }
        self.track_mode: str | None = None

    def assign_radiation_integrals(self, radiation_integrals, bending_radius):
        """
        Function to handle the synchrotron radiation integrals from an
        input array or a bending radius input.
        For more about synchrotron radiation damping and integral
        definition, please refer to (non-exhaustive list):
        A. Wolski, CAS Advanced Accelerator Physics, 19-29 August 2013
        H. Wiedemann, Particle Accelerator Physics, Chapter Equilibrium
        Particle Distribution, p. 384, Third Edition, Springer, 2007
        """
        if radiation_integrals is None:
            if bending_radius is None:
                if hasattr(self.ring, "I2"):
                    self.I2 = self.ring.I2
                    self.I3 = self.ring.I3
                    self.I4 = self.ring.I4
                    self.jz = 2.0 + self.I4 / self.I2
                else:
                    raise MissingParameterError(
                        "Synchrotron radiation damping "
                        "and quantum excitation require"
                        " either the bending radius "
                        + "for an isomagnetic ring, or the "
                        "first five synchrotron radiation "
                        "integrals."
                    )
            else:
                self.rho = bending_radius
                self.I2 = 2.0 * np.pi / self.rho
                self.I3 = 2.0 * np.pi / self.rho**2.0
                self.I4 = (
                    self.ring.ring_circumference
                    * self.ring.alpha_0[0, 0]
                    / self.rho**2.0
                )
                self.jz = 2.0 + self.I4 / self.I2
        else:
            if not isinstance(radiation_integrals, (np.ndarray, list)):
                raise TypeError(
                    f"Expected a list or a NDArray as an input. "
                    f"Received type(radiation_integrals)="
                    f"{type(radiation_integrals)}."
                )
            else:
                integrals = np.array(radiation_integrals)
                if len(integrals) < 5:
                    raise ValueError(
                        f"Length of radiation integrals must be "
                        f"> 5, but is {len(integrals)}"
                    )
                if bending_radius is not None:
                    warnings.warn(
                        "Synchrotron radiation integrals prevail. "
                        "'bending radius' is ignored."
                    )
                self.I2 = integrals[1]
                self.I3 = integrals[2]
                self.I4 = integrals[3]
                self.jz = 2.0 + self.I4 / self.I2

    # Method to compute the SR parameters
    def calculate_SR_params(self):
        i_turn = self.rf_params.counter[0]

        # Energy loss per turn/RF section [eV]
        self.U0 = (
            self.c_gamma
            * self.ring.energy[0, i_turn] ** 4.0
            * self.I2
            / (2.0 * np.pi)
            * self.rf_params.section_length
            / self.ring.ring_circumference
        )

        # Damping time [turns]
        self.tau_z = 2.0 / self.jz * self.ring.energy[0, i_turn] / self.U0

        # Equilibrium energy spread
        self.sigma_dE = np.sqrt(
            self.c_q
            * self.ring.gamma[0, i_turn] ** 2.0
            * self.I3
            / (self.jz * self.I2)
        )

    # Print SR parameters
    def print_SR_params(self):
        i_turn = self.rf_params.counter[0]

        print("------- Synchrotron radiation parameters -------")
        print(f"jz = {self.jz:1.8f}")
        if self.rf_params.section_length == self.ring.ring_circumference:
            print(f"Energy loss per turn = {self.U0 / 1e9:1.4f} GeV/turn")
            print(f"Damping time = {self.tau_z:1.4f} turns")
        else:
            print(
                "Energy loss per RF section = {0:1.4f} GeV/section".format(
                    self.U0 * 1e-9
                )
            )
            print(
                "Energy loss per turn = {0:1.4f} GeV/turn".format(
                    self.U0
                    * 1e-9
                    * self.ring.ring_circumference
                    / self.rf_params.section_length
                )
            )
            print(
                "Damping time = {0:1.4f} turns".format(
                    self.tau_z
                    * self.rf_params.section_length
                    / self.ring.ring_circumference
                )
            )
        print(
            f"Equilibrium energy spread = {self.sigma_dE * 100:1.4f}%"
            + f"({self.sigma_dE * self.ring.energy[0, i_turn] * 1e-6:1.4f}) MeV"
        )
        print("------------------------------------------------")

    def track_SR_python(self):
        """
        Adds the effect of synchrotron radiation damping on the beam
        coordinates. Quantum excitation ignored.
        """
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (
            i_turn != 0
            and self.ring.energy[0, i_turn] != self.ring.energy[0, i_turn - 1]
        ):
            self.calculate_SR_params()
        for i in range(self.n_kicks):
            self.beam.dE += -(
                2.0 / self.tau_z / self.n_kicks * self.beam.dE
                + self.U0 / self.n_kicks
            )

    def track_full_python(self):
        """
        Adds the effect of synchrotron radiation damping and quantum
        excitation on the beam coordinates.
        """
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (
            i_turn != 0
            and self.ring.energy[0, i_turn] != self.ring.energy[0, i_turn - 1]
        ):
            self.calculate_SR_params()
        for i in range(self.n_kicks):
            self.beam.dE += -(
                2.0 / self.tau_z / self.n_kicks * self.beam.dE
                # synchrotron radiation damping
                + self.U0 / self.n_kicks
                # energy lost due to synchrotron radiation
                - 2.0
                * self.sigma_dE
                / np.sqrt(self.tau_z * self.n_kicks)
                * self.beam.energy
                * np.random.normal(size=self.beam.n_macroparticles)
            )
            # quantum excitation kick

    # Track particles with SR only (without quantum excitation)
    # C implementation
    def track_SR_C(self):
        """
        Adds the effect of synchrotron radiation damping on the beam
        coordinates. Quantum excitation ignored.
        """
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (
            i_turn != 0
            and self.ring.energy[0, i_turn] != self.ring.energy[0, i_turn - 1]
        ):
            self.calculate_SR_params()

        bm.synchrotron_radiation(
            self.beam.dE, self.U0, self.n_kicks, self.tau_z
        )

    # Track particles with SR and quantum excitation. C implementation
    def track_full_C(self):
        """
        Adds the effect of synchrotron radiation damping and quantum
        excitation on the beam coordinates.
        """
        i_turn = self.rf_params.counter[0]
        # Recalculate SR parameters if energy changes
        if (
            i_turn != 0
            and self.ring.energy[0, i_turn] != self.ring.energy[0, i_turn - 1]
        ):
            self.calculate_SR_params()

        bm.synchrotron_radiation_full(
            self.beam.dE,
            self.U0,
            self.n_kicks,
            self.tau_z,
            self.sigma_dE,
            self.ring.energy[0, i_turn],
        )

    def to_gpu(self, recursive=True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, "_device") and self._device == "GPU":
            return

        # No arrays need to be transfered

        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, "_device") and self._device == "CPU":
            return

        # No arrays need to be transfered

        # to make sure it will not be called again
        self._device: DeviceType = "CPU"
