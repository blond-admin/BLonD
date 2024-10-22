# Future
from __future__ import annotations

# General imports
import numpy as np
import scipy.constants as cont

try:
    import cupy as cp
    _CP = True
except ImportError:
    _CP = False

from typing import TYPE_CHECKING

# BLonD imports
from ..utils import bmath as bm

if TYPE_CHECKING:
    from typing import Iterable

    from numpy.typing import ArrayLike as ArrayLike
    from numpy.typing import NDArray as NDArray

    from ..beam.beam import Beam


_M_E = cont.physical_constants['electron mass energy equivalent in MeV'][0]*1E6
_R_E = cont.physical_constants['classical electron radius'][0]


class ElectronCooling:

    def __init__(self, beam: Beam, length: float,
                 gun_voltage: float | ArrayLike,
                 e_energy_spread: float | ArrayLike,
                 density_electrons: float | ArrayLike,
                 cycle_time: Iterable[float], counter: list[int]):

        self.beam = beam
        self.length = length

        self.gun_voltage = gun_voltage
        self.e_energy_spread = e_energy_spread
        self.density_electrons = density_electrons
        self._interpolate(cycle_time)

        self._counter = counter

        self._part_mass = self.beam.Particle.mass
        self._part_charge = self.beam.Particle.charge
        self._device = 'CPU'


    def _cooling_force(self, turn: int, particle_velocity: float | NDArray):

        v_gun = self.gun_voltage[turn]
        density = self.density_electrons[turn]
        e_spread = self.e_energy_spread[turn]

        e_beta = 1/bm.sqrt(1 + _M_E**2/((_M_E + v_gun)**2 - _M_E**2))
        e_beta_spread = 1/bm.sqrt(1 + _M_E**2 / ((_M_E + v_gun*e_spread)**2
                                               - _M_E**2))

        rel_vel = particle_velocity - e_beta * cont.c

        e_vel_spread = e_beta_spread*cont.c

        vel_fact = rel_vel/(bm.abs(rel_vel)**3 + 2*e_vel_spread**2)
        charge_fact = self._part_charge**2 * cont.e**2

        force = (-12*np.pi*charge_fact * cont.c**2 * _R_E
                 * density*self.length*vel_fact)

        return force


    def track(self):

        part_energy = self.beam.dE + self._part_mass
        part_beta = 1/bm.sqrt(1 + self._part_mass**2
                              / (part_energy**2 - self._part_mass**2))

        particle_velocity = part_beta * cont.c

        self.beam.dE += self._cooling_force(self.counter[0], particle_velocity)


    def _interpolate(self, cycle_time: Iterable[float]):

        if hasattr(self.gun_voltage, "__iter__"):
            self.gun_voltage = np.interp(cycle_time, self.gun_voltage[0],
                                         self.gun_voltage[1])
        else:
            self.gun_voltage = np.zeros_like(cycle_time) + self.gun_voltage


        if hasattr(self.e_energy_spread, "__iter__"):
            self.e_energy_spread = np.interp(cycle_time,
                                             self.e_energy_spread[0],
                                             self.e_energy_spread[1])
        else:
            self.e_energy_spread = (np.zeros_like(cycle_time)
                                    + self.e_energy_spread)


        if hasattr(self.density_electrons, "__iter__"):
            self.density_electrons = np.interp(cycle_time,
                                               self.density_electrons[0],
                                               self.density_electrons[1])
        else:
            self.density_electrons = (np.zeros_like(cycle_time)
                                      + self.density_electrons)


    def to_gpu(self, recursive = True):

        if not _CP:
            raise RuntimeError("Cannot send to GPU, cupy not available.")

        if self._device == "CPU":
            self.gun_voltage = cp.array(self.gun_voltage)
            self.e_energy_spread = cp.array(self.e_energy_spread)
        else:
            raise RuntimeError("Cannot send to GPU, already there")


    def to_cpu(self):

        if not _CP:
            raise RuntimeError("Cannot retrieve from GPU, cupy not available.")

        if self._device == "GPU":
            self.gun_voltage = self.gun_voltage.get()
            self.e_energy_spread = self.e_energy_spread.get()
        else:
            raise RuntimeError("Cannot send to CPU, already there")
