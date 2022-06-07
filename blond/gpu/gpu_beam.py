from __future__ import division
import numpy as np
import cupy as cp
from ..utils import bmath as bm
# , driver as drv, tools
from types import MethodType
from ..gpu.cupy_butils_wrap import stdKernel
from ..gpu import grid_size, block_size

from ..beam.beam import Beam

gllc = bm.getMod().get_function("gpu_losses_longitudinal_cut")
glec = bm.getMod().get_function("gpu_losses_energy_cut")
glbe = bm.getMod().get_function("gpu_losses_energy_cut")


class GpuBeam(Beam):

    @property
    def dE(self):
        return self.dE_obj.my_array

    @dE.setter
    def dE(self, value):
        self.dE_obj.my_array = value

    @property
    def dev_dE(self):
        return self.dE_obj.dev_my_array

    @dev_dE.setter
    def dev_dE(self, value):
        self.dE_obj.dev_my_array = value

    @property
    def dt(self):
        return self.dt_obj.my_array

    @dt.setter
    def dt(self, value):
        self.dt_obj.my_array = value

    @property
    def dev_dt(self):
        return self.dt_obj.dev_my_array

    @dev_dt.setter
    def dev_dt(self, value):
        self.dt_obj.dev_my_array = value

    @property
    def id(self):
        return self.id_obj.my_array

    @id.setter
    def id(self, value):
        self.id_obj.my_array = value

    @property
    def dev_id(self):
        return self.id_obj.dev_my_array

    @dev_id.setter
    def dev_id(self, value):
        self.id_obj.dev_my_array = value

    @property
    def n_macroparticles_lost(self):
        """
        Gpu Equivalent for n_macroparticles_lost
        """
        return self.n_macroparticles - int(cp.sum(self.dev_id))

    def losses_longitudinal_cut(self, dt_min, dt_max):
        """
        Gpu Equivalent for losses_longitudinal_cut
        """
        gllc(args = (self.dev_dt, self.dev_id, np.int32(self.n_macroparticles), bm.precision.real_t(dt_min),
             bm.precision.real_t(dt_max)),
             grid=grid_size, block=block_size)
        self.id_obj.invalidate_cpu()

    def losses_energy_cut(self, dE_min, dE_max):
        """
        Gpu Equivalent for losses_energy_cut
        """
        glec(args = (self.dev_dE, self.dev_id, np.int32(self.n_macroparticles), bm.precision.real_t(dE_min),
             bm.precision.real_t(dE_max)),
             grid=grid_size, block=block_size)
        self.id_obj.invalidate_cpu()

    def losses_below_energy(self, dE_min):
        """
        Gpu Equivalent for losses_below_energy
        """
        glbe(args = (self.dev_dE, self.dev_id, np.int32(self.n_macroparticles), bm.precision.real_t(dE_min)),
             grid=grid_size, block=block_size)
        self.id_obj.invalidate_cpu()

    def statistics(self):
        """
        Gpu Equivalent for statistics
        """
        ones_sum = bm.precision.real_t(cp.sum(self.dev_id))
        self.mean_dt = bm.precision.real_t(cp.sum(self.dev_dt * self.dev_id)) / ones_sum
        self.mean_dE = bm.precision.real_t(cp.sum(self.dev_dE * self.dev_id)) / ones_sum

        self.sigma_dt = np.sqrt(stdKernel(self.dev_dt, self.dev_id, self.mean_dt).get() / ones_sum)
        self.sigma_dE = np.sqrt(stdKernel(self.dev_dE, self.dev_id, self.mean_dE).get() / ones_sum)

        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_dt  # in eVs
