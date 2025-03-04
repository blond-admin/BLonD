from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cupy as cp
import numpy as np

from .beam_abstract import BeamBaseClass
from ..input_parameters.rf_parameters import RFStation
from ..input_parameters.ring import Ring
from ..trackers.utilities import is_in_separatrix

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyNDArray
    from typing import Callable, Dict
    from numpy.typing import NDArray


class MultiGpuArray:
    def __init__(self, array_cpu: NDArray, axis=0):
        """Array that is split on the memory of several GPUs one machine

        Parameters
        ----------
        array_cpu
            The array to be distributed on several GPUs
        axis
            Which axis of the array to split
        """
        self.gpu_arrays: Dict[int, CupyNDArray] = {}

        n_gpus = cp.cuda.runtime.getDeviceCount()
        sub_arrays = np.split(array_cpu, n_gpus, axis=axis)
        for gpu_i, array_tmp in enumerate(sub_arrays):
            with cp.cuda.Device(gpu_i):
                # upload to GPU
                self.gpu_arrays[gpu_i] = cp.array(array_tmp)

    def map(self, func: Callable, results: Optional[NDArray] = None):
        """Map function along all GPUs

        Parameters
        ----------
        func
            The function to calculate on the array
            results[gpu_i] = func(array)
        results
            The result must provide the correct shape,
            if the func returns array-like data
            results[gpu_i,:] = func(array)[:]

        Returns
        -------
        results
            Described above

        """
        if results is None:
            results = np.empty(len(self.gpu_arrays.keys()))
        else:
            assert results.shape[0] == len(self.gpu_arrays.keys()), (
                f"{ results.shape=}"
            )
        # todo make concurrent
        for gpu_i, array in self.gpu_arrays.items():
            with cp.cuda.Device(gpu_i):
                results[gpu_i] = func(array)
        return results

    def map_inplace(self, func: Callable, out: MultiGpuArray):
        assert len(self.gpu_arrays.keys()) == len(out.gpu_arrays.keys())
        # todo make concurrent
        for gpu_i, array in self.gpu_arrays.items():
            with cp.cuda.Device(gpu_i):
                func(array, out=out.gpu_arrays[gpu_i])


class BeamDistributedSingleNode(BeamBaseClass):
    def __init__(
        self,
        ring: Ring,
        intensity: float,
        dE: NDArray,
        dt: NDArray,
        id: NDArray,  # TODO
    ):
        """Special version of beam, which storage of dE, dt and id distributed on several GPUs"""
        assert len(dE) == len(dt), f"{len(dE)=}, but {len(dt)=}"
        assert len(dE) == len(id), f"{len(dE)=}, but {len(id)=}"

        super().__init__(
            ring=ring, n_macroparticles=len(dE), intensity=intensity
        )
        self.dE_multi_gpu = MultiGpuArray(dE)
        self.dt_multi_gpu = MultiGpuArray(dt)
        self.id_multi_gpu = MultiGpuArray(id)

    def download_ids(self):
        ids = np.concatenate(
            [
                self.id_multi_gpu.gpu_arrays[gpu_i].asnumpy()
                for gpu_i in range(self.n_gpus)
            ]
        )
        return ids

    def download_dts(self):
        dts = np.concatenate(
            [
                self.dt_multi_gpu.gpu_arrays[gpu_i].asnumpy()
                for gpu_i in range(self.n_gpus)
            ]
        )
        return dts

    def download_dEs(self):
        dEs = np.concatenate(
            [
                self.dE_multi_gpu.gpu_arrays[gpu_i].asnumpy()
                for gpu_i in range(self.n_gpus)
            ]
        )
        return dEs

    @property
    def n_gpus(self):
        return cp.cuda.runtime.getDeviceCount()

    @property
    def n_macroparticles_alive(self) -> int:
        counts = self.id_multi_gpu.map(cp.count_nonzero)
        return np.sum(counts)

    def eliminate_lost_particles(self):
        n_macroparticles_new = 0
        for gpu_i in range(self.n_gpus):
            with cp.cuda.Device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]
                select_alive = id_tmp != 0
                n_alive = cp.sum(select_alive)
                if n_alive == (len(select_alive) - 1):
                    pass
                elif cp.sum(select_alive) > 0:
                    self.n_macroparticles_eliminated += cp.sum(~select_alive)
                    self.dE_multi_gpu.gpu_arrays[gpu_i] = dE_tmp[select_alive]
                    self.dt_multi_gpu.gpu_arrays[gpu_i] = dt_tmp[select_alive]

                    self.id_multi_gpu.gpu_arrays[gpu_i] = cp.arange(
                        (n_macroparticles_new + 1),
                        (n_macroparticles_new + 1)  # next line
                        + len(self.dE_multi_gpu.gpu_arrays[gpu_i]),
                    )
                n_macroparticles_new += len(
                    self.dE_multi_gpu.gpu_arrays[gpu_i]
                )
        if n_macroparticles_new == 0:
            # AllParticlesLost
            raise RuntimeError(
                "ERROR in Beams: all particles lost and" + " eliminated!"
            )

    def statistics(self) -> None:
        self.mean_dt = self.dt_mean(ignore_id_0=True)
        self.sigma_dt = self.dt_std(ignore_id_0=True)
        # self._mpi_sumsq_dt # todo

        self.mean_dE = self.dt_mean(ignore_id_0=True)
        self.sigma_dE = self.dt_std(ignore_id_0=True)
        # self._mpi_sumsq_dE # todo

        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_dt  # in eVs

    def losses_separatrix(self, ring: Ring, rf_station: RFStation) -> None:
        for gpu_i in self.dt_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]

                # todo does this even work on GPU ever??
                # write test for is_in_separatrix()
                lost_index = ~is_in_separatrix(
                    ring, rf_station, self, dt_tmp, dE_tmp
                )

                self.id_multi_gpu.gpu_arrays[gpu_i][lost_index] = 0

    def losses_longitudinal_cut(self, dt_min: float, dt_max: float) -> None:
        for gpu_i in self.dt_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]

                lost_index = (dt_tmp < dt_min) | (dt_tmp > dt_max)
                self.id_multi_gpu.gpu_arrays[gpu_i][lost_index] = 0

    def losses_energy_cut(self, dE_min: float, dE_max: float) -> None:
        for gpu_i in self.dt_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]

                lost_index = (dE_tmp < dE_min) | (dE_tmp > dE_max)
                self.id_multi_gpu.gpu_arrays[gpu_i][lost_index] = 0

    def losses_below_energy(self, dE_min: float):
        for gpu_i in self.dt_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]

                lost_index = dE_tmp < dE_min
                self.id_multi_gpu.gpu_arrays[gpu_i][lost_index] = 0

    def dE_mean(self, ignore_id_0: bool = False):
        means = np.empty(self.n_gpus)
        for gpu_i in self.dE_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]

                if ignore_id_0:
                    mask = id_tmp > 0
                    means[gpu_i] = cp.mean(dE_tmp[mask])
                else:
                    means[gpu_i] = cp.mean(dE_tmp)

        return float(np.mean(means))

    def dE_std(self, ignore_id_0: bool = False):
        stds = np.empty(self.n_gpus)
        for gpu_i in self.dE_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]

                if ignore_id_0:
                    mask = id_tmp > 0
                    stds[gpu_i] = cp.std(dE_tmp[mask])
                else:
                    stds[gpu_i] = cp.std(dE_tmp)

        return float(np.mean(stds))

    def dt_mean(self, ignore_id_0: bool = False):
        means = np.empty(self.n_gpus)
        for gpu_i in self.dt_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]

                if ignore_id_0:
                    mask = id_tmp > 0
                    means[gpu_i] = cp.mean(dt_tmp[mask])
                else:
                    means[gpu_i] = cp.mean(dt_tmp)

        return float(np.mean(means))

    def dt_std(self, ignore_id_0: bool = False):
        stds = np.empty(self.n_gpus)
        for gpu_i in self.dt_multi_gpu.gpu_arrays.keys():
            with cp.cuda.Device(gpu_i):
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]

                if ignore_id_0:
                    mask = id_tmp > 0
                    stds[gpu_i] = cp.std(dt_tmp[mask])
                else:
                    stds[gpu_i] = cp.std(dt_tmp)

        return float(np.mean(stds))
