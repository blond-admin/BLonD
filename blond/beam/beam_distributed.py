from __future__ import annotations

from concurrent import futures
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
    def __init__(
        self, array_cpu: NDArray, *, axis=0, mock_n_gpus: Optional[int] = None
    ):
        """Array that is split to the memory of several GPUs one machine

        Parameters
        ----------
        array_cpu
            The array to be distributed on several GPUs
        axis
            Which axis of the array to split
        mock_n_gpus
            Pretend to have n_gpus when the system has only one.
            This should be used for testing only.
        """
        self.mock_n_gpus = mock_n_gpus

        self.gpu_arrays: Dict[int, CupyNDArray] = {}

        if self.mock_n_gpus is not None:
            n_gpus = self.mock_n_gpus
        else:
            n_gpus = cp.cuda.runtime.getDeviceCount()

        sub_arrays = np.split(array_cpu, n_gpus, axis=axis)
        for gpu_i, array_tmp in enumerate(sub_arrays):
            with self.get_device(gpu_i=gpu_i):
                # upload to GPU
                self.gpu_arrays[gpu_i] = cp.array(array_tmp)

    def get_device(self, gpu_i: int):
        """Get the device of the selected GPU"""
        if self.mock_n_gpus:
            device_i = cp.cuda.Device(0)
        else:
            device_i = cp.cuda.Device(gpu_i)
        return device_i

    def map(self, func: Callable, **kwargs):
        """Map function along all GPUs

        Parameters
        ----------
        func
            The function to calculate on the array
            func(array_gpu_i)

        Returns
        -------
        results
            List of results of func(array_gpu_i)
        """

        results = []
        # Executor will run tasks concurrently
        with futures.ThreadPoolExecutor() as executor:
            futures_ = []

            # Submit the function to be executed on each GPU
            for gpu_i, array in self.gpu_arrays.items():
                futures_.append(
                    executor.submit(
                        self._map_gpu_helper,
                        *(func, gpu_i, array, kwargs),
                    )
                )

            # Wait for all futures to complete and collect their results
            for future in futures_:
                result = future.result()  # Get the result of each future
                if result is not None:  # Optional check for None values
                    results.append(
                        result
                    )  # Append the result to the results list
                else:
                    raise ValueError(f"{result=}")
        return results

    def map_inplace(self, func: Callable, out: MultiGpuArray):
        """Map function along all GPUs and write on 'out'

        Parameters
        ----------
        func
            The function to calculate on the array
            func(array, out=out.gpu_arrays[gpu_i])
        out
            The function should write its results
            on the out array.

        """
        assert len(self.gpu_arrays.keys()) == len(out.gpu_arrays.keys())

        # Executor will run tasks concurrently
        with futures.ThreadPoolExecutor() as executor:
            futures_ = []

            # Submit the function to be executed on each GPU
            for gpu_i, array in self.gpu_arrays.items():
                futures_.append(
                    executor.submit(
                        self._map_inplace_gpu_helper,
                        *(func, gpu_i, array, out),
                    )
                )

            # Wait for all futures to complete
            futures.wait(futures_)

    def _map_inplace_gpu_helper(
        self,
        func: Callable,
        gpu_i: int,
        array: CupyNDArray,
        out: MultiGpuArray,
    ):
        # Perform device context management and the actual function
        with self.get_device(gpu_i=gpu_i):
            func(array, out=out.gpu_arrays[gpu_i])

    def _map_gpu_helper(
        self, func: Callable, gpu_i: int, array: CupyNDArray, kwargs: Dict
    ):
        # Perform device context management and the actual function
        with self.get_device(gpu_i=gpu_i):
            return func(array, **kwargs)

    def download_array(self):
        """Get arrays from several GPUs as one CPU array"""
        array = np.concatenate(
            [self.gpu_arrays[gpu_i].get() for gpu_i in self.gpu_arrays.keys()]
        )
        return array


class BeamDistributedSingleNode(BeamBaseClass):
    def __init__(
        self,
        ring: Ring,
        intensity: float,
        dE: NDArray,
        dt: NDArray,
        id: NDArray,  # TODO
        mock_n_gpus: Optional[int] = None,
    ):
        """Special version of beam, which storage of dE, dt and id distributed on several GPUs"""
        assert len(dE) == len(dt), f"{len(dE)=}, but {len(dt)=}"
        assert len(dE) == len(id), f"{len(dE)=}, but {len(id)=}"

        super().__init__(
            ring=ring, n_macroparticles=len(dE), intensity=intensity
        )
        self.__dE_multi_gpu = MultiGpuArray(dE, mock_n_gpus=mock_n_gpus)
        self.__dt_multi_gpu = MultiGpuArray(dt, mock_n_gpus=mock_n_gpus)
        self.__id_multi_gpu = MultiGpuArray(id, mock_n_gpus=mock_n_gpus)

    @property
    def dE_multi_gpu(self):
        return self.__dE_multi_gpu

    @property
    def dt_multi_gpu(self):
        return self.__dt_multi_gpu

    @property
    def id_multi_gpu(self):
        return self.__id_multi_gpu

    def map(self, func: Callable, **kwargs):
        """Map function along all GPUs

        Parameters
        ----------
        func
            The function to calculate on the array
            func(array_gpu_i)

        Returns
        -------
        results
            List of results of func(array_gpu_i)
        """

        results = []
        # Executor will run tasks concurrently
        with futures.ThreadPoolExecutor() as executor:
            futures_ = []

            # Submit the function to be executed on each GPU
            for gpu_i in range(self.n_gpus):
                futures_.append(
                    executor.submit(
                        self._map_gpu_helper,
                        *(func, gpu_i, kwargs),
                    )
                )

            # Wait for all futures to complete and collect their results
            for future in futures_:
                result = future.result()  # Get the result of each future
                if result is not None:  # Optional check for None values
                    results.append(result)
        return results

    def _map_gpu_helper(self, func: Callable, gpu_i: int, kwargs: Dict):
        # Perform device context management and the actual function
        with self.dE_multi_gpu.get_device(gpu_i=gpu_i):
            return func(
                dt_gpu_i=self.dt_multi_gpu.gpu_arrays[gpu_i],
                dE_gpu_i=self.dE_multi_gpu.gpu_arrays[gpu_i],
                id_gpu_i=self.id_multi_gpu.gpu_arrays[gpu_i],
                **kwargs,
            )

    def download_ids(self):
        return self.id_multi_gpu.download_array()

    def download_dts(self):
        return self.dt_multi_gpu.download_array()

    def download_dEs(self):
        return self.dE_multi_gpu.download_array()

    @property
    def n_gpus(self) -> int:
        if self.dE_multi_gpu.mock_n_gpus is not None:
            return self.dE_multi_gpu.mock_n_gpus
        else:
            return cp.cuda.runtime.getDeviceCount()

    @property
    def n_macroparticles_alive(self) -> int:
        counts = self.id_multi_gpu.map(lambda x: int(cp.count_nonzero(x)))
        return np.sum(counts)

    def eliminate_lost_particles(self):
        n_macroparticles_new = 0
        for gpu_i in range(self.n_gpus):
            # sequential because of 'id_multi_gpu' dependence on 'n_macroparticles_new'
            with self.dE_multi_gpu.get_device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]
                select_alive = id_tmp != 0
                n_alive = cp.sum(select_alive)
                if n_alive == (len(select_alive) - 1):
                    pass
                else:
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
        self.n_macroparticles = n_macroparticles_new
        if n_macroparticles_new == 0:
            # AllParticlesLost
            raise RuntimeError(
                "ERROR in Beams: all particles lost and" + " eliminated!"
            )

    def statistics(self) -> None:
        self.mean_dt = self.dt_mean(ignore_id_0=True)
        self.sigma_dt = self.dt_std(ignore_id_0=True)
        # self._mpi_sumsq_dt # todo

        self.mean_dE = self.dE_mean(ignore_id_0=True)
        self.sigma_dE = self.dE_std(ignore_id_0=True)
        # self._mpi_sumsq_dE # todo

        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_dt  # in eVs

    def losses_separatrix(self, ring: Ring, rf_station: RFStation) -> None:
        self.map(
            _losses_separatrix_helper,
            beam=self,
            ring=ring,
            rf_station=rf_station,
        )

    def losses_longitudinal_cut(self, dt_min: float, dt_max: float) -> None:
        self.map(_losses_longitudinal_cut_helper, dt_min=dt_min, dt_max=dt_max)

    def losses_energy_cut(self, dE_min: float, dE_max: float) -> None:
        self.map(_losses_energy_cut_helper, dE_min=dE_min, dE_max=dE_max)

    def losses_below_energy(self, dE_min: float):
        self.map(_losses_below_energy_helper, dE_min=dE_min)

    def dE_mean(self, ignore_id_0: bool = False):
        if ignore_id_0:
            means = self.map(_dE_mean_helper_ignore_id_0)
        else:
            means = self.dE_multi_gpu.map(_dE_mean_helper)

        return float(np.mean(means))

    def dE_std(self, ignore_id_0: bool = False):
        mean = self.dE_mean(ignore_id_0=ignore_id_0)
        if ignore_id_0:
            sums = self.map(_dE_std_helper_ignore_id_0, mean=mean)
            N = self.n_macroparticles_alive
        else:
            sums = self.dE_multi_gpu.map(_dE_std_helper, mean=mean)
            N = self.n_macroparticles

        sums = np.array(sums)
        return np.sqrt(np.sum(sums) / N)

    def dt_mean(self, ignore_id_0: bool = False):
        if ignore_id_0:
            means = self.map(_dt_mean_helper_ignore_id_0)
        else:
            means = self.dt_multi_gpu.map(_dt_mean_helper)

        return float(np.mean(means))

    def dt_std(self, ignore_id_0: bool = False):
        mean = self.dt_mean(ignore_id_0=ignore_id_0)
        if ignore_id_0:
            sums = self.map(_dt_std_helper_ignore_id_0, mean=mean)
            N = self.n_macroparticles_alive
        else:
            sums = self.dt_multi_gpu.map(_dt_std_helper, mean=mean)
            N = self.n_macroparticles

        sums = np.array(sums)
        return np.sqrt(np.sum(sums) / N)


def _losses_separatrix_helper(
    dt_gpu_i, dE_gpu_i, id_gpu_i, beam, ring, rf_station
):
    lost_index = ~is_in_separatrix(ring, rf_station, beam, dt_gpu_i, dE_gpu_i)
    id_gpu_i[lost_index] = 0


def _losses_longitudinal_cut_helper(
    dt_gpu_i, dE_gpu_i, id_gpu_i, dt_min, dt_max
):
    id_gpu_i[(dt_gpu_i < dt_min) | (dt_gpu_i > dt_max)] = 0


def _losses_energy_cut_helper(dt_gpu_i, dE_gpu_i, id_gpu_i, dE_min, dE_max):
    id_gpu_i[(dE_gpu_i < dE_min) | (dE_gpu_i > dE_max)] = 0


def _losses_below_energy_helper(dt_gpu_i, dE_gpu_i, id_gpu_i, dE_min):
    id_gpu_i[dE_gpu_i < dE_min] = 0


def _dE_mean_helper(dE_gpu_i):
    return float(cp.mean(dE_gpu_i))


def _dE_mean_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i):
    mask = id_gpu_i > 0
    masked = dE_gpu_i[mask]
    if len(masked) == 0:
        return None
    else:
        return float(cp.mean(masked))


def _dE_std_helper(dE_gpu_i, mean):
    tmp = dE_gpu_i - mean
    return float(cp.sum(tmp * tmp))


def _dE_std_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i, mean):
    mask = id_gpu_i > 0
    if not cp.any(mask):
        return None
    else:
        tmp = dE_gpu_i[mask] - mean
        return float(cp.sum(tmp * tmp))


def _dt_mean_helper(dt_gpu_i):
    return float(cp.mean(dt_gpu_i))


def _dt_mean_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i):
    mask = id_gpu_i > 0
    masked = dt_gpu_i[mask]
    if len(masked) == 0:
        return None
    else:
        return float(cp.mean(masked))


def _dt_std_helper(dt_gpu_i, mean):
    tmp = dt_gpu_i - mean
    return float(cp.sum(tmp * tmp))


def _dt_std_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i, mean):
    mask = id_gpu_i > 0
    if not cp.any(mask):
        return None
    else:
        tmp = dt_gpu_i[mask] - mean
        return float(cp.sum(tmp * tmp))
