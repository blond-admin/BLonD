from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cupy as cp
import numpy as np

from ._beam_distributed_gluecode import (
    _kick_helper,
    _drift_helper,
    _losses_separatrix_helper,
    _losses_longitudinal_cut_helper,
    _losses_energy_cut_helper,
    _losses_below_energy_helper,
    _dE_mean_helper,
    _dE_mean_helper_ignore_id_0,
    _dE_std_helper,
    _dE_std_helper_ignore_id_0,
    _dt_mean_helper,
    _dt_mean_helper_ignore_id_0,
    _dt_std_helper,
    _dt_std_helper_ignore_id_0,
    _linear_interp_kick_helper,
    _kickdrift_considering_periodicity_helper,
)
from ..beam.beam import Beam
from ..beam.beam_abstract import BeamBaseClass
from ..gpu.butils_wrap_cupy import (
    slice_beam,
)
from ..input_parameters.rf_parameters import RFStation
from ..input_parameters.ring import Ring
from ..utils import precision

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyNDArray
    from typing import Callable, Dict
    from numpy.typing import NDArray as NumpyNDArray

    from ..utils.types import SolverTypes

max_n_gpus = cp.cuda.runtime.getDeviceCount()
all_devices = [cp.cuda.Device(gpu_i) for gpu_i in range(max_n_gpus)]
streams = [cp.cuda.Stream(non_blocking=True) for i in range(16)]  # todo


def get_device(gpu_i: int):
    """Get the device of the selected GPU"""

    if max_n_gpus == 1:
        device_i = streams[gpu_i]
    else:
        device_i = all_devices[gpu_i]
    return device_i


class DistributedMultiGpuArray:
    # DEVELOPER NOTE:
    # At the time of writing (Q1 2025),
    # cupyx.distributed.array.DistributedArray
    # exists already, but most features are "not supported"
    # for this reason it is not used for now

    def __init__(
        self,
        array_cpu: NumpyNDArray,
        *,
        axis=0,
        mock_n_gpus: Optional[int] = None,
    ):
        """Array that is split to the memory of several GPUs one machine

        Parameters
        ----------
        array_cpu
            The CPU array to be distributed on several GPUs
        axis
            Which axis of the array to split
        mock_n_gpus
            Pretend to have n_gpus when the system has only one.
            This should be used for testing only.
        """

        self.mock_n_gpus = mock_n_gpus

        if self.mock_n_gpus is not None:
            n_gpus = self.mock_n_gpus
        else:
            n_gpus = max_n_gpus

        self.gpu_arrays: Dict[int, CupyNDArray] = {}
        self.buffers: Dict[int, CupyNDArray] = {}
        self.buffers_int: Dict[int, CupyNDArray] = {}

        sub_arrays = np.array_split(array_cpu, n_gpus, axis=axis)
        for gpu_i, array_tmp in enumerate(sub_arrays):
            with get_device(gpu_i=gpu_i):
                # upload to GPU
                self.gpu_arrays[gpu_i] = cp.array(
                    array_tmp, dtype=array_cpu.dtype
                )

                self.buffers[gpu_i] = cp.empty((1,), dtype=array_cpu.dtype)
                self.buffers_int[gpu_i] = cp.empty((1,), dtype=int)
        print(f"{len(self.gpu_arrays)=}")

    def get_buffer(self):
        results = []
        for gpu_i, buffer in self.buffers.items():
            with get_device(gpu_i=gpu_i):
                results.append(buffer[0].get())
        return results

    def get_buffer_int(self):
        results = []
        for gpu_i, buffer in self.buffers_int.items():
            with get_device(gpu_i=gpu_i):
                results.append(buffer[0].get())
        return results

    def map_no_result(self, func: Callable, **kwargs):
        """Execute function on all GPUs

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

        for gpu_i, array in self.gpu_arrays.items():
            with get_device(gpu_i=gpu_i):
                func(array, **kwargs)
        cp.cuda.runtime.deviceSynchronize()

    def map_float(self, func: Callable, **kwargs):
        """Execute function on all GPUs

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

        for gpu_i, array in self.gpu_arrays.items():
            with get_device(gpu_i=gpu_i):
                val = func(array, **kwargs)
                self.buffers[gpu_i][0] = val

    def map_int(self, func: Callable, **kwargs):
        """Execute function on all GPUs

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

        for gpu_i, array in self.gpu_arrays.items():
            with get_device(gpu_i=gpu_i):
                val = func(array, **kwargs)
                self.buffers_int[gpu_i][0] = val

    def min(self):
        """Minimum of entire array"""
        self.map_float(lambda x: cp.min(x))
        return np.min(self.get_buffer())

    def max(self):
        """Maximum of entire array"""
        self.map_float(lambda x: cp.max(x))
        return np.max(self.get_buffer())

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
        dE: NumpyNDArray,
        dt: NumpyNDArray,
        id_: NumpyNDArray,
        mock_n_gpus: Optional[int] = None,
    ):
        """Special version of beam, which storage of dE, dt and id distributed on several GPUs

        Parameters
        ----------
        ring
            Class containing the general properties of the synchrotron.
        intensity
            total intensity of the beam in number of charges [].
        dE
            Beam arrival times with respect to synchronous time [s].
        dt
            Beam energy offset with respect to the synchronous particle [eV].
        id_
            Index of the particle (might change during execution)
        mock_n_gpus
            Pretend to have n_gpus when the system has only one.
            This should be used for testing only.
        """
        assert len(dE) == len(dt), f"{len(dE)=}, but {len(dt)=}"
        assert len(dE) == len(id_), f"{len(dE)=}, but {len(id_)=}"

        super().__init__(
            ring=ring, n_macroparticles=len(dE), intensity=intensity
        )
        self.__dE_multi_gpu = DistributedMultiGpuArray(
            dE, mock_n_gpus=mock_n_gpus
        )
        self.__dt_multi_gpu = DistributedMultiGpuArray(
            dt, mock_n_gpus=mock_n_gpus
        )
        self.__id_multi_gpu = DistributedMultiGpuArray(
            id_, mock_n_gpus=mock_n_gpus
        )
        self.profile_multi_gpu = {}
        self.n_bins = -1

        self.buffers_float: Dict[int, CupyNDArray] = {}
        self.buffers_int: Dict[int, CupyNDArray] = {}
        for gpu_i, array_tmp in enumerate(self.__dE_multi_gpu.gpu_arrays):
            with get_device(gpu_i=gpu_i):
                self.buffers_float[gpu_i] = cp.empty((1,), dtype=dE.dtype)
                self.buffers_int[gpu_i] = cp.empty((1,), dtype=int)

    @staticmethod
    def from_beam(beam: Beam, ring: Ring, mock_n_gpus: Optional[int] = None):
        _beam = BeamDistributedSingleNode(
            ring=ring,
            intensity=beam.intensity,
            dE=beam.dE,
            dt=beam.dt,
            id_=beam.id,
            mock_n_gpus=mock_n_gpus,
        )
        return _beam

    def get_buffer(self):
        results = []
        for gpu_i, buffer_i in self.buffers_float.items():
            with get_device(gpu_i=gpu_i):
                results.append(buffer_i[0].get())
        return results

    def get_buffer_int(self):
        results = []
        for gpu_i, buffer_i in self.buffers_int.items():
            with get_device(gpu_i=gpu_i):
                results.append(buffer_i[0].get())
        return results

    def _init_profile_multi_gpu(self, n_bins):
        if self.n_bins == n_bins:
            return
        from blond.utils import precision

        self.n_bins = n_bins
        self.profile_multi_gpu = {}
        for gpu_i in range(len(self.dE_multi_gpu.gpu_arrays)):
            with get_device(gpu_i=gpu_i):
                self.profile_multi_gpu[gpu_i] = cp.zeros(
                    n_bins, dtype=precision.real_t
                )

    @property
    def dE_multi_gpu(self):
        return self.__dE_multi_gpu

    @property
    def dt_multi_gpu(self):
        return self.__dt_multi_gpu

    @property
    def id_multi_gpu(self):
        return self.__id_multi_gpu

    def map_no_result(self, func: Callable, **kwargs):
        """Execute function on all GPUs

        Parameters
        ----------
        func
            The function to calculate on the array
            value_float = func(array_gpu_i)

        Returns
        -------
        results
            List of results of func(array_gpu_i)
        """

        for gpu_i in range(self.n_gpus):
            with get_device(gpu_i):
                func(
                    dt_gpu_i=self.dt_multi_gpu.gpu_arrays[gpu_i],
                    dE_gpu_i=self.dE_multi_gpu.gpu_arrays[gpu_i],
                    id_gpu_i=self.id_multi_gpu.gpu_arrays[gpu_i],
                    **kwargs,
                )

    def map(self, func: Callable, **kwargs):
        """Execute function on all GPUs

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

        for gpu_i in range(self.n_gpus):
            with get_device(gpu_i):
                self.buffers_float[gpu_i][0] = func(
                    dt_gpu_i=self.dt_multi_gpu.gpu_arrays[gpu_i],
                    dE_gpu_i=self.dE_multi_gpu.gpu_arrays[gpu_i],
                    id_gpu_i=self.id_multi_gpu.gpu_arrays[gpu_i],
                    **kwargs,
                )
        return results

    def download_ids(self):
        """Collects the IDs from all GPUs to one CPU array"""
        return self.id_multi_gpu.download_array()

    def download_dts(self):
        """Collects dt from all GPUs to one CPU array"""
        return self.dt_multi_gpu.download_array()

    def download_dEs(self):
        """Collects dE from all GPUs to one CPU array"""
        return self.dE_multi_gpu.download_array()

    @property
    def n_gpus(self) -> int:
        """Number of GPUs"""
        if self.dE_multi_gpu.mock_n_gpus is not None:
            return self.dE_multi_gpu.mock_n_gpus
        else:
            return max_n_gpus

    @property
    def n_macroparticles_alive(self) -> int:
        """Number of macro-particles marked as alive (id ≠ 0)"""
        self.id_multi_gpu.map_int(lambda x: cp.count_nonzero(x))
        counts = self.id_multi_gpu.get_buffer_int()
        return np.sum(counts)

    def eliminate_lost_particles(self):
        """Eliminate lost particles from the beam coordinate arrays"""
        n_macroparticles_new = 0
        for gpu_i in range(self.n_gpus):
            # sequential because of 'id_multi_gpu' dependence on 'n_macroparticles_new'
            with get_device(gpu_i):
                dE_tmp = self.dE_multi_gpu.gpu_arrays[gpu_i]
                dt_tmp = self.dt_multi_gpu.gpu_arrays[gpu_i]
                id_tmp = self.id_multi_gpu.gpu_arrays[gpu_i]
                select_alive: CupyNDArray = id_tmp[:] != 0  # noqa
                n_alive = cp.sum(select_alive)
                if n_alive == (len(select_alive) - 1):
                    pass
                else:
                    self.n_macroparticles_eliminated += cp.sum(~select_alive)
                    self.dE_multi_gpu.gpu_arrays[gpu_i] = cp.ascontiguousarray(
                        dE_tmp[select_alive]
                    )
                    self.dt_multi_gpu.gpu_arrays[gpu_i] = cp.ascontiguousarray(
                        dt_tmp[select_alive]
                    )

                    self.id_multi_gpu.gpu_arrays[gpu_i] = cp.arange(
                        (n_macroparticles_new + 1),
                        (n_macroparticles_new + 1)  # next line
                        + len(self.dE_multi_gpu.gpu_arrays[gpu_i]),
                    )

                n_macroparticles_new += int(n_alive)
        self.n_macroparticles = n_macroparticles_new
        if n_macroparticles_new == 0:
            # AllParticlesLost
            raise RuntimeError(
                "ERROR in Beams: all particles lost and" + " eliminated!"
            )

    def statistics(self) -> None:
        r"""Update statistics of dE and dE array

        Notes
        -----
        Following attributes are updated:
        - mean_dt
        - mean_dE
        - sigma_dt
        - sigma_dE
        """
        self.mean_dt = self.dt_mean(ignore_id_0=True)
        self.sigma_dt = self.dt_std(ignore_id_0=True)
        # self._mpi_sumsq_dt # todo

        self.mean_dE = self.dE_mean(ignore_id_0=True)
        self.sigma_dE = self.dE_std(ignore_id_0=True)
        # self._mpi_sumsq_dE # todo

        self.epsn_rms_l = np.pi * self.sigma_dE * self.sigma_dt  # in eVs

    def losses_separatrix(self, ring: Ring, rf_station: RFStation) -> None:
        """Mark particles outside separatrix as not-alive (id=0)

        Parameters
        ----------
        ring : Ring
            Class containing the general properties of the synchrotron
        rf_station : RFStation
            Class containing all the RF parameters for all the RF systems
            in one ring segment

        """
        # todo make concurrent execution possible
        self.map_no_result(
            _losses_separatrix_helper,
            ring=ring,
            beam=self,
            rf_station=rf_station,
        )

    def losses_longitudinal_cut(self, dt_min: float, dt_max: float) -> None:
        """Mark particles outside time range as not-alive (id=0)

        Parameters
        ----------
        dt_min : float
            Lower limit (dt=dt_min is kept)
        dt_max : float
            Upper limit (dt=dt_max is kept)
        """
        self.map_no_result(
            _losses_longitudinal_cut_helper, dt_min=dt_min, dt_max=dt_max
        )

    def losses_energy_cut(self, dE_min: float, dE_max: float) -> None:
        """Mark particles outside energy range as not-alive (id=0)

        Parameters
        ----------
        dE_min : float
            Lower limit (dE=dE_min is kept)
        dE_max : float
            Upper limit (dE=dE_max is kept)
        """
        self.map_no_result(
            _losses_energy_cut_helper, dE_min=dE_min, dE_max=dE_max
        )

    def losses_below_energy(self, dE_min: float):
        """Mark particles outside energy range as not-alive (id=0)

        Parameters
        ----------
        dE_min : float
            Lower limit (dE=dE_min is kept)
        """
        self.map_no_result(_losses_below_energy_helper, dE_min=dE_min)

    def dE_mean(self, ignore_id_0: bool = False):
        """Calculate mean of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            self.map(_dE_mean_helper_ignore_id_0)
            means = self.get_buffer()

        else:
            self.dE_multi_gpu.map_float(_dE_mean_helper)
            means = self.dE_multi_gpu.get_buffer()

        return float(np.nanmean(means))

    def dE_std(self, ignore_id_0: bool = False):
        """Calculate standard deviation of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        mean = self.dE_mean(ignore_id_0=ignore_id_0)
        if ignore_id_0:
            self.map(_dE_std_helper_ignore_id_0, mean=mean)
            sums = self.get_buffer()
            N = self.n_macroparticles_alive
        else:
            self.dE_multi_gpu.map_float(_dE_std_helper, mean=mean)
            sums = self.dE_multi_gpu.get_buffer()
            N = self.n_macroparticles

        return np.sqrt(np.nansum(sums) / N)

    def dt_min(self):  # todo ignore lost particles?
        """Minimum of all 'dt'"""
        return self.dt_multi_gpu.min()

    def dE_min(self):  # todo ignore lost particles?
        """Minimum of all 'dE'"""
        return self.dE_multi_gpu.min()

    def dt_max(self):  # todo ignore lost particles?
        """Maximum of all 'dt'"""
        return self.dt_multi_gpu.max()

    def dE_max(self):  # todo ignore lost particles?
        """Maximum of all 'dE'"""
        return self.dE_multi_gpu.max()

    def dt_mean(self, ignore_id_0: bool = False):
        """Calculate mean of time

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            self.map(_dt_mean_helper_ignore_id_0)
            means = self.get_buffer()
        else:
            self.dt_multi_gpu.map_float(_dt_mean_helper)
            means = self.dt_multi_gpu.get_buffer()
        return float(np.nanmean(means))

    def dt_std(self, ignore_id_0: bool = False):
        """Calculate standard deviation of time

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        mean = self.dt_mean(ignore_id_0=ignore_id_0)
        if ignore_id_0:
            self.map(_dt_std_helper_ignore_id_0, mean=mean)
            sums = self.get_buffer()
            N = self.n_macroparticles_alive
        else:
            self.dt_multi_gpu.map_float(_dt_std_helper, mean=mean)
            sums = self.dt_multi_gpu.get_buffer()
            N = self.n_macroparticles
        return np.sqrt(np.nansum(sums) / N)

    def slice_beam(
        self, profile: CupyNDArray, cut_left: float, cut_right: float
    ):
        """Computes a histogram of the dt coordinates"""
        self._init_profile_multi_gpu(n_bins=len(profile))
        for gpu_i, dt_multi_gpu in self.dt_multi_gpu.gpu_arrays.items():
            with get_device(gpu_i=gpu_i):
                slice_beam(
                    dt=dt_multi_gpu,
                    profile=self.profile_multi_gpu[gpu_i],
                    cut_left=cut_left,
                    cut_right=cut_right,
                )
        hist = self.profile_multi_gpu[0].get()
        if len(self.profile_multi_gpu) > 1:
            for hist_tmp in self.profile_multi_gpu.values():
                hist[:] += hist_tmp[:].get()
        if hasattr(profile, "get"):
            profile[:] = cp.array(hist)[:]
        else:
            profile[:] = hist[:]

    def kick(
        self,
        rf_station: RFStation,
        acceleration_kicks: NumpyNDArray | CupyNDArray,
        turn_i: int,
    ):
        # send them to the specific GPU during execution inside _kick_helper
        voltage = rf_station.voltage[:, turn_i]
        omega_rf = rf_station.omega_rf[:, turn_i]
        phi_rf = rf_station.phi_rf[:, turn_i]

        self.map_no_result(
            _kick_helper,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=rf_station.particle.charge,
            n_rf=rf_station.n_rf,
            acceleration_kick=acceleration_kicks[turn_i],
        )

    def drift(self, rf_station: RFStation, solver: SolverTypes, turn_i: int):
        self.map_no_result(
            _drift_helper,
            solver=solver,
            t_rev=float(rf_station.t_rev[turn_i]),
            length_ratio=float(rf_station.length_ratio),
            alpha_order=float(rf_station.alpha_order),
            eta_0=float(rf_station.eta_0[turn_i]),
            eta_1=float(rf_station.eta_1[turn_i]),
            eta_2=float(rf_station.eta_2[turn_i]),
            alpha_0=float(rf_station.alpha_0[turn_i]),
            alpha_1=float(rf_station.alpha_1[turn_i]),
            alpha_2=float(rf_station.alpha_2[turn_i]),
            beta=float(rf_station.beta[turn_i]),
            energy=float(rf_station.energy[turn_i]),
        )

    def linear_interp_kick(
        self,
        voltage: NumpyNDArray | CupyNDArray,
        bin_centers: NumpyNDArray | CupyNDArray,
        charge: float,
        acceleration_kick: float,
    ):
        self.map_no_result(
            _linear_interp_kick_helper,
            voltage=voltage,  # will get uploaded to each device in helper
            bin_centers=bin_centers,  # will get uploaded to each device in helper
            charge=charge,
            acceleration_kick=acceleration_kick,
        )

    def kickdrift_considering_periodicity(
        self,
        acceleration_kicks: NumpyNDArray | CupyNDArray,
        rf_station: RFStation,
        solver: SolverTypes,
        turn_i: int,
    ):
        self.map_no_result(
            _kickdrift_considering_periodicity_helper,
            acceleration_kick=precision.real_t(acceleration_kicks[turn_i]),
            rf_station=rf_station,
            solver=solver,
            turn=turn_i,
        )
