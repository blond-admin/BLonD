from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cupy as cp
import numpy as np

from .beam_abstract import BeamBaseClass
from ..gpu.butils_wrap_cupy import kick, drift, slice_beam
from ..input_parameters.rf_parameters import RFStation
from ..input_parameters.ring import Ring
from ..trackers.utilities import is_in_separatrix

if TYPE_CHECKING:
    from cupy.typing import NDArray as CupyNDArray
    from typing import Callable, Dict
    from numpy.typing import NDArray


def get_device(gpu_i: int):
    """Get the device of the selected GPU"""
    max_n_gpus = cp.cuda.runtime.getDeviceCount()

    if max_n_gpus == 1:
        device_i = cp.cuda.Stream(non_blocking=True)
    else:
        device_i = cp.cuda.Device(gpu_i)
    return device_i


class DistributedMultiGpuArray:
    # DEVELOPER NOTE:
    # At the time of writing (Q1 2025),
    # cupyx.distributed.array.DistributedArray
    # exists already, but most features are "not supported"
    # for this reason it is not used for now

    def __init__(
        self, array_cpu: NDArray, *, axis=0, mock_n_gpus: Optional[int] = None
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
            n_gpus = cp.cuda.runtime.getDeviceCount()

        self.gpu_arrays: Dict[int, CupyNDArray] = {}
        sub_arrays = np.array_split(array_cpu, n_gpus, axis=axis)
        for gpu_i, array_tmp in enumerate(sub_arrays):
            with get_device(gpu_i=gpu_i):
                # upload to GPU
                self.gpu_arrays[gpu_i] = cp.array(
                    array_tmp, dtype=array_cpu.dtype
                )
        print(f"{len(self.gpu_arrays)=}")

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

        for gpu_i, array in self.gpu_arrays.items():
            with get_device(gpu_i=gpu_i):
                results.append(
                    func(array, **kwargs),
                )

        return results

    def min(self):
        """Minimum of entire array"""
        return np.min(self.map(lambda x: float(cp.min(x))))

    def max(self):
        """Maximum of entire array"""
        return np.max(self.map(lambda x: float(cp.max(x))))

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
        id
            Index of the particle (might change during execution)
        mock_n_gpus
            Pretend to have n_gpus when the system has only one.
            This should be used for testing only.
        """
        assert len(dE) == len(dt), f"{len(dE)=}, but {len(dt)=}"
        assert len(dE) == len(id), f"{len(dE)=}, but {len(id)=}"

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
            id, mock_n_gpus=mock_n_gpus
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
            results.append(
                func(
                    dt_gpu_i=self.dt_multi_gpu.gpu_arrays[gpu_i],
                    dE_gpu_i=self.dE_multi_gpu.gpu_arrays[gpu_i],
                    id_gpu_i=self.id_multi_gpu.gpu_arrays[gpu_i],
                    **kwargs,
                ),
            )
        cp.cuda.runtime.deviceSynchronize()
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
            return cp.cuda.runtime.getDeviceCount()

    @property
    def n_macroparticles_alive(self) -> int:
        """Number of macro-particles marked as alive (id ≠ 0)"""
        counts = self.id_multi_gpu.map(lambda x: int(cp.count_nonzero(x)))
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
                select_alive: NDArray = id_tmp[:] != 0  # noqa
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
        self.map(
            _losses_separatrix_helper,
            beam=self,
            ring=ring,
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
        self.map(_losses_longitudinal_cut_helper, dt_min=dt_min, dt_max=dt_max)

    def losses_energy_cut(self, dE_min: float, dE_max: float) -> None:
        """Mark particles outside energy range as not-alive (id=0)

        Parameters
        ----------
        dE_min : float
            Lower limit (dE=dE_min is kept)
        dE_max : float
            Upper limit (dE=dE_max is kept)
        """
        self.map(_losses_energy_cut_helper, dE_min=dE_min, dE_max=dE_max)

    def losses_below_energy(self, dE_min: float):
        """Mark particles outside energy range as not-alive (id=0)

        Parameters
        ----------
        dE_min : float
            Lower limit (dE=dE_min is kept)
        """
        self.map(_losses_below_energy_helper, dE_min=dE_min)

    def dE_mean(self, ignore_id_0: bool = False):
        """Calculate mean of energy

        Parameters
        ----------
        ignore_id_0
            If True, particles with id = 0 are ignored
        """
        if ignore_id_0:
            means = self.map(_dE_mean_helper_ignore_id_0)
        else:
            means = self.dE_multi_gpu.map(_dE_mean_helper)

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
            sums = self.map(_dE_std_helper_ignore_id_0, mean=mean)
            N = self.n_macroparticles_alive
        else:
            sums = self.dE_multi_gpu.map(_dE_std_helper, mean=mean)
            N = self.n_macroparticles

        sums = np.array(sums)
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
            means = self.map(_dt_mean_helper_ignore_id_0)
        else:
            means = self.dt_multi_gpu.map(_dt_mean_helper)

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
            sums = self.map(_dt_std_helper_ignore_id_0, mean=mean)
            N = self.n_macroparticles_alive
        else:
            sums = self.dt_multi_gpu.map(_dt_std_helper, mean=mean)
            N = self.n_macroparticles

        sums = np.array(sums)
        return np.sqrt(np.nansum(sums) / N)

    def histogram(self, out, cut_left, cut_right):  # todo rewrite using bmath
        """Computes a histogram of the dt coordinates"""
        histograms = self.dt_multi_gpu.map(
            slice_beam_helper,
            n_bins=len(out),
            cut_left=cut_left,
            cut_right=cut_right,
        )
        hist = histograms[0]
        if len(histograms) > 1:
            for hist_tmp in histograms[1:]:
                hist[:] += hist_tmp[:]

        return cp.array(hist)

    def kick(
        self,
        voltage: NDArray,
        omega_rf: NDArray,
        phi_rf: NDArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ):
        # accept only CPU arrays,
        # send them to the specific GPU during execution
        assert not hasattr(voltage, "get")
        assert not hasattr(omega_rf, "get")
        assert not hasattr(phi_rf, "get")
        self.map(
            kick_helper,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            n_rf=n_rf,
            acceleration_kick=acceleration_kick,
        )

    def drift(
        self,
        solver,
        t_rev,
        length_ratio,
        alpha_order,
        eta_0,
        eta_1,
        eta_2,
        alpha_0,
        alpha_1,
        alpha_2,
        beta,
        energy,
    ):
        self.map(
            drift_helper,
            solver=solver,
            t_rev=t_rev,
            length_ratio=length_ratio,
            alpha_order=alpha_order,
            eta_0=eta_0,
            eta_1=eta_1,
            eta_2=eta_2,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            beta=beta,
            energy=energy,
        )


def slice_beam_helper(
    dt_gpu_i,
    n_bins,
    cut_left,
    cut_right,
):
    from ..utils import precision

    profile = cp.zeros(n_bins, dtype=precision.real_t)
    slice_beam(
        dt=dt_gpu_i, profile=profile, cut_left=cut_left, cut_right=cut_right
    )
    return profile.get()


def kick_helper(
    dt_gpu_i,
    dE_gpu_i,
    id_gpu_i,
    voltage,
    omega_rf,
    phi_rf,
    charge,
    n_rf,
    acceleration_kick,
):
    kick(
        dt=dt_gpu_i,
        dE=dE_gpu_i,
        voltage=cp.array(voltage),  # so that voltage is on each device
        omega_rf=cp.array(omega_rf),  # so that voltage is on each device
        phi_rf=cp.array(phi_rf),  # so that voltage is on each device
        charge=charge,
        n_rf=n_rf,
        acceleration_kick=acceleration_kick,
    )


def drift_helper(
    dt_gpu_i,
    dE_gpu_i,
    id_gpu_i,
    solver,
    t_rev,
    length_ratio,
    alpha_order,
    eta_0,
    eta_1,
    eta_2,
    alpha_0,
    alpha_1,
    alpha_2,
    beta,
    energy,
):
    drift(
        dt=dt_gpu_i,
        dE=dE_gpu_i,
        solver=solver,
        t_rev=t_rev,
        length_ratio=length_ratio,
        alpha_order=alpha_order,
        eta_0=eta_0,
        eta_1=eta_1,
        eta_2=eta_2,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        beta=beta,
        energy=energy,
    )


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
        return np.nan
    else:
        return float(cp.mean(masked))


def _dE_std_helper(dE_gpu_i, mean):
    tmp = dE_gpu_i - mean
    return float(cp.sum(tmp * tmp))


def _dE_std_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i, mean):
    mask = id_gpu_i > 0
    if not cp.any(mask):
        return np.nan
    else:
        tmp = dE_gpu_i[mask] - mean
        return float(cp.sum(tmp * tmp))


def _dt_mean_helper(dt_gpu_i):
    return float(cp.mean(dt_gpu_i))


def _dt_mean_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i):
    mask = id_gpu_i > 0
    masked = dt_gpu_i[mask]
    if len(masked) == 0:
        return np.nan
    else:
        return float(cp.mean(masked))


def _dt_std_helper(dt_gpu_i, mean):
    tmp = dt_gpu_i - mean
    return float(cp.sum(tmp * tmp))


def _dt_std_helper_ignore_id_0(dt_gpu_i, dE_gpu_i, id_gpu_i, mean):
    mask = id_gpu_i > 0
    if not cp.any(mask):
        return np.nan
    else:
        tmp = dt_gpu_i[mask] - mean
        return float(cp.sum(tmp * tmp))
