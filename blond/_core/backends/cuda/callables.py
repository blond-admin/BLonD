from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np

from blond._core.backends.backend import Specials, backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as CupyArray

_filepath = os.path.realpath(__file__)
_basepath = os.sep.join(_filepath.split(os.sep)[:-1])
_compute_capability = cp.cuda.Device(0).compute_capability

if backend.float == np.float32:
    path = os.path.join(
        _basepath,
        f"kernels_sm_{_compute_capability}_single.cubin",
    )
    assert os.path.isfile(path), f"{path=}"
    gpu_module = cp.RawModule(
        path=path,
    )
elif backend.float == np.float64:
    path = os.path.join(
        _basepath,
        f"kernels_sm_{_compute_capability}_double.cubin",
    )
    assert os.path.isfile(path), f"{path=}"
    gpu_module = cp.RawModule(
        path=path,
    )
else:
    raise TypeError(backend.float)

_drift_simple = gpu_module.get_function("drift_simple")
_beam_phase = gpu_module.get_function("beam_phase")
_kick_multi_harmonic = gpu_module.get_function("kick_multi_harmonic")
_kick_single_harmonic = gpu_module.get_function("kick_single_harmonic")
_sm_histogram = gpu_module.get_function("sm_histogram")
_hybrid_histogram = gpu_module.get_function("hybrid_histogram")
_gm_linear_interp_kick_help = gpu_module.get_function("lik_only_gm_copy")
_gm_linear_interp_kick_comp = gpu_module.get_function("lik_only_gm_comp")

default_blocks = 2 * cp.cuda.Device(0).attributes["MultiProcessorCount"]
default_threads = cp.cuda.Device(0).attributes["MaxThreadsPerBlock"]
max_shared_memory_per_block = cp.cuda.Device(0).attributes[
    "MaxSharedMemoryPerBlock"
]
blocks = int(os.environ.get("GPU_BLOCKS", default_blocks))
threads = int(os.environ.get("GPU_THREADS", default_threads))
grid_size = (blocks, 1, 1)
block_size = (threads, 1, 1)


class CudaSpecials(Specials):
    @staticmethod
    def loss_box(top: float, bottom: float, left: float, right: float) -> None:
        raise NotImplementedError()

    @staticmethod
    def kick_single_harmonic(
        dt: CupyArray | CupyArray,
        dE: CupyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: np.flaot32 | np.float64,
        acceleration_kick: np.flaot32 | np.float64,
    ) -> None:
        assert dt.dtype == backend.float
        assert dE.dtype == backend.float
        assert isinstance(charge, backend.float)
        assert isinstance(voltage, backend.float)
        assert isinstance(omega_rf, backend.float)
        assert isinstance(phi_rf, backend.float)
        assert isinstance(acceleration_kick, backend.float)

        _kick_single_harmonic(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                charge,  # charge
                voltage,  # voltage
                omega_rf,  # omega_RF
                phi_rf,  # phi_RF
                np.int32(len(dE)),  # n_macroparticles
                acceleration_kick,  # acc_kick
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def kick_multi_harmonic(
        dt: CupyArray | CupyArray,
        dE: CupyArray | CupyArray,
        voltage: CupyArray,
        omega_rf: CupyArray,
        phi_rf: CupyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        """assert dt.dtype == backend.float
        assert dE.dtype == backend.float
        assert phi_rf.dtype == backend.float
        assert voltage.dtype == backend.float
        assert omega_rf.dtype == backend.float
        assert isinstance(charge, backend.float)
        assert isinstance(acceleration_kick, backend.float)"""

        _kick_multi_harmonic(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                np.int32(len(voltage)),  # n_rf
                charge,  # charge
                voltage,  # voltage
                omega_rf,  # omega_RF
                phi_rf,  # phi_RF
                np.int32(len(dE)),  # n_macroparticles
                acceleration_kick,  # acc_kick
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def drift_simple(
        dt: CupyArray,
        dE: CupyArray,
        T: np.float32 | np.float64,
        eta_0: np.float32 | np.float64,
        beta: np.float32 | np.float64,
        energy: np.float32 | np.float64,
    ) -> None:
        assert dt.dtype == backend.float
        assert dE.dtype == backend.float
        assert isinstance(T, backend.float)
        assert isinstance(eta_0, backend.float)
        assert isinstance(beta, backend.float)
        assert isinstance(energy, backend.float)
        _drift_simple(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                T,  # T
                eta_0,  # eta_zero
                beta,  # beta
                energy,  # energy
                np.int32(len(dE)),  # n_macroparticles
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def drift_legacy(
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        alpha_order: int,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def drift_exact(
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def kick_induced_voltage(
        dt: CupyArray,
        dE: CupyArray,
        voltage: CupyArray,
        bin_centers: CupyArray,
        charge: np.flaot32 | np.float64,
        acceleration_kick: np.flaot32 | np.float64,
    ) -> None:
        assert dt.dtype == backend.float
        assert dE.dtype == backend.float
        assert voltage.dtype == backend.float
        assert bin_centers.dtype == backend.float
        assert isinstance(charge, backend.float)
        assert isinstance(acceleration_kick, backend.float)

        glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), backend.float)
        _gm_linear_interp_kick_help(
            args=(
                dt,
                dE,
                voltage,
                bin_centers,
                backend.float(charge),
                np.int32(bin_centers.size),
                np.int32(dt.size),
                acceleration_kick,
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )

        _gm_linear_interp_kick_comp(
            args=(
                dt,
                dE,
                voltage,
                bin_centers,
                backend.float(charge),
                np.int32(bin_centers.size),
                np.int32(dt.size),
                acceleration_kick,
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )

    @staticmethod
    def histogram(
        array_read: CupyArray,
        array_write: CupyArray,
        start: np.float32 | np.float64,
        stop: np.float32 | np.float64,
    ) -> None:
        assert array_read.dtype == backend.float
        assert array_write.dtype == backend.float
        assert isinstance(start, backend.float)
        assert isinstance(stop, backend.float)

        n_slices = array_write.size
        array_write.fill(0)

        if 4 * n_slices < max_shared_memory_per_block:
            _sm_histogram(
                args=(
                    array_read,
                    array_write,
                    start,
                    stop,
                    np.uint32(n_slices),
                    np.uint32(len(array_read)),
                ),
                grid=grid_size,
                block=block_size,
                shared_mem=4 * n_slices,
            )
        else:
            _hybrid_histogram(
                args=(
                    array_read,
                    array_write,
                    start,
                    stop,
                    np.uint32(n_slices),
                    np.uint32(len(array_read)),
                    np.int32(max_shared_memory_per_block / 4),
                ),
                grid=grid_size,
                block=block_size,
                shared_mem=max_shared_memory_per_block,
            )

    @staticmethod
    def beam_phase(
        hist_x: CupyArray,
        hist_y: CupyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> np.float32 | np.float64:
        """assert hist_x.dtype == backend.float
        assert hist_y.dtype == backend.float
        assert isinstance(alpha, backend.float), type(alpha)
        assert isinstance(omega_rf, backend.float), type(alpha)
        assert isinstance(phi_rf, backend.float), type(alpha)
        assert isinstance(bin_size, backend.float), type(alpha)"""
        result = cp.zeros(2, dtype=backend.float)
        _beam_phase(
            args=(
                hist_x,  # hist_x
                hist_y,  # hist_y
                result,  # result
                alpha,  # alpha
                omega_rf,  # omega_rf
                phi_rf,  # phi_rf
                bin_size,  # bin_size
                np.int32(len(hist_x)),  # n_bins
            ),
            block=block_size,
            grid=grid_size,
            shared_mem=2 * block_size[0] * np.dtype(backend.float).itemsize,
        )
        return backend.float(result[0].get() / result[1].get())
