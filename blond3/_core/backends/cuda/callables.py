import os
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupy.typing import NDArray as CupyArray
from numpy._typing import NDArray as NumpyArray

from blond3._core.backends.backend import Specials
from blond3._core.backends.backend import backend
from blond3.handle_results.helpers import callers_relative_path

if TYPE_CHECKING:  # pragma: no cover
    pass
if backend.float == np.float32:
    gpu_module = cp.RawModule(
        path=callers_relative_path(
            "kernels_sm_75_single.cubin",
            stacklevel=1,
        )
    )
elif backend.float == np.float64:
    gpu_module = cp.RawModule(
        path=callers_relative_path(
            "kernels_sm_75_double.cubin",
            stacklevel=1,
        )
    )
else:
    raise TypeError(backend.float)

_drift_simple = gpu_module.get_function("drift_simple")
_beam_phase = gpu_module.get_function("beam_phase")
_kick_multi_harmonic = gpu_module.get_function("kick_multi_harmonic")
_kick_single_harmonic = gpu_module.get_function("kick_single_harmonic")

default_blocks = 2 * cp.cuda.Device(0).attributes["MultiProcessorCount"]
default_threads = cp.cuda.Device(0).attributes["MaxThreadsPerBlock"]
max_shared_memory_per_block = cp.cuda.Device(0).attributes["MaxSharedMemoryPerBlock"]
blocks = int(os.environ.get("GPU_BLOCKS", default_blocks))
threads = int(os.environ.get("GPU_THREADS", default_threads))
grid_size = (blocks, 1, 1)
block_size = (threads, 1, 1)


class CudaSpecials(Specials):
    @staticmethod
    def loss_box(self, a, b, c, d) -> None:
        raise NotImplementedError()

    @staticmethod
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ):
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
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ):
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
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ):
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
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_order: int,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ):
        raise NotImplementedError()

    @staticmethod
    def drift_exact(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        beta: float,
        energy: float,
    ):
        raise NotImplementedError()

    @staticmethod
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ):
        raise NotImplementedError()

    @staticmethod
    def histogram(
        array_read: NumpyArray, array_write: NumpyArray, start: float, stop: float
    ):
        raise NotImplementedError()

    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
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
        return backend.float(result[0]) / backend.float(result[1])
