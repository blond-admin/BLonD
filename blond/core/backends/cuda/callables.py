# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `CudaSpecials` and helper functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np

from blond.core.backends.backend import Specials, backend
from blond.generals.hashing_ import hash_in_folder

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore

_filepath = os.path.realpath(__file__)
_compute_capability = cp.cuda.Device(0).compute_capability


folder = os.path.dirname(os.path.abspath(__file__))

hash_ = hash_in_folder(
    folder=folder,
    extensions=(".py", ".cu"),
    recursive=False,
)
_basepath = os.path.join(folder, "compiled", hash_)


def reload_cuda_backend(  # NOQA: D102
    floattype: type[np.float32 | np.float64],
) -> CudaSpecials:
    """Load and link the according CUDA backend.

    Parameters
    ----------
    floattype
        Float type to compile the backend for.
        32 or 64 bit.

    Returns
    -------
    CudaSpecials
        The `CudaSpecials` class.

    """
    if floattype == np.float32:
        path = os.path.join(
            _basepath,
            f"kernels_sm_{_compute_capability}_single.cubin",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"The compiled CUDA backend was notfound at {path=}"
            )
        gpu_module = cp.RawModule(
            path=path,
        )
    elif floattype == np.float64:
        path = os.path.join(
            _basepath,
            f"kernels_sm_{_compute_capability}_double.cubin",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"The compiled CUDA backend was notfound at {path=}"
            )
        gpu_module = cp.RawModule(
            path=path,
        )
    else:
        raise TypeError(floattype)

    _drift_simple = gpu_module.get_function("drift_simple")
    _beam_phase = gpu_module.get_function("beam_phase")
    _kick_multi_harmonic = gpu_module.get_function("kick_multi_harmonic")
    _kick_single_harmonic = gpu_module.get_function("kick_single_harmonic")
    _sm_histogram = gpu_module.get_function("sm_histogram")
    _hybrid_histogram = gpu_module.get_function("hybrid_histogram")
    _gm_linear_interp_kick_help = gpu_module.get_function("lik_only_gm_copy")
    _gm_linear_interp_kick_comp = gpu_module.get_function("lik_only_gm_comp")
    _loss_box = gpu_module.get_function("loss_box")

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
        def loss_box(
            e_max: float,
            e_min: float,
            t_min: float,
            t_max: float,
            dt: CupyArray,
            dE: CupyArray,
            flags: CupyArray,
        ) -> None:
            assert dt.dtype == backend.float
            assert dE.dtype == backend.float
            assert dE.dtype == backend.float
            assert isinstance(e_max, backend.float)
            assert isinstance(e_min, backend.float)
            assert isinstance(t_min, backend.float)
            assert isinstance(t_max, backend.float)

            _loss_box(
                args=(
                    e_max,
                    e_min,
                    t_min,
                    t_max,
                    dt,
                    dE,
                    flags,
                    np.int32(len(dE)),  # n_macroparticles
                ),
                block=block_size,
                grid=grid_size,
            )

        @staticmethod
        def kick_single_harmonic(
            dt: CupyArray | CupyArray,
            dE: CupyArray | CupyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous

            _kick_single_harmonic(
                args=(
                    dt,  # beam_dt
                    dE,  # beam_dE
                    floattype(charge),  # charge
                    floattype(voltage),  # voltage
                    floattype(omega_rf),  # omega_RF
                    floattype(phi_rf),  # phi_RF
                    np.int32(len(dE)),  # n_macroparticles
                    floattype(acceleration_kick),  # acc_kick
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
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert phi_rf.dtype == floattype
            assert voltage.dtype == floattype
            assert omega_rf.dtype == floattype

            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert omega_rf.flags.c_contiguous
            assert phi_rf.flags.c_contiguous

            _kick_multi_harmonic(
                args=(
                    dt,  # beam_dt
                    dE,  # beam_dE
                    np.int32(len(voltage)),  # n_rf
                    floattype(charge),  # charge
                    voltage,  # voltage
                    omega_rf,  # omega_RF
                    phi_rf,  # phi_RF
                    np.int32(len(dE)),  # n_macroparticles
                    floattype(acceleration_kick),  # acc_kick
                ),
                block=block_size,
                grid=grid_size,
            )

        @staticmethod
        def drift_simple(
            dt: CupyArray,
            dE: CupyArray,
            T: float,
            eta_0: float,
            beta: float,
            energy: float,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous

            # Cast Python floats to backend floattype
            T = floattype(T)
            eta_0 = floattype(eta_0)
            beta = floattype(beta)
            energy = floattype(energy)

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
            charge: float,
            acceleration_kick: float,
        ) -> None:
            assert dt.dtype == floattype
            assert dE.dtype == floattype
            assert voltage.dtype == floattype
            assert bin_centers.dtype == floattype
            assert dt.flags.c_contiguous
            assert dE.flags.c_contiguous
            assert voltage.flags.c_contiguous
            assert bin_centers.flags.c_contiguous

            # Cast Python floats to backend floattype
            charge = floattype(charge)
            acceleration_kick = floattype(acceleration_kick)

            glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), floattype)
            _gm_linear_interp_kick_help(
                args=(
                    dt,
                    dE,
                    voltage,
                    bin_centers,
                    charge,
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
                    floattype(charge),
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
            start: float,
            stop: float,
        ) -> None:
            assert array_read.dtype == floattype
            assert array_write.dtype == floattype
            assert array_read.flags.c_contiguous
            assert array_write.flags.c_contiguous

            # Cast Python floats to backend floattype
            start = floattype(start)
            stop = floattype(stop)

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
        ) -> float:
            assert hist_x.dtype == floattype
            assert hist_y.dtype == floattype
            assert hist_x.flags.c_contiguous
            assert hist_y.flags.c_contiguous

            # Cast Python floats to backend floattype
            alpha = floattype(alpha)
            omega_rf = floattype(omega_rf)
            phi_rf = floattype(phi_rf)
            bin_size = floattype(bin_size)

            result = cp.zeros(2, dtype=floattype)
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
                shared_mem=2 * block_size[0] * np.dtype(floattype).itemsize,
            )
            return floattype(result[0].get() / result[1].get())

        @staticmethod
        def move_flagged_elements_to_end(
            flag: int,
            flags: CupyArray,
            dt: CupyArray,
            dE: CupyArray,
            ids: CupyArray,
        ):
            # TODO write a kernel that works with gpu kernels
            #  to have a smaller memory footprint.
            flag = np.int32(flag)
            assert flags.dtype == np.int32
            assert dt.dtype == backend.float
            assert dE.dtype == backend.float
            assert ids.dtype == np.int32

            select = flags == flag
            order = cp.argsort(select)

            flags[:] = flags[order]
            dt[:] = dt[order]
            dE[:] = dE[order]
            ids[:] = ids[order]

            n_new = len(ids) - cp.sum(select)
            return n_new

    return CudaSpecials


CudaSpecials = reload_cuda_backend(backend.float)
