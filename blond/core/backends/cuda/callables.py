# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `CudaSpecials` and helper functions."""

from __future__ import annotations

import itertools
import os
import time
from typing import TYPE_CHECKING

import cupy as cp  # type: ignore
import numpy as np

from blond.core.backends.backend import Specials
from blond.core.backends.cuda.compiled_dir_handler import cuda_compiled_dir
from blond.generals.compiled_cache import mark_used

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore

_filepath = os.path.realpath(__file__)
_compute_capability = cp.cuda.Device(0).compute_capability

FLOAT = np.float64

folder = os.path.dirname(os.path.abspath(__file__))

# Same toolchain-aware directory the compiler writes to.
_basepath = str(cuda_compiled_dir(folder))


path = os.path.join(
    _basepath,
    f"kernels_sm_{_compute_capability}_double.cubin",
)
if not os.path.isfile(path):
    from blond.core.backends.cuda.compile import compile_cuda_library

    print("CUDA backend was not found.. Trying to compile.")
    compile_cuda_library()

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"The compiled CUDA backend was not found at {path=}.\n"
            f"Has the backend been compiled?"
            f"{__file__.replace('callables.py', 'compile.py')}:1"  # :1 to
            # make PyCharm automatically link the correct file
        )
gpu_module = cp.RawModule(
    path=path,
)
# Refresh the LRU stamp on the cache dir we loaded from.
mark_used(_basepath)

_drift_simple = gpu_module.get_function("drift_simple")
_drift_exact = gpu_module.get_function("drift_exact")
_beam_phase = gpu_module.get_function("beam_phase")
_kick_multi_harmonic = gpu_module.get_function("kick_multi_harmonic")
_kick_single_harmonic = gpu_module.get_function("kick_single_harmonic")
_sm_histogram = gpu_module.get_function("sm_histogram")
_hybrid_histogram = gpu_module.get_function("hybrid_histogram")
_gm_linear_interp_kick_help = gpu_module.get_function("lik_only_gm_copy")
_gm_linear_interp_kick_comp = gpu_module.get_function("lik_only_gm_comp")
_loss_box = gpu_module.get_function("loss_box")
_histogram_sparse = gpu_module.get_function("histogram_sparse")
_wake_from_pole_residue = gpu_module.get_function("wake_from_pole_residue")
_apply_sr_without_quantum_excitation = gpu_module.get_function(
    "apply_sr_without_quantum_excitation"
)
_apply_sr_with_quantum_excitation = gpu_module.get_function(
    "apply_sr_with_quantum_excitation"
)

default_blocks = 2 * cp.cuda.Device(0).attributes["MultiProcessorCount"]
default_threads = cp.cuda.Device(0).attributes["MaxThreadsPerBlock"]
max_shared_memory_per_block = cp.cuda.Device(0).attributes[
    "MaxSharedMemoryPerBlock"
]
blocks = int(os.environ.get("GPU_BLOCKS", default_blocks))
threads = int(os.environ.get("GPU_THREADS", default_threads))
grid_size = (blocks, 1, 1)
block_size = (threads, 1, 1)
_quantum_excitation_seed_counter = itertools.count(time.time_ns())


class CudaSpecials(Specials):  # NOQA: D101
    @staticmethod
    def get_max_threads() -> int:  # NOQA: D102
        """
        Return the max number of threads this backend's kernels may use.

        Returns
        -------
        max_threads
            Maximum number of threads this backend's kernels may use.
        """
        return 1

    @staticmethod
    def loss_box(  # NOQA: D102
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: CupyArray,
        dE: CupyArray,
        flags: CupyArray,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert flags.device != "cpu", (
            f"Requires Cupy array, but got {type(flags)}."
        )

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert flags.dtype == np.int32

        assert isinstance(e_max, FLOAT)
        assert isinstance(e_min, FLOAT)
        assert isinstance(t_min, FLOAT)
        assert isinstance(t_max, FLOAT)

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
    def kick_single_harmonic(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        _kick_single_harmonic(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                FLOAT(charge),  # charge
                FLOAT(voltage),  # voltage
                FLOAT(omega_rf),  # omega_RF
                FLOAT(phi_rf),  # phi_RF
                np.int32(len(dE)),  # n_macroparticles
                FLOAT(acceleration_kick),  # acc_kick
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def kick_multi_harmonic(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: CupyArray,
        omega_rf: CupyArray,
        phi_rf: CupyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert phi_rf.device != "cpu", (
            f"Requires Cupy array, but got {type(phi_rf)}."
        )
        assert voltage.device != "cpu", (
            f"Requires Cupy array, but got {type(voltage)}."
        )
        assert omega_rf.device != "cpu", (
            f"Requires Cupy array, but got {type(omega_rf)}."
        )

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert phi_rf.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert omega_rf.dtype == FLOAT

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
                FLOAT(charge),  # charge
                voltage,  # voltage
                omega_rf,  # omega_RF
                phi_rf,  # phi_RF
                np.int32(len(dE)),  # n_macroparticles
                FLOAT(acceleration_kick),  # acc_kick
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def sum_1d_array(array: CupyArray) -> float:
        """Return the sum of 1d array."""
        assert array.device != "cpu", (
            f"Requires Cupy array, but got {type(array)}."
        )
        return cp.sum(array)

    @staticmethod
    def dot_product_1d_array(array_1: CupyArray, array_2: CupyArray):  # NOQA: D102
        assert array_1.device != "cpu", (
            f"Requires Cupy array, but got {type(array_1)}."
        )
        assert array_2.device != "cpu", (
            f"Requires Cupy array, but got {type(array_2)}."
        )

        """Return the sum of dot product of two 1d arrays."""
        return cp.dot(array_1, array_2)

    @staticmethod
    def drift_simple(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous

        # Cast Python floats to backend floattype
        T = FLOAT(T)
        eta_0 = FLOAT(eta_0)
        beta = FLOAT(beta)
        energy = FLOAT(energy)

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
    def drift_exact(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        T: float,
        alpha_0: float,
        higher_alpha: CupyArray,
        beta: float,
        energy: float,
    ) -> None:
        assert dt.device != "cpu"
        assert dE.device != "cpu"
        assert higher_alpha.device != "cpu"

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert higher_alpha.dtype == FLOAT

        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert higher_alpha.flags.c_contiguous

        T = FLOAT(T)
        alpha_0 = FLOAT(alpha_0)
        beta = FLOAT(beta)
        energy = FLOAT(energy)

        _drift_exact(
            args=(
                dt,  # beam_dt
                dE,  # beam_dE
                T,  # t_rev
                alpha_0,  # alpha_zero
                higher_alpha,  # higher_alpha
                np.int32(len(higher_alpha)),  # n_alpha
                beta,  # beta
                energy,  # energy
                np.int32(len(dE)),  # n_macroparticles
            ),
            block=block_size,
            grid=grid_size,
        )
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."

    @staticmethod
    def kick_interpolated(  # NOQA: D102
        dt: CupyArray,
        dE: CupyArray,
        voltage: CupyArray,
        bin_centers: CupyArray,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert voltage.device != "cpu", (
            f"Requires Cupy array, but got {type(voltage)}."
        )
        assert bin_centers.device != "cpu", (
            f"Requires Cupy array, but got {type(bin_centers)}."
        )

        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert bin_centers.dtype == FLOAT
        assert dt.flags.c_contiguous
        assert dE.flags.c_contiguous
        assert voltage.flags.c_contiguous
        assert bin_centers.flags.c_contiguous

        n_slices = bin_centers.size
        if n_slices >= 2:  # noqa: PLR2004
            diffs = cp.diff(bin_centers)
            if not cp.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                raise ValueError(
                    "bin_centers is not uniformly spaced (looks like a "
                    "sparse/multi-island EquidistantMultiProfile.hist_x). "
                    "Either pass this profile's sparse metadata "
                    "(first_left_cut, left_cut_distance, cut_width, "
                    "bins_per_profile, filling_pattern, "
                    "bucket_index_to_memory_index), e.g. via "
                    "`profile.sparse_kick_metadata`, or use "
                    "EquidistantMultiProfile.profiles[i].hist_x for a "
                    "single bucket."
                )

        # Cast Python floats to backend floattype
        charge = FLOAT(charge)
        acceleration_kick = FLOAT(acceleration_kick)

        glob_vkick_factor = cp.empty(2 * (bin_centers.size - 1), FLOAT)
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
                FLOAT(charge),
                np.int32(bin_centers.size),
                np.int32(dt.size),
                acceleration_kick,
                glob_vkick_factor,
            ),
            grid=grid_size,
            block=block_size,
        )

    @staticmethod
    def histogram(  # NOQA: D102
        array_read: CupyArray,
        array_write: CupyArray,
        start: float,
        stop: float,
    ) -> None:
        assert array_read.device != "cpu", (
            f"Requires Cupy array, but got {type(array_read)}."
        )
        assert array_write.device != "cpu", (
            f"Requires Cupy array, but got {type(array_write)}."
        )

        assert array_read.dtype == FLOAT
        assert array_write.dtype == FLOAT
        assert array_read.flags.c_contiguous
        assert array_write.flags.c_contiguous

        # Cast Python floats to backend floattype
        start = FLOAT(start)
        stop = FLOAT(stop)

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
    def beam_phase(  # NOQA: D102
        hist_x: CupyArray,
        hist_y: CupyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        assert hist_x.device != "cpu", (
            f"Requires Cupy array, but got {type(hist_x)}."
        )
        assert hist_y.device != "cpu", (
            f"Requires Cupy array, but got {type(hist_y)}."
        )

        assert hist_x.dtype == FLOAT
        assert hist_y.dtype == FLOAT
        assert hist_x.flags.c_contiguous
        assert hist_y.flags.c_contiguous

        # Cast Python floats to backend floattype
        alpha = FLOAT(alpha)
        omega_rf = FLOAT(omega_rf)
        phi_rf = FLOAT(phi_rf)
        bin_size = FLOAT(bin_size)

        result = cp.zeros(2, dtype=FLOAT)
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
            shared_mem=2 * block_size[0] * np.dtype(FLOAT).itemsize,
        )
        return FLOAT(result[0].get() / result[1].get())

    @staticmethod
    def apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
        beam_dE: CupyArray,
        energy_lost: float,
        longitudinal_damping_time: float,
        natural_energy_spread: float,
        total_energy: float,
        disable_quantum_excitation: bool = False,
    ) -> None:
        """
        Apply synchrotron radiation and quantum excitation energy kicks.

        Single fused CUDA kernel — one launch, one pass over ``beam_dE``,
        no auxiliary noise buffer. The Gaussian noise is drawn with
        NVIDIA's cuRAND device library (one state per thread), so the
        memory footprint is just the beam itself.

        Parameters
        ----------
        beam_dE
            Macro-particle energy coordinates, in [eV]. CuPy array,
            modified in place.
        energy_lost
            Energy lost through the considered synchrotron segment,
            in [eV per turn].
        longitudinal_damping_time
            Longitudinal damping time, in [turn].
        natural_energy_spread
            Natural energy spread, [dimensionless].
        total_energy
            Beam total reference energy, in [eV].
        disable_quantum_excitation
           Disables the quantum excitation kick.
        """
        assert beam_dE.device != "cpu", (
            f"Requires Cupy array, but got {type(beam_dE)}."
        )
        assert beam_dE.dtype == FLOAT
        assert beam_dE.flags.c_contiguous

        damping_factor = FLOAT(1.0 - 2.0 / longitudinal_damping_time)
        energy_lost_typed = FLOAT(energy_lost)
        n_macroparticles = np.int32(len(beam_dE))
        if disable_quantum_excitation:
            _apply_sr_without_quantum_excitation(
                args=(
                    beam_dE,
                    damping_factor,
                    energy_lost_typed,
                    n_macroparticles,
                ),
                block=block_size,
                grid=grid_size,
            )
        else:
            noise_scale = FLOAT(
                2.0
                * natural_energy_spread
                / float(np.sqrt(longitudinal_damping_time))
                * total_energy
            )
            # The counter guarantees a distinct seed per launch even on
            # platforms where consecutive clock reads return the same value.
            # Each thread uses its tid as the cuRAND subsequence.
            base_seed = np.uint64(next(_quantum_excitation_seed_counter))
            _apply_sr_with_quantum_excitation(
                args=(
                    beam_dE,
                    damping_factor,
                    energy_lost_typed,
                    noise_scale,
                    base_seed,
                    n_macroparticles,
                ),
                block=block_size,
                grid=grid_size,
            )

    @staticmethod
    def move_flagged_elements_to_end(  # NOQA: D102
        flag: int,
        flags: CupyArray,
        dt: CupyArray,
        dE: CupyArray,
        ids: CupyArray,
    ):
        assert flags.device != "cpu", (
            f"Requires Cupy array, but got {type(flags)}."
        )
        assert dt.device != "cpu", f"Requires Cupy array, but got {type(dt)}."
        assert dE.device != "cpu", f"Requires Cupy array, but got {type(dE)}."
        assert ids.device != "cpu", (
            f"Requires Cupy array, but got {type(ids)}."
        )

        # TODO write a kernel that works with gpu kernels
        #  to have a smaller memory footprint.
        flag = np.int32(flag)
        assert flags.dtype == np.int32
        assert dt.dtype == FLOAT
        assert dE.dtype == FLOAT
        assert ids.dtype == np.int32

        select = flags == flag
        order = cp.argsort(select)

        flags[:] = flags[order]
        dt[:] = dt[order]
        dE[:] = dE[order]
        ids[:] = ids[order]

        n_new = len(ids) - cp.sum(select)
        return n_new

    @staticmethod
    def histogram_sparse(  # NOQA: D102
        x: CupyArray,
        out: CupyArray,
        first_left_cut: float,
        left_cut_distance: float,
        cut_width: float,
        bins_per_profile: int,
        n_active_profiles: int,
        filling_pattern: CupyArray,
        bucket_index_to_memory_index: CupyArray,
    ) -> None:
        assert x.device != "cpu", f"Requires Cupy array, but got {type(x)}."
        assert out.device != "cpu", (
            f"Requires Cupy array, but got {type(out)}."
        )
        assert filling_pattern.device != "cpu", (
            f"Requires Cupy array, but got {type(filling_pattern)}."
        )
        assert bucket_index_to_memory_index.device != "cpu", (
            f"Requires Cupy array, but got {type(bucket_index_to_memory_index)}."
        )

        assert x.dtype == FLOAT
        assert out.dtype == FLOAT
        assert filling_pattern.dtype == np.bool
        assert bucket_index_to_memory_index.dtype == np.int32

        assert x.flags.c_contiguous
        assert out.flags.c_contiguous
        assert filling_pattern.flags.c_contiguous
        assert bucket_index_to_memory_index.flags.c_contiguous

        out[:] = 0
        _histogram_sparse(
            args=(
                x,  # input
                out,  # output
                FLOAT(first_left_cut),  # first_left_cut
                FLOAT(left_cut_distance),  # left_cut_distance
                FLOAT(cut_width),  # cut_width
                np.int32(bins_per_profile),  # bins_per_profile
                np.int32(len(filling_pattern)),  # n_buckets
                np.int32(len(x)),  # n_macroparticles
                filling_pattern,  # input
                bucket_index_to_memory_index,  # input
            ),
            block=block_size,
            grid=grid_size,
        )

    @staticmethod
    def wake_from_pole_residue(
        # read
        profile: CupyArray,
        profile_dts: CupyArray,
        poles: CupyArray,
        residues: CupyArray,
        is_counterrotating_beam: bool,
        counterrotating_pole_signs: CupyArray,
        update_on_bin: CupyArray,
        factor: float,
        # write
        states: CupyArray,
        voltage: CupyArray,
        voltage_threaded: CupyArray,
    ) -> None:
        """
        Apply poles based on the `profile` to generate `voltage`.

        Parameters
        ----------
        profile
            Beam profile histogram.
        profile_dts
            Base for time step, connected to `update_on_bin`.
        poles
            Complex poles of an equivalent circuit model.
        residues
            Complex residues of an equivalent circuit model.
        is_counterrotating_beam
            If true, the current beam is counter-rotating.
        counterrotating_pole_signs
            Array per pole, -1 if the sign of the impedance is flipped
            for a counter-rotating beam.
        update_on_bin
            Index when to trigger an update of dt. For speedup.
            E.g. For profile no.: ``0,0,0,1,1,1,1,2,2,2``
            one needs ``update_on_bin = [0,3,7]``.
        factor
            To convert `profile` to current per bin [A].
        states
            Complex state vector, length ``n_poles + 1``.
            The last element stores ``t_start`` in its real part.
        voltage
            Output voltage, in [V].
        voltage_threaded
            Unused on the CUDA backend (kept for API parity with CPU
            backends); pole contributions are reduced into `voltage`
            directly via atomic adds.
        """
        assert profile.device != "cpu", (
            f"Requires Cupy array, but got {type(profile)}."
        )
        assert profile_dts.device != "cpu", (
            f"Requires Cupy array, but got {type(profile_dts)}."
        )
        assert poles.device != "cpu", (
            f"Requires Cupy array, but got {type(poles)}."
        )
        assert residues.device != "cpu", (
            f"Requires Cupy array, but got {type(residues)}."
        )
        assert counterrotating_pole_signs.device != "cpu", (
            f"Requires Cupy array, but got {type(counterrotating_pole_signs)}."
        )
        assert states.device != "cpu", (
            f"Requires Cupy array, but got {type(states)}."
        )
        assert voltage.device != "cpu", (
            f"Requires Cupy array, but got {type(voltage)}."
        )
        assert update_on_bin.device != "cpu", (
            f"Requires Cupy array, but got {type(update_on_bin)}."
        )

        complex_dtype = np.complex64 if np.float32 == FLOAT else np.complex128
        assert profile.dtype == FLOAT
        assert profile_dts.dtype == FLOAT
        assert voltage.dtype == FLOAT
        assert counterrotating_pole_signs.dtype == FLOAT
        assert poles.dtype == complex_dtype
        assert residues.dtype == complex_dtype
        assert states.dtype == complex_dtype
        assert update_on_bin.dtype == np.int32

        assert profile.flags.c_contiguous
        assert profile_dts.flags.c_contiguous
        assert poles.flags.c_contiguous
        assert residues.flags.c_contiguous
        assert counterrotating_pole_signs.flags.c_contiguous
        assert states.flags.c_contiguous
        assert voltage.flags.c_contiguous
        assert update_on_bin.flags.c_contiguous

        n_bins = int(profile.shape[0])
        n_poles = int(poles.shape[0])
        n_updates = int(update_on_bin.shape[0])
        n_profile_dts = int(profile_dts.shape[0])

        # states has length n_poles + 1; last entry stores t_start.
        assert states.shape[0] == n_poles + 1
        assert residues.shape[0] == n_poles
        assert counterrotating_pole_signs.shape[0] == n_poles
        assert voltage.shape[0] == n_bins

        # Output is reduced across poles via atomicAdd; must start at zero.
        voltage.fill(0)

        if n_poles == 0 or n_bins == 0:
            return

        # View complex arrays as interleaved real/imag float arrays without
        # copying. A C-contiguous complex array maps 1:1 to 2*N reals.
        poles_r = poles.view(FLOAT)
        residues_r = residues.view(FLOAT)
        states_r = states.view(FLOAT)

        # One thread per pole. Each thread runs the full n_bins-long state
        # recurrence sequentially; there is no benefit from oversubscribing.
        MAX_POLES = 128
        threads_per_block = MAX_POLES if n_poles >= MAX_POLES else 32
        blocks_poles = (n_poles + threads_per_block - 1) // threads_per_block

        _wake_from_pole_residue(
            args=(
                profile,
                profile_dts,
                poles_r,
                residues_r,
                np.int32(1 if is_counterrotating_beam else 0),
                counterrotating_pole_signs,
                update_on_bin,
                FLOAT(factor),
                states_r,
                voltage,
                np.int32(n_bins),
                np.int32(n_poles),
                np.int32(n_updates),
                np.int32(n_profile_dts),
            ),
            block=(threads_per_block, 1, 1),
            grid=(blocks_poles, 1, 1),
        )
