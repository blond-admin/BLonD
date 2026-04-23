# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `NumbaSpecials` and helper functions."""
# pragma: no cover

from __future__ import annotations

import logging
from functools import cache, wraps
from typing import TYPE_CHECKING

import numba  # type: ignore
import numpy as np
from numba import boolean, complex128, int32, njit, prange, void

from blond.core.backends.backend import Specials
from blond.core.backends.python.callables import (
    _move_flagged_elements_to_end_py,
)
from blond.core.beam.flags import BeamFlags

from .fastmath import fast_sin

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

logger = logging.getLogger(__name__)


def enforce_precision(dtype):
    """Decorator to convert float inputs to a consistent precision."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, float):
                    new_args.append(dtype(arg))
                else:
                    new_args.append(arg)

            new_kwargs = {
                k: (dtype(v) if isinstance(v, float) else v)
                for k, v in kwargs.items()
            }

            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator


def enforce_return_precision(dtype):
    """Decorator to convert float outputs to a consistent precision."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return dtype(func(*args, **kwargs))

        return wrapper

    return decorator


@cache  # or set a limit like maxsize=128
def recompile_numba_backend(  # NOQA PLR0915 # NOQA: D102
    floattype: type[np.float32 | np.float64],
):
    """
    Helper to recompile `NumbaSpecials` when the backend changed.

    Parameters
    ----------
    floattype
        Float type to compile the backend for.
        `np.float32` or `np.float64` bit.

    Returns
    -------
    NumbaSpecials
        The `NumbaSpecials` backend.
    """
    logger.info(f"Compiling numba for {floattype}")

    nb_i = numba.int32

    if floattype == np.float32:
        nb_f = numba.float32

    elif floattype == np.float64:
        nb_f = numba.float64

    else:
        raise TypeError(floattype)

    sig_dt = nb_f[:]
    sig_dE = nb_f[:]
    sig_singleharmonic_voltage = nb_f
    sig_singleharmonic_omega_rf = nb_f
    sig_singleharmonic_phi_rf = nb_f
    sig_charge = nb_f
    sig_acceleration_kick = nb_f
    sig_voltage_multi_harmonic = nb_f[:]
    sig_omega_rf_multi_harmonic = nb_f[:]
    sig_phi_rf_multi_harmonic = nb_f[:]
    sig_n_rf_multi_harmonic = nb_i

    sig_t_rev = nb_f
    sig_T = nb_f
    sig_eta_0 = nb_f
    sig_alpha_0 = nb_f
    sig_higher_alpha = nb_f[:]
    sig_beta = nb_f
    sig_energy = nb_f

    sig_voltage = nb_f[:]
    sig_bin_centers = nb_f[:]

    # function signatures
    sig_kick_single_harmonic = void(
        sig_dt,
        sig_dE,
        sig_singleharmonic_voltage,
        sig_singleharmonic_omega_rf,
        sig_singleharmonic_phi_rf,
        sig_charge,
        sig_acceleration_kick,
    )

    sig_kick_multi_harmonic = void(
        sig_dt,
        sig_dE,
        sig_voltage_multi_harmonic,
        sig_omega_rf_multi_harmonic,
        sig_phi_rf_multi_harmonic,
        sig_charge,
        sig_n_rf_multi_harmonic,
        sig_acceleration_kick,
    )

    sig_sum_1d_array = nb_f(nb_f[:])

    sig_dot_product_1d_array = nb_f(nb_f[:], nb_f[:])

    sig_drift_simple = void(
        sig_dt,
        sig_dE,
        sig_T,
        sig_eta_0,
        sig_beta,
        sig_energy,
    )

    sig_drift_exact = void(
        sig_dt,  # dt: NumpyArray,
        sig_dE,  # dE: NumpyArray,
        sig_t_rev,  # T: float,
        sig_alpha_0,  # alpha_0: float,
        sig_higher_alpha,  # higher_alpha: NumpyArray,
        sig_beta,  # beta: float,
        sig_energy,  # energy: float,
    )

    sig_kick_induced_voltage = void(
        sig_dt,
        sig_dE,
        sig_voltage,
        sig_bin_centers,
        sig_charge,
        sig_acceleration_kick,
    )
    sig_array_read = nb_f[:]
    sig_array_write = nb_f[:]
    sig_start = nb_f
    sig_stop = nb_f
    sig_histogram = (
        sig_array_read,
        sig_array_write,
        sig_start,
        sig_stop,
    )

    sig_histogram_sparse = (
        sig_array_read,  # x: NumpyArray,
        sig_array_write,  # out: NumpyArray,
        nb_f,  # first_left_cut: float,
        nb_f,  # left_cut_distance: float,
        nb_f,  # cut_width: float,
        numba.int32,  # bins_per_profile: int,
        numba.int32,  # n_profiles: int,
        numba.bool[:],  # filling_pattern: NumpyArray,
        numba.int32[:],  # bucket_index_to_memory_index: NumpyArray,
    )

    sig_hist_x = nb_f[:]
    sig_hist_y = nb_f[:]
    sig_alpha = nb_f
    sig_omega_rf = nb_f
    sig_phi_rf = nb_f
    sig_bin_size = nb_f

    sig_beam_phase = nb_f(
        sig_hist_x,
        sig_hist_y,
        sig_alpha,
        sig_omega_rf,
        sig_phi_rf,
        sig_bin_size,
    )
    # Internal definition, to make `njit` compile with the correct signature,
    # that is eiter 32 or 64 bit, defined by the backend.

    sig_flag = numba.int32
    sig_flags = numba.int32[:]
    sig_ids = nb_i[:]
    sig_move_flagged_elements_to_end = nb_i(
        sig_flag,
        sig_flags,
        sig_dt,
        sig_dE,
        sig_ids,
    )

    sig_top = nb_f
    sig_bottom = nb_f
    sig_left = nb_f
    sig_right = nb_f

    sig_loss_box = (
        sig_top,
        sig_bottom,
        sig_left,
        sig_right,
        sig_dt,
        sig_dE,
        sig_flags,
    )

    _move_flagged_elements_to_end_nb = njit(sig_move_flagged_elements_to_end)(
        _move_flagged_elements_to_end_py
    )
    _lost = BeamFlags.LOST.value

    class NumbaSpecials(Specials):  # pragma: no cover
        @staticmethod
        @enforce_precision(floattype)
        @enforce_return_precision(floattype)
        @njit(
            sig_beam_phase,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def beam_phase(
            hist_x: NumpyArray,
            hist_y: NumpyArray,
            alpha: float,
            omega_rf: float,
            phi_rf: float,
            bin_size: float,
        ) -> float:
            n = len(hist_x)

            f_sin = np.zeros_like(hist_x)
            f_cos = np.zeros_like(hist_x)

            for i in prange(n):
                exp_term_i = np.exp(alpha * hist_x[i])
                angle_i = omega_rf * hist_x[i] + phi_rf
                sin_term = np.sin(angle_i)
                cos_term = np.cos(angle_i)

                # Prepare the function values for integration
                val = exp_term_i * hist_y[i]
                f_sin[i] = val * sin_term
                f_cos[i] = val * cos_term

            scoeff = 0.0
            for i in range(n - 1):
                scoeff += 0.5 * (f_sin[i] + f_sin[i + 1]) * bin_size

            ccoeff = 0.0
            for i in range(n - 1):
                ccoeff += 0.5 * (f_cos[i] + f_cos[i + 1]) * bin_size

            return scoeff / ccoeff

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_histogram,
            parallel=True,
            fastmath=True,
            cache=False,
        )
        def histogram(
            array_read: NumpyArray,
            array_write: NumpyArray,
            start: float,
            stop: float,
        ) -> None:
            n_threads = numba.get_num_threads()  # this prevents caching
            width = stop - start
            n_bins = len(array_write)
            bin_step = width / n_bins
            inv_bin_step = 1 / bin_step
            array_tmp = np.zeros((n_threads, n_bins))
            array_write[:] = 0
            for i in prange(len(array_read)):
                curr_thread = numba.get_thread_id()
                if array_read[i] == stop:
                    array_tmp[curr_thread, -1] += 1
                    continue
                idx = (array_read[i] - start) * inv_bin_step
                if idx < 0 or idx >= n_bins:
                    continue
                else:
                    array_tmp[curr_thread, int(idx)] += 1
            array_write[:] = np.sum(array_tmp, axis=0)

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_loss_box,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def loss_box(
            e_max: np.float32 | np.float64,
            e_min: np.float32 | np.float64,
            t_min: np.float32 | np.float64,
            t_max: np.float32 | np.float64,
            dt: NumpyArray,
            dE: NumpyArray,
            flags: NumpyArray,
        ) -> None:
            for i in prange(len(dt)):
                select = (
                    (dE[i] > e_max)
                    | (dE[i] < e_min)
                    | (dt[i] < t_min)
                    | (dt[i] > t_max)
                )
                if select:
                    flags[i] = _lost

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_kick_single_harmonic,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def kick_single_harmonic(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            voltage_kick = charge * voltage
            for i in prange(len(dt)):
                dE[i] += (
                    voltage_kick * fast_sin(omega_rf * dt[i] + phi_rf)
                    + acceleration_kick
                )

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_drift_simple,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def drift_simple(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            eta_0: float,
            beta: float,
            energy: float,
        ) -> None:
            """Function to apply drift equation of motion."""
            # solver_decoded = solver.decode(encoding='utf_8')

            coeff = T * eta_0 / (beta * beta * energy)
            for i in prange(len(dt)):
                dt[i] += coeff * dE[i]

        @staticmethod
        @enforce_precision(floattype)
        @njit(sig_kick_multi_harmonic, parallel=True, fastmath=False)
        def kick_multi_harmonic(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            omega_rf: NumpyArray,
            phi_rf: NumpyArray,
            charge: float,
            n_rf: int,
            acceleration_kick: float,
        ) -> None:
            for i in prange(len(dt)):
                dti = dt[i]
                de_sum = 0.0
                for j in range(n_rf):
                    de_sum += (
                        charge
                        * voltage[j]
                        * fast_sin(omega_rf[j] * dti + phi_rf[j])
                    )
                dE[i] += de_sum + acceleration_kick

        @staticmethod
        @enforce_precision(floattype)
        @enforce_return_precision(floattype)
        @njit(sig_sum_1d_array, parallel=True, cache=False, fastmath=True)
        def sum_1d_array(
            array_1: NumpyArray,
        ):
            acc = floattype(0.0)
            for idx in prange(array_1.shape[0]):
                acc += array_1[idx]
            return acc

        @staticmethod
        @enforce_precision(floattype)
        @enforce_return_precision(floattype)
        @njit(
            sig_dot_product_1d_array, parallel=True, cache=False, fastmath=True
        )
        def dot_product_1d_array(array_1: NumpyArray, array_2: NumpyArray):
            acc = floattype(0.0)
            for idx in prange(array_1.shape[0]):
                acc += array_1[idx] * array_2[idx]
            return acc

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_drift_exact,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def drift_exact(
            dt: NumpyArray,
            dE: NumpyArray,
            T: float,
            alpha_0: float,
            higher_alpha: NumpyArray,
            beta: float,
            energy: float,
        ) -> None:
            inv_beta_sq = 1.0 / (beta * beta)
            inv_energy = 1.0 / energy
            inv_energy_sq = inv_energy * inv_energy

            n_alpha = len(higher_alpha)

            for i in prange(len(dt)):
                dEi = dE[i]

                delta = (
                    np.sqrt(
                        1.0
                        + inv_beta_sq
                        * (dEi * dEi * inv_energy_sq + 2.0 * dEi * inv_energy)
                    )
                    - 1.0
                )

                poly = 1.0 + alpha_0 * delta

                if n_alpha > 0:
                    delta_power = delta * delta  # starts at δ²

                    for k in range(n_alpha):
                        poly += higher_alpha[k] * delta_power
                        delta_power *= delta  # next power

                dt[i] += T * (
                    poly * (1.0 + dEi * inv_energy) / (1.0 + delta) - 1.0
                )

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_kick_induced_voltage,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def kick_induced_voltage(
            dt: NumpyArray,
            dE: NumpyArray,
            voltage: NumpyArray,
            bin_centers: NumpyArray,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            dx = (bin_centers[-1] - bin_centers[0]) / (len(bin_centers) - 1)
            inv_dx = 1 / dx
            x_min = bin_centers[0]
            x_max = bin_centers[-1]
            for i in prange(len(dE)):
                x = dt[i]

                if x < x_min or x >= x_max:
                    continue
                else:
                    idx = int((x - x_min) * inv_dx)
                    x0 = x_min + idx * dx
                    # x1 = x0 + dx
                    y0 = voltage[idx]
                    y1 = voltage[idx + 1]

                    # Linear interpolation
                    v = y0 + (y1 - y0) * inv_dx * (x - x0)
                    dE[i] += charge * v + acceleration_kick

        @staticmethod
        def move_flagged_elements_to_end(
            flag: int,
            flags: NumpyArray,  # also purged
            dt: NumpyArray,
            dE: NumpyArray,
            ids: NumpyArray,
        ):
            # TODO parallel version of sorting
            n_new = _move_flagged_elements_to_end_nb(
                flag=np.int32(flag),
                flags=flags,
                dt=dt,
                dE=dE,
                ids=ids,
            )
            return n_new

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            sig_histogram_sparse,
            parallel=True,
            fastmath=True,
            cache=False,
        )
        def histogram_sparse(
            x: NumpyArray,
            out: NumpyArray,
            first_left_cut: float,
            left_cut_distance: float,
            cut_width: float,
            bins_per_profile: int,
            n_active_profiles: int,
            filling_pattern: NumpyArray,
            bucket_index_to_memory_index: NumpyArray,
        ) -> None:
            """
            Sparse histogram with strided memory layout (gaps between profiles).

            Parameters
            ----------
            x
                An array, e.g., the particle ``dt`` values.
            out
                Output histogram ``(n_filled_buckets * bins_per_profile)``.
            first_left_cut
                Start of the first histogram.
            left_cut_distance
                Distance between the start of each histogram.
            cut_width
                Distance between left and right edge of the histogram.
            bins_per_profile
                Number of bins per bucket.
            n_active_profiles
                Number of non-empty buckets.
            filling_pattern
                Filling pattern as a boolean array
                where ``True`` means filled bucket.
            bucket_index_to_memory_index
                Maps bucket index to memory index.
                For a ``filling_pattern = [1, 0, 0, 1]``
                ``bucket_index_to_memory_index = [0, 0, 0, 8]`` with
                ``bins_per_profile = 8``.
                Use `_gen_array_bucket_index_to_memory_index` to generate this.
            """
            n_threads = numba.get_num_threads()  # this prevents caching
            array_tmp = np.zeros((n_threads, len(out)))

            ive_profile_dist = 1 / left_cut_distance
            inv_bin_step = bins_per_profile / cut_width
            n_buckets = len(filling_pattern)

            for i in prange(len(x)):
                thread_i = numba.get_thread_id()

                xi = x[i]

                bucket_i = int((xi - first_left_cut) * ive_profile_dist)

                if bucket_i < 0 or bucket_i >= n_buckets:
                    continue
                if not filling_pattern[bucket_i]:
                    continue

                start_loc = first_left_cut + bucket_i * left_cut_distance
                stop_loc = start_loc + cut_width

                if xi == stop_loc:
                    write_idx = (
                        bucket_index_to_memory_index[bucket_i]
                        + bins_per_profile
                        - 1
                    )
                    array_tmp[thread_i, write_idx] += 1
                    continue

                idx = int((xi - start_loc) * inv_bin_step)
                if idx < 0 or idx >= bins_per_profile:
                    continue
                else:
                    write_idx = int(
                        bucket_index_to_memory_index[bucket_i] + idx
                    )
                    array_tmp[thread_i, write_idx] += 1

            out[:] = np.sum(array_tmp, axis=0)

        @staticmethod
        @enforce_precision(floattype)
        @njit(
            void(
                nb_f[:],
                nb_f[:],
                complex128[:],
                complex128[:],
                boolean,
                nb_f[:],
                complex128[:],
                nb_f[:],
                nb_f[:, :],
                int32[:],
                nb_f,
            ),
            parallel=True,
            fastmath=True,
            cache=False,
        )
        def wake_from_pole_residue(
            # read
            profile,
            profile_dts,
            poles,
            residues,
            beam_counter_rotation_flag,
            cr_pole_flip_flags,
            # write
            states,
            voltage,
            voltage_threaded,
            update_on_bin,
            factor,
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
                Complex poles of an equivalent circuit.
            residues
                Complex residues of an equivalent circuit.
            beam_counter_rotation_flag
                If true, the current beam is counter-rotating.
            cr_pole_flip_flags
                Array per pole, -1 if the sign of the impedance is flipped
                for a counter-rotating beam.
            states
                Complex state vector, initially ``(0 + 0j)``.
            voltage
                Output voltage, in [V].
            voltage_threaded
                Cached `voltage` array per thread. For speedup.
            update_on_bin
                Index when to trigger an update of dt. For speedup.
                E.g. For profile No.: `0,0,0,1,1,1,1,2,2,2`
                one needs `update_on_bin = [0,3,7]`.

            factor
                To convert `profile` to current per bin [A].
            """
            n_poles = len(poles)
            two_factor = 2 * factor
            n_bins = len(profile)

            voltage[:] = 0  # reset to zero from previous call
            voltage_threaded[:, :] = 0  # reset to zero from previous call
            if not (voltage_threaded.shape[0] == numba.get_num_threads()):
                raise Exception
            for pole_i in prange(n_poles):
                thread_i = numba.get_thread_id()

                cr_pole_flip = 1.0
                if (
                    beam_counter_rotation_flag
                    and cr_pole_flip_flags[pole_i] == -1
                ):
                    cr_pole_flip = -1.0

                # y[n] = profile[n] + exp(p * dt) * y[n-1]
                # V[n] = 2 * Re(r * y[n])
                # state = 0.0 + 0.0j
                i_update = 0
                update_on_bin_i = update_on_bin[i_update]

                pole = complex(poles[pole_i])
                residue = complex(residues[pole_i])
                state = complex(states[pole_i])

                t_start = states[-1]

                for bin_i in range(n_bins):
                    profile_i_half = complex(0.5 * profile[bin_i])

                    if bin_i == update_on_bin_i:
                        if bin_i == 0:
                            t_jump = profile_dts[0] - t_start + 0j
                        else:
                            t_jump = (
                                profile_dts[bin_i]
                                - profile_dts[bin_i - 1]
                                + 0j
                            )
                        state *= np.exp(pole * t_jump)
                        dt = profile_dts[bin_i + 1] - profile_dts[bin_i]
                        decay = np.exp(pole * dt)

                        i_update += 1
                        if i_update < len(update_on_bin):
                            update_on_bin_i = update_on_bin[i_update]
                    else:
                        state *= decay
                    state += cr_pole_flip * profile_i_half
                    amp = float(np.real(residue * state))
                    voltage_threaded[thread_i, bin_i] += (
                        cr_pole_flip * two_factor * amp
                    )
                    state += cr_pole_flip * profile_i_half
                states[pole_i] = state

            for thread_i in prange(numba.get_num_threads()):
                voltage += voltage_threaded[thread_i, :]
            states[-1] = profile_dts[-1]

    return NumbaSpecials


if TYPE_CHECKING:  # pragma: no cover
    from blond import backend

    NumbaSpecials = recompile_numba_backend(backend.float)
