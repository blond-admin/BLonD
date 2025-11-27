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
from numba import njit, prange, void

from blond.core.backends.backend import Specials
from blond.core.backends.python.callables import (
    _move_flagged_elements_to_end_py,
)
from blond.core.beam.base import BeamFlags

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
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


@cache  # or set a limit like maxsize=128
def recompile_numba_backend(  # NOQA PLR0915 # NOQA: D102
    floattype: type[np.float32 | np.float64],
):
    """Helper to recompile `NumbaSpecials` when the backend changed.

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

    if floattype == np.float32:
        nb_f = numba.float32
        nb_i = numba.int32

    elif floattype == np.float64:
        nb_f = numba.float64
        nb_i = numba.int64

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
    sig_length_ratio = nb_f
    sig_eta_0 = nb_f
    sig_eta_1 = nb_f
    sig_eta_2 = nb_f
    sig_alpha_0 = nb_f
    sig_alpha_1 = nb_f
    sig_alpha_2 = nb_f
    sig_alpha_order = nb_i
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

    sig_drift_simple = void(
        sig_dt,
        sig_dE,
        sig_T,
        sig_eta_0,
        sig_beta,
        sig_energy,
    )
    sig_drift_legacy = void(
        sig_dt,
        sig_dE,
        sig_t_rev,
        sig_length_ratio,
        sig_alpha_order,
        sig_eta_0,
        sig_eta_1,
        sig_eta_2,
        sig_beta,
        sig_energy,
    )

    sig_drift_exact = void(
        sig_dt,
        sig_dE,
        sig_t_rev,
        sig_length_ratio,
        sig_alpha_0,
        sig_alpha_1,
        sig_alpha_2,
        sig_beta,
        sig_energy,
    )

    sig_mask = numba.bool[:, :]
    sig_buffer = nb_f[:]
    sig_meta_params_multibunch = (
        sig_dt,
        sig_dE,
        sig_mask,
        sig_buffer,
        sig_buffer,
        sig_buffer,
        sig_buffer,
        sig_buffer,
        nb_f,
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
            top: np.float32 | np.float64,
            bottom: np.float32 | np.float64,
            left: np.float32 | np.float64,
            right: np.float32 | np.float64,
            dt: NumpyArray,
            dE: NumpyArray,
            flags: NumpyArray,
        ) -> None:
            for i in prange(len(dt)):
                select = (
                    (dE[i] > top)
                    | (dE[i] < bottom)
                    | (dt[i] < left)
                    | (dt[i] > right)
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
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            voltage: float,
            omega_rf: float,
            phi_rf: float,
            charge: float,
            acceleration_kick: float,
        ) -> None:
            voltage_kick = charge * voltage
            for i in prange(len(dt)):
                dE[i] += (
                    voltage_kick * np.sin(omega_rf * dt[i] + phi_rf)
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
        @njit(
            sig_meta_params_multibunch,
            parallel=True,
            fastmath=True,
            cache=True,
        )
        def meta_params_multibunch(
            dt: NumpyArray,
            dE: NumpyArray,
            mask: NumpyArray,
            sigma_dt_buffer: NumpyArray,
            sigma_dE_buffer: NumpyArray,
            mean_dt_buffer: NumpyArray,
            mean_dE_buffer: NumpyArray,
            emittance_buffer: NumpyArray,
            t_rf: float,
        ) -> None:
            for bucket in prange(len(mask)):
                sigma_dt_buffer[bucket] = np.std(dt[mask[bucket]])
                sigma_dE_buffer[bucket] = np.std(dE[mask[bucket]])
                mean_dt_buffer[bucket] = (
                    np.mean(dt[mask[bucket]]) - bucket * t_rf
                )
                # correct to value of first bucket
                mean_dE_buffer[bucket] = np.mean(dE[mask[bucket]])
                emittance_buffer[bucket] = np.sqrt(
                    np.average(dE[mask[bucket]] ** 2)
                    * np.average(dt[mask[bucket]] ** 2)
                    - np.average(dE[mask[bucket]] * dt[mask[bucket]]) ** 2
                )

        @staticmethod
        @enforce_precision(floattype)
        @njit(sig_kick_multi_harmonic, parallel=True, fastmath=False)
        def kick_multi_harmonic(
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
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
                        * np.sin(omega_rf[j] * dti + phi_rf[j])
                    )
                dE[i] += de_sum + acceleration_kick

        @staticmethod
        @enforce_precision(floattype)
        @njit(sig_drift_legacy, parallel=True, fastmath=False)
        def drift_legacy(
            dt: NumpyArray,
            dE: NumpyArray,
            t_rev: float,
            length_ratio: float,
            alpha_order: int,
            eta_0: float,
            eta_1: float,
            eta_2: float,
            beta: float,
            energy: float,
        ) -> None:  # pragma: no cover # TODO
            T = t_rev * length_ratio
            coeff = 1.0 / (beta * beta * energy)
            eta0 = eta_0 * coeff
            eta1 = eta_1 * coeff * coeff
            eta2 = eta_2 * coeff * coeff * coeff
            for i in prange(len(dt)):
                dEi = dE[i]
                if alpha_order == 0:
                    dt[i] += T * (1.0 / (1.0 - eta0 * dEi) - 1.0)
                elif alpha_order == 1:
                    dt[i] += T * (
                        1.0 / (1.0 - eta0 * dEi - eta1 * dEi * dEi) - 1.0
                    )
                else:
                    dt[i] += T * (
                        1.0
                        / (
                            1.0
                            - eta0 * dEi
                            - eta1 * dEi * dEi
                            - eta2 * dEi * dEi * dEi
                        )
                        - 1.0
                    )

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
            t_rev: float,
            length_ratio: float,
            alpha_0: float,
            alpha_1: float,
            alpha_2: float,
            beta: float,
            energy: float,
        ) -> None:  # pragma: no cover # TODO
            T = t_rev * length_ratio
            invbetasq = 1 / (beta * beta)
            invenesq = 1 / (energy * energy)
            # double beam_delta;
            for i in prange(len(dt)):
                beam_delta = (
                    np.sqrt(
                        1.0
                        + invbetasq
                        * (dE[i] * dE[i] * invenesq + 2.0 * dE[i] / energy)
                    )
                    - 1.0
                )

                dt[i] += T * (
                    (
                        1.0
                        + alpha_0 * beam_delta
                        + alpha_1 * (beam_delta * beam_delta)
                        + alpha_2 * (beam_delta * beam_delta * beam_delta)
                    )
                    * (1.0 + dE[i] / energy)
                    / (1.0 + beam_delta)
                    - 1.0
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

                if x <= x_min or x >= x_max:
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
            flags: NumpyArray | CupyArray,  # also purged
            dt: NumpyArray | CupyArray,
            dE: NumpyArray | CupyArray,
            ids: NumpyArray | CupyArray,
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

    return NumbaSpecials


if TYPE_CHECKING:
    from blond import backend

    NumbaSpecials = recompile_numba_backend(backend.float)
