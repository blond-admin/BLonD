from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import njit, prange

from ..backend import Specials
from ..backend import backend

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy._typing import NDArray as NumpyArray

if backend.float == np.float32:
    nb_f = numba.float32
    nb_i = numba.int32
    nb_c = numba.complex64


elif backend.float == np.float64:
    nb_f = numba.float64
    nb_i = numba.int64
    nb_c = numba.complex128

else:
    raise Exception(backend.float)

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
sig_kick_single_harmonic = (
    sig_dt,
    sig_dE,
    sig_singleharmonic_voltage,
    sig_singleharmonic_omega_rf,
    sig_singleharmonic_phi_rf,
    sig_charge,
    sig_acceleration_kick,
)


sig_kick_multi_harmonic = (
    sig_dt,
    sig_dE,
    sig_voltage_multi_harmonic,
    sig_omega_rf_multi_harmonic,
    sig_phi_rf_multi_harmonic,
    sig_charge,
    sig_n_rf_multi_harmonic,
    sig_acceleration_kick,
)

sig_drift_simple = (
    sig_dt,
    sig_dE,
    sig_T,
    sig_eta_0,
    sig_beta,
    sig_energy,
)
sig_drift_legacy = (
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

sig_drift_exact = (
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


sig_kick_induced_voltage = (
    sig_dt,
    sig_dE,
    sig_voltage,
    sig_bin_centers,
    sig_charge,
    sig_acceleration_kick,
)


class NumbaSpecials(Specials):
    @staticmethod
    def loss_box(self, a, b, c, d) -> None:
        pass

    @staticmethod
    @njit(sig_kick_single_harmonic, parallel=True, fastmath=True)
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ):
        voltage_kick = charge * voltage
        for i in prange(len(dt)):
            dE[i] += (
                voltage_kick * np.sin(omega_rf * dt[i] + phi_rf) + acceleration_kick
            )

    @staticmethod
    @njit(sig_drift_simple, parallel=True, fastmath=True)
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ):
        """
        Function to apply drift equation of motion
        """

        # solver_decoded = solver.decode(encoding='utf_8')

        coeff = T * eta_0 / (beta * beta * energy)
        for i in prange(len(dt)):
            dt[i] += coeff * dE[i]

    @staticmethod
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
    ):
        for i in prange(len(dt)):
            dti = dt[i]
            de_sum = 0.0
            for j in range(n_rf):
                de_sum += charge * voltage[j] * np.sin(omega_rf[j] * dti + phi_rf[j])
            dE[i] += de_sum + acceleration_kick

    @staticmethod
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
    ):
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
                dt[i] += T * (1.0 / (1.0 - eta0 * dEi - eta1 * dEi * dEi) - 1.0)
            else:
                dt[i] += T * (
                    1.0 / (1.0 - eta0 * dEi - eta1 * dEi * dEi - eta2 * dEi * dEi * dEi)
                    - 1.0
                )

    @staticmethod
    @njit(sig_drift_exact, parallel=True, fastmath=True)
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
    ):
        T = t_rev * length_ratio
        invbetasq = 1 / (beta * beta)
        invenesq = 1 / (energy * energy)
        # double beam_delta;
        for i in prange(len(dt)):
            beam_delta = (
                np.sqrt(
                    1.0 + invbetasq * (dE[i] * dE[i] * invenesq + 2.0 * dE[i] / energy)
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
    @njit(sig_kick_induced_voltage, parallel=True, fastmath=True)
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ):
        dx = (bin_centers[-1] - bin_centers[0]) / (len(bin_centers) - 1)
        inv_dx = 1 / dx
        x_min = bin_centers[0]
        x_max = bin_centers[-1]
        for i in prange(len(dE)):
            x = dt[i]

            if x <= x_min:
                continue
            elif x >= x_max:
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
