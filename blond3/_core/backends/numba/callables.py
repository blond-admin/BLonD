from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import njit, prange
from cupy.typing import NDArray as CupyArray
from numpy._typing import NDArray as NumpyArray

from ..backend import backend
from ..backend import Specials

if TYPE_CHECKING:  # pragma: no cover
    pass

from ..python.callables import PythonSpecials

if backend.float == np.float32:
    nb_f = numba.float32

elif backend.float == np.float64:
    nb_f = numba.float32
else:
    raise Exception(backend.float)

sig_dt = nb_f[:]
sig_dE = nb_f[:]
sig_singleharmonic_voltage = nb_f
sig_singleharmonic_omega_rf = nb_f
sig_singleharmonic_phi_rf = nb_f
sig_singleharmonic_charge = nb_f
sig_singleharmonic_acceleration_kick = nb_f

sig_t_rev = nb_f
sig_length_ratio = nb_f
sig_eta_0 = nb_f
sig_beta = nb_f
sig_energy = nb_f

# function signatures
sig_kick_single_harmonic = (
    sig_dt,
    sig_dE,
    sig_singleharmonic_voltage,
    sig_singleharmonic_omega_rf,
    sig_singleharmonic_phi_rf,
    sig_singleharmonic_charge,
    sig_singleharmonic_acceleration_kick,
)


sig_drift_simple = (
    sig_dt,
    sig_dE,
    sig_t_rev,
    sig_length_ratio,
    sig_eta_0,
    sig_beta,
    sig_energy,
)


class NumbaSpecials(Specials):
    @staticmethod
    def loss_box(self, a, b, c, d) -> None:
        pass

    @staticmethod
    @njit(sig_kick_single_harmonic, parallel=True)
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
    @njit(sig_drift_simple, parallel=True)
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        t_rev: float,
        length_ratio: float,
        eta_0: float,
        beta: float,
        energy: float,
    ):
        """
        Function to apply drift equation of motion
        """

        # solver_decoded = solver.decode(encoding='utf_8')

        T = t_rev * length_ratio

        coeff = T * eta_0 / (beta * beta * energy)
        for i in prange(len(dt)):
            dt[i] += coeff * dE[i]

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
        pass

    @staticmethod
    def drift_legacy(
        dt: NumpyArray,
        dE: NumpyArray,
        t_rev: float,
        length_ratio: float,
        alpha_order,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ):
        pass

    @staticmethod
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
        pass
