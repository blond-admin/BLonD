from typing import TYPE_CHECKING

import numpy as np

from ..backend import Specials

if TYPE_CHECKING:  # pragma: no cover

    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray

class PythonSpecials(Specials):
    @staticmethod
    def kick_multi_harmonic(dt: NumpyArray, dE: NumpyArray, voltage: NumpyArray,
             omega_rf: NumpyArray, phi_rf: NumpyArray,
             charge: float, n_rf: int, acceleration_kick: float):
        """
        Function to apply RF kick on the particles with sin function
        """

        voltage_kick = charge * voltage

        for j in range(n_rf):
            dE += voltage_kick[j] * np.sin(omega_rf[j] * dt + phi_rf[j])

        dE[:] += acceleration_kick


    @staticmethod
    def drift_simple(dt: NumpyArray, dE: NumpyArray, t_rev: float,
              length_ratio: float, eta_0: float, beta: float, energy: float):
        """
        Function to apply drift equation of motion
        """

        # solver_decoded = solver.decode(encoding='utf_8')

        T = t_rev * length_ratio

        coeff = eta_0 / (beta * beta * energy)
        dt += T * coeff * dE

    @staticmethod
    def drift_legacy(dt: NumpyArray, dE: NumpyArray, t_rev: float,
                     length_ratio: float, alpha_order, eta_0: float,
                     eta_1: float, eta_2: float, beta: float, energy: float):
        """
        Function to apply drift equation of motion
        """

        # solver_decoded = solver.decode(encoding='utf_8')

        T = t_rev * length_ratio
        coeff = 1. / (beta * beta * energy)
        eta0 = eta_0 * coeff
        eta1 = eta_1 * coeff * coeff
        eta2 = eta_2 * coeff * coeff * coeff

        if alpha_order == 0:
            dt += T * (1. / (1. - eta0 * dE) - 1.)
        elif alpha_order == 1:
            dt += T * (1. / (1. - eta0 * dE
                             - eta1 * dE * dE) - 1.)
        else:
            dt += T * (1. / (1. - eta0 * dE
                             - eta1 * dE * dE
                             - eta2 * dE * dE * dE) - 1.)


    @staticmethod
    def drift_exact(dt: NumpyArray, dE: NumpyArray, t_rev: float,
                     length_ratio: float, alpha_0: float,
                     alpha_1: float, alpha_2: float, beta: float, energy: float):
        """
        Function to apply drift equation of motion
        """

        # solver_decoded = solver.decode(encoding='utf_8')

        T = t_rev * length_ratio
        invbetasq = 1 / (beta * beta)
        invenesq = 1 / (energy * energy)
        # double beam_delta;

        beam_delta = np.sqrt(1. + invbetasq *
                             (dE * dE * invenesq + 2. * dE / energy)) - 1.

        dt += T * (
                (1. + alpha_0 * beam_delta +
                 alpha_1 * (beam_delta * beam_delta) +
                 alpha_2 * (beam_delta * beam_delta * beam_delta)) *
                (1. + dE / energy) / (1. + beam_delta) - 1.)

