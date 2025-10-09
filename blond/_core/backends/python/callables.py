from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..backend import Specials

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class PythonSpecials(Specials):
    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> np.float32 | np.float64:
        scoeff = np.trapezoid(  # type: ignore
            np.exp(alpha * hist_x)
            * np.sin(omega_rf * hist_x + phi_rf)
            * hist_y,
            dx=bin_size,
        )
        ccoeff = np.trapezoid(  # type: ignore
            np.exp(alpha * hist_x)
            * np.cos(omega_rf * hist_x + phi_rf)
            * hist_y,
            dx=bin_size,
        )

        return scoeff / ccoeff

    @staticmethod
    def histogram(
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: np.float32 | np.float64,
        stop: np.float32 | np.float64,
    ) -> None:
        array_write[:], _ = np.histogram(
            array_read,
            range=(float(start), float(stop)),
            bins=len(array_write),
        )

    @staticmethod
    def loss_box(
        top: float, bottom: float, left: float, right: float
    ) -> None:  # TODO
        raise NotImplementedError

    @staticmethod
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: np.flaot32 | np.float64,
        acceleration_kick: np.flaot32 | np.float64,
    ) -> None:
        voltage_kick = charge * voltage

        dE[:] += (
            voltage_kick * np.sin(omega_rf * dt[:] + phi_rf)
            + acceleration_kick
        )

    @staticmethod
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
        """Function to apply RF kick on the particles with sin function."""
        voltage_kick = charge * voltage

        for j in range(n_rf):
            dE += voltage_kick[j] * np.sin(omega_rf[j] * dt + phi_rf[j])

        dE[:] += acceleration_kick

    @staticmethod
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        T: np.float32 | np.float64,
        eta_0: np.float32 | np.float64,
        beta: np.float32 | np.float64,
        energy: np.float32 | np.float64,
    ) -> None:
        """Function to apply drift equation of motion."""
        # solver_decoded = solver.decode(encoding='utf_8')

        coeff = eta_0 / (beta * beta * energy)
        dt += T * coeff * dE

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
    ) -> None:  # pragma: no cover # TODO
        """Function to apply drift equation of motion."""
        # solver_decoded = solver.decode(encoding='utf_8')

        coeff = 1.0 / (beta * beta * energy)
        eta0 = eta_0 * coeff
        eta1 = eta_1 * coeff * coeff
        eta2 = eta_2 * coeff * coeff * coeff

        if alpha_order == 0:
            dt += T * (1.0 / (1.0 - eta0 * dE) - 1.0)
        elif alpha_order == 1:
            dt += T * (1.0 / (1.0 - eta0 * dE - eta1 * dE * dE) - 1.0)
        else:
            dt += T * (
                1.0 / (1.0 - eta0 * dE - eta1 * dE * dE - eta2 * dE * dE * dE)
                - 1.0
            )

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
    ) -> None:  # pragma: no cover # TODO
        """Function to apply drift equation of motion."""
        # solver_decoded = solver.decode(encoding='utf_8')

        invbetasq = 1 / (beta * beta)
        invenesq = 1 / (energy * energy)
        # double beam_delta;

        beam_delta = (
            np.sqrt(1.0 + invbetasq * (dE * dE * invenesq + 2.0 * dE / energy))
            - 1.0
        )

        dt += T * (
            (
                1.0
                + alpha_0 * beam_delta
                + alpha_1 * (beam_delta * beam_delta)
                + alpha_2 * (beam_delta * beam_delta * beam_delta)
            )
            * (1.0 + dE / energy)
            / (1.0 + beam_delta)
            - 1.0
        )

    @staticmethod
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: np.flaot32 | np.float64,
        acceleration_kick: np.flaot32 | np.float64,
    ) -> None:
        """Interpolated kick method.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        voltage
            Array of voltages along `bin_centers`, in [V]
        bin_centers
            Positions of `voltage`, in [s]
        charge
            Particle charge, as number of elementary charges `e` []
        acceleration_kick
            Energy, in [eV], which is added to all particles.
            This is intended to subtract the target energy from the RF
            energy gain in one common call.

        """
        n_slices = len(bin_centers)
        inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])

        fbin = np.floor((dt - bin_centers[0]) * inv_bin_width).astype(np.int32)

        helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
        helper2 = (
            charge * voltage[:-1] - bin_centers[:-1] * helper1
        ) + acceleration_kick

        for i in range(len(dt)):
            # fbin = int(np.floor((dt[i]-bin_centers[0])*inv_bin_width))
            if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
                dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]
