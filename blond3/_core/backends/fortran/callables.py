from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from blond3._core.backends.backend import Specials


def find_kick_module_so(file: str):
    # Get the file where this function is defined
    current_file = inspect.getfile(find_kick_module_so)
    # Get the directory of that file
    folder = os.path.dirname(current_file)

    # List all files in the directory and filter
    for filename in os.listdir(folder):
        if filename.endswith(".so") and file in filename:
            return os.path.join(folder, filename)
    raise FileNotFoundError(file)


def add_backend(module_name):
    module_path = find_kick_module_so(module_name)
    # Load it explicitly
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)
    # (Optional) Add to sys.modules to make it importable elsewhere
    sys.modules[module_name] = loaded_module
    return loaded_module


drift_module = add_backend(
    module_name="drift_module",
)
kick_module = add_backend(
    module_name="kick_module",
)

kick_induced_module = add_backend(
    module_name="kick_induced_module",
)


histogram_module = add_backend(
    module_name="histogram_module",
)

beam_phase_module = add_backend(
    module_name="beam_phase_module",
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy._typing import NDArray as NumpyArray


class FortranSpecials(Specials):
    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        beam_phase_module.beam_phase(
            bin_centers=hist_x,
            profile=hist_y,
            alpha=alpha,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            bin_size=bin_size,
            n_bins=len(hist_x),
        )

    @staticmethod
    def histogram(
        array_read: NumpyArray, array_write: NumpyArray, start: float, stop: float
    ):
        histogram_module.histogram(
            array_read,
            array_out=array_write,
            n_macroparticles=len(array_read),
            n_slices=len(array_write),
            cut_left=start,
            cut_right=stop,
        )

    @staticmethod
    def loss_box(self, a, b, c, d) -> None:
        pass

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
        kick_module.kick_single_harmonic(
            dt=dt,
            de=dE,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            acceleration_kick=acceleration_kick,
            n=len(dt),
        )
        pass

    @staticmethod
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

        drift_module.drift_simple(
            dt=dt, de=dE, t=T, eta_0=eta_0, beta=beta, energy=energy, n=len(dt)
        )
        pass

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
        kick_module.kick_multi_harmonic(
            dt=dt,
            de=dE,
            n_rf=n_rf,
            charge=charge,
            voltage=voltage[:],
            omega_rf=omega_rf[:],
            phi_rf=phi_rf[:],
            n_macroparticles=len(dt),
            acc_kick=acceleration_kick,
        )

    @staticmethod
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
        raise NotImplementedError

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
        raise NotImplementedError
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
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ):
        kick_induced_module.linear_interp_kick(
            beam_dt=dt,
            beam_de=dE,
            voltage_array=voltage,
            bin_centers=bin_centers,
            charge=charge,
            n_slices=len(bin_centers),
            n_macroparticles=len(dt),
            acc_kick=acceleration_kick,
        )
