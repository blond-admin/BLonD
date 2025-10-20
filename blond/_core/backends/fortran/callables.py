from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
import warnings
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from ...._core.backends.backend import Specials, backend
from ...._generals._hashing import hash_in_folder

logger = logging.getLogger(__name__)


def _add_dll_directory(command):
    """Add the bin directory of some executable to the DLL search path.

    Parameters
    ----------
    command
        A command line command, e.g. gcc or gfortran

    Returns
    -------

    """
    _gfortran_path = shutil.which(command)
    if not _gfortran_path:
        raise RuntimeError(f'shutil.which("{command}") = {_gfortran_path}')
    gfortran_bin_directory = os.path.dirname(_gfortran_path)
    logger.debug(f"Added {gfortran_bin_directory=} to the DLL search path.")
    os.add_dll_directory(gfortran_bin_directory)


_using_windows = os.name == "nt"
if _using_windows:
    try:
        # this is implemented to prevent
        # add_backend(..) from crashing on windows
        _add_dll_directory("gfortran")
    except RuntimeError as exc:
        warnings.warn(str(exc), stacklevel=1)


def find_module_so(file: str) -> str:
    # Get the directory of that file
    this_folder = os.path.dirname(os.path.abspath(__file__))
    hash_ = hash_in_folder(
        folder=this_folder,
        extensions=(".py", ".f90"),
        recursive=False,
    )
    folder = os.path.join(this_folder, "compiled", hash_)

    # List all files in the directory and filter
    for filename in os.listdir(folder):
        is_lib = (
            filename.endswith(".so")  # linux
            or filename.endswith(".pyd")  # windows
        )
        if is_lib and file in filename:
            return os.path.join(folder, filename)
    raise FileNotFoundError(file)


def add_backend(module_name: str) -> ModuleType:
    module_path = find_module_so(module_name)
    # Load it explicitly
    spec = importlib.util.spec_from_file_location(
        module_name, str(module_path)
    )
    if spec is not None:
        loaded_module = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(loaded_module)
        else:
            raise Exception(
                f"Failed to load spec from {module_name} at {module_path}"
            )
        # (Optional) Add to sys.modules to make it importable elsewhere
        sys.modules[module_name] = loaded_module
    else:
        raise Exception(
            f"Failed to load spec from {module_name} at {module_path}"
        )
    return loaded_module


if backend.float == np.float32:
    libblond_fortran = add_backend(module_name="libblond32")

elif backend.float == np.float64:
    libblond_fortran = add_backend(module_name="libblond64")

else:
    raise TypeError(backend.float)


if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class FortranSpecials(Specials):
    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: np.float32 | np.float64,
        omega_rf: np.float32 | np.float64,
        phi_rf: np.float32 | np.float64,
        bin_size: np.float32 | np.float64,
    ) -> np.float32 | np.float64:
        return libblond_fortran.beam_phase_module.beam_phase(
            bin_centers=hist_x,
            profile=hist_y,
            alpha=alpha,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            bin_size=bin_size,
            n_bins=np.int32(len(hist_x)),
        )

    @staticmethod
    def histogram(
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: np.float32 | np.float64,
        stop: np.float32 | np.float64,
    ) -> None:
        libblond_fortran.histogram(
            array_read,
            array_out=array_write,
            n_macroparticles=np.int32(len(array_read)),
            n_slices=np.int32(len(array_write)),
            cut_left=start,
            cut_right=stop,
        )

    @staticmethod
    def loss_box(
        top: np.float32 | np.float64,
        bottom: np.float32 | np.float64,
        left: np.float32 | np.float64,
        right: float,
    ) -> None:
        pass

    @staticmethod
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: np.float32 | np.float64,
        omega_rf: np.float32 | np.float64,
        phi_rf: np.float32 | np.float64,
        charge: np.float32 | np.float64,
        acceleration_kick: np.float32 | np.float64,
    ) -> None:
        assert dt.dtype == backend.float
        assert dE.dtype == backend.float
        assert isinstance(voltage, backend.float)
        assert isinstance(omega_rf, backend.float)
        assert isinstance(phi_rf, backend.float)
        assert isinstance(charge, backend.float)
        assert isinstance(acceleration_kick, backend.float)
        libblond_fortran.kick_single_harmonic(
            dt=dt,
            de=dE,
            voltage=voltage,
            omega_rf=omega_rf,
            phi_rf=phi_rf,
            charge=charge,
            acceleration_kick=acceleration_kick,
            n=np.int32(len(dt)),
        )
        pass

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
        libblond_fortran.drift_simple(
            dt=dt,
            de=dE,
            t=T,
            eta_0=eta_0,
            beta=beta,
            energy=energy,
            n=np.int32(len(dt)),
        )
        pass

    @staticmethod
    def kick_multi_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: np.float32 | np.float64,
        n_rf: int,
        acceleration_kick: np.float32 | np.float64,
    ) -> None:
        libblond_fortran.kick_multi_harmonic(
            dt=dt,
            de=dE,
            n_rf=n_rf,
            charge=charge,
            voltage=voltage[:],
            omega_rf=omega_rf[:],
            phi_rf=phi_rf[:],
            n_macroparticles=np.int32(len(dt)),
            acc_kick=acceleration_kick,
        )

    @staticmethod
    def drift_legacy(
        dt: NumpyArray,
        dE: NumpyArray,
        t_rev: np.float32 | np.float64,
        length_ratio: np.float32 | np.float64,
        alpha_order: int,
        eta_0: np.float32 | np.float64,
        eta_1: np.float32 | np.float64,
        eta_2: float,
        beta: float,
        energy: float,
    ) -> None:
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
    ) -> None:
        raise NotImplementedError
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
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: np.float32 | np.float64,
        acceleration_kick: np.float32 | np.float64,
    ) -> None:
        libblond_fortran.linear_interp_kick(
            beam_dt=dt,
            beam_de=dE,
            voltage_array=voltage,
            bin_centers=bin_centers,
            charge=charge,
            n_slices=np.int32(len(bin_centers)),
            n_macroparticles=np.int32(len(dt)),
            acc_kick=acceleration_kick,
        )
