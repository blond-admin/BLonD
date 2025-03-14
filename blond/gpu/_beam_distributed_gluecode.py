from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupy.typing import NDArray as CupyNDArray
from numpy.typing import NDArray

from blond.beam.beam import Beam
from blond.gpu.butils_wrap_cupy import (
    kick,
    drift,
    losses_separatrix,
    losses_longitudinal_cut,
    losses_energy_cut,
    losses_below_energy,
    linear_interp_kick,
    kickdrift_considering_periodicity,
)

if TYPE_CHECKING:
    from blond.input_parameters.rf_parameters import RFStation
    from blond.input_parameters.ring import Ring
    from blond.utils.types import SolverTypes


def _kick_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    voltage,
    omega_rf,
    phi_rf,
    charge: float,
    n_rf: int,
    acceleration_kick: float,
):
    # copy voltage etc. to the active device.
    # We can do this because anyway each turn there is another voltage
    # rf_station.voltage[:, turn_i]

    kick(
        dt=dt_gpu_i,
        dE=dE_gpu_i,
        voltage=cp.array(voltage),  # so that voltage is on each device
        omega_rf=cp.array(omega_rf),  # so that voltage is on each device
        phi_rf=cp.array(phi_rf),  # so that voltage is on each device
        charge=charge,
        n_rf=n_rf,
        acceleration_kick=acceleration_kick,
    )


def _drift_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    solver: SolverTypes,
    t_rev: float,
    length_ratio: float,
    alpha_order: float,
    eta_0: float,
    eta_1: float,
    eta_2: float,
    alpha_0: float,
    alpha_1: float,
    alpha_2: float,
    beta: float,
    energy: float,
):
    drift(
        dt=dt_gpu_i,
        dE=dE_gpu_i,
        solver=solver,
        t_rev=t_rev,
        length_ratio=length_ratio,
        alpha_order=alpha_order,
        eta_0=eta_0,
        eta_1=eta_1,
        eta_2=eta_2,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        beta=beta,
        energy=energy,
    )


def _losses_separatrix_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    ring: Ring,
    beam: Beam,
    rf_station: RFStation,
):
    losses_separatrix(
        ring=ring,
        rf_station=rf_station,
        beam=beam,
        dt=dt_gpu_i,
        dE=dE_gpu_i,
        id=id_gpu_i,
    )


def _losses_longitudinal_cut_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    dt_min: float,
    dt_max: float,
):
    losses_longitudinal_cut(
        dt=dt_gpu_i, id=id_gpu_i, dt_min=dt_min, dt_max=dt_max
    )


def _losses_energy_cut_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    dE_min: float,
    dE_max: float,
):
    losses_energy_cut(dE=dE_gpu_i, id=id_gpu_i, dE_min=dE_min, dE_max=dE_max)


def _losses_below_energy_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    dE_min: float,
):
    losses_below_energy(dE=dE_gpu_i, id=id_gpu_i, dE_min=dE_min)


def _dE_mean_helper(dE_gpu_i: CupyNDArray):
    return cp.mean(dE_gpu_i)


def _dE_mean_helper_ignore_id_0(
    dt_gpu_i: CupyNDArray, dE_gpu_i: CupyNDArray, id_gpu_i: CupyNDArray
):
    mask: CupyNDArray = id_gpu_i > 0  # noqa
    masked = dE_gpu_i[mask]  # might be written more efficient  because
    # masking makes array copies
    if len(masked) == 0:
        return np.nan
    else:
        return cp.mean(masked)


def _dE_std_helper(dE_gpu_i: CupyNDArray, mean: float):
    tmp = dE_gpu_i - mean
    return cp.sum(tmp * tmp)


def _dE_std_helper_ignore_id_0(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    mean: float,
):
    mask: CupyNDArray = id_gpu_i > 0  # noqa
    if not cp.any(mask):
        return np.nan
    else:
        tmp = dE_gpu_i[mask] - mean  # might be written more efficient,
        # because masking makes array copies
        return cp.sum(tmp * tmp)


def _dt_mean_helper(dt_gpu_i: CupyNDArray):
    return cp.mean(dt_gpu_i)


def _dt_mean_helper_ignore_id_0(
    dt_gpu_i: CupyNDArray, dE_gpu_i: CupyNDArray, id_gpu_i: CupyNDArray
):
    mask: CupyNDArray = id_gpu_i > 0  # noqa
    masked = dt_gpu_i[mask]  # might be written more efficient# because
    # masking makes array copies
    if len(masked) == 0:
        return np.nan
    else:
        return cp.mean(masked)


def _dt_std_helper(dt_gpu_i: CupyNDArray, mean: float):
    tmp = dt_gpu_i - mean
    return cp.sum(tmp * tmp)


def _dt_std_helper_ignore_id_0(
    dt_gpu_i: CupyNDArray, dE_gpu_i: CupyNDArray, id_gpu_i: CupyNDArray, mean
):
    mask: CupyNDArray = id_gpu_i > 0  # noqa
    if not cp.any(mask):
        return np.nan
    else:
        tmp = dt_gpu_i[mask] - mean  # might be written more efficient,
        # because masking makes array copies
        return cp.sum(tmp * tmp)


def _linear_interp_kick_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    voltage: NDArray | CupyNDArray,
    bin_centers: NDArray | CupyNDArray,
    charge: float,
    acceleration_kick: float,
):
    linear_interp_kick(
        dt=dt_gpu_i,
        dE=dE_gpu_i,
        voltage=cp.array(voltage),  # so that voltage is on each device
        bin_centers=cp.array(
            bin_centers
        ),  # so that bin_centers is on each device
        charge=charge,
        acceleration_kick=acceleration_kick,
    )


def _kickdrift_considering_periodicity_helper(
    dt_gpu_i: CupyNDArray,
    dE_gpu_i: CupyNDArray,
    id_gpu_i: CupyNDArray,
    acceleration_kick: float,
    rf_station: RFStation,
    solver: SolverTypes,
    turn: int,
):
    kickdrift_considering_periodicity(
        acceleration_kick=acceleration_kick,
        beam_dE=dE_gpu_i,
        beam_dt=dt_gpu_i,
        rf_station=rf_station,
        solver=solver,
        turn=turn,
    )
