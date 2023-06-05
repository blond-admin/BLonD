'''
@author: Konstantinos Iliakis, George Tsapatsaris
'''

import cupy as cp
import numpy as np

from ..utils import precision
from . import GPU_DEV


def rf_volt_comp(voltage, omega_rf, phi_rf, bin_centers):
    """Calculate the rf voltage at each profile bin

    Args:
        voltage (float array): _description_
        omega_rf (float array): _description_
        phi_rf (float array): _description_
        bin_centers (float array): _description_

    Returns:
        float array: the calculated rf_voltage
    """

    rf_volt_comp_kernel = GPU_DEV.mod.get_function("rf_volt_comp")

    
    assert voltage.dtype == precision.real_t
    assert omega_rf.dtype == precision.real_t
    assert phi_rf.dtype == precision.real_t
    assert bin_centers.dtype == precision.real_t

    rf_voltage = cp.zeros(bin_centers.size, precision.real_t)

    rf_volt_comp_kernel(args=(voltage, omega_rf, phi_rf, bin_centers,
                              np.int32(voltage.size), np.int32(bin_centers.size), rf_voltage),
                        block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)
    return rf_voltage


def kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick):
    """Apply the energy kick

    Args:
        dt (float array): the time coordinate
        dE (float array): the energy coordinate
        voltage (float array): _description_
        omega_rf (float array): _description_
        phi_rf (float array): _description_
        charge (float): _description_
        n_rf (int): _description_
        acceleration_kick (float): _description_
    """
    kick_kernel = GPU_DEV.mod.get_function("simple_kick")

    assert dt.dtype == precision.real_t
    assert dE.dtype == precision.real_t
    assert omega_rf.dtype == precision.real_t
    assert phi_rf.dtype == precision.real_t

    voltage_kick = cp.empty(voltage.size, precision.real_t)
    voltage_kick = charge * voltage

    kick_kernel(args=(dt,
                      dE,
                      np.int32(n_rf),
                      voltage_kick,
                      omega_rf,
                      phi_rf,
                      np.int32(dt.size),
                      precision.real_t(acceleration_kick)),
                block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


def drift(dt, dE, solver, t_rev, length_ratio, alpha_order, eta_0,
          eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy):
    """Apply the time drift function.

    Args:
        dt (_type_): _description_
        dE (_type_): _description_
        solver (_type_): _description_
        t_rev (_type_): _description_
        length_ratio (_type_): _description_
        alpha_order (_type_): _description_
        eta_0 (_type_): _description_
        eta_1 (_type_): _description_
        eta_2 (_type_): _description_
        alpha_0 (_type_): _description_
        alpha_1 (_type_): _description_
        alpha_2 (_type_): _description_
        beta (_type_): _description_
        energy (_type_): _description_
    """
    drift_kernel = GPU_DEV.mod.get_function("drift")

    solver = solver.decode('utf-8')
    if solver == "simple":
        solver = np.int32(0)
    elif solver == "legacy":
        solver = np.int32(1)
    else:
        solver = np.int32(2)

    if not isinstance(t_rev, float):
        t_rev = float(t_rev)

    drift_kernel(args=(dt,
                       dE,
                       solver,
                       precision.real_t(t_rev), precision.real_t(length_ratio),
                       precision.real_t(alpha_order), precision.real_t(eta_0),
                       precision.real_t(eta_1), precision.real_t(eta_2),
                       precision.real_t(alpha_0), precision.real_t(alpha_1),
                       precision.real_t(alpha_2),
                       precision.real_t(beta), precision.real_t(energy),
                       np.int32(dt.size)),
                 block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


def linear_interp_kick(dt, dE, voltage,
                       bin_centers, charge,
                       acceleration_kick):
    """An accelerated version of the kick function.

    Args:
        dt (_type_): _description_
        dE (_type_): _description_
        voltage (_type_): _description_
        bin_centers (_type_): _description_
        charge (_type_): _description_
        acceleration_kick (_type_): _description_
    """
    gm_linear_interp_kick_help = GPU_DEV.mod.get_function("lik_only_gm_copy")
    gm_linear_interp_kick_comp = GPU_DEV.mod.get_function("lik_only_gm_comp")

    assert dt.dtype == precision.real_t
    assert dE.dtype == precision.real_t
    assert voltage.dtype == precision.real_t
    assert bin_centers.dtype == precision.real_t

    macros = dt.size
    slices = bin_centers.size

    glob_vkick_factor = cp.empty(2 * (slices - 1), precision.real_t)
    gm_linear_interp_kick_help(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     precision.real_t(acceleration_kick),
                                     glob_vkick_factor),
                               grid=GPU_DEV.grid_size, block=GPU_DEV.block_size)

    gm_linear_interp_kick_comp(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     precision.real_t(acceleration_kick),
                                     glob_vkick_factor),
                               grid=GPU_DEV.grid_size, block=GPU_DEV.block_size)



def slice_beam(dt, profile, cut_left, cut_right):
    """Constant space slicing with a constant frame.

    Args:
        dt (_type_): _description_
        profile (_type_): _description_
        cut_left (_type_): _description_
        cut_right (_type_): _description_
    """
    sm_histogram = GPU_DEV.mod.get_function("sm_histogram")
    hybrid_histogram = GPU_DEV.mod.get_function("hybrid_histogram")



    assert dt.dtype == precision.real_t

    n_slices = profile.size
    profile.fill(0)

    if not isinstance(cut_left, float):
        cut_left = float(cut_left)
    if not isinstance(cut_right, float):
        cut_right = float(cut_right)

    if 4 * n_slices < GPU_DEV.attributes['MaxSharedMemoryPerBlock']:
        sm_histogram(args=(dt, profile, precision.real_t(cut_left),
                           precision.real_t(cut_right), np.uint32(n_slices),
                           np.uint32(dt.size)),
                     grid=GPU_DEV.grid_size, block=GPU_DEV.block_size, shared_mem=4 * n_slices)
    else:
        hybrid_histogram(args=(dt, profile, precision.real_t(cut_left),
                               precision.real_t(cut_right), np.uint32(n_slices),
                               np.uint32(dt.size), np.int32(
            GPU_DEV.attributes['MaxSharedMemoryPerBlock'] / 4)),
            grid=GPU_DEV.grid_size, block=GPU_DEV.block_size,
            shared_mem=GPU_DEV.attributes['MaxSharedMemoryPerBlock'])


def synchrotron_radiation(dE, U0, n_kicks, tau_z):
    """Track particles with SR only (without quantum excitation)

    Args:
        dE (_type_): _description_
        U0 (_type_): _description_
        n_kicks (_type_): _description_
        tau_z (_type_): _description_
    """
    synch_rad = GPU_DEV.mod.get_function("synchrotron_radiation")

    assert dE.dtype == precision.real_t

    synch_rad(args=(dE, precision.real_t(U0 / n_kicks), np.int32(dE.size),
                    precision.real_t(tau_z * n_kicks),
                    np.int32(n_kicks)),
              block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


def synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy):
    """Track particles with SR and quantum excitation.

    Args:
        dE (_type_): _description_
        U0 (_type_): _description_
        n_kicks (_type_): _description_
        tau_z (_type_): _description_
        sigma_dE (_type_): _description_
        energy (_type_): _description_
    """
    synch_rad_full = GPU_DEV.mod.get_function("synchrotron_radiation_full")

    assert dE.dtype == precision.real_t

    synch_rad_full(args=(dE, precision.real_t(U0 / n_kicks), np.int32(dE.size),
                         precision.real_t(sigma_dE),
                         precision.real_t(tau_z * n_kicks),
                         precision.real_t(energy), np.int32(n_kicks)),
                   block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


@cp.fuse(kernel_name='beam_phase_helper')
def __beam_phase_helper(bin_centers, profile, alpha, omega_rf, phi_rf):
    """Helper function, used by beam_phase

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        alpha (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_

    Returns:
        _type_: _description_
    """
    base = cp.exp(alpha * bin_centers) * profile
    a = omega_rf * bin_centers + phi_rf
    return base * cp.sin(a), base * cp.cos(a)


def beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, bin_size):
    """Beam phase measured at the main RF frequency and phase. The beam is
       convolved with the window function of the band-pass filter of the
       machine. The coefficients of sine and cosine components determine the
       beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
       phase is already w.r.t. the instantaneous RF phase.*

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        alpha (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_
        bin_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert bin_centers.dtype == precision.real_t
    assert profile.dtype == precision.real_t

    array1, array2 = __beam_phase_helper(
        bin_centers, profile, alpha, omega_rf, phi_rf)
    # due to the division, the bin_size is not needed
    scoeff = cp.trapz(array1, dx=1)
    ccoeff = cp.trapz(array2, dx=1)

    return float(scoeff / ccoeff)


@cp.fuse(kernel_name='beam_phase_fast_helper')
def __beam_phase_fast_helper(bin_centers, profile, omega_rf, phi_rf):
    """Helper function used by beam_phase_fast

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_

    Returns:
        _type_: _description_
    """
    arr = omega_rf * bin_centers + phi_rf
    return profile * cp.sin(arr), profile * cp.cos(arr)


def beam_phase_fast(bin_centers, profile, omega_rf, phi_rf, bin_size):
    """Simplified, faster variation of the beam_phase function

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_
        bin_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert bin_centers.dtype == precision.real_t
    assert profile.dtype == precision.real_t

    array1, array2 = __beam_phase_fast_helper(
        bin_centers, profile, omega_rf, phi_rf)
    # due to the division, the bin_size is not needed
    scoeff = cp.trapz(array1, dx=1)
    ccoeff = cp.trapz(array2, dx=1)

    return float(scoeff / ccoeff)
