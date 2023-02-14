import numpy as np
import cupy as cp
from ..utils import bmath as bm

grid_size, block_size = bm.gpuDev().grid_size, bm.gpuDev().block_size
kernels = bm.gpuDev().mod

# Load all required CUDA kernels
rf_volt_comp_kernel = kernels.get_function("rf_volt_comp")
kick_kernel = kernels.get_function("simple_kick")
drift_kernel = kernels.get_function("drift")
gm_linear_interp_kick_help = kernels.get_function("lik_only_gm_copy")
gm_linear_interp_kick_comp = kernels.get_function("lik_only_gm_comp")
gm_linear_interp_kick_drift_comp = kernels.get_function("lik_drift_only_gm_comp")
hybrid_histogram = kernels.get_function("hybrid_histogram")
sm_histogram = kernels.get_function("sm_histogram")
synch_rad = kernels.get_function("synchrotron_radiation")
synch_rad_full = kernels.get_function("synchrotron_radiation_full")


def rf_volt_comp(voltage, omega_rf, phi_rf, bin_centers):
    assert voltage.dtype == bm.precision.real_t
    assert omega_rf.dtype == bm.precision.real_t
    assert phi_rf.dtype == bm.precision.real_t
    assert bin_centers.dtype == bm.precision.real_t

    rf_voltage = cp.zeros(bin_centers.size, bm.precision.real_t)

    rf_volt_comp_kernel(args=(voltage, omega_rf, phi_rf, bin_centers,
              np.int32(voltage.size), np.int32(bin_centers.size), rf_voltage),
        block=block_size, grid=grid_size)
    return rf_voltage


def kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick):
    assert dt.dtype == bm.precision.real_t
    assert dE.dtype == bm.precision.real_t
    assert omega_rf.dtype == bm.precision.real_t
    assert phi_rf.dtype == bm.precision.real_t

    voltage_kick = cp.empty(voltage.size, bm.precision.real_t)
    voltage_kick = charge * voltage

    kick_kernel(args=(dt,
                      dE,
                      np.int32(n_rf),
                      voltage_kick,
                      omega_rf,
                      phi_rf,
                      np.int32(dt.size),
                      bm.precision.real_t(acceleration_kick)),
                block=block_size, grid=grid_size)  


def drift(dt, dE, solver, t_rev, length_ratio, alpha_order, eta_0,
              eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy):

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
                bm.precision.real_t(t_rev), bm.precision.real_t(length_ratio),
                bm.precision.real_t(alpha_order), bm.precision.real_t(eta_0),
                bm.precision.real_t(eta_1), bm.precision.real_t(eta_2),
                bm.precision.real_t(alpha_0), bm.precision.real_t(alpha_1),
                bm.precision.real_t(alpha_2),
                bm.precision.real_t(beta), bm.precision.real_t(energy),
                np.int32(dt.size)),
          block=block_size, grid=grid_size)


def linear_interp_kick(dt, dE, voltage,
                           bin_centers, charge,
                           acceleration_kick):
    assert dt.dtype == bm.precision.real_t
    assert dE.dtype == bm.precision.real_t
    assert voltage.dtype == bm.precision.real_t
    assert bin_centers.dtype == bm.precision.real_t

    macros = dt.size
    slices = bin_centers.size

    
    glob_vkick_factor = cp.empty(2*(slices - 1), bm.precision.real_t)
    gm_linear_interp_kick_help(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     bm.precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.real_t(acceleration_kick),
                                     glob_vkick_factor),
                               grid=grid_size, block=block_size)

    gm_linear_interp_kick_comp(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     bm.precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.real_t(acceleration_kick),
                                     glob_vkick_factor),
                               grid=grid_size, block=block_size)


def linear_interp_kick_drift(dt, dE, total_voltage, bin_centers, charge, acc_kick,
                                t_rev, length_ratio, alpha_order, eta_0, beta, energy):
    assert dt.dtype == bm.precision.real_t
    assert dE.dtype == bm.precision.real_t
    assert total_voltage.dtype == bm.precision.real_t
    assert bin_centers.dtype == bm.precision.real_t
    

    macros = dt.size
    slices = bin_centers.size

    glob_vkick_factor = cp.empty(2*(slices - 1), bm.precision.real_t)

    gm_linear_interp_kick_help(args=(dt,
                                     dE,
                                     total_voltage,
                                     bin_centers,
                                     bm.precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.real_t(acc_kick),
                                     glob_vkick_factor),
                               grid=grid_size, block=block_size)
    gm_linear_interp_kick_drift_comp(args=(dt,
                                           dE,
                                           total_voltage,
                                           bin_centers,
                                           bm.precision.real_t(charge),
                                           np.int32(slices),
                                           np.int32(macros),
                                           bm.precision.real_t(acc_kick),
                                           glob_vkick_factor,
                                           bm.precision.real_t(t_rev),
                                           bm.precision.real_t(length_ratio),
                                           bm.precision.real_t(eta_0),
                                           bm.precision.real_t(beta),
                                           bm.precision.real_t(energy)),
                                     grid=grid_size, block=block_size)


def slice(dt, profile, cut_left, cut_right):

    assert dt.dtype == bm.precision.real_t
    
    n_slices = profile.size
    profile.fill(0)

    if not isinstance(cut_left, float):
        cut_left = float(cut_left)
    if not isinstance(cut_right, float):
        cut_right = float(cut_right)

    if 4*n_slices < bm.gpuDev().attributes['MaxSharedMemoryPerBlock']:
        sm_histogram(args=(dt, profile, bm.precision.real_t(cut_left),
                           bm.precision.real_t(cut_right), np.uint32(n_slices),
                           np.uint32(dt.size)),
                     grid=grid_size, block=block_size, shared_mem=4*n_slices)
    else:
        hybrid_histogram(args=(dt, profile, bm.precision.real_t(cut_left),
                               bm.precision.real_t(cut_right), np.uint32(n_slices),
                               np.uint32(dt.size), np.int32(
            bm.gpuDev().attributes['MaxSharedMemoryPerBlock']/4)),
            grid=grid_size, block=block_size,
            shared_mem=bm.gpuDev().attributes['MaxSharedMemoryPerBlock'])


def synchrotron_radiation(dE, U0, n_kicks, tau_z):
    assert dE.dtype == bm.precision.real_t
   
    synch_rad(args=(dE, bm.precision.real_t(U0/n_kicks), np.int32(dE.size),
                    bm.precision.real_t(tau_z * n_kicks),
                    np.int32(n_kicks)),
              block=block_size, grid=grid_size)


def synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy):
    assert dE.dtype == bm.precision.real_t
    
    synch_rad_full(args=(dE, bm.precision.real_t(U0/n_kicks), np.int32(dE.size),
                         bm.precision.real_t(sigma_dE),
                         bm.precision.real_t(tau_z * n_kicks),
                         bm.precision.real_t(energy), np.int32(n_kicks)),
                   block=block_size, grid=grid_size)


@cp.fuse(kernel_name='beam_phase_helper')
def __beam_phase_helper(bin_centers, profile, alpha, omega_rf, phi_rf):
    base = cp.exp(alpha * bin_centers) * profile
    a = omega_rf * bin_centers + phi_rf
    return base * cp.sin(a), base * cp.cos(a)


def beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, bin_size):
    assert bin_centers.dtype == bm.precision.real_t
    assert profile.dtype == bm.precision.real_t

    array1, array2 = __beam_phase_helper(
        bin_centers, profile, alpha, omega_rf, phi_rf)
    # due to the division, the bin_size is not needed
    scoeff = cp.trapz(array1, dx=1)
    ccoeff = cp.trapz(array2, dx=1)

    return float(scoeff / ccoeff)


@cp.fuse(kernel_name='beam_phase_fast_helper')
def __beam_phase_fast_helper(bin_centers, profile, omega_rf, phi_rf):
    a = omega_rf * bin_centers + phi_rf
    return profile * cp.sin(a), profile * cp.cos(a)


def beam_phase_fast(bin_centers, profile, omega_rf, phi_rf, bin_size):
    assert bin_centers.dtype == bm.precision.real_t
    assert profile.dtype == bm.precision.real_t

    array1, array2 = __beam_phase_fast_helper(
        bin_centers, profile, omega_rf, phi_rf)
    # due to the division, the bin_size is not needed
    scoeff = cp.trapz(array1, dx=1)
    ccoeff = cp.trapz(array2, dx=1)

    return float(scoeff / ccoeff)


# cugradient = kernels.get_function("cugradient")

# def convolve(signal, kernel, mode='full', result=None):
#     if mode != 'full':
#         # ConvolutionError
#         raise RuntimeError('[convolve] Only full mode is supported')
#     if result is None:
#         result = cp.empty(len(signal) + len(kernel) - 1,
#                           dtype=bm.precision.real_t)
#     result = bm.irfft(bm.rfft(signal) * bm.rfft(kernel))
#     return result


# def interp(x, xp, yp, left=None, right=None, result=None):
#     cuinterp = kernels.get_function("cuinterp")

#     if not left:
#         left = yp[0]
#     if not right:
#         right = yp[-1]
#     if result is None:
#         result = cp.empty(x.size, bm.precision.real_t)

#     cuinterp(args=(x, np.int32(x.size),
#                    xp, np.int32(xp.size),
#                    yp, result,
#                    bm.precision.real_t(left), bm.precision.real_t(right)),
#              block=block_size, grid=grid_size)
#     return result
