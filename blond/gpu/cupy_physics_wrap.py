import numpy as np
import cupy as cp
# from ..gpu import block_size, grid_size
# from ..gpu.cupy_array import get_gpuarray
from ..utils import bmath as bm

grid_size, block_size = bm.gpuDev().grid_size, bm.gpuDev().block_size
kernels = bm.gpuDev().mod


def gpu_rf_volt_comp(voltage, omega_rf, phi_rf, bin_centers):
    assert voltage.dtype == bm.precision.real_t
    assert omega_rf.dtype == bm.precision.real_t
    assert phi_rf.dtype == bm.precision.real_t
    assert bin_centers.dtype == bm.precision.real_t

    rf_voltage = cp.empty(bin_centers.size, bm.precision.real_t)

    rvc = kernels.get_function("rf_volt_comp")

    rvc(args=(voltage, omega_rf, phi_rf, bin_centers,
              np.int32(voltage.size), np.int32(bin_centers.size), rf_voltage),
        block=block_size, grid=grid_size)
    return rf_voltage


def gpu_kick(dt, dE, voltage, omega_rf, phi_rf, charge, n_rf, acceleration_kick):
    assert dt.dtype == bm.precision.real_t
    assert dE.dtype == bm.precision.real_t
    assert omega_rf.dtype == bm.precision.real_t
    assert phi_rf.dtype == bm.precision.real_t

    kick_kernel = kernels.get_function("simple_kick")

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
                block=block_size, grid=grid_size)  # , time_kernel=True)


def gpu_drift(dt, dE, solver, t_rev, length_ratio, alpha_order, eta_0,
              eta_1, eta_2, alpha_0, alpha_1, alpha_2, beta, energy):

    solver = solver.decode('utf-8')
    if solver == "simple":
        solver = np.int32(0)
    elif solver == "legacy":
        solver = np.int32(1)
    else:
        solver = np.int32(2)

    drift = kernels.get_function("drift")

    drift(args=(dt,
                dE,
                solver,
                bm.precision.real_t(t_rev), bm.precision.real_t(length_ratio),
                bm.precision.real_t(alpha_order), bm.precision.real_t(eta_0),
                bm.precision.real_t(eta_1), bm.precision.real_t(eta_2),
                bm.precision.real_t(alpha_0), bm.precision.real_t(alpha_1),
                bm.precision.real_t(alpha_2),
                bm.precision.real_t(beta), bm.precision.real_t(energy),
                np.int32(dt.size)),
          block=block_size, grid=grid_size)  # , time_kernel=True)


def gpu_linear_interp_kick(dt, dE, voltage,
                           bin_centers, charge,
                           acceleration_kick):
    assert dt.dtype == bm.precision.real_t
    assert dE.dtype == bm.precision.real_t
    assert voltage.dtype == bm.precision.real_t
    assert bin_centers.dtype == bm.precision.real_t

    macros = dt.size
    slices = bin_centers.size

    gm_linear_interp_kick_help = kernels.get_function("lik_only_gm_copy")
    gm_linear_interp_kick_comp = kernels.get_function("lik_only_gm_comp")

    voltage_kick = cp.empty(slices - 1, bm.precision.real_t)
    dev_factor = cp.empty(slices - 1, bm.precision.real_t)
    gm_linear_interp_kick_help(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     bm.precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.real_t(acceleration_kick),
                                     voltage_kick,
                                     dev_factor),
                               grid=grid_size, block=block_size)  # ,time_kernel=True)

    gm_linear_interp_kick_comp(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     bm.precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.real_t(acceleration_kick),
                                     voltage_kick,
                                     dev_factor),
                               grid=grid_size, block=block_size)  # ,time_kernel=True)


def gpu_linear_interp_kick_drift(dt, dE, total_voltage, bin_centers, charge, acc_kick,
                                 solver, t_rev, length_ratio, alpha_order, eta_0, eta_1,
                                 eta_2, beta, energy):
    assert dt.dtype == bm.precision.real_t
    assert dE.dtype == bm.precision.real_t
    assert total_voltage.dtype == bm.precision.real_t
    assert bin_centers.dtype == bm.precision.real_t
    gm_linear_interp_kick_drift_comp = kernels.get_function(
        "lik_drift_only_gm_comp")
    gm_linear_interp_kick_help = kernels.get_function("lik_only_gm_copy")

    macros = dt.size
    slices = bin_centers.size

    voltage_kick = cp.empty(slices - 1, bm.precision.real_t)
    factor = cp.empty(slices - 1, bm.precision.real_t)

    gm_linear_interp_kick_help(args=(dt,
                                     dE,
                                     total_voltage,
                                     bin_centers,
                                     bm.precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     bm.precision.real_t(acc_kick),
                                     voltage_kick,
                                     factor),
                               grid=grid_size, block=block_size)  # ,time_kernel=True)
    gm_linear_interp_kick_drift_comp(args=(dt,
                                           dE,
                                           total_voltage,
                                           bin_centers,
                                           bm.precision.real_t(charge),
                                           np.int32(slices),
                                           np.int32(macros),
                                           bm.precision.real_t(acc_kick),
                                           voltage_kick,
                                           factor,
                                           bm.precision.real_t(t_rev),
                                           bm.precision.real_t(length_ratio),
                                           bm.precision.real_t(eta_0),
                                           bm.precision.real_t(beta),
                                           bm.precision.real_t(energy)),
                                     grid=grid_size, block=block_size)  # ,time_kernel=True)


def gpu_slice(dt, profile, cut_left, cut_right):

    assert dt.dtype == bm.precision.real_t
    hybrid_histogram = kernels.get_function("hybrid_histogram")
    sm_histogram = kernels.get_function("sm_histogram")

    n_slices = profile.size
    profile.fill(0)
    if 4*n_slices < bm.gpuDev().attributes['MaxSharedMemoryPerBlock']:
        sm_histogram(args=(dt, profile, bm.precision.real_t(cut_left),
                           bm.precision.real_t(cut_right), np.uint32(n_slices),
                           np.uint32(dt.size)),
                     grid=grid_size, block=block_size, shared_mem=4*n_slices)  # , time_kernel=True)
    else:
        hybrid_histogram(args=(dt, profile, bm.precision.real_t(cut_left),
                               bm.precision.real_t(
                                   cut_right), np.uint32(n_slices),
                               np.uint32(dt.size), np.int32(
            bm.gpuDev().attributes['MaxSharedMemoryPerBlock']/4)),
            grid=grid_size, block=block_size,
            shared_mem=bm.gpuDev().attributes['MaxSharedMemoryPerBlock'])  # , time_kernel=True)
    return profile


def gpu_synchrotron_radiation(dE, U0, n_kicks, tau_z):
    assert dE.dtype == bm.precision.real_t
    synch_rad = kernels.get_function("synchrotron_radiation")

    synch_rad(args=(dE, bm.precision.real_t(U0), np.int32(dE.size),
                    bm.precision.real_t(tau_z),
                    np.int32(n_kicks)),
              block=block_size,
              grid=(bm.gpuDev().attributes['MultiProcessorCount'], 1, 1))


def gpu_synchrotron_radiation_full(dE, U0, n_kicks, tau_z, sigma_dE, energy):
    assert dE.dtype == bm.precision.real_t
    synch_rad_full = kernels.get_function("synchrotron_radiation_full")

    synch_rad_full(args=(dE, bm.precision.real_t(U0), np.int32(dE.size),
                         bm.precision.real_t(
                             sigma_dE), bm.precision.real_t(energy),
                         np.int32(n_kicks), np.int32(1)),
                   block=block_size, grid=grid_size)


def gpu_beam_phase(bin_centers, profile, alpha, omega_rf, phi_rf, bin_size):
    assert bin_centers.dtype == bm.precision.real_t
    assert profile.dtype == np.int32
    # assert omega_rf.dtype == bm.precision.real_t
    # assert phi_rf.dtype == bm.precision.real_t

    beam_phase_v2 = kernels.get_function("beam_phase_v2")
    beam_phase_sum = kernels.get_function("beam_phase_sum")

    array1 = cp.empty(bin_centers.size, bm.precision.real_t)
    array2 = cp.empty(bin_centers.size, bm.precision.real_t)

    dev_scoeff = cp.empty(1, bm.precision.real_t)
    dev_coeff = cp.empty(1, bm.precision.real_t)

    beam_phase_v2(args=(bin_centers, profile,
                        bm.precision.real_t(alpha),
                        bm.precision.real_t(omega_rf),
                        bm.precision.real_t(phi_rf),
                        bm.precision.real_t(bin_size),
                        array1, array2, np.int32(bin_centers.size)),
                  block=block_size, grid=grid_size)

    beam_phase_sum(args=(array1, array2, dev_scoeff, dev_coeff,
                         np.int32(bin_centers.size)), block=(512, 1, 1),
                   grid=(1, 1, 1))  # , time_kernel=True)
    to_ret = dev_scoeff[0].get()
    return to_ret

# def gpu_dumb_slice(dev_dt, dev_n_macroparticles, cut_left, cut_right):
#     """This is only here for benchmarks"""
#     dumb_histogram = kernels.get_function("histogram")

#     n_slices = dev_n_macroparticles.size
#     dev_n_macroparticles.fill(0)
#     dumb_histogram( args = (dev_dt, dev_n_macroparticles, bm.precision.real_t(cut_left),
#                    bm.precision.real_t(cut_right), np.uint32(n_slices),
#                    np.uint32(dev_dt.size)),
#                    grid=grid_size, block=block_size)#, time_kernel=True)
#     return dev_n_macroparticles