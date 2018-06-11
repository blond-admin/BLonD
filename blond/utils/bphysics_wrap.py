'''
BLonD physics wrapper functions

@author Konstantinos Iliakis
@date 20.10.2017
'''

import ctypes as ct
import numpy as np
# from setup_cpp import libblondphysics as __lib
from .. import libblond as __lib


def __getPointer(x):
    return x.ctypes.data_as(ct.c_void_p)


def __getLen(x):
    return ct.c_int(len(x))


def rf_volt_comp(voltages, omega_rf, phi_rf, ring):
    # voltages = np.ascontiguousarray(ring.voltage[:, ring.counter[0]])
    # omega_rf = np.ascontiguousarray(ring.omega_rf[:, ring.counter[0]])
    # phi_rf = np.ascontiguousarray(ring.phi_rf[:, ring.counter[0]])

    rf_voltage = np.zeros(len(ring.profile.bin_centers))

    __lib.rf_volt_comp(__getPointer(voltages),
                       __getPointer(omega_rf),
                       __getPointer(phi_rf),
                       __getPointer(ring.profile.bin_centers),
                       __getLen(voltages),
                       __getLen(rf_voltage),
                       __getPointer(rf_voltage))
    return rf_voltage


def kick(ring, dt, dE, turn):
    voltage_kick = np.ascontiguousarray(ring.charge*ring.voltage[:, turn])
    omegarf_kick = np.ascontiguousarray(ring.omega_rf[:, turn])
    phirf_kick = np.ascontiguousarray(ring.phi_rf[:, turn])

    __lib.kick(__getPointer(dt),
               __getPointer(dE),
               ct.c_int(ring.n_rf),
               __getPointer(voltage_kick),
               __getPointer(omegarf_kick),
               __getPointer(phirf_kick),
               __getLen(dt),
               ct.c_double(ring.acceleration_kick[turn]))


def drift(ring, dt, dE, turn):

    __lib.drift(__getPointer(dt),
                __getPointer(dE),
                ct.c_char_p(ring.solver),
                ct.c_double(ring.t_rev[turn]),
                ct.c_double(ring.length_ratio),
                ct.c_double(ring.alpha_order),
                ct.c_double(ring.eta_0[turn]),
                ct.c_double(ring.eta_1[turn]),
                ct.c_double(ring.eta_2[turn]),
                ct.c_double(ring.rf_params.beta[turn]),
                ct.c_double(ring.rf_params.energy[turn]),
                __getLen(dt))


def linear_interp_kick(ring, dt, dE, turn):
    __lib.linear_interp_kick(__getPointer(dt),
                             __getPointer(dE),
                             __getPointer(ring.total_voltage),
                             __getPointer(ring.profile.bin_centers),
                             ct.c_double(ring.beam.Particle.charge),
                             ct.c_int(ring.profile.n_slices),
                             ct.c_int(ring.beam.n_macroparticles),
                             ct.c_double(ring.acceleration_kick[turn]))


def linear_interp_time_translation(ring, dt, dE, turn):
    pass


def slice(profile):
    __lib.histogram(__getPointer(profile.Beam.dt),
                    __getPointer(profile.n_macroparticles),
                    ct.c_double(profile.cut_left),
                    ct.c_double(profile.cut_right),
                    ct.c_int(profile.n_slices),
                    ct.c_int(profile.Beam.n_macroparticles))


def slice_smooth(profile):
    __lib.smooth_histogram(__getPointer(profile.Beam.dt),
                           __getPointer(profile.n_macroparticles),
                           ct.c_double(profile.cut_left),
                           ct.c_double(profile.cut_right),
                           ct.c_int(profile.n_slices),
                           ct.c_int(profile.Beam.n_macroparticles))


def music_track(music):
    __lib.music_track(__getPointer(music.beam.dt),
                      __getPointer(music.beam.dE),
                      __getPointer(music.induced_voltage),
                      __getPointer(music.array_parameters),
                      __getLen(music.beam.dt),
                      ct.c_double(music.alpha),
                      ct.c_double(music.omega_bar),
                      ct.c_double(music.const),
                      ct.c_double(music.coeff1),
                      ct.c_double(music.coeff2),
                      ct.c_double(music.coeff3),
                      ct.c_double(music.coeff4))


def music_track_multiturn(music):
    __lib.music_track_multiturn(__getPointer(music.beam.dt),
                                __getPointer(music.beam.dE),
                                __getPointer(music.induced_voltage),
                                __getPointer(music.array_parameters),
                                __getLen(music.beam.dt),
                                ct.c_double(music.alpha),
                                ct.c_double(music.omega_bar),
                                ct.c_double(music.const),
                                ct.c_double(music.coeff1),
                                ct.c_double(music.coeff2),
                                ct.c_double(music.coeff3),
                                ct.c_double(music.coeff4))


def synchrotron_radiation(SyncRad, turn):
    __lib.synchrotron_radiation(
        __getPointer(SyncRad.beam.dE),
        ct.c_double(SyncRad.U0 / SyncRad.n_kicks),
        ct.c_int(SyncRad.beam.n_macroparticles),
        ct.c_double(SyncRad.tau_z * SyncRad.n_kicks),
        ct.c_int(SyncRad.n_kicks))


def synchrotron_radiation_full(SyncRad, turn):
    __lib.synchrotron_radiation_full(
        __getPointer(SyncRad.beam.dE),
        ct.c_double(SyncRad.U0 / SyncRad.n_kicks),
        ct.c_int(SyncRad.beam.n_macroparticles),
        ct.c_double(SyncRad.sigma_dE),
        ct.c_double(SyncRad.tau_z * SyncRad.n_kicks),
        ct.c_double(SyncRad.general_params.energy[0, turn]),
        __getPointer(SyncRad.random_array),
        ct.c_int(SyncRad.n_kicks))
