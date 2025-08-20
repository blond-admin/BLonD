import numpy as np
from unittest.mock import Mock
import json
import matplotlib.pyplot as plt
from blond3 import WakeField
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import (
    AnalyticSingleTurnResonatorSolver,
)
from scipy.constants import e, c
import cst.results


def functest():
    # CST settings: open BC at z, magnetic symmetry planes, ec1 parameters from https://cds.cern.ch/record/533324, f_cutoff = 2.5GHz, WF length = 5m
    # create bunch with sigma of 40mm --> set this as profile, convolute with potential to get wake for the first 5 meters
    sigma_z = 40e-3
    # R_over_Q = np.array([51.94, 13.7312, 0.0915, 2.638805, 2.132499, 2.712645, 4.064])
    # q_factor = np.array([4.15e8, 4.416e5, 38791, 70.629, 59.224, 35.6335, 23.2348])
    # freq = np.array([1.30192e9, 2.4508e9, 2.70038e9, 3.0675e9, 3.083e9, 3.34753e9, 3.42894e9])
    with open(
        "../../unittests_blond3/physics/impedances/resources/TESLA_until_4.5GHz_CST_WG_ports_finer_5_PM.json",
        "r",
        encoding="utf-8",
    ) as cst_modes_EM_file:
        cst_modes_dict = json.load(cst_modes_EM_file)

    freq, q_factor, R_over_Q = [], [], []
    for mode in cst_modes_dict:
        if cst_modes_dict[mode]["Qext"] < 200:
            continue
        freq.append(cst_modes_dict[mode]["freq"])
        q_factor.append(cst_modes_dict[mode]["Qext"])
        R_over_Q.append(cst_modes_dict[mode]["R/Q_||"])
    freq = np.array(freq)
    q_factor = np.array(q_factor)
    R_over_Q = np.array(R_over_Q)

    R_shunt = R_over_Q * q_factor

    res = Resonators(
        quality_factors=q_factor, shunt_impedances=R_shunt, center_frequencies=freq
    )
    analy = AnalyticSingleTurnResonatorSolver()
    csts_stupid_number = 8.548921333333334
    # csts_stupid_number = 8.54
    bunch_time = np.linspace(
        -sigma_z * csts_stupid_number / c, csts_stupid_number * sigma_z / c, 2**12
    )
    bunch = np.exp(-0.5 * (bunch_time / (sigma_z / c)) ** 2)

    analy._parent_wakefield = Mock(WakeField)
    analy._parent_wakefield.profile.cut_left = -sigma_z * csts_stupid_number / c
    analy._parent_wakefield.profile.cut_right = csts_stupid_number * sigma_z / c
    analy._parent_wakefield.profile.bin_size = bunch_time[1] - bunch_time[0]
    analy._parent_wakefield.profile.hist_x = bunch_time
    analy._parent_wakefield.profile.hist_y = bunch / np.sum(bunch)

    analy._parent_wakefield.sources = (res,)

    beam = Mock(BeamBaseClass)
    beam.n_particles = int(1e3)
    beam.particle_type.charge = 1 / e
    beam.n_macroparticles_partial.return_value = int(1e3)
    # n_particles == n_macroparticles, integrated bunch is 1 --> all normalized to 1C

    analy._wake_pot_vals_needs_update = True

    calced_voltage = analy.calc_induced_voltage(beam=beam)

    # cst_result = np.load("../../unittests_blond3/physics/impedances/resources/TESLA_ec1_WF_pot.npz")
    # time_axis = cst_result["s_axis"] / c
    # pot_axis = cst_result["pot_axis"] * 1e12  # pC

    project = cst.results.ProjectFile(
        r"D:\CB_2\BLonD_verification\TESLA_ec1_WF-WG_ports.cst", allow_interactive=True
    )
    # time_axis = np.array(project.get_3d().get_result_item("1D Results\\saved\\Z_analy_id_tb").get_xdata()) / 1e3 / c
    # pot_axis = np.array(project.get_3d().get_result_item("1D Results\\saved\\Z_analy_id_tb").get_ydata()) * 1e12
    time_axis = (
        np.array(
            project.get_3d()
            .get_result_item(
                "1D Results\\Particle Beams\\ParticleBeam1\\Wake potential\\Z"
            )
            .get_xdata()
        )
        / 1e3
        / c
    )
    pot_axis = (
        np.array(
            project.get_3d()
            .get_result_item(
                "1D Results\\Particle Beams\\ParticleBeam1\\Wake potential\\Z"
            )
            .get_ydata()
        )
        * 1e12
    )

    plt.plot(
        np.interp(bunch_time, time_axis, pot_axis),
        label="wake_pot",
    )
    plt.plot(calced_voltage, label="calced voltage")
    # plt.plot(np.interp(bunch_time, time_axis, pot_axis)[:len(calced_voltage)], label="wake_pot")
    # plt.plot(calced_voltage[:len(calced_voltage)], label="calced voltage")
    plt.legend()
    plt.show()

def convolution_scratch(pad=True):
    profile = np.zeros(21)
    profile[10] = 1
    profile /= np.sum(profile)
    starttime = 50e-9
    stoptime = 52e-9
    profile_time_orig = np.linspace(starttime, stoptime, num=21, endpoint=True)

    extended_profile = np.pad(profile, (0, 10))
    extended_time_orig = np.linspace(starttime, stoptime + 1e-9, num=31, endpoint=True)

    bin_size = 1e-10
    if pad:
        left_extend = len(profile_time_orig) + int(profile_time_orig[0] / bin_size)
        right_extend = len(profile_time_orig) - left_extend - 1

        profile_time = np.linspace(
            profile_time_orig[0]
            - left_extend * bin_size,
            profile_time_orig[-1]
            + right_extend * bin_size,
            int(
                len(profile_time_orig) + left_extend + right_extend
            ),
            endpoint=True,
        )
        shift_prof = 0

        left_extend = len(extended_time_orig) + int(extended_time_orig[0] / bin_size)
        right_extend = len(extended_time_orig) - left_extend - 1
        # left_extend = np.ceil((len(extended_time_orig) - 1) / 2)
        # right_extend = np.floor((len(extended_time_orig) - 1) / 2)

        extended_time = np.linspace(
            extended_time_orig[0]
            - left_extend * bin_size,
            extended_time_orig[-1]
            + right_extend * bin_size,
            int(
                len(extended_time_orig) + left_extend + right_extend
            ),
            endpoint=True,
        )
        shift_prof_ext = 0

    else:
        left_extend = 10
        right_extend = 0
        profile_time = np.linspace(
            profile_time_orig[0]
            - left_extend * bin_size,
            profile_time_orig[-1]
            + right_extend * bin_size,
            int(
                len(profile_time_orig) + left_extend + right_extend
            ),
            endpoint=True,
        )
        left_extend = 10
        right_extend = 0
        extended_time = np.linspace(
            extended_time_orig[0]
            - left_extend * bin_size,
            extended_time_orig[-1]
            + right_extend * bin_size,
            int(
                len(extended_time_orig) + left_extend + right_extend
            ),
            endpoint=True,
        )
    extended_time[
        np.abs(extended_time)
        <= 1e-10
        * np.finfo(float).eps
        * len(extended_time)
        ] = 0.0
    profile_time[
        np.abs(profile_time)
        <= 1e-10
        * np.finfo(float).eps
        * len(profile_time)
        ] = 0.0

    res = Resonators(center_frequencies=1e9, quality_factors=1e10, shunt_impedances=1e5)
    extended_kernel = res.get_wake(extended_time)
    wake = res.get_wake(profile_time)
    if pad:
        voltage = np.convolve(profile, wake, mode="valid")
        extended_voltage = np.convolve(extended_profile, extended_kernel, mode="valid")
        plt.plot(voltage[shift_prof:shift_prof+len(profile)])
        plt.plot(extended_voltage[shift_prof_ext:shift_prof_ext+len(extended_profile)], ls="--")
        # plt.xlim(5, 15)
        plt.show()
        plt.plot(voltage)
        plt.plot(extended_voltage, ls="--")
        # plt.xlim(5, 15)
        plt.show()
    else:
        voltage = np.convolve(profile[::-1], wake, mode="full")
        extended_voltage = np.convolve(extended_profile[::-1], extended_kernel, mode="full")
        plt.plot(voltage[21:21 + len(profile)])
        plt.plot(extended_voltage[31:31 + len(extended_profile)], ls="--")
        plt.show()


if __name__ == "__main__":
    functest()
    # convolution_scratch()
