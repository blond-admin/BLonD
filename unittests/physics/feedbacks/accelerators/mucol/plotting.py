"""
Plotting helpers for the muon-collider RCS cavity-feedback driver.

Extracted from ``mucol_cav_fdbk.py``: interactive/diagnostic plots for the
observations recorded by :func:`mucol_cav_fdbk.setup_and_run` (bunch
statistics, induced-voltage vs feedback-voltage comparisons and the
generator power / antenna voltage of a PI-feedback run). Not a test module.
"""

import matplotlib.pyplot as plt
import numpy as np

from blond.generals.cupy.no_cupy_import import copy_to_cpu


def plot_generator_power_and_voltage(
    cav_fdbk_obs_list,
    R_over_Q,
    Q_L,
    station_index=0,
    filename=None,
    show=True,
):
    """
    Plot the generator power and antenna voltage of a real simulation run.

    Real-simulation analogue of the diagnostic plots in
    ``test_generator_current_pi_feedback.py``. It reads the per-profile
    (fine-grid) generator current, antenna voltage and beam current
    recorded by an :class:`IQCavityFeedbackObservation` (one per RF
    station) and shows, turn by turn: the klystron forward power, the peak
    bunch current, and the antenna-voltage swing the bunch induces across
    the profile.

    The fine grid is used rather than the coarse grid because the latter is
    NaN-padded to a fixed length and its single-sample bunch values produce
    misleading peaks; the fine grid resolves the actual bunch cleanly.

    Use together with ``setup_and_run(..., use_pi_feedback=True)`` and the
    ``cav_fdbk_obs_list`` it returns.

    Parameters
    ----------
    cav_fdbk_obs_list
        List of :class:`IQCavityFeedbackObservation`, one per RF station.
    R_over_Q
        Geometric shunt impedance of the cavity [Ohm].
    Q_L
        Loaded quality factor of the cavity.
    station_index
        Which station's observation to plot.
    filename
        If given, save the figure to this path.
    show
        If True, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    obs = cav_fdbk_obs_list[station_index]
    i_gen = copy_to_cpu(obs.i_gen_fine)
    v_ant = copy_to_cpu(obs.v_ant_fine)
    i_beam = copy_to_cpu(obs.i_beam_fine)
    turns = np.arange(i_gen.shape[0])

    # Per fine-grid sample generator power, P = 0.5 (R/Q) Q_L |I_gen|^2.
    power = 0.5 * R_over_Q * Q_L * np.abs(i_gen) ** 2
    # Klystron forward power (steady level over the profile).
    gen_power = np.median(power, axis=1)
    # Peak bunch current and the beam-loading voltage swing along the bunch.
    peak_i_beam = np.abs(i_beam).max(axis=1)
    v_swing = np.abs(v_ant).max(axis=1) - np.abs(v_ant).min(axis=1)

    fig, (ax_v, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    fig.suptitle(
        "Real RCS simulation: generator power & antenna voltage (PI feedback)"
    )

    ax_v.plot(turns, v_swing / 1e6, color="C0", marker="o")
    ax_v.set_ylabel("antenna-voltage swing\nover the bunch [MV]")

    line_power = ax_p.plot(
        turns,
        gen_power / 1e3,
        color="C3",
        marker="o",
        label="generator power",
    )[0]
    ax_p.set_ylabel("generator power [kW]", color="C3")
    ax_p.tick_params(axis="y", labelcolor="C3")
    ax_p.set_xlabel("turn")
    ax_pb = ax_p.twinx()
    line_beam = ax_pb.plot(
        turns,
        peak_i_beam,
        color="C2",
        ls=":",
        marker="^",
        label="peak beam current",
    )[0]
    ax_pb.set_ylabel("peak beam current [A]")
    ax_p.legend(handles=[line_power, line_beam], loc="best")

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=120)
    if show:
        plt.show()
    return fig


def plot_results(bunch_obs_list, n_turns_list, ind_volt_obs_list):
    """
    Plot the bunch length (sigma_dt) for the MTW and feedback runs.

    Parameters
    ----------
    bunch_obs_list
        List of bunch observations; index 0 is MTW, index 1 the feedback run.
    n_turns_list
        Number of turns per run (unused in the plot itself).
    ind_volt_obs_list
        List of induced-voltage observations (unused in the plot itself).
    """
    plt.title("sigma_dt")
    plt.plot(bunch_obs_list[0].sigma_dt, label="MTW")
    plt.plot(bunch_obs_list[1].sigma_dt, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("rms_emittance")
    plt.plot(bunch_obs_list[0].rms_emittance, label="MTW")
    plt.plot(bunch_obs_list[1].rms_emittance, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("sigma_dE")
    plt.plot(bunch_obs_list[0].sigma_dE, label="MTW")
    plt.plot(bunch_obs_list[1].sigma_dE, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("mean_dt")
    plt.plot(bunch_obs_list[0].mean_dt, label="MTW")
    plt.plot(bunch_obs_list[1].mean_dt, ls="--", label="fdbk")
    plt.legend()
    plt.show()

    plt.title("mean_dE")
    plt.plot(bunch_obs_list[0].mean_dE, label="MTW")
    plt.plot(bunch_obs_list[1].mean_dE, ls="--", label="fdbk")
    plt.legend()
    plt.show()


def plot_ind_volt_cav_fdbk_voltage(
    ind_volt_obs_list, cav_fdbk_obs_list, n_turns_in: int, n_stations: int
):
    """
    Plot induced voltage against the cavity-feedback voltage per station.

    Parameters
    ----------
    ind_volt_obs_list
        List of induced-voltage observations.
    cav_fdbk_obs_list
        List of cavity-feedback observations.
    n_turns_in
        Number of turns that were simulated.
    n_stations
        Number of RF stations around the ring.
    """
    # plt.clf()
    # fix, ax = plt.subplots()
    # plt.title("ind_volt vs fdbk_kick")
    #
    # ax.plot(
    #     ind_volt_obs_list[0][0].induced_voltage[0], ls="--", label="ind_volt"
    # )
    # # ax.plot(cav_fdbk_obs_list[1][0].v_corr[0] * 30e6, label="rel_volt correction")
    # ax.plot(
    #     np.abs(cav_fdbk_obs_list[1][0].v_ant_fine[0]), label="abs v_ant_fine"
    # )
    # ax.plot(
    #     np.real(cav_fdbk_obs_list[1][0].v_ant_fine[0]), label="real v_ant_fine"
    # )

    # ax2 = ax.twinx()
    # ax2.plot(ind_volt_obs_list[0][0].beam_profile[0], label="beam profile")
    # ax2.plot(
    #     cav_fdbk_obs_list[1][0].v_corr[0] * 30e6, label="rel_volt correction"
    # )
    # ax2.plot(cav_fdbk_obs_list[1][0].phi_corr[0], label="rel_volt correction")
    #
    # plt.legend()
    # plt.show(block=False)
    #
    # fix, ax = plt.subplots()
    # plt.title("coarse")
    #
    # ax.plot(cav_fdbk_obs_list[1][0].v_ant_coarse[0], label="ind_volt")
    #
    # ax2 = ax.twinx()
    # ax2.plot(ind_volt_obs_list[0][0].beam_profile[0], label="beam profile")
    # # ax2.plot(cav_fdbk_obs_list[1][0].v_corr[0] * 30e6, label="rel_volt correction")
    #
    # plt.legend()
    # plt.show(block=False)

    #
    # turn_idx = 1
    # fig, ax = plt.subplots(2, 2, sharex=True)
    # ax[0, 0].set_title("through stations")
    # for idx in range(3):
    #     clr = ax[0, 0]._get_lines.get_next_color()
    #     ax[0, 0].plot(
    #         ind_volt_obs_list[0][idx].total_voltage[turn_idx],
    #         color=clr,
    #         label="MTW",
    #     )
    #     ax[0, 0].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].kick_voltage_fine[turn_idx]),
    #         ls="--",
    #         color=clr,
    #         label="real fdbk",
    #     )
    #     cavity_voltage_raw = (
    #         ind_volt_obs_list[0][idx].total_voltage[turn_idx]
    #         - ind_volt_obs_list[0][idx].induced_voltage[turn_idx]
    #     )
    #     ax[0, 1].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].kick_voltage_fine[turn_idx])
    #         - cavity_voltage_raw,
    #         ls="--",
    #         color=clr,
    #         label="real fdbk",
    #     )
    #     ax[0, 1].plot(
    #         np.imag(cav_fdbk_obs_list[1][idx].kick_voltage_fine[turn_idx]),
    #         # - cavity_voltage_raw,
    #         ls=":",
    #         color=clr,
    #         label="real fdbk",
    #     )
    #     ax[0, 1].plot(
    #         ind_volt_obs_list[0][idx].induced_voltage[turn_idx],
    #         color=clr,
    #         label="MTW",
    #     )
    #     ax[1, 0].set_title("v_corr")
    #     ax[1, 0].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].v_corr[turn_idx]),
    #         color=clr,
    #         label="v_corr",
    #     )
    #     ax[1, 1].set_title("phi_corr")
    #     ax[1, 1].plot(
    #         np.real(cav_fdbk_obs_list[1][idx].phi_corr[turn_idx]),
    #         color=clr,
    #         label="phi_corr",
    #     )
    #
    #     # ax[1].plot(np.abs(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]), label="abs fdbk")
    # plt.tight_layout()
    # # plt.legend()
    # plt.show(block=False)

    trn_idx = 0
    plt.figure("coarse_voltage")
    plt.plot(np.real(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k")
    plt.plot(np.imag(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k")
    # plt.plot(np.abs(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k")
    trn_idx = 1
    plt.plot(
        np.real(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]),
        color="b",
        ls="--",
    )
    plt.plot(
        np.imag(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]),
        color="b",
        ls="--",
    )
    # plt.plot(np.abs(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="b", ls="--")
    plt.show(block=False)

    trn_idx = 0
    plt.figure("coarse_voltage_2")
    plt.plot(
        np.angle(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="k"
    )
    trn_idx = 1
    plt.plot(
        np.angle(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]),
        color="b",
        ls="--",
    )
    # plt.plot(np.imag(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="b", ls="--")
    # plt.plot(np.abs(cav_fdbk_obs_list[1][0].v_ant_coarse[trn_idx]), color="b", ls="--")
    plt.show(block=False)

    fig, ax = plt.subplots(2, 2, sharex=True)
    for idx in range(n_turns_in):
        clr = ax[0, 0]._get_lines.get_next_color()
        ax[0, 0].plot(
            ind_volt_obs_list[0][0].total_voltage[idx], color=clr, label="MTW"
        )
        ax[0, 0].plot(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]),
            ls="--",
            color=clr,
            label="real fdbk",
        )
        cavity_voltage_raw = (
            ind_volt_obs_list[0][0].total_voltage[idx]
            - ind_volt_obs_list[0][0].induced_voltage[idx]
        )
        feedback_argmax = np.argmax(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx][0:300])
            - cavity_voltage_raw[0:300]
        )
        print(f"feedback argmax {feedback_argmax}")
        ax[0, 1].plot(
            np.real(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx])
            - cavity_voltage_raw,
            ls="--",
            color=clr,
            label="real fdbk",
        )
        ax[0, 1].plot(
            np.imag(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]),
            # - cavity_voltage_raw,
            ls=":",
            color=clr,
            label="real fdbk",
        )
        print(
            f"ind volt argmax {np.argmax(ind_volt_obs_list[0][0].induced_voltage[idx][0:300])}"
        )
        ax[0, 1].plot(
            ind_volt_obs_list[0][0].induced_voltage[idx],
            color=clr,
            label="MTW",
        )
        ax[1, 0].set_title("v_corr")
        ax[1, 0].plot(
            np.real(cav_fdbk_obs_list[1][0].v_corr[idx]),
            color=clr,
            label="v_corr",
        )
        ax[1, 1].set_title("phi_corr")
        ax[1, 1].plot(
            np.real(cav_fdbk_obs_list[1][0].phi_corr[idx]),
            color=clr,
            label="phi_corr",
        )

        # ax[1].plot(np.abs(cav_fdbk_obs_list[1][0].kick_voltage_fine[idx]), label="abs fdbk")
    plt.tight_layout()
    # plt.legend()
    plt.show()
