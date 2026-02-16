import unittest
from pathlib import Path

import numpy as np

from blond import (
    StaticProfile,
)
from blond.experimental.physics.feedbacks.accelerators.lhc.cavity_feedback import (
    LHCCavityLoop,
    LHCCavityLoopCommissioning,
)
from blond.experimental.physics.feedbacks.transfer_function import (
    TransferFunction,
)
from blond.handle_results.helpers import callers_relative_path


class TestLHCTransferFunction(unittest.TestCase):
    blond2_data = np.load(
        Path(
            callers_relative_path(
                "../resources/lhc_cavity_control_transfer_function_freq.npz",
                stacklevel=1,
            )
        )
    )

    @staticmethod
    def measure_transfer_function(
        open_loop: bool = False, open_otfb: bool = False
    ):
        f_rf = 400.789e6
        harmonic = 35640

        G_a = 6.79e-6  # Analog FB gain [A/V]
        G_d = 10  # Digital FB gain [-]
        tau_loop = 650e-9  # Overall loop delay [s]
        tau_a = 170e-6  # Analog FB delay [s]
        tau_d = 400e-6  # Digital FB delay [s]
        a_comb = 15 / 16  # Comb filter alpha [-]
        Q_L = 20000  # Loaded Quality factor [-]
        tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
        tau_o = 110e-6

        commissioning = LHCCavityLoopCommissioning(
            alpha=a_comb,
            G_a=G_a,
            G_d=G_d,
            tau_a=tau_a,
            tau_d=tau_d,
            tau_o=tau_o,
            open_tuner=True,
            excitation=True,
            open_loop=open_loop,
            open_otfb=open_otfb,
        )

        cavity_control = LHCCavityLoop(
            StaticProfile(0, 2.5e-9, 2**6),
            RFFB=commissioning,
            tau_loop=tau_loop,
            Q_L=Q_L,
            tau_otfb=tau_comp,
            n_pretrack=200,
        )
        cavity_control.set_hardware_commissioning(
            omega_rf=2 * np.pi * f_rf, harmonic=harmonic
        )

        transfer_function = TransferFunction(
            cavity_control.V_EXC_IN,
            cavity_control.V_EXC_OUT,
            T_s=cavity_control.T_s,
        )

        transfer_function.analyse(3564 * 5)

        return transfer_function.H_est, transfer_function.f_est

    def test_open_loop_transfer_function(self):
        tf_est, freq_est = self.measure_transfer_function(
            open_loop=True, open_otfb=True
        )

        np.testing.assert_allclose(
            np.log(np.abs(tf_est)),
            np.log(np.abs(self.blond2_data["open_loop_transfer_function"])),
            atol=1e-2,
            err_msg="Error in amplitude of open-loop transfer function",
        )
        np.testing.assert_allclose(
            np.angle(tf_est),
            np.angle(self.blond2_data["open_loop_transfer_function"]),
            atol=7,
            err_msg="Error in phase of open-loop transfer function",
        )

    def test_closed_loop_transfer_function(self):
        tf_est, freq_est = self.measure_transfer_function(
            open_loop=False, open_otfb=True
        )

        np.testing.assert_allclose(
            np.log(np.abs(tf_est)),
            np.log(np.abs(self.blond2_data["closed_loop_transfer_function"])),
            atol=1e-2,
            err_msg="Error in amplitude of closed-loop transfer function",
        )
        np.testing.assert_allclose(
            np.angle(tf_est),
            np.angle(self.blond2_data["closed_loop_transfer_function"]),
            atol=7,
            err_msg="Error in phase of closed-loop transfer function",
        )

    def test_close_loop_transfer_function_with_otfb(self):
        tf_est, freq_est = self.measure_transfer_function(
            open_loop=False, open_otfb=False
        )

        np.testing.assert_allclose(
            np.log(np.abs(tf_est)),
            np.log(np.abs(self.blond2_data["full_loop_transfer_function"])),
            atol=6e-1,
            err_msg="Error in amplitude of closed-loop transfer function with otfb",
        )
        np.testing.assert_allclose(
            np.angle(tf_est),
            np.angle(self.blond2_data["full_loop_transfer_function"]),
            atol=7,
            err_msg="Error in phase of closed-loop transfer function with otfb",
        )
