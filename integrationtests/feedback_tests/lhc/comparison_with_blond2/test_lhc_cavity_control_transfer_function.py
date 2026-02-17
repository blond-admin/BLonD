"""Test LHC feedback with injection phase error."""

import unittest

import numpy as np

f_rf = 400.789e6
harmonic = 35640
C = 26658.8832  # Machine circumference [m]
p_s = 450e9  # Synchronous momentum [eV/c]
V = 1e6
gamma_t = 53.8  # Transition gamma
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

G_a = 6.79e-6  # Analog FB gain [A/V]
G_d = 10  # Digital FB gain [-]
G_otfb = 10
R_over_Q = 45  # Cavity R/Q [Ohms]
G_gen = 1
tau_loop = 650e-9  # Overall loop delay [s]
tau_a = 170e-6  # Analog FB delay [s]
tau_d = 400e-6  # Digital FB delay [s]
a_comb = 15 / 16  # Comb filter alpha [-]
Q_L = 20000  # Loaded Quality factor [-]
tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
tau_o = 110e-6


class TestLHCTransferFunction(unittest.TestCase):
    """Test LHC feedback transfer functions."""

    @staticmethod
    def measure_transfer_function_blond3(
        open_loop: bool = False, open_otfb: bool = False
    ):
        """Measure full transfer function in blond3."""
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

    @classmethod
    def setUpClass(cls):
        """Measure full transfer function in blond2 and save the results."""

        def measure_transfer_functions_blond2():  # noqa: PLR0915
            """Measure full transfer function in blond2."""
            from blond.legacy.blond2.beam.beam import Beam, Proton
            from blond.legacy.blond2.beam.profile import CutOptions, Profile
            from blond.legacy.blond2.input_parameters.rf_parameters import (
                RFStation,
            )
            from blond.legacy.blond2.input_parameters.ring import Ring
            from blond.legacy.blond2.llrf.cavity_feedback import (
                LHCCavityLoop,
                LHCCavityLoopCommissioning,
            )
            from blond.legacy.blond2.llrf.transfer_function import (
                TransferFunction,
            )

            ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=1)
            rf = RFStation(ring, [harmonic], [V], [0])
            beam = Beam(ring, 1, 1)
            profile = Profile(
                beam,
                CutOptions=CutOptions(
                    cut_left=2 * rf.t_rf[0, 0],
                    cut_right=3 * rf.t_rf[0, 0],
                    n_slices=64,
                ),
            )

            commissioning = LHCCavityLoopCommissioning(
                alpha=a_comb,
                d_phi_ad=0,
                G_a=G_a,
                G_d=G_d,
                G_o=G_otfb,
                tau_a=tau_a,
                tau_d=tau_d,
                tau_o=tau_o,
                open_drive=False,
                open_loop=True,
                open_otfb=True,
                open_rffb=False,
                open_tuner=True,
                full_detuning=False,
                excitation=True,
                enable_klystron=False,
            )

            cavity_feedback = LHCCavityLoop(
                rf,
                profile,
                n_cavities=1,
                f_c=rf.omega_rf[0, 0] / (2 * np.pi),
                G_gen=G_gen,
                n_pretrack=200,
                Q_L=Q_L,
                R_over_Q=R_over_Q,
                tau_loop=tau_loop,
                tau_otfb=tau_comp,
                RFFB=commissioning,
            )

            n_turns_excite = 200
            cavity_feedback.track_no_beam_excitation(n_turns_excite)

            transfer_function = TransferFunction(
                cavity_feedback.V_EXC_IN,
                cavity_feedback.V_EXC_OUT,
                T_s=cavity_feedback.T_s,
            )

            transfer_function.analyse(3564 * 5)

            cls.open_loop_transfer_function_blond2 = transfer_function.H_est
            cls.open_loop_freq_blond2 = transfer_function.f_est

            commissioning = LHCCavityLoopCommissioning(
                alpha=a_comb,
                d_phi_ad=0,
                G_a=G_a,
                G_d=G_d,
                G_o=G_otfb,
                tau_a=tau_a,
                tau_d=tau_d,
                tau_o=tau_o,
                open_drive=False,
                open_loop=False,
                open_otfb=True,
                open_rffb=False,
                open_tuner=True,
                full_detuning=False,
                excitation=True,
                enable_klystron=False,
            )

            cavity_feedback = LHCCavityLoop(
                rf,
                profile,
                n_cavities=1,
                f_c=rf.omega_rf[0, 0] / (2 * np.pi),
                G_gen=G_gen,
                n_pretrack=200,
                Q_L=Q_L,
                R_over_Q=R_over_Q,
                tau_loop=tau_loop,
                tau_otfb=tau_comp,
                RFFB=commissioning,
            )

            n_turns_excite = 200
            cavity_feedback.track_no_beam_excitation(n_turns_excite)

            transfer_function = TransferFunction(
                cavity_feedback.V_EXC_IN,
                cavity_feedback.V_EXC_OUT,
                T_s=cavity_feedback.T_s,
            )

            transfer_function.analyse(3564 * 5)

            cls.closed_loop_transfer_function_blond2 = transfer_function.H_est
            cls.closed_loop_freq_blond2 = transfer_function.f_est

            # Closed-loop transfer function with otfb measurement
            commissioning = LHCCavityLoopCommissioning(
                alpha=a_comb,
                d_phi_ad=0,
                G_a=G_a,
                G_d=G_d,
                G_o=G_otfb,
                tau_a=tau_a,
                tau_d=tau_d,
                tau_o=tau_o,
                open_drive=False,
                open_loop=False,
                open_otfb=False,
                open_rffb=False,
                open_tuner=True,
                full_detuning=False,
                excitation=True,
                enable_klystron=False,
            )

            cavity_feedback = LHCCavityLoop(
                rf,
                profile,
                n_cavities=1,
                f_c=rf.omega_rf[0, 0] / (2 * np.pi),
                G_gen=G_gen,
                n_pretrack=200,
                Q_L=Q_L,
                R_over_Q=R_over_Q,
                tau_loop=tau_loop,
                tau_otfb=tau_comp,
                RFFB=commissioning,
            )

            n_turns_excite = 200
            cavity_feedback.track_no_beam_excitation(n_turns_excite)

            transfer_function = TransferFunction(
                cavity_feedback.V_EXC_IN,
                cavity_feedback.V_EXC_OUT,
                T_s=cavity_feedback.T_s,
            )

            transfer_function.analyse(3564 * 5)

            cls.full_loop_transfer_function_blond2 = transfer_function.H_est
            cls.full_loop_freq_blond2 = transfer_function.f_est

        measure_transfer_functions_blond2()

    def test_open_loop_transfer_function(self):
        """Measure open loop transfer function in blond3 and compare with blond2."""
        tf_est, freq_est = self.measure_transfer_function_blond3(
            open_loop=True, open_otfb=True
        )

        np.testing.assert_allclose(
            np.log(np.abs(tf_est)),
            np.log(np.abs(self.open_loop_transfer_function_blond2)),
            atol=1e-2,
            err_msg="Error in amplitude of open-loop transfer function",
        )
        np.testing.assert_allclose(
            np.angle(tf_est),
            np.angle(self.open_loop_transfer_function_blond2),
            atol=7,
            err_msg="Error in phase of open-loop transfer function",
        )

    def test_closed_loop_transfer_function(self):
        """Measure closed loop transfer function in blond3 and compare with blond2."""
        tf_est, freq_est = self.measure_transfer_function_blond3(
            open_loop=False, open_otfb=True
        )

        np.testing.assert_allclose(
            np.log(np.abs(tf_est)),
            np.log(np.abs(self.closed_loop_transfer_function_blond2)),
            atol=1e-2,
            err_msg="Error in amplitude of closed-loop transfer function",
        )
        np.testing.assert_allclose(
            np.angle(tf_est),
            np.angle(self.closed_loop_transfer_function_blond2),
            atol=7,
            err_msg="Error in phase of closed-loop transfer function",
        )

    def test_close_loop_transfer_function_with_otfb(self):
        """Measure open loop transfer function with otfb in blond3 and compare with blond2."""
        tf_est, freq_est = self.measure_transfer_function_blond3(
            open_loop=False, open_otfb=False
        )

        np.testing.assert_allclose(
            np.log(np.abs(tf_est)),
            np.log(np.abs(self.full_loop_transfer_function_blond2)),
            atol=6e-1,
            err_msg="Error in amplitude of closed-loop transfer function with otfb",
        )
        np.testing.assert_allclose(
            np.angle(tf_est),
            np.angle(self.full_loop_transfer_function_blond2),
            atol=7,
            err_msg="Error in phase of closed-loop transfer function with otfb",
        )
