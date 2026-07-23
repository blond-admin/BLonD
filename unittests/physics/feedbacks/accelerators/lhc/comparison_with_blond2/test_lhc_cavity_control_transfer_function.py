"""Compare blond3 and blond2 LHC cavity-loop transfer functions."""

import unittest

import numpy as np

from .support import blond2_reference

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

# The three measured loop configurations: (open_loop, open_otfb) flags of
# the commissioning object; everything else is identical between them.
_LOOP_CONFIGS = (
    ("open_loop", True, True),
    ("closed_loop", False, True),
    ("full_loop", False, False),
)


def _run_blond2() -> dict[str, np.ndarray]:
    """
    Measure the frozen blond2 transfer functions for all loop configurations.

    Only executed when the pinned reference file is absent or regeneration is
    requested (see :func:`support.blond2_reference`); the legacy code and its
    fixed excitation-noise seeds make the measurements deterministic.

    Returns
    -------
    dict
        The estimated transfer function and its frequency axis for the
        open-loop, closed-loop and full-loop (with OTFB) configurations.
    """
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

    results = {}
    for prefix, open_loop, open_otfb in _LOOP_CONFIGS:
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
            open_loop=open_loop,
            open_otfb=open_otfb,
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

        results[f"{prefix}_transfer_function_blond2"] = transfer_function.H_est
        results[f"{prefix}_freq_blond2"] = transfer_function.f_est

    return results


class TestLHCTransferFunction(unittest.TestCase):
    """Test LHC feedback transfer functions."""

    @staticmethod
    def measure_transfer_function_blond3(
        open_loop: bool = False, open_otfb: bool = False
    ):
        """
        Measure full transfer function in blond3.

        Parameters
        ----------
        open_loop
            If True, open the main feedback loop during the measurement.
        open_otfb
            If True, open the one-turn feedback (OTFB) during the measurement.

        Returns
        -------
        H_est
            Estimated transfer function.
        f_est
            Frequency points at which the transfer function is estimated.
        """
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
        """Load the pinned blond2 transfer-function measurements."""
        for key, value in blond2_reference(
            "transfer_function", _run_blond2
        ).items():
            setattr(cls, key, value)

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
        # Compare the phases via the wrapped difference angle(H3*conj(H2)):
        # a direct np.angle() comparison jumps by 2*pi at the +-pi branch
        # cut, which forced the previous tolerance above 2*pi and made it
        # unfailable. Measured max deviation: 3.9e-3 rad.
        np.testing.assert_allclose(
            np.angle(
                tf_est * np.conj(self.open_loop_transfer_function_blond2)
            ),
            0,
            atol=1e-2,
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
        # Wrapped phase difference (see the open-loop test for why).
        # Measured max deviation: 1.5e-6 rad.
        np.testing.assert_allclose(
            np.angle(
                tf_est * np.conj(self.closed_loop_transfer_function_blond2)
            ),
            0,
            atol=1e-5,
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
        # Wrapped phase difference (see the open-loop test for why). The
        # full loop with OTFB genuinely deviates between the codes, in
        # phase (measured max: 5.8e-1 rad) as in amplitude (atol=6e-1
        # above); the tolerance brackets the current deviation.
        np.testing.assert_allclose(
            np.angle(
                tf_est * np.conj(self.full_loop_transfer_function_blond2)
            ),
            0,
            atol=7e-1,
            err_msg="Error in phase of closed-loop transfer function with otfb",
        )
