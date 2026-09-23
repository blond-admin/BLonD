"""Golden tests comparing the compiled LHC cavity loop to the reference.

``LHCCavityFeedback.track_one_turn_reference`` is the readable, per-sample
Python implementation; ``track_one_turn`` runs the compiled kernel. Both
must produce the same buffers for every commissioning flag combination.

The comparison starts from a warmed-up, physically sensible state: the
cavity loop has a large open-loop gain, so seeding the buffers with
arbitrary noise drives it far from any operating point and lets it diverge
within a few turns, which would compare NaN against NaN.
"""

import dataclasses
import unittest

import numpy as np

from blond import StaticProfile
from blond.physics.feedbacks.accelerators.lhc import (
    LHCCavityFeedback,
    LHCCavityFeedbackCommissioning,
)
from blond.physics.feedbacks.accelerators.lhc.track_kernel import (
    N_SIGNALS,
    SIGNAL_NAMES,
)
from blond.physics.feedbacks.buffers import TwoTurnArray

F_RF = 400.789e6
HARMONIC = 35640

# Set point voltage [V] per cavity, of the order of an LHC flat-bottom
# voltage of 5 MV shared between 8 cavities.
SETPOINT_VOLTAGE = 5e6 / 8

N_WARMUP_TURNS = 10
N_TURNS_COMPARED = 3

# Relative size of the perturbation added to the warmed-up state, so that
# every buffer carries a distinct, non-degenerate signal.
PERTURBATION = 1e-3

# The linear signal path reproduces the reference bit for bit. The
# clamping and saturation branches rebuild a complex number from its
# magnitude and phase, where `math.atan2` and `np.angle` differ in the last
# bits; the loop gain and the double integrator in the tuner CIC filter
# amplify that difference over a turn, so those cases get a tolerance.
RTOL_LINEAR = 0.0
RTOL_NONLINEAR = 1e-9


def make_cavity_feedback(
    commissioning: LHCCavityFeedbackCommissioning,
) -> LHCCavityFeedback:
    """Build a stand-alone LHC cavity feedback without pre-tracking."""
    profile = StaticProfile(cut_left=0, cut_right=2.5e-9, n_bins=4)

    cavity_feedback = LHCCavityFeedback(
        profile=profile, commissioning=commissioning, n_pretrack=0
    )
    cavity_feedback.disable_fine_grid = True
    cavity_feedback.set_hardware_commissioning(
        omega_rf=2 * np.pi * F_RF, harmonic=HARMONIC
    )
    return cavity_feedback


def buffer_names(cavity_feedback: LHCCavityFeedback) -> list[str]:
    """Names of all two-turn buffers on the coarse grid."""
    return [
        field.name
        for field in dataclasses.fields(cavity_feedback.buffers_coarse)
        if isinstance(
            getattr(cavity_feedback.buffers_coarse, field.name), TwoTurnArray
        )
    ]


def snapshot_buffers(cavity_feedback: LHCCavityFeedback) -> dict:
    """Copy the full two-turn state of every coarse buffer."""
    return {
        name: getattr(
            cavity_feedback.buffers_coarse, name
        ).two_turn_view.copy()
        for name in buffer_names(cavity_feedback)
    }


def restore_buffers(
    cavity_feedback: LHCCavityFeedback, snapshot: dict
) -> None:
    """Write a snapshot back into the coarse buffers."""
    for name, data in snapshot.items():
        getattr(cavity_feedback.buffers_coarse, name).two_turn_view[:] = data


class TrackKernelAgreementMixin:
    """Compare the compiled kernel against the reference implementation."""

    commissioning_kwargs: dict = {}
    rtol: float = RTOL_LINEAR
    is_nonlinear: bool = False

    def build(self) -> LHCCavityFeedback:
        """
        Build a warmed-up feedback for this flag combination.

        The loop is first driven to its operating point with the nonlinear
        branches switched off, then perturbed, and only then given a
        switch-and-protect threshold low enough for the clamping and
        saturation branches to act on the largest samples.
        """
        commissioning = LHCCavityFeedbackCommissioning(
            **self.commissioning_kwargs
        )
        cavity_feedback = make_cavity_feedback(commissioning)
        buffers = cavity_feedback.buffers_coarse

        buffers.v_setpoint.prev[:] = SETPOINT_VOLTAGE
        buffers.v_setpoint.curr[:] = SETPOINT_VOLTAGE

        clamping, saturation = commissioning.clamping, commissioning.saturation
        commissioning.clamping = False
        commissioning.saturation = False
        for _turn in range(N_WARMUP_TURNS):
            cavity_feedback.track_one_turn()
        commissioning.clamping, commissioning.saturation = clamping, saturation

        rng = np.random.default_rng(1234)
        for name in buffer_names(cavity_feedback):
            view = getattr(buffers, name).two_turn_view
            scale = np.max(np.abs(view))
            if scale == 0.0:
                scale = 1.0
            view += (
                PERTURBATION
                * scale
                * (
                    rng.standard_normal(len(view))
                    + 1j * rng.standard_normal(len(view))
                )
            )

        cavity_feedback.i_swap_threshold = 0.9 * np.max(
            np.abs(buffers.i_feedback_out.two_turn_view)
        )
        return cavity_feedback

    def track(self, cavity_feedback: LHCCavityFeedback, compiled: bool):
        """Track the comparison turns with one of the two implementations."""
        for _turn in range(N_TURNS_COMPARED):
            if compiled:
                cavity_feedback.track_one_turn()
            else:
                cavity_feedback.track_one_turn_reference()

    def test_kernel_matches_reference(self):
        cavity_feedback = self.build()
        initial_state = snapshot_buffers(cavity_feedback)

        self.track(cavity_feedback, compiled=True)
        kernel_state = snapshot_buffers(cavity_feedback)

        restore_buffers(cavity_feedback, initial_state)
        self.track(cavity_feedback, compiled=False)
        reference_state = snapshot_buffers(cavity_feedback)

        for name, reference in reference_state.items():
            with self.subTest(buffer=name):
                self.assertTrue(
                    np.all(np.isfinite(reference)),
                    msg=f"reference state of {name} diverged; "
                    f"the comparison would be vacuous",
                )
                np.testing.assert_allclose(
                    kernel_state[name],
                    reference,
                    rtol=self.rtol,
                    atol=0.0,
                    err_msg=f"kernel and reference disagree on {name}",
                )

    def test_kernel_changes_state(self):
        """Guard against a kernel that silently does nothing."""
        cavity_feedback = self.build()
        before = snapshot_buffers(cavity_feedback)

        cavity_feedback.track_one_turn()
        after = snapshot_buffers(cavity_feedback)

        self.assertFalse(
            np.array_equal(before["i_gen"], after["i_gen"]),
            msg="tracking one turn left the generator current unchanged",
        )

    def test_nonlinear_branch_is_exercised(self):
        """The clamping and saturation branches must actually be hit."""
        if not self.is_nonlinear:
            self.skipTest("flag combination has no nonlinear branch")

        cavity_feedback = self.build()
        cavity_feedback.track_one_turn()

        above_threshold = np.count_nonzero(
            np.abs(cavity_feedback.buffers_coarse.i_feedback_out.curr)
            > cavity_feedback.i_swap_threshold
        )
        self.assertGreater(
            above_threshold,
            0,
            msg="no sample exceeded the switch-and-protect threshold, "
            "so the nonlinear branch was never taken",
        )


class TestTrackKernelDefault(TrackKernelAgreementMixin, unittest.TestCase):
    commissioning_kwargs = {}


class TestTrackKernelOpenLoops(TrackKernelAgreementMixin, unittest.TestCase):
    commissioning_kwargs = {
        "open_loop": True,
        "open_otfb": True,
        "open_drive": True,
        "open_rffb": True,
    }


class TestTrackKernelKlystron(TrackKernelAgreementMixin, unittest.TestCase):
    commissioning_kwargs = {"enable_klystron": True}


class TestTrackKernelClamping(TrackKernelAgreementMixin, unittest.TestCase):
    commissioning_kwargs = {"clamping": True}
    rtol = RTOL_NONLINEAR
    is_nonlinear = True


class TestTrackKernelSaturation(TrackKernelAgreementMixin, unittest.TestCase):
    commissioning_kwargs = {"saturation": True}
    rtol = RTOL_NONLINEAR
    is_nonlinear = True


class TestTrackKernelAllHardware(TrackKernelAgreementMixin, unittest.TestCase):
    commissioning_kwargs = {
        "clamping": True,
        "saturation": True,
        "enable_klystron": True,
    }
    rtol = RTOL_NONLINEAR
    is_nonlinear = True


class TestSignalBlockAliasing(unittest.TestCase):
    """The named buffers must stay views into the shared signal block.

    The kernel writes into the block, and the rest of the model reads the
    named buffers. If a buffer were ever rebound to a fresh array, both
    would keep working in isolation while silently drifting apart.
    """

    def setUp(self):
        self.cavity_feedback = make_cavity_feedback(
            LHCCavityFeedbackCommissioning()
        )
        self.buffers = self.cavity_feedback.buffers_coarse

    def test_block_has_one_row_per_signal(self):
        self.assertEqual(
            self.buffers.signals.shape,
            (N_SIGNALS, 2 * self.cavity_feedback.n_coarse),
        )
        self.assertEqual(N_SIGNALS, len(SIGNAL_NAMES))

    def test_every_two_turn_buffer_has_a_row(self):
        self.assertCountEqual(
            buffer_names(self.cavity_feedback), list(SIGNAL_NAMES)
        )

    def test_buffers_are_views_into_the_block(self):
        for row, name in enumerate(SIGNAL_NAMES):
            with self.subTest(signal=name):
                buffer = getattr(self.buffers, name)
                self.assertTrue(
                    np.shares_memory(
                        buffer.two_turn_view, self.buffers.signals
                    )
                )
                # A write through the block must be visible in the buffer,
                # at the row the kernel writes to.
                self.buffers.signals[row, 0] = 3.0 + 4.0j
                self.assertEqual(buffer.prev[0], 3.0 + 4.0j)

    def test_writes_through_buffers_reach_the_block(self):
        self.buffers.v_ant.curr[5] = 1.5 - 2.5j
        row = SIGNAL_NAMES.index("v_ant")
        column = self.cavity_feedback.n_coarse + 5
        self.assertEqual(self.buffers.signals[row, column], 1.5 - 2.5j)


if __name__ == "__main__":
    unittest.main()
