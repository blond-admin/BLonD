import unittest

from blond.physics.feedbacks.accelerators.lhc.beam_feedback import (
    LHCBeamControl,
)
from blond.physics.feedbacks.accelerators.ps.beam_feedback import (
    PSBeamControl,
)
from blond.physics.feedbacks.accelerators.psb.beam_feedback import (
    PSBBeamControl,
)
from blond.physics.feedbacks.accelerators.sps.beam_feedback import (
    SPSBeamControl,
)
from blond.testing.backend_testing import BLonDTestCase

BEAM_CONTROLS = (
    LHCBeamControl,
    PSBeamControl,
    PSBBeamControl,
    SPSBeamControl,
)


class TestBeamFeedbackBase(BLonDTestCase):
    def test___init___forwards_section_index_and_name(self):
        for beam_control in BEAM_CONTROLS:
            with self.subTest(beam_control=beam_control.__name__):
                feedback = beam_control(
                    profile=None, section_index=3, name="my_feedback"
                )
                self.assertEqual(feedback.section_index, 3)
                self.assertEqual(feedback.name, "my_feedback")

    def test___init___stores_documented_options(self):
        for beam_control in BEAM_CONTROLS:
            with self.subTest(beam_control=beam_control.__name__):
                feedback = beam_control(
                    profile=None,
                    delay=4,
                    window_coefficient=0.5,
                    time_offset=1e-9,
                    sample_de=2,
                )
                self.assertEqual(feedback._delay, 4)
                self.assertEqual(feedback.window_coefficient, 0.5)
                self.assertEqual(feedback.time_offset, 1e-9)
                self.assertEqual(feedback.sample_de, 2)

    def test___init___rejects_unknown_kwargs(self):
        for beam_control in BEAM_CONTROLS:
            with (
                self.subTest(beam_control=beam_control.__name__),
                self.assertRaises(TypeError),
            ):
                beam_control(profile=None, dealy=4)  # typo of `delay`


if __name__ == "__main__":
    unittest.main()
