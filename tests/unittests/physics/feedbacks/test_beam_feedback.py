import unittest
from unittest.mock import Mock

import numpy as np
import pytest

from blond import Beam, proton
from blond.core.backends.mpi_distributed.helpers import mpi_is_distributed
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


class TestRadialDifference(BLonDTestCase):
    MOMENTUM_COMPACTION_FACTOR = 0.027
    CIRCUMFERENCE = 628.0  # [m]
    TOTAL_ENERGY = 26e9  # [eV]

    def _radial_difference(self, dE, sample_de):
        """`PSBeamControl` with `radial_difference` applied to `dE`."""
        feedback = PSBeamControl(profile=None, sample_de=sample_de)
        feedback._simulation = Mock(
            ring=Mock(
                momentum_compaction_factor=self.MOMENTUM_COMPACTION_FACTOR,
                circumference=self.CIRCUMFERENCE,
            )
        )
        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(
            dt=np.zeros(len(dE)),
            dE=dE,
            reference_total_energy=self.TOTAL_ENERGY,
            mpi_mode="root-distributes",
        )
        feedback.radial_difference(beam=beam)
        return feedback, beam

    def _expected_drho(self, average_de, beam):
        ring_radius = self.CIRCUMFERENCE / (2 * np.pi)
        return (
            self.MOMENTUM_COMPACTION_FACTOR
            * ring_radius
            * average_de
            / (beam.reference.beta**2 * beam.reference.total_energy)
        )

    def test_radial_difference_uses_every_sample_de_particle(self):
        if mpi_is_distributed():
            self.skipTest("Serial values; see the MPI test")
        feedback, beam = self._radial_difference(
            dE=np.arange(12.0), sample_de=2
        )
        # mean of 0, 2, ..., 10 [eV]
        self.assertAlmostEqual(feedback.average_de, 5.0)
        self.assertAlmostEqual(
            feedback.drho, self._expected_drho(5.0, beam), places=15
        )

    @pytest.mark.mpi
    def test_radial_difference_is_global_mean_over_ranks(self):
        if not mpi_is_distributed():
            self.skipTest("Only with MPI")
        # Rank 0 gets 0 eV and the last rank 12 eV particles, so a
        # rank-local mean would give 0 or 12 there instead of 6.
        dE = np.concatenate((np.zeros(6), np.full(6, 12.0)))
        feedback, beam = self._radial_difference(dE=dE, sample_de=1)
        self.assertAlmostEqual(feedback.average_de, 6.0)
        self.assertAlmostEqual(
            feedback.drho, self._expected_drho(6.0, beam), places=15
        )


if __name__ == "__main__":
    unittest.main()
