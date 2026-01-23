import copy
import unittest
from random import random

import numpy as np

from blond import (
    Ring,
    SingleHarmonicRFStation,
    SynchrotronRadiationMaster,
)
from blond.physics.drifts import DriftBaseClass


class TestSynchrotronRadiationMaster(unittest.TestCase):
    def setUp(self):
        self.synchrotron_radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        self.SR_ring = Ring(
            10.0, radiation_integrals=self.synchrotron_radiation_integrals
        )
        self.ring = copy.deepcopy(self.SR_ring)

    def test_inputs(self):
        SRHandler = SynchrotronRadiationMaster()
        self.assertFalse(SRHandler.verbose)
        self.assertFalse(SRHandler._disable_quantum_excitation)
        self.assertEqual(SRHandler.track_before_element_type, DriftBaseClass)

        self.assertIsNone(SRHandler._simulation)
        self.assertEqual(SRHandler._turn_i, 0)
        self.assertIsNone(SRHandler._simulation)
        self.assertIsNone(SRHandler._natural_energy_spread)
        self.assertIsNone(SRHandler._energy_loss_per_turn)
        self.assertIsNone(SRHandler._energy_loss_per_turn)

        self.assertListEqual(SRHandler.generated_children, [])

        SRHandleriso = SynchrotronRadiationMaster(
            _disable_quantum_excitation=True,
            verbose=True,
            track_before_element_type=SingleHarmonicRFStation,
        )
        self.assertTrue(SRHandleriso.verbose)
        self.assertTrue(SRHandleriso._disable_quantum_excitation)
        self.assertEqual(
            SRHandleriso.track_before_element_type, SingleHarmonicRFStation
        )

        with self.assertRaisesRegex(
            TypeError,
            expected_regex=f"Expected a list or numpy.ndarray as an input. Received"
            f" {type('not an array')}.",
        ):
            self.SRHandler = SynchrotronRadiationMaster()

    def test__str__(self):
        SRM = SynchrotronRadiationMaster()
        self.assertRegex(
            SRM.__str__(),
            (
                f"Synchrotron radiation master class set up for the "
                f" ring. Simulation currently set for turn "
                f"{0}. \n Generated "
                f"{0} "
                f"synchrotron radiation elements."
            ),
        )

    def test_set_synchrotron_radiation_integrals(self):
        ring = Ring(90.65874532 * 1e3)
        SRM = SynchrotronRadiationMaster()
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Synchrotron radiation damping "
            "and quantum excitation require"
            " either the bending radius " + "for an isomagnetic ring, or the "
            "first five synchrotron radiation "
            "integrals.",
        ):
            SRM._set_synchrotron_radiation_integrals(ring=ring)

        SRM._set_synchrotron_radiation_integrals(
            ring=ring, bending_radius=14428.78745218723
        )
        np.testing.assert_array_equal(
            SRM._synchrotron_radiation_integrals,
            np.array(
                [
                    0,
                    0.0004354617689116441,
                    3.018006678347967e-08,
                    0,
                    0,
                ]
            ),
        )
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Synchrotron radiation damping "
            "and quantum excitation require"
            " either the bending radius " + "for an isomagnetic ring, or the "
            "first five synchrotron radiation "
            "integrals.",
        ):
            SRM._set_synchrotron_radiation_integrals(ring=ring)

        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Could not transform the input into an array",
        ):
            SRM._set_synchrotron_radiation_integrals(
                ring=ring,
                radiation_integrals=[
                    [random() for k in range(5)],
                    [random() for k in range(7)],
                ],
            )

        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex=f"Expected a list or numpy.ndarray as an input. Received"
            f" {type('not an array')}.",
        ):
            SRM._set_synchrotron_radiation_integrals(
                ring=ring, radiation_integrals="not an array"
            )

        SRM._set_synchrotron_radiation_integrals(
            ring=ring, radiation_integrals=self.synchrotron_radiation_integrals
        )
        np.testing.assert_array_equal(
            SRM._synchrotron_radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        SRM._set_synchrotron_radiation_integrals(
            ring=ring,
            radiation_integrals=self.synchrotron_radiation_integrals,
            bending_radius=0000,
        )
        np.testing.assert_array_equal(
            SRM._synchrotron_radiation_integrals,
            self.synchrotron_radiation_integrals,
        )

        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        SRM._set_synchrotron_radiation_integrals(ring=ring)
        np.testing.assert_array_equal(
            SRM._synchrotron_radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        SRM._set_synchrotron_radiation_integrals(
            ring=ring, radiation_integrals=np.zeros(5)
        )
        np.testing.assert_array_equal(
            SRM._synchrotron_radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        SRM._set_synchrotron_radiation_integrals(
            ring=ring, bending_radius=10000
        )
        np.testing.assert_array_equal(
            SRM._synchrotron_radiation_integrals,
            self.synchrotron_radiation_integrals,
        )

    # def test_properties(self):
    #     SRM = SynchrotronRadiationMaster(
    #         radiation_integrals=self.synchrotron_radiation_integrals,
    #     )
    #     np.assert
    #
    # def test_print_synchrotron_radiation_parameters(self):
    #
    # def test_compute_turn_by_turn_synchrotron_radiation_parameters(self):
    #
    # def test_generate_synchrotron_radiation_subclasses(self):
    #
    #
