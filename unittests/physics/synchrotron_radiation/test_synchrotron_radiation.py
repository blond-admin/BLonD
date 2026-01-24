import copy
import unittest
from random import random
from unittest.mock import Mock

import numpy as np

from blond import (
    MultiHarmonicRFStation,
    ReferenceEnergyChange,
    Ring,
    SingleHarmonicRFStation,
    SynchrotronRadiationMaster,
    backend,
)
from blond.handle_results.observables_as_elements import (
    BunchObservationMetaParams,
)
from blond.physics.cavities import RFStationBaseClass
from blond.physics.drifts import DriftBaseClass, DriftSimple
from blond.physics.synchrotron_radiation.synchrotron_radiation import (
    _SynchrotronRadiationDrift,
    _SynchrotronRadiationSection,
)


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

        self.decimal = 6 if backend.float == np.float32 else 12

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

    def test_generate_synchrotron_radiation_subclasses_errors(self):
        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        momentum_compaction_factor = 0.646747216157 / (90.65874532 * 1e3)
        self.cavity = SingleHarmonicRFStation()
        self.cavity.harmonic = 242400
        self.cavity.voltage = 50.1e6
        self.cavity.phi_rf = 0

        ring.add_element(self.cavity)

        number_of_sections = 5
        for i in range(number_of_sections):
            drift = DriftSimple(
                name=f"drift{i + 1}",
                orbit_length=ring.circumference / number_of_sections,
                momentum_compaction_factor=momentum_compaction_factor
                / number_of_sections,
                section_index=i,
            )
            ring.add_element(drift, section_index=i)

        # Check the waring and exception are raised
        SRM = SynchrotronRadiationMaster()
        SRM.generated_children = 1
        with self.assertRaisesRegex(
            expected_exception=Warning,
            expected_regex="Synchrotron radiation subclasses have already been "
            "generated. Command ignored",
        ):
            SRM.generate_synchrotron_radiation_subclasses(ring=ring)

        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[DriftBaseClass, SingleHarmonicRFStation]
        )
        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="Inhomogeneous element classes.",
        ):
            SRM.generate_synchrotron_radiation_subclasses(
                ring=ring,
            )

        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[BunchObservationMetaParams]
        )
        with self.assertRaises(
            expected_exception=TypeError,
        ):
            SRM.generate_synchrotron_radiation_subclasses(
                ring=ring,
            )

    def test_generate_synchrotron_radiation_subclasses_drift_trackers(self):
        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        momentum_compaction_factor = 0.646747216157 / (90.65874532 * 1e3)
        self.cavity = SingleHarmonicRFStation()
        self.cavity.harmonic = 242400
        self.cavity.voltage = 50.1e6
        self.cavity.phi_rf = 0
        ring.add_element(self.cavity)

        number_of_sections = 5
        for i in range(number_of_sections):
            drift = DriftSimple(
                name=f"drift{i + 1}",
                orbit_length=ring.circumference / number_of_sections,
                momentum_compaction_factor=momentum_compaction_factor
                / number_of_sections,
                section_index=i,
            )
            ring.add_element(drift, section_index=i)
        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[DriftBaseClass]
        )
        ring_SRdrifts = copy.deepcopy(ring)
        SRM.generate_synchrotron_radiation_subclasses(ring=ring_SRdrifts)

        # Check the corrected number of _SynchrotronRadiationDrift trackers
        # have been generated
        SRdrifttracker_list = ring_SRdrifts.elements.get_elements(
            class_=_SynchrotronRadiationDrift
        )
        self.assertEqual(len(SRdrifttracker_list), number_of_sections)
        self.assertEqual(len(SRdrifttracker_list), len(SRM.generated_children))
        self.assertEqual(
            SRM.number_of_generated_synchrotron_radiation_classes,
            len(SRM.generated_children),
        )

        for i in range(number_of_sections):
            # Ensures the created trackers are the expected ones, and located
            # before the drifts
            assert (
                ring_SRdrifts.elements.elements[1 + 2 * i]
                == SRM.generated_children[i]
            )

            # verifies the synchrotron radiation integral share of each tracker
            np.testing.assert_array_almost_equal(
                ring_SRdrifts.elements.elements[
                    1 + 2 * i
                ].share_of_synchrotron_radiation_integrals,
                self.synchrotron_radiation_integrals / 5,
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                ring_SRdrifts.elements.elements[
                    1 + 2 * i
                ].synchrotron_radiation_integrals_drift,
                self.synchrotron_radiation_integrals / 5,
                decimal=self.decimal,
            )

    def test_generate_synchrotron_radiation_subclasses_cavity_trackers(self):
        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        momentum_compaction_factor = 0.646747216157 / (90.65874532 * 1e3)
        self.cavity = SingleHarmonicRFStation()
        self.cavity.harmonic = 242400
        self.cavity.voltage = 50.1e6
        self.cavity.phi_rf = 0
        ring.add_element(self.cavity)

        number_of_sections = 5
        for i in range(number_of_sections):
            drift = DriftSimple(
                name=f"drift{i + 1}",
                orbit_length=ring.circumference / number_of_sections,
                momentum_compaction_factor=momentum_compaction_factor
                / number_of_sections,
                section_index=i,
            )
            ring.add_element(drift, section_index=i)

        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[SingleHarmonicRFStation]
        )
        ring_SRdrifts = copy.deepcopy(ring)
        SRM.generate_synchrotron_radiation_subclasses(ring=ring_SRdrifts)

        # Check the corrected number of _SynchrotronRadiationDrift trackers
        # have been generated
        SRsectiontracker_list = ring_SRdrifts.elements.get_elements(
            class_=_SynchrotronRadiationSection
        )
        self.assertEqual(len(SRsectiontracker_list), 1)
        self.assertEqual(
            len(SRsectiontracker_list), len(SRM.generated_children)
        )
        self.assertEqual(
            SRM.number_of_generated_synchrotron_radiation_classes,
            len(SRM.generated_children),
        )

        # Ensures the created trackers are the expected ones, and located
        # before the drifts
        assert ring_SRdrifts.elements.elements[1] == SRM.generated_children[0]

        # verifies the synchrotron radiation integral share of each tracker
        np.testing.assert_array_almost_equal(
            ring_SRdrifts.elements.elements[
                1
            ].share_of_synchrotron_radiation_integrals,
            self.synchrotron_radiation_integrals,
            decimal=self.decimal,
        )
        np.testing.assert_array_almost_equal(
            ring_SRdrifts.elements.elements[
                1
            ].synchrotron_radiation_integrals_section,
            self.synchrotron_radiation_integrals,
            decimal=self.decimal,
        )

    def test_generate_synchrotron_radiation_subclasses_cavities_trackers(self):
        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        momentum_compaction_factor = 0.646747216157 / (90.65874532 * 1e3)

        number_of_sections = 4
        for i in range(number_of_sections):
            rf_station = SingleHarmonicRFStation(section_index=i)
            rf_station.harmonic = 242400
            rf_station.voltage = 50.1e6
            rf_station.phi_rf = 0
            ring.add_element(rf_station)
            drift = DriftSimple(
                name=f"drift{i + 1}",
                orbit_length=ring.circumference / number_of_sections,
                momentum_compaction_factor=momentum_compaction_factor
                / number_of_sections,
                section_index=i,
            )
            ring.add_element(drift, section_index=i)

        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[SingleHarmonicRFStation]
        )
        ring_SRdrifts = copy.deepcopy(ring)
        SRM.generate_synchrotron_radiation_subclasses(ring=ring_SRdrifts)

        # Check the corrected number of _SynchrotronRadiationDrift trackers
        # have been generated
        SRsectiontracker_list = ring_SRdrifts.elements.get_elements(
            class_=_SynchrotronRadiationSection
        )
        self.assertEqual(len(SRsectiontracker_list), number_of_sections)
        self.assertEqual(
            len(SRsectiontracker_list), len(SRM.generated_children)
        )
        self.assertEqual(
            SRM.number_of_generated_synchrotron_radiation_classes,
            len(SRM.generated_children),
        )

        for i in range(number_of_sections):
            # Ensures the created trackers are the expected ones, and located
            # before the drifts
            assert (
                ring_SRdrifts.elements.elements[1 + 3 * i]
                == SRM.generated_children[i]
            )

            # verifies the synchrotron radiation integral share of each tracker
            np.testing.assert_array_almost_equal(
                ring_SRdrifts.elements.elements[
                    1 + 3 * i
                ].share_of_synchrotron_radiation_integrals,
                self.synchrotron_radiation_integrals / number_of_sections,
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                ring_SRdrifts.elements.elements[
                    1 + 3 * i
                ].synchrotron_radiation_integrals_section,
                self.synchrotron_radiation_integrals / number_of_sections,
                decimal=self.decimal,
            )
