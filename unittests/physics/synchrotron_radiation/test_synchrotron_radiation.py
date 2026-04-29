from __future__ import annotations

import copy
import unittest
from random import random
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light as c0

from blond import (
    Ring,
    SingleHarmonicRFStation,
    backend,
    positron,
)
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.particle_types import ParticleType, electron
from blond.core.reference_clock.reference_clock import ReferenceCoordinates
from blond.generals.distributed.distributed_array import DistributedArray
from blond.handle_results.observables_as_elements import (
    BunchObservationMetaParams,
)
from blond.physics.drifts import DriftBaseClass, DriftSimple
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    SynchrotronRadiationMaster,
    _SynchrotronRadiationTracker,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class BeamBaseClassTester(BeamBaseClass):
    def __init__(
        self,
        intensity: int | float,
        particle_type: ParticleType,
        is_counter_rotating: bool = False,
        is_distributed=False,
    ):
        super().__init__(
            intensity=intensity,
            particle_type=particle_type,
            is_counter_rotating=is_counter_rotating,
            is_distributed=is_distributed,
        )
        # self.reference = Mock(ReferenceCoordinates)
        self.reference._particle_type = particle_type
        self.reference.time = 0
        self.reference_beta = 0.99
        self.reference_velocity = self.reference_beta * c0
        self.reference_gamma = np.sqrt(1 - 0.99**2)  # beta**2
        self.reference_total_energy = 20e9
        self.reference.total_energy = 20e9
        self._dE = DistributedArray(
            np.linspace(-1e6, 1e6, 10, dtype=backend.float)
        )  #
        # delta E
        # in eV
        self._dt = DistributedArray(
            np.linspace(-1e-6, 1e-6, 10, dtype=backend.float)
        )  # delta t
        # in s
        self._flags = np.zeros(10, dtype=np.int32)
        self._ids = np.arange(10, dtype=np.int32)

    def ratio(self) -> float:
        return self.intensity / self.common_array_size

    def setup_beam(
        self,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray = None,
        reference_time: float | None = None,
        reference_total_energy: float | None = None,
        mpi_mode: Literal["root-distributes", "all-ranks"] = "all-ranks",
        **kwargs,
    ) -> None:
        """Sets beam array attributes for simulation

        Parameters
        ----------
        mpi_mode
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
        flags
            Macro-particle flags
        reference_time
            Time of the reference frame (global time), in [s]
        reference_total_energy
            Time of the reference frame (global total energy), in [eV]
        mpi_mode
            Specifies how the particle data is distributed across multiple ranks (processing
            units) in a parallel environment:

            - "root-distributes": The root node (rank 0) holds the full array and splits it
              into smaller chunks, which are then distributed to all ranks, including rank 0.
              Each rank stores its own chunk of the data. This mode is useful when loading
              large datasets (e.g., with `np.loadtxt(...)`) and distributing parts of the data
              across ranks.

            - "all-ranks": Each rank independently generates and stores a full copy of the data.
              While this mode uses more memory, it can be simpler to implement in scenarios where
              each rank needs to work with its own independent data (e.g., generating separate
              random distributions with `np.random.randn()`).
        **kwargs
            Keyword arguments to make the non-abstract implementation
            extendable.
        """
        pass

    def plot_hist2d(self):
        pass

    def dE_max(self) -> float:
        pass

    def dt_min(self) -> float:
        pass

    def dt_max(self) -> float:
        pass

    def dE_min(self) -> float:
        pass

    def common_array_size(self) -> int:
        pass

    def rms_emittance(self) -> int:
        pass


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
            10.0,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        self.ring = copy.deepcopy(self.SR_ring)

        self.decimal = 6 if backend.float == np.float32 else 12

    def test_inputs(self):
        SRHandler = SynchrotronRadiationMaster()
        self.assertFalse(SRHandler._disable_quantum_excitation)
        self.assertEqual(
            SRHandler.track_before_element_type,
            [
                DriftBaseClass,
            ],
        )

        self.assertIsNone(SRHandler._simulation)
        self.assertIsNone(SRHandler._simulation)
        self.assertIsNone(SRHandler._natural_energy_spread)
        self.assertIsNone(SRHandler._energy_loss_per_turn)
        self.assertIsNone(SRHandler._energy_loss_per_turn)

        self.assertListEqual(SRHandler.generated_children, [])

        SRHandleriso = SynchrotronRadiationMaster(
            disable_quantum_excitation=True,
            track_before_element_type=[
                SingleHarmonicRFStation,
            ],
        )
        self.assertTrue(SRHandleriso._disable_quantum_excitation)
        self.assertEqual(
            SRHandleriso.track_before_element_type,
            [
                SingleHarmonicRFStation,
            ],
        )

    def test__str__(self):
        SRM = SynchrotronRadiationMaster()
        self.assertRegex(
            SRM.__str__(),
            (
                f"Synchrotron radiation master class set up for the"
                f" ring. \n Generated "
                f"{0} "
                f"synchrotron radiation elements."
            ),
        )

    def test_radiation_integrals_internal_setter(self):
        SRM = SynchrotronRadiationMaster()
        ring = Ring(circumference=90.65874532 * 1e3)

        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Synchrotron radiation damping "
            "and quantum excitation require"
            " either the bending radius " + "for an isomagnetic ring, or the "
            "first five synchrotron radiation "
            "integrals.",
        ):
            SRM._radiation_integrals_internal_setter(ring=ring)

        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Synchrotron radiation damping "
            "and quantum excitation require"
            " either the bending radius " + "for an isomagnetic ring, or the "
            "first five synchrotron radiation "
            "integrals.",
        ):
            SRM._radiation_integrals_internal_setter(
                ring=ring, bending_radius="not a float"
            )

        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Could not transform the input into an array",
        ):
            SRM._radiation_integrals_internal_setter(
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
            SRM._radiation_integrals_internal_setter(
                ring=ring, radiation_integrals="not an array"
            )

        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="The first five synchrotron radiation integrals are "
            "required.",
        ):
            SRM._radiation_integrals_internal_setter(
                ring=ring, radiation_integrals=[1, 2, 3, 4]
            )

    def test_set_radiation_integrals(self):
        SRM = SynchrotronRadiationMaster()
        ## ring has synchrotron radiation integrals

        ring = Ring(circumference=90.65874532 * 1e3)
        ring._radiation_integrals = self.synchrotron_radiation_integrals
        ring._momentum_compaction_factor = 0
        SRM._set_radiation_integrals(
            ring=ring,
            radiation_integrals=self.synchrotron_radiation_integrals / 10,
            bending_radius=14428.78745218723,
        )
        np.testing.assert_array_equal(
            ring.radiation_integrals,
            self.synchrotron_radiation_integrals,
        )

        SRM._set_radiation_integrals(
            ring=ring, radiation_integrals=np.zeros(5)
        )
        np.testing.assert_array_equal(
            ring.radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        SRM._set_radiation_integrals(ring=ring, bending_radius=10000)
        np.testing.assert_array_equal(
            ring.radiation_integrals,
            self.synchrotron_radiation_integrals,
        )

        ## ring does not have synchrotron radiation integrals
        ring = Ring(circumference=90.65874532 * 1e3)
        ring.add_drifts(
            n_sections=1, n_drifts_per_section=1, momentum_compaction_factor=0
        )
        SRM._set_radiation_integrals(
            ring=ring, bending_radius=14428.78745218723
        )

        np.testing.assert_array_equal(
            ring._radiation_integrals,
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
        ring = Ring(circumference=90.65874532 * 1e3)
        ring.add_drifts(
            n_sections=1, n_drifts_per_section=1, momentum_compaction_factor=0
        )
        SRM._set_radiation_integrals(
            ring=ring, radiation_integrals=self.synchrotron_radiation_integrals
        )
        np.testing.assert_array_equal(
            ring._radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        np.testing.assert_array_equal(
            ring.radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        ring = Ring(circumference=90.65874532 * 1e3)
        ring.add_drifts(
            n_sections=1, n_drifts_per_section=1, momentum_compaction_factor=0
        )
        SRM._set_radiation_integrals(
            ring=ring,
            radiation_integrals=self.synchrotron_radiation_integrals,
            bending_radius=0000,
        )
        np.testing.assert_array_equal(
            ring._radiation_integrals,
            self.synchrotron_radiation_integrals,
        )
        ring = Ring(
            circumference=90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        SRM._set_radiation_integrals(ring=ring)
        np.testing.assert_array_equal(
            ring._radiation_integrals,
            self.synchrotron_radiation_integrals,
        )

        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Radiation integrals input ignored. "
            "Using the ring's.",
        ):
            SRM._set_radiation_integrals(
                ring=ring,
                radiation_integrals=self.synchrotron_radiation_integrals,
            )
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Bending radius input ignored. "
            "Using the ring's radiation integrals.",
        ):
            SRM._set_radiation_integrals(ring=ring, bending_radius=10)

    def test_user_warning_set_radiation_integrals(self):
        SRM = SynchrotronRadiationMaster()
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Radiation integrals input ignored. "
            "Using the ring's.",
        ):
            SRM._user_warning_set_radiation_integrals(
                radiation_integrals=self.synchrotron_radiation_integrals,
                bending_radius=None,
            )
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Bending radius input ignored. "
            "Using the ring's radiation integrals.",
        ):
            SRM._user_warning_set_radiation_integrals(
                radiation_integrals=None,
                bending_radius=10e3,
            )
        self.assertIsNone(
            SRM._user_warning_set_radiation_integrals(
                radiation_integrals=None,
                bending_radius=None,
            )
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
        self.cavity.phi_rf_design = 0

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
        SRM.generated_children = [None]
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Synchrotron radiation subclasses have already been "
            "generated. Command ignored",
        ):
            SRM.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)

        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[DriftBaseClass, SingleHarmonicRFStation]
        )
        with self.assertRaisesRegex(
            expected_exception=TypeError,
            expected_regex="Unsupported list of elements.",
        ):
            SRM.prepare_ring_for_synchrotron_radiation_tracking(
                ring=ring,
            )

        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[BunchObservationMetaParams]
        )
        with self.assertRaises(
            expected_exception=TypeError,
        ):
            SRM.prepare_ring_for_synchrotron_radiation_tracking(
                ring=ring,
            )

    def test_get_share_of_radiation_integrals_drifts(self):
        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals * 10,
        )

        drift = DriftSimple(
            orbit_length=ring.circumference,
            momentum_compaction_factor=1e-4,
            section_index=0,
        )
        SRM = SynchrotronRadiationMaster()
        calculated_share_SR_int = SRM._get_share_of_radiation_integrals_drifts(
            ring=ring, drift_list=[drift]
        )
        np.testing.assert_array_equal(
            calculated_share_SR_int[0],
            self.synchrotron_radiation_integrals * 10,
        )

        drift._radiation_integrals = self.synchrotron_radiation_integrals / 100

        calculated_share_SR_int = SRM._get_share_of_radiation_integrals_drifts(
            ring=ring, drift_list=[drift]
        )
        np.testing.assert_array_equal(
            calculated_share_SR_int[0],
            self.synchrotron_radiation_integrals / 100,
        )

        drift2 = DriftSimple(
            orbit_length=ring.circumference / 4,
            momentum_compaction_factor=1e-4,
            section_index=0,
        )
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="Either all drifts should have defined radiation ",
        ):
            SRM._get_share_of_radiation_integrals_drifts(
                ring=ring, drift_list=[drift, drift2]
            )

        drift2._radiation_integrals = self.synchrotron_radiation_integrals / 5

        calculated_share_SR_int = SRM._get_share_of_radiation_integrals_drifts(
            ring=ring, drift_list=[drift, drift2]
        )

        np.testing.assert_array_equal(
            calculated_share_SR_int,
            [
                self.synchrotron_radiation_integrals / 100,
                self.synchrotron_radiation_integrals / 5,
            ],
        )

    def test_set_share_of_radiation_integrals_cavities(self):
        ring = Mock(Ring)
        ring.circumference = 90.65874532 * 1e3
        ring.radiation_integrals = self.synchrotron_radiation_integrals * 10
        cavity = SingleHarmonicRFStation()
        cavity.harmonic = 242400
        cavity.voltage = 50.1e6
        cavity.phi_rf_design = 0

        SRM = SynchrotronRadiationMaster()

        element_list = [cavity]
        calculated_share_SR_int = (
            SRM._get_share_of_radiation_integrals_cavities(
                ring=ring, cavity_list=element_list
            )
        )
        np.testing.assert_array_equal(
            calculated_share_SR_int[0],
            self.synchrotron_radiation_integrals * 10,
        )
        cavity2 = SingleHarmonicRFStation()
        cavity2._section_index = 3
        ring.section_lengths = (
            np.array([1 / 4, 1 / 3 / 4, 2 / 3 / 4, 1 / 2])
        ) * ring.circumference

        element_list = [cavity, cavity2]

        calculated_share_SR_int = (
            SRM._get_share_of_radiation_integrals_cavities(
                ring=ring, cavity_list=element_list
            )
        )
        np.testing.assert_array_equal(
            calculated_share_SR_int[0],
            self.synchrotron_radiation_integrals * 10 / 2,
        )
        np.testing.assert_array_equal(
            calculated_share_SR_int[1],
            self.synchrotron_radiation_integrals * 10 / 2,
        )

    def yolo(
        self,
        i: int,
        momentum_compaction_factor: float,
        number_of_sections: int,
        ring: Ring,
    ):
        drift = DriftSimple(
            name=f"drift{i + 1}",
            orbit_length=ring.circumference / number_of_sections,
            momentum_compaction_factor=momentum_compaction_factor
            / number_of_sections,
            section_index=i,
        )
        ring.add_element(drift, section_index=i)

    def test_generate_synchrotron_radiation_subclasses_drift_trackers(self):
        ring = Ring(
            90.65874532 * 1e3,
            radiation_integrals=self.synchrotron_radiation_integrals,
        )
        momentum_compaction_factor = 0.646747216157 / (90.65874532 * 1e3)
        self.cavity = SingleHarmonicRFStation()
        self.cavity.harmonic = 242400
        self.cavity.voltage = 50.1e6
        self.cavity.phi_rf_design = 0
        ring.add_element(self.cavity)

        number_of_sections = 5
        for i in range(number_of_sections):
            self.yolo(i, momentum_compaction_factor, number_of_sections, ring)
        SRM = SynchrotronRadiationMaster(
            track_before_element_type=[DriftBaseClass]
        )
        ring_SRdrifts = copy.deepcopy(ring)
        SRM.prepare_ring_for_synchrotron_radiation_tracking(ring=ring_SRdrifts)

        # Check the corrected number of _SynchrotronRadiationDrift trackers
        # have been generated
        SRdrifttracker_list = ring_SRdrifts.elements.get_elements(
            class_=_SynchrotronRadiationTracker
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
            sr_drift: _SynchrotronRadiationTracker = (
                ring_SRdrifts.elements.elements[1 + 2 * i]
            )
            assert sr_drift == SRM.generated_children[i]

            # verifies the synchrotron radiation integral share of each tracker
            np.testing.assert_array_almost_equal(
                sr_drift.share_of_radiation_integrals,
                self.synchrotron_radiation_integrals / 5,
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                sr_drift.share_of_radiation_integrals,
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
        self.cavity.phi_rf_design = 0
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
        SRM.prepare_ring_for_synchrotron_radiation_tracking(ring=ring_SRdrifts)

        # Check the corrected number of _SynchrotronRadiationDrift trackers
        # have been generated
        SRsectiontracker_list = ring_SRdrifts.elements.get_elements(
            class_=_SynchrotronRadiationTracker
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
        sr_drifts = ring_SRdrifts.elements.elements[1]
        assert sr_drifts == SRM.generated_children[0]

        # verifies the synchrotron radiation integral share of each tracker
        np.testing.assert_array_almost_equal(
            sr_drifts.share_of_radiation_integrals,
            self.synchrotron_radiation_integrals,
            decimal=self.decimal,
        )
        np.testing.assert_array_almost_equal(
            sr_drifts.radiation_integrals_tracker,
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
            rf_station.phi_rf_design = 0
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
        SRM.prepare_ring_for_synchrotron_radiation_tracking(ring=ring_SRdrifts)

        # Check the corrected number of _SynchrotronRadiationDrift trackers
        # have been generated
        SRsectiontracker_list = ring_SRdrifts.elements.get_elements(
            class_=_SynchrotronRadiationTracker
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
                ].share_of_radiation_integrals,
                self.synchrotron_radiation_integrals / number_of_sections,
                decimal=self.decimal,
            )
            np.testing.assert_array_almost_equal(
                ring_SRdrifts.elements.elements[
                    1 + 3 * i
                ].radiation_integrals_tracker,
                self.synchrotron_radiation_integrals / number_of_sections,
                decimal=self.decimal,
            )

    def test_compute_synchrotron_radiation_parameters(self):
        radiation_integrals = np.array(
            [
                0.646747216157,
                0.000593654931851,
                5.6814536525e-08,
                5.92870407301e-09,
                1.698280783e-11,
            ]
        )
        ring = Ring(
            circumference=90.65874532 * 1e3,
            radiation_integrals=radiation_integrals,
        )
        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)

        beam.reference.total_energy = 20e9
        beam.particle_type = positron

        SRM = SynchrotronRadiationMaster()

        SRM.compute_synchrotron_radiation_parameters(ring=ring, beam=beam)

        self.assertAlmostEqual(
            SRM._energy_loss_per_turn,
            1337317.6296824566,
            places=self.decimal,
        )
        self.assertAlmostEqual(
            SRM._longitudinal_damping_time,
            14955.235531740671,
            places=self.decimal,
        )
        self.assertAlmostEqual(
            SRM._natural_energy_spread,
            0.00016759685785477585,
            places=self.decimal,
        )

        self.assertEqual(SRM._natural_energy_spread, SRM.natural_energy_spread)
        self.assertEqual(
            SRM._longitudinal_damping_time, SRM.longitudinal_damping_time
        )
        self.assertEqual(SRM._energy_loss_per_turn, SRM.energy_loss_per_turn)

    def test_print_synchrotron_radiation_parameters(self):
        SRM = SynchrotronRadiationMaster()
        radiation_integrals = np.array(
            [
                0.646747216157,
                0.000593654931851,
                5.6814536525e-08,
                5.92870407301e-09,
                1.698280783e-11,
            ]
        )
        ring = Ring(
            circumference=90.65874532 * 1e3,
            radiation_integrals=radiation_integrals,
        )

        beam = Mock(BeamBaseClass)
        beam.reference = Mock(ReferenceCoordinates)

        beam.reference.total_energy = 20e9
        beam.particle_type = positron

        self.assertRegex(
            SRM.print_synchrotron_radiation_parameters(ring=ring, beam=beam),
            expected_regex=(
                f"Synchrotron radiation parameters for the beam energy "
                f"{beam.reference.total_energy}"
                + f"Energy lost: {1337317.6296824566} eV per turn,"
                + f"Longitudinal damping time:"
                f" {14955.235531740671} turns,"
                + f"Natural energy spread: {0.00016759685785477585}"
            ),
        )

    def test_schedule(self):
        SRM = SynchrotronRadiationMaster()
        radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        ring = Ring(
            circumference=90.65874532 * 1e3,
            radiation_integrals=radiation_integrals,
        )
        momentum_compaction_factor = 0.646747216157 / (90.65874532 * 1e3)

        number_of_sections = 4
        number_of_turns = 10
        for i in range(number_of_sections):
            rf_station = SingleHarmonicRFStation(section_index=i)
            rf_station.harmonic = 242400
            rf_station.voltage = 50.1e6
            rf_station.phi_rf_design = 0
            ring.add_element(rf_station)
            drift = DriftSimple(
                name=f"drift{i + 1}",
                orbit_length=ring.circumference / number_of_sections,
                momentum_compaction_factor=momentum_compaction_factor
                / number_of_sections,
                section_index=i,
            )
            ring.add_element(drift, section_index=i)
        SRM.prepare_ring_for_synchrotron_radiation_tracking(
            ring=ring, radiation_integrals=radiation_integrals
        )
        SRM.schedule(
            attribute="share_of_radiation_integrals",
            value=np.array(
                [
                    radiation_integrals * 1 / (k + 1)
                    for k in range(number_of_turns)
                ]
            ),
        )
        beam = BeamBaseClassTester(
            intensity=1e12,
            particle_type=electron,
            is_counter_rotating=False,
            is_distributed=False,
        )
        for i, SRClass_child in enumerate(SRM.generated_children):
            self.assertTrue(
                "share_of_radiation_integrals"
                in SRClass_child.intended_for_scheduling
            )
            for k in range(number_of_turns):
                SRClass_child.apply_schedules(
                    turn_i=k,
                    reference_time=float(beam.reference.time),
                )
                np.testing.assert_array_almost_equal(
                    SRClass_child.share_of_radiation_integrals,
                    1 / (k + 1) * radiation_integrals / number_of_sections,
                    decimal=self.decimal,
                )
