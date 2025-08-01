import unittest
from unittest.mock import Mock

import numpy as np

from blond3 import Simulation
from blond3._core.base import BeamPhysicsRelevant
from blond3._core.beam.base import BeamBaseClass
from blond3._core.ring.beam_physics_relevant_elements import (
    pprint,
    BeamPhysicsRelevantElements,
)
from blond3.physics.cavities import CavityBaseClass
from blond3.physics.drifts import DriftBaseClass


class TestFunctions(unittest.TestCase):
    def test_pprint_executes(self):
        pprint(v=np.array(10))


class TestBeamPhysicsRelevantElements(unittest.TestCase):
    def setUp(self):
        self.beam_physics_relevant_elements = BeamPhysicsRelevantElements()
        element1 = Mock(spec=DriftBaseClass)
        element1.orbit_length = 0.5
        element1.section_index = 0
        element1.name = "element1"
        self.beam_physics_relevant_elements.add_element(element1)

        element2 = Mock(spec=CavityBaseClass)
        element2.section_index = 0
        element2.name = "element2"
        self.beam_physics_relevant_elements.add_element(element2)

        element3 = Mock(spec=DriftBaseClass)
        element3.orbit_length = 0.5
        element3.section_index = 1
        element3.name = "element3"
        self.beam_physics_relevant_elements.add_element(element3)

        element4 = Mock(spec=CavityBaseClass)
        element4.section_index = 1
        element4.name = "element4"
        self.beam_physics_relevant_elements.add_element(element4)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test__check_section_indexing(self):
        self.beam_physics_relevant_elements._check_section_indexing()

    def test_add_element(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 0
        # asert that element is inserted at the end of section 0,
        # which has already 2 elements
        self.beam_physics_relevant_elements.add_element(element=element)
        assert self.beam_physics_relevant_elements.elements[2] is element

    def test_add_element2(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 1
        # asert that element is inserted at the end of section 0,
        # which has already 2 elements
        self.beam_physics_relevant_elements.add_element(element=element)
        assert self.beam_physics_relevant_elements.elements[-1] is element

    def test_insert_element(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 0
        self.beam_physics_relevant_elements.insert(element=element,
                                                   insert_at=0)
        assert self.beam_physics_relevant_elements.elements[0] is element

    def test_check_insertion_compatibility(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 1
        with self.assertRaises(AssertionError,
                               msg='The element section index is incompatible '
                                 'with the requested location. Please allow '
                                 'overwrite for automatic handling.'):
            self.beam_physics_relevant_elements.insert(element=element,
                                                   insert_at=0)
        element.section_index = 1
        with self.assertRaises(AssertionError,
                               msg='The element section index is incompatible '
                                   'with the requested location. Please allow '
                                   'overwrite for automatic handling.'):
            self.beam_physics_relevant_elements.insert(element=element,
                                                       insert_at=1)
        element.section_index = 0
        with self.assertRaises(AssertionError,
                               msg='The element section index is incompatible '
                                   'with the requested location. Please allow '
                                   'overwrite for automatic handling.'):
            self.beam_physics_relevant_elements.insert(element=element,
                                                       insert_at=len(self.beam_physics_relevant_elements.elements))
        element.section_index = 50
        with self.assertRaises(AssertionError,
                               msg=f'The element must be inserted within ['
                                 f'0:{len(self.beam_physics_relevant_elements.elements)+1}] indexes. '):
            self.beam_physics_relevant_elements.insert(element=element,
                                                       insert_at=len(
                                                           self.beam_physics_relevant_elements.elements))

    def test_count(self):
        assert (
            self.beam_physics_relevant_elements.count(
                class_=DriftBaseClass, section_i=0
            )
            == 1
        )
        assert (
            self.beam_physics_relevant_elements.count(
                class_=CavityBaseClass, section_i=0
            )
            == 1
        )
        assert (
            self.beam_physics_relevant_elements.count(
                class_=BeamPhysicsRelevant, section_i=0
            )
            == 2
        )
        assert (
            self.beam_physics_relevant_elements.count(
                class_=CavityBaseClass, section_i=1
            )
            == 1
        )
        assert (
            self.beam_physics_relevant_elements.count(
                class_=BeamPhysicsRelevant, section_i=None
            )
            == 4
        )

    def test_get_element(self):
        self.beam_physics_relevant_elements.get_element(
            class_=DriftBaseClass, section_i=0
        )
        with self.assertRaises(AssertionError):
            self.beam_physics_relevant_elements.get_element(
                class_=DriftBaseClass, section_i=None
            )

    def test_get_elements(self):
        elements = self.beam_physics_relevant_elements.get_elements(
            class_=DriftBaseClass, section_i=None
        )
        assert len(elements) == 2

    def test_get_order_info(self):
        self.beam_physics_relevant_elements.get_order_info()

    def test_get_section_circumference_orbit_lengths(self):
        orbit_length = self.beam_physics_relevant_elements.get_sections_orbit_length()
        self.assertEqual(orbit_length[0], 0.5)
        self.assertEqual(orbit_length[1], 0.5)

    def test_get_sections_indices(self):
        indices = self.beam_physics_relevant_elements.get_sections_indices()
        self.assertEqual(indices, (0, 1))

    def test_n_elements(self):
        self.assertEqual(4, self.beam_physics_relevant_elements.n_elements)

    def test_n_sections(self):
        self.assertEqual(2, self.beam_physics_relevant_elements.n_sections)

    def test_on_init_simulation(self):
        simulation = Mock(spec=Simulation)

        self.beam_physics_relevant_elements.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self):
        simulation = Mock(spec=Simulation)
        beam = Mock(spec=BeamBaseClass)
        self.beam_physics_relevant_elements.on_run_simulation(
            simulation=simulation, n_turns=10, turn_i_init=0, beam=beam
        )

    def test_print_order(self):
        self.beam_physics_relevant_elements.print_order()

    def test_reorder(self):
        self.beam_physics_relevant_elements.reorder()
        expected = (
            "CavityBaseClass",
            "DriftBaseClass",
            "CavityBaseClass",
            "DriftBaseClass",
        )
        actual = tuple(
            [
                e._spec_class.__name__
                for e in self.beam_physics_relevant_elements.elements
            ]
        )
        self.assertEqual(expected, actual)

    def test_reorder_section(self):
        # TODO: implement test for `reorder_section`
        self.beam_physics_relevant_elements.reorder_section(section_index=0)
        # self.beam_physics_relevant_elements.reorder_section(section_index=1)
        expected = (
            "CavityBaseClass",
            "DriftBaseClass",
            "DriftBaseClass",
            "CavityBaseClass",
        )
        actual = tuple(
            [
                e._spec_class.__name__
                for e in self.beam_physics_relevant_elements.elements
            ]
        )
        section = tuple(
            [e.section_index for e in self.beam_physics_relevant_elements.elements]
        )
        self.assertEqual((0, 0, 1, 1), section)
        self.assertEqual(expected, actual)

    def test_reorder_sections(self):
        # TODO: implement test for `reorder_section`
        self.beam_physics_relevant_elements.reorder_section(section_index=0)
        self.beam_physics_relevant_elements.reorder_section(section_index=1)
        expected = (
            "CavityBaseClass",
            "DriftBaseClass",
            "CavityBaseClass",
            "DriftBaseClass",
        )
        actual = tuple(
            [
                e._spec_class.__name__
                for e in self.beam_physics_relevant_elements.elements
            ]
        )
        section = tuple(
            [e.section_index for e in self.beam_physics_relevant_elements.elements]
        )
        self.assertEqual((0, 0, 1, 1), section)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
