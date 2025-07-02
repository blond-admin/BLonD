import unittest
from unittest.mock import Mock

import numpy as np

from blond3 import Ring, Simulation
from blond3._core.base import BeamPhysicsRelevant
from blond3.physics.cavities import CavityBaseClass
from blond3.physics.drifts import DriftBaseClass


class TestRing(unittest.TestCase):
    def setUp(self):
        # TODO: implement test for `__init__`
        self.ring = Ring(circumference=23, bending_radius=None)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_circumference(self):
        ring = Ring(10.0)
        self.assertTrue(np.isclose(ring._circumference, 10.0))
        self.assertTrue(np.isclose(ring.circumference, 10.0))

    def test_bending_radius(self):
        ring = Ring(10.0, bending_radius=2.0)
        self.assertEqual(ring._bending_radius, 2.0)
        self.assertEqual(ring.bending_radius, 2.0)

    def test_bending_radius_calculated_if_none(self):
        ring = Ring(2 * np.pi)  # Should produce radius 1.0
        expected_radius = 1.0
        self.assertTrue(np.isclose(ring._bending_radius, expected_radius))

    def test_add_element_fails_no_section_index(self):
        with self.assertRaises(AssertionError):
            element = Mock(spec=BeamPhysicsRelevant)
            self.ring.add_element(
                element=element,
                reorder=False,
                deepcopy=False,
                section_index=None,
            )

    def test_add_element(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 0
        self.ring.add_element(
            element=element,
            reorder=False,
            deepcopy=False,
            section_index=None,
        )
        assert self.ring.elements.elements[0] is element

    def test_add_element_section_index(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 0
        self.ring.add_element(
            element=element,
            reorder=False,
            deepcopy=False,
            section_index=10,
        )
        self.assertEqual(element._section_index, 10)

    def test_add_element_deepcopy(self):
        element = Mock(spec=BeamPhysicsRelevant)
        element.section_index = 0
        initial_element = element
        self.ring.add_element(
            element=initial_element,
            reorder=False,
            deepcopy=True,
            section_index=None,
        )
        assert self.ring.elements.elements[0] is not element

    def test_add_element_reorder(self):
        drift = Mock(spec=DriftBaseClass)
        cavity = Mock(spec=CavityBaseClass)
        drift.section_index = 0
        cavity.section_index = 0

        self.ring.add_element(
            element=drift,
            reorder=False,
            deepcopy=False,
            section_index=None,
        )
        self.ring.add_element(
            element=cavity,
            reorder=True,
            deepcopy=False,
            section_index=None,
        )
        assert self.ring.elements.elements == (cavity, drift)

    def test_add_elements(self):
        element1 = Mock(spec=BeamPhysicsRelevant)
        element2 = Mock(spec=BeamPhysicsRelevant)
        element1.section_index = 0
        element2.section_index = 0

        self.ring.add_elements(
            elements=[element1, element2],
            reorder=False,
            deepcopy=False,
            section_index=None,
        )

        assert self.ring.elements.elements[0] is element1
        assert self.ring.elements.elements[1] is element2

    def test_elements(self):
        element1 = Mock(spec=BeamPhysicsRelevant)
        element1.section_index = 0
        self.ring.add_elements([element1 for i in range(10)], deepcopy=True)
        self.assertEqual(self.ring.elements.n_elements, 10)
        for element in self.ring.elements.elements:
            assert element is not element1

    def test_n_cavities(self):
        element1 = Mock(spec=BeamPhysicsRelevant)
        element1.section_index = 0
        cavity1 = Mock(spec=CavityBaseClass)
        cavity1.section_index = 0
        self.ring.add_elements(
            [element1 for i in range(10)] + [cavity1 for i in range(10)], deepcopy=True
        )
        self.assertEqual(self.ring.n_cavities, 10)

    def test_on_init_simulation(self):
        simulation = Mock(spec=Simulation)
        drift = Mock(spec=DriftBaseClass)
        drift.section_index = 0
        drift.share_of_circumference = 1
        cavity = Mock(spec=CavityBaseClass)
        cavity.section_index = 0
        self.ring.add_elements((drift, cavity))

        self.ring.on_init_simulation(simulation=simulation)

    def test_on_run_simulation(self):
        simulation = Mock(spec=Simulation)
        self.ring.on_run_simulation(simulation=simulation, n_turns=10, turn_i_init=5)

if __name__ == "__main__":
    unittest.main()
