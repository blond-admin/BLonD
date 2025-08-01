import unittest
from unittest.mock import Mock

import numpy as np

from blond3 import Ring, Simulation
from blond3._core.base import BeamPhysicsRelevant
from blond3._core.beam.base import BeamBaseClass
from blond3.physics.cavities import CavityBaseClass
from blond3.physics.drifts import DriftBaseClass

class BeamPhysicsRelevantHelper(BeamPhysicsRelevant):
    def track(self, beam: BeamBaseClass) -> None:
        pass

    def on_init_simulation(self, simulation: Simulation) -> None:
        pass

    def on_run_simulation(self, simulation: Simulation, beam: BeamBaseClass,
                          n_turns: int, turn_i_init: int, **kwargs) -> None:
        pass


class TestRing(unittest.TestCase):
    def setUp(self):
        # TODO: implement test for `__init__`
        self.ring = Ring(10.0)

    def test___init__(self):
        pass  # calls __init__ in  self.setUp

    def test_circumference(self):
        ring = Ring(10.0)
        self.assertTrue(np.isclose(ring.circumference, 10.0))
        self.assertTrue(np.isclose(ring.closed_orbit_length, 0.0))

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
        assert self.ring.elements.elements == [cavity, drift]

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
        element3 = Mock(spec=BeamPhysicsRelevant)
        element3.section_index = 0
        location = 1
        self.ring.insert_element(
            element=element3,
            insert_at=location,
            deepcopy=False,
        )
        assert self.ring.elements.elements[1] is element3

    def test_insert_element_several_locations(self):
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
        element3 = Mock(spec=BeamPhysicsRelevant)
        element3.section_index = 0
        location = [0, 2]
        locations_in_the_new_ring = self.ring.insert_element(
            element=element3,
            insert_at=location,
            deepcopy=False,
            allow_section_index_overwrite=False,
        )
        assert (self.ring.elements.elements[locations_in_the_new_ring[0]] is 
                element3)
        assert (self.ring.elements.elements[locations_in_the_new_ring[1]] is 
                element3)

        element4 = Mock(spec=BeamPhysicsRelevant)
        element4.section_index = 5
        location = [0, 2]

        with self.assertRaises(AssertionError,
                               msg='The element section index is incompatible '
                                   'with the requested location. Please allow '
                                   'overwrite for automatic handling.'):
            self.ring.insert_element(
                element=element4,
                insert_at=location,
                deepcopy=False,
                allow_section_index_overwrite=False,
            )

        with self.assertRaises(AssertionError,
                               msg='Cannot overwrite the section indexes with '
                          'deepcopy == False.'):
            self.ring.insert_element(
                element=element4,
                insert_at=location,
                deepcopy=False,
                allow_section_index_overwrite=True,
            )
        element4 = Mock(spec=BeamPhysicsRelevant)
        element4.section_index = 5
        location = [1,2,5]
        with self.assertRaises(AssertionError,
                               msg=f'The element must be inserted within ['
                                 f'0:{len(self.ring.elements.elements)}] indexes.'):
            self.ring.insert_element(
                element=element4,
                insert_at=location,
                deepcopy=False,
                allow_section_index_overwrite=True,
            )

    def test_insert_element_section_compatibility(self):
        element1 = BeamPhysicsRelevantHelper()
        element2 = BeamPhysicsRelevantHelper()
        element3 = BeamPhysicsRelevantHelper()
        element4 = BeamPhysicsRelevantHelper()
        element1._section_index = 0
        element2._section_index = 1
        element3._section_index = 2
        element4._section_index = 3

        self.ring.add_elements(
            elements=[element1, element2, element3, element4],
            reorder=False,
            deepcopy=False,
            section_index=None,
        )

        element5 = Mock(spec=BeamPhysicsRelevant)
        element5._section_index = 10
        location = [0, 2, 4]
        locations_in_the_new_ring = self.ring.insert_element(
            element=element3,
            insert_at=location,
            deepcopy=True,
            allow_section_index_overwrite=True,
        )
        assert self.ring.elements.elements[locations_in_the_new_ring[0]]._section_index == 0
        assert self.ring.elements.elements[locations_in_the_new_ring[
            1]]._section_index == 2
        assert self.ring.elements.elements[locations_in_the_new_ring[
            2]]._section_index == 3

    def test_insert_elements(self):
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
        element3 = Mock(spec=BeamPhysicsRelevant)
        element3.section_index = 0
        element4 = Mock(spec=BeamPhysicsRelevant)
        element4.section_index = 0
        location = 1
        self.ring.insert_elements(
            elements=[element3, element4],
            insert_at=location,
            deepcopy=False,
        )
        assert self.ring.elements.elements[1] is element3
        assert self.ring.elements.elements[2] is element4


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
        element3 = Mock(spec=BeamPhysicsRelevant)
        element3.section_index = 0
        location = 1
        self.ring.insert_element(
            element=element3,
            insert_at=location,
            deepcopy=False,
        )
        assert self.ring.elements.elements[1] is element3

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
        beam = Mock(spec=BeamBaseClass)

        self.ring.on_run_simulation(
            simulation=simulation,
            n_turns=10,
            turn_i_init=5,
            beam=beam,
        )

    def test_effective_circumference(self):
        drift = Mock(spec=DriftBaseClass)
        drift.orbit_length = 123
        cavity = Mock(spec=CavityBaseClass)
        drift.section_index = 0
        cavity.section_index = 0
        self.ring.add_elements((drift, cavity))
        self.assertEqual(123, self.ring.closed_orbit_length)

    def test_effective_circumference2(self):
        drift = Mock(spec=DriftBaseClass)
        drift.orbit_length = 123
        drift2 = Mock(spec=DriftBaseClass)
        drift2.orbit_length = 123
        cavity = Mock(spec=CavityBaseClass)
        drift.section_index = 0
        drift2.section_index = 0
        cavity.section_index = 0
        self.ring.add_elements((drift, drift2, cavity))
        self.assertEqual(2 * 123, self.ring.closed_orbit_length)

    def test_assert_circumference(self):
        with self.assertRaises(AssertionError):
            self.ring._circumference = 12
            drift = Mock(spec=DriftBaseClass)
            drift.orbit_length = 123
            drift2 = Mock(spec=DriftBaseClass)
            drift2.orbit_length = 123
            cavity = Mock(spec=CavityBaseClass)
            drift.section_index = 0
            drift2.section_index = 0
            cavity.section_index = 0
            self.ring.add_elements((drift, drift2, cavity))
            self.ring.assert_circumference()  # fails
        self.ring._circumference = 2 * 123
        self.ring.assert_circumference()  # works

    def test_add_drifts2(self):
        self.ring._circumference = 129
        self.ring.add_drifts(12, 3)
        self.assertEqual(3 * 12, self.ring.elements.n_elements)
        self.assertEqual(3, self.ring.elements.n_sections)
        self.ring.assert_circumference()  # works


if __name__ == "__main__":
    unittest.main()
