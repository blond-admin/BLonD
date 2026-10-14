import gc
import unittest
import weakref
from copy import deepcopy

from blond.core.offenders import (
    WeakBackReference,
    hold_strongly,
)
from blond.testing.backend_testing import BLonDTestCase


class _Parent:
    pass


class _Child:
    parent = WeakBackReference()
    other = WeakBackReference()


class TestWeakBackReference(BLonDTestCase):
    def setUp(self):
        gc.disable()
        self.addCleanup(gc.enable)

    def test_unset_reads_none(self):
        self.assertIsNone(_Child().parent)

    def test_reads_back_assigned(self):
        parent = _Parent()
        child = _Child()
        child.parent = parent
        self.assertIs(child.parent, parent)

    def test_assign_none(self):
        child = _Child()
        child.parent = _Parent()
        child.parent = None
        self.assertIsNone(child.parent)

    def test_does_not_keep_referent_alive(self):
        parent = _Parent()
        parent.child = _Child()
        parent.child.parent = parent
        parent_ref = weakref.ref(parent)
        del parent
        self.assertIsNone(parent_ref())

    def test_read_after_referent_freed_raises(self):
        # Using an element after dropping its simulation must say so,
        # instead of failing later on an attribute of None.
        child = _Child()
        child.parent = _Parent()
        with self.assertRaisesRegex(ReferenceError, "_Child.parent"):
            _ = child.parent

    def test_deepcopy_points_to_the_copied_referent(self):
        parent = _Parent()
        parent.child = _Child()
        parent.child.parent = parent
        parent_copy = deepcopy(parent)
        self.assertIs(parent_copy.child.parent, parent_copy)

    def test_deepcopy_points_to_a_referent_copied_later(self):
        # The child may be copied before its referent, e.g. a feedback
        # listed before its RF stations in the same ring.
        parent = _Parent()
        child = _Child()
        child.parent = parent
        child_copy, parent_copy = deepcopy([child, parent])
        self.assertIs(child_copy.parent, parent_copy)


class TestHoldStrongly(BLonDTestCase):
    # Without a simulation (e.g. `headless` constructors) nothing else owns
    # the stand-ins assigned to back-references, so the object must.
    def setUp(self):
        gc.disable()
        self.addCleanup(gc.enable)

    def test_keeps_referent_alive(self):
        child = _Child()
        hold_strongly(child, "parent")
        child.parent = _Parent()
        self.assertIsInstance(child.parent, _Parent)

    def test_converts_already_assigned_reference(self):
        parent = _Parent()
        child = _Child()
        child.parent = parent
        hold_strongly(child, "parent")
        del parent
        self.assertIsInstance(child.parent, _Parent)

    def test_other_attributes_stay_weak(self):
        child = _Child()
        hold_strongly(child, "other")
        child.parent = _Parent()
        with self.assertRaises(ReferenceError):
            _ = child.parent

    def test_other_instances_stay_weak(self):
        hold_strongly(_Child(), "parent")
        child = _Child()
        child.parent = _Parent()
        with self.assertRaises(ReferenceError):
            _ = child.parent

    def test_deepcopy_points_to_the_copied_referent(self):
        parent = _Parent()
        parent.child = _Child()
        hold_strongly(parent.child, "parent")
        parent.child.parent = parent
        parent_copy = deepcopy(parent)
        self.assertIs(parent_copy.child.parent, parent_copy)


if __name__ == "__main__":
    unittest.main()
