import copy
import unittest

from blond.generals.exceptions_ import BLonDException
from blond.generals.late_init import (
    LateInit,
    NotInitialisedError,
    check_filled,
    reset_late_init,
    unfilled,
)
from blond.testing.backend_testing import BLonDTestCase


class _Owner:
    energy: LateInit[float] = LateInit(
        "_Owner.setup()", doc="Energy offset [eV]."
    )
    time: LateInit[float] = LateInit("_Owner.setup()")
    plain = 0.0

    def setup(self, energy: float) -> None:
        self.energy = energy


class _Child(_Owner):
    extra: LateInit[int] = LateInit("_Child.setup()")


class _ExplodingLateInit(LateInit):
    def __get__(self, instance, owner=None):
        raise RuntimeError("descriptor consulted")


class TestLateInit(BLonDTestCase):
    def test_read_before_set_raises(self):
        owner = _Owner()
        with self.assertRaises(NotInitialisedError) as context:
            _ = owner.energy
        self.assertIn("_Owner.energy", str(context.exception))
        self.assertIn("_Owner.setup()", str(context.exception))

    def test_error_is_attribute_error_and_blond_exception(self):
        self.assertTrue(issubclass(NotInitialisedError, AttributeError))
        self.assertTrue(issubclass(NotInitialisedError, BLonDException))

    def test_hasattr_reports_fill_state(self):
        owner = _Owner()
        self.assertFalse(hasattr(owner, "energy"))
        owner.setup(1.0)
        self.assertTrue(hasattr(owner, "energy"))

    def test_read_after_set(self):
        owner = _Owner()
        owner.setup(1.5)
        self.assertEqual(owner.energy, 1.5)

    def test_value_stored_under_public_name(self):
        owner = _Owner()
        owner.setup(2.0)
        self.assertEqual(vars(owner), {"energy": 2.0})

    def test_instances_are_independent(self):
        first, second = _Owner(), _Owner()
        first.setup(1.0)
        self.assertFalse(hasattr(second, "energy"))

    def test_delete_resets(self):
        owner = _Owner()
        owner.setup(1.0)
        del owner.energy
        self.assertFalse(hasattr(owner, "energy"))
        with self.assertRaises(AttributeError):
            del owner.energy

    def test_error_carries_name_and_object(self):
        owner = _Owner()
        with self.assertRaises(NotInitialisedError) as context:
            _ = owner.energy
        self.assertEqual(context.exception.name, "energy")
        self.assertIs(context.exception.obj, owner)

    def test_is_non_data_descriptor(self):
        # A `__set__` or `__delete__` would route every read of a filled
        # attribute through the Python-level `__get__`, several times
        # slower than a plain attribute read.
        self.assertFalse(hasattr(LateInit, "__set__"))
        self.assertFalse(hasattr(LateInit, "__delete__"))

    def test_filled_read_bypasses_descriptor(self):
        owner = _Owner()
        owner.setup(1.0)
        descriptor = vars(_Owner)["energy"]
        descriptor.__class__ = _ExplodingLateInit
        try:
            self.assertEqual(owner.energy, 1.0)
        finally:
            descriptor.__class__ = LateInit

    def test_reset_late_init_clears_filled_and_unfilled(self):
        owner = _Owner()
        owner.setup(1.0)
        reset_late_init(owner, "energy", "time")
        self.assertFalse(hasattr(owner, "energy"))
        self.assertFalse(hasattr(owner, "time"))
        # Unfilled attributes are skipped silently
        reset_late_init(owner, "energy", "time")

    def test_reset_late_init_rejects_other_attributes(self):
        owner = _Owner()
        with self.assertRaises(TypeError):
            reset_late_init(owner, "plain")
        with self.assertRaises(TypeError):
            reset_late_init(owner, "missing")

    def test_unfilled_lists_unfilled_in_declaration_order(self):
        owner = _Owner()
        self.assertEqual(unfilled(owner), ("energy", "time"))
        owner.setup(1.0)
        self.assertEqual(unfilled(owner), ("time",))
        owner.time = 2.0
        self.assertEqual(unfilled(owner), ())

    def test_unfilled_includes_inherited(self):
        self.assertEqual(
            sorted(unfilled(_Child())), ["energy", "extra", "time"]
        )

    def test_check_filled_raises_for_unfilled(self):
        owner = _Owner()
        owner.setup(1.0)
        with self.assertRaisesRegex(NotInitialisedError, "_Owner.time"):
            check_filled(owner)
        owner.time = 2.0
        check_filled(owner)

    def test_class_access_returns_descriptor(self):
        self.assertIsInstance(_Owner.energy, LateInit)
        self.assertEqual(_Owner.energy.__doc__, "Energy offset [eV].")

    def test_deepcopy(self):
        owner = _Owner()
        self.assertFalse(hasattr(copy.deepcopy(owner), "energy"))
        owner.setup(3.0)
        self.assertEqual(copy.deepcopy(owner).energy, 3.0)


if __name__ == "__main__":
    unittest.main()
