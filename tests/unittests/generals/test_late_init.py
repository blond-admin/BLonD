import copy
import unittest

from blond.generals.exceptions_ import NotInitialisedError
from blond.generals.late_init import _LateInit, late_init_attributes
from blond.testing.backend_testing import BLonDTestCase


class _Owner:
    energy: _LateInit[float] = _LateInit("_Owner.setup()")
    time: _LateInit[float] = _LateInit("_Owner.setup()")
    plain = 0.0

    def setup(self, energy: float) -> None:
        self.energy = energy


class _Child(_Owner):
    extra: _LateInit[int] = _LateInit("_Child.setup()")


class _Mixin:
    offset: _LateInit[float] = _LateInit("_Mixin.setup()")


class _MultiOwner(_Owner, _Mixin):
    pass


class _Redeclarer(_Owner):
    energy: _LateInit[float] = _LateInit("_Redeclarer.setup()")


class _NoLateInit:
    plain = 0.0


class _ExplodingLateInit(_LateInit):
    def __get__(self, instance, owner=None):
        raise RuntimeError("descriptor consulted")


class TestLateInit(BLonDTestCase):
    def test_read_before_set_raises(self):
        owner = _Owner()
        with self.assertRaises(NotInitialisedError) as context:
            _ = owner.energy
        self.assertIn("_Owner.energy", str(context.exception))
        self.assertIn("_Owner.setup()", str(context.exception))

    def test_hasattr_reports_fill_state(self):
        owner = _Owner()
        self.assertNotHasAttr(owner, "energy")
        owner.setup(1.0)
        self.assertHasAttr(owner, "energy")

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
        self.assertNotHasAttr(second, "energy")

    def test_direct_assignment(self):
        owner = _Owner()
        self.assertNotHasAttr(owner, "energy")
        owner.energy = 1
        self.assertHasAttr(owner, "energy")

    def test_delete_resets(self):
        owner = _Owner()
        owner.setup(1.0)
        del owner.energy
        self.assertNotHasAttr(owner, "energy")
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
        self.assertNotHasAttr(_LateInit, "__set__")
        self.assertNotHasAttr(_LateInit, "__delete__")

    def test_filled_read_bypasses_descriptor(self):
        owner = _Owner()
        owner.setup(1.0)
        descriptor = vars(_Owner)["energy"]
        descriptor.__class__ = _ExplodingLateInit
        try:
            self.assertEqual(owner.energy, 1.0)
        finally:
            descriptor.__class__ = _LateInit

    def test_class_access_returns_descriptor(self):
        self.assertIsInstance(_Owner.energy, _LateInit)

    def test_doc_is_none_so_sphinx_skips_the_class_docstring(self):
        self.assertIsNone(_Owner.energy.__doc__)

    def test_deepcopy(self):
        owner = _Owner()
        self.assertNotHasAttr(copy.deepcopy(owner), "energy")
        owner.setup(3.0)
        self.assertEqual(copy.deepcopy(owner).energy, 3.0)


class TestLateInitAttributes(BLonDTestCase):
    def test_lists_declared_names_in_declaration_order(self):
        self.assertEqual(late_init_attributes(_Owner), ("energy", "time"))

    def test_accepts_an_instance(self):
        self.assertEqual(late_init_attributes(_Owner()), ("energy", "time"))

    def test_empty_without_declarations(self):
        self.assertEqual(late_init_attributes(_NoLateInit), ())

    def test_includes_inherited(self):
        self.assertEqual(
            sorted(late_init_attributes(_Child)),
            ["energy", "extra", "time"],
        )

    def test_includes_every_base_under_multiple_inheritance(self):
        self.assertEqual(
            sorted(late_init_attributes(_MultiOwner)),
            ["energy", "offset", "time"],
        )

    def test_redeclaring_an_inherited_name_does_not_duplicate(self):
        self.assertEqual(
            sorted(late_init_attributes(_Redeclarer)),
            ["energy", "time"],
        )

    def test_independent_of_fill_state(self):
        owner = _Owner()
        declared = late_init_attributes(owner)
        owner.setup(1.0)
        self.assertEqual(late_init_attributes(owner), declared)


if __name__ == "__main__":
    unittest.main()
