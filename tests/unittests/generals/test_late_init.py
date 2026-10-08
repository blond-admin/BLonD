import copy
import unittest

from blond.generals.exceptions_ import NotInitialisedError
from blond.generals.late_init import (
    AssignedDuringTracking,
    InitalisedInternally,
    SetBy,
    ToBeDefined,
    _LateInit,
    late_init_attributes,
)
from blond.testing.backend_testing import BLonDTestCase


class _Owner:
    energy: InitalisedInternally[float] = InitalisedInternally(
        "`_Owner.setup()`"
    )
    time: InitalisedInternally[float] = InitalisedInternally(
        "`_Owner.setup()`"
    )
    plain = 0.0

    def setup(self, energy: float) -> None:
        self.energy = energy


class _Child(_Owner):
    extra: InitalisedInternally[int] = InitalisedInternally("`_Child.setup()`")


class _Mixin:
    offset: InitalisedInternally[float] = InitalisedInternally(
        "`_Mixin.setup()`"
    )


class _MultiOwner(_Owner, _Mixin):
    pass


class _Redeclarer(_Owner):
    energy: InitalisedInternally[float] = InitalisedInternally(
        "`_Redeclarer.setup()`"
    )


class _NoLateInit:
    plain = 0.0


class _Categorised:
    by_framework: InitalisedInternally[float] = InitalisedInternally(
        "a setup hook"
    )
    by_tracking: AssignedDuringTracking[float] = AssignedDuringTracking(
        "`track()`"
    )
    by_user: ToBeDefined[float] = ToBeDefined("the user")


class _ExplodingLateInit(_LateInit):
    def __get__(self, instance, owner=None):
        raise RuntimeError("descriptor consulted")


class TestLateInit(BLonDTestCase):
    def test_read_before_set_raises(self):
        owner = _Owner()
        with self.assertRaises(NotInitialisedError) as context:
            _ = owner.energy
        self.assertIn("_Owner.energy", str(context.exception))
        self.assertIn("`_Owner.setup()`", str(context.exception))

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
        original = type(descriptor)
        descriptor.__class__ = _ExplodingLateInit
        try:
            self.assertEqual(owner.energy, 1.0)
        finally:
            descriptor.__class__ = original

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


class TestLateInitSubclasses(BLonDTestCase):
    def test_each_category_is_a_late_init(self):
        for category in (
            InitalisedInternally,
            AssignedDuringTracking,
            ToBeDefined,
        ):
            with self.subTest(category=category.__name__):
                self.assertTrue(issubclass(category, _LateInit))

    def test_instances_satisfy_the_docs_skip_hook(self):
        for name in ("by_framework", "by_tracking", "by_user"):
            with self.subTest(attribute=name):
                self.assertIsInstance(vars(_Categorised)[name], _LateInit)

    def test_unfilled_read_names_what_fills_it(self):
        owner = _Categorised()
        with self.assertRaises(NotInitialisedError) as context:
            _ = owner.by_tracking
        self.assertIn("`track()`", str(context.exception))

    def test_read_after_set(self):
        owner = _Categorised()
        owner.by_user = 2.5
        self.assertEqual(owner.by_user, 2.5)

    def test_registered_like_the_base_class(self):
        self.assertEqual(
            late_init_attributes(_Categorised),
            ("by_framework", "by_tracking", "by_user"),
        )


class _Routed:
    by_argument: ToBeDefined[float] = ToBeDefined(SetBy.ARGUMENT)
    by_schedule: ToBeDefined[float] = ToBeDefined(SetBy.ARGUMENT_OR_SCHEDULE)
    by_update: ToBeDefined[float] = ToBeDefined(SetBy.UPDATE_ATTRIBUTES)
    by_run: InitalisedInternally[float] = InitalisedInternally(
        SetBy.RUN_SIMULATION
    )


def _message_for(route: SetBy | str) -> str:
    """
    Raise on an unfilled attribute declared with ``route``.

    Parameters
    ----------
    route
        The ``set_by`` to declare the attribute with.

    Returns
    -------
    str
        The resulting error message.
    """
    owner = type("_Probe", (), {"attr": ToBeDefined(route)})
    try:
        owner().attr
    except NotInitialisedError as exc:
        return str(exc)
    raise AssertionError("reading an unfilled attribute did not raise")


class TestSetBy(BLonDTestCase):
    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            _LateInit("`never()`")

    def test_owner_placeholder_becomes_the_reading_class(self):
        with self.assertRaises(NotInitialisedError) as context:
            _ = _Routed().by_update
        self.assertIn(
            "`_Routed.update_attributes(...)`", str(context.exception)
        )

    def test_attribute_placeholder_becomes_the_attribute_name(self):
        with self.assertRaises(NotInitialisedError) as context:
            _ = _Routed().by_argument
        self.assertIn(
            "the `by_argument` argument of `_Routed(...)`",
            str(context.exception),
        )

    def test_attribute_placeholder_is_repr_quoted_for_schedule(self):
        with self.assertRaises(NotInitialisedError) as context:
            _ = _Routed().by_schedule
        self.assertIn("attribute='by_schedule'", str(context.exception))

    def test_route_without_placeholders_renders_verbatim(self):
        with self.assertRaises(NotInitialisedError) as context:
            _ = _Routed().by_run
        self.assertIn(
            "`Simulation.run_simulation(...)`", str(context.exception)
        )

    def test_every_route_renders_without_leftover_placeholders(self):
        for route in SetBy:
            with self.subTest(route=route.name):
                message = _message_for(route)
                self.assertNotIn("{", message)
                self.assertNotIn("}", message)

    def test_a_plain_string_is_accepted(self):
        self.assertIn(
            "`Simulation.load_results(...)`",
            _message_for("`Simulation.load_results(...)`"),
        )

    def test_message_starts_with_the_route_after_the_framing(self):
        # Each route names what the user calls, so it must survive into
        # the message rather than being replaced by a hook name.
        for route in SetBy:
            with self.subTest(route=route.name):
                rendered = str(route).format(owner="_Probe", attribute="attr")
                self.assertIn(rendered, _message_for(route))


if __name__ == "__main__":
    unittest.main()
