import copy
import unittest

from blond.generals.exceptions_ import NotInitialisedError
from blond.generals.late_init import (
    AssignedDuringTracking,
    InitialisedInternally,
    SetBy,
    ToBeDefined,
    _LateInit,
    check_filled,
    late_init_attributes,
    unfilled,
)
from blond.testing.backend_testing import BLonDTestCase


class _Owner:
    energy: InitialisedInternally[float] = InitialisedInternally(
        "`_Owner.setup()`"
    )
    time: InitialisedInternally[float] = InitialisedInternally(
        "`_Owner.setup()`"
    )
    plain = 0.0

    def setup(self, energy: float) -> None:
        self.energy = energy


class _Child(_Owner):
    extra: InitialisedInternally[int] = InitialisedInternally(
        "`_Child.setup()`"
    )


class _Mixin:
    offset: InitialisedInternally[float] = InitialisedInternally(
        "`_Mixin.setup()`"
    )


class _MultiOwner(_Owner, _Mixin):
    pass


class _Redeclarer(_Owner):
    energy: InitialisedInternally[float] = InitialisedInternally(
        "`_Redeclarer.setup()`"
    )


class _Shadower(_Owner):
    energy = 1.0
    time = 2.0


class _NoLateInit:
    plain = 0.0


class _Categorised:
    by_framework: InitialisedInternally[float] = InitialisedInternally(
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
            InitialisedInternally,
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
    by_run: InitialisedInternally[float] = InitialisedInternally(
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


class _BrokenProperty:
    late: ToBeDefined[float] = ToBeDefined("the user")

    @property
    def broken(self) -> float:
        raise AttributeError("not a late init")


class TestUnfilled(BLonDTestCase):
    def test_excludes_a_filled_name(self):
        owner = _Owner()
        owner.setup(1.0)
        self.assertEqual(unfilled(owner), ("time",))

    def test_empty_when_all_are_filled(self):
        owner = _Owner()
        owner.setup(1.0)
        owner.time = 2.0
        self.assertEqual(unfilled(owner), ())

    def test_empty_without_declarations(self):
        self.assertEqual(unfilled(_NoLateInit()), ())

    def test_follows_declaration_order_across_the_mro(self):
        self.assertEqual(
            unfilled(_MultiOwner()), late_init_attributes(_MultiOwner)
        )

    def test_a_deleted_name_becomes_unfilled_again(self):
        owner = _Owner()
        owner.setup(1.0)
        del owner.energy
        self.assertIn("energy", unfilled(owner))

    def test_excludes_a_name_shadowed_by_a_plain_class_value(self):
        self.assertEqual(unfilled(_Shadower()), ())

    def test_filters_to_one_category(self):
        self.assertEqual(unfilled(_Categorised(), ToBeDefined), ("by_user",))

    def test_each_category_selects_its_own(self):
        owner = _Categorised()
        for category, expected in (
            (InitialisedInternally, ("by_framework",)),
            (AssignedDuringTracking, ("by_tracking",)),
            (ToBeDefined, ("by_user",)),
        ):
            with self.subTest(category=category.__name__):
                self.assertEqual(unfilled(owner, category), expected)

    def test_omitting_the_category_reports_every_one(self):
        self.assertEqual(
            unfilled(_Categorised()),
            ("by_framework", "by_tracking", "by_user"),
        )

    def test_a_filled_name_is_excluded_from_its_category(self):
        owner = _Categorised()
        owner.by_user = 1.0
        self.assertEqual(unfilled(owner, ToBeDefined), ())

    def test_resolves_a_category_declared_on_a_base_class(self):
        self.assertEqual(
            unfilled(_Child(), InitialisedInternally),
            late_init_attributes(_Child),
        )

    def test_keeps_declaration_order(self):
        self.assertEqual(
            unfilled(_Owner(), InitialisedInternally), ("energy", "time")
        )


class TestCheckFilled(BLonDTestCase):
    def test_passes_when_every_name_is_filled(self):
        owner = _Owner()
        owner.setup(1.0)
        self.assertIsNone(check_filled(owner, "energy"))

    def test_without_names_checks_every_declared_attribute(self):
        with self.assertRaises(NotInitialisedError) as context:
            check_filled(_Categorised())
        message = str(context.exception)
        for name in ("by_framework", "by_tracking", "by_user"):
            with self.subTest(attribute=name):
                self.assertIn(name, message)

    def test_without_names_passes_when_every_one_is_filled(self):
        owner = _Categorised()
        owner.by_framework = 1.0
        owner.by_tracking = 2.0
        owner.by_user = 3.0
        self.assertIsNone(check_filled(owner))

    def test_without_declarations_nothing_is_checked(self):
        self.assertIsNone(check_filled(_NoLateInit()))

    def test_ignores_names_that_are_filled(self):
        owner = _Categorised()
        owner.by_user = 1.0
        with self.assertRaises(NotInitialisedError) as context:
            check_filled(owner, "by_user", "by_tracking")
        self.assertNotIn("by_user", str(context.exception))

    def test_single_unfilled_keeps_the_descriptor_message(self):
        owner = _Categorised()
        with self.assertRaises(NotInitialisedError) as expected:
            _ = owner.by_user
        with self.assertRaises(NotInitialisedError) as actual:
            check_filled(owner, "by_user")
        self.assertEqual(str(actual.exception), str(expected.exception))
        self.assertEqual(actual.exception.name, "by_user")
        self.assertIs(actual.exception.obj, owner)

    def test_reports_every_unfilled_attribute_in_one_error(self):
        owner = _Categorised()
        with self.assertRaises(NotInitialisedError) as context:
            check_filled(owner, "by_framework", "by_tracking", "by_user")
        message = str(context.exception)
        for name in ("by_framework", "by_tracking", "by_user"):
            with self.subTest(attribute=name):
                self.assertIn(name, message)

    def test_keeps_the_route_of_each_unfilled_attribute(self):
        owner = _Categorised()
        with self.assertRaises(NotInitialisedError) as context:
            check_filled(owner, "by_framework", "by_tracking", "by_user")
        message = str(context.exception)
        for route in ("a setup hook", "`track()`", "the user"):
            with self.subTest(route=route):
                self.assertIn(route, message)

    def test_does_not_stop_at_the_first_unfilled_attribute(self):
        owner = _Categorised()
        with self.assertRaises(NotInitialisedError) as context:
            check_filled(owner, "by_framework", "by_user")
        self.assertIn("by_user", str(context.exception))

    def test_lists_in_the_order_the_names_are_given(self):
        owner = _Categorised()
        with self.assertRaises(NotInitialisedError) as context:
            check_filled(owner, "by_user", "by_framework")
        message = str(context.exception)
        self.assertLess(
            message.index("by_user"), message.index("by_framework")
        )

    def test_an_unrelated_attribute_error_is_not_aggregated(self):
        with self.assertRaises(AttributeError) as context:
            check_filled(_BrokenProperty(), "broken", "late")
        self.assertNotIsInstance(context.exception, NotInitialisedError)
        self.assertIn("not a late init", str(context.exception))


if __name__ == "__main__":
    unittest.main()
