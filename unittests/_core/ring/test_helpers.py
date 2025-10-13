import unittest

from blond._core.ring.helpers import (
    _build_dependency_graph,
    _topological_sort,
    get_dependencies,
    get_elements,
    get_init_order,
    requires,
)


class A:
    def __init__(self) -> None:
        pass

    def common(self):
        pass


class B:
    def __init__(self) -> None:
        pass

    @requires(["A"])
    def common(self):
        pass


class C:
    def __init__(self) -> None:
        pass

    @requires(["B"])
    def common(self):
        pass


class D:
    def __init__(self) -> None:
        pass

    @requires(["B", "C"])
    def common(self):
        pass


class AD(A, D):
    def __init__(self) -> None:
        super().__init__()


class TestFunctions(unittest.TestCase):
    def test_build_dependency_graph_executes(self) -> None:
        a = A()
        b = B()
        graph, in_degree, all_classes = _build_dependency_graph(
            instances=(a, b), dependency_attribute="common.requires"
        )

    def test_get_elements(self) -> None:
        a = A()
        elements_selected = get_elements(elements=(a, B(), C()), _class=A)
        assert elements_selected[0] is a

    def test_get_elements2(self) -> None:
        a = A()
        ad = AD()
        elements_selected = get_elements(elements=(a, B(), C(), ad), _class=A)
        assert elements_selected[0] is a
        assert elements_selected[1] is ad

    def test_get_init_order(self) -> None:
        a = A()
        b = B()
        sorted_classes = get_init_order(
            instances=(a, b), dependency_attribute="common.requires"
        )
        assert sorted_classes == ["A", "B"]

    def test_get_init_order2(self) -> None:
        sorted_classes = get_init_order(
            instances=(A(), B(), C(), D()),
            dependency_attribute="common.requires",
        )
        assert sorted_classes == ["A", "B", "C", "D"]

    def test_get_init_order3(self) -> None:
        a = A()
        b = D()
        sorted_classes = get_init_order(
            instances=(a, b), dependency_attribute="common.requires"
        )
        assert sorted_classes == ["A", "D"]

    def test_requires(self) -> None:
        class A:
            @requires([""])
            def method(self):
                pass

        assert A.method.requires == [""]

    @unittest.skip
    def test_topological_sort(self) -> None:
        # TODO: implement test for `topological_sort`
        _topological_sort(graph=None, in_degree=None, all_classes=None)

    def test_get_dependencies(self) -> None:
        class A:
            pass

        class B:
            @requires(["A", "B"])
            def on_init_simulation(self):
                return

        res = get_dependencies(
            cls_=B,
            dependency_attribute="on_init_simulation.requires",
        )
        self.assertEqual(res, ["A", "B"])

    def test_get_dependencies_raise(self) -> None:
        class A:
            pass

        class B:
            @requires("A")
            def on_init_simulation(self):
                return

        with self.assertRaises(Exception):
            get_dependencies(
                cls_=B,
                dependency_attribute="on_init_simulation.requires",
            )

    def test_get_dependencies2(self) -> None:
        class A:
            pass

        class B:
            requires = ["A", "B"]

        res = get_dependencies(
            cls_=B,
            dependency_attribute="requires",
        )
        self.assertEqual(res, ["A", "B"])

    def test_get_dependencies_raise2(self) -> None:
        class A:
            pass

        class B:
            requires = "A"

        with self.assertRaises(Exception):
            get_dependencies(
                cls_=B,
                dependency_attribute="requires",
            )


if __name__ == "__main__":
    unittest.main()
