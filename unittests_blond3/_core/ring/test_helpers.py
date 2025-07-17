import unittest

from blond3._core.ring.helpers import (
    _build_dependency_graph,
    get_dependencies,
    get_elements,
    get_init_order,
    requires,
    _topological_sort,
)


class A:
    def __init__(self):
        pass

    def common(self):
        pass


class B:
    def __init__(self):
        pass

    @requires(["A"])
    def common(self):
        pass


class C:
    def __init__(self):
        pass

    @requires(["B"])
    def common(self):
        pass


class D:
    def __init__(self):
        pass

    @requires(["B", "C"])
    def common(self):
        pass


class AD(A, D):
    def __init__(self):
        super().__init__()


class TestFunctions(unittest.TestCase):
    def test_build_dependency_graph_executes(self):
        a = A()
        b = B()
        graph, in_degree, all_classes = _build_dependency_graph(
            instances=(a, b), dependency_attribute="common.requires"
        )

    def test_get_elements(self):
        a = A()
        elements_selected = get_elements(elements=(a, B(), C()), class_=A)
        assert elements_selected[0] is a

    def test_get_elements2(self):
        a = A()
        ad = AD()
        elements_selected = get_elements(elements=(a, B(), C(), ad), class_=A)
        assert elements_selected[0] is a
        assert elements_selected[1] is ad

    def test_get_init_order(self):
        a = A()
        b = B()
        sorted_classes = get_init_order(
            instances=(a, b), dependency_attribute="common.requires"
        )
        assert sorted_classes == ["A", "B"]

    def test_get_init_order2(self):
        sorted_classes = get_init_order(
            instances=(A(), B(), C(), D()), dependency_attribute="common.requires"
        )
        assert sorted_classes == ["A", "B", "C", "D"]

    def test_get_init_order3(self):
        a = A()
        b = D()
        sorted_classes = get_init_order(
            instances=(a, b), dependency_attribute="common.requires"
        )
        assert sorted_classes == ["A", "D"]

    def test_requires(self):
        class A:
            @requires([""])
            def method(self):
                pass

        assert A.method.requires == [""]

    @unittest.skip
    def test_topological_sort(self):
        # TODO: implement test for `topological_sort`
        _topological_sort(graph=None, in_degree=None, all_classes=None)

    @unittest.skip
    def test_get_dependencies(self):
        # TODO: implement test for `get_dependencies`
        get_dependencies(cls_=None, dependency_attribute=None)


if __name__ == "__main__":
    unittest.main()
