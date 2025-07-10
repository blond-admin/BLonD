from __future__ import annotations

import inspect
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Iterable, Any, List, Tuple, Type, TypeVar

    T = TypeVar("T")


def requires(argument: List[str]):
    """Decorator to manage execution order of decorated functions

    Parameters
    ----------
    argument
        List of class names that are required before executing
        the decorated function
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        # allow strings to prevent cyclic imports
        assert all([isinstance(a, str) for a in argument])
        wrapper.requires = argument
        return wrapper

    return decorator


def get_elements(elements: Iterable, class_: Type[T]) -> Tuple[T, ...]:
    """
    Find all elements of a certain type

    Parameters
    ----------
    elements
        List of instances that might match isinstance(element, class_)
    class_
        Return only elements that are instance of this class

    Returns
    -------
    filtered_elements
        List of filtered elements that match class_

    """
    return tuple(filter(lambda x: isinstance(x, class_), elements))


def get_init_order(instances: Iterable[Any], dependency_attribute: str) -> list[Any]:
    """
    Get order to be initialized elements

    Notes
    -----
    To be used in combination with `@requires(["ClassName1", "ClassName2"])`

    Parameters
    ----------
    instances
        Instances to be sorted
    dependency_attribute
        Attribute that is used for sorting
        e.g. "on_init_simulation.requires"

    Returns
    -------
    sorted_classes_filtered
        Sorted `instances`

    """
    graph, in_degree, all_classes = _build_dependency_graph(
        instances, dependency_attribute
    )
    sorted_classes = _topological_sort(graph, in_degree, all_classes)
    sorted_classes_filtered = []
    for cls in sorted_classes:
        if any([cls == type(i).__name__ for i in instances]):
            sorted_classes_filtered.append(cls)

    return sorted_classes_filtered


def _build_dependency_graph(
    instances: Iterable[Any], dependency_attribute: str
) -> (defaultdict[Any, list], defaultdict[Any, int], set):
    """Function to build a dependency graph

    Parameters
    ----------
    instances
        Instances to be sorted
    dependency_attribute
        Attribute that is used for sorting
        e.g. "on_init_simulation.requires"
    """

    graph = defaultdict(list)  # Directed graph: dependency -> list of dependent classes
    in_degree = defaultdict(
        int
    )  # Count of incoming edges (dependencies) for each class
    all_classes = set()  # Set to keep track of all involved classes

    # Iterate through the types (classes) of all given instances
    for cls in [type(o) for o in instances]:
        all_classes.add(cls.__name__)  # Register the class
        # Traverse the class's MRO (method resolution order) to get parent classes as well
        # For each dependency declared in 'on_init_simulation_dependencies'
        dependencies = set()
        for cls_ in inspect.getmro(cls):
            deps = get_dependencies(cls_, dependency_attribute)
            for dep_ in deps:
                dependencies.add(dep_)
        for dep in dependencies:
            graph[dep].append(cls.__name__)  # Add edge: dep -> cls
            in_degree[cls.__name__] += 1  # Increment in-degree count for the class
            all_classes.add(dep)  # Ensure the dependency class is also tracked
        pass
    return graph, in_degree, all_classes


def get_dependencies(cls_: type, dependency_attribute: str):
    """
    Investigate on which classes this class depends

    Parameters
    ----------
    cls_
        Investigated class
    dependency_attribute
        Attribute that is used for sorting
        e.g. "on_init_simulation.requires"


    Returns
    -------

    """
    if "." in dependency_attribute:
        if dependency_attribute.count(".") != 1:
            raise NotImplementedError(f"Only one . allowed in {dependency_attribute=}")
        atr1, atr2 = dependency_attribute.split(".")
        attr = getattr(cls_, atr1, None)
        if attr is not None:
            attr = getattr(attr, atr2, [])
            if not isinstance(attr, list):
                raise Exception(type(attr))
        else:
            attr = []
        if not isinstance(attr, list):
            raise Exception(type(attr))
    else:
        attr = getattr(cls_, dependency_attribute, [])
        if not isinstance(attr, list):
            raise Exception(type(attr))
    return attr


def _topological_sort(
    graph: defaultdict[Any, list], in_degree: defaultdict[Any, int], all_classes: set
) -> list[Any]:
    """Function to perform topological sort on the dependency graph"""
    # Initialize queue with classes that have no dependencies (in-degree 0)
    queue = deque([cls for cls in all_classes if in_degree[cls] == 0])
    sorted_classes = []  # List to store the sorted order

    # Kahn's algorithm for topological sorting
    while queue:
        cls = queue.popleft()
        sorted_classes.append(cls)
        # Reduce in-degree of dependent classes
        for neighbor in graph[cls]:
            in_degree[neighbor] -= 1
            # If in-degree becomes 0, add to queue
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all classes are sorted, there is a cycle in the graph
    if len(sorted_classes) != len(all_classes):
        raise ValueError("Cyclic dependency detected")
    return sorted_classes
