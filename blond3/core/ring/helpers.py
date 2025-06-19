from __future__ import annotations

import inspect
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable, Any, List, Tuple, Type, TypeVar

    T = TypeVar("T")


def requires(argument: List):
    def decorator(function):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        wrapper.requires = argument
        return wrapper

    return decorator


def get_elements(elements: Iterable, class_: Type[T]) -> Tuple[T, ...]:
    return tuple(filter(lambda x: isinstance(x, class_), elements))


def get_init_order(instances: Iterable[Any], dependency_attribute: str):
    graph, in_degree, all_classes = build_dependency_graph(
        instances, dependency_attribute
    )
    sorted_classes = topological_sort(graph, in_degree, all_classes)
    return sorted_classes


def build_dependency_graph(
    instances: Iterable[Any], dependency_attribute: str
) -> (defaultdict[Any, list], defaultdict[Any, int], set):
    """Function to build a dependency graph based on
    'late_init_dependencies' defined in classes"""

    graph = defaultdict(list)  # Directed graph: dependency -> list of dependent classes
    in_degree = defaultdict(
        int
    )  # Count of incoming edges (dependencies) for each class
    all_classes = set()  # Set to keep track of all involved classes

    # Iterate through the types (classes) of all given instances
    for cls in [type(o) for o in instances]:
        all_classes.add(cls)  # Register the class
        # Traverse the class's MRO (method resolution order) to get parent classes as well
        # For each dependency declared in 'late_init_dependencies'
        dependencies = set()
        for cls_ in inspect.getmro(cls):
            deps = get_dependencies(cls_, dependency_attribute)
            for dep_ in deps:
                dependencies.add(dep_)
        for dep in dependencies:
            graph[dep].append(cls)  # Add edge: dep -> cls
            in_degree[cls] += 1  # Increment in-degree count for the class
            all_classes.add(dep)  # Ensure the dependency class is also tracked
    return graph, in_degree, all_classes


def get_dependencies(cls_: type, dependency_attribute: str):
    if "." in dependency_attribute:
        if dependency_attribute.count(".") != 1:
            raise NotImplementedError(
                f"Only one . allowed in " f"{dependency_attribute=}"
            )
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


def topological_sort(
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
