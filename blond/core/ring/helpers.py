# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helper functions to deal with the late-init methods of `Simulation`."""

from __future__ import annotations

import inspect
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable
    from typing import (
        Any,
        TypeVar,
    )

    # To reference a function that has the same output type
    # like an input type
    T = TypeVar("T")


def requires(dependencies: list[str]) -> Callable:
    """
    Decorator to manage execution order of decorated functions.

    This is useful when you need to enforce or track an execution order between
    functions or methods—especially in frameworks, pipelines, or initialization
    sequences where certain components must be processed first.

    Parameters
    ----------
    dependencies
        A list of class names that are intended to be initialized
        before the decorated function should be executed.

    Returns
    -------
    Callable
        A decorator that adds a `requires` attribute to the decorated function.

    Notes
    -----
    - Dependencies are expressed as strings to avoid cyclic imports.
    - The decorator does *not* enforce order by itself; it simply attaches
      metadata (`.requires`) to the wrapped function that other systems can
      inspect.

    Examples
    --------
    >>> from blond.core.ring.helpers import requires, get_required_order
    >>>
    >>>
    >>> class ClassA:
    ...     def common(self):
    ...         pass
    >>>
    >>>
    >>> class ClassB:
    ...     @requires(["ClassA",])
    ...     def common(self):
    ...         pass
    >>>
    >>>
    >>> sorted_classes = get_required_order(
    ...     instances=(ClassB(), ClassA()),
    ...     dependency_attribute="common.requires",
    ... )
    >>> # The classes are sorted according to their requirements
    >>> # sorted_classes = ['ClassA', 'ClassB']
    """
    if not all(isinstance(dep, str) for dep in dependencies):
        raise TypeError("All dependencies must be strings.")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # No additional behavior—simply pass through to the wrapped function.
            return func(*args, **kwargs)

        # Attach the dependency metadata
        wrapper.requires = dependencies  # type: ignore[attr-defined]
        return wrapper

    return decorator


def filter_elements(elements: Iterable, _class: type[T]) -> tuple[T, ...]:
    """
    Find all elements of a certain type.

    Parameters
    ----------
    elements
        List of instances that might match isinstance(element, _class).
    _class
        Return only elements that are instance of this class.

    Returns
    -------
    filtered_elements
        List of filtered elements that match _class.
    """
    return tuple(filter(lambda x: isinstance(x, _class), elements))


def get_required_order(
    instances: Iterable[Any], dependency_attribute: str
) -> list[Any]:
    """
    Get order to be initialized elements.

    Parameters
    ----------
    instances
        Instances to be sorted.
    dependency_attribute
        Attribute that is used for sorting
        e.g. "on_init_simulation.requires".

    Returns
    -------
    sorted_classes_filtered
        Sorted `instances`.

    Notes
    -----
    To be used in combination with `@requires(["ClassName1", "ClassName2"])`.
    """
    graph, in_degree, all_classes = _build_dependency_graph(
        instances, dependency_attribute
    )
    sorted_classes = _topological_sort(graph, in_degree, all_classes)
    sorted_classes_filtered = []
    for cls in sorted_classes:
        if any(cls == type(i).__name__ for i in instances):
            sorted_classes_filtered.append(cls)

    return sorted_classes_filtered


def _build_dependency_graph(
    instances: Iterable[Any], dependency_attribute: str
) -> tuple[defaultdict[Any, list], defaultdict[Any, int], set]:
    """
    Function to build a dependency graph.

    Parameters
    ----------
    instances
        Instances to be sorted.
    dependency_attribute
        Attribute that is used for sorting
        e.g. "on_init_simulation.requires".

    Returns
    -------
    graph
        Directed graph mapping dependencies to dependent classes.
    in_degree
        Count of incoming edges for each class.
    all_classes
        Set of all involved class names.
    """
    graph = defaultdict(
        list
    )  # Directed graph: dependency -> list of dependent classes
    in_degree: defaultdict = defaultdict(
        int
    )  # Count of incoming edges (dependencies) for each class
    all_classes: set[str] = {
        o.__class__.__name__ for o in instances
    }  # Set to keep track of all involved classes

    # Iterate through the types (classes) of all given instances
    for class_of_instance in (type(ins) for ins in instances):
        # Traverse the class's MRO (method resolution order) to get parent classes as well
        # For each dependency declared in 'on_init_simulation_dependencies'
        instance_depends_on: set[str] = set()
        for inherited_class in inspect.getmro(class_of_instance):
            dependencies_tmp = get_dependencies(
                inherited_class, dependency_attribute
            )
            for dep_ in dependencies_tmp:
                instance_depends_on.add(dep_)

        for dependency in instance_depends_on:
            depends_on_instances = _get_matching_instances(
                instances, dependency
            )

            # If a node is not implemented, which is in the dependency tree,
            # the simulation would not be possible.
            assert len(depends_on_instances) > 0, (
                f"Missing instance of {dependency} which is a dependency of {class_of_instance.__name__}"
            )

            for dep in depends_on_instances:
                graph[dep].append(
                    class_of_instance.__name__
                )  # Add edge: dep -> cls
                in_degree[class_of_instance.__name__] += (
                    1  # Increment in-degree count for the class
                )
    return graph, in_degree, all_classes


def _get_matching_instances(
    instances: Iterable[object], dependency: str
) -> list[str]:
    """
    Return the names of instance types that match dependency within the MRO.

    Return the names of instance types whose method-resolution order (MRO)
    includes a class with the given name.

    Parameters
    ----------
    instances : Iterable[object]
        An iterable of instantiated objects to inspect.
    dependency : str
        The class name to search for within each instance's MRO.

    Returns
    -------
    list[str]
        A list of class names (one per instance) whose MRO contains
        a class named `dependency`.

    Notes
    -----
    This checks only the class *names* in the MRO, not the class objects
    themselves. For example, if multiple unrelated classes share the same
    name, they will all be considered matches.
    """
    return [
        instance_type.__name__
        for instance_type in [type(ins) for ins in instances]
        if dependency in [c.__name__ for c in inspect.getmro(instance_type)]
    ]


def get_dependencies(cls_: type, dependency_attribute: str) -> list:
    """
    Investigate on which classes this class depends.

    Parameters
    ----------
    cls_
        Investigated class.
    dependency_attribute
        Attribute that is used for sorting
        e.g. "on_init_simulation.requires".

    Returns
    -------
    dependencies
        List of class names this class depends on.
    """
    if "." in dependency_attribute:
        if dependency_attribute.count(".") != 1:
            raise NotImplementedError(
                f"Only one . allowed in {dependency_attribute=}"
            )
        atr1, atr2 = dependency_attribute.split(".")
        attr = getattr(cls_, atr1, None)
        if attr is not None:
            attr = getattr(attr, atr2, [])
            if not isinstance(attr, list):
                raise Exception(type(attr))
        else:
            attr = []
    else:
        attr = getattr(cls_, dependency_attribute, [])
        if not isinstance(attr, list):
            raise Exception(type(attr))
    return attr


def _topological_sort(
    graph: defaultdict[Any, list],
    in_degree: defaultdict[Any, int],
    all_classes: set,
) -> list[Any]:
    """
    Function to perform topological sort on the dependency graph.

    Parameters
    ----------
    graph
        Directed graph mapping dependencies to dependent classes.
    in_degree
        Count of incoming edges for each class.
    all_classes
        Set of all involved class names.

    Returns
    -------
    sorted_classes
        List of class names in topologically sorted order.
    """
    # this import is here because of sphinx warning
    # `list assignment index out of range [autodoc]`
    from collections import deque

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
