from typing import Iterable, TypeVar

T = TypeVar("T")


def all_equal(iterable: Iterable[T]) -> bool:
    """
    Check if all elements in the iterable are equal.

    Parameters
    ----------
    iterable : Iterable[T]
        An iterable containing elements to be compared.

    Returns
    -------
    bool
        True if all elements are equal or the iterable is empty,
        False otherwise.

    Examples
    --------
    >>> all_equal([1, 1, 1])
    True

    >>> all_equal([1, 2, 1])
    False

    >>> all_equal([])
    True
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True  # Empty iterable → considered all equal
    return all(x == first for x in iterator)
