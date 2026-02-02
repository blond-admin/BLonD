# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Collection of helpers to develop new functions and modules."""

from collections.abc import Sequence


def check_inputs_length_consistency(*args: tuple[Sequence]):
    """
    Check if the tuple of arguments have the same length.

    Parameters
    ----------
    *args
        Tuple of Sequence.
    """
    lengths = []
    for a in args:
        if isinstance(a, Sequence):
            lengths.append(len(a))
    if len(set(lengths)) > 1:
        raise ValueError(
            "Input sequences of more than one element have different lengths."
        )
