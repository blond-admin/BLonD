# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Helpers for string formatting."""

import math


def si_format(num, decimals=2) -> str:
    """
    Get a string with SI-postfix from a ``float``.

    Parameters
    ----------
    num
        Floating number to evaluate.
    decimals
        Number of decimals to output.

    Returns
    -------
    si_postfix_num
        Number with SI-postfix, e.g. ``si_format(1e3) = '1k'``.
    """
    if num == 0:
        return "0"

    prefixes = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "µ",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }

    exponent = int(math.floor(math.log10(abs(num)) / 3) * 3)
    exponent = max(min(exponent, 24), -24)

    value = num / (10**exponent)
    return f"{value:.{decimals}f}{prefixes[exponent]}"
