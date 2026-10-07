# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Kernel wrappers must pass array pointers through ``_get_pointer``.

``x.ctypes.data_as(...)`` builds a fresh numpy ``_ctypes`` helper and a new
ctypes object on *every* call. Measured on an i5-11500 that is ~1.3 us per
array, so a kernel taking two arrays pays ~2.7 us before the C code starts
-- which is a large share of a small or medium kernel call, and is paid
again on every turn of the tracking loop.

``_get_pointer`` caches the ``c_void_p`` keyed by array identity, holding
only a weak reference and re-validating it on every hit. It is safe because
a numpy array never relocates its data buffer while alive and BLonD never
resizes kernel arrays in place.

Most wrappers already use it; the ones that did not were simply
inconsistent, not deliberate. This test keeps them that way, because the
difference is invisible in results and only shows up as a slower
simulation.

Authors: Simon Lauber
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from blond.core.backends.cpp import callables
from blond.testing.backend_testing import BLonDTestCase

_CALLABLES_SOURCE = Path(inspect.getfile(callables))


def _fresh_pointer_call_sites() -> list[str]:
    """
    Find every ``<array>.ctypes.data_as(...)`` in the C++ wrappers.

    Returns
    -------
    offenders
        ``line:code`` for each call that builds a pointer from scratch
        instead of using the cached ``_get_pointer`` helper.
    """
    source = _CALLABLES_SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(_CALLABLES_SOURCE))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # match `<something>.ctypes.data_as(...)`
        if not isinstance(func, ast.Attribute) or func.attr != "data_as":
            continue
        owner = func.value
        if not isinstance(owner, ast.Attribute) or owner.attr != "ctypes":
            continue
        offenders.append(f"{node.lineno}: {lines[node.lineno - 1].strip()}")
    return offenders


class TestKernelPointersAreCached(BLonDTestCase):
    """The C++ wrappers must not rebuild ctypes pointers per call."""

    def test_no_fresh_ctypes_pointer_in_wrappers(self):
        """Every kernel argument pointer goes through ``_get_pointer``."""
        offenders = _fresh_pointer_call_sites()
        self.assertEqual(
            [],
            offenders,
            msg=(
                "These wrappers build a ctypes pointer on every call "
                "(~1.3 us per array). Use `_get_pointer(<array>)` instead, "
                "as the other kernels do:\n  " + "\n  ".join(offenders)
            ),
        )
