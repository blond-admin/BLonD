# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Replaces itself with ``cupy``, or raises the missing-CuPy hint."""

import sys

from blond.generals.cupy_ import import_cupy_with_error_hint

sys.modules[__name__] = import_cupy_with_error_hint()
