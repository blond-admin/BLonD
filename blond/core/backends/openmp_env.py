# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""OpenMP environment defaults applied before any OpenMP runtime starts."""

from __future__ import annotations

import os

__all__ = ["OMP_WAIT_POLICY", "set_default_openmp_wait_policy"]

OMP_WAIT_POLICY = "OMP_WAIT_POLICY"


def set_default_openmp_wait_policy() -> None:
    """
    Default ``OMP_WAIT_POLICY`` to ``passive`` unless the user set it.

    BLonD's per-turn kernels are short (order one millisecond of
    memory-bound work per call), so the tracking loop crosses an OpenMP
    barrier thousands of times per second. With libgomp's default
    ``active`` wait policy every worker busy-spins at each of those
    barriers instead of sleeping. As soon as the machine is even slightly
    contended -- the Python main thread, a hyperthread sibling, or a
    second BLonD process -- one worker is descheduled and the rest burn
    whole cores spinning for it, which was measured to cost a factor ~20
    in the ``kick`` and ``drift`` kernels of both the ``numba`` and
    ``cpp`` backends.

    ``passive`` makes a waiting thread sleep rather than spin. The number
    of OpenMP threads and the parallel regions themselves are unchanged,
    so this does not reduce the available parallelism -- it only removes
    the busy-waiting between kernel invocations.

    The value is only set when ``OMP_WAIT_POLICY`` is absent from the
    environment, so an explicit choice is never overridden. It must be
    applied before the OpenMP runtime is loaded, because libgomp reads
    the variable once at initialisation.

    Notes
    -----
    On a dedicated, otherwise idle machine running long-lived parallel
    regions, ``active`` can be marginally faster. Set
    ``OMP_WAIT_POLICY=active`` before starting Python to restore it.
    """
    os.environ.setdefault(OMP_WAIT_POLICY, "passive")


# Applied on import so that merely importing this module -- which must
# happen before any OpenMP runtime is loaded -- is enough to take effect.
set_default_openmp_wait_policy()
