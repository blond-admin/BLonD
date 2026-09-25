# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Number of threads worth using for the C++ ``histogram``.

The three constants below are properties of the machine, not of the
algorithm: each one is a ratio of two hardware speeds. They live in
Python so that they can be set at runtime, without recompiling::

    from blond.core.backends.cpp import histogram_n_threads
    histogram_n_threads.BYTES_PER_PARTICLE = 170  # a laptop

The defaults are set for the kind of node BLonD runs its large jobs on,
a dual-socket AMD EPYC (Zen 2/3) server as found behind LXPLUS and its
batch system: 512 KiB of L2 per core, eight DDR4-3200 channels per
socket, threads woken through a hypervisor. The values are ESTIMATES
FROM THOSE SPECS, cross-checked only for plausibility on a laptop; they
have not been measured on such a node. A wrong value never changes the
result of ``histogram``, only how far its runtime is from the optimum,
and the runtime is flat around the optimum: a factor 2 in a constant
costs a few percent.
"""

from __future__ import annotations

from math import sqrt

from blond.core.backends.backend import INDEX_DTYPE

__all__ = [
    "BYTES_PER_PARTICLE",
    "CACHED_BYTES",
    "PARTICLES_PER_THREAD",
    "histogram_n_threads",
]

#: Thread wake-up, in particles: the latency until a sleeping thread
#: counts, over the time one core needs per particle. A thread handed
#: fewer particles than this costs more than it saves. BLonD defaults to
#: a passive OpenMP wait policy, so the threads really do sleep between
#: two calls. Some 50 us to wake a thread in a virtual machine over some
#: 1 ns per particle on a server core. A desktop wakes faster and counts
#: faster, its ratio is about half of this. Erring high is cheap, a few
#: percent near the limit; erring low is not, a thread woken for nothing
#: costs its whole wake-up.
PARTICLES_PER_THREAD = 65536

#: Private cache of one core, in bytes (its L2). A private histogram up
#: to this size is zeroed and reduced without touching main memory,
#: which is free next to the counting. 512 KiB on Zen 2/3; 256 KiB on
#: older Xeons, 1 MiB and more on newer Xeons and on Zen 4.
CACHED_BYTES = 512 * 1024

#: Counting against streaming, in bytes per particle: the time ONE
#: thread needs per particle when every increment misses the cache,
#: times the bytes ALL threads together stream over the memory bus per
#: second. I.e. how many bytes of private histogram the machine can zero
#: and reduce in the time one thread counts one particle into main
#: memory. This is where the memory bus of the machine hides: the
#: counting is bound by the latency of main memory, which more channels
#: do not improve, the zeroing and reducing by its bandwidth, which they
#: do. So a server with many channels affords more threads than a
#: desktop. Some 10 ns per particle times some 100 GB/s per socket gives
#: 1000; halved, as the first touch of a freshly allocated histogram
#: costs page faults on top of the streaming. For comparison, a laptop
#: with two LPDDR5 channels measured 170.
BYTES_PER_PARTICLE = 512

_COUNTER_BYTES = INDEX_DTYPE().itemsize


def histogram_n_threads(
    n_macroparticles: int, n_slices: int, max_threads: int
) -> int:
    """
    Return the number of threads worth using for the C++ ``histogram``.

    Counting splits among the threads, but every thread also pays for
    itself: it is woken up, and it zeroes a private histogram that the
    reduction then reads back. Two limits follow. Their form holds on any
    machine, their constants (module attributes, read on every call) do
    not:

    - Each thread needs `PARTICLES_PER_THREAD` particles.
    - Private histograms in main memory share one memory bus, so zeroing
      and reducing them takes ``n_threads * 2 * histogram_bytes`` of bus
      time, while the counting, bound by latency instead, still splits:
      ``n_macroparticles * BYTES_PER_PARTICLE / n_threads`` in the same
      unit. The sum ``a * n_threads + b / n_threads`` is smallest at
      ``sqrt(b / a)``.

    The counters are integers, so the thread count never changes the
    result of ``histogram``, only its runtime.

    Parameters
    ----------
    n_macroparticles
        Number of particles to be counted.
    n_slices
        Number of bins of the histogram.
    max_threads
        Largest allowed result, e.g. ``omp_get_max_threads()``.

    Returns
    -------
    n_threads
        Size of the OpenMP team, ``1 <= n_threads <= max_threads``.
    """
    n_threads = n_macroparticles / PARTICLES_PER_THREAD

    # One slot past the profile: the kernel's trash bin.
    histogram_bytes = (n_slices + 1) * _COUNTER_BYTES
    if histogram_bytes > CACHED_BYTES:
        n_threads_bus = sqrt(
            n_macroparticles * BYTES_PER_PARTICLE / (2 * histogram_bytes)
        )
        n_threads = min(n_threads, n_threads_bus)

    return max(1, min(int(n_threads), max_threads))
