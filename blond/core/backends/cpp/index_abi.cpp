// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

/**
Reports the compiled ABI of `index_t` to the Python side.

`blond_common.h` and `INDEX_DTYPE` in blond/core/backends/backend.py declare
the macro-particle index type independently of one another, and ctypes
validates nothing about the widths it passes. Handing a 4-byte value to a
kernel that reads 8 (or the reverse) reads or writes adjacent memory inside
the kernel rather than raising, so the mismatch surfaces as corrupted
particle data far from its cause. The exports below let the Python wrapper
compare what it is about to pass against what this library was actually
compiled with, once, when the library is loaded.

@Date: 14.09.2026
*/

#include "blond_common.h"
#include <type_traits>

// Signedness is not negotiable. The comparison on the Python side only checks
// that the two declarations AGREE, so an unsigned index_t paired with an
// unsigned INDEX_DTYPE would pass it while every signed loop counter in the
// kernels wrapped silently instead of going negative.
static_assert(std::is_signed<index_t>::value,
              "index_t must be a signed integer type");

// The purpose of index_t is to count past 2^31 - 1 macro-particles (see the
// typedef in blond_common.h). Narrowing it is a deliberate decision that has
// to be taken here as well as at the typedef, not something that should be
// possible by editing one line.
static_assert(sizeof(index_t) >= 8,
              "index_t must be at least 64-bit wide");

/// Size of `index_t` in bytes, as compiled into this library.
extern "C" int blond_index_t_size() { return (int)sizeof(index_t); }

/// 1 if `index_t` is a signed type, 0 otherwise, as compiled into this
/// library.
extern "C" int blond_index_t_is_signed() {
    return std::is_signed<index_t>::value ? 1 : 0;
}
