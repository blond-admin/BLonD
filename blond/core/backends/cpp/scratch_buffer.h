// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Scratch memory for kernels that run every turn.

#pragma once

#include <cstddef>
#include <vector>

// Return `size` elements of `buffer`, growing it only when it is too small.
// Pass a function-local `static thread_local std::vector`: the buffer then
// persists and is reused across calls, so a kernel called every turn
// allocates only when its size grows, not on every call. Grown memory is
// zero-filled once; callers must not rely on the content.
// `thread_local` gives every calling thread its own buffer: `ctypes` releases
// the GIL, so Python threads running separate simulations in one process can
// call the same kernel concurrently. Get the buffer outside the kernel's
// OpenMP region; its worker threads then share the caller's buffer.
template <typename T>
T *reuse_scratch(std::vector<T> &buffer, const std::size_t size) {
  if (size > buffer.size()) {
    buffer.resize(size);
  }
  return buffer.data();
}
