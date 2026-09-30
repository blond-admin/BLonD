// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Kernel call records of the deferred specials (cpp_deferred,
// cuda_deferred): a KernelCallHeader followed by one kernel's Args struct.
// Each Args struct is mirrored by the `args_dtype` of a DeferrableKernel in
// kernel_call_records.py; keep fields, order and KernelId values in step.
// The numpy dtypes use `align=True`, so both sides pad the same way, and
// both backends compare every sizeof(Args) with its dtype when loading.
//
// Include after `real_t` and `index_t` are defined:
// blond_common.h on the C++ side, kernels.cu on the CUDA side.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define BLOND_HOST_DEVICE __host__ __device__
#else
#define BLOND_HOST_DEVICE
#endif

static_assert(sizeof(real_t) == 8, "records assume 64-bit real_t");
static_assert(sizeof(index_t) == 8, "records assume 64-bit index_t");

// The position of each kernel in DEFERRABLE_KERNELS.
enum class KernelId : std::uint32_t {
  KickSingleHarmonic = 0,
  KickMultiHarmonic = 1,
  DriftSimple = 2,
  DriftLikeLineSegment = 3,
  DriftExact = 4,
  KickInterpolated = 5,
};
constexpr int KERNEL_COUNT = 6;

struct KernelCallHeader {
  KernelId kernel_id;
  std::uint32_t record_size_bytes; // header + Args, multiple of 8
};

// Plain C arrays: the layout must match the numpy dtypes.
// NOLINTBEGIN(*-avoid-c-arrays)
struct KickSingleHarmonicArgs {
  real_t voltage;
  real_t omega_rf;
  real_t phi_rf;
  real_t charge;
  real_t acceleration_kick;
};

struct KickMultiHarmonicArgs {
  std::int32_t n_rf;
  real_t voltage[32];
  real_t omega_rf[32];
  real_t phi_rf[32];
  real_t charge;
  real_t acceleration_kick;
};

struct DriftSimpleArgs {
  real_t T;
  real_t eta_0;
  real_t beta;
  real_t energy;
};

struct DriftLikeLineSegmentArgs {
  real_t T;
  real_t eta_0;
  real_t beta;
  real_t energy;
};

struct DriftExactArgs {
  real_t T;
  real_t alpha_0;
  real_t beta;
  real_t energy;
  std::int32_t n_alpha;
  real_t higher_alpha[8];
};

struct KickInterpolatedArgs {
  const real_t *voltage_kick_table;
  index_t voltage_kick_table_length;
  real_t acceleration_kick;
};

// A macro, so the CUDA side can initialise a __device__ array
// from the same list (kernels.cu).
#define KERNEL_CALL_ARGS_SIZES_INITIALIZER                                     \
  {                                                                            \
      sizeof(KickSingleHarmonicArgs), sizeof(KickMultiHarmonicArgs),           \
      sizeof(DriftSimpleArgs),        sizeof(DriftLikeLineSegmentArgs),        \
      sizeof(DriftExactArgs),         sizeof(KickInterpolatedArgs),            \
  }
constexpr std::uint32_t KERNEL_CALL_ARGS_SIZES[KERNEL_COUNT] =
    KERNEL_CALL_ARGS_SIZES_INITIALIZER;
// NOLINTEND(*-avoid-c-arrays)

// The records are packed back to back in a byte buffer, hence the
// casts from the header to its Args and to the next header.
// NOLINTBEGIN(*-reinterpret-cast,*-pointer-arithmetic)
template <class Args>
BLOND_HOST_DEVICE inline const Args &
record_args(const KernelCallHeader *record) {
  return *reinterpret_cast<const Args *>(
      reinterpret_cast<const char *>(record) + sizeof(KernelCallHeader));
}

BLOND_HOST_DEVICE inline const KernelCallHeader *
next_record(const KernelCallHeader *record) {
  return reinterpret_cast<const KernelCallHeader *>(
      reinterpret_cast<const char *>(record) + record->record_size_bytes);
}
// NOLINTEND(*-reinterpret-cast,*-pointer-arithmetic)

// The only switch over KernelId. `visitor(args)` resolves to the
// backend's overload for that Args type; a missing overload does
// not compile. The pragma lets a host-only visitor instantiate it
// under nvcc without a __host__ __device__ mismatch warning.
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <class Visitor>
BLOND_HOST_DEVICE inline void visit_kernel_call(const KernelCallHeader *record,
                                                const Visitor &visitor) {
  switch (record->kernel_id) {
  case KernelId::KickSingleHarmonic:
    visitor(record_args<KickSingleHarmonicArgs>(record));
    break;
  case KernelId::KickMultiHarmonic:
    visitor(record_args<KickMultiHarmonicArgs>(record));
    break;
  case KernelId::DriftSimple:
    visitor(record_args<DriftSimpleArgs>(record));
    break;
  case KernelId::DriftLikeLineSegment:
    visitor(record_args<DriftLikeLineSegmentArgs>(record));
    break;
  case KernelId::DriftExact:
    visitor(record_args<DriftExactArgs>(record));
    break;
  case KernelId::KickInterpolated:
    visitor(record_args<KickInterpolatedArgs>(record));
    break;
  }
}
