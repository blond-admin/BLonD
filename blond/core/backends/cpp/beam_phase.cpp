// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include "blond_common.h"
#include "blondmath.h"
#include "openmp.h"
#include <vector>

real_t trapz_const_delta(const real_t *__restrict__ f, const real_t deltaX,
                         const int nsub) {
  // initialize the partial sum to be f(a)+f(b) and
  // deltaX to be the step size using nsub subdivisions
  real_t psum = (f[0] + f[nsub - 1]) / 2.; // f(a)+f(b);

// increment the partial sum
#pragma omp parallel for reduction(+ : psum)
  for (int i = 1; i < nsub - 1; ++i)
    psum += f[i];

  // multiply the sum by the constant deltaX/2.0
  // return approximation
  return deltaX * psum;
}

extern "C" real_t beam_phase(const real_t *__restrict__ bin_centers,
                             const real_t *__restrict__ profile,
                             const real_t alpha, const real_t omega_rf,
                             const real_t phi_rf, const real_t bin_size,
                             const int n_bins) {
  // Use std::vector to avoid manual memory management and heap fragmentation
  std::vector<real_t> base(n_bins);
  std::vector<real_t> array1(n_bins);
  std::vector<real_t> array2(n_bins);

#pragma omp parallel for
  for (int i = 0; i < n_bins; ++i) {
    base[i] = FAST_EXP(alpha * bin_centers[i]) * profile[i];
  }

#pragma omp parallel for
  for (int i = 0; i < n_bins; ++i) {
    const real_t a = omega_rf * bin_centers[i] + phi_rf;
    array1[i] = base[i] * FAST_SIN(a);
    array2[i] = base[i] * FAST_COS(a);
  }

  real_t scoeff = trapz_const_delta(array1.data(), bin_size, n_bins);
  real_t ccoeff = trapz_const_delta(array2.data(), bin_size, n_bins);

  return scoeff / ccoeff;
}

extern "C" real_t beam_phase_fast(const real_t *__restrict__ bin_centers,
                                  const real_t *__restrict__ profile,
                                  const real_t omega_rf, const real_t phi_rf,
                                  const real_t bin_size, const int n_bins) {
  // Use std::vector to avoid manual memory management and heap fragmentation
  std::vector<real_t> array1(n_bins);
  std::vector<real_t> array2(n_bins);

#pragma omp parallel for
  for (int i = 0; i < n_bins; ++i) {
    const real_t a = omega_rf * bin_centers[i] + phi_rf;
    array1[i] = profile[i] * FAST_SIN(a);
    array2[i] = profile[i] * FAST_COS(a);
  }

  real_t scoeff = trapz_const_delta(array1.data(), bin_size, n_bins);
  real_t ccoeff = trapz_const_delta(array2.data(), bin_size, n_bins);

  return scoeff / ccoeff;
}
