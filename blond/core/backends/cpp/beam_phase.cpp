// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include "blond_common.h"
#include "openmp.h"

namespace {

// Integrands of bin `i`: weight * sin(phase) and weight * cos(phase), with
// weight = exp(alpha * bin_center) * profile (exp dropped if !with_exp)
// and phase = omega_rf * bin_center + phi_rf.
template <bool with_exp>
inline void sin_cos_integrands(const real_t *__restrict__ bin_centers,
                               const real_t *__restrict__ profile,
                               const real_t alpha, const real_t omega_rf,
                               const real_t phi_rf, const int i,
                               real_t &sin_integrand, real_t &cos_integrand) {
  real_t weight = profile[i];
  if (with_exp) {
    weight *= FAST_EXP(alpha * bin_centers[i]);
  }
  const real_t phase = omega_rf * bin_centers[i] + phi_rf;
  sin_integrand = weight * FAST_SIN(phase);
  cos_integrand = weight * FAST_COS(phase);
}

// Ratio of the trapezoidal integrals (step `bin_size`) of the sin and cos
// integrands. Both are summed in one reduction, so no per-call scratch
// arrays are needed.
template <bool with_exp>
real_t sin_cos_integral_ratio(const real_t *__restrict__ bin_centers,
                              const real_t *__restrict__ profile,
                              const real_t alpha, const real_t omega_rf,
                              const real_t phi_rf, const real_t bin_size,
                              const int n_bins) {
  real_t sin_first = 0;
  real_t cos_first = 0;
  real_t sin_last = 0;
  real_t cos_last = 0;
  sin_cos_integrands<with_exp>(bin_centers, profile, alpha, omega_rf, phi_rf, 0,
                               sin_first, cos_first);
  sin_cos_integrands<with_exp>(bin_centers, profile, alpha, omega_rf, phi_rf,
                               n_bins - 1, sin_last, cos_last);
  // Trapezoidal rule: the end points count half.
  real_t sin_sum = (sin_first + sin_last) / 2.;
  real_t cos_sum = (cos_first + cos_last) / 2.;

#pragma omp parallel for reduction(+ : sin_sum, cos_sum)
  for (int i = 1; i < n_bins - 1; ++i) {
    real_t sin_integrand = 0;
    real_t cos_integrand = 0;
    sin_cos_integrands<with_exp>(bin_centers, profile, alpha, omega_rf, phi_rf,
                                 i, sin_integrand, cos_integrand);
    sin_sum += sin_integrand;
    cos_sum += cos_integrand;
  }

  return (bin_size * sin_sum) / (bin_size * cos_sum);
}

} // namespace

extern "C" real_t beam_phase(const real_t *__restrict__ bin_centers,
                             const real_t *__restrict__ profile,
                             const real_t alpha, const real_t omega_rf,
                             const real_t phi_rf, const real_t bin_size,
                             const int n_bins) {
  return sin_cos_integral_ratio<true>(bin_centers, profile, alpha, omega_rf,
                                      phi_rf, bin_size, n_bins);
}

extern "C" real_t beam_phase_fast(const real_t *__restrict__ bin_centers,
                                  const real_t *__restrict__ profile,
                                  const real_t omega_rf, const real_t phi_rf,
                                  const real_t bin_size, const int n_bins) {
  return sin_cos_integral_ratio<false>(bin_centers, profile, real_t(0),
                                       omega_rf, phi_rf, bin_size, n_bins);
}
