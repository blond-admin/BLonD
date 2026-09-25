// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

/**
C++ Math library
@Author: Leonard Thiele
@Date: 27.04.2026
*/
#include "blond_common.h"
#include "openmp.h"

extern "C" real_t sum_1d_array(const real_t *__restrict__ array_1,
                               const index_t n) {
    real_t acc = 0.0;

#pragma omp parallel for reduction(+ : acc)
    for (index_t idx = 0; idx < n; ++idx) {
        acc += array_1[idx];
    }

    return acc;
}

extern "C" real_t dot_product_1d_array(const real_t *__restrict__ array_1,
                                       const real_t *__restrict__ array_2,
                                       const index_t n) {
    real_t acc = 0.0;

#pragma omp parallel for reduction(+ : acc)
    for (index_t idx = 0; idx < n; ++idx) {
        acc += array_1[idx] * array_2[idx];
    }

    return acc;
}

// The five phase-space sums of a beam in ONE pass over dt and dE:
// sums = {sum(dt), sum(dE), sum(dt^2), sum(dE^2), sum(dt * dE)}.
// Every first- and second-order statistic (means, RMS sizes, RMS
// emittance) is a function of these, so fusing them reads each particle
// array once instead of once per sum.
extern "C" void phase_space_sums(const real_t *__restrict__ dt,
                                 const real_t *__restrict__ dE,
                                 const index_t n,
                                 real_t *__restrict__ sums) {
    real_t dt_sum = 0.0;
    real_t dE_sum = 0.0;
    real_t dt_dt_sum = 0.0;
    real_t dE_dE_sum = 0.0;
    real_t dt_dE_sum = 0.0;

#pragma omp parallel for reduction(+ : dt_sum, dE_sum, dt_dt_sum, dE_dE_sum, dt_dE_sum)
    for (index_t idx = 0; idx < n; ++idx) {
        const real_t dt_i = dt[idx];
        const real_t dE_i = dE[idx];
        dt_sum += dt_i;
        dE_sum += dE_i;
        dt_dt_sum += dt_i * dt_i;
        dE_dE_sum += dE_i * dE_i;
        dt_dE_sum += dt_i * dE_i;
    }

    sums[0] = dt_sum;
    sums[1] = dE_sum;
    sums[2] = dt_dt_sum;
    sums[3] = dE_dE_sum;
    sums[4] = dt_dE_sum;
}
