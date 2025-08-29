#include "blond_common.h"
#include "blondmath.h"
#include "openmp.h"

real_t trapz_const_delta(const real_t *__restrict__ f,
                         const real_t deltaX,
                         const int nsub)
    {
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

extern "C" real_t beam_phase(const real_t * __restrict__ bin_centers,
                             const real_t * __restrict__ profile,
                             const real_t alpha,
                             const real_t omega_rf,
                             const real_t phi_rf,
                             const real_t bin_size,
                             const int n_bins)
{
    real_t *base = new real_t[n_bins];
    real_t *array1 = new real_t[n_bins];
    real_t *array2 = new real_t[n_bins];

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

    real_t scoeff = trapz_const_delta(array1, bin_size, n_bins);
    real_t ccoeff = trapz_const_delta(array2, bin_size, n_bins);

    delete[] base;
    delete[] array1;
    delete[] array2;

    return scoeff / ccoeff;
}


extern "C" real_t beam_phase_fast(const real_t * __restrict__ bin_centers,
                             const real_t * __restrict__ profile,
                             const real_t omega_rf,
                             const real_t phi_rf,
                             const real_t bin_size,
                             const int n_bins)
{
    real_t *array1 = new real_t[n_bins];
    real_t *array2 = new real_t[n_bins];

    #pragma omp parallel for
    for (int i = 0; i < n_bins; ++i) {
        const real_t a = omega_rf * bin_centers[i] + phi_rf;
        array1[i] = profile[i] * FAST_SIN(a);
        array2[i] = profile[i] * FAST_COS(a);
    }

    real_t scoeff = trapz_const_delta(array1, bin_size, n_bins);
    real_t ccoeff = trapz_const_delta(array2, bin_size, n_bins);

    delete[] array1;
    delete[] array2;

    return scoeff / ccoeff;
}
