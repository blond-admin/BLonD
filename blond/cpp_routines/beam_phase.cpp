#include "blondmath.h"


extern "C" double beam_phase(const double * __restrict__ bin_centers,
                             const double * __restrict__ profile,
                             const double alpha,
                             const double omega_rf,
                             const double phi_rf,
                             const double bin_size,
                             const int n_bins)
{
    double *base = new double[n_bins];
    double *array1 = new double[n_bins];
    double *array2 = new double[n_bins];

    #pragma omp parallel for
    for (int i = 0; i < n_bins; ++i) {
        base[i] = fast_exp(alpha * bin_centers[i]) * profile[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_bins; ++i) {
        const double a = omega_rf * bin_centers[i] + phi_rf;
        array1[i] = base[i] * fast_sin(a);
        array2[i] = base[i] * fast_cos(a);
    }

    double scoeff = trapz_const_delta(array1, bin_size, n_bins);
    double ccoeff = trapz_const_delta(array2, bin_size, n_bins);

    delete[] base;
    delete[] array1;
    delete[] array2;

    return scoeff / ccoeff;
}

