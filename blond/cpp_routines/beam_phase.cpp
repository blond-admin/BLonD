#include "blondmath.h"
#include "openmp.h"


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


extern "C" double beam_phase_fast(const double * __restrict__ bin_centers,
                             const double * __restrict__ profile,
                             const double omega_rf,
                             const double phi_rf,
                             const double bin_size,
                             const int n_bins)
{
    double *array1 = new double[n_bins];
    double *array2 = new double[n_bins];

    #pragma omp parallel for
    for (int i = 0; i < n_bins; ++i) {
        const double a = omega_rf * bin_centers[i] + phi_rf;
        array1[i] = profile[i] * fast_sin(a);
        array2[i] = profile[i] * fast_cos(a);
    }

    double scoeff = trapz_const_delta(array1, bin_size, n_bins);
    double ccoeff = trapz_const_delta(array2, bin_size, n_bins);

    delete[] array1;
    delete[] array2;

    return scoeff / ccoeff;
}


extern "C" float beam_phasef(const float * __restrict__ bin_centers,
                             const float * __restrict__ profile,
                             const float alpha,
                             const float omega_rf,
                             const float phi_rf,
                             const float bin_size,
                             const int n_bins)
{
    float *base = new float[n_bins];
    float *array1 = new float[n_bins];
    float *array2 = new float[n_bins];

    #pragma omp parallel for
    for (int i = 0; i < n_bins; ++i) {
        base[i] = fast_expf(alpha * bin_centers[i]) * profile[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_bins; ++i) {
        const float a = omega_rf * bin_centers[i] + phi_rf;
        array1[i] = base[i] * fast_sinf(a);
        array2[i] = base[i] * fast_cosf(a);
    }

    float scoeff = trapz_const_deltaf(array1, bin_size, n_bins);
    float ccoeff = trapz_const_deltaf(array2, bin_size, n_bins);

    delete[] base;
    delete[] array1;
    delete[] array2;

    return scoeff / ccoeff;
}
