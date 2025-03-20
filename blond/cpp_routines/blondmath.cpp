/**
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/


C++ Math library
@Author: Konstantinos Iliakis
@Date: 20.10.2017
*/

#include <cmath>
#include <algorithm>
#include <functional>
#include <complex>
#include <cstdint>

#include "blondmath.h"
#include "blond_common.h"
#include "openmp.h"

#ifdef BOOST
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#else
#include <random>
#include <thread>
#endif

extern "C"
{

    void random_normal(real_t *__restrict__ arr, const real_t mean,
                       const real_t scale, const int size, long unsigned int seed)
    {
        /** @brief Function that generates numbers from a normal distribution
         * Similar to numpy.random.normal
         * @arr: array to fill with random numbers
         * @mean: mean of the distribution
         * @scale: standard deviation of the distribution
         * @size: size of the array
         * @seed: seed for the random number generator
         */
#ifdef BOOST
        using namespace boost;
#else
        using namespace std;
#endif

#pragma omp parallel
        {
            static __thread mt19937_64 *gen = nullptr;
            if (!gen)
                gen = new mt19937_64(seed + omp_get_thread_num());
            static thread_local normal_distribution<real_t> dist(0.0, 1.0);
#pragma omp for
            for (int i = 0; i < size; i++)
            {
                arr[i] = mean + scale*dist(*gen);
            }
        }
    }

    void where_more_than(const real_t *__restrict__ data, const int n,
                         const real_t c1,
                         bool *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            res[i] = data[i] > c1;
        }
    }

    void where_less_than(const real_t *__restrict__ data, const int n,
                         const real_t c1,
                         bool *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            res[i] = data[i] < c1;
        }
    }

    void where_more_less_than(const real_t *__restrict__ data, const int n,
                              const real_t c1, const real_t c2,
                              bool *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            res[i] = (data[i] > c1) && (data[i] < c2);
        }
    }

    int where(const real_t *__restrict__ dt, const int n_macroparticles,
              const real_t constant1, const real_t constant2)
    {
        int s = 0;
#pragma omp parallel for reduction(+ : s)
        for (int i = 0; i < n_macroparticles; i++)
        {
            s += (dt[i] < constant2 && dt[i] > constant1) ? 1 : 0;
        }
        return s;
    }

    void add_int_vector(const int *__restrict__ a,
                        const int *__restrict__ b,
                        const int size,
                        int *__restrict__ result)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void add_uint16_vector(const uint16_t *__restrict__ a,
                           const uint16_t *__restrict__ b,
                           const int size,
                           uint16_t *__restrict__ result)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void add_uint32_vector(const uint32_t *__restrict__ a,
                           const uint32_t *__restrict__ b,
                           const int size,
                           uint32_t *__restrict__ result)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void add_longint_vector(const long *__restrict__ a,
                            const long *__restrict__ b,
                            const int size,
                            long *__restrict__ result)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void add_double_vector(const double *__restrict__ a,
                           const double *__restrict__ b,
                           const int size,
                           double *__restrict__ result)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void add_float_vector(const float *__restrict__ a,
                          const float *__restrict__ b,
                          const int size,
                          float *__restrict__ result)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            result[i] = a[i] + b[i];
        }
    }

    void add_int_vector_inplace(int *__restrict__ a,
                                const int *__restrict__ b,
                                const int size)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            a[i] = a[i] + b[i];
        }
    }

    void add_uint16_vector_inplace(uint16_t *__restrict__ a,
                                   const uint16_t *__restrict__ b,
                                   const int size)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            a[i] = a[i] + b[i];
        }
    }

    void add_uint32_vector_inplace(uint32_t *__restrict__ a,
                                   const uint32_t *__restrict__ b,
                                   const int size)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            a[i] = a[i] + b[i];
        }
    }

    void add_longint_vector_inplace(long *__restrict__ a,
                                    const long *__restrict__ b,
                                    const int size)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            a[i] = a[i] + b[i];
        }
    }

    void add_double_vector_inplace(double *__restrict__ a,
                                   const double *__restrict__ b,
                                   const int size)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            a[i] = a[i] + b[i];
        }
    }

    void add_float_vector_inplace(float *__restrict__ a,
                                  const float *__restrict__ b,
                                  const int size)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            a[i] = a[i] + b[i];
        }
    }

    void convolution(const real_t *__restrict__ signal,
                     const int SignalLen,
                     const real_t *__restrict__ kernel,
                     const int KernelLen,
                     real_t *__restrict__ res)
    {
        const int size = KernelLen + SignalLen - 1;

#pragma omp parallel for
        for (int n = 0; n < size; ++n)
        {
            res[n] = 0;
            const int kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
            const int kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
            // uint j = n - kmin;
            for (int k = kmin; k <= kmax; k++)
            {
                res[n] += signal[k] * kernel[n - k];
                //--j;
            }
        }
    }

    real_t mean(const real_t *__restrict__ data, const int n)
    {
        real_t m = 0;
#pragma omp parallel for reduction(+ : m)
        for (int i = 0; i < n; ++i)
        {
            m += data[i];
        }
        return m / n;
    }

    real_t stdev(const real_t *__restrict__ data,
                 const int n)
    {
        const real_t m = mean(data, n);
        real_t sum_deviation = 0.0;

#pragma omp parallel for reduction(+ : sum_deviation)
        for (int i = 0; i < n; ++i)
            sum_deviation += (data[i] - m) * (data[i] - m);
        return sqrt(sum_deviation / n);
    }

    real_t fast_sin(real_t x) { return FAST_SIN(x); }
    real_t fast_cos(real_t x) { return FAST_COS(x); }
    real_t fast_exp(real_t x) { return FAST_EXP(x); }

    void fast_sinv(const real_t *__restrict__ in,
                   const int size,
                   real_t *__restrict__ out)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = FAST_SIN(in[i]);
    }

    void fast_cosv(const real_t *__restrict__ in,
                   const int size,
                   real_t *__restrict__ out)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = FAST_COS(in[i]);
    }

    void fast_expv(const real_t *__restrict__ in,
                   const int size,
                   real_t *__restrict__ out)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = FAST_EXP(in[i]);
    }

    /**
    Parameters are like python's np.interp

    @x: x-coordinates of the interpolated values
    @N: The x array size
    @xp: The x-coords of the data points, !!must be sorted!!
    @M: The xp array size
    @yp: the y-coords of the data points
    @left: value to return for x < xp[0]
    @right: value to return for x > xp[last]
    @y: the interpolated values, same shape as x
    */
    void interp(const real_t *__restrict__ x,
                const int N,
                const real_t *__restrict__ xp,
                const int M,
                const real_t *__restrict__ yp,
                const real_t left,
                const real_t right,
                real_t *__restrict__ y)
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            int pos = std::lower_bound(xp, xp + M, x[i]) - xp;
            if (pos == M)
                y[i] = right;
            else if (xp[pos] == x[i])
                y[i] = yp[pos];
            else if (pos == 0)
                y[i] = left;
            else
            {
                y[i] = yp[pos - 1] +
                       (yp[pos] - yp[pos - 1]) * (x[i] - xp[pos - 1]) /
                           (xp[pos] - xp[pos - 1]);
            }
        }
    }

    /**
    @x: x-coordinates of the interpolated values
    @N: The x array size
    @xp: The x-coords of the data points, !!must be sorted!!
    @M: The xp array size
    @yp: the y-coords of the data points
    @left: value to return for x < xp[0]
    @right: value to return for x > xp[last]
    @y: the interpolated values, same shape as x
    */
    void interp_const_space(const real_t *__restrict__ x,
                            const int N,
                            const real_t *__restrict__ xp,
                            const int M,
                            const real_t *__restrict__ yp,
                            const real_t left,
                            const real_t right,
                            real_t *__restrict__ y)
    {

        const int offset = std::lower_bound(xp, xp + M, x[0]) - xp;
        const real_t c = (x[0] - xp[0] + (1 - offset) * (xp[1] - xp[0])) / (xp[1] - xp[0]);

#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            const int pos = i + offset;
            if (pos >= M)
                y[i] = right;
            else if (pos == 0)
                y[i] = left;
            else
                y[i] = yp[pos - 1] + (yp[pos] - yp[pos - 1]) * c;
            // else if (xp[pos] == x[i])
            //     y[i] = yp[pos];
        }
    }

    void interp_const_bin(const real_t *__restrict__ x, const int N,
                          const real_t *__restrict__ xp,
                          const real_t *__restrict__ yp, const int M,
                          const real_t left, const real_t right,
                          real_t *__restrict__ y)
    {
        /**
         * @brief Linear interpolation with constant bin size (space between xp values is constant)
         * 
         * @x: x-coordinates of the interpolated values
         * @N: The x array size
         * @xp: The x-coords of the data points, !!must be sorted!!
         * @yp: the y-coords of the data points
         * @M: The xp array size
         * @left: value to return for x < xp[0]
         * @right: value to return for x > xp[last]
         * @y: the interpolated values, same shape as x
         */
        const real_t dx = xp[1] - xp[0];
        const real_t inv_dx = 1.0 / dx;
        const real_t xp0 = xp[0];
        const real_t xplast = xp[M - 1];

#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            if (x[i] < xp0)
                y[i] = left;
            else if (x[i] > xplast)
                y[i] = right;
            else
            {
                const real_t fpos = (x[i] - xp0) * inv_dx;
                const int pos = floor(fpos);
                y[i] = yp[pos] + (yp[pos + 1] - yp[pos]) * (fpos - pos);
            }
        }
    }

    // Function to implement integration of f(x) over the interval
    // [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_wo_initial(const real_t *__restrict__ f,
                             const real_t deltaX,
                             const int nsub,
                             real_t *__restrict__ psum)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        const real_t half_dx = deltaX / 2.0;
        psum[0] = (f[1] + f[0]) * half_dx;

        for (int i = 1; i < nsub - 1; ++i)
            psum[i] = psum[i - 1] + (f[i + 1] + f[i]) * half_dx;
    }

    // Function to implement integration of f(x) over the interval
    // [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_w_initial(const real_t *__restrict__ f,
                            const real_t deltaX,
                            const real_t initial,
                            const int nsub,
                            real_t *__restrict__ psum)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        const real_t half_dx = deltaX / 2.0;

        psum[0] = initial;
        psum[1] = (f[1] + f[0]) * half_dx;
        // increment the partial sum
        for (int i = 2; i < nsub; ++i)
            psum[i] = psum[i - 1] + (f[i] + f[i - 1]) * half_dx;
    }

    real_t trapz_var_delta(const real_t *__restrict__ f,
                           const real_t *__restrict__ deltaX,
                           const int nsub)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions

        real_t psum = 0.0;
// increment the partial sum
#pragma omp parallel for reduction(+ : psum)
        for (int i = 1; i < nsub; ++i)
            psum += (f[i] + f[i - 1]) * (deltaX[i] - deltaX[i - 1]);

        return psum / 2.;
    }

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

    int min_idx(const real_t *__restrict__ a, int size)
    {
        return (int)(std::min_element(a, a + size) - a);
    }

    int max_idx(const real_t *__restrict__ a, int size)
    {
        return (int)(std::max_element(a, a + size) - a);
    }

    void linspace(const real_t start, const real_t end, const int n,
                  real_t *__restrict__ out)
    {
        const real_t step = (end - start) / (n - 1);
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
            out[i] = start + i * step;
    }

    void arange_double(const double start, const double stop,
                       const double step,
                       double *__restrict__ out)
    {
        const int size = (int)std::ceil((stop - start) / step);
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = start + i * step;
    }

    void arange_float(const float start, const float stop,
                      const float step,
                      float *__restrict__ out)
    {
        const int size = (int)std::ceil((stop - start) / step);
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = start + i * step;
    }

    void arange_int(const int start, const int stop,
                    const int step,
                    int *__restrict__ out)
    {
        const int size = (int)std::ceil((stop - start) / step);
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = start + i * step;
    }

    real_t sum(const real_t *__restrict__ data, const int n)
    {
        real_t m = 0.0;
#pragma omp parallel for reduction(+ : m)
        for (int i = 0; i < n; ++i)
            m += data[i];
        return m;
    }

    void sort_double(double *__restrict__ in, const int n, bool reverse)
    {
        if (reverse)
            std::sort(in, in + n, std::greater<double>());
        else
            std::sort(in, in + n);
    }

    void sort_float(float *__restrict__ in, const int n, bool reverse)
    {
        if (reverse)
            std::sort(in, in + n, std::greater<float>());
        else
            std::sort(in, in + n);
    }

    void sort_int(int *__restrict__ in, const int n, bool reverse)
    {
        if (reverse)
            std::sort(in, in + n, std::greater<int>());
        else
            std::sort(in, in + n);
    }

    void sort_longint(int64_t *__restrict__ in, const int n, bool reverse)
    {
        if (reverse)
            std::sort(in, in + n, std::greater<int64_t>());
        else
            std::sort(in, in + n);
    }

    void scalar_mul_int32(const int *__restrict__ a, const int b,
                          const int n, int *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b;
        }
    }

    void scalar_mul_int64(const long *__restrict__ a, const long b,
                          const int n, long *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b;
        }
    }

    void scalar_mul_float32(const float *__restrict__ a, const float b,
                            const int n, float *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b;
        }
    }

    void scalar_mul_float64(const double *__restrict__ a, const double b,
                            const int n, double *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b;
        }
    }

    void scalar_mul_complex64(const std::complex<float> *__restrict__ a,
                              const std::complex<float> b,
                              const int n, std::complex<float> *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b;
        }
    }

    void scalar_mul_complex128(const std::complex<double> *__restrict__ a,
                               const std::complex<double> b,
                               const int n, std::complex<double> *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b;
        }
    }

    void vector_mul_int32(const int *__restrict__ a, const int *__restrict__ b,
                          const int n, int *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b[i];
        }
    }

    void vector_mul_int64(const long *__restrict__ a, const long *__restrict__ b,
                          const int n, long *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b[i];
        }
    }

    void vector_mul_float32(const float *__restrict__ a, const float *__restrict__ b,
                            const int n, float *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b[i];
        }
    }

    void vector_mul_float64(const double *__restrict__ a, const double *__restrict__ b,
                            const int n, double *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b[i];
        }
    }

    void vector_mul_complex64(const std::complex<float> *__restrict__ a,
                              const std::complex<float> *__restrict__ b,
                              const int n, std::complex<float> *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b[i];
        }
    }

    void vector_mul_complex128(const std::complex<double> *__restrict__ a,
                               const std::complex<double> *__restrict__ b,
                               const int n, std::complex<double> *__restrict__ res)
    {
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            res[i] = a[i] * b[i];
        }
    }
}
