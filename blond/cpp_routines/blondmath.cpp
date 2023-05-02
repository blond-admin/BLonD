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


#include "sin.h"
#include "exp.h"
#include "cos.h"
#include <cmath>
#include <algorithm>
#include <functional>
#include "blondmath.h"
#include "openmp.h"

using namespace std;

extern "C" {

    void where_more_than(const double *__restrict__ data, const int n,
                         const double c1,
                         bool *__restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            res[i] = data[i] > c1;
        }
    }

    void where_less_than(const double *__restrict__ data, const int n,
                         const double c1,
                         bool *__restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            res[i] = data[i] < c1;
        }
    }

    void where_more_less_than(const double *__restrict__ data, const int n,
                              const double c1, const double c2,
                              bool *__restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            res[i] = (data[i] > c1) && (data[i] < c2);
        }
    }


    void where_more_thanf(const float *__restrict__ data, const int n,
                          const float c1,
                          bool *__restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            res[i] = data[i] > c1;
        }
    }

    void where_less_thanf(const float *__restrict__ data, const int n,
                          const float c1,
                          bool *__restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            res[i] = data[i] < c1;
        }
    }

    void where_more_less_thanf(const float *__restrict__ data, const int n,
                               const float c1, const float c2,
                               bool *__restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            res[i] = (data[i] > c1) && (data[i] < c2);
        }
    }


    int where(const double *__restrict__ dt, const int n_macroparticles,
              const double constant1, const double constant2)
    {
        int s = 0;
        #pragma omp parallel for reduction(+:s)
        for (int i = 0; i < n_macroparticles; i++) {
            s += (dt[i] < constant2 && dt[i] > constant1) ? 1 : 0;
        }
        return s;
    }


    int wheref(const float *__restrict__ dt, const int n_macroparticles,
               const float constant1, const float constant2)
    {
        int s = 0;
        #pragma omp parallel for reduction(+:s)
        for (int i = 0; i < n_macroparticles; i++) {
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
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }

    void add_uint16_vector(const uint16_t *__restrict__ a,
                           const uint16_t *__restrict__ b,
                           const int size,
                           uint16_t *__restrict__ result)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }

    void add_uint32_vector(const uint32_t *__restrict__ a,
                           const uint32_t *__restrict__ b,
                           const int size,
                           uint32_t *__restrict__ result)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }


    void add_longint_vector(const long *__restrict__ a,
                            const long *__restrict__ b,
                            const int size,
                            long *__restrict__ result)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }


    void add_double_vector(const double * __restrict__ a,
                           const double * __restrict__ b,
                           const int size,
                           double * __restrict__ result)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }


    void add_float_vector(const float * __restrict__ a,
                          const float * __restrict__ b,
                          const int size,
                          float * __restrict__ result)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }


    void add_int_vector_inplace(int *__restrict__ a,
                                const int *__restrict__ b,
                                const int size)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] + b[i];
        }
    }

    void add_uint16_vector_inplace(uint16_t *__restrict__ a,
                                   const uint16_t *__restrict__ b,
                                   const int size)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] + b[i];
        }
    }

    void add_uint32_vector_inplace(uint32_t *__restrict__ a,
                                   const uint32_t *__restrict__ b,
                                   const int size)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] + b[i];
        }
    }


    void add_longint_vector_inplace(long *__restrict__ a,
                                    const long *__restrict__ b,
                                    const int size)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] + b[i];
        }
    }


    void add_double_vector_inplace(double * __restrict__ a,
                                   const double * __restrict__ b,
                                   const int size)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] + b[i];
        }
    }

    void add_float_vector_inplace(float * __restrict__ a,
                                  const float * __restrict__ b,
                                  const int size)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] + b[i];
        }
    }



    void convolution(const double * __restrict__ signal,
                     const int SignalLen,
                     const double * __restrict__ kernel,
                     const int KernelLen,
                     double * __restrict__ res)
    {
        const int size = KernelLen + SignalLen - 1;

        #pragma omp parallel for
        for (int n = 0; n < size; ++n) {
            res[n] = 0;
            const int kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
            const int kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
            // uint j = n - kmin;
            for (int k = kmin; k <= kmax; k++) {
                res[n] += signal[k] * kernel[n - k];
                //--j;
            }
        }
    }


    void convolutionf(const float * __restrict__ signal,
                      const int SignalLen,
                      const float * __restrict__ kernel,
                      const int KernelLen,
                      float * __restrict__ res)
    {
        const int size = KernelLen + SignalLen - 1;

        #pragma omp parallel for
        for (int n = 0; n < size; ++n) {
            res[n] = 0;
            const int kmin = (n >= KernelLen - 1) ? n - (KernelLen - 1) : 0;
            const int kmax = (n < SignalLen - 1) ? n : SignalLen - 1;
            // uint j = n - kmin;
            for (int k = kmin; k <= kmax; k++) {
                res[n] += signal[k] * kernel[n - k];
                //--j;
            }
        }
    }


    double mean(const double * __restrict__ data, const int n)
    {
        double m = 0;
        #pragma omp parallel for reduction(+:m)
        for (int i = 0; i < n; ++i) {
            m += data[i];
        }
        return m / n;
    }

    double stdev(const double * __restrict__ data,
                 const int n)
    {
        const double m = mean(data, n);
        double sum_deviation = 0.0;

        #pragma omp parallel for reduction(+:sum_deviation)
        for (int i = 0; i < n; ++i)
            sum_deviation += (data[i] - m) * (data[i] - m);
        return sqrt(sum_deviation / n);
    }

    float meanf(const float * __restrict__ data, const int n)
    {
        float m = 0;
        #pragma omp parallel for reduction(+:m)
        for (int i = 0; i < n; ++i) {
            m += data[i];
        }
        return m / n;
    }

    float stdevf(const float * __restrict__ data,
                 const int n)
    {
        const float m = meanf(data, n);
        float sum_deviation = 0.0;

        #pragma omp parallel for reduction(+:sum_deviation)
        for (int i = 0; i < n; ++i)
            sum_deviation += (data[i] - m) * (data[i] - m);
        return sqrt(sum_deviation / n);
    }


    double fast_sin(double x) {return vdt::fast_sin(x);}
    double fast_cos(double x) {return vdt::fast_cos(x);}
    double fast_exp(double x) {return vdt::fast_exp(x);}

    float fast_sinf(float x) {return vdt::fast_sinf(x);}
    float fast_cosf(float x) {return vdt::fast_cosf(x);}
    float fast_expf(float x) {return vdt::fast_expf(x);}

    void fast_sinv(const double * __restrict__ in,
                   const int size,
                   double * __restrict__ out)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = fast_sin(in[i]);
    }

    void fast_cosv(const double * __restrict__ in,
                   const int size,
                   double * __restrict__ out)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = fast_cos(in[i]);
    }

    void fast_expv(const double * __restrict__ in,
                   const int size,
                   double * __restrict__ out)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = fast_exp(in[i]);
    }


    void fast_sinvf(const float * __restrict__ in,
                    const int size,
                    float * __restrict__ out)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = fast_sin(in[i]);
    }

    void fast_cosvf(const float * __restrict__ in,
                    const int size,
                    float * __restrict__ out)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = fast_cos(in[i]);
    }

    void fast_expvf(const float * __restrict__ in,
                    const int size,
                    float * __restrict__ out)
    {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i)
            out[i] = fast_exp(in[i]);
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
    void interp(const double * __restrict__ x,
                const int N,
                const double * __restrict__ xp,
                const int M,
                const double * __restrict__ yp,
                const double left,
                const double right,
                double * __restrict__ y)
    {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            int pos = std::lower_bound(xp, xp + M, x[i]) - xp;
            if (pos == M)
                y[i] = right;
            else if (xp[pos] == x[i])
                y[i] = yp[pos];
            else if (pos == 0)
                y[i] = left;
            else {
                y[i] = yp[pos - 1] +
                       (yp[pos] - yp[pos - 1]) * (x[i] - xp[pos - 1]) /
                       (xp[pos] - xp[pos - 1]);
            }
        }
    }

    void interpf(const float * __restrict__ x,
                 const int N,
                 const float * __restrict__ xp,
                 const int M,
                 const float * __restrict__ yp,
                 const float left,
                 const float right,
                 float * __restrict__ y)
    {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            int pos = std::lower_bound(xp, xp + M, x[i]) - xp;
            if (pos == M)
                y[i] = right;
            else if (xp[pos] == x[i])
                y[i] = yp[pos];
            else if (pos == 0)
                y[i] = left;
            else {
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
    void interp_const_space(const double * __restrict__ x,
                            const int N,
                            const double * __restrict__ xp,
                            const int M,
                            const double * __restrict__ yp,
                            const double left,
                            const double right,
                            double * __restrict__ y)
    {

        const int offset = std::lower_bound(xp, xp + M, x[0]) - xp;
        const double c = (x[0] - xp[0] + (1 - offset) * (xp[1] - xp[0]))
                         / (xp[1] - xp[0]);

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
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

    void interp_const_spacef(const float * __restrict__ x,
                             const int N,
                             const float * __restrict__ xp,
                             const int M,
                             const float * __restrict__ yp,
                             const float left,
                             const float right,
                             float * __restrict__ y)
    {

        const int offset = std::lower_bound(xp, xp + M, x[0]) - xp;
        const float c = (x[0] - xp[0] + (1 - offset) * (xp[1] - xp[0]))
                        / (xp[1] - xp[0]);

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
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


// Function to implement integration of f(x) over the interval
// [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_wo_initial(const double * __restrict__ f,
                             const double deltaX,
                             const int nsub,
                             double * __restrict__ psum)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        const double half_dx = deltaX / 2.0;
        psum[0] = (f[1] + f[0]) * half_dx;

        for (int i = 1; i < nsub - 1; ++i)
            psum[i] = psum[i - 1] + (f[i + 1] + f[i]) * half_dx;
    }

// Function to implement integration of f(x) over the interval
// [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_w_initial(const double * __restrict__ f,
                            const double deltaX,
                            const double initial,
                            const int nsub,
                            double * __restrict__ psum)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        const double half_dx = deltaX / 2.0;

        psum[0] = initial;
        psum[1] = (f[1] + f[0]) * half_dx;
        // increment the partial sum
        for (int i = 2; i < nsub; ++i)
            psum[i] = psum[i - 1] + (f[i] + f[i - 1]) * half_dx;
    }


    double trapz_var_delta(const double * __restrict__ f,
                           const double * __restrict__ deltaX,
                           const int nsub)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions

        double psum = 0.0;
        // increment the partial sum
        #pragma omp parallel for reduction(+ : psum)
        for (int i = 1; i < nsub; ++i)
            psum += (f[i] + f[i - 1]) * (deltaX[i] - deltaX[i - 1]);

        return psum / 2.;
    }

    double trapz_const_delta(const double * __restrict__ f,
                             const double deltaX,
                             const int nsub)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        double psum = (f[0] + f[nsub - 1]) / 2.; // f(a)+f(b);

        // increment the partial sum
        #pragma omp parallel for reduction(+ : psum)
        for (int i = 1; i < nsub - 1; ++i)
            psum += f[i];

        // multiply the sum by the constant deltaX/2.0
        // return approximation
        return deltaX * psum;;
    }

    int min_idx(const double * __restrict__ a, int size)
    {
        return (int) (std::min_element(a, a + size) - a);
    }

    int max_idx(const double * __restrict__ a, int size)
    {
        return (int) (std::max_element(a, a + size) - a);
    }


    void linspace(const double start, const double end, const int n,
                  double * __restrict__ out)
    {
        const double step = (end - start) / (n - 1);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) out[i] = start + i * step;
    }

    void arange_double(const double start, const double stop,
                       const double step,
                       double * __restrict__ out)
    {
        const int size = (int) std::ceil((stop - start) / step);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) out[i] = start + i * step;
    }


// Function to implement integration of f(x) over the interval
// [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_wo_initialf(const float * __restrict__ f,
                              const float deltaX,
                              const int nsub,
                              float * __restrict__ psum)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        const float half_dx = deltaX / 2.0;
        psum[0] = (f[1] + f[0]) * half_dx;

        for (int i = 1; i < nsub - 1; ++i)
            psum[i] = psum[i - 1] + (f[i + 1] + f[i]) * half_dx;
    }

// Function to implement integration of f(x) over the interval
// [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_w_initialf(const float * __restrict__ f,
                             const float deltaX,
                             const float initial,
                             const int nsub,
                             float * __restrict__ psum)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        const float half_dx = deltaX / 2.0;

        psum[0] = initial;
        psum[1] = (f[1] + f[0]) * half_dx;
        // increment the partial sum
        for (int i = 2; i < nsub; ++i)
            psum[i] = psum[i - 1] + (f[i] + f[i - 1]) * half_dx;
    }


    float trapz_var_deltaf(const float * __restrict__ f,
                           const float * __restrict__ deltaX,
                           const int nsub)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions

        float psum = 0.0;
        // increment the partial sum
        #pragma omp parallel for reduction(+ : psum)
        for (int i = 1; i < nsub; ++i)
            psum += (f[i] + f[i - 1]) * (deltaX[i] - deltaX[i - 1]);

        return psum / 2.;
    }

    float trapz_const_deltaf(const float * __restrict__ f,
                             const float deltaX,
                             const int nsub)
    {
        // initialize the partial sum to be f(a)+f(b) and
        // deltaX to be the step size using nsub subdivisions
        float psum = (f[0] + f[nsub - 1]) / 2.; // f(a)+f(b);

        // increment the partial sum
        #pragma omp parallel for reduction(+ : psum)
        for (int i = 1; i < nsub - 1; ++i)
            psum += f[i];

        // multiply the sum by the constant deltaX/2.0
        // return approximation
        return deltaX * psum;;
    }

    int min_idxf(const float * __restrict__ a, int size)
    {
        return (int) (std::min_element(a, a + size) - a);
    }

    int max_idxf(const float * __restrict__ a, int size)
    {
        return (int) (std::max_element(a, a + size) - a);
    }


    void linspacef(const float start, const float end, const int n,
                   float * __restrict__ out)
    {
        const float step = (end - start) / (n - 1);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) out[i] = start + i * step;
    }

    void arange_float(const float start, const float stop,
                      const float step,
                      float * __restrict__ out)
    {
        const int size = (int) std::ceil((stop - start) / step);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) out[i] = start + i * step;
    }



    void arange_int(const int start, const int stop,
                    const int step,
                    int * __restrict__ out)
    {
        const int size = (int) std::ceil((stop - start) / step);
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) out[i] = start + i * step;
    }





    double sum(const double * __restrict__ data, const int n)
    {
        double m = 0.0;
        #pragma omp parallel for reduction(+ : m)
        for (int i = 0; i < n; ++i) m += data[i];
        return m;
    }


    void sort_double(double * __restrict__ in, const int n, bool reverse)
    {
        if (reverse) std::sort(in, in + n, std::greater<double>());
        else std::sort(in, in + n);
    }



    float sumf(const float * __restrict__ data, const int n)
    {
        float m = 0.0;
        #pragma omp parallel for reduction(+ : m)
        for (int i = 0; i < n; ++i) m += data[i];
        return m;
    }


    void sort_float(float * __restrict__ in, const int n, bool reverse)
    {
        if (reverse) std::sort(in, in + n, std::greater<float>());
        else std::sort(in, in + n);
    }


    void sort_int(int * __restrict__ in, const int n, bool reverse)
    {
        if (reverse) std::sort(in, in + n, std::greater<int>());
        else std::sort(in, in + n);
    }


    void sort_longint(long int * __restrict__ in, const int n, bool reverse)
    {
        if (reverse) std::sort(in, in + n, std::greater<long int>());
        else std::sort(in, in + n);
    }

    void scalar_mul_int32(const int * __restrict__ a, const int b,
                          const int n, int * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b;
        }
        // std::transform(a, a + n, res, bind2nd(multiplies<int>(), b));
    }

    void scalar_mul_int64(const long * __restrict__ a, const long b,
                          const int n, long * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b;
        }
        // std::transform(a, a + n, res, bind2nd(multiplies<long>(), b));
    }

    void scalar_mul_float32(const float * __restrict__ a, const float b,
                            const int n, float * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b;
        }
        // std::transform(a, a + n, res, bind2nd(multiplies<float>(), b));
    }

    void scalar_mul_float64(const double * __restrict__ a, const double b,
                            const int n, double * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b;
        }
        // std::transform(a, a + n, res, bind2nd(multiplies<double>(), b));
    }

    void scalar_mul_complex64(const complex<float> * __restrict__ a,
                              const complex<float> b,
                              const int n, complex<float> * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b;
        }
        // std::transform(a, a + n, res, bind2nd(multiplies<complex<float>>(), b));
    }

    void scalar_mul_complex128(const complex<double> * __restrict__ a,
                               const complex<double> b,
                               const int n, complex<double> * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b;
        }
        // std::transform(a, a + n, res, bind2nd(multiplies<complex<double>>(), b));
    }

    void vector_mul_int32(const int * __restrict__ a, const int *__restrict__ b,
                          const int n, int * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b[i];
        }
        // std::transform(a, a + n, b, res, multiplies<int>());
    }

    void vector_mul_int64(const long * __restrict__ a, const long *__restrict__ b,
                          const int n, long * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b[i];
        }
        // std::transform(a, a + n, b, res, multiplies<long>());
    }

    void vector_mul_float32(const float * __restrict__ a, const float *__restrict__ b,
                            const int n, float * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b[i];
        }
        // std::transform(a, a + n, b, res, multiplies<float>());

    }

    void vector_mul_float64(const double * __restrict__ a, const double *__restrict__ b,
                            const int n, double * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b[i];
        }
        // std::transform(a, a + n, b, res, multiplies<double>());

    }

    void vector_mul_complex64(const complex<float> * __restrict__ a,
                              const complex<float> *__restrict__ b,
                              const int n, complex<float> * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b[i];
        }
        // std::transform(a, a + n, b, res, multiplies<complex<float>>());

    }

    void vector_mul_complex128(const complex<double> * __restrict__ a,
                               const complex<double> *__restrict__ b,
                               const int n, complex<double> * __restrict__ res)
    {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            res[i] = a[i] * b[i];
        }
        // std::transform(a, a + n, b, res, multiplies<complex<double>>());

    }


}

