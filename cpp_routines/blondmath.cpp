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

#ifdef PARALLEL
#include <omp.h>
#endif


extern "C" {

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


    double fast_sin(double x) {return vdt::fast_sin(x);}

    double fast_cos(double x) {return vdt::fast_cos(x);}

    double fast_exp(double x) {return vdt::fast_exp(x);}

    // float fast_sinf(float x) {return vdt::fast_sinf(x);}
    // float fast_cosf(float x) {return vdt::fast_cosf(x);}
    // float fast_expf(float x) {return vdt::fast_expf(x);}

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
                  double *__restrict__ out)
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
        if(reverse) std::sort(in, in + n, std::greater<double>());
        else std::sort(in, in + n);
    }

    void sort_int(int * __restrict__ in, const int n, bool reverse)
    {
        if(reverse) std::sort(in, in + n, std::greater<int>());
        else std::sort(in, in + n);
    }


    void sort_longint(long int * __restrict__ in, const int n, bool reverse)
    {
        if(reverse) std::sort(in, in + n, std::greater<long int>());
        else std::sort(in, in + n);
    }

}
