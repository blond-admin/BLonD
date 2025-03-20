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
#pragma once

#include <complex>
#include <cstdint>
#include "blond_common.h"

extern "C"
{
  void random_normal(real_t *__restrict__ arr, const real_t mean,
                     const real_t scale, const int size,
                     long unsigned int seed = 1234);

  void where_more_than(const real_t *__restrict__ data, const int n,
                       const real_t c1,
                       bool *__restrict__ res);

  void where_less_than(const real_t *__restrict__ data, const int n,
                       const real_t c1,
                       bool *__restrict__ res);

  void where_more_less_than(const real_t *__restrict__ data, const int n,
                            const real_t c1, const real_t c2,
                            bool *__restrict__ res);

  void where_less_thanf(const float *__restrict__ data, const int n,
                        const float c1,
                        bool *__restrict__ res);

  int where(const real_t *__restrict__ dt, const int n_macroparticles,
            const real_t constant1, const real_t constant2);

  // int wheref(const float *__restrict__ dt, const int n_macroparticles,
  //            const float constant1, const float constant2);

  void where_more_than(const real_t *__restrict__ data, const int n,
                       const real_t c1,
                       bool *__restrict__ res);

  void where_less_than(const real_t *__restrict__ data, const int n,
                       const real_t c1,
                       bool *__restrict__ res);

  void where_more_less_than(const real_t *__restrict__ data, const int n,
                            const real_t c1, const real_t c2,
                            bool *__restrict__ res);

  void convolution(const real_t *__restrict__ signal,
                   const int SignalLen,
                   const real_t *__restrict__ kernel,
                   const int KernelLen,
                   real_t *__restrict__ res);

  real_t mean(const real_t *__restrict__ data, const int n);
  real_t stdev(const real_t *__restrict__ data,
               const int n);
  real_t fast_sin(real_t x);
  real_t fast_cos(real_t x);
  real_t fast_exp(real_t x);
  void fast_sinv(const real_t *__restrict__ in,
                 const int size,
                 real_t *__restrict__ out);
  void fast_cosv(const real_t *__restrict__ in,
                 const int size,
                 real_t *__restrict__ out);
  void fast_expv(const real_t *__restrict__ in,
                 const int size,
                 real_t *__restrict__ out);

  void add_int_vector(const int *__restrict__ a,
                      const int *__restrict__ b,
                      const int size,
                      int *__restrict__ result);

  void add_int_vector_inplace(int *__restrict__ a,
                              const int *__restrict__ b,
                              const int size);

  void add_uint16_vector(const uint16_t *__restrict__ a,
                         const uint16_t *__restrict__ b,
                         const int size,
                         uint16_t *__restrict__ result);

  void add_uint16_vector_inplace(uint16_t *__restrict__ a,
                                 const uint16_t *__restrict__ b,
                                 const int size);

  void add_uint32_vector(const uint32_t *__restrict__ a,
                         const uint32_t *__restrict__ b,
                         const int size,
                         uint32_t *__restrict__ result);

  void add_uint32_vector_inplace(uint32_t *__restrict__ a,
                                 const uint32_t *__restrict__ b,
                                 const int size);

  void add_longint_vector(const long *__restrict__ a,
                          const long *__restrict__ b,
                          const int size,
                          long *__restrict__ result);

  void add_longint_vector_inplace(long *__restrict__ a,
                                  const long *__restrict__ b,
                                  const int size);

  void add_double_vector(const double *__restrict__ a,
                         const double *__restrict__ b,
                         const int size,
                         double *__restrict__ result);

  void add_double_vector_inplace(double *__restrict__ a,
                                 const double *__restrict__ b,
                                 const int size);

  void add_float_vector(const float *__restrict__ a,
                        const float *__restrict__ b,
                        const int size,
                        float *__restrict__ result);

  void add_float_vector_inplace(float *__restrict__ a,
                                const float *__restrict__ b,
                                const int size);

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
              real_t *__restrict__ y);

  void interp_const_space(const real_t *__restrict__ x,
                          const int N,
                          const real_t *__restrict__ xp,
                          const int M,
                          const real_t *__restrict__ yp,
                          const real_t left,
                          const real_t right,
                          real_t *__restrict__ y);

  void interp_const_bin(const real_t *__restrict__ x,
                        const int N,
                        const real_t *__restrict__ xp,
                        const real_t *__restrict__ yp,
                        const int M,
                        const real_t left,
                        const real_t right,
                        real_t *__restrict__ y);

  // Function to implement integration of f(x); over the interval
  // [a,b] using the trapezoid rule with nsub subdivisions.
  void cumtrapz_wo_initial(const real_t *__restrict__ f,
                           const real_t deltaX,
                           const int nsub,
                           real_t *__restrict__ psum);

  // Function to implement integration of f(x); over the interval
  // [a,b] using the trapezoid rule with nsub subdivisions.
  void cumtrapz_w_initial(const real_t *__restrict__ f,
                          const real_t deltaX,
                          const real_t initial,
                          const int nsub,
                          real_t *__restrict__ psum);

  real_t trapz_var_delta(const real_t *__restrict__ f,
                         const real_t *__restrict__ deltaX,
                         const int nsub);

  real_t trapz_const_delta(const real_t *__restrict__ f,
                           const real_t deltaX,
                           const int nsub);

  int min_idx(const real_t *__restrict__ a, int size);
  int max_idx(const real_t *__restrict__ a, int size);
  void linspace(const real_t start, const real_t end, const int n,
                real_t *__restrict__ out);

  void arange_double(const double start, const double stop,
                     const double step,
                     double *__restrict__ out);

  void arange_float(const float start, const float stop,
                    const float step,
                    float *__restrict__ out);

  void arange_int(const int start, const int stop,
                  const int step,
                  int *__restrict__ out);

  real_t sum(const real_t *__restrict__ data, const int n);
  void sort_double(double *__restrict__ in, const int n, bool reverse);

  void sort_int(int *__restrict__ in, const int n, bool reverse);
  void sort_longint(int64_t *__restrict__ in, const int n, bool reverse);

  void scalar_mul_int32(const int *__restrict__ a, const int b,
                        const int n, int *__restrict__ res);

  void scalar_mul_int64(const long *__restrict__ a, const long b,
                        const int n, long *__restrict__ res);

  void scalar_mul_float32(const float *__restrict__ a, const float b,
                          const int n, float *__restrict__ res);

  void scalar_mul_float64(const double *__restrict__ a, const double b,
                          const int n, double *__restrict__ res);

  void scalar_mul_complex64(const std::complex<float> *__restrict__ a,
                            const std::complex<float> b,
                            const int n, std::complex<float> *__restrict__ res);

  void scalar_mul_complex128(const std::complex<double> *__restrict__ a,
                             const std::complex<double> b,
                             const int n, std::complex<double> *__restrict__ res);

  void vector_mul_int32(const int *__restrict__ a, const int *__restrict__ b,
                        const int n, int *__restrict__ res);

  void vector_mul_int64(const long *__restrict__ a, const long *__restrict__ b,
                        const int n, long *__restrict__ res);

  void vector_mul_float32(const float *__restrict__ a, const float *__restrict__ b,
                          const int n, float *__restrict__ res);

  void vector_mul_float64(const double *__restrict__ a, const double *__restrict__ b,
                          const int n, double *__restrict__ res);

  void vector_mul_complex64(const std::complex<float> *__restrict__ a,
                            const std::complex<float> *__restrict__ b,
                            const int n, std::complex<float> *__restrict__ res);

  void vector_mul_complex128(const std::complex<double> *__restrict__ a,
                             const std::complex<double> *__restrict__ b,
                             const int n, std::complex<double> *__restrict__ res);
}
