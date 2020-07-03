/*
 * fft.cpp
 *
 *  Created on: Mar 21, 2016
 *      Author: kiliakis
 */

#ifdef USEFFTW3

#include "fft.h"
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <functional>
#include <iostream>
#include "openmp.h"


// FFTW_PATIENT: run a lot of ffts to discover the best plan.
// Will not use the multithreaded version unless the fft size
// is big enough

// FFTW_MEASURE : run some ffts to find the best plan.
// (first run should take some more seconds)

// FFTW_ESTIMATE : dont run any ffts, just make an estimation.
// Usually leads to suboptimal solutions

// FFTW_DESTROY_INPUT : use the original input to store arbitaty data.
// May yield better performance but the input is not usable any more.
// Can be combined with all the above

static std::vector<fft_plan_t> planV;
static std::vector<fftf_plan_t> planVf;
static bool hasBeenInit = false;
const unsigned FFTW_FLAGS = FFTW_MEASURE | FFTW_DESTROY_INPUT;

using namespace std;
// Parameters are like python's numpy.fft.rfft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the transformed array
extern "C" {

    fftw_plan init_fft(const int n,  complex128_t *in, complex128_t *out,
                       const int sign = FFTW_FORWARD,
                       const unsigned flag = FFTW_ESTIMATE,
                       const int threads = 1)
    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
#endif
        // cout << "Threads: " << threads << "\n";

        fftw_complex *a = reinterpret_cast<fftw_complex *>(in);
        fftw_complex *b = reinterpret_cast<fftw_complex *>(out);
        return fftw_plan_dft_1d(n, a, b, sign, flag);
    }

    fftw_plan init_rfft(const int n, double *in, complex128_t *out,
                        const unsigned flag = FFTW_ESTIMATE,
                        const int threads = 1)

    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
#endif
        // cout << "Threads: " << threads << "\n";
        fftw_complex *b = reinterpret_cast<fftw_complex *>(out);
        return fftw_plan_dft_r2c_1d(n, in, b, flag);
    }

    fftw_plan init_irfft(const int n, complex128_t *in, double *out,
                         const unsigned flag = FFTW_ESTIMATE,
                         const int threads = 1)
    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
#endif

        // cout << "Threads: " << threads << "\n";

        fftw_complex *b = reinterpret_cast<fftw_complex *>(in);
        return fftw_plan_dft_c2r_1d(n, b, out, flag);
    }


    fftw_plan init_irfft_packed(const int n, const int howmany, complex128_t *in, double *out,
                                const unsigned flag = FFTW_ESTIMATE,
                                const int threads = 1)
    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftw_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftw_plan_with_nthreads(threads);
            }
        }
#endif
        // cout << "Threads: " << threads << "\n";

        fftw_complex *b = reinterpret_cast<fftw_complex *>(in);

        return fftw_plan_many_dft_c2r(1, &n, howmany,
                                      b, NULL,
                                      1, n / 2 + 1,
                                      out, NULL,
                                      1, n,
                                      flag);
        // return fftw_plan_dft_c2r_2d(n, n1, b, out, flag);
    }

    // void run_fft(const fftw_plan &p) { fftw_execute(p);}

    // void destroy_fft(fftw_plan &p) { fftw_destroy_plan(p); }




    fft_plan_t find_plan(int fftSize, int inSize, fft_type_t type, int threads,
                         vector<fft_plan_t> &v)
    {
        // const uint flag = FFTW_FLAGS;
        auto it =
        find_if(v.begin(), v.end(), [inSize, fftSize, type](const fft_plan_t &s) {
            return ((s.inSize == inSize) && (s.fftSize == fftSize) && (s.type == type)
                    && (s.howmany == 1));
        });

        if (it == v.end()) {
            fft_plan_t plan;
            plan.inSize = inSize;
            plan.fftSize = fftSize;
            plan.type = type;

            if (type == FFT) {
                fftw_complex *in =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);
                fftw_complex *out =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);

                auto p = init_fft(fftSize, reinterpret_cast<complex128_t *>(in),
                                  reinterpret_cast<complex128_t *>(out),
                                  FFTW_FORWARD, FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;
            } else if (type == IFFT) {

                fftw_complex *in =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);
                fftw_complex *out =
                    (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fftSize);
                auto p = init_fft(fftSize, reinterpret_cast<complex128_t *>(in),
                                  reinterpret_cast<complex128_t *>(out),
                                  FFTW_BACKWARD, FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else if (type == RFFT) {
                double *in = (double *)fftw_malloc(sizeof(double) * inSize);
                fftw_complex *out = (fftw_complex *)fftw_malloc(
                                        sizeof(fftw_complex) * fftSize);
                auto p = init_rfft(inSize, in, reinterpret_cast<complex128_t *>(out),
                                   FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else if (type == IRFFT) {
                fftw_complex *in = (fftw_complex *)fftw_malloc(
                                       sizeof(fftw_complex) * inSize);
                double *out = (double *)fftw_malloc(sizeof(double) * fftSize);
                auto p = init_irfft(fftSize, reinterpret_cast<complex128_t *>(in), out,
                                    FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else {
                printf("[fft::find_plan]: ERROR Wrong fft type!\n");
                exit(-1);
            }

            v.push_back(plan);
            return plan;
        } else {
            return *it;
        }
    }



    fft_plan_t find_plan_packed(int fftSize, int howmany, int inSize, fft_type_t type, int threads,
                                vector<fft_plan_t> &v)
    {
        // const uint flag = FFTW_FLAGS;
        auto it =
        find_if(v.begin(), v.end(), [inSize, fftSize, howmany, type](const fft_plan_t &s) {
            return ((s.inSize == inSize) && (s.fftSize == fftSize) && (s.type == type)
                    && (s.howmany == howmany));
        });

        if (it == v.end()) {
            fft_plan_t plan;
            plan.inSize = inSize;
            plan.fftSize = fftSize;
            plan.howmany = howmany;
            plan.type = type;

            if (type == IRFFT) {
                fftw_complex *in = (fftw_complex *)fftw_malloc(
                                       sizeof(fftw_complex) * inSize * howmany);
                double *out = (double *)fftw_malloc(sizeof(double) * fftSize * howmany);

                auto p = init_irfft_packed(fftSize, howmany, reinterpret_cast<complex128_t *>(in), out,
                                           FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else {
                printf("[fft::find_plan]: ERROR Wrong fft type!\n");
                exit(-1);
            }

            v.push_back(plan);
            return plan;
        } else {
            return *it;
        }
    }



    void destroy_plans()
    {
        for (auto &i : planV) {
            fftw_destroy_plan(i.p);
            fftw_free(i.in);
            fftw_free(i.out);
        }
        planV.clear();

        for (auto &i : planVf) {
            fftwf_destroy_plan(i.p);
            fftwf_free(i.in);
            fftwf_free(i.out);
        }
        planVf.clear();

    }

    // rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // @n: Number of points in the input to use. If more than inSize, pad zeros.
    //     if less crop.
    void rfft(double *in, const int inSize,
              complex128_t *out, int n,
              const int threads)
    {
        n = n == 0 ? n = inSize : n;
        const int outSize = n / 2 + 1;

        auto plan = find_plan(outSize, n, RFFT, threads, planV);
        auto from = (double *)plan.in;
        auto to = (complex128_t *)plan.out;

        if (n <= inSize)
            copy(in, in + n, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + n, 0.0);
        }

        fftw_execute(plan.p);
        copy(to, to + outSize, out);
    }

    // Inverse of rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // Missing n: size of output
    void irfft(complex128_t *in, const int inSize,
               double *out, int outSize,
               const int threads)
    {
        outSize = outSize == 0 ? outSize = 2 * (inSize - 1) : outSize;
        const int n = outSize / 2 + 1;

        auto plan = find_plan(outSize, n, IRFFT, threads, planV);
        auto from = (complex128_t *)plan.in;
        auto to = (double *)plan.out;

        if (n <= inSize)
            copy(in, in + n, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + n, 0.0);
        }
        // for(int i =0; i < n; i++)
        //     cout << from[i] << "\t";
        // cout << "\n";

        fftw_execute(plan.p);

        transform(to, to + outSize, out,
                  bind2nd(divides<double>(), outSize));
    }

    // Inverse of rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // n: size of output
    // howmnay: how many ffts of size n0 to perform
    void irfft_packed(complex128_t *in, const int n0, const int howmany,
                      double *out, int outSize,
                      const int threads)
    {
        outSize = outSize == 0 ? outSize = 2 * (n0 - 1) : outSize;

        const int n = outSize / 2 + 1;

        auto plan = find_plan_packed(outSize, howmany, n, IRFFT, threads, planV);

        auto from = (complex128_t *) plan.in;
        auto to = (double *) plan.out;


        if (n <= n0)
            copy(in, in + howmany * n, (complex128_t *) from);
        else {
            copy(in, in + howmany * n0, (complex128_t *) from);
            fill((complex128_t *) from + howmany * n0, (complex128_t *) from + howmany * n, 0.0);
        }
        fftw_execute(plan.p);

        transform(to, to + howmany * outSize, out,
                  bind2nd(divides<double>(), outSize));


    }


    // Parameters are like python's numpy.fft.ifft
    // @in:  input data
    // @n:   number of points to use. If n < in.size() then the input is cropped
    //       if n > in.size() then input is padded with zeros
    // @out: the inverse Fourier transform of input data
    void ifft(complex128_t *in, const int inSize,
              complex128_t *out, int fftSize,
              const int threads)
    {
        if (fftSize == 0) fftSize = inSize;

        auto plan = find_plan(fftSize, inSize, IFFT, threads, planV);
        auto from = (complex128_t *)plan.in;
        auto to = (complex128_t *)plan.out;
        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }

        fftw_execute(plan.p);

        transform(&to[0], &to[fftSize], out,
                  bind2nd(divides<complex128_t>(), fftSize));
    }


// Parameters are like python's numpy.fft.fft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the transformed array
    void fft(complex128_t *in, const int inSize,
             complex128_t *out, int fftSize,
             const int threads)
    {
        if (fftSize == 0) fftSize = inSize;

        auto plan = find_plan(fftSize, inSize, FFT, threads, planV);
        auto from = (complex128_t *)plan.in;
        auto to = (complex128_t *)plan.out;

        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }
        fftw_execute(plan.p);

        copy(&to[0], &to[fftSize], out);
    }



// Same as python's numpy.fft.rfftfreq
// @ n: window length
// @ d (optional) : Sample spacing
// @return: A vector of length (n div 2) + 1 of the sample frequencies
    void rfftfreq(const int n, double *out, const double d)
    {
        const double factor = 1.0 / (d * n);
        #pragma omp parallel for
        for (int i = 0; i < n / 2 + 1; ++i) {
            out[i] = i * factor;
        }
    }


    void fft_convolution(double * signal, const int signalLen,
                         double * kernel, const int kernelLen,
                         double * res, const int threads)
    {
        const size_t realSize = signalLen + kernelLen - 1;
        const size_t complexSize = realSize / 2 + 1;
        complex128_t *z1 = (complex128_t *) fftw_alloc_complex (2 * complexSize);
        complex128_t *z2 = z1 + complexSize;

        rfft(signal, signalLen, z1, realSize, threads);
        rfft(kernel, kernelLen, z2, realSize, threads);

        transform(z1, z1 + complexSize, z2, z1,
                  multiplies<complex128_t>());

        irfft(z1, complexSize, res, realSize, threads);

        fftw_free(z1);
    }



    fftwf_plan init_fftf(const int n,  complex64_t *in, complex64_t *out,
                         const int sign = FFTW_FORWARD,
                         const unsigned flag = FFTW_ESTIMATE,
                         const int threads = 1)
    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftwf_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftwf_plan_with_nthreads(threads);
            }
        }
#endif
        // cout << "Threads: " << threads << "\n";

        fftwf_complex *a = reinterpret_cast<fftwf_complex *>(in);
        fftwf_complex *b = reinterpret_cast<fftwf_complex *>(out);
        return fftwf_plan_dft_1d(n, a, b, sign, flag);
    }

    fftwf_plan init_rfftf(const int n, float *in, complex64_t *out,
                          const unsigned flag = FFTW_ESTIMATE,
                          const int threads = 1)

    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftwf_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftwf_plan_with_nthreads(threads);
            }
        }
#endif
        // cout << "Threads: " << threads << "\n";
        fftwf_complex *b = reinterpret_cast<fftwf_complex *>(out);
        return fftwf_plan_dft_r2c_1d(n, in, b, flag);
    }

    fftwf_plan init_irfftf(const int n, complex64_t *in, float *out,
                           const unsigned flag = FFTW_ESTIMATE,
                           const int threads = 1)
    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftwf_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftwf_plan_with_nthreads(threads);
            }
        }
#endif

        // cout << "Threads: " << threads << "\n";

        fftwf_complex *b = reinterpret_cast<fftwf_complex *>(in);
        return fftwf_plan_dft_c2r_1d(n, b, out, flag);
    }


    fftwf_plan init_irfft_packedf(const int n, const int howmany, complex64_t *in, float *out,
                                  const unsigned flag = FFTW_ESTIMATE,
                                  const int threads = 1)
    {
#ifdef FFTW3PARALLEL
        if (threads > 1) {
            if (!hasBeenInit) {
                if (fftwf_init_threads() == 0)
                    cout << "[fft.cpp:init_rfft] Thread initialisation error\n";
                hasBeenInit = true;
                fftwf_plan_with_nthreads(threads);
            }
        }
#endif
        // cout << "Threads: " << threads << "\n";

        fftwf_complex *b = reinterpret_cast<fftwf_complex *>(in);

        return fftwf_plan_many_dft_c2r(1, &n, howmany,
                                       b, NULL,
                                       1, n / 2 + 1,
                                       out, NULL,
                                       1, n,
                                       flag);
        // return fftwf_plan_dft_c2r_2d(n, n1, b, out, flag);
    }


    fftf_plan_t find_planf(int fftSize, int inSize, fft_type_t type, int threads,
                           vector<fftf_plan_t> &v)
    {
        // const uint flag = FFTW_FLAGS;
        auto it =
        find_if(v.begin(), v.end(), [inSize, fftSize, type](const fftf_plan_t &s) {
            return ((s.inSize == inSize) && (s.fftSize == fftSize) && (s.type == type)
                    && (s.howmany == 1));
        });

        if (it == v.end()) {
            fftf_plan_t plan;
            plan.inSize = inSize;
            plan.fftSize = fftSize;
            plan.type = type;

            if (type == FFT) {
                fftwf_complex *in =
                    (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftSize);
                fftwf_complex *out =
                    (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftSize);

                auto p = init_fftf(fftSize, reinterpret_cast<complex64_t *>(in),
                                   reinterpret_cast<complex64_t *>(out),
                                   FFTW_FORWARD, FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;
            } else if (type == IFFT) {

                fftwf_complex *in =
                    (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftSize);
                fftwf_complex *out =
                    (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftSize);
                auto p = init_fftf(fftSize, reinterpret_cast<complex64_t *>(in),
                                   reinterpret_cast<complex64_t *>(out),
                                   FFTW_BACKWARD, FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else if (type == RFFT) {
                float *in = (float *)fftwf_malloc(sizeof(float) * inSize);
                fftwf_complex *out = (fftwf_complex *)fftwf_malloc(
                                         sizeof(fftwf_complex) * fftSize);
                auto p = init_rfftf(inSize, in, reinterpret_cast<complex64_t *>(out),
                                    FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else if (type == IRFFT) {
                fftwf_complex *in = (fftwf_complex *)fftwf_malloc(
                                        sizeof(fftwf_complex) * inSize);
                float *out = (float *)fftwf_malloc(sizeof(float) * fftSize);
                auto p = init_irfftf(fftSize, reinterpret_cast<complex64_t *>(in), out,
                                     FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else {
                printf("[fft::find_plan]: ERROR Wrong fft type!\n");
                exit(-1);
            }

            v.push_back(plan);
            return plan;
        } else {
            return *it;
        }
    }



    fftf_plan_t find_plan_packedf(int fftSize, int howmany, int inSize, fft_type_t type, int threads,
                                  vector<fftf_plan_t> &v)
    {
        // const uint flag = FFTW_FLAGS;
        auto it =
        find_if(v.begin(), v.end(), [inSize, fftSize, howmany, type](const fftf_plan_t &s) {
            return ((s.inSize == inSize) && (s.fftSize == fftSize) && (s.type == type)
                    && (s.howmany == howmany));
        });

        if (it == v.end()) {
            fftf_plan_t plan;
            plan.inSize = inSize;
            plan.fftSize = fftSize;
            plan.howmany = howmany;
            plan.type = type;

            if (type == IRFFT) {
                fftwf_complex *in = (fftwf_complex *)fftwf_malloc(
                                        sizeof(fftwf_complex) * inSize * howmany);
                float *out = (float *)fftwf_malloc(sizeof(float) * fftSize * howmany);

                auto p = init_irfft_packedf(fftSize, howmany, reinterpret_cast<complex64_t *>(in), out,
                                            FFTW_FLAGS, threads);
                plan.p = p;
                plan.in = in;
                plan.out = out;

            } else {
                printf("[fft::find_plan]: ERROR Wrong fft type!\n");
                exit(-1);
            }

            v.push_back(plan);
            return plan;
        } else {
            return *it;
        }
    }



    // rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // @n: Number of points in the input to use. If more than inSize, pad zeros.
    //     if less crop.
    void rfftf(float *in, const int inSize,
               complex64_t *out, int n,
               const int threads)
    {
        n = n == 0 ? n = inSize : n;
        const int outSize = n / 2 + 1;

        auto plan = find_planf(outSize, n, RFFT, threads, planVf);
        auto from = (float *)plan.in;
        auto to = (complex64_t *)plan.out;

        if (n <= inSize)
            copy(in, in + n, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + n, 0.0);
        }

        fftwf_execute(plan.p);
        copy(to, to + outSize, out);
    }

    // Inverse of rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // Missing n: size of output
    void irfftf(complex64_t *in, const int inSize,
                float *out, int outSize,
                const int threads)
    {
        outSize = outSize == 0 ? outSize = 2 * (inSize - 1) : outSize;
        const int n = outSize / 2 + 1;

        auto plan = find_planf(outSize, n, IRFFT, threads, planVf);
        auto from = (complex64_t *)plan.in;
        auto to = (float *)plan.out;

        if (n <= inSize)
            copy(in, in + n, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + n, 0.0);
        }
        // for(int i =0; i < n; i++)
        //     cout << from[i] << "\t";
        // cout << "\n";

        fftwf_execute(plan.p);

        transform(to, to + outSize, out,
                  bind2nd(divides<float>(), outSize));
    }

    // Inverse of rfft
    // @in: input vector which must be the result of a rfft
    // @out: irfft of input, always real
    // n: size of output
    // howmnay: how many ffts of size n0 to perform
    void irfft_packedf(complex64_t *in, const int n0, const int howmany,
                       float *out, int outSize,
                       const int threads)
    {
        outSize = outSize == 0 ? outSize = 2 * (n0 - 1) : outSize;

        const int n = outSize / 2 + 1;

        auto plan = find_plan_packedf(outSize, howmany, n, IRFFT, threads, planVf);

        auto from = (complex64_t *) plan.in;
        auto to = (float *) plan.out;


        if (n <= n0)
            copy(in, in + howmany * n, (complex64_t *) from);
        else {
            copy(in, in + howmany * n0, (complex64_t *) from);
            fill((complex64_t *) from + howmany * n0, (complex64_t *) from + howmany * n, 0.0);
        }
        fftwf_execute(plan.p);

        transform(to, to + howmany * outSize, out,
                  bind2nd(divides<float>(), outSize));


    }


    // Parameters are like python's numpy.fft.ifft
    // @in:  input data
    // @n:   number of points to use. If n < in.size() then the input is cropped
    //       if n > in.size() then input is padded with zeros
    // @out: the inverse Fourier transform of input data
    void ifftf(complex64_t *in, const int inSize,
               complex64_t *out, int fftSize,
               const int threads)
    {
        if (fftSize == 0) fftSize = inSize;

        auto plan = find_planf(fftSize, inSize, IFFT, threads, planVf);
        auto from = (complex64_t *)plan.in;
        auto to = (complex64_t *)plan.out;
        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }

        fftwf_execute(plan.p);

        transform(&to[0], &to[fftSize], out,
                  bind2nd(divides<complex64_t>(), fftSize));
    }


// Parameters are like python's numpy.fft.fft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the transformed array
    void fftf(complex64_t *in, const int inSize,
              complex64_t *out, int fftSize,
              const int threads)
    {
        if (fftSize == 0) fftSize = inSize;

        auto plan = find_planf(fftSize, inSize, FFT, threads, planVf);
        auto from = (complex64_t *)plan.in;
        auto to = (complex64_t *)plan.out;

        if (fftSize <= inSize)
            copy(in, in + fftSize, from);
        else {
            copy(in, in + inSize, from);
            fill(from + inSize, from + fftSize, 0.0);
        }
        fftwf_execute(plan.p);

        copy(&to[0], &to[fftSize], out);
    }



// Same as python's numpy.fft.rfftfreq
// @ n: window length
// @ d (optional) : Sample spacing
// @return: A vector of length (n div 2) + 1 of the sample frequencies
    void rfftfreqf(const int n, float *out, const float d)
    {
        const float factor = 1.0 / (d * n);
        #pragma omp parallel for
        for (int i = 0; i < n / 2 + 1; ++i) {
            out[i] = i * factor;
        }
    }


    void fft_convolutionf(float * signal, const int signalLen,
                          float * kernel, const int kernelLen,
                          float * res, const int threads)
    {
        const size_t realSize = signalLen + kernelLen - 1;
        const size_t complexSize = realSize / 2 + 1;
        complex64_t *z1 = (complex64_t *) fftwf_alloc_complex (2 * complexSize);
        complex64_t *z2 = z1 + complexSize;

        rfftf(signal, signalLen, z1, realSize, threads);
        rfftf(kernel, kernelLen, z2, realSize, threads);

        transform(z1, z1 + complexSize, z2, z1,
                  multiplies<complex64_t>());

        irfftf( z1, complexSize, res, realSize, threads);

        fftwf_free(z1);
    }


}

#else
// empty file
#endif
