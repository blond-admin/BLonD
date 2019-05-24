/*
 * fft.h
 *
 *  Created on: Mar 21, 2016
 *      Author: kiliakis
 */

#ifndef INCLUDE_FFT_H_
#define INCLUDE_FFT_H_

#include <complex>
#include <fftw3.h>


typedef std::complex<double> complex_t;
enum fft_type_t { FFT, IFFT, RFFT, IRFFT };

struct fft_plan_t {
  fftw_plan p;      // fftw_plan
  int inSize;       // input size
  int fftSize;      // fft size
  int howmany = 1;  // for packed ffts
  fft_type_t type;
  void *in;
  void *out;
};




extern "C" {
  void destroy_plans();

  void rfft(double *in, const int inSize,
            complex_t *out, int fftSize = 0,
            const int threads = 1);

// Parameters are like python's numpy.fft.fft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the transformed array
  void fft(complex_t *in, const int inSize,
           complex_t *out, int fftSize = 0,
           const int threads = 1);


// Parameters are like python's numpy.fft.ifft
// @in:  input data
// @n:   number of points to use. If n < in.size() then the input is cropped
//       if n > in.size() then input is padded with zeros
// @out: the inverse Fourier transform of input data
  void ifft(complex_t *in, const int inSize,
            complex_t *out, int fftSize = 0,
            const int threads = 1);


// Inverse of rfft
// @in: input vector which must be the result of a rfft
// @out: irfft of input, always real
// Missing n: size of output
  void irfft(complex_t *in, const int inSize,
             double *out, int fftSize = 0,
             const int threads = 1);

  void irfft_packed(complex_t *in, const int n0, const int howmany,
                    double *out, int fftSize = 0,
                    const int threads = 1);


// Same as python's numpy.fft.rfftfreq
// @ n: window length
// @ d (optional) : Sample spacing
// @return: A vector of length (n div 2) + 1 of the sample frequencies
  void rfftfreq(const int n, double *out, const double d = 1.0);



  void fft_convolution(double * signal, const int signalLen,
                       double * kernel, const int kernelLen,
                       double * res, const int threads = 1);
}


#endif /* INCLUDE_FFT_H_ */
