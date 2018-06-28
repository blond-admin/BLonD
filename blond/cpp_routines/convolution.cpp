/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Parallelised C++ routine that calculates the linear convolution of two arrays.
// Author: K. Iliakis



extern "C" void convolution(const double *__restrict signal,
                                   const int SignalLen,
                                   const double *__restrict kernel,
                                   const int KernelLen, double *__restrict res)
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

