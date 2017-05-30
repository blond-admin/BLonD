/*
 * Copyright 2016 CERN. This software is distributed under the
 * terms of the GNU General Public Licence version 3 (GPL Version 3), 
 * copied verbatim in the file LICENCE.md.
 * In applying this licence, CERN does not waive the privileges and immunities 
 * granted to it by virtue of its status as an Intergovernmental Organization or 
 * submit itself to any jurisdiction.
 * Project website: http://blond.web.cern.ch/
 * */


//Author:  Simon Albright, Konstantinos Iliakis

#include <stdlib.h>
// #include <stdio.h>
#include <math.h>
// #include <iostream>



extern "C" void fast_resonator_real_imag(double *__restrict__ impedanceReal,
        double *__restrict__ impedanceImag,
        const double *__restrict__ frequencies,
        const double *__restrict__ shunt_impedances,
        const double *__restrict__ Q_values,
        const double *__restrict__ resonant_frequencies,
        const int n_resonators,
        const int n_frequencies)
{


    for (int res = 0; res < n_resonators; res++) {
        const double Qsquare = Q_values[res] * Q_values[res];
        #pragma omp parallel for
        for (int freq = 1; freq < n_frequencies; freq++) {
            const double commonTerm = (frequencies[freq]
                                       / resonant_frequencies[res]
                                       - resonant_frequencies[res]
                                       / frequencies[freq]);

            impedanceReal[freq] += shunt_impedances[res]
                                   / (1.0 + Qsquare * commonTerm * commonTerm);

            impedanceImag[freq] -= shunt_impedances[res]
                                   * (Q_values[res] * commonTerm)
                                   / (1.0 + Qsquare * commonTerm * commonTerm);
        }
    }

}
