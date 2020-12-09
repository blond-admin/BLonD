/*
 * Copyright 2014-2017 CERN. This software is distributed under the
 * terms of the GNU General Public Licence version 3 (GPL Version 3), 
 * copied verbatim in the file LICENCE.md.
 * In applying this licence, CERN does not waive the privileges and immunities 
 * granted to it by virtue of its status as an Intergovernmental Organization or 
 * submit itself to any jurisdiction.
 * Project website: http://blond.web.cern.ch/
 * */

// Optimised C++ routine that calculates the impedance of a resonator.
// Author:  Simon Albright, Konstantinos Iliakis, Danilo Quartullo

#include <stdlib.h>
#include <math.h>


extern "C" void fast_resonator_real_imag(double *__restrict__ impedanceReal,
        double *__restrict__ impedanceImag,
        const double *__restrict__ frequencies,
        const double *__restrict__ shunt_impedances,
        const double *__restrict__ Q_values,
        const double *__restrict__ resonant_frequencies,
        const int n_resonators,
        const int n_frequencies)
        
{   /*
    This function takes as an input a list of resonators parameters and 
    computes the impedance in an optimised way.
    
    Parameters
    ---------- 
    frequencies : float array
        array of frequency in Hz
    shunt_impedances : float array
        array of shunt impedances in Ohm
    Q_values : float array
        array of quality factors
    resonant_frequencies : float array
        array of resonant frequency in Hz
    n_resonators : int
        number of resonantors
    n_frequencies : int
        length of the array 'frequencies'
    
    Returns
    -------
    impedanceReal : float array
        real part of the impedance
    impedanceImag : float array
        imaginary part of the impedance
      */


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


extern "C" void fast_resonator_real_imagf(float *__restrict__ impedanceReal,
        float *__restrict__ impedanceImag,
        const float *__restrict__ frequencies,
        const float *__restrict__ shunt_impedances,
        const float *__restrict__ Q_values,
        const float *__restrict__ resonant_frequencies,
        const int n_resonators,
        const int n_frequencies)
        
{   /*
    This function takes as an input a list of resonators parameters and 
    computes the impedance in an optimised way.
    
    Parameters
    ---------- 
    frequencies : float array
        array of frequency in Hz
    shunt_impedances : float array
        array of shunt impedances in Ohm
    Q_values : float array
        array of quality factors
    resonant_frequencies : float array
        array of resonant frequency in Hz
    n_resonators : int
        number of resonantors
    n_frequencies : int
        length of the array 'frequencies'
    
    Returns
    -------
    impedanceReal : float array
        real part of the impedance
    impedanceImag : float array
        imaginary part of the impedance
      */


    for (int res = 0; res < n_resonators; res++) {
        const float Qsquare = Q_values[res] * Q_values[res];
        #pragma omp parallel for
        for (int freq = 1; freq < n_frequencies; freq++) {
            const float commonTerm = (frequencies[freq]
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

