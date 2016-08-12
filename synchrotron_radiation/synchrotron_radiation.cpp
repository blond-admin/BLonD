/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates and applies synchrotron radiation (SR)
// damping term
// Author: Juan F. Esteban Mueller

#include <stdlib.h>

// This function calculates and applies only the synchrotron radiation damping term
extern "C" void synchrotron_radiation(double * __restrict__ beam_dE, const double U0, 
                            const int n_macroparticles, const double tau_z, 
                            const int n_kicks){

    // SR damping constant
    const double const_synch_rad = 2.0 / tau_z;

    for (int j=0; j<n_kicks; j++){
        // SR damping term due to energy spread
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] -= const_synch_rad * beam_dE[i];
    
        // Average energy change due to SR
        #pragma omp parallel for
        for (int i = 0; i < n_macroparticles; i++)
            beam_dE[i] -= U0;
    }
}
