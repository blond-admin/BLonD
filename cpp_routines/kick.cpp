#include "sin.h"
#include <iostream>
#include <cmath>
#include <valarray>
#include <stdio.h>
using namespace std;
using namespace vdt;
using uint = unsigned int;

extern "C" void kick(const double * __restrict__ beam_theta, double * __restrict__ beam_dE, const int n_rf, 
						const double * __restrict__ voltage, const double * __restrict__ harmonic, 
						const double * __restrict__ phi_offset, const uint n_macroparticles){
  
 const double *fbeam_theta = (const double*) __builtin_assume_aligned(beam_theta, 64);
 double *fbeam_dE = (double*) __builtin_assume_aligned(beam_dE, 64);
  
 for (int j = 0; j < n_rf; j++) 
 	for (uint i = 0; i < n_macroparticles; i++) 
        fbeam_dE[i] = fbeam_dE[i] + voltage[j] * fast_sin(harmonic[j] * fbeam_theta[i] + phi_offset[j]);

}