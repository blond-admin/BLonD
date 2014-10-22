#include "sin.h"
using namespace vdt;
using uint = unsigned int;


extern "C" void kick(const double * __restrict__ beam_theta, double * __restrict__ beam_dE, const int n_rf, 
						const double * __restrict__ voltage, const double * __restrict__ harmonic, 
						const double * __restrict__ phi_offset, const uint n_macroparticles){
  
for (int j = 0; j < n_rf; j++) 
 	for (uint i = 0; i < n_macroparticles; i++) 
            beam_dE[i] = beam_dE[i] + voltage[j] * fast_sin(harmonic[j] * beam_theta[i] + phi_offset[j]);

}
