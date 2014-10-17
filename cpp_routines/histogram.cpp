#include <cstring>
using uint = unsigned int;


extern "C" void histogram(const double * __restrict__ finput,
			   uint * __restrict__ foutput,
               const double cut_left,
               const double cut_right,
               const uint n_slices,
               const uint n_macroparticles){
	
   const double *input = (const double*) __builtin_assume_aligned(finput, 16);
   uint *output = (uint*) __builtin_assume_aligned(foutput, 16);
   
   uint i;
   
   for (i = 0; i < n_slices; i++) {
   	  output[i] = 0;
   }	
  
   const double inv_bin_width = n_slices / (cut_right - cut_left);
   uint ffbin;
   
   for (i = 0; i < n_macroparticles; i++) {
      const double a = input[i];
      if ((a < cut_left)||(a > cut_right))
        	continue;
      const double fbin = (a - cut_left) * inv_bin_width;
      ffbin = (uint)(fbin);
      output[ffbin] = output[ffbin] + 1;
   }
}


