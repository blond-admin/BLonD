using uint = unsigned int;


extern "C" void histogram(const double * __restrict__ input,
			   double * __restrict__ output,
               const double cut_left,
               const double cut_right,
               const uint n_slices,
               const uint n_macroparticles){
	
   uint i;
   double a;
   double fbin;
   uint ffbin;
   const double inv_bin_width = n_slices / (cut_right - cut_left);
   
   for (i = 0; i < n_slices; i++) {
   	  output[i] = 0.0;
   }	
  
   for (i = 0; i < n_macroparticles; i++) {
      a = input[i];
      if ((a < cut_left)||(a > cut_right))
        	continue;
      fbin = (a - cut_left) * inv_bin_width;
      ffbin = (uint)(fbin);
      output[ffbin] = output[ffbin] + 1.0;
   }
}


