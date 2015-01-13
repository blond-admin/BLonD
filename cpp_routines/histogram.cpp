/*
Copyright 2015 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the histogram
// Author: Danilo Quartullo

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
