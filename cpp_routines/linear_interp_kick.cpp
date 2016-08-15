/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/


// Optimised C++ routine that calculates the kick of a voltage array on particles
// Authors: Juan F. Esteban Mueller, Alexandre Lasheen

extern "C" void linear_interp_kick(
		double * __restrict__ beam_dt,
		double * __restrict__ beam_dE,
		double * __restrict__ voltage_array,
		double * __restrict__ bin_centers,
        const int n_slices,
		const int n_macroparticles){

	
	const double inv_bin_width = (n_slices-1) / (bin_centers[n_slices-1] - bin_centers[0]);
    
    #pragma omp parallel for
    for (int i = 0; i < n_macroparticles; i++) {
        double a;
        double voltageKick;
        int ffbin; 
    	a = beam_dt[i];
    	ffbin = (int)((a - bin_centers[0]) * inv_bin_width);
    	if ((a < bin_centers[0])||(a > bin_centers[n_slices-1]))
    		voltageKick = 0.;
    	else
    		voltageKick = voltage_array[ffbin] + (a - bin_centers[ffbin]) * (voltage_array[ffbin+1]-voltage_array[ffbin]) * inv_bin_width;
    	beam_dE[i] = beam_dE[i] + voltageKick;
    }

}
