/*
Copyright 2015 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the kick of a voltage array on particles
// Author: Juan F. Esteban Muller, Alexandre Lasheen


using uint = unsigned int;

extern "C" void linear_interp_kick(
		double * __restrict__ beam_dt,
		double * __restrict__ beam_dE,
		double * __restrict__ voltage_array,
		double * __restrict__ bin_centers,
        const uint n_slices,
		const uint n_macroparticles){

	double a;
	uint i;
	double fbin;
	uint ffbin;
	double inducedVoltageKick;
	const double inv_bin_width = (n_slices-1) / (bin_centers[n_slices-1] - bin_centers[0]);

	for (i = 0; i < n_macroparticles; i++) {
		a = beam_dt[i];
		fbin = (a - bin_centers[0]) * inv_bin_width;
		ffbin = (uint)(fbin);
		if ((a < bin_centers[0])||(a > bin_centers[n_slices-1]))
			inducedVoltageKick = 0.;
		else
			inducedVoltageKick = voltage_array[ffbin] + (a - bin_centers[ffbin]) * (voltage_array[ffbin+1]-voltage_array[ffbin]) * inv_bin_width;
		beam_dE[i] = beam_dE[i] + inducedVoltageKick;
	}

}
