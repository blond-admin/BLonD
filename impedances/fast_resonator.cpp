/*
 * Copyright 2016 CERN. This software is distributed under the
 * terms of the GNU General Public Licence version 3 (GPL Version 3), 
 * copied verbatim in the file LICENCE.md.
 * In applying this licence, CERN does not waive the privileges and immunities 
 * granted to it by virtue of its status as an Intergovernmental Organization or 
 * submit itself to any jurisdiction.
 * Project website: http://blond.web.cern.ch/
 * */


//Author:  Simon Albright

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern "C" void fast_resonator_real(double * __restrict__ impedanceReal,
					const double * __restrict__ frequencies,
					const double * __restrict__ shunt_impedances,
					const double * __restrict__ Q_values,
					const double * __restrict__ resonant_frequencies,
					const int n_resonators,
					const int n_frequencies)

{

int res, freq;

	#pragma omp parallel for
	for (res = 0; res < n_resonators; res++)
		#pragma omp parallel for
		for (freq = 1; freq < n_frequencies; freq++)

			impedanceReal[freq] += shunt_impedances[res] / (1.0 + pow(Q_values[res], 2)*pow((frequencies[freq]/resonant_frequencies[res] - resonant_frequencies[res]/frequencies[freq]), 2));

}


extern "C" void fast_resonator_imag(double * __restrict__ impedanceImag,
					const double * __restrict__ frequencies,
					const double * __restrict__ shunt_impedances,
					const double * __restrict__ Q_values,
					const double * __restrict__ resonant_frequencies,
					const int n_resonators,
					const int n_frequencies)

{

int res, freq;

		#pragma omp parallel for
		for (res = 0; res < n_resonators; res++)
			#pragma omp parallel for
			for (freq = 1; freq < n_frequencies; freq++)

				impedanceImag[freq] -= shunt_impedances[res] * (Q_values[res] * (frequencies[freq]/resonant_frequencies[res] - resonant_frequencies[res]/frequencies[freq]))
				 / (1.0 + pow(Q_values[res], 2)*pow((frequencies[freq]/resonant_frequencies[res] - resonant_frequencies[res]/frequencies[freq]), 2));


}

