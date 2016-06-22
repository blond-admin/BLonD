/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or 
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/





// Optimised C++ routine that calculates the drift.
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

#include <string.h>

extern "C" void drift(double * __restrict__ beam_dt, 
                      const double * __restrict__ beam_dE, 
                      const char * __restrict__ solver,
	                  const double T0, const double length_ratio, 
	                  const double alpha_order, const double eta_zero, 
	                  const double eta_one, const double eta_two,
                      const double beta, const double energy, 
                      const int n_macroparticles){

int i;
double T = T0*length_ratio;

if ( strcmp (solver,"simple") == 0 )
	{
		double coeff = eta_zero/(beta*beta*energy);
        #pragma omp parallel for
		for (int i = 0; i < n_macroparticles; i++)
			beam_dt[i] += T*coeff*beam_dE[i];
	}

else
	{
		const double coeff = 1./(beta*beta*energy);
		const double eta0 = eta_zero*coeff;
		const double eta1 = eta_one*coeff*coeff;
		const double eta2 = eta_two*coeff*coeff*coeff;

		if ( alpha_order == 1 )
    		for ( i = 0; i < n_macroparticles; i++ )
        		beam_dt[i] += T*(1./(1. - eta0*beam_dE[i]) - 1.);
    	else if (alpha_order == 2)
        	for ( i = 0; i < n_macroparticles; i++ )
        		beam_dt[i] += T*(1./(1. - eta0*beam_dE[i]
        		           - eta1*beam_dE[i]*beam_dE[i]) - 1.);
    	else
        	for ( i = 0; i < n_macroparticles; i++ )
        		beam_dt[i] += T*(1./(1. - eta0*beam_dE[i]
        		           - eta1*beam_dE[i]*beam_dE[i]
        		           - eta2*beam_dE[i]*beam_dE[i]*beam_dE[i]) - 1.);
	}

}

