/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates the kicks
// Author: Danilo Quartullo, Helga Timko, Alexandre Lasheen

// #include "sin.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>

// using namespace vdt;
using namespace std;

struct particle {
    double de;
    double dt;
    bool operator<(const particle &o) const
    {
        return dt < o.dt;
    }
};


extern "C" void music_track(double *__restrict__ beam_dt,
                            double *__restrict__ beam_dE,
                            double *__restrict__ induced_voltage,
                            const int n_macroparticles,
                            const double alpha,
                            const double omega_bar,
                            const double cnst,
                            const double coeff1,
                            const double coeff2,
                            const double coeff3,
                            const double coeff4)
{
    clock_t start;
    double duration;

    start = std::clock();

    

//     sort(&beam_dt[0], &beam_dt[n_macroparticles]);
    vector<particle> particles; particles.reserve(n_macroparticles);
    for (int i = 0; i < n_macroparticles; i++)
        particles.push_back({beam_dE[i], beam_dt[i]});
    sort(particles.begin(), particles.end());
    for (int i = 0; i < n_macroparticles; i++) {
        beam_dE[i] = particles[i].de;
        beam_dt[i] = particles[i].dt;
    }
    
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;

    cout<<"printf: "<< duration <<'\n';
    beam_dE[0] += induced_voltage[0];
    double input_first_component = 1;
    double input_second_component = 0;

    for (int i = 0; i < n_macroparticles - 1; i++) {
        const double time_difference = beam_dt[i + 1] - beam_dt[i];
        const double exp_term = exp(-alpha * time_difference);
        const double cos_term = cos(omega_bar * time_difference);
        const double sin_term = sin(omega_bar * time_difference);
        const double product_first_component =
            exp_term * ((cos_term + coeff1 * sin_term)
                        * input_first_component + coeff2 * sin_term
                        * input_second_component);

        const double product_second_component =
            exp_term * (coeff3 * sin_term * input_first_component
                        + (cos_term + coeff4 * sin_term)
                        * input_second_component);
        induced_voltage[i + 1] = cnst * (0.5 + product_first_component);
        beam_dE[i + 1] += induced_voltage[i + 1];
        input_first_component = product_first_component + 1;
        input_second_component = product_second_component;
    }
    
    



}

