/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates MuSiC track method
// Author: Danilo Quartullo, Konstantinos Iliakis

#include "sin.h"
#include "cos.h"
#include "exp.h"

#ifdef PARALLEL
#include <parallel/algorithm>
#else
#include <algorithm>
#endif

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

using namespace vdt;

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
                            double *__restrict__ array_parameters,
                            double *__restrict__ input_first_component,
                            double *__restrict__ input_second_component,
                            const int n_macroparticles,
                            const int n_resonators,
                            const double *__restrict__ alpha,
                            const double *__restrict__ omega_bar,
                            const double *__restrict__ cnst,
                            const double *__restrict__ coeff1,
                            const double *__restrict__ coeff2,
                            const double *__restrict__ coeff3,
                            const double *__restrict__ coeff4)
{

//     std::chrono::time_point<std::chrono::high_resolution_clock>start;
//     std::chrono::duration<double> duration(0.0);
//     start = std::chrono::system_clock::now();

    std::vector<particle> particles; particles.reserve(n_macroparticles);
    for (int i = 0; i < n_macroparticles; i++)
        particles.push_back({beam_dE[i], beam_dt[i]});
#ifdef PARALLEL
    __gnu_parallel::sort(particles.begin(), particles.end());
#else
    std::sort(particles.begin(), particles.end());
#endif
    for (int i = 0; i < n_macroparticles; i++) {
        beam_dE[i] = particles[i].de;
        beam_dt[i] = particles[i].dt;
    }

//     duration = std::chrono::system_clock::now() - start;
//     std::cout << "sorting time: " << duration.count() << '\n';

//     start = std::chrono::system_clock::now();

    // beam_dE[0] += induced_voltage[0];
    // double input_first_component = 1;
    // double input_second_component = 0;
    for (int i = 1; i < n_macroparticles; i++)
        induced_voltage[i] = 0;

    #pragma omp parallel for
    for (int j = 0; j < n_resonators; j++) {
        input_first_component[j] = 1.0;
        input_second_component[j] = 0.0;
        for (int i = 0; i < n_macroparticles - 1; i++) {
            const double time_difference = beam_dt[i + 1] - beam_dt[i];
            const double exp_term = fast_exp(-alpha[j] * time_difference);
            const double cos_term = fast_cos(omega_bar[j] * time_difference);
            const double sin_term = fast_sin(omega_bar[j] * time_difference);

            const double product_first_component =
                exp_term * ((cos_term + coeff1[j] * sin_term)
                            * input_first_component[j] + coeff2[j] * sin_term
                            * input_second_component[j]);

            const double product_second_component =
                exp_term * (coeff3[j] * sin_term * input_first_component[j]
                            + (cos_term + coeff4[j] * sin_term)
                            * input_second_component[j]);
            // beam_dE[i + 1] += induced_voltage[i + 1];
            input_first_component[j] = product_first_component + 1;
            input_second_component[j] = product_second_component;

            #pragma omp atomic
            induced_voltage[i + 1] += cnst[j] * (0.5 + product_first_component);

        }
    }
    for (int i = 0; i < n_macroparticles; i++)
        beam_dE[i] += induced_voltage[i];

    array_parameters[1] = beam_dt[n_macroparticles - 1];

//     duration = std::chrono::system_clock::now() - start;
//     std::cout << "tracking time: " << duration.count() << '\n';


}


extern "C" void music_track_multiturn(double *__restrict__ beam_dt,
                                      double *__restrict__ beam_dE,
                                      double *__restrict__ induced_voltage,
                                      double *__restrict__ array_parameters,
                                      const int n_macroparticles,
                                      const double alpha,
                                      const double omega_bar,
                                      const double cnst,
                                      const double coeff1,
                                      const double coeff2,
                                      const double coeff3,
                                      const double coeff4)
{

    std::vector<particle> particles; particles.reserve(n_macroparticles);
    for (int i = 0; i < n_macroparticles; i++)
        particles.push_back({beam_dE[i], beam_dt[i]});
#ifdef PARALLEL
    __gnu_parallel::sort(particles.begin(), particles.end());
#else
    std::sort(particles.begin(), particles.end());
#endif
    for (int i = 0; i < n_macroparticles; i++) {
        beam_dE[i] = particles[i].de;
        beam_dt[i] = particles[i].dt;
    }

    const double time_difference_0 = beam_dt[0] + array_parameters[2] - array_parameters[3];
    const double exp_term = fast_exp(-alpha * time_difference_0);
    const double cos_term = fast_cos(omega_bar * time_difference_0);
    const double sin_term = fast_sin(omega_bar * time_difference_0);

    const double product_first_component =
        exp_term * ((cos_term + coeff1 * sin_term)
                    * array_parameters[0] + coeff2 * sin_term
                    * array_parameters[1]);

    const double product_second_component =
        exp_term * (coeff3 * sin_term * array_parameters[0]
                    + (cos_term + coeff4 * sin_term)
                    * array_parameters[1]);

    induced_voltage[0] = cnst * (0.5 + product_first_component);
    beam_dE[0] += induced_voltage[0];
    double input_first_component = product_first_component + 1;
    double input_second_component = product_second_component;


    for (int i = 0; i < n_macroparticles - 1; i++) {
        const double time_difference = beam_dt[i + 1] - beam_dt[i];
        const double exp_term = fast_exp(-alpha * time_difference);
        const double cos_term = fast_cos(omega_bar * time_difference);
        const double sin_term = fast_sin(omega_bar * time_difference);

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

    array_parameters[0] = input_first_component;
    array_parameters[1] = input_second_component;
    array_parameters[3] = beam_dt[n_macroparticles - 1];
}


