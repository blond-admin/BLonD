/*
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routine that calculates MuSiC trakc method
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

using namespace vdt;

struct particle {
    double de;
    double dt;
    bool operator<(const particle &o) const
    {
        return dt < o.dt;
    }
};

struct Comparator {
    const double *dt;
    Comparator(const double *_dt) : dt(_dt) {}
    bool operator()(const int i1, const int i2) const
    {
        return dt[i1] < dt[i2];
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
    // NOTE make sure that this is actually sorting beam_dE with regards to
    // beam_dt
// #ifdef PARALLEL
//     // std::cout << "parallel code\n";
//     __gnu_parallel::sort(&beam_dE[0], &beam_dE[n_macroparticles],
//                          Comparator(beam_dt));
//     __gnu_parallel::sort(&beam_dt[0], &beam_dt[n_macroparticles]);
// #else
//     // std::cout << "serial code\n";
//     std::sort(&beam_dE[0], &beam_dE[n_macroparticles],
//               Comparator(beam_dt));
//     std::sort(&beam_dt[0], &beam_dt[n_macroparticles]);
// #endif


//     duration = std::chrono::system_clock::now() - start;
//     std::cout << "sorting time: " << duration.count() << '\n';

//     start = std::chrono::system_clock::now();

    beam_dE[0] += induced_voltage[0];
    double input_first_component = 1;
    double input_second_component = 0;
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



//     duration = std::chrono::system_clock::now() - start;
//     std::cout << "tracking time: " << duration.count() << '\n';


}

