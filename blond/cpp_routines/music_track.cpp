/*
Copyright 2014-2017 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/
*/

// Optimised C++ routines for the MuSiC algorithm.
// Author: Danilo Quartullo, Konstantinos Iliakis


#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

#ifdef PARALLEL
#include <parallel/algorithm>
#else
#include <algorithm>
#endif

#include "blond_common.h"
#include "openmp.h"


// using namespace vdt;


// Definition of struct particle
template <typename T>
struct particle {
    T de;
    T dt;
    bool operator<(const particle &o) const
    {
        return dt < o.dt;
    }
};


extern "C" void music_track(real_t *__restrict__ beam_dt,
                            real_t *__restrict__ beam_dE,
                            real_t *__restrict__ induced_voltage,
                            real_t *__restrict__ array_parameters,
                            const int n_macroparticles,
                            const real_t alpha,
                            const real_t omega_bar,
                            const real_t cnst,
                            const real_t coeff1,
                            const real_t coeff2,
                            const real_t coeff3,
                            const real_t coeff4)
{
    /*
    This function calculates the single-turn induced voltage and updates the
    energies of the particles.

    Parameters
    ----------
    beam_dt : float array
        Longitudinal coordinates [s]
    beam_dE : float array
        Initial energies [V]
    induced_voltage : float array
        array used to store the output of the computation
    array_parameters : float array
        See documentation in music.py
    n_macroparticles : int
        number of macro-particles
    alpha, omega_bar, cnst, coeff1, coeff2, coeff3, coeff4 : floats
        See documentation in music.py

    Returns
    -------
    induced_voltage : float array
        Computed induced voltage.
    beam_dE : float array
        Array of energies updated.
    */


    // Particle sorting with respect to dt
    std::vector<particle<real_t>> particles; particles.reserve(n_macroparticles);
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

    // MuSiC algorithm
    beam_dE[0] += induced_voltage[0];
    real_t input_first_component = 1;
    real_t input_second_component = 0;
    for (int i = 0; i < n_macroparticles - 1; i++) {
        const real_t time_difference = beam_dt[i + 1] - beam_dt[i];
        const real_t exp_term = FAST_EXP(-alpha * time_difference);
        const real_t cos_term = FAST_COS(omega_bar * time_difference);
        const real_t sin_term = FAST_SIN(omega_bar * time_difference);

        const real_t product_first_component =
            exp_term * ((cos_term + coeff1 * sin_term)
                        * input_first_component + coeff2 * sin_term
                        * input_second_component);

        const real_t product_second_component =
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


extern "C" void music_track_multiturn(real_t *__restrict__ beam_dt,
                                      real_t *__restrict__ beam_dE,
                                      real_t *__restrict__ induced_voltage,
                                      real_t *__restrict__ array_parameters,
                                      const int n_macroparticles,
                                      const real_t alpha,
                                      const real_t omega_bar,
                                      const real_t cnst,
                                      const real_t coeff1,
                                      const real_t coeff2,
                                      const real_t coeff3,
                                      const real_t coeff4)
{   /*
    This function calculates the multi-turn induced voltage and updates the
    energies of the particles.
    Parameters and Returns as for music_track.
    */


    // Particle sorting with respect to dt
    std::vector<particle<real_t>> particles; particles.reserve(n_macroparticles);
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

    // First computation of MuSiC relative to the voltage coming from the
    // previous turn
    const real_t time_difference_0 = beam_dt[0] + array_parameters[2] - array_parameters[3];
    const real_t exp_term = FAST_EXP(-alpha * time_difference_0);
    const real_t cos_term = FAST_COS(omega_bar * time_difference_0);
    const real_t sin_term = FAST_SIN(omega_bar * time_difference_0);

    const real_t product_first_component =
        exp_term * ((cos_term + coeff1 * sin_term)
                    * array_parameters[0] + coeff2 * sin_term
                    * array_parameters[1]);

    const real_t product_second_component =
        exp_term * (coeff3 * sin_term * array_parameters[0]
                    + (cos_term + coeff4 * sin_term)
                    * array_parameters[1]);

    induced_voltage[0] = cnst * (0.5 + product_first_component);
    beam_dE[0] += induced_voltage[0];
    real_t input_first_component = product_first_component + 1;
    real_t input_second_component = product_second_component;

    // MuSiC algorithm for the current turn
    for (int i = 0; i < n_macroparticles - 1; i++) {
        const real_t time_difference = beam_dt[i + 1] - beam_dt[i];
        const real_t exp_term = FAST_EXP(-alpha * time_difference);
        const real_t cos_term = FAST_COS(omega_bar * time_difference);
        const real_t sin_term = FAST_SIN(omega_bar * time_difference);

        const real_t product_first_component =
            exp_term * ((cos_term + coeff1 * sin_term)
                        * input_first_component + coeff2 * sin_term
                        * input_second_component);

        const real_t product_second_component =
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
