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


#include "sin.h"
#include "cos.h"
#include "exp.h"

#include "openmp.h"

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


extern "C" void music_track(double *__restrict__ beam_dt,
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
    std::vector<particle<double>> particles; particles.reserve(n_macroparticles);
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

    array_parameters[0] = input_first_component;
    array_parameters[1] = input_second_component;
    array_parameters[3] = beam_dt[n_macroparticles - 1];

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
{   /*
    This function calculates the multi-turn induced voltage and updates the
    energies of the particles.
    Parameters and Returns as for music_track.
    */


    // Particle sorting with respect to dt
    std::vector<particle<double>> particles; particles.reserve(n_macroparticles);
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

    // MuSiC algorithm for the current turn
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



extern "C" void music_trackf(float *__restrict__ beam_dt,
                             float *__restrict__ beam_dE,
                             float *__restrict__ induced_voltage,
                             float *__restrict__ array_parameters,
                             const int n_macroparticles,
                             const float alpha,
                             const float omega_bar,
                             const float cnst,
                             const float coeff1,
                             const float coeff2,
                             const float coeff3,
                             const float coeff4)
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
    std::vector<particle<float>> particles; particles.reserve(n_macroparticles);
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
    float input_first_component = 1;
    float input_second_component = 0;
    for (int i = 0; i < n_macroparticles - 1; i++) {
        const float time_difference = beam_dt[i + 1] - beam_dt[i];
        const float exp_term = fast_exp(-alpha * time_difference);
        const float cos_term = fast_cos(omega_bar * time_difference);
        const float sin_term = fast_sin(omega_bar * time_difference);

        const float product_first_component =
            exp_term * ((cos_term + coeff1 * sin_term)
                        * input_first_component + coeff2 * sin_term
                        * input_second_component);

        const float product_second_component =
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


extern "C" void music_track_multiturnf(float *__restrict__ beam_dt,
                                       float *__restrict__ beam_dE,
                                       float *__restrict__ induced_voltage,
                                       float *__restrict__ array_parameters,
                                       const int n_macroparticles,
                                       const float alpha,
                                       const float omega_bar,
                                       const float cnst,
                                       const float coeff1,
                                       const float coeff2,
                                       const float coeff3,
                                       const float coeff4)
{   /*
    This function calculates the multi-turn induced voltage and updates the
    energies of the particles.
    Parameters and Returns as for music_track.
    */


    // Particle sorting with respect to dt
    std::vector<particle<float>> particles; particles.reserve(n_macroparticles);
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
    const float time_difference_0 = beam_dt[0] + array_parameters[2] - array_parameters[3];
    const float exp_term = fast_exp(-alpha * time_difference_0);
    const float cos_term = fast_cos(omega_bar * time_difference_0);
    const float sin_term = fast_sin(omega_bar * time_difference_0);

    const float product_first_component =
        exp_term * ((cos_term + coeff1 * sin_term)
                    * array_parameters[0] + coeff2 * sin_term
                    * array_parameters[1]);

    const float product_second_component =
        exp_term * (coeff3 * sin_term * array_parameters[0]
                    + (cos_term + coeff4 * sin_term)
                    * array_parameters[1]);

    induced_voltage[0] = cnst * (0.5 + product_first_component);
    beam_dE[0] += induced_voltage[0];
    float input_first_component = product_first_component + 1;
    float input_second_component = product_second_component;

    // MuSiC algorithm for the current turn
    for (int i = 0; i < n_macroparticles - 1; i++) {
        const float time_difference = beam_dt[i + 1] - beam_dt[i];
        const float exp_term = fast_exp(-alpha * time_difference);
        const float cos_term = fast_cos(omega_bar * time_difference);
        const float sin_term = fast_sin(omega_bar * time_difference);

        const float product_first_component =
            exp_term * ((cos_term + coeff1 * sin_term)
                        * input_first_component + coeff2 * sin_term
                        * input_second_component);

        const float product_second_component =
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



