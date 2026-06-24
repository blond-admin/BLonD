// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENSE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

// Optimised C++ routines for the MuSiC algorithm.
// Author: Danilo Quartullo, Konstantinos Iliakis

#include <cmath>

#include "blond_common.h"
#include "openmp.h"

// Sorting of the macro-particles by dt is done in the BLonD3 `Music` element
// (so that all per-particle arrays stay consistent), hence these kernels
// assume the incoming beam_dt/beam_dE are already sorted ascending by dt.

extern "C" void music_track(real_t *__restrict__ beam_dt,
                            real_t *__restrict__ beam_dE,
                            real_t *__restrict__ induced_voltage,
                            real_t *__restrict__ array_parameters,
                            const int n_macroparticles, const real_t alpha,
                            const real_t omega_bar, const real_t cnst,
                            const real_t coeff1, const real_t coeff2,
                            const real_t coeff3, const real_t coeff4,
                            const int multiturn) {
  /*
  This function calculates the induced voltage of one resonator and updates
  the energies of the particles.

  Parameters
  ----------
  beam_dt : float array
      Longitudinal coordinates [s], sorted ascending.
  beam_dE : float array
      Initial energies [V], updated in place.
  induced_voltage : float array
      array used to store the output of the computation
  array_parameters : float array
      [input_first, input_second, t_rev, last_dt]; see music.py
  n_macroparticles : int
      number of macro-particles
  alpha, omega_bar, cnst, coeff1, coeff2, coeff3, coeff4 : floats
      See documentation in music.py
  multiturn : int
      0 for the first turn (recurrence starts fresh), non-zero to bridge
      the wake from the previous turn across the revolution gap.

  Returns
  -------
  induced_voltage : float array
      Computed induced voltage.
  beam_dE : float array
      Array of energies updated.

  Note
  ----
  The caller is responsible for sorting the macro-particles by ``beam_dt``
  (ascending). Sorting is done in the BLonD3 element so that all per-particle
  arrays (ids, flags, ...) stay consistent, hence it is not repeated here.
  */

  real_t product_first_component;
  real_t product_second_component;
  if (multiturn) {
    // Bridge the wake coming from the previous turn.
    const real_t time_difference_0 =
        beam_dt[0] + array_parameters[2] - array_parameters[3];
    const real_t exp_term = FAST_EXP(-alpha * time_difference_0);
    const real_t cos_term = FAST_COS(omega_bar * time_difference_0);
    const real_t sin_term = FAST_SIN(omega_bar * time_difference_0);
    product_first_component =
        exp_term * ((cos_term + coeff1 * sin_term) * array_parameters[0] +
                    coeff2 * sin_term * array_parameters[1]);
    product_second_component =
        exp_term * (coeff3 * sin_term * array_parameters[0] +
                    (cos_term + coeff4 * sin_term) * array_parameters[1]);
  } else {
    // Turn 1: no previous-turn wake to bridge.
    product_first_component = 0;
    product_second_component = 0;
  }

  induced_voltage[0] = cnst * (0.5 + product_first_component);
  beam_dE[0] += induced_voltage[0];
  real_t input_first_component = product_first_component + 1;
  real_t input_second_component = product_second_component;

  for (int i = 0; i < n_macroparticles - 1; i++) {
    const real_t time_difference = beam_dt[i + 1] - beam_dt[i];
    const real_t exp_term = FAST_EXP(-alpha * time_difference);
    const real_t cos_term = FAST_COS(omega_bar * time_difference);
    const real_t sin_term = FAST_SIN(omega_bar * time_difference);

    const real_t next_first_component =
        exp_term * ((cos_term + coeff1 * sin_term) * input_first_component +
                    coeff2 * sin_term * input_second_component);

    const real_t next_second_component =
        exp_term * (coeff3 * sin_term * input_first_component +
                    (cos_term + coeff4 * sin_term) * input_second_component);

    induced_voltage[i + 1] = cnst * (0.5 + next_first_component);
    beam_dE[i + 1] += induced_voltage[i + 1];
    input_first_component = next_first_component + 1;
    input_second_component = next_second_component;
  }

  array_parameters[0] = input_first_component;
  array_parameters[1] = input_second_component;
  array_parameters[3] = beam_dt[n_macroparticles - 1];
}
