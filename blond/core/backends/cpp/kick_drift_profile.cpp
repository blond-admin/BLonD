// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#include "blond_common.h"

#include <math.h>
#include <stdlib.h> // mmalloc()
#include <string.h> // memset()

#include "blond_common.h"
#include "openmp.h"

extern "C" void kick_drift_profile(
    real_t *__restrict__ beam_dt, real_t *__restrict__ beam_dE,
    const real_t charge, const real_t voltage, const real_t omega_RF,
    const real_t phi_RF, const real_t T, const real_t eta_zero,
    const real_t beta, const real_t energy, real_t *__restrict__ output,
    const real_t cut_left, const real_t cut_right, const int n_slices,
    const int n_macroparticles, const real_t acc_kick) {

  const real_t coeff = T * eta_zero / (beta * beta * energy);

  // Number of Iterations of the inner loop
  // STEP=32: 2x AVX-512 ops for the bin computation, better loop amortisation.
  const int STEP = 32;
  const real_t inv_bin_width = n_slices / (cut_right - cut_left);

  // allocate memory for the thread_private histogram
  int **histo = (int **)malloc(omp_get_max_threads() * sizeof(int *));
  histo[0] = (int *)malloc(omp_get_max_threads() * n_slices * sizeof(int));
  for (int i = 0; i < omp_get_max_threads(); i++)
    histo[i] = (*histo + n_slices * i);

#pragma omp parallel
  {
    const int id = omp_get_thread_num();
    const int threads = omp_get_num_threads();
    memset(histo[id], 0, n_slices * sizeof(int));
    // Use int directly to avoid float→int conversion in the scatter loop.
    int fbin[STEP];
#pragma omp for schedule(static)
    for (int i = 0; i < n_macroparticles; i += STEP) {
      const int loop_count =
          n_macroparticles - i > STEP ? STEP : n_macroparticles - i;

      // Phase 1: Apply kick and drift, write back, compute bin indices.
      // This loop is free of histogram scatter and branches on fbin, so the
      // compiler can vectorise the kick+drift+floor block with SIMD.
      for (int j = 0; j < loop_count; j++) {
        real_t dEij = beam_dE[i + j];
        real_t dtij = beam_dt[i + j];
        dEij += charge * voltage * FAST_SIN(omega_RF * dtij + phi_RF) + acc_kick;
        dtij += coeff * dEij;
        beam_dE[i + j] = dEij;
        beam_dt[i + j] = dtij;
        int bin = (int)((dtij - cut_left) * inv_bin_width);
        // Clamp the right edge exactly (check the post-drift value)
        if (dtij == cut_right) bin = n_slices - 1;
        fbin[j] = bin;
      }

      // Phase 2: Scatter to histogram (serial — gather/scatter can't be vectorised).
      for (int j = 0; j < loop_count; j++) {
        if (fbin[j] >= 0 && fbin[j] < n_slices)
          histo[id][fbin[j]]++;
      }
    }

// Reduce to a single histogram
#pragma omp for
    for (int i = 0; i < n_slices; i++) {
      output[i] = 0.;
      for (int t = 0; t < threads; t++)
        output[i] += histo[t][i];
    }
  }

  // free memory
  free(histo[0]);
  free(histo);
}
