// Copyright CERN. This software is distributed under the
// terms of the GNU General Public Licence version 3 (GPL Version 3),
// copied verbatim in the file LICENCE.txt.
// In applying this licence, CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization or
// submit itself to any jurisdiction.
// Project website: http://blond.web.cern.ch/

#ifdef USEFLOAT
    typedef float real_t;
#else
    typedef double real_t;
#endif

extern "C"
__global__ void drift_simple(
                     real_t * __restrict__ beam_dt,
                     real_t * __restrict__ beam_dE,
                     const real_t T,
                     const real_t eta_zero,
                     const real_t beta,
                     const real_t energy,
                     const int n_macroparticles
                     )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t coeff = T * eta_zero / (beta * beta * energy);
    for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)
        beam_dt[i] +=  coeff * beam_dE[i];
}


extern "C"
__global__ void kick_single_harmonic(
    real_t  * __restrict__ beam_dt,
    real_t  * __restrict__ beam_dE,
    const real_t charge,
    const real_t voltage,
    const real_t omega_RF,
    const real_t phi_RF,
    const int n_macroparticles,
    const real_t acc_kick
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        beam_dE[i] += charge * voltage * sin(omega_RF*beam_dt[i] + phi_RF) + acc_kick;
    }
}

extern "C"
__global__ void kick_multi_harmonic(
    real_t  * __restrict__ beam_dt,
    real_t  * __restrict__ beam_dE,
    const int n_rf,
    const real_t charge,
    const real_t  * __restrict__ voltage,
    const real_t  * __restrict__ omega_RF,
    const real_t  * __restrict__ phi_RF,
    const int n_macroparticles,
    const real_t acc_kick
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t my_beam_dt;
    real_t my_beam_dE;

    if (n_rf == 1) {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x)
            beam_dE[i] += charge * voltage[0] * sin(omega_RF[0]*beam_dt[i] + phi_RF[0]) + acc_kick;

    } else if (n_rf == 2) {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x){
            const real_t dE_sum = (
                charge * voltage[0] * sin(omega_RF[0]*beam_dt[i] + phi_RF[0])
              + charge * voltage[1] * sin(omega_RF[1]*beam_dt[i] + phi_RF[1])
              );
            beam_dE[i] += dE_sum + acc_kick;
        }

    } else if (n_rf == 3) {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x){
            const real_t dE_sum = (
                charge * voltage[0] * sin(omega_RF[0]*beam_dt[i] + phi_RF[0])
              + charge * voltage[1] * sin(omega_RF[1]*beam_dt[i] + phi_RF[1])
              + charge * voltage[2] * sin(omega_RF[2]*beam_dt[i] + phi_RF[2])
              );
            beam_dE[i] += dE_sum + acc_kick;
        }
    } else if (n_rf == 4) {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x){
            const real_t dE_sum = (
                charge * voltage[0] * sin(omega_RF[0]*beam_dt[i] + phi_RF[0])
              + charge * voltage[1] * sin(omega_RF[1]*beam_dt[i] + phi_RF[1])
              + charge * voltage[2] * sin(omega_RF[2]*beam_dt[i] + phi_RF[2])
              + charge * voltage[3] * sin(omega_RF[3]*beam_dt[i] + phi_RF[3])
              );
            beam_dE[i] += dE_sum + acc_kick;
        }
    } else {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
            my_beam_dt = beam_dt[i];
            my_beam_dE = beam_dE[i];
            for (int j = 0; j < n_rf; j++) {
                my_beam_dE += charge * voltage[j] * sin(omega_RF[j]*my_beam_dt + phi_RF[j]);
            }
            beam_dE[i] = my_beam_dE + acc_kick;
        }
    }



}


extern "C"
__global__ void beam_phase(const real_t* __restrict__ hist_x,
                           const real_t* __restrict__ hist_y,
                           real_t* result,
                           real_t alpha,
                           real_t omega_rf,
                           real_t phi_rf,
                           real_t bin_size,
                           int n_bins)
{
    extern __shared__ real_t shared[];

    real_t* sin_partial = shared;
    real_t* cos_partial = shared + blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    real_t sin_val = 0.0;
    real_t cos_val = 0.0;

    if (i < n_bins) {
        real_t x = hist_x[i];
        real_t prof = hist_y[i];
        real_t phase = omega_rf * x + phi_rf;
        real_t base = exp(alpha * x) * prof;

        real_t coeff = ((i == 0) || (i == n_bins - 1)) ? 1.0 : 2.0;

        sin_val = coeff * base * sin(phase);
        cos_val = coeff * base * cos(phase);
    }

    sin_partial[threadIdx.x] = sin_val;
    cos_partial[threadIdx.x] = cos_val;

    __syncthreads();

    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sin_partial[threadIdx.x] += sin_partial[threadIdx.x + s];
            cos_partial[threadIdx.x] += cos_partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Only thread 0 adds to global memory
    if (threadIdx.x == 0) {
        atomicAdd(&result[0], sin_partial[0]);
        atomicAdd(&result[1], cos_partial[0]);
    }
}



extern "C"
__global__ void hybrid_histogram(
                                 const real_t * __restrict__  input,
                                 real_t * __restrict__  output,
                                 const real_t cut_left,
                                 const real_t cut_right,
                                 const unsigned int n_slices,
                                 const unsigned int n_macroparticles,
                                 const int capacity
                                 )
{
    extern __shared__ int block_hist[];
    //reset shared memory
    for (int i = threadIdx.x; i < capacity; i += blockDim.x)
        block_hist[i] = 0;
    __syncthreads();
    int const tid = threadIdx.x + blockDim.x * blockIdx.x;
    int target_bin;
    real_t const inv_bin_width = n_slices / (cut_right - cut_left);

    const int low_tbin = (n_slices / 2) - (capacity / 2);
    const int high_tbin = low_tbin + capacity;


    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        if (input[i] == cut_right){
            target_bin = n_slices - 1;
            if (target_bin >= low_tbin && target_bin < high_tbin)
                atomicAdd(&(block_hist[target_bin - low_tbin]), 1);
            else
                atomicAdd(&(output[target_bin]), 1);
            continue;
        }
        target_bin = floor((input[i] - cut_left) * inv_bin_width);
        if (target_bin < 0 || target_bin >= n_slices)
            continue;
        if (target_bin >= low_tbin && target_bin < high_tbin)
            atomicAdd(&(block_hist[target_bin - low_tbin]), 1);
        else
            atomicAdd(&(output[target_bin]), 1);

    }
    __syncthreads();
    for (int i = threadIdx.x; i < capacity; i += blockDim.x)
        atomicAdd(&output[low_tbin + i], (real_t) block_hist[i]);
}


extern "C"
__global__ void sm_histogram(const real_t * __restrict__  input,
                             real_t * __restrict__  output,
                             const real_t cut_left,
                             const real_t cut_right,
                             const unsigned int n_slices,
                             const unsigned int n_macroparticles)
{
    extern __shared__ int block_hist[];
    for (int i = threadIdx.x; i < n_slices; i += blockDim.x)
        block_hist[i] = 0;
    __syncthreads();
    int const tid = threadIdx.x + blockDim.x * blockIdx.x;
    int target_bin;
    real_t const inv_bin_width = n_slices / (cut_right - cut_left);
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        target_bin = floor((input[i] - cut_left) * inv_bin_width);

        if (input[i] == cut_right){
            target_bin = n_slices - 1;
            atomicAdd(&(block_hist[target_bin]), 1);
            continue;
        }

        if (target_bin < 0 || target_bin >= n_slices)
            continue;

        atomicAdd(&(block_hist[target_bin]), 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_slices; i += blockDim.x)
        atomicAdd(&output[i], (real_t) block_hist[i]);
}




extern "C"
__global__ void lik_only_gm_copy(
    real_t * __restrict__ beam_dt,
    real_t * __restrict__ beam_dE,
    const real_t * __restrict__ voltage_array,
    const real_t * __restrict__ bin_centers,
    const real_t charge,
    const int n_slices,
    const int n_macroparticles,
    const real_t acc_kick,
    real_t * __restrict__ glob_vkick_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t const inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1] - bin_centers[0]);


    for (int i = tid; i < n_slices - 1; i += gridDim.x * blockDim.x) {
        glob_vkick_factor[2*i] = charge * (voltage_array[i + 1] - voltage_array[i])
                              * inv_bin_width;
        glob_vkick_factor[2*i+1] = (charge * voltage_array[i] - bin_centers[i] * glob_vkick_factor[2*i])
                         + acc_kick;
    }
}


extern "C"
__global__ void lik_only_gm_comp(
    real_t * __restrict__ beam_dt,
    real_t * __restrict__ beam_dE,
    const real_t * __restrict__ voltage_array,
    const real_t * __restrict__ bin_centers,
    const real_t charge,
    const int n_slices,
    const int n_macroparticles,
    const real_t acc_kick,
    real_t * __restrict__ glob_vkick_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t const inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1] - bin_centers[0]);
    int fbin;
    const real_t bin0 = bin_centers[0];
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        fbin = floor((beam_dt[i] - bin0) * inv_bin_width);
        if ((fbin < n_slices - 1) && (fbin >= 0))
            beam_dE[i] += beam_dt[i] * glob_vkick_factor[2*fbin] + glob_vkick_factor[2*fbin+1];
    }
}


extern "C"
__global__ void loss_box(
                     const real_t e_max,
                     const real_t e_min,
                     const real_t t_min,
                     const real_t t_max,
                     const real_t * dt,
                     const real_t * dE,
                     int * __restrict__ flags,
                     const int n_macroparticles
                     )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x){
        const bool outside = (dE[i] > e_max) || (dE[i] < e_min) || (dt[i] < t_min) || (dt[i] > t_max);
        if (outside){
            flags[i] =  -500; // assume (BeamFlags.LOST.value)
        }
        }
}

extern "C" __global__ void drift_exact(real_t *__restrict__ beam_dt,
                                       const real_t *__restrict__ beam_dE,
                                       const real_t T, const real_t alpha_zero,
                                       const real_t *__restrict__ higher_alpha,
                                       const int n_alpha, const real_t beta,
                                       const real_t energy,
                                       const int n_macroparticles) {
  const real_t inv_beta_sq = 1.0 / (beta * beta);
  const real_t inv_energy = 1.0 / energy;
  const real_t inv_energy_sq = inv_energy * inv_energy;

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = tid; i < n_macroparticles; i = i + blockDim.x * gridDim.x) {

    const real_t dE = beam_dE[i];

    const real_t delta = sqrt(1.0 + inv_beta_sq * (dE * dE * inv_energy_sq +
                                                   2.0 * dE * inv_energy)) -
                         1.0;

    real_t poly = 1.0 + alpha_zero * delta;

    if (n_alpha > 0 && higher_alpha != nullptr) {
      real_t delta_power = delta * delta; // starts at δ²

      for (int k = 0; k < n_alpha; ++k) {
        poly += higher_alpha[k] * delta_power;
        delta_power *= delta; // next power
      }
    }

    beam_dt[i] += T * (poly * (1.0 + dE * inv_energy) / (1.0 + delta) - 1.0);
  }
}


extern "C"
__global__ void histogram_sparse(
    const real_t *__restrict__ input,
    real_t *__restrict__ output,
    const real_t first_left_cut,
    const real_t left_cut_distance,
    const real_t cut_width,
    const int bins_per_profile,
    const int n_buckets,
    const int n_macroparticles,
    const bool *__restrict__ filling_pattern,
    const int *__restrict__ bucket_index_to_memory_index)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    const real_t cut_left0 = first_left_cut;
    const real_t inv_hist_dist = real_t(1) / left_cut_distance;
    const real_t inv_bin_width =
        real_t(bins_per_profile) / cut_width;


    // Loop through input particles and update histograms in shared memory
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        const real_t dt = input[i];

        const int bucket_i = (int)((dt - cut_left0) * inv_hist_dist);
        if (bucket_i >= n_buckets || bucket_i < 0)
            continue;
        if (!filling_pattern[bucket_i]){
            continue;
        }
        const real_t cut_left = cut_left0 + bucket_i * left_cut_distance;
        const real_t cut_right = cut_left + cut_width;

        // Check if the value is within the cut range
        if (dt == cut_right) {
            atomicAdd(&output[bucket_index_to_memory_index[bucket_i] + bins_per_profile - 1], 1);
            continue;
        }
        if (dt < cut_left || dt >= cut_right)
            continue;

        // Calculate the bin index
        const int bin = (int)((dt - cut_left) * inv_bin_width);
        if ((unsigned)bin < (unsigned)bins_per_profile) {
            atomicAdd(&output[bucket_index_to_memory_index[bucket_i] + bin], 1);
        }
    }
    __syncthreads();


}


// Apply pole-residue (vector fitting) model to a beam profile to generate
// induced voltage. Mirrors the CPU/OpenMP implementation in cpp/poles.cpp but
// is parallelized one thread per pole. The per-pole state evolution is
// sequential across bins; different poles are fully independent and contend
// only on the output `voltage` buffer via atomicAdd.
//
// Complex arrays (poles, residues, states) are stored as interleaved real/imag:
//   [re0, im0, re1, im1, ...]
// The last complex element of `states` stores t_start in its real part.
extern "C" __global__ void apply_poles(
    const real_t * __restrict__ profile,
    const real_t * __restrict__ profile_dts,
    const real_t * __restrict__ poles,
    const real_t * __restrict__ residues,
    const int is_counterrotating_beam,
    const real_t * __restrict__ cr_pole_signs,
    real_t * __restrict__ states,
    real_t * __restrict__ voltage,
    const int * __restrict__ update_on_bin,
    const real_t factor,
    const int n_bins,
    const int n_poles,
    const int n_updates,
    const int n_profile_dts)
{
    const int pole_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pole_i >= n_poles) return;

    const real_t two_factor = real_t(2) * factor;
    const real_t t_start = states[2 * n_poles];

    real_t cr_pole_flip = real_t(1);
    if (is_counterrotating_beam && cr_pole_signs[pole_i] == real_t(-1)) {
        cr_pole_flip = real_t(-1);
    }

    const int pole_n = 2 * pole_i;
    const real_t pole_re = poles[pole_n];
    const real_t pole_im = poles[pole_n + 1];
    const real_t res_re  = residues[pole_n];
    const real_t res_im  = residues[pole_n + 1];

    real_t state_re = states[pole_n];
    real_t state_im = states[pole_n + 1];

    int i_update = 0;
    int update_on_bin_i = (n_updates > 0) ? update_on_bin[0] : -1;

    real_t decay_re = real_t(0);
    real_t decay_im = real_t(0);

    for (int bin_i = 0; bin_i < n_bins; ++bin_i) {
        if (bin_i == update_on_bin_i) {
            const real_t t_jump = (bin_i == 0)
                ? (profile_dts[0] - t_start)
                : (profile_dts[bin_i] - profile_dts[bin_i - 1]);

            // state *= exp(pole * t_jump)
            {
                const real_t e_mag = exp(pole_re * t_jump);
                const real_t c     = cos(pole_im * t_jump);
                const real_t s     = sin(pole_im * t_jump);
                const real_t e_re  = e_mag * c;
                const real_t e_im  = e_mag * s;
                const real_t nr    = state_re * e_re - state_im * e_im;
                const real_t ni    = state_re * e_im + state_im * e_re;
                state_re = nr;
                state_im = ni;
            }

            // decay = exp(pole * dt)
            const real_t dt = profile_dts[bin_i + 1] - profile_dts[bin_i];
            {
                const real_t e_mag = exp(pole_re * dt);
                const real_t c     = cos(pole_im * dt);
                const real_t s     = sin(pole_im * dt);
                decay_re = e_mag * c;
                decay_im = e_mag * s;
            }

            ++i_update;
            if (i_update < n_updates) {
                update_on_bin_i = update_on_bin[i_update];
            }
        } else {
            // state *= decay
            const real_t nr = state_re * decay_re - state_im * decay_im;
            const real_t ni = state_re * decay_im + state_im * decay_re;
            state_re = nr;
            state_im = ni;
        }

        const real_t half_step = cr_pole_flip * (real_t(0.5) * profile[bin_i]) * two_factor;

        // First half of the trapezoidal rule.
        state_re += half_step;

        // amp = Re(residue * state)
        const real_t amp = res_re * state_re - res_im * state_im;
        atomicAdd(&voltage[bin_i], cr_pole_flip * amp);

        // Second half of the trapezoidal rule.
        state_re += half_step;
    }

    // Persist state for the next call.
    states[pole_n]     = state_re;
    states[pole_n + 1] = state_im;

    // Only one thread writes t_start for the next call.
    if (pole_i == 0) {
        states[2 * n_poles]     = profile_dts[n_profile_dts - 1];
        states[2 * n_poles + 1] = real_t(0);
    }
}
