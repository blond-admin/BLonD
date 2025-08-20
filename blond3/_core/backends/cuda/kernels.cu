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
