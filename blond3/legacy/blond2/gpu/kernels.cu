#include <cupy/complex.cuh>
#include <curand_kernel.h>

#ifdef USEFLOAT
    typedef float real_t;
    #define CURAND_NORMAL curand_normal
#else
    typedef double real_t;
    #define CURAND_NORMAL curand_normal_double

    // Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap).
    // For example, atomicAdd() for double-precision floating-point numbers is not
    // available on devices with compute capability lower than 6.0 but it can be implemented
    // as follows:
    #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
    #else
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
    #endif
#endif


extern "C"
__global__ void simple_kick(
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
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        my_beam_dt = beam_dt[i];
        my_beam_dE = beam_dE[i];
        for (int j = 0; j < n_rf; j++) {
            my_beam_dE += charge * voltage[j] * sin(omega_RF[j]*my_beam_dt + phi_RF[j]);
        }
        beam_dE[i] = my_beam_dE + acc_kick;
    }
}

extern "C"
__global__ void rf_volt_comp(const real_t * __restrict__ voltage,
                             const real_t * __restrict__ omega_rf,
                             const real_t * __restrict__ phi_rf,
                             const real_t * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             real_t * __restrict__ rf_voltage)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t my_rf_voltage;
    real_t my_bin_centers;
    for (int i = tid; i < n_bins; i += blockDim.x * gridDim.x) {
        my_rf_voltage = rf_voltage[i];
        my_bin_centers = bin_centers[i];
        for (int j = 0; j < n_rf; j++)
            my_rf_voltage += voltage[j] * sin(omega_rf[j] * my_bin_centers + phi_rf[j]);
        rf_voltage[i] = my_rf_voltage;
    }
}

extern "C"
__global__ void drift(real_t * __restrict__ beam_dt,
                     real_t  * __restrict__ beam_dE,
                     const int solver,
                     const real_t T0, const real_t length_ratio,
                     const real_t alpha_order, const real_t eta_zero,
                     const real_t eta_one, const real_t eta_two,
                     const real_t alpha_zero, const real_t alpha_one,
                     const real_t alpha_two,
                     const real_t beta, const real_t energy,
                     const int n_macroparticles)
{
    real_t T = T0 * length_ratio;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if ( solver == 0 )
    {
        real_t coeff = eta_zero / (beta * beta * energy);
        for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)
            beam_dt[i] += T * coeff * beam_dE[i];
    }

    else if ( solver == 1 )
    {
        const real_t coeff = 1. / (beta * beta * energy);
        const real_t eta0 = eta_zero * coeff;
        const real_t eta1 = eta_one * coeff * coeff;
        const real_t eta2 = eta_two * coeff * coeff * coeff;

        if (alpha_order == 0)
            for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) - 1.);
        else if (alpha_order == 1)
            for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]) - 1.);
        else
            for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)
                beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]
                                         - eta1 * beam_dE[i] * beam_dE[i]
                                         - eta2 * beam_dE[i] * beam_dE[i] * beam_dE[i]) - 1.);
    }

    else
    {

        const real_t invbetasq = 1 / (beta * beta);
        const real_t invenesq = 1 / (energy * energy);
        // real_t beam_delta;


        for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)

        {

            real_t beam_delta = sqrt(1. + invbetasq *
                              (beam_dE[i] * beam_dE[i] * invenesq + 2.*beam_dE[i] / energy)) - 1.;

            beam_dt[i] += T * (
                              (1. + alpha_zero * beam_delta +
                               alpha_one * (beam_delta * beam_delta) +
                               alpha_two * (beam_delta * beam_delta * beam_delta)) *
                              (1. + beam_dE[i] / energy) / (1. + beam_delta) - 1.);
        }
    }
}


extern "C"
__global__ void hybrid_histogram(const real_t * __restrict__  input,
                                 real_t * __restrict__  output, const real_t cut_left,
                                 const real_t cut_right, const unsigned int n_slices,
                                 const int n_macroparticles, const int capacity)
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
                             real_t * __restrict__  output, const real_t cut_left,
                             const real_t cut_right, const unsigned int n_slices,
                             const int n_macroparticles)
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
__global__ void lik_drift_only_gm_comp(
    real_t *beam_dt,
    real_t *beam_dE,
    const real_t *voltage_array,
    const real_t *bin_centers,
    const real_t charge,
    const int n_slices,
    const int n_macroparticles,
    const real_t acc_kick,
    real_t *glob_vkick_factor,
    const real_t T0, const real_t length_ratio,
    const real_t eta0, const real_t beta, const real_t energy
)
{
    const real_t T = T0 * length_ratio * eta0 / (beta * beta * energy);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t const inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1] - bin_centers[0]);
    unsigned fbin;
    const real_t bin0 = bin_centers[0];
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        fbin = (unsigned) floor((beam_dt[i] - bin0) * inv_bin_width);
        if ((fbin < n_slices - 1))
            beam_dE[i] += beam_dt[i] * glob_vkick_factor[2*fbin] + glob_vkick_factor[2*fbin+1];
        // beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) -1.);
        beam_dt[i] += T * beam_dE[i];
    }
}

// This function calculates and applies only the synchrotron radiation damping term
extern "C"
__global__ void synchrotron_radiation(
    real_t *  beam_dE,
    const real_t U0,
    const int n_macroparticles,
    const real_t tau_z,
    const int n_kicks)
{

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    // SR damping constant, adjusted for better performance
    const real_t const_synch_rad = 1.0 - 2.0 / tau_z;

    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        // SR damping term due to energy spread and
        // Average energy change due to SR
        for (int j = 0; j < n_kicks; j++) {
            beam_dE[i] = beam_dE[i] * const_synch_rad - U0;
        }
    }
}


// This function calculates and applies synchrotron radiation damping and
// quantum excitation terms
extern "C"
__global__ void synchrotron_radiation_full(
    real_t *  beam_dE,
    const real_t U0,
    const int n_macroparticles,
    const real_t sigma_dE,
    const real_t tau_z,
    const real_t energy,
    const int n_kicks
)
{
    unsigned int seed = 1234;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // Quantum excitation constant
    const real_t const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    
    // Adjusted SR damping constant
    const real_t const_synch_rad = 1.0 - 2.0 / tau_z;

    curandState_t state;
    curand_init(seed, tid, 0, &state);

    // Compute synchrotron radiation damping term and
    // Applies the quantum excitation term
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_kicks; j++) {
            beam_dE[i] = beam_dE[i] * const_synch_rad 
                         + const_quantum_exc * CURAND_NORMAL(&state) - U0;
        }
    }
}


extern "C"
__global__ void kickdrift_considering_periodicity(
real_t * __restrict__ beam_dt,
real_t  * __restrict__ beam_dE,
const real_t t_rev_tmp, // self.rf_params.t_rev[turn + 1]
const int  n_rf,
const real_t * __restrict__ voltage, // self.rf_params.voltage[:, turn]
const real_t * __restrict__ omega_rf, // self.rf_params.omega_rf[:, turn]
const real_t * __restrict__ phi_rf, // self.rf_params.voltage[:, turn]
const real_t charge,
const real_t acc_kick, // self.rf_params.voltage[:, turn]
const real_t coeff, // T0 * length_ratio * eta_zero / (beta * beta * energy // (of turn+1)
const int n_macroparticles
){
    // This is a GPU clone of RingAndRFTracker.kickdrift_considering_periodicity
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    real_t my_beam_dt;
    real_t my_beam_dE;
    real_t sum_;
    bool is_left_of_trev;
    for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x){
        my_beam_dt = beam_dt[i];
        my_beam_dE = beam_dE[i];

        // Distinguish the particles inside the frame from the particles on
        // the right-hand side of the frame.
        is_left_of_trev = my_beam_dt < t_rev_tmp;
        if (my_beam_dt > t_rev_tmp){
            my_beam_dt  -= t_rev_tmp;
        }

        // Synchronize the bunch with the particles that are on the
        // RHS of the current frame applying kick and drift to the
        // bunch
        // After that all the particles are in the new updated frame
        if (is_left_of_trev){
            // kick
            sum_ = 0.0;
            for (int j = 0; j < n_rf; j++) {
                sum_ +=  voltage[j] * sin(omega_rf[j] * my_beam_dt + phi_rf[j]);
            }
            my_beam_dE += charge * sum_ + acc_kick;

            // drift, solver = 0
            my_beam_dt += coeff * my_beam_dE;

        }


        // Check all the particles on the left of the just updated
        // frame and apply a second kick and drift to them with the
        // previous wave after having changed reference.
        if (my_beam_dt < 0){
            my_beam_dt += t_rev_tmp;

            // kick
            sum_ = 0.0;
            for (int j = 0; j < n_rf; j++) {
                sum_ +=  voltage[j] * sin(omega_rf[j] * my_beam_dt + phi_rf[j]);
            }
            my_beam_dE += charge * sum_ + acc_kick;
            // drift, solver = 0
            my_beam_dt += coeff * my_beam_dE;
        }

        beam_dE[i] = my_beam_dE;
        beam_dt[i] = my_beam_dt;

    }

}
