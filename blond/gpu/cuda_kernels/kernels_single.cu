#include <cupy/complex.cuh>
#include <curand_kernel.h>
#define PI 3.141592653589793238462643383279502884197169399375105820974944592307816406286
#define PI_DIV_2 3.141592653589793238462643383279502884197169399375105820974944592307816406286/2

// Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap).
// For example, atomicAdd() for double-precision floating-point numbers is not
// available on devices with compute capability lower than 6.0 but it can be implemented
// as follows:
// #if __CUDA_ARCH__ < 600
// __device__ double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                               (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;

//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));

//     // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//     } while (assumed != old);

//     return __longlong_as_double(old);
// }
// #endif


extern "C"
__global__ void simple_kick(
    const float  * __restrict__ beam_dt,
    float        * __restrict__ beam_dE,
    const int n_rf,
    const float  * __restrict__ voltage,
    const float  * __restrict__ omega_RF,
    const float  * __restrict__ phi_RF,
    const int n_macroparticles,
    const float acc_kick
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float my_beam_dt;
    float my_beam_dE;
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        my_beam_dt = beam_dt[i];
        my_beam_dE = beam_dE[i];
        for (int j = 0; j < n_rf; j++) {
            my_beam_dE += voltage[j] * sinf(omega_RF[j]*my_beam_dt + phi_RF[j]);
        }
        beam_dE[i] = my_beam_dE + acc_kick;
    }
}

extern "C"
__global__ void rf_volt_comp(const float * __restrict__ voltage,
                             const float * __restrict__ omega_rf,
                             const float * __restrict__ phi_rf,
                             const float * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             float * __restrict__ rf_voltage)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float my_rf_voltage;
    float my_bin_centers;
    for (int i = tid; i < n_bins; i += blockDim.x * gridDim.x) {
        my_rf_voltage = rf_voltage[i];
        my_bin_centers = bin_centers[i];
        for (int j = 0; j < n_rf; j++)
            my_rf_voltage += voltage[j] * sinf(omega_rf[j] * my_bin_centers + phi_rf[j]);
        rf_voltage[i] = my_rf_voltage;
    }
}

extern "C"
__global__ void drift(float *beam_dt,
        const float  *beam_dE,
        const int solver,
        const float T0, const float length_ratio,
        const float alpha_order, const float eta_zero,
        const float eta_one, const float eta_two,
        const float alpha_zero, const float alpha_one,
        const float alpha_two,
        const float beta, const float energy,
        const int n_macroparticles)
{
    float T = T0 * length_ratio;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if ( solver == 0 )
    {
        float coeff = eta_zero / (beta * beta * energy);
        for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)
            beam_dt[i] += T * coeff * beam_dE[i];
    }

    else if ( solver == 1 )
    {
        const float coeff = 1. / (beta * beta * energy);
        const float eta0 = eta_zero * coeff;
        const float eta1 = eta_one * coeff * coeff;
        const float eta2 = eta_two * coeff * coeff * coeff;

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

        const float invbetasq = 1 / (beta * beta);
        const float invenesq = 1 / (energy * energy);
        // float beam_delta;


        for (int i=tid; i<n_macroparticles; i=i+blockDim.x*gridDim.x)

        {

            float beam_delta = sqrt(1. + invbetasq *
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
__global__ void hybrid_histogram(float * input,
                                 float * output, const float cut_left,
                                 const float cut_right, const unsigned int n_slices,
                                 const int n_macroparticles, const int capacity)
{
    extern __shared__ int block_hist[];
    //reset shared memory
    for (int i = threadIdx.x; i < capacity; i += blockDim.x)
        block_hist[i] = 0;
    __syncthreads();
    int const tid = threadIdx.x + blockDim.x * blockIdx.x;
    int target_bin;
    float const inv_bin_width = n_slices / (cut_right - cut_left);

    const int low_tbin = (n_slices / 2) - (capacity / 2);
    const int high_tbin = low_tbin + capacity;


    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        target_bin = floorf((input[i] - cut_left) * inv_bin_width);
        if (target_bin < 0 || target_bin >= n_slices)
            continue;
        if (target_bin >= low_tbin && target_bin < high_tbin)
            atomicAdd(&(block_hist[target_bin - low_tbin]), 1);
        else
            atomicAdd(&(output[target_bin]), 1);

    }
    __syncthreads();
    for (int i = threadIdx.x; i < capacity; i += blockDim.x)
        atomicAdd(&output[low_tbin + i], (float) block_hist[i]);
}


extern "C"
__global__ void sm_histogram(const float * __restrict__ input,
                             float * __restrict__ output, const float cut_left,
                             const float cut_right, const unsigned int n_slices,
                             const int n_macroparticles)
{
    extern __shared__ int block_hist[];
    for (int i = threadIdx.x; i < n_slices; i += blockDim.x)
        block_hist[i] = 0;
    __syncthreads();
    int const tid = threadIdx.x + blockDim.x * blockIdx.x;
    int target_bin;
    float const inv_bin_width = n_slices / (cut_right - cut_left);
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        target_bin = floorf((input[i] - cut_left) * inv_bin_width);
        if (target_bin < 0 || target_bin >= n_slices)
            continue;
        atomicAdd(&(block_hist[target_bin]), 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_slices; i += blockDim.x)
        atomicAdd(&output[i], (float) block_hist[i]);
}


extern "C"
__global__ void lik_only_gm_copy(
    float * __restrict__ beam_dt,
    float * __restrict__ beam_dE,
    const float * __restrict__ voltage_array,
    const float * __restrict__ bin_centers,
    const float charge,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick,
    float * __restrict__ glob_vkick_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float const inv_bin_width = (n_slices - 1)
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
    float * __restrict__ beam_dt,
    float * __restrict__ beam_dE,
    const float * __restrict__ voltage_array,
    const float * __restrict__ bin_centers,
    const float charge,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick,
    float * __restrict__ glob_vkick_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float const inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1] - bin_centers[0]);
    int fbin;
    const float bin0 = bin_centers[0];
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        fbin = floorf((beam_dt[i] - bin0) * inv_bin_width);
        if ((fbin < n_slices - 1) && (fbin >= 0))
            beam_dE[i] += beam_dt[i] * glob_vkick_factor[2*fbin] + glob_vkick_factor[2*fbin+1];
    }
}


extern "C"
__global__ void lik_drift_only_gm_comp(
    float *beam_dt,
    float *beam_dE,
    const float *voltage_array,
    const float *bin_centers,
    const float charge,
    const int n_slices,
    const int n_macroparticles,
    const float acc_kick,
    float *glob_vkick_factor,
    const float T0, const float length_ratio,
    const float eta0, const float beta, const float energy
)
{
    const float T = T0 * length_ratio * eta0 / (beta * beta * energy);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float const inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1] - bin_centers[0]);
    unsigned fbin;
    const float bin0 = bin_centers[0];
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        fbin = (unsigned) floorf((beam_dt[i] - bin0) * inv_bin_width);
        if ((fbin < n_slices - 1))
            beam_dE[i] += beam_dt[i] * glob_vkick_factor[2*fbin] + glob_vkick_factor[2*fbin+1];
        // beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) -1.);
        beam_dt[i] += T * beam_dE[i];
    }
}


// This function calculates and applies only the synchrotron radiation damping term
extern "C"
__global__ void synchrotron_radiation(
    float *  beam_dE,
    const float U0,
    const int n_macroparticles,
    const float tau_z,
    const int n_kicks)
{

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    // SR damping constant, adjusted for better performance
    const float const_synch_rad = 1.0 - 2.0 / tau_z;

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
    double *  beam_dE,
    const float U0,
    const int n_macroparticles,
    const float sigma_dE,
    const float tau_z,
    const float energy,
    const int n_kicks
)
{
    unsigned int seed = 1234;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // Quantum excitation constant
    const float const_quantum_exc = 2.0 * sigma_dE / sqrt(tau_z) * energy;
    
    // Adjusted SR damping constant
    const float const_synch_rad = 1.0 - 2.0 / tau_z;

    curandState_t state;
    curand_init(seed, tid, 0, &state);

    // Compute synchrotron radiation damping term and
    // Applies the quantum excitation term
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_kicks; j++) {
            beam_dE[i] = beam_dE[i] * const_synch_rad 
                         + const_quantum_exc * curand_normal(&state) - U0;
        }
    }
}

// extern "C"
// __global__ void cuinterp(float *x,
//                          int x_size,
//                          float *xp,
//                          int xp_size,
//                          float *yp,
//                          float *y,
//                          float left,
//                          float right)
// {
//     if (left == 0.12345)
//         left = yp[0];
//     if (right == 0.12345)
//         right = yp[xp_size - 1];
//     float curr;
//     int lo;
//     int mid;
//     int hi;
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     for (int i = tid; i < x_size; i += blockDim.x * gridDim.x) {
//         //need to find the right bin with binary search
//         // looks like bisect_left
//         curr = x[i];
//         hi = xp_size;
//         lo = 0;
//         while (lo < hi) {
//             mid = (lo + hi) / 2;
//             if (xp[mid] < curr)
//                 lo = mid + 1;
//             else
//                 hi = mid;
//         }
//         if (lo == xp_size)
//             y[i] = right;
//         else if (xp[lo - 1] == curr)
//             y[i] = yp[i];
//         else if (lo <= 1)
//             y[i] = left;
//         else {
//             y[i] = yp[lo - 1] +
//                    (yp[lo] - yp[lo - 1]) * (x[i] - xp[lo - 1]) /
//                    (xp[lo] - xp[lo - 1]);
//         }

//     }
// }

// extern "C"
// __global__ void cugradient(
//     float x,
//     int *y,
//     float *g,
//     int size)
// {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     for (int i = tid + 1; i < size - 1; i += blockDim.x * gridDim.x) {

//         g[i] = (y[i + 1] - y[i - 1]) / (2 * x);
//         // g[i] = (hs*hs*fd + (hd*hd-hs*hs)*fx - hd*hd*fs)/
//         //     (hs*hd*(hd+hs));
//     }
//     if (tid == 0)
//         g[0] = (y[1] - y[0]) / x;
//     if (tid == 32)
//         g[size - 1] = (y[size - 1] - y[size - 2]) / x;
// }