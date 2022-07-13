#include <cupy/complex.cuh>
#include <curand_kernel.h>
#include <stdio.h>
#define REDUCE(a, b) (a+b)
#define BLOCK_SIZE 512
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
__global__ void gpu_losses_longitudinal_cut(
    float *dt,
    float *dev_id,
    const int size,
    const float min_dt,
    const float max_dt)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < size; i += blockDim.x * gridDim.x)
        if ((dt[i] - min_dt) * (max_dt - dt[i]) < 0)
            dev_id[i] = 0;
}

extern "C"
__global__ void gpu_losses_energy_cut(
    float *dE,
    float *dev_id,
    const int size,
    const float min_dE,
    const float max_dE)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < size; i += blockDim.x * gridDim.x)
        if ((dE[i] - min_dE) * (max_dE - dE[i]) < 0)
            dev_id[i] = 0;
}

extern "C"
__global__ void gpu_losses_below_energy(
    float *dE,
    float *dev_id,
    const int size,
    const float min_dE)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < size; i += blockDim.x * gridDim.x)
        if (dE[i] - min_dE < 0)
            dev_id[i] = 0;
}

extern "C"
__global__ void cuinterp(float *x,
                         int x_size,
                         float *xp,
                         int xp_size,
                         float *yp,
                         float *y,
                         float left,
                         float right)
{
    if (left == 0.12345)
        left = yp[0];
    if (right == 0.12345)
        right = yp[xp_size - 1];
    float curr;
    int lo;
    int mid;
    int hi;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid; i < x_size; i += blockDim.x * gridDim.x) {
        //need to find the right bin with binary search
        // looks like bisect_left
        curr = x[i];
        hi = xp_size;
        lo = 0;
        while (lo < hi) {
            mid = (lo + hi) / 2;
            if (xp[mid] < curr)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo == xp_size)
            y[i] = right;
        else if (xp[lo - 1] == curr)
            y[i] = yp[i];
        else if (lo <= 1)
            y[i] = left;
        else {
            y[i] = yp[lo - 1] +
                   (yp[lo] - yp[lo - 1]) * (x[i] - xp[lo - 1]) /
                   (xp[lo] - xp[lo - 1]);
        }

    }
}

extern "C"
__global__ void cugradient(
    float x,
    int *y,
    float *g,
    int size)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = tid + 1; i < size - 1; i += blockDim.x * gridDim.x) {

        g[i] = (y[i + 1] - y[i - 1]) / (2 * x);
        // g[i] = (hs*hs*fd + (hd*hd-hs*hs)*fx - hd*hd*fs)/
        //     (hs*hd*(hd+hs));
    }
    if (tid == 0)
        g[0] = (y[1] - y[0]) / x;
    if (tid == 32)
        g[size - 1] = (y[size - 1] - y[size - 2]) / x;
}


extern "C"
__global__ void gpu_beam_fb_track_other(float *omega_rf,
                                        float *harmonic,
                                        float *dphi_rf,
                                        float *omega_rf_d,
                                        float *phi_rf,
                                        // float pi,
                                        float domega_rf,
                                        int size,
                                        int counter,
                                        int n_rf)
{
    float a, b, c;
    for (int i = threadIdx.x; i < n_rf; i += blockDim.x) {
        a = domega_rf * harmonic[i * size + counter] / harmonic[counter];
        b =  2.0 * PI * harmonic[size * i + counter] * (a + omega_rf[i * size + counter] - omega_rf_d[size * i + counter]) / omega_rf_d[size * i + counter];
        c = dphi_rf[i] +  b;
        omega_rf[i * size + counter] += a;
        dphi_rf[i] +=  b;
        phi_rf[size * i + counter] += c;
    }
}

extern "C"
__global__ void gpu_rf_voltage_calc_mem_ops(float *new_voltages,
        float *new_omega_rf,
        float *new_phi_rf,
        float *voltages,
        float *omega_rf,
        float *phi_rf,
        int start,
        int end,
        int step)
{
    int idx = 0;
    for (int i = threadIdx.x * step + start; i < end; i += blockDim.x * step) {
        new_voltages[idx] = voltages[i];
        new_omega_rf[idx] = omega_rf[i];
        new_phi_rf[idx] = phi_rf[i];
        idx++;
    }
}

extern "C"
__global__ void halve_edges(float *my_array, int size) {
    //__shared__ my_sum;
    int tid = threadIdx.x;
    if (tid == 0) {
        my_array[0] = my_array[0] / 2.;
    }
    if (tid == 32) {
        my_array[size - 1] = my_array[size - 1] / 2.;
    }
}

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
    float sin_res;
    float dummy;
    for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x) {
        my_beam_dt = beam_dt[i];
        for (int j = 0; j < n_rf; j++) {
            sincosf(omega_RF[j]*my_beam_dt + phi_RF[j], &sin_res, &dummy);
            beam_dE[i] += voltage[j] * sin_res;
        }
        beam_dE[i] += acc_kick;
    }
}

extern "C"
__global__ void rf_volt_comp(const float * __restrict__ voltage,
                             const float * __restrict__ omega_rf,
                             const float * __restrict__ phi_rf,
                             const float * __restrict__ bin_centers,
                             const int n_rf,
                             const int n_bins,
                             const int f_rf,
                             float * __restrict__ rf_voltage)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < n_bins; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_rf; j++)
            rf_voltage[i] = voltage[j] * sin(omega_rf[j] * bin_centers[i] + phi_rf[j]);
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
__global__ void histogram(float * input,
                          int * output, const float cut_left,
                          const float cut_right, const int n_slices,
                          const int n_macroparticles)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int target_bin;
    float const inv_bin_width = n_slices / (cut_right - cut_left);
    for (int i = tid; i < n_macroparticles; i = i + blockDim.x * gridDim.x) {
        target_bin = floorf((input[i] - cut_left) * inv_bin_width);
        if (target_bin < 0 || target_bin >= n_slices)
            continue;
        atomicAdd(&(output[target_bin]), 1);
    }
}

extern "C"
__global__ void hybrid_histogram(float * input,
                                 int * output, const float cut_left,
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
        atomicAdd(&output[low_tbin + i], block_hist[i]);
}


extern "C"
__global__ void sm_histogram(const float * __restrict__ input,
                             int * __restrict__ output, const float cut_left,
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
        atomicAdd(&output[i], block_hist[i]);
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
    float * __restrict__ glob_voltageKick,
    float * __restrict__ glob_factor
)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float const inv_bin_width = (n_slices - 1)
                                 / (bin_centers[n_slices - 1] - bin_centers[0]);


    for (int i = tid; i < n_slices - 1; i += gridDim.x * blockDim.x) {
        glob_voltageKick[i] = charge * (voltage_array[i + 1] - voltage_array[i])
                              * inv_bin_width;
        glob_factor[i] = (charge * voltage_array[i] - bin_centers[i] * glob_voltageKick[i])
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
    float * __restrict__ glob_voltageKick,
    float * __restrict__ glob_factor
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
            beam_dE[i] += beam_dt[i] * glob_voltageKick[fbin] + glob_factor[fbin];
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
    float *glob_voltageKick,
    float *glob_factor,
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
            beam_dE[i] += beam_dt[i] * glob_voltageKick[fbin] + glob_factor[fbin];
        // beam_dt[i] += T * (1. / (1. - eta0 * beam_dE[i]) -1.);
        beam_dt[i] += T * beam_dE[i];
    }
}

extern "C"
__global__ void beam_phase_v2(
    const float * __restrict__ bin_centers,
    const int * __restrict__ profile,
    const float alpha,
    // const float * __restrict__ omega_rf_ar,
    // const float * __restrict__ phi_rf_ar,
    const float omega_rf,
    const float phi_rf,
    // const int ind,
    const float bin_size,
    float * __restrict__ array1,
    float * __restrict__ array2,
    const int n_bins)
{
    // float omega_rf = omega_rf_ar[ind];
    // float phi_rf = phi_rf_ar[ind];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float a, b;
    float sin_res, cos_res;
    for (int i = tid; i < n_bins; i += gridDim.x * blockDim.x) {
        a = omega_rf * bin_centers[i] + phi_rf;
        sincosf(a, &sin_res, &cos_res);
        b = expf(alpha * bin_centers[i]) * profile[i];
        array1[i] = b * sin_res;
        array2[i] = b * cos_res;
    }
}

extern "C"
__global__ void beam_phase_sum(
    const float * __restrict__ ar1,
    const float * __restrict__ ar2,
    float * __restrict__ scoeff,
    float * __restrict__ coeff,
    int n_bins)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid == 0) {
        scoeff[0] = 0;
        coeff[0] = 0;
    }
    __syncthreads();
    float my_sum_1 = 0;
    float my_sum_2 = 0;
    if (tid == 0) {
        my_sum_1 += ar1[0] / 2 + ar1[n_bins - 1] / 2;
        my_sum_2 += ar2[0] / 2 + ar2[n_bins - 1] / 2;
    }
    for (int i = tid + 1; i < n_bins - 1; i += gridDim.x * blockDim.x) {
        my_sum_1 += ar1[i];
        my_sum_2 += ar2[i];
    }
    atomicAdd(&(scoeff[0]), my_sum_1);
    atomicAdd(&(coeff[0]), my_sum_2);
    __syncthreads();
    if (tid == 0)
        scoeff[0] = scoeff[0] / coeff[0];

}

// extern "C"
// __global__ void gpu_trapz_custom(
//     float *y,
//     float x,
//     int sz,
//     float *res)
// {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     float my_sum = 0;
//     for (int i = tid; i < sz - 1; i += gridDim.x * blockDim.x)
//         my_sum += (y[i] + y[i + 1]) * x / 2.0;

//     atomicAdd(&(res[0]), my_sum);
// }


// extern "C"
// __global__ void gpu_trapz_stage1(float *out, float *y, float x, int sz,
//                       unsigned int seq_count, unsigned int n)
// {
//     // Needs to be variable-size to prevent the braindead CUDA compiler from
//     // running constructors on this array. Grrrr.
//     extern __shared__ float sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * 512 * seq_count + tid;
//     float acc = 0;
//     for (unsigned s = 0; s < seq_count; ++s)
//     {
//         if (i >= n)
//             break;
//         acc = acc + ((i < sz - 1) ? x * (y[i] + y[i + 1]) / 2.0 : 0.0);
//         i += 512;
//     }
//     sdata[tid] = acc;
//     __syncthreads();
// #if (512 >= 512)
//     if (tid < 256) { sdata[tid] = sdata[tid] + sdata[tid + 256]; }
//     __syncthreads();
// #endif
// #if (512 >= 256)
//     if (tid < 128) { sdata[tid] = sdata[tid] + sdata[tid + 128]; }
//     __syncthreads();
// #endif
// #if (512 >= 128)
//     if (tid < 64) { sdata[tid] = sdata[tid] + sdata[tid + 64]; }
//     __syncthreads();
// #endif
//     if (tid < 32)
//     {
//         // 'volatile' required according to Fermi compatibility guide 1.2.2
//         volatile float *smem = sdata;
//         if (512 >= 64) smem[tid] = smem[tid] + smem[tid + 32];
//         if (512 >= 32) smem[tid] = smem[tid] + smem[tid + 16];
//         if (512 >= 16) smem[tid] = smem[tid] + smem[tid + 8];
//         if (512 >= 8)  smem[tid] = smem[tid] + smem[tid + 4];
//         if (512 >= 4)  smem[tid] = smem[tid] + smem[tid + 2];
//         if (512 >= 2)  smem[tid] = smem[tid] + smem[tid + 1];
//     }
//     if (tid == 0) out[blockIdx.x] = sdata[0];
// }



// extern "C"
// __global__ void gpu_trapz_stage2(float *out, const float *cupy_reduction_inp, float *y, float x, int sz,
//                       unsigned int seq_count, unsigned int n)
// {
//     // Needs to be variable-size to prevent the braindead CUDA compiler from
//     // running constructors on this array. Grrrr.
//     extern __shared__ float sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * 512 * seq_count + tid;
//     float acc = 0;
//     for (unsigned s = 0; s < seq_count; ++s)
//     {
//         if (i >= n)
//             break;
//         acc = acc + (cupy_reduction_inp[i]);
//         i = 512;
//     }
//     sdata[tid] = acc;
//     __syncthreads();
// #if (512 >= 512)
//     if (tid < 256) { sdata[tid] = sdata[tid] + sdata[tid + 256]; }
//     __syncthreads();
// #endif
// #if (512 >= 256)
//     if (tid < 128) { sdata[tid] = sdata[tid] + sdata[tid + 128]; }
//     __syncthreads();
// #endif
// #if (512 >= 128)
//     if (tid < 64) { sdata[tid] = sdata[tid] + sdata[tid + 64]; }
//     __syncthreads();
// #endif
//     if (tid < 32)
//     {
//         // 'volatile' required according to Fermi compatibility guide 1.2.2
//         volatile float *smem = sdata;
//         if (512 >= 64) smem[tid] = smem[tid] + smem[tid + 32];
//         if (512 >= 32) smem[tid] = smem[tid] + smem[tid + 16];
//         if (512 >= 16) smem[tid] = smem[tid] + smem[tid + 8];
//         if (512 >= 8)  smem[tid] = smem[tid] + smem[tid + 4];
//         if (512 >= 4)  smem[tid] = smem[tid] + smem[tid + 2];
//         if (512 >= 2)  smem[tid] = smem[tid] + smem[tid + 1];
//     }
//     if (tid == 0) out[blockIdx.x] = sdata[0];
// }


extern "C"
__global__ void mean_non_zeros_stage1(float *out, float *x, float *id,
                           unsigned int seq_count, unsigned int n)
{
    // Needs to be variable-size to prevent the braindead CUDA compiler from
    // running constructors on this array. Grrrr.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 512 * seq_count + tid;
    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
        if (i >= n)
            break;
        acc = acc + ((id[i] != 0) * x[i]);
        i += 512;
    }
    sdata[tid] = acc;
    __syncthreads();
#if (512 >= 512)
    if (tid < 256) { sdata[tid] = sdata[tid] + sdata[tid + 256]; }
    __syncthreads();
#endif
#if (512 >= 256)
    if (tid < 128) { sdata[tid] = sdata[tid] + sdata[tid + 128]; }
    __syncthreads();
#endif
#if (512 >= 128)
    if (tid < 64) { sdata[tid] = sdata[tid] + sdata[tid + 64]; }
    __syncthreads();
#endif
    if (tid < 32)
    {
        // 'volatile' required according to Fermi compatibility guide 1.2.2
        volatile float *smem = sdata;
        if (512 >= 64) smem[tid] = smem[tid] + smem[tid + 32];
        if (512 >= 32) smem[tid] = smem[tid] + smem[tid + 16];
        if (512 >= 16) smem[tid] = smem[tid] + smem[tid + 8];
        if (512 >= 8)  smem[tid] = smem[tid] + smem[tid + 4];
        if (512 >= 4)  smem[tid] = smem[tid] + smem[tid + 2];
        if (512 >= 2)  smem[tid] = smem[tid] + smem[tid + 1];
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}




extern "C"
__global__ void mean_non_zeros_stage2(float *out, const float *cupy_reduction_inp, float *x, float *id,
                           unsigned int seq_count, unsigned int n)
{
    // Needs to be variable-size to prevent the braindead CUDA compiler from
    // running constructors on this array. Grrrr.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * 512 * seq_count + tid;
    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
        if (i >= n)
            break;
        acc = REDUCE(acc, (cupy_reduction_inp[i]));
        i += 512;
    }
    sdata[tid] = acc;
    __syncthreads();
#if (512 >= 512)
    if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
    __syncthreads();
#endif
#if (512 >= 256)
    if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
    __syncthreads();
#endif
#if (512 >= 128)
    if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
    __syncthreads();
#endif
    if (tid < 32)
    {
        // 'volatile' required according to Fermi compatibility guide 1.2.2
        volatile float *smem = sdata;
        if (512 >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
        if (512 >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
        if (512 >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
        if (512 >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
        if (512 >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
        if (512 >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}




extern "C"
__global__ void stdKernel_stage1(float *out, float *x, float *y, float m,
                      unsigned int seq_count, unsigned int n)
{
    // Needs to be variable-size to prevent the braindead CUDA compiler from
    // running constructors on this array. Grrrr.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * seq_count + tid;
    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
        if (i >= n)
            break;
        acc = REDUCE(acc, ((y[i] != 0) * (x[i] - m) * (x[i] - m)));
        i += BLOCK_SIZE;
    }
    sdata[tid] = acc;
    __syncthreads();
#if (BLOCK_SIZE >= 512)
    if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 256)
    if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 128)
    if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
    __syncthreads();
#endif
    if (tid < 32)
    {
        // 'volatile' required according to Fermi compatibility guide 1.2.2
        volatile float *smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
        if (BLOCK_SIZE >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
        if (BLOCK_SIZE >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
        if (BLOCK_SIZE >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
        if (BLOCK_SIZE >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
        if (BLOCK_SIZE >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}




extern "C"
__global__ void stdKernel_stage2(float *out, const float *cupy_reduction_inp, float *x, float *y, float m,
                      unsigned int seq_count, unsigned int n)
{
    // Needs to be variable-size to prevent the braindead CUDA compiler from
    // running constructors on this array. Grrrr.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * seq_count + tid;
    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
        if (i >= n)
            break;
        acc = REDUCE(acc, (cupy_reduction_inp[i]));
        i += BLOCK_SIZE;
    }
    sdata[tid] = acc;
    __syncthreads();
#if (BLOCK_SIZE >= 512)
    if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 256)
    if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 128)
    if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
    __syncthreads();
#endif
    if (tid < 32)
    {
        // 'volatile' required according to Fermi compatibility guide 1.2.2
        volatile float *smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
        if (BLOCK_SIZE >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
        if (BLOCK_SIZE >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
        if (BLOCK_SIZE >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
        if (BLOCK_SIZE >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
        if (BLOCK_SIZE >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}




extern "C"
__global__ void sum_non_zeros_stage1(float *out, float *x,
                          unsigned int seq_count, unsigned int n)
{
    // Needs to be variable-size to prevent the braindead CUDA compiler from
    // running constructors on this array. Grrrr.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * seq_count + tid;
    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
        if (i >= n)
            break;
        acc = REDUCE(acc, ((x[i] != 0)));
        i += BLOCK_SIZE;
    }
    sdata[tid] = acc;
    __syncthreads();
#if (BLOCK_SIZE >= 512)
    if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 256)
    if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 128)
    if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
    __syncthreads();
#endif
    if (tid < 32)
    {
        // 'volatile' required according to Fermi compatibility guide 1.2.2
        volatile float *smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
        if (BLOCK_SIZE >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
        if (BLOCK_SIZE >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
        if (BLOCK_SIZE >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
        if (BLOCK_SIZE >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
        if (BLOCK_SIZE >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}




extern "C"
__global__ void sum_non_zeros_stage2(float *out, const float *cupy_reduction_inp, float *x,
                          unsigned int seq_count, unsigned int n)
{
    // Needs to be variable-size to prevent the braindead CUDA compiler from
    // running constructors on this array. Grrrr.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * BLOCK_SIZE * seq_count + tid;
    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
        if (i >= n)
            break;
        acc = REDUCE(acc, (cupy_reduction_inp[i]));
        i += BLOCK_SIZE;
    }
    sdata[tid] = acc;
    __syncthreads();
#if (BLOCK_SIZE >= 512)
    if (tid < 256) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 256]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 256)
    if (tid < 128) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 128]); }
    __syncthreads();
#endif
#if (BLOCK_SIZE >= 128)
    if (tid < 64) { sdata[tid] = REDUCE(sdata[tid], sdata[tid + 64]); }
    __syncthreads();
#endif
    if (tid < 32)
    {
        // 'volatile' required according to Fermi compatibility guide 1.2.2
        volatile float *smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] = REDUCE(smem[tid], smem[tid + 32]);
        if (BLOCK_SIZE >= 32) smem[tid] = REDUCE(smem[tid], smem[tid + 16]);
        if (BLOCK_SIZE >= 16) smem[tid] = REDUCE(smem[tid], smem[tid + 8]);
        if (BLOCK_SIZE >= 8)  smem[tid] = REDUCE(smem[tid], smem[tid + 4]);
        if (BLOCK_SIZE >= 4)  smem[tid] = REDUCE(smem[tid], smem[tid + 2]);
        if (BLOCK_SIZE >= 2)  smem[tid] = REDUCE(smem[tid], smem[tid + 1]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}






extern "C"
__global__ void bm_phase_exp_times_scalar_range(float *a, float *b, float c, int *d , long start, long stop, long step)
{
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x * blockDim.x;
    unsigned cta_start = blockDim.x * blockIdx.x;
    long i;
    ;
    if (step < 0)
    {
        for (i = start + (cta_start + tid) * step;
                i > stop; i += total_threads * step)
        {
            a[i] = expf(c * b[i]) * d[i];
        }
    }
    else
    {
        for (i = start + (cta_start + tid) * step;
                i < stop; i += total_threads * step)
        {
            a[i] = expf(c * b[i]) * d[i];
        }
    }
    ;
}



extern "C"
__global__ void bm_phase_mul_add_range(float *a, float b, float *c, float d , long start, long stop, long step)
{
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x * blockDim.x;
    unsigned cta_start = blockDim.x * blockIdx.x;
    long i;
    ;
    if (step < 0)
    {
        for (i = start + (cta_start + tid) * step;
                i > stop; i += total_threads * step)
        {
            a[i] = b * c[i] + d;
        }
    }
    else
    {
        for (i = start + (cta_start + tid) * step;
                i < stop; i += total_threads * step)
        {
            a[i] = b * c[i] + d;
        }
    }
    ;
}



extern "C"
__global__ void bm_sin_cos_range(float *a, float *b, float *c , long start, long stop, long step)
{
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x * blockDim.x;
    unsigned cta_start = blockDim.x * blockIdx.x;
    long i;
    ;
    if (step < 0)
    {
        for (i = start + (cta_start + tid) * step;
                i > stop; i += total_threads * step)
        {
            sincosf(a[i], &b[i], &c[i]);
        }
    }
    else
    {
        for (i = start + (cta_start + tid) * step;
                i < stop; i += total_threads * step)
        {
            sincosf(a[i], &b[i], &c[i]);
        }
    }
    ;
}




extern "C"
__global__ void synchrotron_radiation(
    float *  beam_dE,
    const float U0,
    const int n_macroparticles,
    const float tau_z,
    const int n_kicks)
{

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const float const_synch_rad = 2.0 / tau_z;

    for (int j = 0; j < n_kicks; j++) {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x)
            beam_dE[i] -= const_synch_rad * beam_dE[i] + U0;
    }
}

extern "C"
__global__ void synchrotron_radiation_full(
    float *  beam_dE,
    const float U0,
    const int n_macroparticles,
    const float sigma_dE,
    const float tau_z,
    const float energy,
    const int n_kicks
)
{   unsigned int seed = 0;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    const float const_quantum_exc = 2.0 * sigma_dE / sqrtf(tau_z) * energy;
    curandState_t state;
    curand_init(seed, tid, 0, &state);
    const float const_synch_rad = 2.0 / tau_z;
    for (int j = 0; j < n_kicks; j++) {
        for (int i = tid; i < n_macroparticles; i += blockDim.x * gridDim.x)
            beam_dE[i] -= const_synch_rad * beam_dE[i] + U0 - const_quantum_exc * curand_normal(&state);
    }
}
