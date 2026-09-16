# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# KernelAbstractions kernels. Written once, compiled for every supported
# device. They must stay free of any device-specific API so that the CUDA
# extension needs nothing but `wrap_array`.
#
# Semantics mirror `blond/core/backends/python/callables.py` (readable
# reference) and the loop form of `blond/core/backends/numba/callables.py`.
#
# The simplest per-particle loops additionally have a range function (see
# "Chunked CPU particle loops" at the end of this file), which the CPU runs
# instead of the per-particle kernel.

# Particle layouts.
#
# Every per-particle kernel loops over the particles of its work-item, given
# by a layout argument, so that one kernel serves every launch pattern: one
# particle per work-item on the CPU, and on a GPU a fixed number of
# work-items striding through the particles, as the cuda backend does.
# Launching one GPU work-item per particle instead costs up to 2x
# (kick_interpolated at 1e8 particles: 61 against 32 ms).

"""
    SingleParticleLayout()

Layout with one particle per work-item: work-item `i` handles particle `i`.
"""
struct SingleParticleLayout end

"""
    StrideLayout(n_particles, stride)

Layout of `stride` work-items sharing `n_particles` particles: work-item `i`
handles particles ``i, i + stride, i + 2 stride, ...``. The integer type of
both fields indexes the particles; a GPU steps through `Int32` faster than
through `Int64`.
"""
struct StrideLayout{T <: Integer}
    n_particles::T
    stride::T
end

"""
    particle_indices(layout, item) -> AbstractRange

Particles handled by work-item `item` of `layout`.
"""
@inline particle_indices(::SingleParticleLayout, item::Int) = item:item

@inline function particle_indices(layout::StrideLayout{T}, item::Int) where {T}
    return T(item):layout.stride:layout.n_particles
end

"""
    GPU_WORKGROUP_SIZE

Work-items per workgroup of the strided GPU kernels: the largest workgroup
every GPU generation supports.
"""
const GPU_WORKGROUP_SIZE = 1024

"""
    gpu_workgroups(device) -> Int

Number of workgroups the strided kernels are launched with on `device`: one
per streaming multiprocessor. Fewer leave multiprocessors idle, more only
add workgroups to schedule. Defined by the device extensions.
"""
function gpu_workgroups end

"""
    bounded_floor_int32(x) -> Int32

Return ``floor(x)`` as `Int32` wherever that conversion is exact, and -1
elsewhere (magnitudes of at least ``2^31``, infinities and NaN).

The decision is taken from the bits rather than by comparing floats, which
a GPU evaluates ~1.5x faster: the biased exponent is below ``1023 + 31``
exactly when the magnitude is below ``2^31``. The conversion of any other
value is computed but never selected.
"""
@inline function bounded_floor_int32(x::Float64)::Int32
    exponent = (reinterpret(UInt64, x) >> 52) & 0x00000000000007ff
    return ifelse(
        exponent < 0x000000000000041e, unsafe_trunc(Int32, floor(x)), Int32(-1)
    )
end

"""
    interpolation_interval(layout, position, n_intervals) -> Integer

Zero-based interval of `position` in ``[0, n_intervals)``, or -1.

Strided GPU kernels decide with [`bounded_floor_int32`] and integer
comparisons, which a GPU evaluates faster; the CPU keeps the float
comparisons, which it evaluates faster.
"""
@inline function interpolation_interval(
    ::StrideLayout, position::Float64, n_intervals::Integer
)::Int32
    interval = bounded_floor_int32(position)
    inside = (interval >= 0) & (interval < n_intervals)
    return ifelse(inside, interval, Int32(-1))
end

@inline function interpolation_interval(
    ::SingleParticleLayout, position::Float64, n_intervals::Integer
)::Int
    inside = (position >= 0.0) & (position < n_intervals)
    return ifelse(inside, unsafe_trunc(Int, position), -1)
end

@kernel function kick_single_harmonic_kernel!(
    dt, dE, voltage_kick, omega_rf, phi_rf, acceleration_kick, layout
)
    item = @index(Global, Linear)
    @inbounds for i in particle_indices(layout, item)
        dE[i] +=
            voltage_kick * sin(omega_rf * dt[i] + phi_rf) + acceleration_kick
    end
end

@kernel function kick_multi_harmonic_kernel!(
    dt, dE, voltage, omega_rf, phi_rf, n_rf, charge, acceleration_kick, layout
)
    item = @index(Global, Linear)
    @inbounds for i in particle_indices(layout, item)
        dt_i = dt[i]
        accumulator = dE[i]
        for j in 1:n_rf
            accumulator +=
                (charge * voltage[j]) * sin(omega_rf[j] * dt_i + phi_rf[j])
        end
        dE[i] = accumulator + acceleration_kick
    end
end

"""
    MAX_SPECIALISED_HARMONICS

Most RF harmonics for which the GPU multi-harmonic kick has a kernel
specialised on their number, see [`kick_specialised_harmonics_kernel!`].
"""
const MAX_SPECIALISED_HARMONICS = 4

# Multi-harmonic kick with the harmonics read once per work-item instead of
# once per particle and harmonic, and the loop over them unrolled: 1.03x
# as fast as the cuda backend for two harmonics (1e6-1e8 particles, T400),
# against 0.92x for `kick_multi_harmonic_kernel!`. Same sine and summation
# order, so the result is bitwise identical. `fast_sin` is no help on a
# GPU: it is ~1.4x slower there than the libdevice sine.
@kernel function kick_specialised_harmonics_kernel!(
    dt, dE, voltage, omega_rf, phi_rf, ::Val{N}, charge, acceleration_kick,
    layout,
) where {N}
    item = @index(Global, Linear)
    @inbounds begin
        amplitudes = ntuple(j -> charge * voltage[j], Val(N))
        harmonic_omegas = ntuple(j -> omega_rf[j], Val(N))
        harmonic_phases = ntuple(j -> phi_rf[j], Val(N))
        for i in particle_indices(layout, item)
            dt_i = dt[i]
            accumulator = dE[i]
            for j in 1:N
                accumulator +=
                    amplitudes[j] *
                    sin(harmonic_omegas[j] * dt_i + harmonic_phases[j])
            end
            dE[i] = accumulator + acceleration_kick
        end
    end
end

@kernel function drift_simple_kernel!(dt, dE, coefficient, layout)
    item = @index(Global, Linear)
    @inbounds for i in particle_indices(layout, item)
        dt[i] += coefficient * dE[i]
    end
end

@kernel function drift_exact_kernel!(
    dt,
    dE,
    drift_time,
    alpha_0,
    higher_alpha,
    n_alpha,
    inverse_beta_squared,
    inverse_energy,
    layout,
)
    item = @index(Global, Linear)
    inverse_energy_squared = inverse_energy * inverse_energy
    @inbounds for i in particle_indices(layout, item)
        energy_offset = dE[i]
        beam_delta =
            sqrt(
                1.0 +
                inverse_beta_squared * (
                    energy_offset * energy_offset * inverse_energy_squared +
                    2.0 * energy_offset * inverse_energy
                ),
            ) - 1.0
        polynomial = 1.0 + alpha_0 * beam_delta
        delta_power = beam_delta * beam_delta
        for k in 1:n_alpha
            polynomial += higher_alpha[k] * delta_power
            delta_power *= beam_delta
        end
        dt[i] +=
            drift_time * (
                polynomial * (1.0 + energy_offset * inverse_energy) /
                (1.0 + beam_delta) - 1.0
            )
    end
end

@kernel function loss_box_kernel!(
    dt, dE, flags, e_max, e_min, t_min, t_max, lost_flag, layout
)
    item = @index(Global, Linear)
    @inbounds for i in particle_indices(layout, item)
        is_lost =
            (dE[i] > e_max) ||
            (dE[i] < e_min) ||
            (dt[i] < t_min) ||
            (dt[i] > t_max)
        if is_lost
            flags[i] = lost_flag
        end
    end
end

# Histogram on a GPU with workgroup-private bins.
#
# Every work-item throwing its value straight at the global histogram makes
# all of them contend on the same few atomic locations (a 1000-bin
# histogram of 1e6 particles is ~10x slower than the cpp/cuda kernels that
# way). Instead each workgroup accumulates into a private copy of the
# bins in workgroup-local (shared) memory, and only the per-workgroup
# partial sums are added to the global histogram.
#
# As in the other strided GPU kernels (see "Particle layouts"), one
# workgroup runs per streaming multiprocessor and every work-item strides
# through the input by the number of work-items.
#
# Local memory is a compile-time constant of the kernel, so histograms
# with more bins than `HISTOGRAM_LOCAL_BINS` are built in several passes
# over the input, each pass covering the next window of bins.
#
# The CPU counts per thread instead, see `count_histogram_slice!`.

"""
    HISTOGRAM_LOCAL_BINS

Number of private bins per workgroup. `Int32` counts of 16 KiB fit into
the workgroup-local memory of every supported GPU generation with room
to spare for other kernels resident on the same multiprocessor.
"""
const HISTOGRAM_LOCAL_BINS = 4096

"""
    histogram_bin(value, start, stop, inverse_bin_width, n_bins) -> Int32

Return the zero-based bin of `value`, or -1 if it falls into no bin, with
the bins of [`histogram_slot`]. The bin is found with
[`bounded_floor_int32`] and integer comparisons, which a GPU evaluates
faster than float comparisons.
"""
@inline function histogram_bin(
    value::Float64,
    start::Float64,
    stop::Float64,
    inverse_bin_width::Float64,
    n_bins::Int32,
)::Int32
    bin = bounded_floor_int32((value - start) * inverse_bin_width)
    if value == stop
        bin = n_bins - Int32(1)
    end
    if bin < Int32(0) || bin >= n_bins
        bin = Int32(-1)
    end
    return bin
end

@kernel function histogram_kernel!(
    array_read,
    array_write,
    n_read,
    n_bins,
    start,
    stop,
    inverse_bin_width,
    bin_offset,
    n_local_bins,
    n_work_items,
)
    workgroup_size = @uniform @groupsize()[1]
    local_counts = @localmem Int32 (HISTOGRAM_LOCAL_BINS,)

    # Phase 1: clear the private bins of this workgroup.
    zero_index = @index(Local, Linear)
    while zero_index <= n_local_bins
        @inbounds local_counts[zero_index] = Int32(0)
        zero_index += workgroup_size
    end

    @synchronize

    # Phase 2: stride through the input and count the values of the current
    # pass window ``[bin_offset, bin_offset + n_local_bins)`` into the
    # private bins. A value in no bin (-1) lands on a local bin below 1.
    # The input is indexed with the type of `n_read`, `Int32` whenever it
    # fits, which a GPU steps through faster than `Int64`.
    value_index = oftype(n_read, @index(Global, Linear))
    while value_index <= n_read
        @inbounds value = array_read[value_index]
        bin = histogram_bin(value, start, stop, inverse_bin_width, n_bins)
        local_bin = bin - bin_offset + Int32(1)
        if Int32(1) <= local_bin <= n_local_bins
            @inbounds Atomix.@atomic local_counts[local_bin] += Int32(1)
        end
        value_index += n_work_items
    end

    @synchronize

    # Phase 3: add the non-empty private bins to the global histogram.
    flush_index = @index(Local, Linear)
    while flush_index <= n_local_bins
        @inbounds count = local_counts[flush_index]
        if count != Int32(0)
            global_bin = bin_offset + flush_index
            @inbounds Atomix.@atomic array_write[global_bin] += Float64(count)
        end
        flush_index += workgroup_size
    end
end

@kernel function histogram_sparse_kernel!(
    x,
    out,
    first_left_cut,
    left_cut_distance,
    cut_width,
    bins_per_profile,
    filling_pattern,
    n_buckets,
    bucket_index_to_memory_index,
    inverse_profile_distance,
    inverse_bin_step,
)
    i = @index(Global, Linear)
    @inbounds begin
        value = x[i]
        bucket_float =
            floor((value - first_left_cut) * inverse_profile_distance)
        if bucket_float >= 0.0 && bucket_float < n_buckets
            bucket_index = unsafe_trunc(Int, bucket_float)
            if filling_pattern[bucket_index + 1]
                start_location =
                    first_left_cut + bucket_index * left_cut_distance
                stop_location = start_location + cut_width
                memory_offset =
                    Int(bucket_index_to_memory_index[bucket_index + 1])
                if value == stop_location
                    write_index = memory_offset + bins_per_profile
                    Atomix.@atomic out[write_index] += 1.0
                elseif value >= start_location && value < stop_location
                    bin_float =
                        floor((value - start_location) * inverse_bin_step)
                    if bin_float >= 0.0 && bin_float < bins_per_profile
                        write_index =
                            memory_offset +
                            unsafe_trunc(Int, bin_float) + 1
                        Atomix.@atomic out[write_index] += 1.0
                    end
                end
            end
        end
    end
end

# Beam phase on a GPU. The trapezoid weights are applied inside the
# kernels, so the host needs one reduction and one synchronisation instead
# of three, and the sine and cosine share one range reduction
# (`fast_sin_cos`). Against the cuda backend (T400): 256 bins 23 against
# 42 µs, 1048576 bins 2.68 against 3.47 ms.

@kernel function beam_phase_values_kernel!(
    values, hist_x, hist_y, n_bins, alpha, omega_rf, phi_rf
)
    i = @index(Global, Linear)
    @inbounds begin
        x = hist_x[i]
        is_end_point = (i == 1) | (i == n_bins)
        weight =
            ifelse(is_end_point, 0.5, 1.0) * (exp(alpha * x) * hist_y[i])
        sine, cosine = fast_sin_cos(omega_rf * x + phi_rf)
        # Real part carries the cosine, imaginary part the sine integrand,
        # so that a single reduction yields both trapezoid coefficients.
        values[i] = complex(weight * cosine, weight * sine)
    end
end

# Large profiles: every work-item strides through the bins, summing its
# integrands privately; the first work-item of each workgroup then adds its
# workgroup's partial sums from workgroup-local memory. The host reads two
# numbers per workgroup with one copy; no atomics, no profile-sized array.
@kernel function beam_phase_workgroup_sums_kernel!(
    workgroup_sums, hist_x, hist_y, n_bins, n_work_items, alpha, omega_rf,
    phi_rf,
)
    workgroup_size = @uniform @groupsize()[1]
    local_sums = @localmem Float64 (2 * GPU_WORKGROUP_SIZE,)

    local_item = @index(Local, Linear)
    sine_sum = 0.0
    cosine_sum = 0.0
    bin = oftype(n_bins, @index(Global, Linear))
    @inbounds while bin <= n_bins
        x = hist_x[bin]
        is_end_point = (bin == 1) | (bin == n_bins)
        weight =
            ifelse(is_end_point, 0.5, 1.0) * (exp(alpha * x) * hist_y[bin])
        sine, cosine = fast_sin_cos(omega_rf * x + phi_rf)
        sine_sum += weight * sine
        cosine_sum += weight * cosine
        bin += n_work_items
    end
    @inbounds local_sums[2 * local_item - 1] = sine_sum
    @inbounds local_sums[2 * local_item] = cosine_sum

    @synchronize

    if @index(Local, Linear) == 1
        workgroup = @index(Group, Linear)
        workgroup_sine = 0.0
        workgroup_cosine = 0.0
        @inbounds for slot in 1:workgroup_size
            workgroup_sine += local_sums[2 * slot - 1]
            workgroup_cosine += local_sums[2 * slot]
        end
        @inbounds workgroup_sums[2 * workgroup - 1] = workgroup_sine
        @inbounds workgroup_sums[2 * workgroup] = workgroup_cosine
    end
end

# Small GPU histograms on one workgroup. Its work-items own the private bins
# exclusively, so the flush *assigns* every bin of the pass window: no
# zeroing pass (`fill!`) and no atomic writes to the output. With only
# `GPU_WORKGROUP_SIZE` strides the counting contends more, so this pays
# only for small inputs, see `HISTOGRAM_SINGLE_WORKGROUP_MAX_VALUES`.
@kernel function histogram_assign_kernel!(
    array_read, array_write, n_read, n_bins, start, stop, inverse_bin_width,
    bin_offset, n_local_bins, n_work_items,
)
    workgroup_size = @uniform @groupsize()[1]
    local_counts = @localmem Int32 (HISTOGRAM_LOCAL_BINS,)

    zero_index = @index(Local, Linear)
    while zero_index <= n_local_bins
        @inbounds local_counts[zero_index] = Int32(0)
        zero_index += workgroup_size
    end

    @synchronize

    value_index = oftype(n_read, @index(Global, Linear))
    while value_index <= n_read
        @inbounds value = array_read[value_index]
        bin = histogram_bin(value, start, stop, inverse_bin_width, n_bins)
        local_bin = bin - bin_offset + Int32(1)
        if Int32(1) <= local_bin <= n_local_bins
            @inbounds Atomix.@atomic local_counts[local_bin] += Int32(1)
        end
        value_index += n_work_items
    end

    @synchronize

    flush_index = @index(Local, Linear)
    while flush_index <= n_local_bins
        @inbounds array_write[bin_offset + flush_index] =
            Float64(local_counts[flush_index])
        flush_index += workgroup_size
    end
end

# Interpolated kicks in two phases.
#
# Within one bin the kick is linear in `dt`: dE += dt * slope + offset.
# Phase 1 computes `slope` and `offset` once per bin, into
# `factors = [slope_1, offset_1, slope_2, offset_2, ...]`; phase 2 only
# locates each particle's bin and applies them. Recomputing the factors
# for every particle (the previous kernels) costs four extra memory reads,
# a division and several products per particle -- about 2x on a GPU. The
# floating-point operations per particle are unchanged, so the result is
# bit-identical.
#
# The range checks test the bin position `x` itself: `x >= 0 && x < n` is
# exactly `floor(x) >= 0 && floor(x) < n` (NaN fails both), and truncation
# equals `floor` for `x >= 0`, so no per-particle `floor` is needed.

"""
    store_interpolated_kick_factors!(factors, bin, voltage, bin_centers,
                                     charge, acceleration_kick,
                                     inverse_bin_width)

Write the slope and offset of the linear kick within `bin` to `factors`.
"""
@inline function store_interpolated_kick_factors!(
    factors,
    bin,
    voltage,
    bin_centers,
    charge,
    acceleration_kick,
    inverse_bin_width,
)
    @inbounds begin
        slope =
            charge * (voltage[bin + 1] - voltage[bin]) * inverse_bin_width
        factors[2 * bin - 1] = slope
        factors[2 * bin] =
            (charge * voltage[bin] - bin_centers[bin] * slope) +
            acceleration_kick
    end
    return nothing
end

@kernel function kick_interpolated_dense_factors_kernel!(
    factors, grid, voltage, bin_centers, n_slices, charge, acceleration_kick
)
    bin = @index(Global, Linear)
    @inbounds begin
        inverse_bin_width =
            (n_slices - 1) / (bin_centers[n_slices] - bin_centers[1])
        store_interpolated_kick_factors!(
            factors,
            bin,
            voltage,
            bin_centers,
            charge,
            acceleration_kick,
            inverse_bin_width,
        )
        # The particle phase reads the grid from the device, so that no
        # scalar has to be copied to the host.
        if bin == 1
            grid[1] = bin_centers[1]
            grid[2] = inverse_bin_width
        end
    end
end

@kernel function kick_interpolated_dense_particles_kernel!(
    dt, dE, factors, grid, n_slices, layout
)
    item = @index(Global, Linear)
    @inbounds begin
        # The grid is read from the device once per work-item.
        first_bin_center = grid[1]
        inverse_bin_width = grid[2]
        for i in particle_indices(layout, item)
            dt_i = dt[i]
            interval = interpolation_interval(
                layout, (dt_i - first_bin_center) * inverse_bin_width,
                n_slices - 1,
            )
            if interval >= 0
                dE[i] +=
                    dt_i * factors[2 * interval + 1] +
                    factors[2 * interval + 2]
            end
        end
    end
end

@kernel function kick_interpolated_sparse_factors_kernel!(
    factors, voltage, bin_centers, charge, acceleration_kick, inverse_bin_width
)
    bin = @index(Global, Linear)
    store_interpolated_kick_factors!(
        factors,
        bin,
        voltage,
        bin_centers,
        charge,
        acceleration_kick,
        inverse_bin_width,
    )
end

@kernel function kick_interpolated_sparse_particles_kernel!(
    dt,
    dE,
    factors,
    first_left_cut,
    left_cut_distance,
    bins_per_profile,
    filling_pattern,
    n_buckets,
    bucket_index_to_memory_index,
    inverse_histogram_distance,
    inverse_bin_width,
    bin_width,
    layout,
)
    item = @index(Global, Linear)
    @inbounds for i in particle_indices(layout, item)
        dt_i = dt[i]
        bucket_position = (dt_i - first_left_cut) * inverse_histogram_distance
        if bucket_position >= 0.0 && bucket_position < n_buckets
            bucket_index = unsafe_trunc(Int, bucket_position)
            if filling_pattern[bucket_index + 1]
                cut_left =
                    first_left_cut + bucket_index * left_cut_distance
                bucket_bin_center0 = cut_left + bin_width / 2.0
                local_bin_position =
                    (dt_i - bucket_bin_center0) * inverse_bin_width
                if local_bin_position >= 0.0 &&
                   local_bin_position < bins_per_profile - 1
                    bin =
                        Int(
                            bucket_index_to_memory_index[bucket_index + 1]
                        ) + unsafe_trunc(Int, local_bin_position) + 1
                    dE[i] += dt_i * factors[2 * bin - 1] + factors[2 * bin]
                end
            end
        end
    end
end

@kernel function move_flagged_scatter_kernel!(
    flags,
    dt,
    dE,
    ids,
    flags_source,
    dt_source,
    dE_source,
    ids_source,
    keep_mask,
    kept_positions,
    n_kept,
)
    i = @index(Global, Linear)
    @inbounds begin
        position = Int(kept_positions[i])
        destination = keep_mask[i] == 1 ? position : n_kept + i - position
        flags[destination] = flags_source[i]
        dt[destination] = dt_source[i]
        dE[destination] = dE_source[i]
        ids[destination] = ids_source[i]
    end
end

@kernel function wake_from_pole_residue_kernel!(
    voltage,
    states,
    profile,
    profile_dts,
    poles,
    residues,
    counterrotating_pole_signs,
    update_on_bin,
    n_bins,
    n_updates,
    n_poles,
    is_counterrotating_beam,
    factor,
    two_factor,
)
    pole_i = @index(Global, Linear)
    @inbounds begin
        # `counterrotating_flip` is intentionally applied to BOTH the state
        # injection and the output amplitude: for the counter-rotating
        # beam's own wake the two factors cancel (flip**2 == 1); only
        # contributions of the other beam, accumulated in the shared
        # `states`, see a net sign flip.
        counterrotating_flip = 1.0
        if is_counterrotating_beam &&
           counterrotating_pole_signs[pole_i] == -1
            counterrotating_flip = -1.0
        end

        pole = poles[pole_i]
        residue = residues[pole_i]
        state = states[pole_i]
        # The last entry of `states` carries the end time of the previous
        # call, not a pole state.
        t_start = states[n_poles + 1]

        # A real pole has no implicit complex conjugate (vector-fitting
        # convention): only a pole with imag != 0 stands in for an
        # unstored conjugate partner and needs the doubled injection.
        injection_factor = imag(pole) == 0 ? factor : two_factor

        i_update = 0
        # An empty `update_on_bin` means "never update"; `decay` stays 0.
        update_bin = n_updates > 0 ? Int(update_on_bin[1]) : -1
        decay = zero(ComplexF64)

        for bin_i in 0:(n_bins - 1)
            profile_half =
                counterrotating_flip * 0.5 * profile[bin_i + 1] *
                injection_factor
            if bin_i == update_bin
                time_jump = if bin_i == 0
                    profile_dts[1] - t_start
                else
                    complex(profile_dts[bin_i + 1] - profile_dts[bin_i])
                end
                state *= exp(pole * time_jump)
                bin_time = profile_dts[bin_i + 2] - profile_dts[bin_i + 1]
                decay = exp(pole * bin_time)
                i_update += 1
                if i_update < n_updates
                    update_bin = Int(update_on_bin[i_update + 1])
                end
            else
                state *= decay
            end
            state += profile_half
            amplitude = counterrotating_flip * real(residue * state)
            Atomix.@atomic voltage[bin_i + 1] += amplitude
            state += profile_half
        end
        states[pole_i] = state
    end
end

@kernel function wake_store_end_time_kernel!(
    states, profile_dts, n_poles, n_profile_dts
)
    i = @index(Global, Linear)
    @inbounds if i == 1
        states[n_poles + 1] = complex(profile_dts[n_profile_dts])
    end
end

@kernel function synchrotron_radiation_kernel!(
    dE, damping_factor, energy_lost, layout
)
    item = @index(Global, Linear)
    @inbounds for i in particle_indices(layout, item)
        dE[i] = damping_factor * dE[i] - energy_lost
    end
end

# Quantum excitation on a GPU uses both normals of one Philox block:
# work-item `item` handles the particles item, item + stride, item +
# 2 stride, ... of its layout, and pairs them up as (item, item + stride),
# (item + 2 stride, item + 3 stride), ... . A pair draws the block whose
# counter is its first particle, so every counter is used by exactly one
# pair and no two particles share a deviate. A lone last particle takes the
# first normal of its own block. Drawing one block per particle instead,
# as before, made the kernel 0.58x as fast as the cuda backend; paired,
# it is 1.02x (1e5-1e8 particles, T400). The loop bound
# `i <= n - stride` cannot overflow, since the launcher guarantees
# `n + stride <= typemax`.
#
# Only the strided GPU layout is supported: the CPU draws its noise with
# `synchrotron_radiation_quantum_excitation_range!`.
@kernel function synchrotron_radiation_quantum_excitation_kernel!(
    dE, damping_factor, noise_scale, energy_lost, key, layout::StrideLayout
)
    item = @index(Global, Linear)
    n_particles = layout.n_particles
    stride = layout.stride
    i = oftype(n_particles, item)
    @inbounds while i <= n_particles - stride
        normal_1, normal_2 = philox_standard_normal_pair(Int(i), key)
        partner = i + stride
        dE[i] =
            damping_factor * dE[i] + (normal_1 * noise_scale - energy_lost)
        dE[partner] =
            damping_factor * dE[partner] +
            (normal_2 * noise_scale - energy_lost)
        i = partner + stride
    end
    @inbounds if i <= n_particles
        normal_1, _ = philox_standard_normal_pair(Int(i), key)
        dE[i] =
            damping_factor * dE[i] + (normal_1 * noise_scale - energy_lost)
    end
end

# Chunked CPU particle loops.
#
# The CPU backend of KernelAbstractions evaluates a kernel one work-item
# at a time, which keeps the compiler from vectorising the per-particle
# arithmetic. On the CPU the loops below therefore run over whole chunks
# of particles instead, each chunk swept in a plain `@simd` loop. Every
# range function takes the arguments of its per-particle kernel (after the
# particle range) and performs exactly the same floating-point operations
# in the same order, except that the kicks use `fast_sin`, as the C++
# backend does.
#
# The chunks are distributed with Polyester's `@batch` rather than with
# KernelAbstractions (see [`sweep_chunks!`]).

"""
    PARTICLES_PER_CHUNK

Macro-particles one work-item of a chunked CPU particle loop sweeps. The
`dt` and `dE` of a chunk (16 KiB) fit into the L1 cache; 1024 is the
measured optimum for ``1e7`` particles on 12 threads (multi-harmonic kick
10.1 ms, against 11.9 ms for 4096 and 20.6 ms for 65536 particles).
"""
const PARTICLES_PER_CHUNK = 1024

"""
    sweep_chunks!(range_function!, n_elements, arguments)

Call ``range_function!(first, last, arguments...)`` for every chunk of
`PARTICLES_PER_CHUNK` of `n_elements` elements, the chunks in parallel.

The chunks run on the thread pool of `thread_pool.jl`, which keeps one
worker per hardware thread: its workers claim chunks one at a time, spin
briefly after a loop and then sleep, so that back-to-back calls start
without waking anyone and a busy machine cannot stall a call. A single
chunk runs on the calling thread.
"""
function sweep_chunks!(range_function!, n_elements::Int, arguments::Tuple)
    run_chunks!(range_function!, n_elements, PARTICLES_PER_CHUNK, arguments)
    return nothing
end

function kick_single_harmonic_range!(
    first_particle, last_particle,
    dt, dE, voltage_kick, omega_rf, phi_rf, acceleration_kick,
)
    @inbounds @simd for i in first_particle:last_particle
        dE[i] +=
            voltage_kick * fast_sin(omega_rf * dt[i] + phi_rf) +
            acceleration_kick
    end
    return nothing
end

function kick_multi_harmonic_range!(
    first_particle, last_particle,
    dt, dE, voltage, omega_rf, phi_rf, n_rf, charge, acceleration_kick,
)
    # One sweep per harmonic keeps the per-particle summation order
    # dE + kick_1 + ... + kick_n_rf + acceleration_kick.
    @inbounds for j in 1:n_rf
        amplitude = charge * voltage[j]
        harmonic_omega_rf = omega_rf[j]
        harmonic_phi_rf = phi_rf[j]
        @simd for i in first_particle:last_particle
            dE[i] +=
                amplitude * fast_sin(harmonic_omega_rf * dt[i] + harmonic_phi_rf)
        end
    end
    @inbounds @simd for i in first_particle:last_particle
        dE[i] += acceleration_kick
    end
    return nothing
end

"""
    MAX_UNROLLED_LENGTH

Longest array [`with_unrolled`] turns into a tuple.
"""
const MAX_UNROLLED_LENGTH = 4

"""
    with_unrolled(f, n_values, arrays...)

Call `f` with the first `n_values` elements of each of `arrays` as a
tuple, whose length is then known at compile time, so that a loop over it
is unrolled; with more than `MAX_UNROLLED_LENGTH` elements call
`f(arrays...)` instead.

Every length has its own branch, so the call is type-stable: no runtime
dispatch on the length, and one compiled `f` per length.
"""
@inline function with_unrolled(
    f::F, n_values::Int, arrays::Vararg{Any, N}
) where {F, N}
    n_values == 0 && return f(ntuple(_ -> (), Val(N))...)
    n_values == 1 && return f(unrolled_prefixes(arrays, Val(1))...)
    n_values == 2 && return f(unrolled_prefixes(arrays, Val(2))...)
    n_values == 3 && return f(unrolled_prefixes(arrays, Val(3))...)
    n_values == 4 && return f(unrolled_prefixes(arrays, Val(4))...)
    return f(arrays...)
end

"""
    unrolled_prefixes(arrays, Val(length)) -> Tuple

The first `length` elements of each of `arrays`, each as a tuple.
"""
@inline function unrolled_prefixes(arrays::Tuple, ::Val{L}) where {L}
    return map(array -> ntuple(j -> @inbounds(array[j]), Val(L)), arrays)
end

function kick_multi_harmonic_unrolled_range!(
    first_particle, last_particle,
    dt, dE, amplitudes::NTuple{N, Float64}, omega_rf::NTuple{N, Float64},
    phi_rf::NTuple{N, Float64}, acceleration_kick,
) where {N}
    # The harmonics are a tuple, so the inner loop is unrolled and the
    # particles are swept once, keeping the summation order
    # dE + kick_1 + ... + kick_N + acceleration_kick. Two harmonics, 1e5
    # particles on one thread: 2.9 against 3.1 ns per particle for one
    # sweep per harmonic.
    @inbounds @simd for i in first_particle:last_particle
        dt_i = dt[i]
        accumulator = dE[i]
        for j in 1:N
            accumulator += amplitudes[j] * fast_sin(omega_rf[j] * dt_i + phi_rf[j])
        end
        dE[i] = accumulator + acceleration_kick
    end
    return nothing
end

function drift_simple_range!(first_particle, last_particle, dt, dE, coefficient)
    @inbounds @simd for i in first_particle:last_particle
        dt[i] += coefficient * dE[i]
    end
    return nothing
end

function drift_exact_range!(
    first_particle, last_particle,
    dt, dE, drift_time, alpha_0, higher_alpha, n_alpha,
    inverse_beta_squared, inverse_energy,
)
    inverse_energy_squared = inverse_energy * inverse_energy
    @inbounds @simd for i in first_particle:last_particle
        energy_offset = dE[i]
        # `Base.sqrt` throws for a negative argument, and that check keeps
        # the loop from vectorising (1e5 particles on one thread: 3.9
        # against 1.1 ns per particle). Unchecked, a negative argument
        # yields NaN, as in NumPy.
        beam_delta =
            @fastmath(sqrt)(
                1.0 +
                inverse_beta_squared * (
                    energy_offset * energy_offset * inverse_energy_squared +
                    2.0 * energy_offset * inverse_energy
                ),
            ) - 1.0
        polynomial = 1.0 + alpha_0 * beam_delta
        delta_power = beam_delta * beam_delta
        # `higher_alpha` holds exactly `n_alpha` factors. For a tuple (see
        # `apply_drift_exact!`) its length is a compile-time constant, so
        # this loop is unrolled.
        for k in 1:length(higher_alpha)
            polynomial += higher_alpha[k] * delta_power
            delta_power *= beam_delta
        end
        dt[i] +=
            drift_time * (
                polynomial * (1.0 + energy_offset * inverse_energy) /
                (1.0 + beam_delta) - 1.0
            )
    end
    return nothing
end

"""
    @novectorize for ... end

Keep LLVM from vectorising the annotated loop.
"""
macro novectorize(loop)
    push!(
        loop.args[2].args,
        Expr(:loopinfo, (Symbol("llvm.loop.vectorize.enable"), false)),
    )
    return esc(loop)
end

# Loss box on the CPU, in two forms, see `LOSS_BOX_SELECT_MAX_PARTICLES`.

function loss_box_select_range!(
    first_particle, last_particle,
    dt, dE, flags, e_max, e_min, t_min, t_max, lost_flag,
)
    # Non-short-circuiting comparisons and a select that writes back every
    # flag, so that the loop vectorises without masked stores.
    @inbounds @simd for i in first_particle:last_particle
        is_lost =
            (dE[i] > e_max) | (dE[i] < e_min) | (dt[i] < t_min) |
            (dt[i] > t_max)
        flags[i] = ifelse(is_lost, lost_flag, flags[i])
    end
    return nothing
end

function loss_box_range!(
    first_particle, last_particle,
    dt, dE, flags, e_max, e_min, t_min, t_max, lost_flag,
)
    # A scalar branch that stores only the flags of lost particles, as the
    # C++ backend does. Left to LLVM, this loop becomes a vectorised
    # masked store, which is twice as slow on flags that NumPy allocated
    # with `np.zeros`: 1e6 particles on six threads took 591 µs vectorised
    # and 243 µs scalar.
    @inbounds @novectorize for i in first_particle:last_particle
        if (dE[i] > e_max) || (dE[i] < e_min) || (dt[i] < t_min) ||
           (dt[i] > t_max)
            flags[i] = lost_flag
        end
    end
    return nothing
end

function kick_interpolated_dense_range!(
    first_particle, last_particle,
    dt, dE, factors, first_bin_center, inverse_bin_width, n_intervals,
)
    # A plain branch: the gathers of the factors keep a branch-free loop
    # from vectorising, and as a scalar loop the select is slower (1e5
    # particles on one thread: 0.81 ns per particle with the branch, 1.61
    # with the select).
    @inbounds for i in first_particle:last_particle
        dt_i = dt[i]
        position = (dt_i - first_bin_center) * inverse_bin_width
        if position >= 0.0 && position < n_intervals
            interval = unsafe_trunc(Int, position)
            dE[i] +=
                dt_i * factors[2 * interval + 1] + factors[2 * interval + 2]
        end
    end
    return nothing
end

function kick_interpolated_sparse_range!(
    first_particle, last_particle,
    dt, dE, factors, first_left_cut, left_cut_distance, bins_per_profile,
    filling_pattern, n_buckets, bucket_index_to_memory_index,
    inverse_histogram_distance, inverse_bin_width, bin_width,
)
    @inbounds for i in first_particle:last_particle
        dt_i = dt[i]
        bucket_position = (dt_i - first_left_cut) * inverse_histogram_distance
        if bucket_position >= 0.0 && bucket_position < n_buckets
            bucket_index = unsafe_trunc(Int, bucket_position)
            if filling_pattern[bucket_index + 1]
                cut_left = first_left_cut + bucket_index * left_cut_distance
                bucket_bin_center0 = cut_left + bin_width / 2.0
                local_bin_position =
                    (dt_i - bucket_bin_center0) * inverse_bin_width
                if local_bin_position >= 0.0 &&
                   local_bin_position < bins_per_profile - 1
                    bin =
                        Int(bucket_index_to_memory_index[bucket_index + 1]) +
                        unsafe_trunc(Int, local_bin_position) + 1
                    dE[i] += dt_i * factors[2 * bin - 1] + factors[2 * bin]
                end
            end
        end
    end
    return nothing
end

function synchrotron_radiation_range!(
    first_particle, last_particle, dE, damping_factor, energy_lost
)
    @inbounds @simd for i in first_particle:last_particle
        dE[i] = damping_factor * dE[i] - energy_lost
    end
    return nothing
end

"""
    beam_phase_integrand_sums(hist_x, hist_y, alpha, omega_rf, phi_rf,
                              first_bin, last_bin) -> (sine_sum, cosine_sum)

Sum the sine- and cosine-weighted beam-phase integrands over a bin range.
"""
@inline function beam_phase_integrand_sums(
    hist_x, hist_y, alpha, omega_rf, phi_rf, first_bin, last_bin
)
    # `fast_exp` and `fast_sin_cos` instead of the library functions, so
    # that the loop vectorises: 65536 bins on one thread take 196 instead
    # of 826 µs. The sums may be reassociated, which moves them by ~1e-14
    # relative for profiles whose integrands cancel.
    sine_sum = 0.0
    cosine_sum = 0.0
    @inbounds @simd for i in first_bin:last_bin
        x = hist_x[i]
        weight = fast_exp(alpha * x) * hist_y[i]
        sine, cosine = fast_sin_cos(omega_rf * x + phi_rf)
        sine_sum += weight * sine
        cosine_sum += weight * cosine
    end
    return sine_sum, cosine_sum
end

function beam_phase_sums_range!(
    first_bin, last_bin, chunk_sums, hist_x, hist_y, alpha, omega_rf, phi_rf
)
    # Every chunk writes only its own two slots, so the chunks need no
    # synchronisation. exp, sin and cos do not vectorise; the gain of the
    # chunks is running them in parallel.
    chunk = (first_bin - 1) ÷ PARTICLES_PER_CHUNK + 1
    sine_sum, cosine_sum = beam_phase_integrand_sums(
        hist_x, hist_y, alpha, omega_rf, phi_rf, first_bin, last_bin
    )
    @inbounds chunk_sums[2 * chunk - 1] = sine_sum
    @inbounds chunk_sums[2 * chunk] = cosine_sum
    return nothing
end

"""
    HISTOGRAM_VALUES_PER_THREAD

Values that justify one more thread in the CPU histogram. Waking a
sleeping Julia thread costs about as much as counting this many values on
one: on 12 threads, histograms of ``3e3``/``1e4``/``3e4`` values took
237/144/181 µs on all threads but 8.7/13/27 µs on one, breaking even at
``1e5`` (72 µs either way) and paying off from ``3e5`` (151 against 199 µs).
"""
const HISTOGRAM_VALUES_PER_THREAD = 100_000

"""
    histogram_slice_count(device, n_read) -> Int

Number of slices, each counted by one thread, the CPU histogram splits
`n_read` values into: one per `HISTOGRAM_VALUES_PER_THREAD` values, at least
one and at most one per thread.
"""
function histogram_slice_count(device::CPU, n_read::Int)::Int
    return clamp(
        cld(n_read, HISTOGRAM_VALUES_PER_THREAD), 1, max_threads(device)
    )
end

"""
    histogram_slot(value, start, stop, inverse_bin_width, n_bins) -> Int

Return the slot `value` is counted in: bin `b` is slot `b + 1`, and slot 1
collects every value that falls into no bin.

Bins follow the other backends: ``floor((value - start) *
inverse_bin_width)`` counts if it lies in ``[0, n_bins)``, and values equal
to `stop` go to the last bin. For ``value >= start`` the scaled position is
non-negative, so truncating it is the floor. The slot is chosen without a
branch, so that a loop over the values vectorises; the conversion of an
out-of-range position is computed but never selected.
"""
@inline function histogram_slot(
    value::Float64,
    start::Float64,
    stop::Float64,
    inverse_bin_width::Float64,
    n_bins::Int,
)::Int
    scaled = (value - start) * inverse_bin_width
    inside = (value >= start) & (scaled < n_bins)
    slot = ifelse(inside, unsafe_trunc(Int, scaled) + 2, 1)
    return ifelse(value == stop, n_bins + 1, slot)
end

"""
    count_histogram_slice!(slice_counts, slice, array_read, n_read,
                           values_per_slice, n_bins, start, stop,
                           inverse_bin_width)

Count slice `slice` of `array_read` into its own bins, stored as
`slice_counts[slice]`, so that the slices share no memory while they count.
Within the slice the slots of a block are computed first, in a vectorised
loop, and counted afterwards.
"""
function count_histogram_slice!(
    slice_counts,
    slice,
    array_read,
    n_read,
    values_per_slice,
    n_bins,
    start,
    stop,
    inverse_bin_width,
)
    first_value = (slice - 1) * values_per_slice + 1
    last_value = min(slice * values_per_slice, n_read)
    counts = zeros(Int, n_bins + 1)
    slots = Vector{Int}(undef, PARTICLES_PER_CHUNK)
    block_start = first_value
    @inbounds while block_start <= last_value
        block_length = min(PARTICLES_PER_CHUNK, last_value - block_start + 1)
        @simd for j in 1:block_length
            slots[j] = histogram_slot(
                array_read[block_start + j - 1],
                start,
                stop,
                inverse_bin_width,
                n_bins,
            )
        end
        for j in 1:block_length
            counts[slots[j]] += 1
        end
        block_start += block_length
    end
    slice_counts[slice] = counts
    return nothing
end

@kernel function histogram_slices_kernel!(
    slice_counts,
    array_read,
    n_read,
    values_per_slice,
    n_bins,
    start,
    stop,
    inverse_bin_width,
)
    # One task per slice, see `count_histogram!`.
    slice = @index(Global, Linear)
    count_histogram_slice!(
        slice_counts,
        slice,
        array_read,
        n_read,
        values_per_slice,
        n_bins,
        start,
        stop,
        inverse_bin_width,
    )
end

function synchrotron_radiation_quantum_excitation_range!(
    first_particle, last_particle,
    dE, damping_factor, noise_scale, energy_lost,
)
    # Each task draws from its own task-local generator, so the noise is
    # generated in parallel and independently per chunk. Drawing it
    # particle by particle saves allocating and filling a beam-sized noise
    # array, which costs more than the draws themselves.
    rng = default_rng()
    @inbounds for i in first_particle:last_particle
        dE[i] =
            damping_factor * dE[i] +
            (randn(rng) * noise_scale - energy_lost)
    end
    return nothing
end
