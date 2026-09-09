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

@kernel function kick_single_harmonic_kernel!(
    dt, dE, voltage_kick, omega_rf, phi_rf, acceleration_kick
)
    i = @index(Global, Linear)
    @inbounds dE[i] +=
        voltage_kick * sin(omega_rf * dt[i] + phi_rf) + acceleration_kick
end

@kernel function kick_multi_harmonic_kernel!(
    dt, dE, voltage, omega_rf, phi_rf, n_rf, charge, acceleration_kick
)
    i = @index(Global, Linear)
    @inbounds begin
        dt_i = dt[i]
        accumulator = dE[i]
        for j in 1:n_rf
            accumulator +=
                (charge * voltage[j]) * sin(omega_rf[j] * dt_i + phi_rf[j])
        end
        dE[i] = accumulator + acceleration_kick
    end
end

@kernel function drift_simple_kernel!(dt, dE, coefficient)
    i = @index(Global, Linear)
    @inbounds dt[i] += coefficient * dE[i]
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
)
    i = @index(Global, Linear)
    @inbounds begin
        inverse_energy_squared = inverse_energy * inverse_energy
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
    dt, dE, flags, e_max, e_min, t_min, t_max, lost_flag
)
    i = @index(Global, Linear)
    @inbounds begin
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

# Histogram with workgroup-private bins.
#
# Every work-item throwing its value straight at the global histogram makes
# all of them contend on the same few atomic locations (a 1000-bin
# histogram of 1e6 particles is ~10x slower than the cpp/cuda kernels that
# way). Instead each workgroup accumulates into a private copy of the
# bins in workgroup-local memory (shared memory on GPUs, a stack array on
# the CPU), and only the per-workgroup partial sums are added to the
# global histogram. The global atomic traffic drops from one add per
# particle to one add per (workgroup, non-empty bin).
#
# Local memory is a compile-time constant of the kernel, so histograms
# with more bins than `HISTOGRAM_LOCAL_BINS` are built in several passes
# over the input, each pass covering the next window of bins.

"""
    HISTOGRAM_LOCAL_BINS

Number of private bins per workgroup. `Int32` counts of 16 KiB fit into
the workgroup-local memory of every supported GPU generation with room
to spare for other kernels resident on the same multiprocessor.
"""
const HISTOGRAM_LOCAL_BINS = 4096

"""
    HISTOGRAM_WORKGROUP_SIZE

Work-items per workgroup for the histogram kernel: the largest workgroup
every GPU generation supports, so that the private bins are shared by as
many work-items as possible. On the CPU backend a workgroup is one task,
so this is also the chunk one thread handles between two flushes.
"""
const HISTOGRAM_WORKGROUP_SIZE = 1024

"""
    HISTOGRAM_VALUES_PER_WORK_ITEM

Values each work-item feeds into the private bins before the workgroup
flushes them. Larger values amortise the flush over more particles but
leave fewer workgroups to spread over the device; 32 is the measured
optimum for ``1e6`` particles on both CPU threads and CUDA. A workgroup
never counts more than `typemax(Int32)` values into one private bin.
"""
const HISTOGRAM_VALUES_PER_WORK_ITEM = 32

"""
    histogram_values_per_workgroup() -> Int

Number of input values one workgroup of the histogram kernel consumes.
"""
histogram_values_per_workgroup()::Int =
    HISTOGRAM_WORKGROUP_SIZE * HISTOGRAM_VALUES_PER_WORK_ITEM

"""
    private_bins_need_atomics(device) -> Val

Whether work-items of one workgroup may race on the private bins.

The CPU backend executes the work-items of a workgroup one after the
other on a single task, so plain increments are exact there and the
locked read-modify-write of an atomic would only cost time. On every
other device the work-items run concurrently and need the atomic.
"""
private_bins_need_atomics(::CPU) = Val(false)
private_bins_need_atomics(::Any) = Val(true)

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
    values_per_workgroup,
    ::Val{NEED_ATOMICS},
) where {NEED_ATOMICS}
    workgroup_size = @uniform @groupsize()[1]
    local_counts = @localmem Int32 (HISTOGRAM_LOCAL_BINS,)

    # Phase 1: clear the private bins of this workgroup.
    zero_index = @index(Local, Linear)
    while zero_index <= n_local_bins
        @inbounds local_counts[zero_index] = Int32(0)
        zero_index += workgroup_size
    end

    @synchronize

    # Phase 2: count this workgroup's slice of the input into the
    # private bins. Only bins of the current pass window
    # ``[bin_offset + 1, bin_offset + n_local_bins]`` are counted.
    local_index = @index(Local, Linear)
    group_index = @index(Group, Linear)
    value_index = (group_index - 1) * values_per_workgroup + local_index
    last_value_index = min(group_index * values_per_workgroup, n_read)
    while value_index <= last_value_index
        @inbounds value = array_read[value_index]
        # Out-of-range values map to bin 0, which no window contains.
        bin_index = 0
        if value == stop
            # The right-most edge belongs to the last bin.
            bin_index = n_bins
        else
            bin_float = floor((value - start) * inverse_bin_width)
            # Range-check in floating point: converting an out-of-range
            # `Float64` to `Int` is undefined behaviour.
            if bin_float >= 0.0 && bin_float < n_bins
                bin_index = unsafe_trunc(Int, bin_float) + 1
            end
        end
        local_bin = bin_index - bin_offset
        if 1 <= local_bin <= n_local_bins
            if NEED_ATOMICS
                @inbounds Atomix.@atomic local_counts[local_bin] += Int32(1)
            else
                @inbounds local_counts[local_bin] += Int32(1)
            end
        end
        value_index += workgroup_size
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

@kernel function beam_phase_values_kernel!(
    values, hist_x, hist_y, alpha, omega_rf, phi_rf
)
    i = @index(Global, Linear)
    @inbounds begin
        weight = exp(alpha * hist_x[i]) * hist_y[i]
        angle = omega_rf * hist_x[i] + phi_rf
        # Real part carries the cosine, imaginary part the sine integrand,
        # so that a single reduction yields both trapezoid coefficients.
        values[i] = complex(weight * cos(angle), weight * sin(angle))
    end
end

@kernel function kick_interpolated_dense_kernel!(
    dt, dE, voltage, bin_centers, n_slices, charge, acceleration_kick
)
    i = @index(Global, Linear)
    @inbounds begin
        inverse_bin_width =
            (n_slices - 1) / (bin_centers[n_slices] - bin_centers[1])
        dt_i = dt[i]
        bin_float = floor((dt_i - bin_centers[1]) * inverse_bin_width)
        if bin_float >= 0.0 && bin_float < n_slices - 1
            bin_index = unsafe_trunc(Int, bin_float) + 1
            helper1 =
                charge * (voltage[bin_index + 1] - voltage[bin_index]) *
                inverse_bin_width
            helper2 =
                (
                    charge * voltage[bin_index] -
                    bin_centers[bin_index] * helper1
                ) + acceleration_kick
            dE[i] += dt_i * helper1 + helper2
        end
    end
end

@kernel function kick_interpolated_sparse_kernel!(
    dt,
    dE,
    voltage,
    bin_centers,
    charge,
    acceleration_kick,
    first_left_cut,
    left_cut_distance,
    cut_width,
    bins_per_profile,
    filling_pattern,
    n_buckets,
    bucket_index_to_memory_index,
    inverse_histogram_distance,
    inverse_bin_width,
    bin_width,
)
    i = @index(Global, Linear)
    @inbounds begin
        dt_i = dt[i]
        bucket_float =
            floor((dt_i - first_left_cut) * inverse_histogram_distance)
        if bucket_float >= 0.0 && bucket_float < n_buckets
            bucket_index = unsafe_trunc(Int, bucket_float)
            if filling_pattern[bucket_index + 1]
                cut_left =
                    first_left_cut + bucket_index * left_cut_distance
                bucket_bin_center0 = cut_left + bin_width / 2.0
                local_bin_float =
                    floor((dt_i - bucket_bin_center0) * inverse_bin_width)
                if local_bin_float >= 0.0 &&
                   local_bin_float < bins_per_profile - 1
                    bin_index =
                        Int(
                            bucket_index_to_memory_index[bucket_index + 1]
                        ) + unsafe_trunc(Int, local_bin_float) + 1
                    helper1 =
                        charge *
                        (voltage[bin_index + 1] - voltage[bin_index]) *
                        inverse_bin_width
                    helper2 =
                        (
                            charge * voltage[bin_index] -
                            bin_centers[bin_index] * helper1
                        ) + acceleration_kick
                    dE[i] += dt_i * helper1 + helper2
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
    dE, damping_factor, energy_lost
)
    i = @index(Global, Linear)
    @inbounds dE[i] = damping_factor * dE[i] - energy_lost
end

@kernel function synchrotron_radiation_quantum_excitation_kernel!(
    dE, noise, damping_factor, noise_scale, energy_lost
)
    i = @index(Global, Linear)
    @inbounds dE[i] =
        damping_factor * dE[i] + (noise[i] * noise_scale - energy_lost)
end
