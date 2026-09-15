# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

using Test
using Aqua
using JET
using LinearAlgebra: dot
using Random
using Atomix: Atomix
using KernelAbstractions:
    KernelAbstractions,
    @groupsize,
    @index,
    @kernel,
    @localmem,
    @synchronize,
    @uniform

using BLonDKernels

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

"""Raw pointer of an array as the `Int` the Python side passes."""
raw_pointer(array) = Int(UInt(pointer(array)))

identity_to_host(array) = array

# ----------------------------------------------------------------------
# reference implementations (plain Julia loops, mirroring
# `blond/core/backends/python/callables.py`)
# ----------------------------------------------------------------------

function reference_kick_single_harmonic(
    dt, dE, voltage, omega_rf, phi_rf, charge, acceleration_kick
)
    result = copy(dE)
    voltage_kick = charge * voltage
    for i in eachindex(dt)
        result[i] +=
            voltage_kick * sin(omega_rf * dt[i] + phi_rf) + acceleration_kick
    end
    return result
end

function reference_kick_multi_harmonic(
    dt, dE, voltage, omega_rf, phi_rf, charge, acceleration_kick
)
    result = copy(dE)
    for i in eachindex(dt)
        accumulator = result[i]
        for j in eachindex(voltage)
            accumulator +=
                (charge * voltage[j]) * sin(omega_rf[j] * dt[i] + phi_rf[j])
        end
        result[i] = accumulator + acceleration_kick
    end
    return result
end

function reference_drift_simple(dt, dE, drift_time, eta_0, beta, energy)
    result = copy(dt)
    coefficient = drift_time * (eta_0 / (beta * beta * energy))
    for i in eachindex(result)
        result[i] += coefficient * dE[i]
    end
    return result
end

function reference_drift_exact(
    dt, dE, drift_time, alpha_0, higher_alpha, beta, energy
)
    result = copy(dt)
    inverse_beta_squared = 1.0 / (beta * beta)
    inverse_energy = 1.0 / energy
    inverse_energy_squared = inverse_energy * inverse_energy
    for i in eachindex(result)
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
        for k in eachindex(higher_alpha)
            polynomial += higher_alpha[k] * delta_power
            delta_power *= beam_delta
        end
        result[i] +=
            drift_time * (
                polynomial * (1.0 + energy_offset * inverse_energy) /
                (1.0 + beam_delta) - 1.0
            )
    end
    return result
end

function reference_loss_box(
    e_max, e_min, t_min, t_max, dt, dE, flags, lost_flag
)
    result = copy(flags)
    for i in eachindex(dt)
        if (dE[i] > e_max) || (dE[i] < e_min) ||
           (dt[i] < t_min) || (dt[i] > t_max)
            result[i] = lost_flag
        end
    end
    return result
end

function reference_beam_phase(
    hist_x, hist_y, alpha, omega_rf, phi_rf, bin_size
)
    n_bins = length(hist_x)
    sine_values = zeros(Float64, n_bins)
    cosine_values = zeros(Float64, n_bins)
    for i in 1:n_bins
        weight = exp(alpha * hist_x[i]) * hist_y[i]
        angle = omega_rf * hist_x[i] + phi_rf
        sine_values[i] = weight * sin(angle)
        cosine_values[i] = weight * cos(angle)
    end
    sine_coefficient = 0.0
    cosine_coefficient = 0.0
    for i in 1:(n_bins - 1)
        sine_coefficient += 0.5 * (sine_values[i] + sine_values[i + 1]) *
                            bin_size
        cosine_coefficient += 0.5 *
                              (cosine_values[i] + cosine_values[i + 1]) *
                              bin_size
    end
    return sine_coefficient / cosine_coefficient
end

# The global-atomic histogram the privatised kernel replaced; kept as
# the performance baseline of the "workgroup privatisation" testset.
@kernel function naive_histogram_kernel!(
    array_read, array_write, n_bins, start, stop, inverse_bin_width
)
    i = @index(Global, Linear)
    @inbounds begin
        value = array_read[i]
        if value == stop
            Atomix.@atomic array_write[n_bins] += 1.0
        else
            bin_float = floor((value - start) * inverse_bin_width)
            if bin_float >= 0.0 && bin_float < n_bins
                bin_index = unsafe_trunc(Int, bin_float) + 1
                Atomix.@atomic array_write[bin_index] += 1.0
            end
        end
    end
end

# The workgroup-private histogram the per-thread slices (CPU) and the
# strided launch (GPU) replaced; kept as the performance baseline of the
# "histogram! beats the workgroup kernel" testset. Each workgroup counts
# its share of the input into workgroup-local bins and flushes them.
const WORKGROUP_HISTOGRAM_SIZE = 1024
const WORKGROUP_HISTOGRAM_VALUES_PER_ITEM = 32

@kernel function workgroup_histogram_kernel!(
    array_read,
    array_write,
    n_read,
    n_bins,
    start,
    stop,
    inverse_bin_width,
    values_per_workgroup,
    ::Val{NEED_ATOMICS},
) where {NEED_ATOMICS}
    workgroup_size = @uniform @groupsize()[1]
    local_counts = @localmem Int32 (4096,)
    zero_index = @index(Local, Linear)
    while zero_index <= n_bins
        @inbounds local_counts[zero_index] = Int32(0)
        zero_index += workgroup_size
    end
    @synchronize
    local_index = @index(Local, Linear)
    group_index = @index(Group, Linear)
    value_index = (group_index - 1) * values_per_workgroup + local_index
    last_value_index = min(group_index * values_per_workgroup, n_read)
    while value_index <= last_value_index
        @inbounds value = array_read[value_index]
        bin_index = 0
        if value == stop
            bin_index = n_bins
        else
            bin_float = floor((value - start) * inverse_bin_width)
            if bin_float >= 0.0 && bin_float < n_bins
                bin_index = unsafe_trunc(Int, bin_float) + 1
            end
        end
        if bin_index != 0
            if NEED_ATOMICS
                @inbounds Atomix.@atomic local_counts[bin_index] += Int32(1)
            else
                @inbounds local_counts[bin_index] += Int32(1)
            end
        end
        value_index += workgroup_size
    end
    @synchronize
    flush_index = @index(Local, Linear)
    while flush_index <= n_bins
        @inbounds count = local_counts[flush_index]
        if count != Int32(0)
            @inbounds Atomix.@atomic array_write[flush_index] += Float64(count)
        end
        flush_index += workgroup_size
    end
end

"""Run `workgroup_histogram_kernel!` over at most 4096 bins."""
function workgroup_histogram!(device, values, n_values, out, n_bins, start, stop)
    fill!(out, 0.0)
    values_per_workgroup =
        WORKGROUP_HISTOGRAM_SIZE * WORKGROUP_HISTOGRAM_VALUES_PER_ITEM
    kernel! = workgroup_histogram_kernel!(device)
    kernel!(
        values,
        out,
        n_values,
        n_bins,
        start,
        stop,
        n_bins / (stop - start),
        values_per_workgroup,
        Val(!(device isa KernelAbstractions.CPU));
        ndrange=cld(n_values, values_per_workgroup) * WORKGROUP_HISTOGRAM_SIZE,
        workgroupsize=WORKGROUP_HISTOGRAM_SIZE,
    )
    return nothing
end

# The per-particle interpolated kick the two-phase kernels replaced (it
# recomputes the bin factors for every particle); kept as the performance
# baseline of the "kick_interpolated_dense! precomputed factors" testset.
@kernel function per_particle_kick_interpolated_dense_kernel!(
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

# The per-particle kernel applying a preallocated noise array (the previous
# GPU implementation); kept as the baseline of the parallel noise timing.
@kernel function noise_array_quantum_excitation_kernel!(
    dE, noise, damping_factor, noise_scale, energy_lost
)
    i = @index(Global, Linear)
    @inbounds dE[i] =
        damping_factor * dE[i] + (noise[i] * noise_scale - energy_lost)
end

# The chunked quantum-excitation sweep that fills a preallocated noise
# array (the previous CPU implementation); kept as the performance
# baseline of the "quantum excitation without noise array" testset.
function noise_array_quantum_excitation_range!(
    first_particle,
    last_particle,
    dE,
    noise,
    damping_factor,
    noise_scale,
    energy_lost,
)
    randn!(view(noise, first_particle:last_particle))
    @inbounds @simd for i in first_particle:last_particle
        dE[i] =
            damping_factor * dE[i] + (noise[i] * noise_scale - energy_lost)
    end
    return nothing
end

# The beam phase over a complex values array reduced three times (the
# previous implementation, still used on GPUs); kept as the performance
# baseline of the "beam_phase chunked sums" testset.
function values_array_beam_phase(
    device, hist_x, hist_y, alpha, omega_rf, phi_rf, bin_size
)
    n_bins = length(hist_x)
    values = similar(hist_x, ComplexF64)
    kernel! = BLonDKernels.beam_phase_values_kernel!(device)
    kernel!(values, hist_x, hist_y, alpha, omega_rf, phi_rf; ndrange=n_bins)
    KernelAbstractions.synchronize(device)
    edge_values = sum(@view values[1:1]) + sum(@view values[n_bins:n_bins])
    integral = (sum(values) - 0.5 * edge_values) * bin_size
    return imag(integral) / real(integral)
end

function reference_histogram(array_read, n_bins, start, stop)
    result = zeros(Float64, n_bins)
    inverse_bin_width = n_bins / (stop - start)
    for value in array_read
        if value == stop
            result[n_bins] += 1.0
            continue
        end
        bin_float = floor((value - start) * inverse_bin_width)
        if bin_float < 0.0 || bin_float >= n_bins
            continue
        end
        result[Int(bin_float) + 1] += 1.0
    end
    return result
end

function reference_kick_interpolated_dense(
    dt, dE, voltage, bin_centers, charge, acceleration_kick
)
    result = copy(dE)
    n_slices = length(bin_centers)
    n_slices < 2 && return result
    inverse_bin_width =
        (n_slices - 1) / (bin_centers[end] - bin_centers[1])
    helper1 = [
        charge * (voltage[i + 1] - voltage[i]) * inverse_bin_width
        for i in 1:(n_slices - 1)
    ]
    helper2 = [
        (charge * voltage[i] - bin_centers[i] * helper1[i]) +
        acceleration_kick for i in 1:(n_slices - 1)
    ]
    for i in eachindex(dt)
        bin_float = floor((dt[i] - bin_centers[1]) * inverse_bin_width)
        if bin_float >= 0.0 && bin_float < n_slices - 1
            bin_index = Int(bin_float) + 1
            result[i] += dt[i] * helper1[bin_index] + helper2[bin_index]
        end
    end
    return result
end

function reference_kick_interpolated_sparse(
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
    bucket_index_to_memory_index,
)
    result = copy(dE)
    n_slices = length(bin_centers)
    inverse_bin_width = bins_per_profile / cut_width
    helper1 = [
        charge * (voltage[i + 1] - voltage[i]) * inverse_bin_width
        for i in 1:(n_slices - 1)
    ]
    helper2 = [
        (charge * voltage[i] - bin_centers[i] * helper1[i]) +
        acceleration_kick for i in 1:(n_slices - 1)
    ]
    n_buckets = length(filling_pattern)
    inverse_histogram_distance = 1.0 / left_cut_distance
    bin_width = cut_width / bins_per_profile
    for i in eachindex(dt)
        bucket_float =
            floor((dt[i] - first_left_cut) * inverse_histogram_distance)
        (bucket_float < 0.0 || bucket_float >= n_buckets) && continue
        bucket_index = Int(bucket_float)
        filling_pattern[bucket_index + 1] || continue
        cut_left = first_left_cut + bucket_index * left_cut_distance
        bucket_bin_center0 = cut_left + bin_width / 2.0
        local_bin_float =
            floor((dt[i] - bucket_bin_center0) * inverse_bin_width)
        (local_bin_float < 0.0 || local_bin_float >= bins_per_profile - 1) &&
            continue
        local_bin = Int(local_bin_float)
        memory_index =
            bucket_index_to_memory_index[bucket_index + 1] + local_bin
        result[i] +=
            dt[i] * helper1[memory_index + 1] + helper2[memory_index + 1]
    end
    return result
end

function reference_histogram_sparse(
    x,
    n_out,
    first_left_cut,
    left_cut_distance,
    cut_width,
    bins_per_profile,
    filling_pattern,
    bucket_index_to_memory_index,
)
    result = zeros(Float64, n_out)
    n_buckets = length(filling_pattern)
    inverse_profile_distance = 1.0 / left_cut_distance
    inverse_bin_step = bins_per_profile / cut_width
    for value in x
        bucket_float =
            floor((value - first_left_cut) * inverse_profile_distance)
        (bucket_float < 0.0 || bucket_float >= n_buckets) && continue
        bucket_index = Int(bucket_float)
        filling_pattern[bucket_index + 1] || continue
        start_location = first_left_cut + bucket_index * left_cut_distance
        stop_location = start_location + cut_width
        memory_offset = bucket_index_to_memory_index[bucket_index + 1]
        if value == stop_location
            result[memory_offset + bins_per_profile] += 1.0
            continue
        end
        (value < start_location || value >= stop_location) && continue
        local_bin = Int(floor((value - start_location) * inverse_bin_step))
        (local_bin < 0 || local_bin >= bins_per_profile) && continue
        result[memory_offset + local_bin + 1] += 1.0
    end
    return result
end

function reference_move_flagged_elements_to_end(flag, flags, dt, dE, ids)
    flags_out = copy(flags)
    dt_out = copy(dt)
    dE_out = copy(dE)
    ids_out = copy(ids)
    i = 1
    j = length(flags_out)
    while i <= j
        if flags_out[i] != flag
            i += 1
        else
            flags_out[i], flags_out[j] = flags_out[j], flags_out[i]
            dt_out[i], dt_out[j] = dt_out[j], dt_out[i]
            dE_out[i], dE_out[j] = dE_out[j], dE_out[i]
            ids_out[i], ids_out[j] = ids_out[j], ids_out[i]
            j -= 1
        end
    end
    return j, flags_out, dt_out, dE_out, ids_out
end

function reference_wake_from_pole_residue(
    profile,
    profile_dts,
    poles,
    residues,
    is_counterrotating_beam,
    counterrotating_pole_signs,
    update_on_bin,
    factor,
    states,
)
    n_poles = length(poles)
    n_bins = length(profile)
    two_factor = 2 * factor
    voltage = zeros(Float64, n_bins)
    states_out = copy(states)
    t_start = states[end]
    for pole_i in 1:n_poles
        counterrotating_flip = 1.0
        if is_counterrotating_beam &&
           counterrotating_pole_signs[pole_i] == -1
            counterrotating_flip = -1.0
        end
        i_update = 0
        update_bin = isempty(update_on_bin) ? -1 : Int(update_on_bin[1])
        pole = poles[pole_i]
        residue = residues[pole_i]
        state = states[pole_i]
        injection_factor = imag(pole) == 0 ? factor : two_factor
        decay = 0.0 + 0.0im
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
                if i_update < length(update_on_bin)
                    update_bin = Int(update_on_bin[i_update + 1])
                end
            else
                state *= decay
            end
            state += profile_half
            voltage[bin_i + 1] +=
                counterrotating_flip * real(residue * state)
            state += profile_half
        end
        states_out[pole_i] = state
    end
    states_out[end] = profile_dts[end]
    return voltage, states_out
end

function reference_music_track(
    beam_dt,
    beam_dE,
    parameter_array,
    alpha,
    omega_bar,
    const_factor,
    coeff1,
    coeff2,
    coeff3,
    coeff4,
    time_since_last_track,
    multiturn,
)
    n = length(beam_dt)
    dE_out = copy(beam_dE)
    induced_voltage = zeros(Float64, n)
    parameters_out = copy(parameter_array)
    if multiturn
        time_difference =
            beam_dt[1] + time_since_last_track - parameter_array[3]
        exp_term = exp(-alpha * time_difference)
        cos_term = cos(omega_bar * time_difference)
        sin_term = sin(omega_bar * time_difference)
        product_first =
            exp_term * (
                (cos_term + coeff1 * sin_term) * parameter_array[1] +
                coeff2 * sin_term * parameter_array[2]
            )
        product_second =
            exp_term * (
                coeff3 * sin_term * parameter_array[1] +
                (cos_term + coeff4 * sin_term) * parameter_array[2]
            )
    else
        product_first = 0.0
        product_second = 0.0
    end
    induced_voltage[1] = const_factor * (0.5 + product_first)
    dE_out[1] += induced_voltage[1]
    input_first = product_first + 1.0
    input_second = product_second
    for i in 1:(n - 1)
        time_difference = beam_dt[i + 1] - beam_dt[i]
        exp_term = exp(-alpha * time_difference)
        cos_term = cos(omega_bar * time_difference)
        sin_term = sin(omega_bar * time_difference)
        product_first =
            exp_term * (
                (cos_term + coeff1 * sin_term) * input_first +
                coeff2 * sin_term * input_second
            )
        product_second =
            exp_term * (
                coeff3 * sin_term * input_first +
                (cos_term + coeff4 * sin_term) * input_second
            )
        induced_voltage[i + 1] = const_factor * (0.5 + product_first)
        dE_out[i + 1] += induced_voltage[i + 1]
        input_first = product_first + 1.0
        input_second = product_second
    end
    parameters_out[1] = input_first
    parameters_out[2] = input_second
    parameters_out[3] = beam_dt[n]
    return dE_out, induced_voltage, parameters_out
end

# ----------------------------------------------------------------------
# device-generic test suite
# ----------------------------------------------------------------------

function run_device_tests(
    label,
    device,
    to_device,
    to_host;
    run_jet::Bool,
    strict_float::Bool,
    gpu_allocated=nothing,
)
    # A GPU is free to contract `a * b + c` into a fused multiply-add,
    # which differs from the CPU result by an ULP or two. Compare
    # bit-exactly only where that contraction cannot happen.
    same(actual, expected) =
        strict_float ? actual == expected :
        isapprox(actual, expected; rtol=1e-14, atol=0.0)
    @testset verbose = true "$label" begin
        @testset "device helpers" begin
            @test (@inferred BLonDKernels.max_threads(device)) isa Int
            @test (@inferred BLonDKernels.synchronize_device(device)) ===
                  nothing
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.max_threads(
                    device
                )
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.synchronize_device(
                    device
                )
            end
        end

        @testset "kick_single_harmonic!" begin
            dt_host = collect(range(1e-9, 10e-9; length=10))
            dE_host = collect(range(1e9, 10e9; length=10))
            voltage = 1e3
            omega_rf = 2 * pi * 400e3
            phi_rf = 0.3
            charge = 1.0
            acceleration_kick = -1.0
            expected = reference_kick_single_harmonic(
                dt_host, dE_host, voltage, omega_rf, phi_rf, charge,
                acceleration_kick,
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            @test (@inferred BLonDKernels.kick_single_harmonic!(
                device,
                raw_pointer(dt),
                raw_pointer(dE),
                length(dt_host),
                voltage,
                omega_rf,
                phi_rf,
                charge,
                acceleration_kick,
            )) === nothing
            @test to_host(dE) ≈ expected rtol = 1e-14
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.kick_single_harmonic!(
                    device, raw_pointer(dt), raw_pointer(dE),
                    length(dt_host), voltage, omega_rf, phi_rf, charge,
                    acceleration_kick,
                )
            end
        end

        @testset "kick_multi_harmonic!" begin
            dt_host = collect(range(1e-9, 10e-9; length=10))
            dE_host = collect(range(1e9, 10e9; length=10))
            voltage_host = collect(range(1e6, 5e6; length=3))
            omega_host = collect(range(200e6, 400e6; length=3))
            phi_host = collect(range(0, 2 * pi; length=3))
            charge = 1.0
            acceleration_kick = -1.0
            expected = reference_kick_multi_harmonic(
                dt_host, dE_host, voltage_host, omega_host, phi_host,
                charge, acceleration_kick,
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            voltage = to_device(voltage_host)
            omega_rf = to_device(omega_host)
            phi_rf = to_device(phi_host)
            @test (@inferred BLonDKernels.kick_multi_harmonic!(
                device,
                raw_pointer(dt),
                raw_pointer(dE),
                length(dt_host),
                raw_pointer(voltage),
                raw_pointer(omega_rf),
                raw_pointer(phi_rf),
                length(voltage_host),
                charge,
                acceleration_kick,
            )) === nothing
            @test to_host(dE) ≈ expected rtol = 1e-14
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.kick_multi_harmonic!(
                    device, raw_pointer(dt), raw_pointer(dE),
                    length(dt_host), raw_pointer(voltage),
                    raw_pointer(omega_rf), raw_pointer(phi_rf),
                    length(voltage_host), charge, acceleration_kick,
                )
            end
        end

        @testset "drift_simple!" begin
            dt_host = collect(range(1e-9, 10e-9; length=10))
            dE_host = collect(range(1e9, 10e9; length=10))
            drift_time = 5.0
            eta_0 = 0.3
            beta = 0.9
            energy = 10.0
            expected = reference_drift_simple(
                dt_host, dE_host, drift_time, eta_0, beta, energy
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            @test (@inferred BLonDKernels.drift_simple!(
                device,
                raw_pointer(dt),
                raw_pointer(dE),
                length(dt_host),
                drift_time,
                eta_0,
                beta,
                energy,
            )) === nothing
            @test to_host(dt) ≈ expected rtol = 1e-14
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.drift_simple!(
                    device, raw_pointer(dt), raw_pointer(dE),
                    length(dt_host), drift_time, eta_0, beta, energy,
                )
            end
        end

        @testset "drift_exact!" begin
            for higher_alpha_host in ([1.0, 2.0], Float64[])
                dt_host = collect(range(1e-9, 10e-9; length=10))
                dE_host = collect(range(1e9, 10e9; length=10))
                drift_time = 5.0
                alpha_0 = 1.0
                beta = 0.9
                energy = 10.0
                expected = reference_drift_exact(
                    dt_host, dE_host, drift_time, alpha_0, higher_alpha_host,
                    beta, energy,
                )
                dt = to_device(dt_host)
                dE = to_device(dE_host)
                higher_alpha = to_device(higher_alpha_host)
                @test (@inferred BLonDKernels.drift_exact!(
                    device,
                    raw_pointer(dt),
                    raw_pointer(dE),
                    length(dt_host),
                    drift_time,
                    alpha_0,
                    raw_pointer(higher_alpha),
                    length(higher_alpha_host),
                    beta,
                    energy,
                )) === nothing
                @test to_host(dt) ≈ expected rtol = 1e-14
                if run_jet
                    @test_opt target_modules = (BLonDKernels,) BLonDKernels.drift_exact!(
                        device, raw_pointer(dt), raw_pointer(dE),
                        length(dt_host), drift_time, alpha_0,
                        raw_pointer(higher_alpha), length(higher_alpha_host),
                        beta, energy,
                    )
                end
            end
        end

        @testset "loss_box!" begin
            n = 50
            dt_host = collect(range(-20, 20; length=n))
            dE_host = collect(range(-2, 2; length=n))
            flags_host = Int32.(0:(n - 1))
            lost_flag = Int32(2)
            expected = reference_loss_box(
                1.0, -1.0, -10.0, 10.0, dt_host, dE_host, flags_host,
                lost_flag,
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            flags = to_device(flags_host)
            @test (@inferred BLonDKernels.loss_box!(
                device,
                1.0,
                -1.0,
                -10.0,
                10.0,
                raw_pointer(dt),
                raw_pointer(dE),
                raw_pointer(flags),
                n,
                lost_flag,
            )) === nothing
            @test to_host(flags) == expected
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.loss_box!(
                    device, 1.0, -1.0, -10.0, 10.0, raw_pointer(dt),
                    raw_pointer(dE), raw_pointer(flags), n, lost_flag,
                )
            end
        end

        @testset "sum_1d_array / dot_product_1d_array" begin
            rng = MersenneTwister(1234)
            values_host = rand(rng, 10_000)
            other_host = rand(rng, 10_000)
            values = to_device(values_host)
            other = to_device(other_host)
            result_sum = @inferred BLonDKernels.sum_1d_array(
                device, raw_pointer(values), length(values_host)
            )
            @test result_sum isa Float64
            @test result_sum ≈ sum(values_host) rtol = 1e-12
            result_dot = @inferred BLonDKernels.dot_product_1d_array(
                device, raw_pointer(values), raw_pointer(other),
                length(values_host),
            )
            @test result_dot isa Float64
            @test result_dot ≈ dot(values_host, other_host) rtol = 1e-12

            empty_host = Float64[]
            empty_device = to_device(empty_host)
            @test BLonDKernels.sum_1d_array(
                device, raw_pointer(empty_device), 0
            ) == 0.0
            @test BLonDKernels.dot_product_1d_array(
                device, raw_pointer(empty_device),
                raw_pointer(empty_device), 0,
            ) == 0.0
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.sum_1d_array(
                    device, raw_pointer(values), length(values_host)
                )
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.dot_product_1d_array(
                    device, raw_pointer(values), raw_pointer(other),
                    length(values_host),
                )
            end
        end

        @testset "histogram!" begin
            values_host = [-1e30, -1e12, -12.0, 0.0, 8.0, 1e12, 1e30]
            n_bins = 21
            start = -12.0
            stop = 8.0
            expected = reference_histogram(values_host, n_bins, start, stop)
            @test sum(expected) == 3.0
            values = to_device(values_host)
            # pre-filled with ones: the entry must zero the output
            out = to_device(ones(Float64, n_bins))
            @test (@inferred BLonDKernels.histogram!(
                device,
                raw_pointer(values),
                length(values_host),
                raw_pointer(out),
                n_bins,
                start,
                stop,
            )) === nothing
            @test to_host(out) == expected
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.histogram!(
                    device, raw_pointer(values), length(values_host),
                    raw_pointer(out), n_bins, start, stop,
                )
            end
        end

        @testset "histogram! workgroup privatisation" begin
            # Many values (several workgroups, partial last workgroup,
            # heavy contention on few bins) and more bins than fit into
            # one workgroup's local memory (multi-pass path). The
            # values are chosen so that every case also has entries
            # exactly on `stop`, below `start` and above `stop`.
            rng = MersenneTwister(1234)
            n_values = 1_000_003
            start = -1.0
            stop = 3.0
            values_host = 4.0 .* rand(rng, n_values) .- 1.5
            values_host[1:100] .= stop
            values_host[101:200] .= start
            values = to_device(values_host)
            local_bins = BLonDKernels.HISTOGRAM_LOCAL_BINS
            for n_bins in (1, 7, 1000, local_bins, 3 * local_bins + 5)
                expected = reference_histogram(
                    values_host, n_bins, start, stop
                )
                out = to_device(ones(Float64, n_bins))
                BLonDKernels.histogram!(
                    device,
                    raw_pointer(values),
                    n_values,
                    raw_pointer(out),
                    n_bins,
                    start,
                    stop,
                )
                @test to_host(out) == expected
            end
            # The privatised kernel must beat a naive global-atomic
            # kernel (the previous implementation) on the same device,
            # measured relative to each other so the test does not
            # depend on the machine. Both are timed after compilation.
            n_bins = 1000
            out = to_device(zeros(Float64, n_bins))
            function run_privatised()
                BLonDKernels.histogram!(
                    device, raw_pointer(values), n_values, raw_pointer(out),
                    n_bins, start, stop,
                )
                BLonDKernels.synchronize_device(device)
            end
            naive_kernel! = naive_histogram_kernel!(device)
            function run_naive()
                fill!(out, 0.0)
                naive_kernel!(
                    values, out, n_bins, start, stop,
                    n_bins / (stop - start); ndrange=n_values,
                )
                KernelAbstractions.synchronize(device)
            end
            run_privatised()
            run_naive()
            elapsed_privatised =
                minimum(@elapsed(run_privatised()) for _ in 1:20)
            elapsed_naive = minimum(@elapsed(run_naive()) for _ in 1:20)
            @test elapsed_privatised < elapsed_naive
        end

        if device isa KernelAbstractions.CPU
            @testset "histogram_slice_count" begin
                # Waking a thread costs about as much as counting
                # `HISTOGRAM_VALUES_PER_THREAD` values on one, so smaller
                # inputs stay on a single task and larger ones get one
                # slice per that many values, up to one per thread.
                per_thread = BLonDKernels.HISTOGRAM_VALUES_PER_THREAD
                n_threads = BLonDKernels.max_threads(device)
                slice_count(n) = @inferred BLonDKernels.histogram_slice_count(
                    device, n
                )
                @test slice_count(1) == 1
                @test slice_count(per_thread) == 1
                @test slice_count(per_thread + 1) == min(n_threads, 2)
                @test slice_count(3 * per_thread) == min(n_threads, 3)
                @test slice_count(1000 * per_thread) == n_threads
            end
        end

        @testset "histogram! beats the workgroup kernel" begin
            # The per-thread slices on the CPU and the strided launch on a
            # GPU must clearly beat the workgroup kernel (the previous
            # implementation) on the same device, timed relative to each
            # other after compilation. The gain comes from vectorising the
            # bin computation on the CPU and from its integer-only range
            # decision on a GPU; `--check-bounds=yes` (forced by `Pkg.test`)
            # prevents the former and adds a bounds check to every access
            # in both GPU kernels, which hides the latter. Run
            # `Pkg.test(julia_args=`--check-bounds=auto`)`.
            if Base.JLOptions().check_bounds == 1
                @test_skip "histogram timing needs --check-bounds=auto"
            else
                rng = MersenneTwister(4321)
                n_values = 1_000_000
                n_bins = 21
                start = -12.0
                stop = 8.0
                values = to_device(20.0 .* rand(rng, n_values) .- 10.0)
                out = to_device(zeros(Float64, n_bins))
                function run_entry()
                    BLonDKernels.histogram!(
                        device, raw_pointer(values), n_values,
                        raw_pointer(out), n_bins, start, stop,
                    )
                    BLonDKernels.synchronize_device(device)
                end
                function run_workgroup()
                    workgroup_histogram!(
                        device, values, n_values, out, n_bins, start, stop
                    )
                    BLonDKernels.synchronize_device(device)
                end
                # Sustained load first: an idle GPU runs in a low power
                # state that would slow down whichever is timed first.
                for _ in 1:50
                    run_entry()
                    run_workgroup()
                end
                elapsed_entry = minimum(@elapsed(run_entry()) for _ in 1:20)
                elapsed_workgroup =
                    minimum(@elapsed(run_workgroup()) for _ in 1:20)
                @test elapsed_entry < 0.75 * elapsed_workgroup
            end
        end

        @testset "beam_phase" begin
            hist_x_host = collect(range(-10, 10; length=21))
            hist_y_host = 10.0^2 .- hist_x_host .^ 2
            alpha = 1.5
            omega_rf = 2.5
            phi_rf = 3.5
            bin_size = 1.0
            expected = reference_beam_phase(
                hist_x_host, hist_y_host, alpha, omega_rf, phi_rf, bin_size
            )
            hist_x = to_device(hist_x_host)
            hist_y = to_device(hist_y_host)
            result = @inferred BLonDKernels.beam_phase(
                device,
                raw_pointer(hist_x),
                raw_pointer(hist_y),
                length(hist_x_host),
                alpha,
                omega_rf,
                phi_rf,
                bin_size,
            )
            @test result isa Float64
            @test result ≈ expected rtol = 1e-12
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.beam_phase(
                    device, raw_pointer(hist_x), raw_pointer(hist_y),
                    length(hist_x_host), alpha, omega_rf, phi_rf, bin_size,
                )
            end
        end

        @testset "kick_interpolated_dense!" begin
            dt_host = collect(range(-5, 5; length=20))
            bin_centers_host = collect(range(-4, 4; length=20))
            voltage_host = bin_centers_host .^ 2
            charge = 10.0
            acceleration_kick = 0.5
            dE_host = zeros(Float64, length(dt_host))
            expected = reference_kick_interpolated_dense(
                dt_host, dE_host, voltage_host, bin_centers_host, charge,
                acceleration_kick,
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            voltage = to_device(voltage_host)
            bin_centers = to_device(bin_centers_host)
            @test (@inferred BLonDKernels.kick_interpolated_dense!(
                device,
                raw_pointer(dt),
                raw_pointer(dE),
                length(dt_host),
                raw_pointer(voltage),
                raw_pointer(bin_centers),
                length(bin_centers_host),
                charge,
                acceleration_kick,
            )) === nothing
            @test same(to_host(dE), expected)
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.kick_interpolated_dense!(
                    device, raw_pointer(dt), raw_pointer(dE),
                    length(dt_host), raw_pointer(voltage),
                    raw_pointer(bin_centers), length(bin_centers_host),
                    charge, acceleration_kick,
                )
            end

            # extreme values must never be kicked (no undefined float ->
            # int conversion)
            outlier_host = [-1e30, -1e12, -4.5, 0.0, 4.5, 1e12, 1e30]
            outlier_dt = to_device(outlier_host)
            outlier_dE = to_device(zeros(Float64, length(outlier_host)))
            BLonDKernels.kick_interpolated_dense!(
                device, raw_pointer(outlier_dt), raw_pointer(outlier_dE),
                length(outlier_host), raw_pointer(voltage),
                raw_pointer(bin_centers), length(bin_centers_host), charge,
                acceleration_kick,
            )
            outlier_result = to_host(outlier_dE)
            @test outlier_result[[1, 2, 3, 5, 6, 7]] == zeros(6)
            @test outlier_result[4] != 0.0

            # a single-bin profile has no width -> no particle is kicked
            single_bin_centers = to_device([0.0])
            single_voltage = to_device([1.0])
            single_dE = to_device(zeros(Float64, length(dt_host)))
            BLonDKernels.kick_interpolated_dense!(
                device, raw_pointer(dt), raw_pointer(single_dE),
                length(dt_host), raw_pointer(single_voltage),
                raw_pointer(single_bin_centers), 1, charge,
                acceleration_kick,
            )
            @test to_host(single_dE) == zeros(length(dt_host))
        end

        @testset "kick_interpolated_sparse!" begin
            bins_per_profile = 4
            filling_pattern_host = [true, false, false, true]
            bucket_index_to_memory_index_host =
                Int32[0, 0, 0, bins_per_profile]
            first_left_cut = 0.0
            left_cut_distance = 1.0
            cut_width = 1.0
            bin_width = cut_width / bins_per_profile
            bin_centers_host = vcat(
                [
                    first_left_cut + bucket * left_cut_distance +
                    bin_width * (index + 0.5)
                    for bucket in (0, 3) for index in 0:(bins_per_profile - 1)
                ]...,
            )
            voltage_host = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
            # particles: on the first bin of the second island, and in an
            # unfilled bucket (must not be kicked)
            dt_host = [bin_centers_host[5], 1.5]
            dE_host = zeros(Float64, length(dt_host))
            charge = 1.0
            acceleration_kick = 0.0
            expected = reference_kick_interpolated_sparse(
                dt_host, dE_host, voltage_host, bin_centers_host, charge,
                acceleration_kick, first_left_cut, left_cut_distance,
                cut_width, bins_per_profile, filling_pattern_host,
                bucket_index_to_memory_index_host,
            )
            @test expected[2] == 0.0
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            voltage = to_device(voltage_host)
            bin_centers = to_device(bin_centers_host)
            filling_pattern = to_device(filling_pattern_host)
            bucket_index_to_memory_index =
                to_device(bucket_index_to_memory_index_host)
            @test (@inferred BLonDKernels.kick_interpolated_sparse!(
                device,
                raw_pointer(dt),
                raw_pointer(dE),
                length(dt_host),
                raw_pointer(voltage),
                raw_pointer(bin_centers),
                length(bin_centers_host),
                charge,
                acceleration_kick,
                first_left_cut,
                left_cut_distance,
                cut_width,
                bins_per_profile,
                raw_pointer(filling_pattern),
                length(filling_pattern_host),
                raw_pointer(bucket_index_to_memory_index),
            )) === nothing
            @test same(to_host(dE), expected)
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.kick_interpolated_sparse!(
                    device, raw_pointer(dt), raw_pointer(dE),
                    length(dt_host), raw_pointer(voltage),
                    raw_pointer(bin_centers), length(bin_centers_host),
                    charge, acceleration_kick, first_left_cut,
                    left_cut_distance, cut_width, bins_per_profile,
                    raw_pointer(filling_pattern),
                    length(filling_pattern_host),
                    raw_pointer(bucket_index_to_memory_index),
                )
            end

            # single filled bucket: sparse must agree bit-exactly with the
            # dense path on the same bucket
            bins_per_profile_single = 8
            first_left_cut_single = -0.37
            cut_width_single = 0.9
            bin_width_single = cut_width_single / bins_per_profile_single
            centers_single = [
                first_left_cut_single + bin_width_single * (index + 0.5)
                for index in 0:(bins_per_profile_single - 1)
            ]
            voltage_single_host = [1.0, 2.0, 5.0, 3.0, 8.0, 1.5, 4.0, 6.0]
            dt_single_host = collect(
                range(
                    first_left_cut_single - 0.1,
                    first_left_cut_single + cut_width_single + 0.1;
                    length=25,
                ),
            )
            dt_single = to_device(dt_single_host)
            voltage_single = to_device(voltage_single_host)
            centers_single_device = to_device(centers_single)
            filling_single = to_device([true])
            memory_index_single = to_device(Int32[0])
            dE_sparse = to_device(zeros(Float64, length(dt_single_host)))
            dE_dense = to_device(zeros(Float64, length(dt_single_host)))
            BLonDKernels.kick_interpolated_sparse!(
                device, raw_pointer(dt_single), raw_pointer(dE_sparse),
                length(dt_single_host), raw_pointer(voltage_single),
                raw_pointer(centers_single_device), bins_per_profile_single,
                1.0, 0.0, first_left_cut_single, 1.0, cut_width_single,
                bins_per_profile_single, raw_pointer(filling_single), 1,
                raw_pointer(memory_index_single),
            )
            BLonDKernels.kick_interpolated_dense!(
                device, raw_pointer(dt_single), raw_pointer(dE_dense),
                length(dt_single_host), raw_pointer(voltage_single),
                raw_pointer(centers_single_device), bins_per_profile_single,
                1.0, 0.0,
            )
            @test to_host(dE_sparse) == to_host(dE_dense)
        end

        if !(device isa KernelAbstractions.CPU)
            @testset "kick_interpolated_dense! precomputed factors" begin
                # Computing the bin factors once per call instead of once
                # per particle must clearly beat the per-particle kernel
                # (the previous implementation) on a GPU.
                n = 10^6
                n_slices = 100
                bin_centers_host = collect(range(-4.0, 4.0; length=n_slices))
                dt = to_device(collect(range(-5.0, 5.0; length=n)))
                dE = to_device(zeros(Float64, n))
                voltage = to_device(bin_centers_host .^ 2)
                bin_centers = to_device(bin_centers_host)
                per_particle_kernel! =
                    per_particle_kick_interpolated_dense_kernel!(device)
                function run_per_particle()
                    per_particle_kernel!(
                        dt, dE, voltage, bin_centers, n_slices, 10.0, 0.5;
                        ndrange=n,
                    )
                    KernelAbstractions.synchronize(device)
                end
                function run_entry()
                    BLonDKernels.kick_interpolated_dense!(
                        device, raw_pointer(dt), raw_pointer(dE), n,
                        raw_pointer(voltage), raw_pointer(bin_centers),
                        n_slices, 10.0, 0.5,
                    )
                    BLonDKernels.synchronize_device(device)
                end
                # Sustained load first: an idle GPU runs in a low power
                # state that would slow down whichever is timed first.
                for _ in 1:200
                    run_per_particle()
                    run_entry()
                end
                elapsed_entry = minimum(@elapsed(run_entry()) for _ in 1:20)
                elapsed_per_particle =
                    minimum(@elapsed(run_per_particle()) for _ in 1:20)
                @test elapsed_entry < 0.75 * elapsed_per_particle
            end

            @testset "quantum excitation allocates no noise array" begin
                # The noise is drawn inside the kernel, so the entry must
                # not allocate a beam-sized noise array (let alone cuRAND's
                # power-of-two padded copy of it).
                n = 2^20 + 3
                dE = to_device(zeros(Float64, n))
                run_entry() = BLonDKernels.apply_synchrotron_radiation!(
                    device, raw_pointer(dE), n, 0.0, 14955.0, 1e-3, 20e9,
                    false,
                )
                run_entry()
                BLonDKernels.synchronize_device(device)
                allocated = Base.invokelatest(gpu_allocated, run_entry)
                @test allocated < sizeof(Float64) * n
            end
        end

        if !(device isa KernelAbstractions.CPU)
            @testset "strided particle kernels visit every particle" begin
                # More particles than work-items, so that every work-item
                # strides several times and the last strides are partial.
                n_work_items =
                    BLonDKernels.gpu_workgroups(device) *
                    BLonDKernels.GPU_WORKGROUP_SIZE
                n = 3 * n_work_items + 5
                dt_host = collect(range(-5.0, 5.0; length=n))
                dE_host = collect(range(-2.0, 2.0; length=n))
                close(actual, expected) =
                    isapprox(actual, expected; rtol=1e-12, atol=0.0)

                dE = to_device(copy(dE_host))
                BLonDKernels.kick_single_harmonic!(
                    device, raw_pointer(to_device(dt_host)), raw_pointer(dE),
                    n, 3.0, 2.0, 0.5, 1.5, 0.25,
                )
                @test close(
                    to_host(dE),
                    reference_kick_single_harmonic(
                        dt_host, dE_host, 3.0, 2.0, 0.5, 1.5, 0.25
                    ),
                )

                voltage_host = [1e3, 5e2]
                omega_rf_host = [2.0, 4.0]
                phi_rf_host = [0.1, pi]
                dE = to_device(copy(dE_host))
                BLonDKernels.kick_multi_harmonic!(
                    device, raw_pointer(to_device(dt_host)), raw_pointer(dE),
                    n, raw_pointer(to_device(voltage_host)),
                    raw_pointer(to_device(omega_rf_host)),
                    raw_pointer(to_device(phi_rf_host)), 2, 1.5, 0.25,
                )
                @test close(
                    to_host(dE),
                    reference_kick_multi_harmonic(
                        dt_host, dE_host, voltage_host, omega_rf_host,
                        phi_rf_host, 1.5, 0.25,
                    ),
                )

                drift_dE_host = collect(range(-1e8, 1e8; length=n))
                drift_dt_host = collect(range(-1e-9, 1e-9; length=n))
                dt = to_device(copy(drift_dt_host))
                BLonDKernels.drift_simple!(
                    device, raw_pointer(dt),
                    raw_pointer(to_device(drift_dE_host)), n, 8.89e-5,
                    3.18e-4, 0.99999786, 450e9,
                )
                @test close(
                    to_host(dt),
                    reference_drift_simple(
                        drift_dt_host, drift_dE_host, 8.89e-5, 3.18e-4,
                        0.99999786, 450e9,
                    ),
                )

                higher_alpha_host = [1e-6, 2e-8]
                dt = to_device(copy(drift_dt_host))
                BLonDKernels.drift_exact!(
                    device, raw_pointer(dt),
                    raw_pointer(to_device(drift_dE_host)), n, 8.89e-5,
                    3.19e-4, raw_pointer(to_device(higher_alpha_host)), 2,
                    0.99999786, 450e9,
                )
                @test close(
                    to_host(dt),
                    reference_drift_exact(
                        drift_dt_host, drift_dE_host, 8.89e-5, 3.19e-4,
                        higher_alpha_host, 0.99999786, 450e9,
                    ),
                )

                flags = to_device(zeros(Int32, n))
                BLonDKernels.loss_box!(
                    device, 1.0, -1.0, -2.5, 2.5,
                    raw_pointer(to_device(dt_host)),
                    raw_pointer(to_device(dE_host)), raw_pointer(flags), n,
                    Int32(-5),
                )
                @test to_host(flags) == reference_loss_box(
                    1.0, -1.0, -2.5, 2.5, dt_host, dE_host, zeros(Int32, n),
                    Int32(-5),
                )

                bin_centers_host = collect(range(-4.0, 4.0; length=20))
                dense_voltage_host = bin_centers_host .^ 2
                dE = to_device(zeros(Float64, n))
                BLonDKernels.kick_interpolated_dense!(
                    device, raw_pointer(to_device(dt_host)), raw_pointer(dE),
                    n, raw_pointer(to_device(dense_voltage_host)),
                    raw_pointer(to_device(bin_centers_host)), 20, 10.0, 0.5,
                )
                @test close(
                    to_host(dE),
                    reference_kick_interpolated_dense(
                        dt_host, zeros(Float64, n), dense_voltage_host,
                        bin_centers_host, 10.0, 0.5,
                    ),
                )

                bins_per_profile = 4
                filling_pattern_host = [true, false, false, true]
                bucket_index_to_memory_index_host =
                    Int32[0, 0, 0, bins_per_profile]
                bin_width = 1.0 / bins_per_profile
                sparse_bin_centers_host = vcat(
                    [
                        bucket + bin_width * (index + 0.5)
                        for bucket in (0, 3)
                        for index in 0:(bins_per_profile - 1)
                    ]...,
                )
                sparse_voltage_host =
                    [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
                sparse_dt_host = collect(range(-0.5, 4.5; length=n))
                dE = to_device(zeros(Float64, n))
                BLonDKernels.kick_interpolated_sparse!(
                    device, raw_pointer(to_device(sparse_dt_host)),
                    raw_pointer(dE), n,
                    raw_pointer(to_device(sparse_voltage_host)),
                    raw_pointer(to_device(sparse_bin_centers_host)),
                    length(sparse_bin_centers_host), 1.0, 0.0, 0.0, 1.0, 1.0,
                    bins_per_profile,
                    raw_pointer(to_device(filling_pattern_host)),
                    length(filling_pattern_host),
                    raw_pointer(to_device(bucket_index_to_memory_index_host)),
                )
                @test close(
                    to_host(dE),
                    reference_kick_interpolated_sparse(
                        sparse_dt_host, zeros(Float64, n),
                        sparse_voltage_host, sparse_bin_centers_host, 1.0,
                        0.0, 0.0, 1.0, 1.0, bins_per_profile,
                        filling_pattern_host,
                        bucket_index_to_memory_index_host,
                    ),
                )

                dE = to_device(copy(dE_host))
                BLonDKernels.apply_synchrotron_radiation!(
                    device, raw_pointer(dE), n, 1.5, 100.0, 1e-3, 1e9, true
                )
                @test close(to_host(dE), (1.0 - 2.0 / 100.0) .* dE_host .- 1.5)

                # Quantum excitation from zero energy without loss is the
                # scaled noise itself: every particle must receive some.
                dE = to_device(zeros(Float64, n))
                BLonDKernels.apply_synchrotron_radiation!(
                    device, raw_pointer(dE), n, 0.0, 100.0, 1e-3, 1e9, false
                )
                @test count(iszero, to_host(dE)) == 0
            end
        end

        @testset "histogram_sparse!" begin
            bins_per_profile = 4
            first_left_cut = -12.0
            left_cut_distance = 8.0
            cut_width = 4.0
            filling_pattern_host = Bool[1, 0, 1, 0, 1, 0]
            bucket_index_to_memory_index_host = Int32[0, 0, 4, 4, 8, 8]
            n_out = 12
            x_host = vcat(
                collect(range(-10, 10; length=21)),
                # left and right edges of every filled window
                [-12.0, -12.0, 4.0, 4.0, 20.0, 20.0],
                [-8.0, 8.0, 24.0],
                # outside any window
                [-12.5, -15.9, -20.5, 8.5, 24.5],
            )
            expected = reference_histogram_sparse(
                x_host, n_out, first_left_cut, left_cut_distance, cut_width,
                bins_per_profile, filling_pattern_host,
                bucket_index_to_memory_index_host,
            )
            x = to_device(x_host)
            out = to_device(ones(Float64, n_out))
            filling_pattern = to_device(filling_pattern_host)
            bucket_index_to_memory_index =
                to_device(bucket_index_to_memory_index_host)
            @test (@inferred BLonDKernels.histogram_sparse!(
                device,
                raw_pointer(x),
                length(x_host),
                raw_pointer(out),
                n_out,
                first_left_cut,
                left_cut_distance,
                cut_width,
                bins_per_profile,
                raw_pointer(filling_pattern),
                length(filling_pattern_host),
                raw_pointer(bucket_index_to_memory_index),
            )) === nothing
            @test to_host(out) == expected
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.histogram_sparse!(
                    device, raw_pointer(x), length(x_host),
                    raw_pointer(out), n_out, first_left_cut,
                    left_cut_distance, cut_width, bins_per_profile,
                    raw_pointer(filling_pattern),
                    length(filling_pattern_host),
                    raw_pointer(bucket_index_to_memory_index),
                )
            end
        end

        @testset "move_flagged_elements_to_end!" begin
            for flags_host in (
                Int32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                Int32[0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
                Int32[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                Int32[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
                n = length(flags_host)
                dt_host = collect(range(0, 10; length=n))
                dE_host = collect(range(0, 10; length=n))
                ids_host = Int32.(0:(n - 1))
                flag = Int32(0)
                expected_n, _, expected_dt, _, _ =
                    reference_move_flagged_elements_to_end(
                        flag, flags_host, dt_host, dE_host, ids_host
                    )
                flags = to_device(copy(flags_host))
                dt = to_device(dt_host)
                dE = to_device(dE_host)
                ids = to_device(ids_host)
                n_new = @inferred BLonDKernels.move_flagged_elements_to_end!(
                    device,
                    flag,
                    raw_pointer(flags),
                    raw_pointer(dt),
                    raw_pointer(dE),
                    raw_pointer(ids),
                    n,
                )
                @test n_new isa Int
                @test n_new == expected_n
                flags_result = to_host(flags)
                dt_result = to_host(dt)
                dE_result = to_host(dE)
                ids_result = to_host(ids)
                @test all(flags_result[1:n_new] .!= flag)
                @test all(flags_result[(n_new + 1):end] .== flag)
                @test sort(dt_result[1:n_new]) ==
                      sort(expected_dt[1:expected_n])
                # dt, dE and ids must stay aligned with each other
                @test dt_result == dE_result
                @test dt_result ≈ dt_host[ids_result .+ 1]
            end
            if run_jet
                flags_host = Int32[0, 1, 0, 1]
                flags = to_device(flags_host)
                dt = to_device([1.0, 2.0, 3.0, 4.0])
                dE = to_device([1.0, 2.0, 3.0, 4.0])
                ids = to_device(Int32[0, 1, 2, 3])
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.move_flagged_elements_to_end!(
                    device, Int32(0), raw_pointer(flags), raw_pointer(dt),
                    raw_pointer(dE), raw_pointer(ids), 4,
                )
            end
        end

        @testset "wake_from_pole_residue!" begin
            for update_on_bin_host in
                (Int32[0], Int32[2], Int32[], Int32[0, 5, 9])
                n_bins = 16
                rng = MersenneTwister(42)
                profile_host = randn(rng, n_bins)
                centers_host = collect(range(0.0, 1e-9; length=n_bins))
                bin_dt = centers_host[2] - centers_host[1]
                # `profile_dts` must hold at least `n_bins + 1` entries
                profile_dts_host = vcat(centers_host, centers_host[end] +
                                                      bin_dt)
                poles_host = ComplexF64[
                    -1e8 + 2 * pi * 1e9im, -2e8 + 2 * pi * 1.5e9im
                ]
                residues_host = ComplexF64[1.0 + 0.5im, 0.7 - 0.2im]
                signs_host = [1.0, -1.0]
                states_host = ComplexF64[
                    0.3 + 0.1im, 0.0 + 0.0im,
                    complex(centers_host[1] - bin_dt),
                ]
                factor = 1.0
                for is_counterrotating in (false, true)
                    expected_voltage, expected_states =
                        reference_wake_from_pole_residue(
                            profile_host,
                            profile_dts_host,
                            poles_host,
                            residues_host,
                            is_counterrotating,
                            signs_host,
                            update_on_bin_host,
                            factor,
                            states_host,
                        )
                    profile = to_device(profile_host)
                    profile_dts = to_device(profile_dts_host)
                    poles = to_device(poles_host)
                    residues = to_device(residues_host)
                    signs = to_device(signs_host)
                    update_on_bin = to_device(update_on_bin_host)
                    states = to_device(copy(states_host))
                    voltage = to_device(ones(Float64, n_bins))
                    @test (@inferred BLonDKernels.wake_from_pole_residue!(
                        device,
                        raw_pointer(profile),
                        n_bins,
                        raw_pointer(profile_dts),
                        length(profile_dts_host),
                        raw_pointer(poles),
                        raw_pointer(residues),
                        length(poles_host),
                        is_counterrotating,
                        raw_pointer(signs),
                        raw_pointer(update_on_bin),
                        length(update_on_bin_host),
                        factor,
                        raw_pointer(states),
                        raw_pointer(voltage),
                    )) === nothing
                    @test to_host(voltage) ≈ expected_voltage rtol = 1e-12
                    @test to_host(states) ≈ expected_states rtol = 1e-12
                    if run_jet
                        @test_opt target_modules = (BLonDKernels,) BLonDKernels.wake_from_pole_residue!(
                            device, raw_pointer(profile), n_bins,
                            raw_pointer(profile_dts),
                            length(profile_dts_host), raw_pointer(poles),
                            raw_pointer(residues), length(poles_host),
                            is_counterrotating, raw_pointer(signs),
                            raw_pointer(update_on_bin),
                            length(update_on_bin_host), factor,
                            raw_pointer(states), raw_pointer(voltage),
                        )
                    end
                end
            end
        end

        @testset "apply_synchrotron_radiation!" begin
            n = 200_000
            initial_dE = 20e9
            energy_lost = 13e6
            longitudinal_damping_time = 14955.0
            natural_energy_spread = 1e-3
            total_energy = 20e9
            damping_factor = 1.0 - 2.0 / longitudinal_damping_time

            # deterministic branch (quantum excitation disabled)
            dE_host = fill(initial_dE, 1000)
            dE = to_device(dE_host)
            @test (@inferred BLonDKernels.apply_synchrotron_radiation!(
                device,
                raw_pointer(dE),
                length(dE_host),
                energy_lost,
                longitudinal_damping_time,
                natural_energy_spread,
                total_energy,
                true,
            )) === nothing
            @test same(
                to_host(dE),
                fill(damping_factor * initial_dE - energy_lost, 1000),
            )

            # noisy branch: mean and standard deviation must match
            dE_noisy = to_device(fill(initial_dE, n))
            BLonDKernels.apply_synchrotron_radiation!(
                device, raw_pointer(dE_noisy), n, energy_lost,
                longitudinal_damping_time, natural_energy_spread,
                total_energy, false,
            )
            result = to_host(dE_noisy)
            expected_mean = damping_factor * initial_dE - energy_lost
            expected_std =
                2.0 * natural_energy_spread /
                sqrt(longitudinal_damping_time) * total_energy
            sample_mean = sum(result) / n
            sample_std = sqrt(sum((result .- sample_mean) .^ 2) / (n - 1))
            @test abs(sample_mean - expected_mean) <
                  max(1e-4 * abs(expected_mean), 5e4)
            @test abs(sample_std / expected_std - 1.0) < 0.02

            # empty beam must be a no-op
            empty_dE = to_device(Float64[])
            @test BLonDKernels.apply_synchrotron_radiation!(
                device, raw_pointer(empty_dE), 0, energy_lost,
                longitudinal_damping_time, natural_energy_spread,
                total_energy, false,
            ) === nothing
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.apply_synchrotron_radiation!(
                    device, raw_pointer(dE), length(dE_host), energy_lost,
                    longitudinal_damping_time, natural_energy_spread,
                    total_energy, true,
                )
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.apply_synchrotron_radiation!(
                    device, raw_pointer(dE), length(dE_host), energy_lost,
                    longitudinal_damping_time, natural_energy_spread,
                    total_energy, false,
                )
            end
        end

        @testset "particle loops across chunk boundaries" begin
            # Several full chunks plus a partial one, so that the first,
            # the inner and the last chunk boundaries are all exercised.
            chunk_length = BLonDKernels.PARTICLES_PER_CHUNK
            n = 3 * chunk_length + 17
            dt_host = collect(range(-5e-9, 5e-9; length=n))
            dE_host = collect(range(1e9, 10e9; length=n))
            charge = 1.0
            acceleration_kick = -1.0

            expected = reference_kick_single_harmonic(
                dt_host, dE_host, 3e6, 2.5e9, 0.3, charge, acceleration_kick
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            BLonDKernels.kick_single_harmonic!(
                device, raw_pointer(dt), raw_pointer(dE), n, 3e6, 2.5e9, 0.3,
                charge, acceleration_kick,
            )
            @test to_host(dE) ≈ expected rtol = 1e-14

            for n_rf in 1:5
                voltage_host = [1e6 * j for j in 1:n_rf]
                omega_host = [2.5e9 * j for j in 1:n_rf]
                phi_host = [0.3 * j for j in 1:n_rf]
                expected = reference_kick_multi_harmonic(
                    dt_host, dE_host, voltage_host, omega_host, phi_host,
                    charge, acceleration_kick,
                )
                dE = to_device(dE_host)
                voltage = to_device(voltage_host)
                omega_rf = to_device(omega_host)
                phi_rf = to_device(phi_host)
                BLonDKernels.kick_multi_harmonic!(
                    device, raw_pointer(dt), raw_pointer(dE), n,
                    raw_pointer(voltage), raw_pointer(omega_rf),
                    raw_pointer(phi_rf), n_rf, charge, acceleration_kick,
                )
                @test to_host(dE) ≈ expected rtol = 1e-14
            end

            expected = reference_drift_simple(
                dt_host, dE_host, 5.0, 0.3, 0.9, 10.0
            )
            dt = to_device(dt_host)
            dE = to_device(dE_host)
            BLonDKernels.drift_simple!(
                device, raw_pointer(dt), raw_pointer(dE), n, 5.0, 0.3, 0.9,
                10.0,
            )
            @test same(to_host(dt), expected)

            longitudinal_damping_time = 14955.0
            damping_factor = 1.0 - 2.0 / longitudinal_damping_time
            dE = to_device(dE_host)
            BLonDKernels.apply_synchrotron_radiation!(
                device, raw_pointer(dE), n, 13e6, longitudinal_damping_time,
                1e-3, 20e9, true,
            )
            @test same(to_host(dE), damping_factor .* dE_host .- 13e6)

            # The beam phase sums its integrands chunk by chunk on the CPU.
            hist_x_host = collect(range(0.0, 1.0; length=n))
            hist_y_host = exp.(-((hist_x_host .- 0.5) ./ 0.2) .^ 2)
            bin_size = hist_x_host[2] - hist_x_host[1]
            expected_phase = reference_beam_phase(
                hist_x_host, hist_y_host, 0.5, 0.8, 0.1, bin_size
            )
            hist_x = to_device(hist_x_host)
            hist_y = to_device(hist_y_host)
            @test BLonDKernels.beam_phase(
                device, raw_pointer(hist_x), raw_pointer(hist_y), n, 0.5,
                0.8, 0.1, bin_size,
            ) ≈ expected_phase rtol = 1e-12

            # Quantum excitation: starting from zero energy without loss,
            # the result is the scaled noise itself. Every chunk must draw
            # its own numbers -- identically seeded chunks would repeat
            # the same noise and still pass the mean/std checks.
            n_chunks = 8
            noise_dE = to_device(zeros(Float64, n_chunks * chunk_length))
            BLonDKernels.apply_synchrotron_radiation!(
                device, raw_pointer(noise_dE), n_chunks * chunk_length, 0.0,
                longitudinal_damping_time, 1e-3, 20e9, false,
            )
            noise = to_host(noise_dE)
            chunks = [
                noise[((chunk - 1) * chunk_length + 1):(chunk * chunk_length)]
                for chunk in 1:n_chunks
            ]
            @test allunique(chunks)
            @test all(allunique, chunks)
            second_noise_dE = to_device(zeros(Float64, n_chunks * chunk_length))
            BLonDKernels.apply_synchrotron_radiation!(
                device, raw_pointer(second_noise_dE), n_chunks * chunk_length,
                0.0, longitudinal_damping_time, 1e-3, 20e9, false,
            )
            @test to_host(second_noise_dE) != noise
        end

        if device isa KernelAbstractions.CPU
            @testset "CPU particle loops beat per-particle kernels" begin
                # The chunked, vectorisable CPU path must clearly beat
                # launching the per-particle kernel (the previous CPU
                # implementation, still used on GPUs), timed relative to
                # each other on the same machine after compilation.
                n = 2^20
                dt = collect(range(-5.0, 5.0; length=n))
                dE = zeros(Float64, n)
                elapsed(run) = (run(); minimum(@elapsed(run()) for _ in 1:20))

                # The kicks gain by vectorisation, which `@inbounds` makes
                # possible. `Pkg.test` forces `--check-bounds=yes`, which
                # ignores `@inbounds`, so the comparison only means
                # something without it: run
                # `Pkg.test(julia_args=`--check-bounds=auto`)`.
                if Base.JLOptions().check_bounds == 1
                    @test_skip "kick timings need --check-bounds=auto"
                    @test_skip "kick timings need --check-bounds=auto"
                else
                    single_kernel! =
                        BLonDKernels.kick_single_harmonic_kernel!(device)
                    elapsed_single_entry = elapsed(
                        () -> BLonDKernels.kick_single_harmonic!(
                            device, raw_pointer(dt), raw_pointer(dE), n,
                            3.0, 2.0, 0.5, 1.0, 0.0,
                        ),
                    )
                    elapsed_single_kernel = elapsed(() -> begin
                        single_kernel!(
                            dt, dE, 3.0, 2.0, 0.5, 0.0,
                            BLonDKernels.SingleParticleLayout();
                            ndrange=n,
                        )
                        KernelAbstractions.synchronize(device)
                    end)
                    @test elapsed_single_entry < 0.8 * elapsed_single_kernel

                    voltage = [1e6, 5e6]
                    omega_rf = [2.0, 4.0]
                    phi_rf = [0.0, pi]
                    multi_kernel! =
                        BLonDKernels.kick_multi_harmonic_kernel!(device)
                    elapsed_multi_entry = elapsed(
                        () -> BLonDKernels.kick_multi_harmonic!(
                            device, raw_pointer(dt), raw_pointer(dE), n,
                            raw_pointer(voltage), raw_pointer(omega_rf),
                            raw_pointer(phi_rf), 2, 1.0, 0.0,
                        ),
                    )
                    elapsed_multi_kernel = elapsed(() -> begin
                        multi_kernel!(
                            dt, dE, voltage, omega_rf, phi_rf, 2, 1.0, 0.0,
                            BLonDKernels.SingleParticleLayout();
                            ndrange=n,
                        )
                        KernelAbstractions.synchronize(device)
                    end)
                    @test elapsed_multi_entry < 0.8 * elapsed_multi_kernel
                end

                # Generating the noise in parallel only pays off with
                # more than one thread.
                if Threads.nthreads() > 1
                    noise = similar(dE)
                    quantum_kernel! =
                        noise_array_quantum_excitation_kernel!(device)
                    elapsed_quantum_entry = elapsed(
                        () -> BLonDKernels.apply_synchrotron_radiation!(
                            device, raw_pointer(dE), n, 0.0, 14955.0, 1e-3,
                            20e9, false,
                        ),
                    )
                    elapsed_quantum_kernel = elapsed(() -> begin
                        randn!(noise)
                        quantum_kernel!(
                            dE, noise, 0.9998, 2e3, 0.0; ndrange=n
                        )
                        KernelAbstractions.synchronize(device)
                    end)
                    @test elapsed_quantum_entry < 0.8 * elapsed_quantum_kernel
                end
            end

            @testset "quantum excitation without noise array" begin
                # Drawing each particle's noise inside its chunk must beat
                # allocating and filling a beam-sized noise array first
                # (the previous CPU implementation). The gain is the saved
                # allocation, not vectorisation, so it also holds under
                # `--check-bounds=yes`.
                n = 2^20
                dE = zeros(Float64, n)
                elapsed(run) = (run(); minimum(@elapsed(run()) for _ in 1:20))
                elapsed_entry = elapsed(
                    () -> BLonDKernels.apply_synchrotron_radiation!(
                        device, raw_pointer(dE), n, 0.0, 14955.0, 1e-3,
                        20e9, false,
                    ),
                )
                elapsed_noise_array = elapsed(() -> begin
                    noise = similar(dE)
                    BLonDKernels.launch_particle_loop!(
                        device,
                        nothing,
                        noise_array_quantum_excitation_range!,
                        n,
                        (dE, noise, 0.9998, 2e3, 0.0),
                    )
                end)
                @test elapsed_entry < 0.75 * elapsed_noise_array
            end

            # Summing the integrands chunk by chunk in parallel pays off
            # through the threads (exp, sin and cos do not vectorise), so
            # it holds under `--check-bounds=yes` but needs threads.
            if Threads.nthreads() > 1
                @testset "beam_phase chunked sums" begin
                    n_bins = 2^18
                    hist_x = collect(range(0.0, 1.0; length=n_bins))
                    hist_y = exp.(-((hist_x .- 0.5) ./ 0.2) .^ 2)
                    bin_size = hist_x[2] - hist_x[1]
                    expected = reference_beam_phase(
                        hist_x, hist_y, 0.5, 0.8, 0.1, bin_size
                    )
                    run_entry() = BLonDKernels.beam_phase(
                        device, raw_pointer(hist_x), raw_pointer(hist_y),
                        n_bins, 0.5, 0.8, 0.1, bin_size,
                    )
                    @test run_entry() ≈ expected rtol = 1e-12
                    elapsed(run) =
                        (run(); minimum(@elapsed(run()) for _ in 1:20))
                    elapsed_entry = elapsed(run_entry)
                    elapsed_values_array = elapsed(
                        () -> values_array_beam_phase(
                            device, hist_x, hist_y, 0.5, 0.8, 0.1, bin_size
                        ),
                    )
                    @test elapsed_entry < 0.75 * elapsed_values_array
                end
            end
        end
    end
end

function run_music_track_tests(device; run_jet::Bool)
    @testset "music_track!" begin
        beam_dt = collect(range(1e-9, 10e-9; length=10))
        beam_dE_start = collect(range(1e9, 10e9; length=10))
        n = length(beam_dt)
        resonator_shunt_impedance = 1e6
        omega_resonator = 2 * pi * 1e9
        quality_factor = 1.0
        n_particles = 1e11
        elementary_charge = 1.602176634e-19
        alpha = omega_resonator / (2 * quality_factor)
        omega_bar = sqrt(omega_resonator^2 - alpha^2)
        const_factor =
            -elementary_charge * resonator_shunt_impedance *
            omega_resonator * n_particles / (n * quality_factor)
        coeff1 = -alpha / omega_bar
        coeff2 =
            -resonator_shunt_impedance * omega_resonator /
            (quality_factor * omega_bar)
        coeff3 =
            omega_resonator * quality_factor /
            (resonator_shunt_impedance * omega_bar)
        coeff4 = alpha / omega_bar
        time_since_last_track = 10.0

        for multiturn in (false, true)
            parameter_array_start = [1.0, 0.0, 0.0]
            expected_dE, expected_voltage, expected_parameters =
                reference_music_track(
                    beam_dt, beam_dE_start, parameter_array_start, alpha,
                    omega_bar, const_factor, coeff1, coeff2, coeff3, coeff4,
                    time_since_last_track, multiturn,
                )
            beam_dt_device = copy(beam_dt)
            beam_dE = copy(beam_dE_start)
            induced_voltage = zeros(Float64, n)
            parameter_array = copy(parameter_array_start)
            @test (@inferred BLonDKernels.music_track!(
                device,
                raw_pointer(beam_dt_device),
                raw_pointer(beam_dE),
                raw_pointer(induced_voltage),
                raw_pointer(parameter_array),
                n,
                alpha,
                omega_bar,
                const_factor,
                coeff1,
                coeff2,
                coeff3,
                coeff4,
                time_since_last_track,
                multiturn,
            )) === nothing
            @test induced_voltage == expected_voltage
            @test beam_dE == expected_dE
            @test parameter_array == expected_parameters
            if run_jet
                @test_opt target_modules = (BLonDKernels,) BLonDKernels.music_track!(
                    device, raw_pointer(beam_dt_device), raw_pointer(beam_dE),
                    raw_pointer(induced_voltage),
                    raw_pointer(parameter_array), n, alpha, omega_bar,
                    const_factor, coeff1, coeff2, coeff3, coeff4,
                    time_since_last_track, multiturn,
                )
            end
        end
    end
end

@testset verbose = true "BLonDKernels" begin
    @testset "Aqua quality assurance" begin
        # `persistent_tasks` spawns a fresh Julia process and precompiles
        # the package again; the remaining checks (ambiguities, unbound
        # arguments, undefined exports, stale/compat deps, type piracy)
        # are what guard the public surface.
        Aqua.test_all(BLonDKernels; persistent_tasks=false)
    end
    @testset "fast_sin" begin
        # Wide sweep plus the quadrant boundaries, where the range
        # reduction switches between the sine and cosine polynomial.
        arguments = vcat(
            collect(range(-1e4, 1e4; length=1_000_001)),
            [k * pi / 4 for k in -16:16],
            [0.0, -0.0, 1e-300, nextfloat(0.0), prevfloat(0.0)],
        )
        @test maximum(
            abs(BLonDKernels.fast_sin(x) - sin(x)) for x in arguments
        ) <= eps()
        @test all(
            BLonDKernels.fast_sin(-x) == -BLonDKernels.fast_sin(x)
            for x in arguments
        )
        @test (@inferred BLonDKernels.fast_sin(0.5)) isa Float64
        @test_opt target_modules = (BLonDKernels,) BLonDKernels.fast_sin(0.5)
    end
    @testset "philox4x32_10" begin
        # Outputs of `curand_Philox4x32_10` from NVIDIA's
        # `curand_philox4x32_x.h` (CUDA 12.9) built for the host; the first
        # three are also Random123's known-answer vectors.
        cases = [
            (
                (0x00000000, 0x00000000, 0x00000000, 0x00000000),
                (0x00000000, 0x00000000),
                (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8),
            ),
            (
                (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
                (0xffffffff, 0xffffffff),
                (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd),
            ),
            (
                (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
                (0xa4093822, 0x299f31d0),
                (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1),
            ),
            (
                (0x00000001, 0x00000000, 0x00000000, 0x00000000),
                (0x12345678, 0x9abcdef0),
                (0xeb897a36, 0x4fcdf6b6, 0xfba23d8c, 0x6eed5b47),
            ),
        ]
        for (counter, key, expected) in cases
            @test BLonDKernels.philox4x32_10(counter, key) == expected
        end
        counter, key, _ = first(cases)
        @test (@inferred BLonDKernels.philox4x32_10(counter, key)) isa
              NTuple{4, UInt32}
        @test_opt target_modules = (BLonDKernels,) BLonDKernels.philox4x32_10(
            counter, key
        )
    end
    @testset "particle layouts" begin
        # One particle per work-item (CPU kernels without a range function).
        single = BLonDKernels.SingleParticleLayout()
        @test collect(BLonDKernels.particle_indices(single, 7)) == [7]
        # Striding: 4 work-items share 10 particles, each item every 4th.
        stride = BLonDKernels.StrideLayout(Int32(10), Int32(4))
        @test collect(BLonDKernels.particle_indices(stride, 3)) == Int32[3, 7]
        @test collect(BLonDKernels.particle_indices(stride, 4)) == Int32[4, 8]
        visited = vcat(
            [collect(BLonDKernels.particle_indices(stride, item)) for item in 1:4]...,
        )
        @test sort(visited) == 1:10
        @test (@inferred BLonDKernels.particle_indices(stride, 2)) isa
              AbstractRange{Int32}
        # An item beyond the particles visits none (fewer particles than
        # work-items).
        @test isempty(BLonDKernels.particle_indices(stride, 11))
        # Floor as `Int32`, or -1 wherever the conversion is not exact.
        floor32 = BLonDKernels.bounded_floor_int32
        @test floor32(2.5) === Int32(2)
        @test floor32(-0.5) === Int32(-1)
        @test floor32(2147483647.5) === typemax(Int32)
        @test floor32(-2147483647.5) === typemin(Int32)
        @test floor32(2147483648.0) === Int32(-1)
        @test floor32(1e30) === Int32(-1)
        @test floor32(Inf) === Int32(-1)
        @test floor32(NaN) === Int32(-1)
    end
    host = @inferred BLonDKernels.host_device()
    @test host isa BLonDKernels.CPU
    @test_opt target_modules = (BLonDKernels,) BLonDKernels.host_device()
    @info "CPU test run" threads = Threads.nthreads()
    run_device_tests(
        "CPU", host, copy, identity_to_host; run_jet=true, strict_float=true
    )
    run_music_track_tests(host; run_jet=true)

    @testset "cuda_device fallback without CUDA" begin
        if !isdefined(Main, :CUDA)
            @test_throws ErrorException BLonDKernels.cuda_device()
        end
    end

    cuda_available = false
    try
        @eval using CUDA
        cuda_available = CUDA.functional()
    catch error
        @warn "CUDA not usable, skipping GPU tests" error
    end
    if cuda_available
        @info "CUDA test run" device = string(CUDA.device())
        cuda_backend = @inferred BLonDKernels.cuda_device()
        @testset "CUDA default stream" begin
            # `use_cuda_default_stream!` builds the stream object from its
            # fields, so it depends on this layout of `CuStream`.
            @test fieldnames(CUDA.CuStream) == (:handle, :valid, :ctx)
            BLonDKernels.use_cuda_default_stream!(cuda_backend)
            @test UInt(CUDA.stream().handle) == 0
            @test CUDA.stream().ctx === CUDA.context()
            @test BLonDKernels.synchronize_device(cuda_backend) === nothing
        end
        # Macros of CUDA.jl only exist once it is loaded at runtime.
        gpu_allocated = @eval run -> CUDA.@allocated run()
        run_device_tests(
            "CUDA",
            cuda_backend,
            Main.CUDA.CuArray,
            Array;
            run_jet=false,
            strict_float=false,
            gpu_allocated,
        )
        @testset "music_track! is CPU only" begin
            @test_throws MethodError BLonDKernels.music_track!(
                cuda_backend, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, false,
            )
        end
    else
        @warn "CUDA not functional -- GPU tests skipped"
    end
end
