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

@kernel function histogram_kernel!(
    array_read, array_write, n_bins, start, stop, inverse_bin_width
)
    i = @index(Global, Linear)
    @inbounds begin
        value = array_read[i]
        if value == stop
            # The right-most edge belongs to the last bin.
            Atomix.@atomic array_write[n_bins] += 1.0
        else
            bin_float = floor((value - start) * inverse_bin_width)
            # Range-check in floating point: converting an out-of-range
            # `Float64` to `Int` is undefined behaviour.
            if bin_float >= 0.0 && bin_float < n_bins
                bin_index = unsafe_trunc(Int, bin_float) + 1
                Atomix.@atomic array_write[bin_index] += 1.0
            end
        end
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
