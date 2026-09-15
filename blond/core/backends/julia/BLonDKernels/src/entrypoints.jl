# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Entry points called from Python (through juliacall). All signatures are
# fully concrete: the device object first, every array as a raw pointer
# (`Int`) whose length is given by the neighbouring count argument, and
# every scalar as `Float64`, `Int`, `Int32` or `Bool`.
#
# Entries return as soon as their kernels are queued. On the CPU they have
# finished by then. On a GPU they run on the CUDA default stream, which CuPy
# uses as well (see `use_cuda_default_stream!`), so that CuPy work queued
# after the call sees their results without waiting; an entry returning a
# scalar waits for it.

"""
    launch_particle_loop!(device, per_particle_kernel, range_function!,
                          n_particles, arguments)

Run a loop over `n_particles` macro-particles on `device`.

On the CPU the particles are swept in chunks of `PARTICLES_PER_CHUNK` by
`range_function!`, one work-item per chunk, so that the compiler can
vectorise each sweep. On every other device `per_particle_kernel` is
launched with one work-item per particle, which is what a GPU parallelises
best. Both receive the same `arguments`.
"""
function launch_particle_loop!(
    device::CPU,
    _,
    range_function!,
    n_particles::Int,
    arguments::Tuple,
)::Nothing
    kernel! = particle_chunks_kernel!(device)
    # A chunk is a full workload of its own: `workgroupsize=1` keeps the
    # CPU backend from merging small ndranges into a single task.
    kernel!(
        range_function!,
        n_particles,
        arguments;
        ndrange=cld(n_particles, PARTICLES_PER_CHUNK),
        workgroupsize=1,
    )
    return nothing
end

function launch_particle_loop!(
    device,
    per_particle_kernel,
    _,
    n_particles::Int,
    arguments::Tuple,
)::Nothing
    kernel! = per_particle_kernel(device)
    kernel!(arguments...; ndrange=n_particles)
    return nothing
end

"""
    apply_quantum_excitation!(device, dE, n_macroparticles,
                              damping_factor, noise_scale, energy_lost)

Damp `dE` and add Gaussian quantum-excitation noise of `noise_scale`.

On the CPU every chunk draws its particles' noise one by one from its
task's generator. Other devices draw every particle's noise inside the
kernel, from a Philox4x32-10 stream keyed per call by the task's generator.
Neither allocates a beam-sized noise array.
"""
function apply_quantum_excitation!(
    device::CPU,
    dE,
    n_macroparticles::Int,
    damping_factor::Float64,
    noise_scale::Float64,
    energy_lost::Float64,
)::Nothing
    launch_particle_loop!(
        device,
        nothing,
        synchrotron_radiation_quantum_excitation_range!,
        n_macroparticles,
        (dE, damping_factor, noise_scale, energy_lost),
    )
    return nothing
end

function apply_quantum_excitation!(
    device,
    dE,
    n_macroparticles::Int,
    damping_factor::Float64,
    noise_scale::Float64,
    energy_lost::Float64,
)::Nothing
    key = (rand(UInt32), rand(UInt32))
    kernel! = synchrotron_radiation_quantum_excitation_kernel!(device)
    kernel!(
        dE,
        damping_factor,
        noise_scale,
        energy_lost,
        key;
        ndrange=n_macroparticles,
    )
    return nothing
end

"""
    kick_single_harmonic!(device, dt, dE, n_macroparticles, voltage,
                          omega_rf, phi_rf, charge, acceleration_kick)

Apply ``dE += charge * voltage * sin(omega_rf * dt + phi_rf) + kick``.

On the CPU the sine is evaluated with [`fast_sin`], as in the C++
backend.
"""
function kick_single_harmonic!(
    device,
    dt_pointer::Int,
    dE_pointer::Int,
    n_macroparticles::Int,
    voltage::Float64,
    omega_rf::Float64,
    phi_rf::Float64,
    charge::Float64,
    acceleration_kick::Float64,
)::Nothing
    n_macroparticles == 0 && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    launch_particle_loop!(
        device,
        kick_single_harmonic_kernel!,
        kick_single_harmonic_range!,
        n_macroparticles,
        (dt, dE, charge * voltage, omega_rf, phi_rf, acceleration_kick),
    )
    return nothing
end

"""
    kick_multi_harmonic!(device, dt, dE, n_macroparticles, voltage,
                         omega_rf, phi_rf, n_rf, charge, acceleration_kick)

Apply the RF kick of `n_rf` harmonics.

On the CPU the sine is evaluated with [`fast_sin`], as in the C++
backend.
"""
function kick_multi_harmonic!(
    device,
    dt_pointer::Int,
    dE_pointer::Int,
    n_macroparticles::Int,
    voltage_pointer::Int,
    omega_rf_pointer::Int,
    phi_rf_pointer::Int,
    n_rf::Int,
    charge::Float64,
    acceleration_kick::Float64,
)::Nothing
    n_macroparticles == 0 && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    voltage = wrap_array_or_empty(device, Float64, voltage_pointer, n_rf)
    omega_rf = wrap_array_or_empty(
        device, Float64, omega_rf_pointer, n_rf
    )
    phi_rf = wrap_array_or_empty(device, Float64, phi_rf_pointer, n_rf)
    launch_particle_loop!(
        device,
        kick_multi_harmonic_kernel!,
        kick_multi_harmonic_range!,
        n_macroparticles,
        (dt, dE, voltage, omega_rf, phi_rf, n_rf, charge, acceleration_kick),
    )
    return nothing
end

"""
    drift_simple!(device, dt, dE, n_macroparticles, drift_time, eta_0,
                  beta, energy)

Apply the linear drift equation of motion.
"""
function drift_simple!(
    device,
    dt_pointer::Int,
    dE_pointer::Int,
    n_macroparticles::Int,
    drift_time::Float64,
    eta_0::Float64,
    beta::Float64,
    energy::Float64,
)::Nothing
    n_macroparticles == 0 && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    coefficient = drift_time * (eta_0 / (beta * beta * energy))
    launch_particle_loop!(
        device,
        drift_simple_kernel!,
        drift_simple_range!,
        n_macroparticles,
        (dt, dE, coefficient),
    )
    return nothing
end

"""
    drift_exact!(device, dt, dE, n_macroparticles, drift_time, alpha_0,
                 higher_alpha, n_alpha, beta, energy)

Apply the exact drift equation of motion with higher-order momentum
compaction factors.
"""
function drift_exact!(
    device,
    dt_pointer::Int,
    dE_pointer::Int,
    n_macroparticles::Int,
    drift_time::Float64,
    alpha_0::Float64,
    higher_alpha_pointer::Int,
    n_alpha::Int,
    beta::Float64,
    energy::Float64,
)::Nothing
    n_macroparticles == 0 && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    higher_alpha = wrap_array_or_empty(
        device, Float64, higher_alpha_pointer, n_alpha
    )
    kernel! = drift_exact_kernel!(device)
    kernel!(
        dt,
        dE,
        drift_time,
        alpha_0,
        higher_alpha,
        n_alpha,
        1.0 / (beta * beta),
        1.0 / energy;
        ndrange=n_macroparticles,
    )
    return nothing
end

"""
    loss_box!(device, e_max, e_min, t_min, t_max, dt, dE, flags,
              n_macroparticles, lost_flag)

Flag every macro-particle outside the given time/energy box as lost.
"""
function loss_box!(
    device,
    e_max::Float64,
    e_min::Float64,
    t_min::Float64,
    t_max::Float64,
    dt_pointer::Int,
    dE_pointer::Int,
    flags_pointer::Int,
    n_macroparticles::Int,
    lost_flag::Int32,
)::Nothing
    n_macroparticles == 0 && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    flags = wrap_array(device, Int32, flags_pointer, n_macroparticles)
    kernel! = loss_box_kernel!(device)
    kernel!(
        dt,
        dE,
        flags,
        e_max,
        e_min,
        t_min,
        t_max,
        lost_flag;
        ndrange=n_macroparticles,
    )
    return nothing
end

"""
    sum_1d_array(device, array, n_elements) -> Float64

Return the sum of a 1d array.
"""
function sum_1d_array(
    device, array_pointer::Int, n_elements::Int
)::Float64
    n_elements == 0 && return 0.0
    array = wrap_array(device, Float64, array_pointer, n_elements)
    return sum(array)
end

"""
    dot_product_1d_array(device, array_1, array_2, n_elements) -> Float64

Return the dot product of two 1d arrays.
"""
function dot_product_1d_array(
    device, array_1_pointer::Int, array_2_pointer::Int, n_elements::Int
)::Float64
    n_elements == 0 && return 0.0
    array_1 = wrap_array(device, Float64, array_1_pointer, n_elements)
    array_2 = wrap_array(device, Float64, array_2_pointer, n_elements)
    return dot(array_1, array_2)
end

"""
    histogram!(device, array_read, n_read, array_write, n_bins, start, stop)

Histogram `array_read` into `n_bins` equidistant bins.

Values equal to `stop` are counted in the last bin, values outside
``[start, stop]`` are dropped.
"""
function histogram!(
    device,
    array_read_pointer::Int,
    n_read::Int,
    array_write_pointer::Int,
    n_bins::Int,
    start::Float64,
    stop::Float64,
)::Nothing
    n_bins == 0 && return nothing
    array_write = wrap_array(device, Float64, array_write_pointer, n_bins)
    fill!(array_write, 0.0)
    n_read == 0 && return nothing
    array_read = wrap_array(device, Float64, array_read_pointer, n_read)
    inverse_bin_width = n_bins / (stop - start)
    values_per_workgroup = histogram_values_per_workgroup()
    n_workgroups = cld(n_read, values_per_workgroup)
    # The workgroup size is passed at launch: building a statically
    # sized kernel from an `Int` is a runtime dispatch.
    kernel! = histogram_kernel!(device)
    need_atomics = private_bins_need_atomics(device)
    # One pass per window of `HISTOGRAM_LOCAL_BINS` bins; a single pass
    # for every histogram that fits into workgroup-local memory.
    for bin_offset in 0:HISTOGRAM_LOCAL_BINS:(n_bins - 1)
        n_local_bins = min(HISTOGRAM_LOCAL_BINS, n_bins - bin_offset)
        kernel!(
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
            need_atomics;
            ndrange=n_workgroups * HISTOGRAM_WORKGROUP_SIZE,
            workgroupsize=HISTOGRAM_WORKGROUP_SIZE,
        )
    end
    return nothing
end

"""
    beam_phase(device, hist_x, hist_y, n_bins, alpha, omega_rf, phi_rf,
               bin_size) -> Float64

Return the beam phase, the ratio of the sine- and cosine-weighted
trapezoidal integrals of the profile.
"""
function beam_phase(
    device,
    hist_x_pointer::Int,
    hist_y_pointer::Int,
    n_bins::Int,
    alpha::Float64,
    omega_rf::Float64,
    phi_rf::Float64,
    bin_size::Float64,
)::Float64
    n_bins == 0 && return 0.0 / 0.0
    hist_x = wrap_array(device, Float64, hist_x_pointer, n_bins)
    hist_y = wrap_array(device, Float64, hist_y_pointer, n_bins)
    return trapezoidal_beam_phase(
        device, hist_x, hist_y, n_bins, alpha, omega_rf, phi_rf, bin_size
    )
end

"""
    trapezoidal_beam_phase(device, hist_x, hist_y, n_bins, alpha, omega_rf,
                           phi_rf, bin_size) -> Float64

Ratio of the sine- and cosine-weighted trapezoidal integrals of a profile.

On the CPU the integrands are summed chunk by chunk in parallel, without
an intermediate array. Other devices store the integrands in a complex
array and reduce it.
"""
function trapezoidal_beam_phase(
    device::CPU,
    hist_x,
    hist_y,
    n_bins::Int,
    alpha::Float64,
    omega_rf::Float64,
    phi_rf::Float64,
    bin_size::Float64,
)::Float64
    chunk_sums = zeros(Float64, 2 * cld(n_bins, PARTICLES_PER_CHUNK))
    launch_particle_loop!(
        device,
        nothing,
        beam_phase_sums_range!,
        n_bins,
        (chunk_sums, hist_x, hist_y, alpha, omega_rf, phi_rf),
    )
    sine_sum = 0.0
    cosine_sum = 0.0
    @inbounds for chunk in 1:(length(chunk_sums) ÷ 2)
        sine_sum += chunk_sums[2 * chunk - 1]
        cosine_sum += chunk_sums[2 * chunk]
    end
    # Trapezoidal rule: the two end points count half.
    first_sine, first_cosine = beam_phase_integrand_sums(
        hist_x, hist_y, alpha, omega_rf, phi_rf, 1, 1
    )
    last_sine, last_cosine = beam_phase_integrand_sums(
        hist_x, hist_y, alpha, omega_rf, phi_rf, n_bins, n_bins
    )
    sine_integral = (sine_sum - 0.5 * (first_sine + last_sine)) * bin_size
    cosine_integral =
        (cosine_sum - 0.5 * (first_cosine + last_cosine)) * bin_size
    return sine_integral / cosine_integral
end

function trapezoidal_beam_phase(
    device,
    hist_x,
    hist_y,
    n_bins::Int,
    alpha::Float64,
    omega_rf::Float64,
    phi_rf::Float64,
    bin_size::Float64,
)::Float64
    values = similar(hist_x, ComplexF64)
    kernel! = beam_phase_values_kernel!(device)
    kernel!(
        values, hist_x, hist_y, alpha, omega_rf, phi_rf; ndrange=n_bins
    )
    # Trapezoidal rule without scalar indexing (which would stall a GPU):
    # the end points are halved through single-element reductions.
    edge_values =
        sum(@view values[1:1]) + sum(@view values[n_bins:n_bins])
    integral = (sum(values) - 0.5 * edge_values) * bin_size
    return imag(integral) / real(integral)
end

"""
    kick_interpolated_dense!(device, dt, dE, n_macroparticles, voltage,
                             bin_centers, n_slices, charge,
                             acceleration_kick)

Interpolated kick on a uniformly spaced `bin_centers` grid.
"""
function kick_interpolated_dense!(
    device,
    dt_pointer::Int,
    dE_pointer::Int,
    n_macroparticles::Int,
    voltage_pointer::Int,
    bin_centers_pointer::Int,
    n_slices::Int,
    charge::Float64,
    acceleration_kick::Float64,
)::Nothing
    # A single-bin (or empty) profile has no width to interpolate across,
    # so no particle can be kicked.
    (n_macroparticles == 0 || n_slices < 2) && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    voltage = wrap_array(device, Float64, voltage_pointer, n_slices)
    bin_centers = wrap_array(
        device, Float64, bin_centers_pointer, n_slices
    )
    factors = KernelAbstractions.allocate(device, Float64, 2 * (n_slices - 1))
    grid = KernelAbstractions.allocate(device, Float64, 2)
    factors_kernel! = kick_interpolated_dense_factors_kernel!(device)
    factors_kernel!(
        factors,
        grid,
        voltage,
        bin_centers,
        n_slices,
        charge,
        acceleration_kick;
        ndrange=n_slices - 1,
    )
    # Queued behind the factors kernel on the same device queue.
    particles_kernel! = kick_interpolated_dense_particles_kernel!(device)
    particles_kernel!(
        dt, dE, factors, grid, n_slices; ndrange=n_macroparticles
    )
    return nothing
end

"""
    kick_interpolated_sparse!(device, dt, dE, n_macroparticles, voltage,
                              bin_centers, n_slices, charge,
                              acceleration_kick, first_left_cut,
                              left_cut_distance, cut_width,
                              bins_per_profile, filling_pattern,
                              n_buckets, bucket_index_to_memory_index)

Interpolated kick on a gapped, multi-island `bin_centers` grid.
"""
function kick_interpolated_sparse!(
    device,
    dt_pointer::Int,
    dE_pointer::Int,
    n_macroparticles::Int,
    voltage_pointer::Int,
    bin_centers_pointer::Int,
    n_slices::Int,
    charge::Float64,
    acceleration_kick::Float64,
    first_left_cut::Float64,
    left_cut_distance::Float64,
    cut_width::Float64,
    bins_per_profile::Int,
    filling_pattern_pointer::Int,
    n_buckets::Int,
    bucket_index_to_memory_index_pointer::Int,
)::Nothing
    (n_macroparticles == 0 || n_buckets == 0) && return nothing
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    voltage = wrap_array(device, Float64, voltage_pointer, n_slices)
    bin_centers = wrap_array(
        device, Float64, bin_centers_pointer, n_slices
    )
    filling_pattern = wrap_array(
        device, Bool, filling_pattern_pointer, n_buckets
    )
    bucket_index_to_memory_index = wrap_array(
        device, Int32, bucket_index_to_memory_index_pointer, n_buckets
    )
    # Without two slices there is no bin to interpolate across.
    n_slices < 2 && return nothing
    inverse_bin_width = bins_per_profile / cut_width
    factors = KernelAbstractions.allocate(device, Float64, 2 * (n_slices - 1))
    factors_kernel! = kick_interpolated_sparse_factors_kernel!(device)
    factors_kernel!(
        factors,
        voltage,
        bin_centers,
        charge,
        acceleration_kick,
        inverse_bin_width;
        ndrange=n_slices - 1,
    )
    # Queued behind the factors kernel on the same device queue.
    particles_kernel! = kick_interpolated_sparse_particles_kernel!(device)
    particles_kernel!(
        dt,
        dE,
        factors,
        first_left_cut,
        left_cut_distance,
        bins_per_profile,
        filling_pattern,
        n_buckets,
        bucket_index_to_memory_index,
        1.0 / left_cut_distance,
        inverse_bin_width,
        cut_width / bins_per_profile;
        ndrange=n_macroparticles,
    )
    return nothing
end

"""
    histogram_sparse!(device, x, n_macroparticles, out, n_out,
                      first_left_cut, left_cut_distance, cut_width,
                      bins_per_profile, filling_pattern, n_buckets,
                      bucket_index_to_memory_index)

Sparse histogram with a strided memory layout (gaps between profiles).
"""
function histogram_sparse!(
    device,
    x_pointer::Int,
    n_macroparticles::Int,
    out_pointer::Int,
    n_out::Int,
    first_left_cut::Float64,
    left_cut_distance::Float64,
    cut_width::Float64,
    bins_per_profile::Int,
    filling_pattern_pointer::Int,
    n_buckets::Int,
    bucket_index_to_memory_index_pointer::Int,
)::Nothing
    n_out == 0 && return nothing
    out = wrap_array(device, Float64, out_pointer, n_out)
    fill!(out, 0.0)
    (n_macroparticles == 0 || n_buckets == 0) && return nothing
    x = wrap_array(device, Float64, x_pointer, n_macroparticles)
    filling_pattern = wrap_array(
        device, Bool, filling_pattern_pointer, n_buckets
    )
    bucket_index_to_memory_index = wrap_array(
        device, Int32, bucket_index_to_memory_index_pointer, n_buckets
    )
    kernel! = histogram_sparse_kernel!(device)
    kernel!(
        x,
        out,
        first_left_cut,
        left_cut_distance,
        cut_width,
        bins_per_profile,
        filling_pattern,
        n_buckets,
        bucket_index_to_memory_index,
        1.0 / left_cut_distance,
        bins_per_profile / cut_width;
        ndrange=n_macroparticles,
    )
    return nothing
end

"""
    move_flagged_elements_to_end!(device, flag, flags, dt, dE, ids,
                                  n_macroparticles) -> Int

Reorder the entries where ``flags == flag`` to the end of the arrays and
return the number of entries that are *not* flagged.

The reordering is a stable partition, computed with a prefix sum plus a
scatter, so that it is race-free on every device. `dt`, `dE`, `ids` and
`flags` stay aligned with each other.
"""
function move_flagged_elements_to_end!(
    device,
    flag::Int32,
    flags_pointer::Int,
    dt_pointer::Int,
    dE_pointer::Int,
    ids_pointer::Int,
    n_macroparticles::Int,
)::Int
    n_macroparticles == 0 && return 0
    flags = wrap_array(device, Int32, flags_pointer, n_macroparticles)
    dt = wrap_array(device, Float64, dt_pointer, n_macroparticles)
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    ids = wrap_array(device, Int32, ids_pointer, n_macroparticles)

    keep_mask = map(flag_value -> ifelse(flag_value != flag, 1, 0), flags)
    kept_positions = cumsum(keep_mask)
    n_kept = Int(sum(keep_mask))

    flags_source = copy(flags)
    dt_source = copy(dt)
    dE_source = copy(dE)
    ids_source = copy(ids)

    kernel! = move_flagged_scatter_kernel!(device)
    kernel!(
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
        n_kept;
        ndrange=n_macroparticles,
    )
    return n_kept
end

"""
    wake_from_pole_residue!(device, profile, n_bins, profile_dts,
                            n_profile_dts, poles, residues, n_poles,
                            is_counterrotating_beam,
                            counterrotating_pole_signs, update_on_bin,
                            n_updates, factor, states, voltage)

Apply an equivalent-circuit pole/residue model to `profile` to obtain the
induced `voltage`. One work item per pole; the per-bin contributions are
accumulated atomically.
"""
function wake_from_pole_residue!(
    device,
    profile_pointer::Int,
    n_bins::Int,
    profile_dts_pointer::Int,
    n_profile_dts::Int,
    poles_pointer::Int,
    residues_pointer::Int,
    n_poles::Int,
    is_counterrotating_beam::Bool,
    counterrotating_pole_signs_pointer::Int,
    update_on_bin_pointer::Int,
    n_updates::Int,
    factor::Float64,
    states_pointer::Int,
    voltage_pointer::Int,
)::Nothing
    voltage = wrap_array(device, Float64, voltage_pointer, n_bins)
    fill!(voltage, 0.0)
    states = wrap_array(device, ComplexF64, states_pointer, n_poles + 1)
    profile_dts = wrap_array(
        device, Float64, profile_dts_pointer, n_profile_dts
    )
    if n_poles > 0 && n_bins > 0
        profile = wrap_array(device, Float64, profile_pointer, n_bins)
        poles = wrap_array(device, ComplexF64, poles_pointer, n_poles)
        residues = wrap_array(
            device, ComplexF64, residues_pointer, n_poles
        )
        counterrotating_pole_signs = wrap_array(
            device, Float64, counterrotating_pole_signs_pointer, n_poles
        )
        update_on_bin = wrap_array_or_empty(
            device, Int32, update_on_bin_pointer, n_updates
        )
        kernel! = wake_from_pole_residue_kernel!(device)
        kernel!(
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
            2 * factor;
            ndrange=n_poles,
        )
    end
    # `states[end]` carries the end time of this call for the next one.
    store_kernel! = wake_store_end_time_kernel!(device)
    store_kernel!(states, profile_dts, n_poles, n_profile_dts; ndrange=1)
    return nothing
end

"""
    apply_synchrotron_radiation!(device, dE, n_macroparticles,
                                 energy_lost, longitudinal_damping_time,
                                 natural_energy_spread, total_energy,
                                 disable_quantum_excitation)

Apply the synchrotron radiation damping and (optionally) the quantum
excitation energy kick.
"""
function apply_synchrotron_radiation!(
    device,
    dE_pointer::Int,
    n_macroparticles::Int,
    energy_lost::Float64,
    longitudinal_damping_time::Float64,
    natural_energy_spread::Float64,
    total_energy::Float64,
    disable_quantum_excitation::Bool,
)::Nothing
    n_macroparticles == 0 && return nothing
    dE = wrap_array(device, Float64, dE_pointer, n_macroparticles)
    damping_factor = 1.0 - 2.0 / longitudinal_damping_time
    if disable_quantum_excitation
        launch_particle_loop!(
            device,
            synchrotron_radiation_kernel!,
            synchrotron_radiation_range!,
            n_macroparticles,
            (dE, damping_factor, energy_lost),
        )
    else
        noise_scale =
            2.0 * natural_energy_spread /
            sqrt(longitudinal_damping_time) * total_energy
        apply_quantum_excitation!(
            device,
            dE,
            n_macroparticles,
            damping_factor,
            noise_scale,
            energy_lost,
        )
    end
    return nothing
end

"""
    music_track!(device, beam_dt, beam_dE, induced_voltage,
                 parameter_array, n_macroparticles, alpha, omega_bar,
                 const_factor, coeff1, coeff2, coeff3, coeff4,
                 time_since_last_track, multiturn)

MuSiC O(n) resonator wake over the ascending-sorted macro-particles.

The recurrence is inherently sequential, so this entry is CPU-only, just
like the C++ and Python reference implementations.
"""
function music_track!(
    device::CPU,
    beam_dt_pointer::Int,
    beam_dE_pointer::Int,
    induced_voltage_pointer::Int,
    parameter_array_pointer::Int,
    n_macroparticles::Int,
    alpha::Float64,
    omega_bar::Float64,
    const_factor::Float64,
    coeff1::Float64,
    coeff2::Float64,
    coeff3::Float64,
    coeff4::Float64,
    time_since_last_track::Float64,
    multiturn::Bool,
)::Nothing
    n_macroparticles == 0 && return nothing
    beam_dt = wrap_array(device, Float64, beam_dt_pointer, n_macroparticles)
    beam_dE = wrap_array(device, Float64, beam_dE_pointer, n_macroparticles)
    induced_voltage = wrap_array(
        device, Float64, induced_voltage_pointer, n_macroparticles
    )
    parameter_array = wrap_array(
        device, Float64, parameter_array_pointer, 3
    )

    product_first = 0.0
    product_second = 0.0
    if multiturn
        # Bridge the wake from the previous turn across the rev. gap.
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
    end

    @inbounds induced_voltage[1] = const_factor * (0.5 + product_first)
    @inbounds beam_dE[1] += induced_voltage[1]

    input_first = product_first + 1.0
    input_second = product_second
    @inbounds for i in 1:(n_macroparticles - 1)
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
        beam_dE[i + 1] += induced_voltage[i + 1]
        input_first = product_first + 1.0
        input_second = product_second
    end

    @inbounds parameter_array[1] = input_first
    @inbounds parameter_array[2] = input_second
    @inbounds parameter_array[3] = beam_dt[n_macroparticles]
    return nothing
end
