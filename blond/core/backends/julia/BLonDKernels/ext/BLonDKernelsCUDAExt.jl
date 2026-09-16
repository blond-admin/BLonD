# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
    BLonDKernelsCUDAExt

CUDA support for `BLonDKernels`, loaded as soon as CUDA.jl is available.

It only adds the device object, the device-specific array wrapping and
the choice of stream -- every kernel itself is shared with the CPU through
KernelAbstractions.
"""
module BLonDKernelsCUDAExt

using BLonDKernels: BLonDKernels
using CUDA: CUDA, CuArray, CuPtr, CuStream

function BLonDKernels.cuda_device()::CUDA.CUDABackend
    return CUDA.CUDABackend()
end

BLonDKernels.max_threads(::CUDA.CUDABackend)::Int = 1

function BLonDKernels.gpu_workgroups(::CUDA.CUDABackend)::Int
    return Int(
        CUDA.CUDACore.attribute(
            CUDA.device(),
            CUDA.CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        ),
    )
end

function BLonDKernels.use_cuda_default_stream!(::CUDA.CUDABackend)::Nothing
    # `CUDA.default_stream()` carries no context, which the synchronization
    # of CUDA.jl needs, and `CuStream` has no public constructor for an
    # existing handle, so the stream object is built from its fields. No
    # finalizer is attached: the default stream is never destroyed.
    handle_type = fieldtype(CuStream, :handle)
    stream = ccall(
        :jl_new_struct,
        Any,
        (Any, Any...),
        CuStream,
        convert(handle_type, C_NULL),
        true,
        CUDA.context(),
    )::CuStream
    CUDA.stream!(stream)
    return nothing
end

function BLonDKernels.wrap_array(
    ::CUDA.CUDABackend, ::Type{T}, pointer_as_int::Int, n_elements::Int
) where {T}
    return unsafe_wrap(
        CuArray{T}, CuPtr{T}(pointer_as_int), n_elements; own=false
    )
end

# Cheap launches of the unchanged KernelAbstractions kernels.
#
# A stock launch of a small kernel costs ~3.8 µs, of which only ~2 µs are
# the driver's `cuLaunchKernelEx`; the rest is argument conversion, stream
# bookkeeping and the compiled-kernel lookup. `launch_kernel!` below caches
# the compiled kernel and its parameter buffer per (kernel, argument types)
# and calls the driver directly, with bitwise-identical results. Through
# the Python API a drift of 1e3 particles takes 5.4 instead of 8.8 µs (cuda
# backend: 5.7 µs), the dense interpolated kick 8.2 instead of 19.7 µs.
#
# The fast path relies on internals of CUDA.jl 6.3 (CUDACore) and
# KernelAbstractions 0.9: KA.launch_config, KA.mkcontext, KA.blocks,
# KA.workitems, the Kernel field `f`; CUDACore.cufunction, the HostKernel
# fields `fun`/`state`, the CuFunction field `handle`, KernelState,
# make_seed, task_local_state!, stream(state), the CuDeviceArray inner
# constructor, CUlaunchConfig, CUlaunchAttribute, CUresult, SUCCESS,
# libcuda; GPUCompiler.isghosttype; the CuArray fields `data`, `offset` and
# `maxsize`. If any of them is missing or has changed, the first launch
# throws, the fast path is switched off with a warning, and every launch
# goes through KernelAbstractions again.
#
# Stream ownership is not tracked: every array launched here must be used
# on the stream the kernels are queued on, which holds for CuPy memory on
# the default stream (the Python wrapper synchronises around calls from
# any other stream) and for buffers the entries allocate themselves.

const KA = BLonDKernels.KernelAbstractions
const CC = CUDA.CUDACore

@inline function device_view(::Type{T}, pointer_as_int::Int, n::Int) where {T}
    pointer = reinterpret(
        CC.LLVMPtr{T, CC.AS.Global}, CC.CuPtr{T}(pointer_as_int)
    )
    return CC.CuDeviceArray{T, 1, CC.AS.Global}(pointer, (n,), n * sizeof(T))
end

function BLonDKernels.wrap_kernel_array(
    ::CUDA.CUDABackend, ::Type{T}, pointer_as_int::Int, n_elements::Int
) where {T}
    return device_view(T, pointer_as_int, n_elements)
end

# CuArray -> CuDeviceArray without `cudaconvert`'s stream-ownership
# bookkeeping. Valid because every array launched here is used on the stream
# the kernels are queued on (CuPy memory on the default stream, or Julia
# buffers allocated on that stream by the entries).
@inline function to_device(array::CuArray{T, N}) where {T, N}
    pointer = convert(CC.CuPtr{T}, array.data[].mem) + array.offset
    return CC.CuDeviceArray{T, N, CC.AS.Global}(
        reinterpret(CC.LLVMPtr{T, CC.AS.Global}, pointer), size(array),
        array.maxsize - array.offset,
    )
end
@inline to_device(argument) = argument

# `context` and `kernel` are untyped, so that a changed CUDA.jl type cannot
# keep this extension from loading; the launch reads them behind a function
# barrier.
mutable struct LaunchSlot{A <: Tuple, N}
    const context::Any
    const kernel::Any
    const values::Base.RefValue{A}
    const pointers::Base.RefValue{NTuple{N, Ptr{Cvoid}}}
    const lock::Threads.SpinLock
end

const LAUNCH_SLOTS = IdDict{Any, Any}()
const LAUNCH_SLOTS_LOCK = ReentrantLock()
const LAUNCH_KERNEL_EX = Ref{Ptr{Cvoid}}(C_NULL)

function launch_kernel_ex()::Ptr{Cvoid}
    if LAUNCH_KERNEL_EX[] == C_NULL
        library = Base.Libc.Libdl.dlopen(CC.libcuda)
        LAUNCH_KERNEL_EX[] = Base.Libc.Libdl.dlsym(library, :cuLaunchKernelEx)
    end
    return LAUNCH_KERNEL_EX[]
end

is_dropped(T) = CC.GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T)

@generated slot_key(::Type{F}, ::Type{C}, ::Type{V}) where {F, C, V} =
    :($(Tuple{F, C, V.parameters...}))
@generated kernel_signature(::Type{C}, ::Type{V}) where {C, V} =
    :($(Tuple{C, V.parameters...}))
@generated function passed_type(::Type{C}, ::Type{V}) where {C, V}
    types = filter(!is_dropped, Any[CC.KernelState, C, V.parameters...])
    return :($(Tuple{types...}))
end
# The values `CUDACore.launch_converted` passes: ghost types dropped.
@generated function passed_values(state, context::C, values::V) where {C, V}
    expressions = Any[]
    is_dropped(CC.KernelState) || push!(expressions, :state)
    is_dropped(C) || push!(expressions, :context)
    for i in 1:fieldcount(V)
        is_dropped(fieldtype(V, i)) || push!(expressions, :(values[$i]))
    end
    return Expr(:tuple, expressions...)
end

@noinline function new_launch_slot(f, signature, device, context, ::Type{A}) where {A}
    kernel = CC.cufunction(
        f, signature; always_inline=device.always_inline, maxthreads=nothing
    )
    values = Ref{A}()
    base = Ptr{Cvoid}(pointer_from_objref(values))
    pointers = Ref(ntuple(i -> base + fieldoffset(A, i), fieldcount(A)))
    return LaunchSlot{A, fieldcount(A)}(
        context, kernel, values, pointers, Threads.SpinLock()
    )
end

@inline function launch_raw(config, slot::LaunchSlot, handle)
    return ccall(
        launch_kernel_ex(), CC.CUresult,
        (Ref{CC.CUlaunchConfig}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
        config, handle,
        Ptr{Ptr{Cvoid}}(pointer_from_objref(slot.pointers)), C_NULL,
    )
end

"""
    FAST_LAUNCH_STATE

Whether `launch_kernel!` takes the fast path: 0 until the first launch,
1 once a fast launch succeeded, -1 after it failed (see the notes above).
"""
const FAST_LAUNCH_STATE = Threads.Atomic{Int8}(0)

function BLonDKernels.launch_kernel!(
    device::CUDA.CUDABackend, kernel_factory, ndrange::Int,
    workgroupsize::Int, arguments::Tuple,
)::Nothing
    if FAST_LAUNCH_STATE[] >= 0
        try
            fast_launch!(device, kernel_factory, ndrange, workgroupsize, arguments)
            FAST_LAUNCH_STATE[] == 0 && (FAST_LAUNCH_STATE[] = 1)
            return nothing
        catch error
            FAST_LAUNCH_STATE[] == 1 && rethrow()
            FAST_LAUNCH_STATE[] = -1
            @warn "BLonDKernels: the fast GPU kernel launch is not supported " *
                  "by this CUDA.jl/KernelAbstractions version; falling back " *
                  "to the regular launch." exception = error
        end
    end
    kernel! = kernel_factory(device)
    kernel!(arguments...; ndrange=ndrange, workgroupsize=workgroupsize)
    return nothing
end

# Launches the very GPU function KernelAbstractions compiles, over the same
# grid, but caches the compiled kernel and the parameter buffer per (kernel,
# argument types) and calls the driver directly.
function fast_launch!(
    device::CUDA.CUDABackend, kernel_factory, ndrange::Int,
    workgroupsize::Int, arguments::Tuple,
)::Nothing
    kernel = kernel_factory(device)
    launch_ndrange, _, iterspace, _ =
        KA.launch_config(kernel, ndrange, workgroupsize)
    blocks = length(KA.blocks(iterspace))
    blocks == 0 && return nothing
    threads = length(KA.workitems(iterspace))
    context = KA.mkcontext(kernel, launch_ndrange, iterspace)
    values = map(to_device, arguments)
    F, C, V = typeof(kernel.f), typeof(context), typeof(values)
    A = passed_type(C, V)
    state = CC.task_local_state!()
    stream = CC.stream(state)
    found = get(LAUNCH_SLOTS, slot_key(F, C, V), nothing)
    slot = if found === nothing || (found::LaunchSlot{A}).context !== state.context
        Base.@lock LAUNCH_SLOTS_LOCK begin
            LAUNCH_SLOTS[slot_key(F, C, V)] = new_launch_slot(
                kernel.f, kernel_signature(C, V), device, state.context, A
            )
        end
    else
        found
    end::LaunchSlot{A}
    config = CC.CUlaunchConfig(
        blocks, 1, 1, threads, 1, 1, 0, stream.handle,
        Ptr{CC.CUlaunchAttribute}(C_NULL), 0,
    )
    result = launch_slot!(slot, slot.kernel, config, context, values, arguments, stream)
    if result != CC.SUCCESS
        # e.g. no current context on this OS thread: the regular launch
        # binds it, or reports the failure with CUDA.jl's diagnostics.
        kernel(arguments...; ndrange=ndrange, workgroupsize=workgroupsize)
    end
    return nothing
end

# Function barrier over the untyped `slot.kernel`.
function launch_slot!(
    slot::LaunchSlot, kernel, config, context, values, arguments, stream
)
    kernel_state = CC.KernelState(
        kernel.state.exception_info, CC.make_seed(kernel)
    )
    return GC.@preserve arguments stream Base.@lock slot.lock begin
        slot.values[] = passed_values(kernel_state, context, values)
        launch_raw(config, slot, kernel.fun.handle)
    end
end

const SCRATCH_VECTORS = Dict{Tuple{Symbol, DataType}, CuArray}()

function BLonDKernels.scratch_vector(
    ::CUDA.CUDABackend, ::Type{T}, n_elements::Int, slot::Symbol
) where {T}
    key = (slot, T)
    buffer = get(SCRATCH_VECTORS, key, nothing)
    if buffer === nothing || length(buffer) < n_elements
        buffer = CuArray{T}(undef, max(n_elements, 1024))
        SCRATCH_VECTORS[key] = buffer
    end
    return buffer::CuArray{T, 1}
end

end # module BLonDKernelsCUDAExt
