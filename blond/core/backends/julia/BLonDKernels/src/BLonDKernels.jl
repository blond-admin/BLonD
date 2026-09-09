# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
    BLonDKernels

Backend-agnostic numeric kernels for BLonD, written once with
KernelAbstractions.jl so that the same source runs on CPU threads and on
GPU devices (CUDA through the `BLonDKernelsCUDAExt` extension).

BLonD (Python) owns all arrays; they are handed over as a raw pointer
(`Int`) plus a length (`Int`) and wrapped zero-copy by [`wrap_array`].
"""
module BLonDKernels

using Atomix: Atomix
using KernelAbstractions: KernelAbstractions, CPU, @index, @kernel
using LinearAlgebra: dot
using Random: randn!

export host_device,
    cuda_device,
    max_threads,
    synchronize_device,
    wrap_array

"""
    host_device() -> KernelAbstractions.CPU

Return the KernelAbstractions device for multi-threaded CPU execution.
"""
host_device()::CPU = CPU()

"""
    cuda_device()

Return the KernelAbstractions device for NVIDIA GPUs.

This fallback only reports that the CUDA extension is not loaded; the
real method is defined in `ext/BLonDKernelsCUDAExt.jl`. It is
deliberately variadic: a fallback with the very same signature as the
extension's method would be a method *overwrite*, which Julia rejects
during precompilation.
"""
function cuda_device(unsupported_arguments...)
    error(
        "BLonDKernels: the CUDA device is only available once CUDA.jl is " *
        "loaded. Run `using CUDA` (and make sure `CUDA.functional()` is " *
        "true) before calling `cuda_device()`.",
    )
end

"""
    max_threads(device) -> Int

Return the number of threads the kernels of `device` may use.
"""
max_threads(::CPU)::Int = Threads.nthreads()

"""
    synchronize_device(device) -> Nothing

Block until all kernels queued on `device` have finished.
"""
function synchronize_device(device)::Nothing
    KernelAbstractions.synchronize(device)
    return nothing
end

"""
    wrap_array(device, T, pointer_as_int, n_elements)

Wrap foreign memory as a device-native array without copying.

One method per device type, so that the concrete array type is inferable
from the type of `device` alone.
"""
function wrap_array(
    ::CPU, ::Type{T}, pointer_as_int::Int, n_elements::Int
) where {T}
    return unsafe_wrap(
        Array, Ptr{T}(pointer_as_int), n_elements; own=false
    )
end

"""
    wrap_array_or_empty(device, T, pointer_as_int, n_elements)

Like [`wrap_array`], but return a device-native empty array when there is
nothing to wrap. BLonD may hand over a null pointer for an empty array
(CuPy does), which `unsafe_wrap` refuses.
"""
function wrap_array_or_empty(
    device, ::Type{T}, pointer_as_int::Int, n_elements::Int
) where {T}
    if n_elements <= 0 || pointer_as_int == 0
        return KernelAbstractions.allocate(device, T, 0)
    end
    return wrap_array(device, T, pointer_as_int, n_elements)
end

include("kernels.jl")
include("entrypoints.jl")

end # module BLonDKernels
