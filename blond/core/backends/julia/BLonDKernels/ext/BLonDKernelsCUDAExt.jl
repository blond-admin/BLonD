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

end # module BLonDKernelsCUDAExt
