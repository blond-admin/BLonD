# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Device hooks the entry points reach for instead of KernelAbstractions
# directly, so that a device extension can wrap arrays, launch kernels and
# hand out scratch buffers more cheaply. The kernels themselves stay
# untouched; `ext/BLonDKernelsCUDAExt.jl` specialises all three for CUDA.

"""
    wrap_kernel_array(device, T, pointer_as_int, n_elements)

Wrap foreign memory for use *only as a kernel argument*. Defaults to
[`wrap_array`]; a GPU extension may return a lighter device-side view that
supports no host-side array operations (`sum`, `fill!`, `similar`, ...).
"""
function wrap_kernel_array(
    device, ::Type{T}, pointer_as_int::Int, n_elements::Int
) where {T}
    return wrap_array(device, T, pointer_as_int, n_elements)
end

"""
    launch_kernel!(device, kernel_factory, ndrange, workgroupsize, arguments)

Launch `kernel_factory(device)` over `ndrange` work-items in workgroups of
`workgroupsize`, passing `arguments`. A GPU extension may launch the same
compiled kernel through a cheaper path.
"""
function launch_kernel!(
    device, kernel_factory, ndrange::Int, workgroupsize::Int, arguments::Tuple
)::Nothing
    kernel! = kernel_factory(device)
    kernel!(arguments...; ndrange=ndrange, workgroupsize=workgroupsize)
    return nothing
end

"""
    scratch_vector(device, T, n_elements, slot::Symbol)

Return a device vector of at least `n_elements` elements for temporary use
within one entry call. Defaults to a fresh allocation; a GPU extension may
hand out a cached, grow-only buffer per `slot`.
"""
function scratch_vector(device, ::Type{T}, n_elements::Int, ::Symbol) where {T}
    return KernelAbstractions.allocate(device, T, n_elements)
end
