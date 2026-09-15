# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Counter-based random numbers for the GPU kernels.
#
# A port of Philox4x32-10 (J. K. Salmon et al., Random123), the generator
# of NVIDIA's `curand_philox4x32_x.h`, from which the `cuda` backend draws
# its quantum-excitation noise. It maps a 128-bit counter and a 64-bit key
# to 128 random bits with integer arithmetic only, so every particle can
# draw its own number inside a kernel, on any device, without any state.

const PHILOX_M4X32_0 = 0xd2511f53
const PHILOX_M4X32_1 = 0xcd9e8d57
const PHILOX_W32_0 = 0x9e3779b9
const PHILOX_W32_1 = 0xbb67ae85

@inline function philox4x32_round(
    counter::NTuple{4, UInt32}, key::NTuple{2, UInt32}
)::NTuple{4, UInt32}
    product_0 = UInt64(PHILOX_M4X32_0) * counter[1]
    product_1 = UInt64(PHILOX_M4X32_1) * counter[3]
    return (
        (product_1 >> 32) % UInt32 ⊻ counter[2] ⊻ key[1],
        product_1 % UInt32,
        (product_0 >> 32) % UInt32 ⊻ counter[4] ⊻ key[2],
        product_0 % UInt32,
    )
end

"""
    philox4x32_10(counter, key) -> NTuple{4, UInt32}

Return the Philox4x32-10 block of `counter` (four words) under `key`
(two words), bit-identical to `curand_Philox4x32_10` of cuRAND.
"""
@inline function philox4x32_10(
    counter::NTuple{4, UInt32}, key::NTuple{2, UInt32}
)::NTuple{4, UInt32}
    key_1, key_2 = key
    for _ in 1:9
        counter = philox4x32_round(counter, (key_1, key_2))
        key_1 += PHILOX_W32_0
        key_2 += PHILOX_W32_1
    end
    return philox4x32_round(counter, (key_1, key_2))
end

"""
    philox_standard_normal(index, key) -> Float64

Return standard normal deviate number `index` of the stream `key`: the
Box-Muller transform of the Philox4x32-10 block of counter `index`.
"""
@inline function philox_standard_normal(
    index::Int, key::NTuple{2, UInt32}
)::Float64
    bits = philox4x32_10(
        (index % UInt32, (index >> 32) % UInt32, 0x00000000, 0x00000000),
        key,
    )
    # Two 53-bit uniforms; the first in (0, 1] so that its logarithm is
    # finite.
    uniform_open =
        ((((UInt64(bits[1]) << 32) | bits[2]) >> 11) + 1) * 0x1p-53
    uniform = (((UInt64(bits[3]) << 32) | bits[4]) >> 11) * 0x1p-53
    return sqrt(-2.0 * log(uniform_open)) * cos(2.0 * pi * uniform)
end
