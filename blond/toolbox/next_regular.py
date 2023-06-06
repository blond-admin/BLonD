
'''
**Optimised next_regular routine to generate Hamming numbers: 
the output will contain at least one factor 2.**

:Authors: **Alexandre Lasheen**, **Juan F. Esteban Mueller**, **Danilo Quartullo** 
'''


def next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    target = -(-target // 2)

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N * 2
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35 * 2
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5 * 2
    if p5 < match:
        match = p5

    return match * 2

# THIS IS THE ORIGINAL PYTHON VERSION #

# def next_regular(target):
#     """
#     Find the next regular number greater than or equal to target.
#     Regular numbers are composites of the prime factors 2, 3, and 5.
#     Also known as 5-smooth numbers or Hamming numbers, these are the optimal
#     size for inputs to FFTPACK.
#
#     Target must be a positive integer.
#     """
#     if target <= 6:
#         return target
#
#     # Quickly check if it's already a power of 2
#     if not (target & (target-1)):
#         return target
#
#     match = float('inf') # Anything found will be smaller
#     p5 = 1
#     while p5 < target:
#         p35 = p5
#         while p35 < target:
#             # Ceiling integer division, avoiding conversion to float
#             # (quotient = ceil(target / p35))
#             quotient = -(-target // p35)
#
#             # Quickly find next power of 2 >= quotient
#             try:
#                 p2 = 2**((quotient - 1).bit_length())
#             except AttributeError:
#                 # Fallback for Python <2.7
#                 p2 = 2**(len(bin(quotient - 1)) - 2)
#
#             N = p2 * p35
#             if N == target:
#                 return N
#             elif N < match:
#                 match = N
#             p35 *= 3
#             if p35 == target:
#                 return p35
#         if p35 < match:
#             match = p35
#         p5 *= 5
#         if p5 == target:
#             return p5
#     if p5 < match:
#         match = p5
#
#     return match
