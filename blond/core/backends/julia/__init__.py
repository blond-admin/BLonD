# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Julia (KernelAbstractions.jl) backends `julia_cpu` and `julia_gpu`.

The numeric kernels of these two backends are written once in Julia
(`BLonDKernels`, next to this file) and compiled by
KernelAbstractions.jl for the host CPU (`julia_cpu`, NumPy arrays) and
for CUDA (`julia_gpu`, CuPy arrays). BLonD keeps owning the arrays;
Julia only wraps their raw pointers.

Importing this package must stay cheap and must never start Julia --
`blond.core.backends.julia.julia_env.is_julia_available` is a pure
`importlib` probe, and the Julia session is only booted by
`ensure_julia_environment`.
"""
