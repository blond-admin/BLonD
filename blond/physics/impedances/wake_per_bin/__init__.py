# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Bin-averaged wakes of the time-domain wakefield sources.

The time-domain solvers apply the wake averaged with the quadratic B-spline
of one bin width (``get_wake_per_bin``) instead of the point-sampled wake;
:mod:`~blond.physics.impedances.wake_per_bin.bin_average` holds that average
in closed form. The mix-ins of the other modules here add it to the
interfaces of :mod:`blond.physics.impedances.base` and to the sources of
:mod:`blond.physics.impedances.sources`.
"""
