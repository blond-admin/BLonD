# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Bin-averaged wakes of the time-domain wakefield sources.

A wake averaged with the quadratic B-spline of one bin width, instead of
sampled at the bin centres;
:mod:`~blond.physics.impedances.wake_per_bin.bin_average` holds that average
in closed form.
"""
