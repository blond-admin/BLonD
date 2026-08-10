# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""LHC VariNoise data (spectral-shape ``gain_y`` for band-limited RF noise)."""

__all__ = ["lhc_spectrum_gain_y"]

import os
from importlib.resources import as_file, files

import numpy as np

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

with as_file(
    files("blond.specifics.cern.lhc.varinoise") / "lhc_spectrum_gain_y.txt"
) as spectrum_file:
    lhc_spectrum_gain_y = np.loadtxt(spectrum_file)
