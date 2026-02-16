# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Untested/unstable code that might be changed in the future."""

__all__ = [
    "FilamentationMatcher",
    "SemiEmpiricMatcher",
    "VariNoise",
]
import warnings

from blond.experimental.beam_preparation.filamentation_matcher import (
    FilamentationMatcher,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.experimental.cycles.noise_generators.vari_noise import (
    VariNoise,
)
from blond.generals.warnings_ import ExperimentalFeaturesWarning

_msg = """
Importing experimental features. These are under development and are
liable to change or be removed without warning.
"""

warnings.warn(_msg, ExperimentalFeaturesWarning, stacklevel=1)
