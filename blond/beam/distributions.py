# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to generate distributions**

:Authors: **Danilo Quartullo**, **Helga Timko**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**, **Theodoros Argyropoulos**,
          **Joel Repond**
"""

from .distribution_generators.singlebunch.bigaussian import bigaussian
from .distribution_generators.singlebunch.matched_from_distribution_function import matched_from_distribution_function
from .distribution_generators.singlebunch.matched_from_line_density import matched_from_line_density
from .distribution_generators.singlebunch.parabolic import parabolic

# allow import from distributions.py
matched_from_line_density = matched_from_line_density
matched_from_distribution_function = matched_from_distribution_function
bigaussian = bigaussian
parabolic = parabolic
