"""
**Module to generate multibunch distributions**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**, **Theodoros Argyropoulos**
"""

from .distribution_generators.multibunch.match_beam_from_distribution import match_beam_from_distribution
from .distribution_generators.multibunch.match_beam_from_distribution_multibatch import \
    match_beam_from_distribution_multibatch
from .distribution_generators.multibunch.matched_from_distribution_density_multibunch import \
    matched_from_distribution_density_multibunch
from .distribution_generators.multibunch.matched_from_line_density_multibunch import \
    matched_from_line_density_multibunch
from .distribution_generators.multibunch.methods import compute_H0
from .distribution_generators.multibunch.methods import compute_x_grid
from .distribution_generators.multibunch.methods import match_a_bunch

# allow import from distributions_multibunch.py
match_beam_from_distribution_multibatch = match_beam_from_distribution_multibatch
matched_from_distribution_density_multibunch = matched_from_distribution_density_multibunch
matched_from_line_density_multibunch = matched_from_line_density_multibunch
match_beam_from_distribution = match_beam_from_distribution
match_beam_from_distribution_multibatch = match_beam_from_distribution_multibatch
compute_x_grid = compute_x_grid
compute_H0 = compute_H0
match_a_bunch = match_a_bunch
