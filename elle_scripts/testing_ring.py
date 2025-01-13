# %%

from blond.beam.beam import Proton
from matplotlib import pyplot as plt

import sys
import unittest

import numpy as np

from blond.beam.beam import Electron
from blond.input_parameters.ring import Ring
from blond.input_parameters.ring_options import convert_data

C = 2 * np.pi * 1100.009  # Ring circumference [m]
gamma_t = 18.0  # Gamma at transition
alpha = 1 / gamma_t ** 2  # Momentum compaction factor
n_turns = 10
n_sections = 8
l_per_section = C/ n_sections
section_lengths = np.full(n_sections, l_per_section)
mon_inj = 1*1e10
mon_ext = 10*1e10

momentum = np.zeros(n_sections * (n_turns + 1))
momentum[0:(n_sections - 1)] = mon_inj
# linear momentum program
momentum[n_sections - 1:] = np.linspace(mon_inj, mon_ext, num=(n_sections * (n_turns + 1)) - (n_sections - 1))
momentum = momentum.reshape(n_turns + 1, n_sections).T


# %%

ring = Ring(ring_length=section_lengths, alpha_0=alpha, 
            synchronous_data=momentum, particle=Proton(), 
            n_turns=n_turns, n_sections=n_sections,
                    synchronous_data_type='momentum')


print(ring.delta_E)

# %%
