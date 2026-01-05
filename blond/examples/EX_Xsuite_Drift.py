# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import logging

import numpy as np
import xtrack as xt

from blond import (
    Beam,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    lead_82,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.handle_results.helpers import callers_relative_path
from blond.interfaces.xsuite.physics.xsuite_drift import DriftXsuite

logging.basicConfig(level=logging.INFO)


def main():
    xsuite_folder = callers_relative_path(
        "./resources/xsuite_lines/SPS_2021_Pb_nominal.json", stacklevel=1
    )
    line = xt.Line.from_json(f"{xsuite_folder}")
    line.cycle(
        name_first_element="actcse.31632"
    )  # make the start at the cavity

    voltage = 3e6
    line["actcse.31632"].voltage = 0  # xsuite cavity =0, only use drift

    ring = Ring(2 * np.pi * 1100.009)
    momentum = 1.4e12

    # slip_factor = -0.017166999
    # mom_compaction = 0.0015175

    cavity1 = SingleHarmonicRfStation()
    cavity1.harmonic = 4620
    cavity1.voltage = voltage
    cavity1.phi_rf = 0

    N_TURNS = int(1e3)

    beam1 = Beam(
        intensity=1.5e11,
        particle_type=lead_82,

    )

    beam1.reference_total_energy = 1e6

    energy_cycle = MagneticCyclePerTurn(
        value_init=momentum,
        values_after_turn=np.linspace(momentum, momentum, N_TURNS),
        reference_particle=lead_82,
    )

    drift1 = DriftXsuite(
        line=line,
        beam=beam1,
    )

    one_turn_map = []
    one_turn_map.extend([cavity1, drift1])
    ring.add_elements(one_turn_map)
    sim = Simulation(
        ring=ring,
        magnetic_cycle=energy_cycle,
    )
    sim.print_one_turn_execution_order()




if __name__ == "__main__":  # pragma: no cover
    main()
