# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import logging
import os

import numpy as np
import xtrack as xt

from blond import (
    Beam,
    BiGaussian,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.handle_results.helpers import callers_relative_path
from blond.handle_results.observables_as_elements import (
    BeamObservationInRingElement,
)
from blond.interfaces.xsuite.physics.xsuite_drift import DriftXsuite

logging.basicConfig(level=logging.INFO)


def main():
    PLOTTING = True

    # Accelerator parameters
    try:
        os.mkdir(callers_relative_path(filename="./results/", stacklevel=1))
    except:
        pass
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    h = 35640  # Harmonic number [-]
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    V = 5e6  # RF voltage [V]
    N_TURNS = 1000

    # Bunch parameters
    N_m = 1000  # Number of macroparticles [-]
    N_p = 1.15e11  # Intensity
    blen = 1.25e-9  # Bunch length [s]


    # xsuite elements ------------------------------------------------
    matrix = xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0,
        frequency_rf=0,
        lag_rf=0,
        momentum_compaction_factor=alpha,
        length=C,
    )

    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})
    line["matrix"].length = C
    line.build_tracker()

    # ---------------------------------------------------------------

    ring = Ring(C)

    cavity1 = SingleHarmonicRfStation()
    cavity1.harmonic = h
    cavity1.voltage = V
    cavity1.phi_rf = 0

    beam1 = Beam(
        intensity=N_p,
        particle_type=proton,
    )

    beam1.reference_total_energy = p_s  # reference total energy vs momentum?

    energy_cycle = MagneticCyclePerTurn(
        value_init=p_s,
        values_after_turn=np.linspace(p_s, p_s, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftXsuite(
        line=line,
        beam=beam1,
        orbit_length=C,
        momentum_compaction_factor=alpha,
    )

    observation = BeamObservationInRingElement(
        each_turn_i=1,
        section_index=0,
        n_turns=2,
        folder=callers_relative_path(filename="./results/", stacklevel=1),
        name=f"observable_{0}",
    )

    one_turn_map = []
    one_turn_map.extend([cavity1, drift1, observation])
    ring.add_elements(one_turn_map)

    sim = Simulation(
        ring=ring,
        magnetic_cycle=energy_cycle,
    )

    sim.print_one_turn_execution_order()

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=BiGaussian(
            sigma_dt=blen * 20,
            sigma_dE=1e-10,
            reinsertion=False,
            seed=1,
            n_macroparticles=N_m,
        ),
    )

    sim.run_simulation(n_turns=N_TURNS, beams=[beam1])
    observation.to_disk()

    if PLOTTING:
        from matplotlib import pyplot as plt

        plt.scatter(observation.dEs[0], observation.dts[0])
        plt.scatter(observation.dEs[-1], observation.dts[-1])
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
