# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numpy as np
import xpart as xp
import xtrack as xt

from blond import SingleHarmonicRFStation, proton
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BLonD3Cavity,
    EnergyUpdate,
)


def main():
    PLOTTING = False
    # Parameters #
    # Accelerator parameters
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    p_f = 450e9  # Synchronous momentum, final
    h = 35640  # Harmonic number [-]
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    V = 5e6  # RF voltage [V]

    # Bunch parameters
    N_p = 1.15e11  # Intensity # where is this used in xtrack?

    # Simulation parameters
    N_TURNS = 100

    # Make First order matrix map (takes care of drift in Xsuite)
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
    line.particle_ref = xp.Particles(p0c=p_s, mass0=xp.PROTON_MASS_EV, q0=1.0)

    momentum = np.linspace(p_s, p_f, N_TURNS)

    # --- Many particle  --- #
    n_part = 100

    rng = np.random.default_rng()
    particles = line.build_particles(
        x=rng.uniform(low=-1e-3, high=1e-3, size=n_part),
        px=rng.uniform(-1e-5, 1e-5, n_part),
        y=rng.uniform(-2e-3, 2e-3, n_part),
        py=rng.uniform(-3e-5, 3e-5, n_part),
        zeta=rng.uniform(-2e-2, 2e-2, n_part),
        delta=rng.uniform(-1e-4, 1e-4, n_part),
    )

    # --- BLonD3Element  --- #
    cavity1 = SingleHarmonicRFStation.headless(
        section_index=1,
        voltage=V,
        harmonic=h,
        phi_rf=0,
        circumference=C,
        total_energy=None,  #
        is_below_transition=None,
    )

    cavity = BLonD3Cavity(
        cavity=cavity1,
        particles=particles,
        line=line,
        initial_intensity=N_p,
    )

    # --- Insert cavity  --- #
    line.insert_element(
        index=0,
        element=cavity,
        name="BLonD_Cavity",
    )

    # --- Insert energy ramp  --- #
    energy_update = EnergyUpdate(momentum=momentum)

    line.insert_element(
        index="matrix", element=energy_update, name="energy_update"
    )

    line.build_tracker()
    line.get_table().show()

    line.track(
        particles,
        num_turns=N_TURNS,
        turn_by_turn_monitor=True,
        with_progress=True,
    )

    if PLOTTING:
        from matplotlib import pyplot as plt

        plt.scatter(
            line.record_last_track.zeta[:, 0],
            line.record_last_track.ptau[:, 0],
        )
        plt.scatter(
            line.record_last_track.zeta[:, -1],
            line.record_last_track.ptau[:, -1],
        )
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
