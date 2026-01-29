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
from scipy.constants import c as c_light

from blond import SingleHarmonicRFStation, proton
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BLonD3Cavity,
    EnergyUpdate,
    blond_to_xsuite_transform,
)


def main():
    PLOTTING = True
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
    blen = 1.25e-9  # Bunch length [s]

    # Simulation parameters
    N_TURNS = 500
    input_dt = 2 * blen - 0.4e-9  # Input particles dt [s]
    input_dE = 0.0  # Input particles dE [eV]

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
    particles = line.build_particles(
        x=np.random.uniform(-1e-3, 1e-3, n_part),
        px=np.random.uniform(-1e-5, 1e-5, n_part),
        y=np.random.uniform(-2e-3, 2e-3, n_part),
        py=np.random.uniform(-3e-5, 3e-5, n_part),
        zeta=np.random.uniform(-2e-2, 2e-2, n_part),
        delta=np.random.uniform(-1e-4, 1e-4, n_part),
    )

    # --- BLonD3Element  --- #
    cavity1 = SingleHarmonicRFStation.headless(
        section_index=1,
        voltage=V,
        harmonic=h,
        phi_rf=-np.pi,  # todo, this is a shift between xsuite and blond
        circumference=C,
        total_energy=None,  # todo dynamically set the energy
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
    line.enable_time_dependent_vars = True
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
            line.record_last_track.delta[:, 0],
        )
        plt.scatter(
            line.record_last_track.zeta[:, -1],
            line.record_last_track.delta[:, -1],
        )
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
