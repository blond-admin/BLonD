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
    # Parameters ----------------------------------------------------------------------------------------------------------
    # Accelerator parameters
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    p_f = 450.1e9  # Synchronous momentum, final
    h = 35640  # Harmonic number [-]
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    V = 5e6  # RF voltage [V]

    # Bunch parameters
    N_p = 1.15e11  # Intensity # where is this used in xtrack?
    blen = 1.25e-9  # Bunch length [s]

    # Simulation parameters
    N_TURNS = 330
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
    line["matrix"].length = C
    print("length", line.get_length())
    line.particle_ref = xp.Particles(p0c=p_s, mass0=xp.PROTON_MASS_EV, q0=1.0)

    # Create necessary blond objects
    momentum = np.linspace(p_s, p_f, N_TURNS)

    omega_rf = 2 * np.pi * c_light * h * line.particle_ref.beta0 / C

    # --- Convert the initial BLonD distribution to xsuite coordinates ---
    zeta, ptau = blond_to_xsuite_transform(
        dt=input_dt,
        de=input_dE,
        beta0=line.particle_ref.beta0[0],
        energy0=line.particle_ref.energy0[0],
        phi_s=0,
        omega_rf=omega_rf,
    )

    # --- Track matrix ---
    particles = line.build_particles(
        x=0,
        y=0,
        px=0,
        py=0,
        zeta=np.copy(zeta),
        ptau=np.copy(ptau),
    )
    # BLonD3 element --------------------------------
    cavity1 = SingleHarmonicRFStation.headless(
        section_index=1,
        voltage=V,
        harmonic=h,
        phi_rf=0,
        circumference=C,
        total_energy=float(
            line.particle_ref.energy0
        ),  # todo fix data types in headless for xsuite
    )
    cavity = BLonD3Cavity(
        cavity=cavity1,
        update_zeta=True,
        particles=particles,
        line=line,
        initial_intensity=N_p,
        particle_type=proton,
    )
    # -----------------------------------------------
    line.insert_element(
        index=0,
        element=cavity,
        name="BLonD_Cavity",
    )

    # Insert energy ramp ----------------------------
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


if __name__ == "__main__":  # pragma: no cover
    main()
