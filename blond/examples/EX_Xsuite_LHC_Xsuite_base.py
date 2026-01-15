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

from blond import (
    Beam,
    SingleHarmonicRfStation,
    proton,

)

from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BLonDElement3, EnergyUpdate, blond_to_xsuite_transform, xsuite_to_blond_transform
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
    dphi = 0  # Phase modulation/offset [rad]

    # Bunch parameters
    N_bunches = 1  # Number of bunches [-]
    N_m = 1  # Number of macroparticles [-]
    N_p = 1.15e11  # Intensity
    blen = 1.25e-9  # Bunch length [s]
    energy_err = 100e6  # Beamenergy error [eV/c]

    # Simulation parameters
    N_TURNS = 330
    N_buckets = 2  # Number of buckets [-]
    dt_plt = 30  # Timestep between plots [-]
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
    line.particle_ref = xp.Particles(p0c=p_s, mass0=xp.PROTON_MASS_EV, q0=1.0)


    # Create necessary blond objects
    cavity1 = SingleHarmonicRfStation()
    cavity1.harmonic = h
    cavity1.voltage = V
    cavity1.phi_rf = 0
    cavity1._turn_i = 0  # needed to initialise
    cavity1.phi_s = 0
    cavity1.calc_omega()

    beam = Beam(intensity=N_p,
                particle_type=proton)

    beam.setup_beam(dt=[input_dt], dE=[input_dE], reference_time=0, reference_total_energy=line.particle_ref.energy0)

    # BLonD3 element
    cavity = BLonDElement3(trackable=cavity1, update_zeta=True, beam=beam)

    line.insert_element(index=0,
        element=cavity, name="BLonD_Cavity",
    )

    # Insert energy ramp
    momentum = np.linspace(p_s, p_f, N_TURNS)
    energy_update = EnergyUpdate(momentum=momentum)

    line.insert_element(
        index="matrix", element=energy_update, name="energy_update"
    )

    # Add particles to line and build tracker
    line.build_tracker()

    # Show table
    line.get_table().show()


    # Simulating ----------------------------------------------------------------------------------------------------------
    print(f"\nSetting up simulation...")

    # --- Convert the initial BLonD distribution to xsuite coordinates ---
    zeta, ptau = blond_to_xsuite_transform(
        beam._dt,
        beam._dE,
        line.particle_ref.beta0[0],
        line.particle_ref.energy0[0],
        phi_s=cavity1.phi_s,
        omega_rf=cavity1._omega_rf,
    )


    # --- Track matrix ---
    particles = line.build_particles(
        x=0, y=0, px=0, py=0, zeta=np.copy(zeta), ptau=np.copy(ptau)
    )

    line.track(
        particles, num_turns=N_TURNS, turn_by_turn_monitor=True, with_progress=True
    )

    mon = line.record_last_track



if __name__ == "__main__":  # pragma: no cover
    main()


