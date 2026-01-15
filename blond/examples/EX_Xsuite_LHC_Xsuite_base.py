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
from scipy.constants import c

from blond import (
    Beam,
    SingleHarmonicRfStation,
    proton,

)
from blond.handle_results.helpers import callers_relative_path
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BlondElement3, EnergyUpdate, blond_to_xsuite_transform, xsuite_to_blond_transform
)





def main():


    cavity1 = SingleHarmonicRfStation()
    cavity1.harmonic = 4620
    cavity1.voltage = V_200
    cavity1.phi_rf = 0
    cavity1._turn_i = 0  # needed to initialise

    p0c = 13.5e9 * 82
    q0 = 82

    mass0 = line.particle_ref.mass0

    line.particle_ref = xp.Particles(mass0=mass0, q0=q0, p0c=p0c)

    num_particles = 1000
    nemitt_x = 2e-6
    nemitt_y = 2e-6

    x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(num_particles)
    y_in_sigmas, py_in_sigmas = xp.generate_2D_gaussian(num_particles)

    zeta, delta = xp.generate_longitudinal_coordinates(
        num_particles=num_particles,
        distribution="gaussian",
        sigma_z=bunch_length * 3e8 / 2,
        line=line,
    )

    particles = line.build_particles(
        zeta=zeta,
        delta=delta,
        x_norm=x_in_sigmas,
        px_norm=px_in_sigmas,
        y_norm=y_in_sigmas,
        py_norm=py_in_sigmas,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
    )

    N_TURNS = int(10)

    beam = Beam(intensity=intensity,
                particle_type=lead_82)
    dt = -particles.zeta / (particles.beta0 * c)
    dE = particles.ptau * particles.beta0 * line.particle_ref.energy0

    beam.setup_beam(dt=dt, dE=dE, reference_time=0, reference_total_energy=line.particle_ref.energy0)

    cavity = BlondElement3(trackable=cavity1, update_zeta=True, beam=beam)

    line[
        "actcse.31632"
    ].voltage = 0  # xsuite cavity =0, use BLonD element longitudinal
    line.insert_element(
        element=cavity, name="BLonD_Cavity_200MHz", at="actcse.31632"
    )  # BLonD inserted there

    zeta_record = []
    delta_record = []

    for turn in range(N_TURNS):
        line.track(particles, num_turns=1)

    zeta_record.append(particles.zeta.copy())
    delta_record.append(particles.delta.copy())

    print("done tracking.")


if __name__ == "__main__":  # pragma: no cover
    main()


