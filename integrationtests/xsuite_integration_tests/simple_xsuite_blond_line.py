# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Running a simple xsuite and blond simulation of SPS."""

import xtrack as xt

from blond import SingleHarmonicRFStation
from blond.handle_results.helpers import callers_relative_path
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BLonD3Cavity,
)


def run_simulation(n_turns: int, init_distribution: dict):
    """
    Run simple xsuite + blond element simulation.

    Parameters
    ----------
    init_distribution

    Returns
    -------
    particles.zeta,
    particles.delta
    """
    sps_line_folder = callers_relative_path(
        "./resources/line_no_spacecharge_and_particle.json", stacklevel=1
    )
    line = xt.load(sps_line_folder)
    length = line.get_length()

    line.set_particle_ref("proton", p0c=26e9)

    N_TURNS = n_turns

    bunch_intensity = 1e11

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=1,
        p0c=26e9,  # 7 TeV
        x=init_distribution["x"],
        px=init_distribution["px"],
        y=init_distribution["y"],
        py=init_distribution["py"],
        zeta=init_distribution["zeta"],
        delta=init_distribution["delta"],
    )

    # --- BLonD3Element  --- #
    cavity1 = SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=3e6,
        harmonic=4620,
        phi_rf=0,
        circumference=line.get_length(),
        total_energy=None,  # todo dynamically set the energy
        is_below_transition=None,
    )

    cavity = BLonD3Cavity(
        cavity=cavity1,
        particles=particles,
        line=line,
        initial_intensity=bunch_intensity,
    )

    phi_s = cavity.calc_phi_s()
    omega_rf = cavity1.calc_main_harmonic_omega_rf(
        beam_beta=cavity.beam.reference.beta, ring_circumference=length
    )
    print("phi_s", "omega_rf", phi_s, omega_rf)

    tab = line.get_table()
    tab_cav = tab.rows[tab.element_type == "Cavity"]
    for nn in tab_cav.name:
        line[nn].voltage = 0

    # --- Insert cavity  --- #
    line.insert_element(
        at="acta.31637",
        element=cavity,
        name="BLonD_Cavity",
    )

    line.build_tracker()

    line.track(
        particles,
        num_turns=N_TURNS,
        turn_by_turn_monitor=True,
        with_progress=True,
    )

    return (
        line.record_last_track.zeta.copy(),
        line.record_last_track.delta.copy(),
    )
