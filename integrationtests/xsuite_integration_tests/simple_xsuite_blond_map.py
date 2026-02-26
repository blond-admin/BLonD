# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Running a simple xsuite map, LHC like."""

import xpart as xp
import xtrack as xt

from blond import SingleHarmonicRFStation
from blond.interfaces.xsuite.physics.blond_element_for_xsuite import (
    BLonD3Cavity,
)


def run_simulation(init_dist, n_turns):
    """
    Run xsuite only simulation.

    Returns
    -------
    line.record_last_track.zeta[:, -1]
    line.record_last_track.delta[:, -1]
    init_dist
    """
    circumference = 26658.8832
    synchronous_momentum = 450e9
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    rf_voltage = 5e6
    harmonic = 35640

    # Bunch parameters
    intensity = 1.15e11  # Intensity # where is this used in xtrack?

    # Simulation parameters
    N_TURNS = n_turns

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
        length=circumference,
    )

    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})
    line.particle_ref = xp.Particles(
        p0c=synchronous_momentum, mass0=xp.PROTON_MASS_EV, q0=1.0
    )

    # --- Many particle  --- #
    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=1,
        p0c=synchronous_momentum,
        x=init_dist["x"],
        px=init_dist["px"],
        y=init_dist["y"],
        py=init_dist["py"],
        zeta=init_dist["zeta"],
        delta=init_dist["delta"],
    )

    # --- BLonD3Element  --- #
    cavity1 = SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=rf_voltage,
        harmonic=harmonic,
        phi_rf=0,
        circumference=circumference,
        total_energy=None,  # todo dynamically set the energy
        is_below_transition=None,
        beam_reference_beta=line.particle_ref.beta0,
    )

    cavity = BLonD3Cavity(
        cavity=cavity1,
        particles=particles,
        line=line,
        initial_intensity=intensity,
    )

    phi_s = cavity.calc_phi_s()

    # --- Insert cavity  --- #
    line.insert_element(
        index=0,
        element=cavity,
        name="BLonD_Cavity",
    )

    line.build_tracker()
    line.get_table().show()

    line.track(
        particles,
        num_turns=N_TURNS,
        turn_by_turn_monitor=True,
        with_progress=True,
    )

    return (
        line.record_last_track.zeta.copy(),
        line.record_last_track.delta.copy(),
        phi_s,
    )
