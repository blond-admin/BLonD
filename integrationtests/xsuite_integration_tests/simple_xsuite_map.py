# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Running a simple xsuite map, LHC like."""

from copy import deepcopy

import numpy as np
import xpart as xp
import xtrack as xt


def run_simulation(n_turns):
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

    # Bunch parameters
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

    xsuite_cavity = xt.Cavity(
        voltage=rf_voltage,
        frequency=400788731.3867354,
        lag=180,
    )
    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})
    line.insert_element(index=0, element=xsuite_cavity, name="xsuite_cavity")

    line.particle_ref = xp.Particles(
        p0c=synchronous_momentum, mass0=xp.PROTON_MASS_EV, q0=1.0
    )

    # --- Many particle  --- #
    n_part = 100
    rng = np.random.default_rng(seed=0)
    particles = line.build_particles(
        x=rng.uniform(-1e-3, 1e-3, n_part),
        px=rng.uniform(-1e-5, 1e-5, n_part),
        y=rng.uniform(-2e-3, 2e-3, n_part),
        py=rng.uniform(-3e-5, 3e-5, n_part),
        zeta=rng.uniform(-2e-2, 2e-2, n_part),
        delta=rng.uniform(-1e-4, 1e-4, n_part),
    )

    init_dist = {
        "x": deepcopy(particles.x),
        "px": deepcopy(particles.px),
        "y": deepcopy(particles.y),
        "py": deepcopy(particles.py),
        "zeta": deepcopy(particles.zeta),
        "delta": deepcopy(particles.delta),
    }

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
        init_dist,
    )
