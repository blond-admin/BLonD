# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Running a simple xsuite simulation of SPS."""

from copy import deepcopy

import numpy as np
import xtrack as xt

from blond.handle_results.helpers import callers_relative_path


def run_simulation(n_turns):
    """
    Run xsuite only simulation.

    Returns
    -------
    line.record_last_track.zeta[:, -1]
    line.record_last_track.delta[:, -1]
    init_dist
    """
    sps_line_folder = callers_relative_path(
        "./resources/line_no_spacecharge_and_particle.json", stacklevel=1
    )
    line = xt.load(sps_line_folder)

    line.set_particle_ref("proton", p0c=26e9)

    tab = line.get_table()
    tab_cav = tab.rows[tab.element_type == "Cavity"]
    for nn in tab_cav.name:
        line[nn].lag = 180
        line[nn].frequency = 200266136.96284187

    N_TURNS = n_turns

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
