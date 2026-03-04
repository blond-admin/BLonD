# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Ramp Blond Xsuite LHC Map Integration Tests."""


# pragma: no cover

from copy import deepcopy

import numpy as np
import xpart as xp
import xtrack as xt
from scipy.constants import c


def run_simulation(n_turns: int):
    """Xsuite only ramp."""
    circumference = 26658.8832
    synchronous_momentum = 450e9
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    rf_voltage = 5e6
    harmonic = 35640

    # Make First order matrix map (takes care of drift in Xsuite)
    matrix = xt.LineSegmentMap(
        longitudinal_mode="nonlinear",
        qx=1.1,
        qy=1.2,
        betx=1.0,
        bety=1.0,
        voltage_rf=0,  # why dont we just add it here???
        frequency_rf=0,
        lag_rf=0,
        momentum_compaction_factor=alpha,
        length=circumference,
    )

    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})

    t_rev = 26658.8832 / c
    p0c_ramp = np.linspace(450e9, 450.1e9, n_turns)
    t_s = np.linspace(0, t_rev * n_turns, n_turns)

    line.particle_ref = xp.Particles(
        p0c=synchronous_momentum, mass0=xp.PROTON_MASS_EV, q0=1.0
    )
    line.energy_program = xt.EnergyProgram(t_s=t_s, p0c=p0c_ramp)

    xsuite_cavity = xt.Cavity(
        voltage=rf_voltage, frequency=400788731.3867354, lag=0
    )

    line.insert_element(index=0, element=xsuite_cavity, name="xsuite_cavity")

    # link rf cavity to the ramp
    t_rf = np.linspace(0, t_rev * n_turns, n_turns)
    f_rev = line.energy_program.get_frev_at_t_s(t_rf)
    h_rf = harmonic
    f_rf = h_rf * f_rev

    n_part = 400
    rng = np.random.default_rng()

    particles = line.build_particles(
        x=rng.uniform(low=-1e-3, high=1e-3, size=n_part),
        px=rng.uniform(-1e-5, 1e-5, n_part),
        y=rng.uniform(-2e-3, 2e-3, n_part),
        py=rng.uniform(-3e-5, 3e-5, n_part),
        zeta=np.linspace(-1.5, 1.5, n_part),
        delta=np.linspace(-1e-4, 1e-4, n_part) * 0,
    )

    # return initial distribution for the next simulation
    init_distribution = {
        "x": deepcopy(particles.x),
        "px": deepcopy(particles.px),
        "y": deepcopy(particles.y),
        "py": deepcopy(particles.py),
        "zeta": deepcopy(particles.zeta),
        "delta": deepcopy(particles.delta),
    }

    line.functions["fun_f_rf"] = xt.FunctionPieceWiseLinear(x=t_rf, y=f_rf)
    line["xsuite_cavity"].frequency = line.functions["fun_f_rf"](
        line.ref["t_turn_s"]
    )

    line.enable_time_dependent_vars = True
    line.build_tracker()
    # the reference energy is updated by xsuite, we should now check the reference energy in BLonD

    line.track(
        particles=particles,
        num_turns=n_turns,
        turn_by_turn_monitor=True,
        with_progress=True,
    )
    # but is it updated at each element, or at each RF station?
    # if it is not smooth, then it will not work

    return (
        init_distribution,
        line.record_last_track.zeta.copy(),
        line.record_last_track.delta.copy(),
    )
