# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Ramp Blond Xsuite LHC Map Integration Tests."""

import numpy as np
import xpart as xp
import xtrack as xt

# pragma: no cover
from scipy.constants import c

from blond import SingleHarmonicRFStation
from blond.interfaces.xsuite import BLonD3Cavity


def run_simulation(n_turns: int, init_distribution: dict):
    """Xsuite and BLonD ramp."""
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
        voltage_rf=0,
        frequency_rf=0,
        lag_rf=0,
        momentum_compaction_factor=alpha,
        length=circumference,
    )

    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})

    # calculate real t_rev
    t_rev = 26658.8832 / c

    t_s = np.linspace(0, t_rev * n_turns, n_turns)

    # linear ramp over first (n_turns - 1)
    p0c_ramp = np.array([450e9, 450.001e9, 450.002e9, 450.003e9, 450e9])

    line.particle_ref = xp.Particles(
        p0c=synchronous_momentum, mass0=xp.PROTON_MASS_EV, q0=1.0
    )

    tw = line.twiss(method="4d")
    alpha_0 = tw["momentum_compaction_factor"]

    line.energy_program = xt.EnergyProgram(
        t_s=t_s, p0c=p0c_ramp
    )  # make it more relativistic?

    # --- BLonD3Element  --- #

    cavity1 = SingleHarmonicRFStation.headless(
        section_index=1,
        voltage=rf_voltage,
        harmonic=harmonic,
        phi_rf=0,
        circumference=circumference,
        total_energy=None,
        is_below_transition=None,
        beam_reference_beta=float(line.particle_ref.beta0[0]),
    )

    particles = line.build_particles(
        x=init_distribution["x"],
        px=init_distribution["px"],
        y=init_distribution["y"],
        py=init_distribution["py"],
        zeta=init_distribution["zeta"],
        delta=init_distribution["delta"],
    )

    blond_cavity = BLonD3Cavity(
        cavity=cavity1,
        particles=particles,
        line=line,
        initial_intensity=1e6,
        momentum_compaction_factor=alpha_0,
    )

    phi_s = cavity1.calc_phi_s_main_harmonic(beam=blond_cavity._beam)
    print("phi_s", phi_s)

    line.insert_element(index=0, element=blond_cavity, name="xsuite_cavity")

    line.enable_time_dependent_vars = True
    line.build_tracker()

    line.track(
        particles=particles,
        num_turns=n_turns,
        turn_by_turn_monitor=True,
        with_progress=True,
    )

    return (
        line.record_last_track.zeta.copy(),
        line.record_last_track.delta.copy(),
    )
