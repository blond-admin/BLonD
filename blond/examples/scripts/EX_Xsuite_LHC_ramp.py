# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

import numpy as np
import xpart as xp
import xtrack as xt
from scipy.constants import c

from blond import SingleHarmonicRFStation
from blond.interfaces.xsuite import BLonD3Cavity


def main():
    n_turns = 10
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    V = 5e6  # RF voltage [V]
    h = 35640  # harmonic number

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
        length=C,
    )

    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})

    t_rev = 26658.8832 / c
    p0c_ramp = np.linspace(450e9, 451e9, n_turns)
    t_s = np.linspace(0, t_rev * n_turns, n_turns)

    line.particle_ref = xp.Particles(p0c=p_s, mass0=xp.PROTON_MASS_EV, q0=1.0)

    tw = line.twiss(method="4d")
    print(tw["momentum_compaction_factor"])

    line.energy_program = xt.EnergyProgram(t_s=t_s, p0c=p0c_ramp)

    # xsuite_cavity = xt.Cavity(voltage=V, frequency=400788731.3867354, lag=0)

    # link rf cavity to the ramp
    # t_rf = np.linspace(0, t_rev * n_turns, n_turns)
    # f_rev = line.energy_program.get_frev_at_t_s(t_rf)
    # h_rf = harmonic
    # f_rf = h_rf * f_rev

    # --- BLonD3Element  --- #
    cavity1 = SingleHarmonicRFStation.headless(
        section_index=1,
        voltage=V,
        harmonic=h,
        phi_rf=0,
        circumference=C,
        total_energy=None,  #
        is_below_transition=None,
    )

    n_part = 20
    rng = np.random.default_rng()

    particles = line.build_particles(
        x=rng.uniform(low=-1e-3, high=1e-3, size=n_part),
        px=rng.uniform(-1e-5, 1e-5, n_part),
        y=rng.uniform(-2e-3, 2e-3, n_part),
        py=rng.uniform(-3e-5, 3e-5, n_part),
        zeta=rng.uniform(-100, 100, n_part),
        delta=rng.uniform(-1e-4, 1e-4, n_part),
    )

    blond_cavity = BLonD3Cavity(
        cavity=cavity1,
        particles=particles,
        line=line,
        initial_intensity=1e6,
    )

    # line.insert_element(index=0, element=xsuite_cavity, name="xsuite_cavity")
    line.insert_element(index=0, element=blond_cavity, name="xsuite_cavity")

    # line.functions["fun_f_rf"] = xt.FunctionPieceWiseLinear(x=t_rf, y=f_rf)
    # line["xsuite_cavity"].frequency = line.functions["fun_f_rf"](
    #    line.ref["t_turn_s"]
    # )

    line.enable_time_dependent_vars = True
    line.build_tracker()
    # the reference energy is updated by xsuite, we should now check the reference energy in BLonD

    line.track(particles=particles, num_turns=2)
    # but is it updated at each element, or at each RF station?
    # if it is not smooth, then it will not work


if __name__ == "__main__":
    main()
