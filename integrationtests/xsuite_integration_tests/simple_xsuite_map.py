# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Running a simple xsuite map, LHC like."""

from copy import deepcopy
import xpart as xp
import xtrack as xt

def main(n_turns):
    """
    Run xsuite only simulation.

    Returns
    -------
    line.record_last_track.zeta[:, -1]
    line.record_last_track.delta[:, -1]
    init_dist
    """

    PLOTTING = True
    # Parameters #
    # Accelerator parameters
    C = 26658.8832  # Machine circumference [m]
    p_s = 450e9  # Synchronous momentum [eV/c]
    p_f = 450e9  # Synchronous momentum, final
    h = 35640  # Harmonic number [-]
    alpha = 0.00034849575112251314  # First order mom. comp. factor [-]
    V = 5e6  # RF voltage [V]

    # Bunch parameters
    N_p = 1.15e11  # Intensity # where is this used in xtrack?

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
        length=C,
    )

    xsuite_cavity = xt.Cavity()
    xsuite_cavity.voltage = V
    xsuite_cavity.frequency = 400.78962
    xsuite_cavity.lag = 0#3.141592653589793*180/np.pi

    # Create line
    line = xt.Line(elements=[matrix], element_names={"matrix"})
    line.insert_element(index=0, element=xsuite_cavity, name='xsuite_cavity')

    line.particle_ref = xp.Particles(p0c=p_s, mass0=xp.PROTON_MASS_EV, q0=1.0)


    # --- Many particle  --- #
    n_part = 1
    particles = line.build_particles(
        x=[0],
        px=[0],
        y=[0],
        py=[0],
        zeta=[0.1e-2],
        delta=[0.1e-4]
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
        line.record_last_track.ptau.copy(),
        init_dist
    )


if __name__ == '__main__':
    main(10)