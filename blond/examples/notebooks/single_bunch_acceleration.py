# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example of Basic Beam Acceleration
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import all blond features for later use.
    """)
    return


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt

    from blond import (
        Beam,
        BeamObservationOncePerTurn,
        DriftSimple,
        MagneticCyclePerTurn,
        RFStationPhaseObservation,
        Ring,
        Simulation,
        SingleHarmonicRFStation,
        backend,
        momentum_compaction_factor,
        proton,
    )
    from blond.experimental import (
        SemiEmpiricMatcher,
    )
    from blond.testing import pytest_active

    # When testing, some variables migh be changed for performace..
    pytest_is_active = pytest_active()
    return (
        Beam,
        BeamObservationOncePerTurn,
        DriftSimple,
        MagneticCyclePerTurn,
        RFStationPhaseObservation,
        Ring,
        SemiEmpiricMatcher,
        Simulation,
        SingleHarmonicRFStation,
        backend,
        momentum_compaction_factor,
        np,
        plt,
        proton,
        pytest_is_active,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the Beam Physics

    Define the beam physics in the synchrotron. In this example, we are analyzing at the LHC.
    """)
    return


@app.cell
def _(
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    np,
    proton,
):
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf_design = 0

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 451e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=55.759505
    )
    beam1 = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    # assembled in the cell that defines the elements, so that marimo knows
    # `sim` depends on them and never runs this before they exist
    sim = Simulation.from_locals(locals())
    return N_TURNS, beam1, rf_station, sim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All the previously defined elements are joined together for a simulation.
    """)
    return


@app.cell
def _(sim):
    sim.print_one_turn_execution_order()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate a Matched Beam
    Now that the physics processes in the synchrotron are defined, we can match the beam to the beam physics.
    """)
    return


@app.cell
def _(SemiEmpiricMatcher, beam1, pytest_is_active, sim):
    sim.prepare_beam(
        beam=beam1,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, 2.5e-9),
            n_macroparticles=1e3 if pytest_is_active else 1e6,
            seed=0,
            maxiter_intensity_effects=0,
            hamilton_to_density_kwargs=dict(
                density_modifier=2.0,  # Controls density profile sharpness
                hamilton_max=40.0,  # Hamiltonian cutoff [eV]
            ),
            animate=True,
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets have a closer look at the generated distribution.
    """)
    return


@app.cell
def _(beam1, plt):
    beam1.plot_hist2d()
    plt.xlabel("Time [s]")
    plt.ylabel("Energy Offset [eV]")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the Full Simulation
    Before running the simulation, we can define which properties to observe during the simulation.

    This will also reserve the memory, and if there are more observations than your computers RAM can hold, it will crash
    at the start of the simulation.
    """)
    return


@app.cell
def _(BeamObservationOncePerTurn, RFStationPhaseObservation, rf_station):
    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=int(1e3 - 1))
    return bunch_observation, phase_observation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's offset the matched beam from its stable position to make the results of this simulation more illustrative.
    """)
    return


@app.cell
def _(beam1):
    beam1._dt.array_local += 0.25e-9
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the Simulation

    We have defined the beam physics, what to observe, and our beam. Lets execute the simulation.
    """)
    return


@app.cell
def _(
    N_TURNS,
    backend,
    beam1,
    bunch_observation,
    phase_observation,
    pytest_is_active,
    sim,
):
    if not pytest_is_active:
        backend.set_specials("numba")

    sim.run_simulation(
        beams=(beam1,),
        n_turns=N_TURNS,
        observe=(phase_observation, bunch_observation),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze the Results
    After running the simulation, we can use the observations to analyze the beam dynamics results.
    """)
    return


@app.cell
def _(bunch_observation, plt):
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.title("Before Simulation")
    plt.hist2d(
        bunch_observation.dts[0, :],
        bunch_observation.dEs[0, :],
        bins=256,
        range=[[0, 2.5e-9], [-8e8, 8e8]],
    )
    plt.subplot(1, 2, 2, sharex=ax)
    plt.title("After Simulation")
    plt.hist2d(
        bunch_observation.dts[-1, :],
        bunch_observation.dEs[-1, :],
        bins=256,
        range=[[0, 2.5e-9], [-8e8, 8e8]],
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
