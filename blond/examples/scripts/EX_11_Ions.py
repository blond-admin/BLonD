# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

"""
Heavy-ion acceleration in a SIS100-like machine.

Demonstrates ParticleType for U-28+ ions and a linear momentum ramp from
injection (153 GeV/c) to extraction (535 GeV/c). Mirrors the legacy EX_07
ion example in the new API.

Notes
-----
Authors:
Alexandre Lasheen (legacy EX_07)
Simon Lauber
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import physical_constants

from blond import (
    AllowPlotting,
    Beam,
    BiGaussian,
    DriftSimple,
    MagneticCyclePerTurn,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
)
from blond.core.beam.particle_types import ParticleType
from blond.handle_results.observables import BeamStatisticsOncePerTurn

# U-28+ at SIS100 (GSI): fully stripped uranium is U-92+, injection uses U-28+
atomic_mass_unit_eV = physical_constants[
    "atomic mass unit-electron volt relationship"
][0]  # [eV]
u28_plus = ParticleType(mass=238.05078826 * atomic_mass_unit_eV, charge=28)

INJECTION_MOMENTUM = 153.37e9  # injection momentum [eV/c]
EXTRACTION_MOMENTUM = 535.62e9  # extraction momentum [eV/c]
N_TURNS = 200  # abridged ramp (full SIS100 ramp is ~45 500 turns)


def main():
    circumference = 1083.6  # SIS100 circumference [m]

    ring = Ring(circumference=circumference)
    energy_cycle = MagneticCyclePerTurn(
        reference_particle=u28_plus,
        value_init=INJECTION_MOMENTUM,
        values_after_turn=np.linspace(
            INJECTION_MOMENTUM, EXTRACTION_MOMENTUM, N_TURNS
        ),
        in_unit="momentum",
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=10,
        voltage=280e3,
        phi_rf=np.pi,
    )
    drift = DriftSimple(orbit_length=circumference)
    drift.momentum_compaction_factor = momentum_compaction_factor(
        transition_gamma=15.59
    )
    beam = Beam(intensity=5e11, particle_type=u28_plus)

    sim = Simulation.from_locals(locals())
    print(
        f"Ion: U-28+  |  mass = {u28_plus.mass:.4e} eV  |  charge = {u28_plus.charge}"
    )

    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=100e-9 / 4,
            reinsertion=False,
            seed=1,
            n_macroparticles=1001,
        ),
        beam=beam,
    )

    bunch_obs = BeamStatisticsOncePerTurn(each_turn_i=10)
    sim.run_simulation(
        beams=(beam,),
        n_turns=N_TURNS,
        observe=(bunch_obs,),
    )

    energy_spread = bunch_obs.bunch_length
    print(f"Final energy spread: {energy_spread[-1] * 1e-6:.3f} MeV")

    with AllowPlotting():
        plt.figure()
        plt.plot(energy_spread * 1e-6, label="energy spread [MeV]")
        plt.xlabel("Observation index (every 10 turns)")
        plt.ylabel("Energy spread [MeV]")
        plt.title("SIS100 U-28+ ramp: energy spread evolution")
        plt.legend()


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
