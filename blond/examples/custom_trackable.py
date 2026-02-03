# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    UserDefinedElement,
    backend,
    proton,
)
from blond.core.beam.base import BeamBaseClass


class TimeRandomizer(UserDefinedElement):
    def __init__(self):
        super().__init__()

    def track(self, beam: BeamBaseClass):
        dt = beam.write_partial_dt()
        dt += backend.random.rand(len(dt))


def main():
    ring = Ring(circumference=42)
    ring.add_element(TimeRandomizer())
    ring.add_element(
        DriftSimple(
            orbit_length=ring.circumference,
            momentum_compaction_factor=1 / (12**2),
        )
    )
    sim = Simulation(
        ring=ring,
        magnetic_cycle=ConstantMagneticCycle(
            reference_particle=proton,
            value=1e9,
            in_unit="momentum",
        ),
    )
    beam = Beam(
        intensity=1e9,
        particle_type=proton,
    )
    beam.setup_beam(
        dt=np.linspace(0, 100e-9),
        dE=np.linspace(0, 100e9),
    )
    sim.run_simulation(
        beams=(beam,),
        n_turns=10,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
