import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    Ring,
    Simulation,
    UserDefinedElement,
    proton,
)
from blond._core.beam.base import BeamBaseClass


class TimeRandomizer(UserDefinedElement):
    def __init__(self):
        super().__init__()

    def track(self, beam: BeamBaseClass):
        dt = beam.write_partial_dt()
        dt += np.random.rand(len(dt))


def main():
    ring = Ring(circumference=42)
    ring.add_element(TimeRandomizer())
    sim = Simulation(
        ring=ring,
        magnetic_cycle=ConstantMagneticCycle(
            reference_particle=proton,
            value=1e9,
            in_unit="momentum",
        ),
    )
    beam = Beam(
        n_particles=1e9,
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
    main()
