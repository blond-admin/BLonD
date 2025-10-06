import matplotlib.pyplot as plt

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicCavity,
    backend,
    proton,
)

backend.set_specials("cpp")  # set any backend you want

ring = Ring(26658.883)  # general definition of ring
cavity1 = SingleHarmonicCavity(harmonic=35640, voltage=6e6, phi_rf=0)
drift1 = DriftSimple(orbit_length=26658.883, transition_gamma=55.759505)
ring.add_elements([cavity1, drift1])  # add elements that resemble one turn

# Define the ramp
magnetic_cycle = ConstantMagneticCycle(value=450e9, reference_particle=proton)

# Define the general beam properties
beam1 = Beam(intensity=1e9, particle_type=proton)

# Assemble simulation, will trigger late-init processes that link the
# objects together
sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
sim.print_one_turn_execution_order()

# As the physics case is defined in the simulation,
# the beam can be populated with particles according to the separatrix.
sim.prepare_beam(
    beam=beam1,
    preparation_routine=BiGaussian(sigma_dt=0.1e-9, n_macroparticles=1e6),
)


plt.figure(0)
plt.subplot(2, 1, 1)
plt.title("Beam before simulation")
beam1.plot_hist2d()

# Artificially introduce offset to show filamentation
dts = beam1.write_partial_dt()
dts += 0.05e-9

sim.run_simulation(
    beams=(beam1,),
    turn_i_init=0,
    n_turns=1e4,
)
plt.figure(0)
plt.subplot(2, 1, 2)
plt.title("Beam after simulation")
beam1.plot_hist2d()
plt.tight_layout()
plt.show()
