import numpy as np

from blond3 import SingleHarmonicCavity, MagneticCyclePerTurn, electron, \
    DriftSimple, BiGaussian, Simulation, Beam


class SynchrotronRadiationSimulation:
    def __init__(self):

        self.synchrotron_radiation_integrals = np.array([0.646747216157, 0.0005936549319, 5.6814536525e-08, 5.92870407301e-09 , 1.71368060083-11])
        cavity1 = SingleHarmonicCavity()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        n_turns = int(10)
        energy_cycle = MagneticCyclePerTurn(
            value_init=20e9,
            values_after_turn=np.linspace(20e9, 20e9, n_turns),
            reference_particle=electron,
            in_unit="momentum",
        )

        drift1 = DriftSimple(
            orbit_length=90.65874532 * 1e3,
        )
        drift1.transition_gamma = (55.759505,)

        beam = Beam(n_particles=1e9, particle_type=electron)
        dt = beam.write_partial_dt()
        dE = beam.write_partial_dE()

        self.beam = beam

        simulation = Simulation.from_locals(locals())
        simulation.print_one_turn_execution_order()

        simulation.prepare_beam(
              beam=beam,
              preparation_routine=BiGaussian(
                  sigma_dt=0.4e-9 / 4,
                  sigma_dE=1e9 / 4,
                  reinsertion=False,
                  seed=1,
                  n_macroparticles=10,
              ),
              turn_i=1,
          )


        # dt[:] += np.load(
        #     'Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
        #     '/damped_distribution_dt_4mm'
        #     '.npy')
        # dE[:] += np.load(
        #     'Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
        #     '/damped_distribution_dE_4mm'
        #     '.npy')
        # self.simulation = simulation
        # self.simulation.run_simulation()