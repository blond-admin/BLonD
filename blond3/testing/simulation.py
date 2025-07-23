from matplotlib import pyplot as plt


class ExampleSimulation01:
    def __init__(self):
        import numpy as np

        from blond3._core.backends.backend import backend, Numpy32Bit
        from blond3.cycles.magnetic_cycle import MagneticCyclePerTurn

        backend.change_backend(Numpy32Bit)
        backend.set_specials("numba")

        from blond3 import (
            Beam,
            proton,
            Ring,
            Simulation,
            SingleHarmonicCavity,
            DriftSimple,
            BiGaussian,
            CavityPhaseObservation,
        )

        ring = Ring(circumference=26658.883)

        cavity1 = SingleHarmonicCavity()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        N_TURNS = int(10)
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
            in_unit="momentum",
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.transition_gamma = (55.759505,)

        beam1 = Beam(n_particles=1e9, particle_type=proton)

        simulation = Simulation.from_locals(locals())
        simulation.print_one_turn_execution_order()

        simulation.prepare_beam(
            beam=beam1,
            preparation_routine=BiGaussian(
                sigma_dt=0.4e-9 / 4,
                sigma_dE=1e9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=10,
            ),
            turn_i=10,
        )

        # sim.beams[0].plot_hist2d()
        # plt.show()
        phase_observation = CavityPhaseObservation(each_turn_i=1, cavity=cavity1)

        # bunch_observation = BunchObservation(each_turn_i=10, batch_size=) # todo
        # batches
        def my_callback(simulation: Simulation):
            if simulation.turn_i.value % 10 != 0:
                return

            plt.scatter(
                simulation.beams[0].read_partial_dt(),
                simulation.beams[0].read_partial_dE(),
            )
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        self.simulation = simulation


class SimulationTwoRfStations:
    def __init__(self):
        import numpy as np

        from blond3.cycles.magnetic_cycle import MagneticCyclePerTurn

        from blond3 import (
            Beam,
            proton,
            Ring,
            Simulation,
            SingleHarmonicCavity,
            DriftSimple,
        )

        circumference = 26658.883
        ring = Ring(circumference=circumference)

        cavity1 = SingleHarmonicCavity(
            section_index=0,
        )
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        cavity2 = SingleHarmonicCavity(
            section_index=1,
        )
        cavity2.harmonic = 35640
        cavity2.voltage = 6e6
        cavity2.phi_rf = 0

        N_TURNS = int(10)
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(
                450e9,
                450e9,
                N_TURNS,
            ),
            reference_particle=proton,
        )

        drift1 = DriftSimple(
            orbit_length=0.5 * circumference,
            section_index=0,
        )
        drift1.transition_gamma = 55.759505
        drift2 = DriftSimple(
            orbit_length=0.5 * circumference,
            section_index=1,
        )
        drift2.transition_gamma = 55.759505
        beam1 = Beam(
            n_particles=1e9,
            particle_type=proton,
        )

        simulation = Simulation.from_locals(locals())

        self.simulation = simulation
