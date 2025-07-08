from matplotlib import pyplot as plt


class ExampleSimulation01:
    def __init__(self):
        import numpy as np

        from blond3._core.backends.backend import backend, Numpy32Bit
        from blond3.cycles.energy_cycle import EnergyCyclePerTurn

        backend.change_backend(Numpy32Bit)
        # backend.set_specials("numba")

        from blond3 import (
            Beam,
            proton,
            Ring,
            Simulation,
            SingleHarmonicCavity,
            DriftSimple,
            BiGaussian,
            RfStationParams,
            CavityPhaseObservation,
        )

        ring = Ring(
            circumference=26658.883,
        )

        cavity1 = SingleHarmonicCavity( )
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0


        N_TURNS = int(10)
        energy_cycle = EnergyCyclePerTurn(
            value_init=450e9, values_after_turn=np.linspace(450e9, 450e9, N_TURNS)
        )

        drift1 = DriftSimple(
            share_of_circumference=1.0,
        )
        drift1.transition_gamma = 55.759505,

        beam1 = Beam(n_particles=1e9, particle_type=proton)

        simulation = Simulation.from_locals(locals())
        simulation.print_one_turn_execution_order()

        simulation.on_prepare_beam(
            preparation_routine=BiGaussian(
                sigma_dt=0.4e-9 / 4,
                sigma_dE=1e9 / 4,
                reinsertion=False,
                seed=1,
                n_macroparticles=10,
            )
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

        from blond3.cycles.energy_cycle import EnergyCyclePerTurn

        from blond3 import (
            Beam,
            proton,
            Ring,
            Simulation,
            SingleHarmonicCavity,
            DriftSimple,
            RfStationParams,
        )

        ring = Ring(circumference=26658.883)

        cavity1 = SingleHarmonicCavity(
            section_index=0,
        )
        cavity1.harmonic=35640
        cavity1.voltage=6e6
        cavity1.phi_rf=0

        cavity2 = SingleHarmonicCavity(
            section_index=1,
        )
        cavity2.harmonic=35640
        cavity2.voltage=6e6
        cavity2.phi_rf=0


        N_TURNS = int(10)
        energy_cycle = EnergyCyclePerTurn(
            value_init=450e9, values_after_turn=np.linspace(450e9, 450e9, N_TURNS)
        )

        drift1 = DriftSimple(share_of_circumference=0.5, section_index=0)
        drift1.transition_gamma=55.759505
        drift2 = DriftSimple(share_of_circumference=0.5, section_index=1)
        drift2.transition_gamma=55.759505
        beam1 = Beam(n_particles=1e9, particle_type=proton)

        simulation = Simulation.from_locals(locals())

        self.simulation = simulation
