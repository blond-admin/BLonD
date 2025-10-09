from matplotlib import pyplot as plt

from blond import MultiHarmonicCavity
from blond._core.backends.backend import backend


class ExampleSimulation01:
    """Simulation with only one drift, one RF."""

    def __init__(self):
        import numpy as np

        from blond import (
            Beam,
            BiGaussian,
            CavityPhaseObservation,
            DriftSimple,
            Ring,
            Simulation,
            SingleHarmonicCavity,
            proton,
        )
        from blond.cycles.magnetic_cycle import MagneticCyclePerTurn

        ring = Ring(circumference=26658.883)

        cavity1 = SingleHarmonicCavity()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        N_TURNS = 10
        energy_cycle = MagneticCyclePerTurn(
            value_init=450e9,
            values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
            reference_particle=proton,
            in_unit="momentum",
        )

        drift1 = DriftSimple(
            orbit_length=26658.883,
        )
        drift1.transition_gamma = 55.759505

        beam1 = Beam(intensity=1e9, particle_type=proton)
        self.beam1 = beam1

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

        phase_observation = CavityPhaseObservation(
            each_turn_i=1, cavity=cavity1
        )

        # bunch_observation = BunchObservation(each_turn_i=10, batch_size=)
        # batches
        def my_callback(simulation: Simulation, beam: Beam):
            if simulation.turn_i.value % 10 != 0:
                return

            plt.scatter(
                beam.read_partial_dt(),
                beam.read_partial_dE(),
            )
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        self.simulation = simulation


class SimulationTwoRfStations:
    """A simulation with two RF stations and according drifts."""

    def __init__(self):
        import numpy as np

        from blond import (
            Beam,
            DriftSimple,
            Ring,
            Simulation,
            SingleHarmonicCavity,
            proton,
        )
        from blond.cycles.magnetic_cycle import MagneticCyclePerTurn

        circumference = 26658.883
        ring = Ring(circumference=circumference)

        cavity1 = MultiHarmonicCavity(
            section_index=0, n_harmonics=1, main_harmonic_idx=0
        )
        cavity1.harmonic = np.array(
            [
                35640.0,
            ],
            dtype=backend.float,
        )
        cavity1.voltage = np.array(
            [
                6e6,
            ],
            dtype=backend.float,
        )
        cavity1.phi_rf = np.array(
            [
                0.0,
            ],
            dtype=backend.float,
        )

        cavity2 = SingleHarmonicCavity(
            section_index=1,
        )
        cavity2.harmonic = backend.float(35640)
        cavity2.voltage = backend.float(6e6)
        cavity2.phi_rf = backend.float(0)

        N_TURNS = 10
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
            intensity=1e9,
            particle_type=proton,
        )

        simulation = Simulation.from_locals(locals())

        self.simulation = simulation
