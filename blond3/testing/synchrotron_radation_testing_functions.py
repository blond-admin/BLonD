import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from blond3 import BiGaussian
from blond3.acc_math.analytic.synchrotron_radiation.synchrotron_radiation_maths import gather_longitudinal_synchrotron_radiation_parameters

from blond3.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond3._core.ring.ring import Ring
from blond3._core.beam.beams import Beam
from blond3._core.beam.particle_types import electron
from blond3._core.simulation.simulation import Simulation
from blond3.physics.cavities import SingleHarmonicCavity
from blond3.physics.drifts import DriftSimple

class SynchrotronRadiationSimulation:
    def __init__(self):

        self.synchrotron_radiation_integrals = np.array([0.646747216157, 0.0005936549319, 5.6814536525e-08, 5.92870407301e-09 , 1.71368060083-11])
        ring = Ring()

        cavity1 = SingleHarmonicCavity()
        cavity1.harmonic = 35640
        cavity1.voltage = 6e6
        cavity1.phi_rf = 0

        N_TURNS = int(10)
        energy_cycle = MagneticCyclePerTurn(
            value_init=182.5e9,
            values_after_turn=np.linspace(182.5e9, 182.5e9, N_TURNS),
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

        simulation.on_prepare_beam(
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
        dt[:] += np.load(
            'Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
            '/damped_distribution_dt_4mm'
            '.npy')
        dE[:] += np.load(
            'Users/lvalle/cernbox/FCC-ee/BLonD_simulations'
            '/damped_distribution_dE_4mm'
            '.npy')
        self.simulation = simulation
        self.simulation.run_simulation()


    def compare_kicks(self):
        particles_energy = self.beam.reference_total_energy + self.beam.read_partial_dE()
        U0, tau_z, sigma0 = gather_longitudinal_synchrotron_radiation_parameters(
             particle_type=self.beam.particle_type,
            energy=particles_energy,
            synchrotron_radiation_integrals = self.synchrotron_radiation_integrals,
         )

        U0_mono, tau_z_mono, sigma0_mono = gather_longitudinal_synchrotron_radiation_parameters(
             particle_type=self.beam.particle_type,
            energy=self.beam.reference_total_energy,
            synchrotron_radiation_integrals = self.synchrotron_radiation_integrals,
         )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        sb.distplot(U0)
        plt.title('Energy lost per particle')

        plt.show()

        # return (
        #  -
        #  - 2.0 / tau_z * self.beam.read_partial_dE()
        #  - 2.0
        #  * sigma0
        #  / np.sqrt(tau_z)
        #  * self.beam.reference_total_energy
        #  * np.random.normal(size=len(self.beam.n_macroparticles_partial()))
        # )


SRS = SynchrotronRadiationSimulation()
SRS.compare_kicks()


