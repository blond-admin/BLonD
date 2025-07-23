from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .base import MatchingRoutine
from .._core.backends.backend import backend
from .._core.helpers import int_from_float_with_warning
from ..acc_math.analytic.hammilton import is_in_separatrix, calc_phi_s_single_harmonic

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional

    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass


def _get_dE_from_dt_core(
    beta: float,
    dt_amplitude: float,
    energy: float,
    eta0: float,
    harmonic: float,
    omega_rf: float,
    particle_charge: float,
    phi_rf: float,
    phi_s: float,
    voltage: float,
) -> float:
    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi
    # Calculate dE_amplitude from dt_amplitude using single-harmonic Hamiltonian
    voltage = particle_charge * voltage
    phi_b = omega_rf * dt_amplitude + phi_s
    dE_amplitude = np.sqrt(
        voltage
        * energy
        * beta**2
        * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s))
        / (np.pi * harmonic * np.fabs(eta0))
    )
    return dE_amplitude


def _get_dE_from_dt(
    simulation: Simulation,
    beam: BeamBaseClass,
    dt_amplitude: float,
) -> float:
    r"""A routine to evaluate the dE amplitude from dt following a single
    RF Hamiltonian.

    Returns
    -------
    dE_amplitude : float
        Full amplitude of the particle oscillation, in [eV]

    """
    from .. import MultiHarmonicCavity
    from ..physics.cavities import SingleHarmonicCavity
    from ..physics.drifts import DriftSimple

    drift: DriftSimple = simulation.ring.elements.get_element(DriftSimple)
    try:
        rf_station: SingleHarmonicCavity = simulation.ring.elements.get_element(
            SingleHarmonicCavity
        )
        main_harmonic_idx = None
    except AssertionError:
        rf_station: MultiHarmonicCavity = simulation.ring.elements.get_element(
            MultiHarmonicCavity
        )
        main_harmonic_idx = rf_station.main_harmonic_idx

    counter = simulation.turn_i.value  # todo might need to be set
    drift.apply_schedules(turn_i=counter, reference_time=beam.reference_time)
    rf_station.apply_schedules(turn_i=counter, reference_time=beam.reference_time)
    energy = beam.reference_total_energy
    beta = beam.reference_beta
    harmonic = rf_station.harmonic
    omega_rf = rf_station.calc_omega(
        beam_beta=beam.reference_beta,
        ring_circumference=simulation.ring.circumference,
    )
    phi_rf = rf_station.phi_rf
    voltage = rf_station.voltage
    if main_harmonic_idx is not None:
        harmonic = harmonic[main_harmonic_idx]
        omega_rf = omega_rf[main_harmonic_idx]
        phi_rf = phi_rf[main_harmonic_idx]
        voltage = voltage[main_harmonic_idx]
    warnings.warn("assuming wrongly phi_s = phi_rf for development, to be resolved")
    phi_s = calc_phi_s_single_harmonic(
        charge=beam.particle_type.charge,
        voltage=voltage,
        phase=phi_rf,
        energy_gain=simulation.magnetic_cycle.get_target_total_energy(
            1, 0, 0, particle_type=beam.particle_type
        )
        - beam.reference_total_energy,
        above_transition=False,  # FIXME
    )
    eta0 = drift.eta_0(gamma=beam.reference_gamma)
    particle_charge = beam.particle_type.charge

    return _get_dE_from_dt_core(
        beta=float(beta),
        dt_amplitude=dt_amplitude,
        energy=float(energy),
        eta0=eta0,
        harmonic=float(harmonic),
        omega_rf=float(omega_rf),
        particle_charge=particle_charge,
        phi_rf=float(phi_rf),
        phi_s=float(phi_s),
        voltage=float(voltage),
    )


class BiGaussian(MatchingRoutine):
    def __init__(
        self,
        n_macroparticles: int | float,
        sigma_dt: float,
        sigma_dE: Optional[float] = None,
        reinsertion: bool = False,
        seed: int = 0,
    ):
        """Beam matching routine to generate a 2D Gaussian particle distribution

        Parameters
        ----------
        n_macroparticles
            Number of macroparticles to be generated
        sigma_dt
            Normal distribution length, in [s]
        sigma_dE
            Normal distribution height, in [eV]
        reinsertion
            If True, only particles within the separatrix are generated
        seed
            Random seed parameter
        """
        super().__init__()
        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self._sigma_dt = sigma_dt
        self._sigma_dE = sigma_dE
        self._reinsertion = reinsertion
        self._seed = seed

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            Simulation context manager
        """

        from .. import MultiHarmonicCavity
        from ..physics.cavities import SingleHarmonicCavity
        from ..physics.drifts import DriftSimple

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
        try:
            rf_station: SingleHarmonicCavity = simulation.ring.elements.get_element(
                SingleHarmonicCavity
            )
            main_harmonic = None
        except AssertionError:
            rf_station: MultiHarmonicCavity = simulation.ring.elements.get_element(
                MultiHarmonicCavity
            )
            main_harmonic = rf_station.main_harmonic_idx
        drift: DriftSimple = simulation.ring.elements.get_element(DriftSimple)
        rf_station.apply_schedules(
            turn_i=0,
            reference_time=0,
        )
        drift.apply_schedules(
            turn_i=0,
            reference_time=0,
        )

        if self._sigma_dE is None:
            sigma_dE = _get_dE_from_dt(
                beam=beam,
                simulation=simulation,
                dt_amplitude=self._sigma_dt,
            )
            # IMPORT
            assert not np.isnan(sigma_dE), "BUG, fix phi_s"
        else:
            sigma_dE = self._sigma_dE

        omega_rf = rf_station.calc_omega(
            beam_beta=beam.reference_beta,
            ring_circumference=simulation.ring.circumference,
        )
        phi_rf = rf_station.phi_rf
        harmonic = rf_station.harmonic
        voltage = rf_station.voltage

        if main_harmonic is not None:
            omega_rf = omega_rf[main_harmonic]
            phi_rf = phi_rf[main_harmonic]
            harmonic = harmonic[main_harmonic]
            voltage = voltage[main_harmonic]
        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=voltage,
            phase=phi_rf,
            energy_gain=simulation.magnetic_cycle.get_target_total_energy(
                0, 0, 0, particle_type=beam.particle_type
            )
            - beam.reference_total_energy,
            above_transition=False,
        )
        # call to legacy
        eta0 = drift.eta_0(gamma=beam.reference_gamma)

        # RF wave is shifted by Pi below transition
        if eta0 < 0:
            phi_rf -= np.pi

        # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
        rng_dt = np.random.default_rng(self._seed)
        rng_dE = np.random.default_rng(self._seed + 1)

        dt = (
            self._sigma_dt
            * rng_dt.normal(size=self.n_macroparticles).astype(
                dtype=backend.float, order="C", copy=False
            )
            + (phi_s - phi_rf) / omega_rf
        )
        dE = sigma_dE * rng_dE.normal(size=self.n_macroparticles).astype(
            dtype=backend.float, order="C"
        )

        # Re-insert if necessary
        if self._reinsertion:
            while True:
                sel = (
                    is_in_separatrix(
                        charge=beam.particle_type.charge,
                        harmonic=harmonic,
                        voltage=voltage,
                        omega_rf=omega_rf,
                        phi_rf_d=phi_rf,
                        phi_s=phi_s,
                        etas=[eta0],
                        beta=beam.reference_beta,
                        total_energy=beam.reference_total_energy,
                        ring_circumference=simulation.ring.circumference,
                        dt=dt,
                        dE=dE,
                    )
                    == False
                )

                n_new = np.sum(sel)
                if n_new == 0:
                    break
                dt[sel] = (
                    self._sigma_dt
                    * rng_dt.normal(size=n_new).astype(
                        dtype=backend.float, order="C", copy=False
                    )
                    + (phi_s - phi_rf) / omega_rf
                )

                dE[sel] = sigma_dE * rng_dE.normal(size=n_new).astype(
                    dtype=backend.float, order="C"
                )
        beam.setup_beam(dt=dt, dE=dE)
