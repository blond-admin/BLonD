# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Minimal executable setup of the muon-collider cavity feedback.

Tracks one turn on a constant-energy ring with two RF stations, each driven
by an ``IQCavityFeedbackTimingClass`` and a PI generator-current controller,
and returns every station's fine-grid antenna-voltage envelope. Nothing is
written to disk. Run from the BLonD root::

    python docs/feedbacks/examples/minimal_feedback.py
"""

from __future__ import annotations

import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    mu_minus,
    mu_plus,
)
from blond.physics.feedbacks.cavity_feedback import IQCavityFeedbackTimingClass
from blond.physics.feedbacks.generator_current_controller import (
    GeneratorCurrentPIController,
)

CIRCUMFERENCE = 5990.0  # [m]
TOTAL_ENERGY = 63e9  # [eV]
MOMENTUM_COMPACTION = 10.395e-4
N_SECTIONS = 2  # even: no station sits at a meeting azimuth of the beams
N_CAVITIES = 2  # cavities represented by one station
VOLTAGE_PER_CAVITY = 0.5e6  # [V]
R_OVER_Q = 518.0  # [Ohm]
Q_L = 1.29e6
N_BINS = 1024
N_MACROPARTICLES = 1000

# Resonant (delta_omega = 0) no-beam equilibrium generator current [A].
GENERATOR_CURRENT = VOLTAGE_PER_CAVITY / (2.0 * R_OVER_Q * Q_L)
# Demonstration gains only -- not an RCS tuning prescription.
GAIN_PROPORTIONAL = 0.1 / (R_OVER_Q * 2.0 * np.pi)  # [A/V]
N_DELAY = 2  # loop delay [coarse samples]


def make_bunch(
    t_rf: float, intensity: float, is_counter_rotating: bool
) -> Beam:
    """
    Create a Gaussian bunch in the middle of the profile window.

    Parameters
    ----------
    t_rf
        RF period [s].
    intensity
        Number of real particles represented by the bunch.
    is_counter_rotating
        If True, a counter-rotating mu- bunch; otherwise a co-rotating mu+.

    Returns
    -------
    Beam
        The bunch; the same seed gives both beams equal coordinates.
    """
    beam = Beam(
        intensity=intensity,
        particle_type=mu_minus if is_counter_rotating else mu_plus,
        is_counter_rotating=is_counter_rotating,
    )
    beam.reference.total_energy = TOTAL_ENERGY
    rng = np.random.default_rng(seed=1)
    beam.setup_beam(
        dt=rng.normal(1.5 * t_rf, 0.06 * t_rf, N_MACROPARTICLES),
        dE=rng.normal(0.0, 10e6, N_MACROPARTICLES),
    )
    return beam


def run_example(
    two_beams: bool = False, intensity: float = 0.0
) -> list[np.ndarray]:
    """
    Track one turn and return each station's fine-grid envelope.

    Parameters
    ----------
    two_beams
        If True, a co-rotating mu+ and a counter-rotating mu- bunch share
        every station's feedback, and the equal bunches share a frozen
        initial histogram. Otherwise a single mu+ bunch with live profiles.
    intensity
        Real particles per bunch. The default 0 gives no beam loading.

    Returns
    -------
    list[numpy.ndarray]
        Per station, the complex station-total antenna-voltage envelope
        ``antenna_voltage_fine_grid`` [V] after its last passage.
    """
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=mu_plus, value=TOTAL_ENERGY, in_unit="total energy"
    )
    t_rev = magnetic_cycle.get_t_rev_init(CIRCUMFERENCE, particle_type=mu_plus)
    # harmonic % (2 * N_SECTIONS) == 0 for the half-drift/station layout.
    harmonic = 25900 - 25900 % (2 * N_SECTIONS)
    t_rf = t_rev / harmonic

    beams = [make_bunch(t_rf, intensity, is_counter_rotating=False)]
    if two_beams:
        beams.append(make_bunch(t_rf, intensity, is_counter_rotating=True))

    ring = Ring(circumference=CIRCUMFERENCE, check_section_indices=False)
    half_drift = CIRCUMFERENCE / N_SECTIONS / 2.0
    elements = []
    feedbacks = []
    for section_index in range(N_SECTIONS):
        profile = StaticProfile.from_rad(
            1.5 * np.pi, 4.5 * np.pi, N_BINS, t_rf, section_index=section_index
        )
        if two_beams:
            profile.track(beams[0])  # histogram the initial bunch once ...
            profile.active = False  # ... and freeze it for both beams
        controller = GeneratorCurrentPIController(
            gain_proportional=GAIN_PROPORTIONAL,
            gain_integral=GAIN_PROPORTIONAL / (30.0 * t_rf),
            generator_current_bias=GENERATOR_CURRENT + 0.0j,
            n_delay=N_DELAY,
        )
        feedback = IQCavityFeedbackTimingClass(
            profile=profile,
            R_over_Q=R_OVER_Q,
            Q_L=Q_L,
            generator_current_bias=GENERATOR_CURRENT + 0.0j,
            n_cavities=N_CAVITIES,
            initial_voltage=VOLTAGE_PER_CAVITY,
            delta_omega=0.0,
            controller=controller,
        )
        station = SingleHarmonicRFStation(
            voltage=N_CAVITIES * VOLTAGE_PER_CAVITY,
            phi_rf=0.0,
            harmonic=harmonic,
            cavity_feedback=feedback,
            profile=profile,
            section_index=section_index,
        )
        drift_kwargs = {
            "orbit_length": half_drift,
            "momentum_compaction_factor": MOMENTUM_COMPACTION,
            "section_index": section_index,
        }
        # A live profile brackets its station, so either beam direction
        # histograms it right before the kick; a frozen one is not re-filled.
        elements += [
            DriftSimple(**drift_kwargs),
            profile,
            station,
            profile,
            DriftSimple(**drift_kwargs),
        ]
        feedbacks.append(feedback)
    # Counter-rotating beams need the layout exactly as written.
    ring.add_elements(elements, reorder=False)

    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    simulation.run_simulation(beams, n_turns=1, show_progressbar=False)
    return [
        feedback.antenna_voltage_fine_grid.copy() for feedback in feedbacks
    ]


if __name__ == "__main__":
    for two_beams in (False, True):
        envelopes = run_example(two_beams=two_beams)
        for station_index, envelope in enumerate(envelopes):
            print(
                f"two_beams={two_beams}, station {station_index}: "
                f"max |V - 1e6| = {np.max(np.abs(envelope - 1e6)):.3e} V, "
                f"max |angle| = {np.max(np.abs(np.angle(envelope))):.3e} rad"
            )
