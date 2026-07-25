"""
Lightweight mock objects shared by the muon-collider feedback tests.

These let the cavity-response solvers and the multi-turn resonator solver be
driven directly on a static profile, without a full ``Beam`` or ``Simulation``.
They expose only the few attributes/methods those solvers read.
"""

from blond import mu_plus


class StubReference:
    """
    Minimal beam reference frame (deepcopy-able).

    Parameters
    ----------
    time
        Reference time of the frame.
    beta
        Relativistic beta of the reference particle.
    """

    def __init__(self, time: float = 0.0, beta: float = 1.0):
        self.time = time
        self.beta = beta


class StubBeam:
    """
    Minimal beam exposing only what the solvers read.

    Parameters
    ----------
    intensity
        Beam intensity (number of particles).
    particle_type
        Particle type of the beam; default ``mu_plus``.
    is_counter_rotating
        Whether the beam circulates against the ring direction.
    """

    def __init__(
        self,
        intensity: float,
        particle_type=mu_plus,
        is_counter_rotating: bool = False,
    ):
        self.particle_type = particle_type
        self.intensity = intensity
        self.is_counter_rotating = is_counter_rotating
        self.reference = StubReference()

    def signed_charge_with_direction(self):
        """
        Return the charge corrected with the direction of the beam.

        Mirrors :meth:`blond.core.beam.base.BeamBaseClass.signed_charge_with_direction`.

        Returns
        -------
        float
            ``particle_type.charge``, negated for a counter-rotating beam.
        """
        return (
            self.particle_type.charge * -1
            if self.is_counter_rotating
            else self.particle_type.charge
        )


class StubRFStation:
    """
    Minimal RF station for the solver's reference/frequency bookkeeping.

    Parameters
    ----------
    omega_rf
        RF angular frequency returned by ``calc_omega_rf_design``.
    """

    def __init__(self, omega_rf: float):
        self._omega_rf = omega_rf

    def track_reference(self, reference, is_counter_rotating):
        """
        Advance the reference (no-op for a static single-pass profile).

        Parameters
        ----------
        reference
            Beam reference frame to advance.
        is_counter_rotating
            Whether the beam is counter-rotating.
        """
        # Static profile, single pass: the reference does not advance.
        pass

    def calc_omega_rf_design(self, beam_beta, ring_circumference):
        """
        Return the fixed design RF angular frequency.

        Parameters
        ----------
        beam_beta
            Relativistic beta of the beam.
        ring_circumference
            Circumference of the ring.

        Returns
        -------
        float
            The fixed design RF angular frequency.
        """
        return self._omega_rf
