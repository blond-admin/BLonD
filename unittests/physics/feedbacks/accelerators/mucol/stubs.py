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
    """

    def __init__(self, intensity: float):
        self.particle_type = mu_plus
        self.intensity = intensity
        self.is_counter_rotating = False
        self.reference = StubReference()


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
