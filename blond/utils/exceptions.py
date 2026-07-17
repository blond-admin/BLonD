
# ===============
# Beam Exceptions
# ===============

class MassError(Exception):
    pass


class AllParticlesLost(Exception):
    pass


class ParticleAdditionError(Exception):
    pass


# ==================================
# Distribution Generation Exceptions
# ==================================

class DistributionError(Exception):
    pass


class GenerationError(Exception):
    pass


# ==================
# Profile Exceptions
# ==================

class CutError(Exception):
    pass


class ProfileDerivativeError(Exception):
    pass


# ====================
# Impedance Exceptions
# ====================

class WakeLengthError(Exception):
    pass


class FrequencyResolutionError(Exception):
    pass


class ResonatorError(Exception):
    pass


class WrongCalcError(Exception):
    pass


class MissingParameterError(Exception):
    pass


# ===========================
# Input Parameters Exceptions
# ===========================

class MomentumError(Exception):
    pass


# ===============
# LLRF Exceptions
# ===============

class PhaseLoopError(Exception):
    pass


class PhaseNoiseError(Exception):
    pass


class FeedbackError(Exception):
    pass


class ImpulseError(Exception):
    pass


# ==================
# Toolbox Exceptions
# ==================

class PhaseSpaceError(Exception):
    pass


class NoiseDiffusionError(Exception):
    pass


# ==================
# Tracker Exceptions
# ==================

class PotentialWellError(Exception):
    pass


class SolverError(Exception):
    pass


class PeriodicityError(Exception):
    pass


class ProfileError(Exception):
    pass


class SynchrotronMotionError(Exception):
    pass


# ===============
# Util Exceptions
# ===============

class ConvolutionError(Exception):
    pass


class IntegrationError(Exception):
    pass


class SortError(Exception):
    pass


# =================
# Global Exceptions
# =================

class InterpolationError(Exception):
    pass


class InputDataError(Exception):
    pass
