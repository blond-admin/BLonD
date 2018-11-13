
#===============
#Beam Exceptions
#===============

class MassError(Exception):
    pass

class AllParticlesLost(Exception):
    pass



#==================================
#Distribution Generation Exceptions
#==================================

class DistributionError(Exception):
    pass

class GenerationError(Exception):
    pass


#==================
#Profile Exceptions
#==================

class CutError(Exception):
    pass

class ProfileDerivativeError(Exception):
    pass


#====================
#Impedance Exceptions
#====================

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


#===========================
#Input Parameters Exceptions
#===========================

class InterpolationError(Exception):
    pass

class InputDataError(Exception):
    pass

class MomentumError(Exception):
    pass


#===============
#LLRF Exceptions
#===============

class PhaseLoopError(Exception):
    pass

class PhaseNoiseError(Exception):
    pass

class FeedbackError(Exception):
    pass















