try:
    from xtrack import Line, Particles, ReferenceEnergyIncrease

    xtrack_available = True
except ModuleNotFoundError:
    xtrack_available = False
