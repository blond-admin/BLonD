<div align="center">
<img src="BLonD2_centered.png" alt="drawing" width="300"/>
</div>

[![Pipeline Status](https://gitlab.cern.ch/blond/BLonD/badges/blonder/pipeline.svg)](https://gitlab.cern.ch/blond/BLonD/-/commits/blonder) [![Coverage Report](https://gitlab.cern.ch/blond/BLonD/badges/blonder/coverage.svg)](https://gitlab.cern.ch/blond/BLonD/-/commits/blonder) [![Latest Release](https://gitlab.cern.ch/blond/BLonD/-/badges/release.svg)](https://gitlab.cern.ch/blond/BLonD/-/releases) [![PyPi](https://img.shields.io/pypi/v/blond.svg)](https://pypi.org/project/blond/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org) [![Documentation Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://blond-code.docs.cern.ch/)


# Beam Longitudinal Dynamics Code (BLonD)




> CERN code for the simulation of longitudinal beam dynamics in synchrotrons.


### Dependencies

* [Python 3.10+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)
* Optional
    * C++ on Linux: [GCC (recommended)](https://gcc.gnu.org/install/), `icc` or `clang`
    * C++ on Windows: [mingw-w64](https://winlibs.com/#download-release)
    * CUDA Compiler Driver - [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)


## Installation
> This does not work until BLonD3 is the main version!
>
> Follow the [Developer Guide](CONTRIBUTING.md) to get BLonD3.
```bash
pip install blond
```
or if a GPU is available
```bash
pip install blond[gpu]
```

### Configuration
Optional backends can be compiled after installation using the commands `blond-compile-cpp` `blond-compile-cuda`,
or `blond-compile-fortran` for improved performance. The backend can be selected in Python using ```backend.set_specials(...)```.

## Documentation
See full documentation [here](https://blond-code.docs.cern.ch/).


## Usage

```python
import matplotlib.pyplot as plt

from blond import (
    Ring,
    SingleHarmonicCavity,
    ConstantMagneticCycle,
    proton,
    Simulation,
    DriftSimple,
    Beam,
    BiGaussian,
    backend,
)

backend.set_specials("cpp") # set any backend you want

ring = Ring(26658.883) # general definition of ring
cavity1 = SingleHarmonicCavity(harmonic=35640, voltage=6e6, phi_rf=0)
drift1 = DriftSimple(orbit_length=26658.883, transition_gamma=55.759505)
ring.add_elements([cavity1, drift1]) # add elements that resemble one turn

# Define the ramp
magnetic_cycle = ConstantMagneticCycle(value=450e9, reference_particle=proton)

# Define the general beam properties
beam1 = Beam(intensity=1e9, particle_type=proton)

# Assemble simulation, will trigger late-init processes that link the
# objects together
sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
sim.print_one_turn_execution_order()

# As the physics case is defined in the simulation,
# the beam can be populated with particles according to the separatrix.
sim.prepare_beam(
    beam=beam1,
    preparation_routine=BiGaussian(sigma_dt=0.1e-9, n_macroparticles=1e6),
)


plt.figure(0)
plt.subplot(2, 1, 1)
plt.title("Beam before simulation")
beam1.plot_hist2d()

# Artificially introduce offset to show filamentation
dts = beam1.write_partial_dt()
dts += 0.05e-9

sim.run_simulation(
    beams=(beam1,),
    turn_i_init=0,
    n_turns=1e4,
)
plt.figure(0)
plt.subplot(2, 1, 2)
plt.title("Beam after simulation")
beam1.plot_hist2d()
plt.tight_layout()
plt.show()
```

## Contributing

See the [Developer Guide](CONTRIBUTING.md) if you want to contribute.

## Further Reading

[Repository](https://gitlab.cern.ch/blond/BLonD)

[Documentation](https://blond-code.docs.cern.ch/)

[Project website](http://blond.web.cern.ch)



## Copyright Notice

*Copyright 2019 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.*
