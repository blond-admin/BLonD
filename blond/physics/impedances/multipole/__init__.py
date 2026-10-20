# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Pole-residue wakefield solver for sparse, multi-turn profiles.

`MultiPoleSparseSolve` evaluates the bin-averaged wake of a vector-fitted
pole-residue model as a far-field recursion plus three near-field taps; it
is re-exported by :mod:`blond.physics.impedances.solvers`.

* :mod:`~blond.physics.impedances.multipole.solver` -- the solver.
* :mod:`~blond.physics.impedances.multipole.model` -- the pole-residue
  model on the bin grid and the poles the far-field recursion keeps.
* :mod:`~blond.physics.impedances.multipole.vector_fit` -- `VectorFit`,
  the vector-fitted impedance model the solver consumes.
* :mod:`~blond.physics.impedances.multipole.checks` -- validation of the
  profile and the poles.

Notes
-----
Authors:
Simon Lauber
"""
