# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""The vector-fitted impedance model a pole-residue solver consumes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

from blond.generals.cupy_.no_cupy_import import is_cupy_array

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


class _VectorFitFields(NamedTuple):
    """The fields of `VectorFit`, see there."""

    # Documented here rather than in an "Attributes" section of `VectorFit`:
    # autodoc already lists the fields as members, so the section would
    # describe each one twice and fail the doc build.
    #: Complex poles :math:`p_k`, in [rad/s].
    poles: NumpyArray
    #: Complex residues :math:`\rho_k`, in [Ohm/s].
    residues: NumpyArray
    #: Signs of the poles to deal with higher order oscillators in
    #: counterrotation, ``1`` for none.
    counterrotation_signs: NumpyArray
    #: Constant term :math:`d`, in [Ohm].
    direct_term: float = 0.0
    #: Inductive term :math:`e`, i.e. the inductance :math:`L` in
    #: :math:`Z = 2\pi i f L`, in [Ohm s].
    inductive_term: float = 0.0


class VectorFit(_VectorFitFields):
    r"""
    A vector-fitted impedance model.

    The impedance is :math:`Z(s) = \sum_k \rho_k / (s - p_k) + d + e\,s`
    with :math:`s = 2\pi i f`, where each complex pole stands in for its
    conjugate partner. The constant term :math:`d` and the inductive term
    :math:`e` have no pole-residue representation; their wakes are
    :math:`d\,\delta(t)` and :math:`e\,\delta'(t)`.

    The arrays are host (NumPy) arrays on every backend: the fit describes
    the model, which a solver builds once and then moves to the device
    whatever it tracks with. Device (CuPy) arrays are rejected.
    """

    __slots__ = ()

    def __new__(
        cls,
        poles: NumpyArray,
        residues: NumpyArray,
        counterrotation_signs: NumpyArray,
        direct_term: float = 0.0,
        inductive_term: float = 0.0,
    ) -> VectorFit:
        """
        Build the fit, rejecting device arrays.

        Parameters
        ----------
        poles
            Complex poles, in [rad/s].
        residues
            Complex residues, in [Ohm/s].
        counterrotation_signs
            Signs of the poles, ``1`` for none.
        direct_term
            Constant term, in [Ohm].
        inductive_term
            Inductive term, in [Ohm s].

        Returns
        -------
        vector_fit
            The vector fit.
        """
        for name, array in (
            ("poles", poles),
            ("residues", residues),
            ("counterrotation_signs", counterrotation_signs),
        ):
            assert not is_cupy_array(array), (
                f"`VectorFit.{name}` must be a host (NumPy) array, but got "
                "a CuPy array; pass it through `copy_to_cpu` first."
            )
        return super().__new__(
            cls,
            poles,
            residues,
            counterrotation_signs,
            direct_term,
            inductive_term,
        )

    @classmethod
    def _make(cls, iterable: Iterable) -> VectorFit:
        # `_replace` builds through `_make`, which would bypass `__new__`.
        return cls(*iterable)
