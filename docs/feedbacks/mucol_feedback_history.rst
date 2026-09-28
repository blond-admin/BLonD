.. _mucol_feedback_development_history:

Muon-collider feedback development history
==========================================

These accounts preserve the reasoning behind earlier fixes. Numerical
values and test counts are historical snapshots, not current validation
results or supported-configuration guarantees. See
:ref:`mucol_cavity_feedback_overview` for the current contracts and
:ref:`mucol_cavity_feedback_tests` for active checks.

Reference-choice debugging
--------------------------

*Why the wrong reference survived so long.* The increment used to be
referenced to the carrier of the passage that ENDS the interval, with the
opposite sign: ``sum_k (omega_k - omega_0) T_seg,k``, with ``omega_0``
the forward frequency of the passage doing the correcting. The two forms
differ by

   ``sum_k (omega_prev - 2 omega_k + omega_0) T_seg,k``

which is a *second* difference of the design-frequency programme, and it
vanishes identically when ``omega`` varies linearly in time. Every
first-order check therefore agreed: on a linear ramp the wrong expression
is numerically the right one, and only the curvature of the programme
survived. What that left was a small residual scaling as the *square* of
the registration phase, which read exactly like an accepted second-order
discretisation artefact rather than a sign-and-reference mistake, and it
was carried as a known limitation for that reason. The lesson generalises:
a bookkeeping term validated only against a linear programme has not been
validated against its own reference choice at all -- pick a test case with
curvature, or compare the two candidate forms directly, as
``test_registration_phase_uses_previous_passage_carrier`` now does.

Separating generator and beam frames
------------------------------------

The source-split coarse state (see *Signal path of one turn*) is what
lets ``Psi`` reach exactly the signal that needs it: the beam-sourced
component's demodulation/readout closure carries ``Psi``, while the
design-anchored generator component sees no registration phase at
readout at all, and the PI regulates the kick-frame sum. This closed
the former driven multi-section readout-phase offset -- one shared
readout phase used to hand ``Psi`` to the generator-driven field too,
walking the RF bucket off the design synchronous phase with no beam at
all -- and it is why the zero-intensity phase neutrality of the readout
is exact rather than approximate (the amplitude-drift half of the same
history, percent-level ``|V_ant|`` growth per turn from rotating the
state, had already been fixed by carrying ``Psi`` on the carrier; both
are pinned by ``TestDrivenSteadyStateFastRamp``,
``TestDrivenFeedbackIsPhaseNeutralWithoutBeam`` and
``TestPIFullTrackingMultiSectionFastRamp``).

Earlier long-horizon measurements
---------------------------------

Referencing the increment to the previous passage's carrier is also what
removed the former secular drift of the undriven multi-section fast-ramp
carried wake against the convolution. Over 20 turns the per-turn error
slope went from ``+0.03184`` to ``-0.00255`` pp/turn at two sections
(turn-19 error ``0.66788 %`` -> ``0.02149 %``) and from ``+0.04275`` to
``-0.00219`` pp/turn at four. What is left is negative at every section
count and bounded: at two sections the endpoint residual sits *below*
the single-section control (``0.02618 %``, slope ``-0.00225``) that
bounds the irreducible multi-turn discretisation residual from below, so
there is no registration artefact left to attribute it to.
``test_multiturn_secular_drift_long_horizon`` records the two- and
single-section post-fix numbers and gates them at slope ``< 0.005``
pp/turn and endpoint ``< 0.05 %``. Single-section rings and
unaccelerated multi-section rings are bit-identical across the fix,
because ``Psi`` is exactly ``0.0`` on both paths.
