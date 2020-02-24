SPS Cavity Loop
===============
:Authors: **Helga Timko**


PRINCIPLES: The Cavity Controller
---------------------------------

The cavity controller is a one-turn feedback, measuring in one turn and correcting in the turn after. It works on the
carrier frequency of the present RF frequency :math:`f_{c}=f_{\mathsf{rf}}`. The cavity response is applied at
the central or resonant frequency of the cavity :math:`f_{r}`, requiring and up- and down-modulation before and after.
The controller loop is built of three entities in the code, see Figure.

.. image:: SPS_OTFB.png
    :align: right
    :width: 1226
    :height: 451

Low-level RF
~~~~~~~~~~~~

The low-level RF (LLRF) contains the comparison between the desired set-point voltage :math:`V_{\mathsf{set}}` and the
actual antenna voltage :math:`V_{\mathsf{ant}}` in the travelling wave cavity (TWC), and acts on the difference
:math:`dV_{\mathsf{gen}}=V_{\mathsf{set}}-V_{\mathsf{ant}}` with the gain :math:`G_{\mathsf{llrf}}`. The comb filter
:math:`H_{\mathsf{comb}}` represents the LLRF filter that reduces the beam loading seen by the beam. It acts bunch by
bunch, and with exactly one turn delay,

.. math::

    dV_{\mathsf{gen,out}, k, n} = a_{\mathsf{comb}} \, dV_{\mathsf{gen,out}, k, n-1} + (1 - a) \, dV_{\mathsf{gen,in}, k, n} \, ,

where :math:`V_{\mathsf{gen,in}}` and :math:`V_{\mathsf{gen,out}}` are at the input and output of the comb filter,
respectively, :math:`k` is the index of the bucket along the ring, and :math:`n` is the index of the turn. The comb
filter constant is :math:`a_{\mathsf{comb}}=15/16` operationally. The output of the comb filter is filtered by the
cavity response :math:`H_{\mathsf{cav}}` represented as a moving average at 40~MS/s. The moving average over :math:`K`
points is

.. math::

    dV_{\mathsf{gen,out}, k} = \frac{1}{K} \sum_{i=k-K}^{k} dV_{\mathsf{gen,in}, i} \, .