# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Various cavity loops for the CERN machines**

:Authors: **Helga Timko**
'''

from __future__ import division
import logging
import time
import numpy as np
import scipy
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.constants import e
import sys

from ..llrf.signal_processing import comb_filter, remove_noise_cartesian, \
    cartesian_to_polar, polar_to_cartesian, modulator, moving_average, \
    rf_beam_current, cartesian_rotation
from ..llrf.impulse_response import SPS3Section200MHzTWC, \
    SPS4Section200MHzTWC, SPS5Section200MHzTWC
from ..llrf.signal_processing import feedforward_filter_TWC3, \
    feedforward_filter_TWC4, feedforward_filter_TWC5
from ..utils import bmath as bm

from blond.toolbox.input_output_tools import print_object_attributes

######

def eq_line(x, m, x0, y0):
    return m*(x - x0) + y0

def eq_parabola(x, p, x0, y0):
    return p*(x - x0)**2 + y0

def get_power_gen_0(Vind_gen_per_cav, Z_0, R_gen):
    ''' RF generator power for f_r = f_rf = 0 (and thus tau = 0) '''
    return 0.5 * Z_0/R_gen**2 * np.abs(Vind_gen_per_cav)**2

def get_power_gen_VI(Vind_tot_per_cav, I_gen_per_cav):
    ''' RF generator power from the product of total voltage and generator current -- TO BENCHMARK '''
    return 0.5 * np.real( Vind_tot_per_cav * np.conjugate(I_gen_per_cav) ) # The imaginary is zero

def get_power_gen_I2(I_gen_per_cav, Z_0):
    ''' RF generator power from generator current (physical, in [A]), for any f_r (and thus any tau) '''
    return 0.5 * Z_0 * np.abs(I_gen_per_cav)**2

def get_power_gen_V2(Vind_gen_per_cav, Z_0, R_gen, tau, d_omega):
    ''' RF generator power from generator voltage, for any f_r (and thus any tau) '''
    return get_power_gen_0(Vind_gen_per_cav, Z_0, R_gen) * ( 1./np.sinc(0.5*tau*d_omega/np.pi) )**2 # the 1. is exp(-1j*phi)

######

class CavityFeedbackCommissioning(object):

    def __init__(self, debug=False, open_loop=False, open_FB=False, open_drive=False, open_FF=False,
                 Vset_imag=True,
                 redef_Zb=False, redef_Ib=False, redef_Vb=True, redef_Vbff=False,
                 add_ff_to_gen=False):
        """Class containing commissioning settings for the cavity feedback

        Parameters
        ----------
        debug : bool
            Debugging output active (True/False); default is False
        open_loop : int(bool)
            Open (True) or closed (False) cavity loop; default is False
        open_FB : int(bool)
            Open (True) or closed (False) feedback; default is False
        open_drive : int(bool)
            Open (True) or closed (False) drive; default is False
        open_FF : int(bool)
            Open (True) or closed (False) feed-forward; default is False
        """

        self.debug = bool(debug)
        # Multiply with zeros if open == True
        self.open_loop = int(np.invert(bool(open_loop)))
        self.open_FB = int(np.invert(bool(open_FB)))
        self.open_drive = int(np.invert(bool(open_drive)))
        self.open_FF = int(np.invert(bool(open_FF)))

        self.use_pretrack_ramp = True #True
        self.use_gen_fine = False #True

        self.Vset_imag = Vset_imag  # Vset at +Q if True, Vset at +I if False

        self.redef_Zb = redef_Zb
        self.redef_Ib = redef_Ib
        self.redef_Vb = redef_Vb
        self.redef_Vbff = redef_Vbff

        self.add_ff_to_gen = add_ff_to_gen # formerly reverse_bg


class SPSCavityFeedback(object):
    """Class determining the turn-by-turn total RF voltage and phase correction
    originating from the individual cavity feedbacks. Assumes two 4-section and
    two 5-section travelling wave cavities in the pre-LS2 scenario and four
    3-section and two 4-section cavities in the post-LS2 scenario. The voltage
    partitioning is proportional to the number of sections.

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        A Profile type class
    G_llrf : float or list
        LLRF Gain [1]; if passed as a float, both 3- and 4-section (4- and
        5-section) cavities have the same G_llrf in the post- (pre-)LS2
        scenario. If passed as a list, the first and second elements correspond
        to the G_llrf of the 3- and 4-section (4- and 5-section) cavity
        feedback in the post- (pre-)LS2 scenario; default is 10
    G_tx : float or list
        Transmitter gain [1] of the cavity feedback; convention same as G_llrf;
        default is 0.5
    a_comb : float
        Comb filter ratio [1]; default is 15/16
    turns :  int
        Number of turns to pre-track without beam
    post_LS2 : bool
        Activates pre-LS2 scenario (False) or post-LS2 scenario (True); default
        is True
    V_part : float
        Voltage partitioning of the shorter cavities; has to be in the range
        (0,1). Default is None and will result in 6/10 for the 3-section
        cavities in the post-LS2 scenario and 4/9 for the 4-section cavities in
        the pre-LS2 scenario
    deltaf0 : float
        Central (resonance) frequency offset :math:`\Delta f_0` in
        :math:`2 \Pi (\f + \Delta f_0)` [Hz]; default is 0

    Attributes
    ----------
    OTFB_1 : class
        An SPSOneTurnFeedback type class; 3/4-section cavity for post/pre-LS2
    OTFB_2 : class
        An SPSOneTurnFeedback type class; 4/5-section cavity for post/pre-LS2
    V_sum : complex array
        Vector sum of RF voltage from all the cavities
    V_corr : float array
        RF voltage correction array to be applied in the tracker
    phi_corr : float array
        RF phase correction array to be applied in the tracker
    logger : logger
        Logger of the present class

    """

    def __init__(self, RFStation, Beam, Profile, G_ff=1, G_llrf=10, G_tx=0.5,
                 a_comb=15/16, turns=1000, post_LS2=True, V_part=None,
                 Commissioning=CavityFeedbackCommissioning(), deltaf0=0,
                 fillpattern=None, power_clamp=False, nollrf=False, outdir='.'):

        # for varnamei in [None]: #['V_ind_gen']: #['Q_gen']: #'V_ind_gen']: #'Q_gen', 'V_ind_gen']: #['V_set', 'dV_err', 'dV_comb', 'dV_del', 'dV_mod', 'dV_Hcav', 'dV_gen', 'V_gen', 'Q_gen', 'V_ind_gen']:
            # Options for commissioning the feedback
            self.Commissioning = Commissioning
            self.outdir = outdir
            self.Commissioning.outdir = outdir

            self.nollrf = nollrf

            self.rf = RFStation
            self.fillpattern = fillpattern

            # Parse input for gains
            if type(G_ff) is list:
                G_ff_1 = G_ff[0]
                G_ff_2 = G_ff[1]
            else:
                G_ff_1 = G_ff
                G_ff_2 = G_ff

            if type(G_llrf) is list:
                G_llrf_1 = G_llrf[0]
                G_llrf_2 = G_llrf[1]
            else:
                G_llrf_1 = G_llrf
                G_llrf_2 = G_llrf

            #if type(G_tx) is list:
            #    G_tx_1 = G_tx[0]
            #    G_tx_2 = G_tx[1]
            #else:
            #    G_tx_1 = G_tx
            #    G_tx_2 = G_tx



            # Voltage partitioning has to be a fraction
            if V_part and V_part*(1 - V_part) < 0:
                raise RuntimeError("SPS cavity feedback: voltage partitioning has to be in the range (0,1)!")

            # Voltage partition proportional to the number of sections # EDIT: 2020.10.01

            if post_LS2:
                ncav1 = 4; ncell1 = 3 # Note that by default each of the 4 3-section cavities contributes with 3/20, tot = 12/20 = 6/10 = 3/5 = 0.60
                ncav2 = 2; ncell2 = 4 # Note that by default each of the 2 4-section cavities contributes with 4/20, tot =  8/20 = 4/10 = 2/5 = 0.40
                # Gtx1 = 0.99468245
                # Gtx2 = 1.002453405
                # Gtx1 = 1.072 # Default: 0.99468245,  1.072 to get the same power than Philippe w/o beam, but the no-beam segment is messed up, Gllrf = 1 or 10?
                # Gtx2 = 1.110 # Default: 1.002453405, 1.110 to get the same power than Philippe w/o beam, but the no-beam segment is messed up, Gllrf = 1 or 10?
                # Gtx1 = 1.0273803844518141  # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) for no-beam, optimized with Gllrf = 1
                # Gtx2 = 1.0601760715276916  # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) for no-beam, optimized with Gllrf = 1
                # Gtx1 = 1.0277831436616858  # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) for no-beam, optimized with Gllrf = 10
                # Gtx2 = 1.0596313424424000  # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) for no-beam, optimized with Gllrf = 10
                # Gtx1 = 1.0273815248448939  # To get exactly Vset in imag(Vant = Vtot = Vind_gen_coarse) for no-beam, at the expense of slightly larger real, optimized with Gllrf = 1
                # Gtx2 = 1.0601770998991529  # To get exactly Vset in imag(Vant = Vtot = Vind_gen_coarse) for no-beam, at the expense of slightly larger real, optimized with Gllrf = 1
                #Gtx1 = 1.0277833755489165  # To get exactly Vset in imag(Vant = Vtot = Vind_gen_coarse) for no-beam, at the expense of slightly larger real, optimized with Gllrf = 10
                #Gtx2 = 1.0596316401311040  # To get exactly Vset in imag(Vant = Vtot = Vind_gen_coarse) for no-beam, at the expense of slightly larger real, optimized with Gllrf = 10
                Gtx1 = 0.9953399221282384  # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) = im() for no-beam, optimized with Gllrf = 10 and phi_TWC = phi_c = phi_rf
                Gtx2 = 1.0031159935705447  # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) = im() for no-beam, optimized with Gllrf = 10 and phi_TWC = phi_c = phi_rf

                # Power clamps per cavity
                if power_clamp:
                    power_clamp_1 = 1.0e6  # [W]
                    power_clamp_2 = 1.6e6  # [W]
                else:
                    power_clamp_1 = False
                    power_clamp_2 = False


            else:
                ncav1 = 2; ncell1 = 4 # Note that by default each of the 2 4-section cavities contributes with  8/18 = 4/9
                ncav2 = 2; ncell2 = 5 # Note that by default each of the 2 5-section cavities contributes with 10/18 = 5/9
                # Gtx1 = 1.002453405
                # Gtx2 = 1.00066015
                Gtx1 = 1.0031159998252943 # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) = im() for no-beam, optimized with Gllrf = 10 and phi_TWC = phi_c = phi_rf
                Gtx2 = 1.0013215173032060 # To get exactly Vset in abs(Vant = Vtot = Vind_gen_coarse) = im() for no-beam, optimized with Gllrf = 10 and phi_TWC = phi_c = phi_rf

                power_clamp_1 = False
                power_clamp_2 = False

            if not V_part:
                # For post_LS2: V_part = 4*3/(4*3 + 2*4) = 12/20 = 6/10 (original) = 0.600
                # For present:  V_part = 2*4/(2*4 + 2*5) =  8/18 = 4/9  (original) = 0.444
                V_part = (ncav1*ncell1)/(ncav1*ncell1 + ncav2*ncell2)

            self.OTFB_1 = SPSOneTurnFeedback(RFStation, Beam, Profile, ncell1,
                                             n_cavities=ncav1, V_part=V_part,
                                             G_ff=float(G_ff_1),
                                             G_llrf=float(G_llrf_1),
                                             G_tx=Gtx1,
                                             a_comb=float(a_comb),
                                             Commissioning=self.Commissioning,
                                             deltaf0=deltaf0,
                                             fillpattern=self.fillpattern,
                                             power_clamp=power_clamp_1,
                                             nollrf=self.nollrf)
            self.OTFB_2 = SPSOneTurnFeedback(RFStation, Beam, Profile, ncell2,
                                             n_cavities=ncav2, V_part=1-V_part,
                                             G_ff=float(G_ff_2),
                                             G_llrf=float(G_llrf_2),
                                             G_tx=Gtx2,
                                             a_comb=float(a_comb),
                                             Commissioning=self.Commissioning,
                                             deltaf0=deltaf0,
                                             fillpattern=self.fillpattern,
                                             power_clamp=power_clamp_2,
                                             nollrf=self.nollrf)

            # Set up logging
            self.logger = logging.getLogger(__class__.__name__)
            self.logger.info("Class initialized")

            # Initialise OTFB without beam
            self.turns = int(turns)
            if turns < 1:
                #FeedbackError
                raise RuntimeError("ERROR in SPSCavityFeedback: 'turns' has to" +
                                   " bfe a positive integer!")

            self.track_init(debug=Commissioning.debug) # varname=varnamei,

    def track(self):

        if self.nollrf:
            self.OTFB_1.track_nollrf()
            self.OTFB_2.track_nollrf()
        else:
            self.OTFB_1.track()
            self.OTFB_2.track()

        self.V_sum = np.copy(self.OTFB_1.V_tot_fine) + np.copy(self.OTFB_2.V_tot_fine)

        self.V_corr, alpha_sum = cartesian_to_polar(self.V_sum)

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        self.phi_corr = 0.5*np.pi*self.Commissioning.Vset_imag - alpha_sum \
            - self.rf.phi_rf[0, self.rf.counter[0]]

    @staticmethod
    def create_ramp(xf=1., yf=1., n=1000, **kwargs):
        '''
        Creates a symmetric PLP ramp  over n points, with the parabolic
        segments extending from (x, y) = (0, 0) to (a*xf, b*yf) and from
        ((1-a)*xf, (1-b)*yf) to  (xf, yf), connected by a linear segment.
        Only a OR b is need to be given (a has preference)

        Parameters
        ----------
        xf : float
            Value on the x-axis at the end of the ramp

        yf : float
            Value on the y-axis at the end of the ramp

        n  : integer
            Number of points in the ramp

        a OR b as kwarg: float in [0, 0.5]
            a is the extent in the x-axis spent in
            the first parabolic segment (the second segment is symmetric);
            b is the corresponding segment in the y-axis spent in the first
            parabolic segment

        Return
        ------
        x, y : Numpy arrays

        [(0,0), (a*xf, b*yf), ((1-a)*xf, (1-b)*yf), (xf,yf)]: List of tuples
        '''


        if 'a' in kwargs.keys():
            # If a is given, ignore b if also given
            a = kwargs['a']
            b = a/(2*(1.-a))
        elif 'b' in kwargs.keys():
            # To use b, only input b (do not input a)
            b = kwargs['b']
            a = 2*b / (1.+2*b)
        else:
            sys.exit('a OR b must be given as input!')

        if a < 0:
            sys.exit('a and b must be in [0, 0.5]!')

        x0 = 0;    y0 = 0
        x1 = a*xf; x2 = (1.-a)*xf
        y1 = b*yf; y2 = (1.-b)*yf

        l1_xrange = (x0, x1)
        l2_xrange = (x1, x2)
        l3_xrange = (x2, xf)

        x = np.linspace(0., 1., int(n))*xf
        y = np.zeros(int(n))

        l1_indices = np.logical_and(x >= l1_xrange[0], x <= l1_xrange[1])
        l2_indices = np.logical_and(x >  l2_xrange[0], x <= l2_xrange[1])
        l3_indices = np.logical_and(x >  l3_xrange[0], x <= l3_xrange[1])

        # First parabolic segment: else-case corresponds to the degenerate case
        # with only linear segment and no parabolic segments
        l1_p = y1/x1**2 if x1 > 0 else 0.
        # Middle linear segment: else-case corresponds to the degenerate case
        # with only parabolic segments, and no linear segment
        l2_m = (y2-y1)/(x2-x1) if x1 != x2 else 0.
        # Second parabolic segment: else-case inherited from l1_p
        l3_p = -l1_p

        y[l1_indices] = eq_parabola(x[l1_indices], l1_p, x0, x0)
        y[l2_indices] = eq_line(    x[l2_indices], l2_m, x1, y1)
        y[l3_indices] = eq_parabola(x[l3_indices], l3_p, xf, yf)

        # Force the start to be zero (in case of precision error)
        x[0] = 0
        y[0] = 0

        return x, y, [(x0,y0), (x1,y1), (x2,y2), (xf,yf)]

    def track_init(self, varname=None, debug=False): #
        r''' Tracking of the SPSCavityFeedback without beam.
        '''

       #self.V_part_ramp = np.array([np.array(elem) for elem in list(zip(np.linspace(0,1,self.turns), np.linspace(0.5*np.pi*self.Vset_imag,0,self.turns)))])
       #self._V_part_ramp = np.array([np.array(elem) for elem in list(zip(np.ones(self.turns), np.zeros(self.turns)))])

        ############

        # if not self.Commissioning.use_pretrack_ramp:

        #     # # OPT 1: No ramp (default):
        #     # fact_ramp = np.ones(self.turns)
        #     self.OTFB_1_V_part_rampfact_fine   = np.array([ None for elem in range(self.turns) ])
        #     self.OTFB_2_V_part_rampfact_fine   = np.array([ None for elem in range(self.turns) ])
        #     self.OTFB_1_V_part_rampfact_coarse = np.array([ None for elem in range(self.turns) ])
        #     self.OTFB_2_V_part_rampfact_coarse = np.array([ None for elem in range(self.turns) ])

        # # # OPT 2: Ramp of voltage over the pretracking turns:
        # # # Using the same normalized ramp for the partioned voltage of each OTFB
        # # t_ramp, fact_ramp, pts = self.create_ramp(xf=self.turns-1, yf=1., n=self.turns, b=0.25)
        # # t_ramp = t_ramp.astype(int)

        # # # For OPT 1 and 2:
        # # # Ramp in magnitude, keep phase
        # # self.OTFB_1_V_part_rampfact_coarse = np.array([ np.array(elem) for elem in list(zip(fact_ramp, np.zeros(self.turns))) ])
        # # self.OTFB_2_V_part_rampfact_coarse = np.array([ np.array(elem) for elem in list(zip(fact_ramp, np.zeros(self.turns))) ])

        if self.Commissioning.use_pretrack_ramp:

            # print(f'fillpattern = {self.fillpattern}, shape = {self.fillpattern.shape}')

            # OPT3: Ramp of voltage along the first turn only (reach max after in the span of self.fillpattern[-1]-self.fillpattern[0] buckets):
            # Using the same normalized ramp for the partioned voltage of each OTFB
            t_ramp_coarse, fact_ramp_coarse_tmp, pts_coarse = self.create_ramp(xf=self.fillpattern[-1]-self.fillpattern[0], yf=1., n=len(self.fillpattern), b=0.25)
            t_ramp_coarse = t_ramp_coarse.astype(int)
            pts_coarse = [ (ptsi[0] + self.fillpattern[0], ptsi[1]) for ptsi in pts_coarse]
            # print(f't_ramp_coarse = {t_ramp_coarse}, shape = {t_ramp_coarse.shape}')
            # print(f'fact_ramp_coarse_tmp = {fact_ramp_coarse_tmp}, shape = {fact_ramp_coarse_tmp.shape}')
            print(f'pts_coarse = {pts_coarse}')
            fact_ramp_coarse = np.zeros(self.OTFB_1.n_coarse) # should be the same for OTFB_2
            fact_ramp_coarse[  self.fillpattern        ] = fact_ramp_coarse_tmp
            fact_ramp_coarse[  self.fillpattern[-1]+1: ] = np.ones(self.OTFB_1.n_coarse - self.fillpattern[-1] - 1)
            for i in range(1,self.fillpattern[1]-self.fillpattern[0]):
                # Keep the same voltage in the empty buckets following a filled bucket (where the voltage was increased)
                fact_ramp_coarse[ self.fillpattern+i ] = fact_ramp_coarse[ self.fillpattern ]
            # self.OTFB_1_V_part_rampfact_coarse_0 = np.array([ None for elem in range(self.turns) ])
            # self.OTFB_2_V_part_rampfact_coarse_0 = np.array([ None for elem in range(self.turns) ])
            self.OTFB_1_V_part_rampfact_coarse_0 = np.array([ fact_ramp_coarse, np.ones(self.OTFB_1.n_coarse)*0.0*np.pi ])
            self.OTFB_2_V_part_rampfact_coarse_0 = np.array([ fact_ramp_coarse, np.ones(self.OTFB_2.n_coarse)*0.0*np.pi ])

            if self.Commissioning.use_gen_fine:

                Ns = self.OTFB_1.profile.n_slices // self.OTFB_1.rf.harmonic[0,0]
                # print(f'Ns = {Ns}')
                # print(int((self.fillpattern[-1]-self.fillpattern[0])*Ns), int(len(self.fillpattern)*Ns+1))

                t_ramp_fine, fact_ramp_fine_tmp, pts_fine = self.create_ramp(xf=int((self.fillpattern[-1]-self.fillpattern[0])*Ns), yf=1., n=int((self.fillpattern[-1]-self.fillpattern[0])*Ns)+1, b=0.25)
                t_ramp_fine = t_ramp_fine.astype(int)
                pts_fine = [ (ptsi[0] + self.fillpattern[0]*Ns, ptsi[1]) for ptsi in pts_fine]
                # print(f't_ramp_fine = {t_ramp_fine}, shape = {t_ramp_fine.shape}')
                # print(f'fact_ramp_fine_tmp = {fact_ramp_fine_tmp}, shape = {fact_ramp_fine_tmp.shape}')
                print(f'pts_fine = {pts_fine}')
                fact_ramp_fine = np.zeros(self.OTFB_1.n_fine) # should be the same for OTFB_2
                fact_ramp_fine[  int( self.fillpattern[ 0]*Ns)   :int((self.fillpattern[-1]*Ns)+1) ] = fact_ramp_fine_tmp
                fact_ramp_fine[  int((self.fillpattern[-1]*Ns)+1):                                 ] = np.ones(self.OTFB_1.n_fine - int(self.fillpattern[-1]*Ns) - 1)
                # self.OTFB_1_V_part_rampfact_fine_0 = np.array([ None for elem in range(self.turns) ])
                # self.OTFB_2_V_part_rampfact_fine_0 = np.array([ None for elem in range(self.turns) ])
                self.OTFB_1_V_part_rampfact_fine_0 = np.array([ fact_ramp_fine, np.ones(self.OTFB_1.n_fine)*0.0*np.pi ])
                self.OTFB_2_V_part_rampfact_fine_0 = np.array([ fact_ramp_fine, np.ones(self.OTFB_2.n_fine)*0.0*np.pi ])

            # quit()

            ############

           # print('')
           #if self.Commissioning.use_gen_fine:
            #     print(f't_ramp_fine   = {t_ramp_fine},   shape = {t_ramp_fine.shape}')
            #     print(f't_ramp_coarse = {t_ramp_coarse}, shape = {t_ramp_coarse.shape}')
            # print(f'fact_ramp_fine   = {fact_ramp_fine},   shape = {fact_ramp_fine.shape}')
            # print(f'fact_ramp_coarse = {fact_ramp_coarse}, shape = {fact_ramp_coarse.shape}')
           # print('')
            if self.Commissioning.use_gen_fine:
                print(f'OTFB_1_V_part_rampfact_fine_0   = {self.OTFB_1_V_part_rampfact_fine_0},   shape = {self.OTFB_1_V_part_rampfact_fine_0.shape}')
                print(f'OTFB_2_V_part_rampfact_fine_0   = {self.OTFB_2_V_part_rampfact_fine_0},   shape = {self.OTFB_2_V_part_rampfact_fine_0.shape}')
            print(f'OTFB_1_V_part_rampfact_coarse_0 = {self.OTFB_1_V_part_rampfact_coarse_0}, shape = {self.OTFB_1_V_part_rampfact_coarse_0.shape}')
            print(f'OTFB_2_V_part_rampfact_coarse_0 = {self.OTFB_2_V_part_rampfact_coarse_0}, shape = {self.OTFB_2_V_part_rampfact_coarse_0.shape}')
           # print('')
            # quit()

            # ####

            # fig, ax = plt.subplots()

            # # Fine:
            # t_ramp = np.arange(self.OTFB_1.n_fine)
            # fact_ramp = np.copy(fact_ramp_fine)
            # pts = np.copy(pts_fine)

            # ax.plot(t_ramp/Ns, fact_ramp, '-', color='blue')
            # for t1 in [pts[i][0] for i in range(len(pts))]:
            #     ax.axvline(t1/Ns, color='blue', ls='--', alpha=0.5)
            # for f1 in [pts[i][1] for i in range(len(pts))]:
            #     ax.axhline(f1, color='blue', ls='--', alpha=0.5)
            # #ax.scatter([0.5*pts[-1][0]], [0.5*pts[-1][1]], color='grey')

            # # Coarse:
            # t_ramp = np.arange(self.OTFB_1.n_coarse)
            # fact_ramp = np.copy(fact_ramp_coarse)
            # pts = np.copy(pts_coarse)

            # ax.plot(t_ramp, fact_ramp, '-.', color='red')
            # for t1 in [pts[i][0] for i in range(len(pts))]:
            #     ax.axvline(t1, color='red', ls=':', alpha=0.5)
            # for f1 in [pts[i][1] for i in range(len(pts))]:
            #     ax.axhline(f1, color='red', ls=':', alpha=0.5)
            # #ax.scatter([0.5*pts[-1][0]], [0.5*pts[-1][1]], color='grey')

            # # ax.set_xlabel('Turns pre-track')
            # ax.set_xlabel('Bin (fine/Ns, coarse)')
            # ax.set_ylabel('Partioned voltage amplitude ramp factor')

            # fig.tight_layout()
            # fig.savefig('plot_ramp_plp.png')
            # plt.cla()
            # plt.close(fig)

            # quit()

        ####

        if debug:

            # matplotlib.use('qt5agg')

            # Check also the fnamefull below, where a label (e.g. variable name) might be appended
            # as well as turns_to_plot
            if varname is None:
                fname = 'plot_cavityfeedback_pretrack'
            else:
                # fname = 'plot_pretrack_360bm-fmeas-conv4-rampNo-genfineNo'
                fname = 'plot_pretrack_powertest'

            # Overrride varname (remove the _coarse at the end) to get a plot for debugging
            # varname = 'Q_gen'

            nrows = 4
            ncols = 3
            fig, ax = plt.subplots(nrows, ncols, sharex=True) #, sharey='row') # Pre-tracking without beam
            fig.set_size_inches(ncols*5.00, nrows*2.50)
            #ax = plt.axes([0.18, 0.1, 0.8, 0.8])

            ax[0,0].set_title(f'{self.OTFB_1.cavtype} TWCs (Vpart = {self.OTFB_1.V_part:.6f}, {self.OTFB_1.V_part/self.OTFB_1.n_cavities:.6f} each)'+'\n')
            ax[0,1].set_title(f'{self.OTFB_2.cavtype} TWCs (Vpart = {self.OTFB_2.V_part:.6f}, {self.OTFB_2.V_part/self.OTFB_1.n_cavities:.6f} each)'+'\n')
            ax[0,2].set_title(f'Total V_ind_gen_coarse ({self.OTFB_1.V_part+self.OTFB_2.V_part:.6f})'+'\n')
            for spi in range(nrows):
                for spj in range(ncols):
                    ax[spi,spj].grid(axis='y')
                    if spi == nrows-1:
                        ax[spi,spj].set_xlabel(r'Time [$\mu$s]')

            # turns_to_plot = np.arange(self.turns)                       # All turns
            # turns_to_plot = np.array([0])                               # First turn
            # turns_to_plot = np.array([1])                               # Second turn
            turns_to_plot = np.array([self.turns-1])                    # Last turn
            # turns_to_plot = np.array([0, self.turns-1])                 # First and last turns
            # turns_to_plot = np.arange(0,self.turns)                     # All turns
            # turns_to_plot = np.arange(self.turns-5,self.turns)          # Last few turns
            # turns_to_plot = np.array([self.turns-2,self.turns-1])       # Last two turns
            # turns_to_plot = np.arange(0,self.turns,int(self.turns/10))  # All turns, but plot only 10 turns evenly spaced
            # turns_to_plot = np.concatenate((np.arange(0,5), np.arange(self.turns-5,self.turns))) # First few and last few turns
            # print(f'turns_to_plot = {turns_to_plot}')

            cmap = plt.get_cmap('coolwarm') # 'jet': rainbow, 'coolwarm': blur to red
            colors = cmap(np.linspace(0,1, len(turns_to_plot)))
            colors[-1][:3] *= 0. # Last one is black (r,g,b,a)
            # print(f'colors = {colors}')

        print('')
        print('i | V_ind_gen_coarse (midpoint) OTFB_1 | OTFB_2:')
        # print('i | Q_gen_coarse (midpoint) OTFB_2:')
        # print('i | V_ind_gen_fine (midpoint) OTFB_2 | V_ind_gen_coarse (midpoint) OTFB_2:')
        for i in range(self.turns):
            self.logger.debug("Pre-tracking w/o beam, iteration %d", i)
            #
            if not self.Commissioning.use_pretrack_ramp or i >= 1:

                # print(f'i = {i}')

                if self.nollrf:
                    self.OTFB_1.track_no_beam_nollrf()
                    self.OTFB_2.track_no_beam_nollrf()
                else:
                    self.OTFB_1.track_no_beam()
                    self.OTFB_2.track_no_beam()

            else:

                # print(f'i = {i}: Using ramp')

                if self.nollrf:
                    self.OTFB_1.track_no_beam_nollrf(self.OTFB_1_V_part_rampfact_fine_0 if self.Commissioning.use_gen_fine else None, self.OTFB_1_V_part_rampfact_coarse_0)
                    self.OTFB_2.track_no_beam_nollrf(self.OTFB_2_V_part_rampfact_fine_0 if self.Commissioning.use_gen_fine else None, self.OTFB_2_V_part_rampfact_coarse_0)
                else:
                    self.OTFB_1.track_no_beam(self.OTFB_1_V_part_rampfact_fine_0 if self.Commissioning.use_gen_fine else None, self.OTFB_1_V_part_rampfact_coarse_0)
                    self.OTFB_2.track_no_beam(self.OTFB_2_V_part_rampfact_fine_0 if self.Commissioning.use_gen_fine else None, self.OTFB_2_V_part_rampfact_coarse_0)

            if i < 3 or i > self.turns - 3:
                print('{0:6d}'.format(i), end=' | ')
                val1 = self.OTFB_1.V_ind_gen_coarse[int(0.5*self.OTFB_1.n_coarse)]
                val2 = self.OTFB_2.V_ind_gen_coarse[int(0.5*self.OTFB_2.n_coarse)]
                # val3 = self.OTFB_2.Q_gen_coarse[int(0.5*self.OTFB_2.n_coarse)]
                print('Re = {0:+.6e}, Im = {1:+.6e}, Abs = {2:+.6e}, Ang = {3:+.6e}'.format(np.real(val1), np.imag(val1), np.abs(val1), np.angle(val1, deg=True)), end=' | ')
                print('Re = {0:+.6e}, Im = {1:+.6e}, Abs = {2:+.6e}, Ang = {3:+.6e}'.format(np.real(val2), np.imag(val2), np.abs(val2), np.angle(val2, deg=True)))
                # print('Re = {0:+.6e}, Im = {1:+.6e}, Abs = {2:+.6e}, Ang = {3:+.6e}'.format(np.real(val3), np.imag(val3), np.abs(val3), np.angle(val3, deg=True)))
                # print('OTFB1:', self.OTFB_1.V_ind_gen_coarse, np.abs(self.OTFB_1.V_ind_gen_coarse), np.angle(self.OTFB_1.V_ind_gen_coarse))
                # print('OTFB2:', self.OTFB_2.V_ind_gen_coarse, np.abs(self.OTFB_2.V_ind_gen_coarse), np.angle(self.OTFB_2.V_ind_gen_coarse))
            #quit()


            if debug and i in turns_to_plot:

                # print(i)
                color_i = colors[ np.where(turns_to_plot == i)[0][0] ]
                # print(f'color_i = {color_i}')

                labeli = f'Turn {i}' if i in [turns_to_plot[0], turns_to_plot[-1]] else None
                # if(self.turns <= 200 or (self.turns > 200 and i%5==0) or i == self.turns-1):

                # BE CAREFUL NOT NO REQUEST A PARAMEER IN THE FINE GRID THAT IT'S ONLY COMPUTED WHEN self.Commissioning.use_gen_fine = True
                if varname is None:
                    varname = 'V_ind_gen'
                    # varname = 'dV_Hcav'
                    fc = 'coarse'
                    var = f'{varname}_{fc}'
                    yfact = 1e6
                    yunit = 'MV'
                else:
                    fc = 'coarse'
                    var = f'{varname}_{fc}'
                    yfact = 1
                    yunit = 'unit'

                npts_i =  getattr(self.OTFB_1, f'n_{fc}')
                time_array_i = getattr(self.OTFB_1, f't_{fc}')[:npts_i]*1e6
                array1 = getattr(self.OTFB_1, f'{var}')[:npts_i]
                array2 = getattr(self.OTFB_2, f'{var}')[:npts_i]
                # Use this if plotting h_gen:
                # array1 = getattr(self.OTFB_1.TWC, f'{var}')[:npts_i]
                # array2 = getattr(self.OTFB_2.TWC, f'{var}')[:npts_i]

                # Re
                ax[0,0].plot(time_array_i,  np.real(array1)/yfact,               color=color_i, ls='-', label=None) #linestyle='', marker='.')
                ax[0,1].plot(time_array_i,  np.real(array2)/yfact,               color=color_i, ls='-', label=labeli) #linestyle='', marker='.')
                # ax[0,2].plot(time_array_i,  np.real(array1 + array2)/yfact,      color=color_i, ls='-', label=None) #linestyle='', marker='.')
                # Im
                ax[1,0].plot(time_array_i,  np.imag(array1)/yfact,               color=color_i, ls='-', label=None) #linestyle='', marker='.')
                ax[1,1].plot(time_array_i,  np.imag(array2)/yfact,               color=color_i, ls='-', label=labeli) #linestyle='', marker='.')
                # ax[1,2].plot(time_array_i,  np.imag(array1 + array2)/yfact,      color=color_i, ls='-', label=None) #linestyle='', marker='.')
                # Abs
                ax[2,0].plot(time_array_i,   np.abs(array1)/yfact,               color=color_i, ls='-', label=None) #linestyle='', marker='.')
                ax[2,1].plot(time_array_i,   np.abs(array2)/yfact,               color=color_i, ls='-', label=labeli) #linestyle='', marker='.')
                # ax[2,2].plot(time_array_i,   np.abs(array1 + array2)/yfact,      color=color_i, ls='-', label=labeli) #linestyle='', marker='.')
                # Ang
                ax[3,0].plot(time_array_i, np.angle(array1)/np.pi*180.,          color=color_i, ls='-', label=None) #linestyle='', marker='.')
                ax[3,1].plot(time_array_i, np.angle(array2)/np.pi*180.,          color=color_i, ls='-', label=labeli) #linestyle='', marker='.')
                # ax[3,2].plot(time_array_i, np.angle(array1 + array2)/np.pi*180., color=color_i, ls='-', label=None) #linestyle='', marker='.')

        # if debug and False: #0 not in turns_to_plot: # Hide set-point for a better yscale of the result (for numerical errors inspection)
        #     for spj in range(ncols):
        #         # The total voltage per cavity types (set-point): there's a 90 deg difference in the definition w.r.t. V_tot_fine, so we compensate by real -> imag, imag -> real, and -90 deg for angle
        #         # Re
        #         ax[0,0].axhline( np.imag(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0])/1e6, ls=':', color='#888888')
        #         ax[0,1].axhline( np.imag(self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/1e6, ls=':', color='#888888')
        #        #ax[0,2].axhline( np.imag(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0] + self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/1e6, ls=':', color='#888888')
        #         ax[0,2].axhline( np.imag(self.OTFB_1.rf.voltage[0,0])/1e6, ls=':', color='#888888') # The latter should ad up identically to this
        #         # Im
        #         ax[1,0].axhline( np.real(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0])/1e6,  ls=':', color='#888888')
        #         ax[1,1].axhline( np.real(self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/1e6,  ls=':', color='#888888')
        #        #ax[1,2].axhline( np.real(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0] + self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/1e6, ls=':', color='#888888')
        #         ax[1,2].axhline( np.real(self.OTFB_1.rf.voltage[0,0])/1e6, ls=':', color='#888888') # The latter should ad up identically to this
        #         # Abs
        #         ax[2,0].axhline(  np.abs(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0])/1e6,  ls=':', color='#888888')
        #         ax[2,1].axhline(  np.abs(self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/1e6,  ls=':', color='#888888')
        #        #ax[2,2].axhline(  np.abs(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0] + self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/1e6, ls=':', color='#888888')
        #         ax[2,2].axhline(  np.abs(self.OTFB_1.rf.voltage[0,0])/1e6, ls=':', color='#888888') # The latter should ad up identically to this
        #         # Ang
        #         ax[3,0].axhline(np.angle(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0])/np.pi*180.-90.,  ls=':', color='#888888')
        #         ax[3,1].axhline(np.angle(self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/np.pi*180.-90.,  ls=':', color='#888888')
        #        #ax[3,2].axhline(np.angle(self.OTFB_1.V_part*self.OTFB_1.rf.voltage[0,0] + self.OTFB_2.V_part*self.OTFB_2.rf.voltage[0,0])/np.pi*180., ls=':', color='#888888')
        #         ax[3,2].axhline(np.angle(self.OTFB_1.rf.voltage[0,0])/np.pi*180.-90., ls=':', color='#888888') # The latter should ad up identically to this

        # c0max = np.argmax(np.abs(getattr(self.OTFB_2, f'{var}')))
        # c0min = np.argmin(np.abs(getattr(self.OTFB_2, f'{var}')))
        # c0 = np.average([c0max, c0min])
        # print(f'Center of pulse: 0.5*({c0max}+{c0min}) = {c0} -> {c0*(time_array_i[1]-time_array_i[0])/1e6} s' )

        print('')

        # quit()

        # # Interpolate from the coarse mesh to the fine mesh of the beam
        # # MYEDIT: 2020.12.14: Important that t_fine and t_coarse have the same
        # # phase refenrece. Also, both time arrays are the same, by their
        # # definitions, for both OTFB_1 and OTFB_2:
        if self.Commissioning.use_gen_fine:
            self.V_sum = np.copy(self.OTFB_1.V_ind_gen_fine) + np.copy(self.OTFB_2.V_ind_gen_fine)
        else:
            self.V_sum = np.interp(self.OTFB_1.t_fine,   # self.OTFB_1.profile.bin_centers,
                                   self.OTFB_1.t_coarse, # self.OTFB_1.rf_centers,
                                   self.OTFB_1.V_ind_gen_coarse + self.OTFB_2.V_ind_gen_coarse)

        self.V_corr, alpha_sum = cartesian_to_polar(self.V_sum)

        if debug:

            # ax[0,2].plot(self.OTFB_1.t_fine/1e-6,  np.real(self.V_sum)/1e6,        color='green', ls='-', label=None) #linestyle='', marker='.')
            # ax[1,2].plot(self.OTFB_1.t_fine/1e-6,  np.imag(self.V_sum)/1e6,        color='green', ls='-', label=None) #linestyle='', marker='.')
            # ax[2,2].plot(self.OTFB_1.t_fine/1e-6,   np.abs(self.V_sum)/1e6,        color='green', ls='-', label=None) #linestyle='', marker='.')
            # ax[3,2].plot(self.OTFB_1.t_fine/1e-6, np.angle(self.V_sum)/np.pi*180., color='green', ls='-', label='Vsum') #linestyle='', marker='.')

            # for spi in range(nrows):
            #     for spj in range(ncols):
            #         ax[spi,spj].set_xlim(11.5,13.50) # The middle of the turn for no-beam (T_rev ~ 23 us)

            ax[0,0].set_ylabel(f'Real [{yunit}]')
            ax[1,0].set_ylabel(f'Imaginary [{yunit}]')
            ax[2,0].set_ylabel(f'Absolute [{yunit}]')
            ax[3,0].set_ylabel(f'Angle [deg]')

            fig.suptitle(var)
            ax[nrows-1,1].legend(loc=4)

            # plt.show()
            fig.tight_layout(rect=[0.0, 0.0, 1.00, 0.950]) #rect=[0.0, 0.0, 1.00, 0.95])
            if fname == 'plot_cavityfeedback_pretrack':
                fnamefull = fname[:]
            else:
                fnamefull = f'{fname}_{var}' #'_500'
                #fnamefull = f'{fname}_{var}_movaveprev' #'_500'
                #fnamefull = f'{fname}_{var}_movaveprev2' #'_500'
            fig.savefig(f'{self.outdir}/{fnamefull}')
            fig.clf()
            print(f'Saving {fnamefull}.png ...')

        # print(f'OTFB_1: Q_gen_abs_max_coarse = {self.OTFB_1.Q_gen_abs_max_coarse}')
        # print(f'OTFB_2: Q_gen_abs_max_coarse = {self.OTFB_2.Q_gen_abs_max_coarse}')
        # quit()

        # Calculate OTFB correction w.r.t. RF voltage and phase in RFStation
        self.V_corr /= self.rf.voltage[0, self.rf.counter[0]]
        #if self.OTFB_1.dphi_rf_opt == self.OTFB_2.dphi_rf_opt and self.OTFB_1.dphi_rf_opt == 0:
        #    # Removing phi_rf removes the rf phase with the correction dphi_rf
        #    # (applied by the tracker in the previous turn if phase loop present).
        #    # This IS a problem since the correction to replace this phase
        #    # was computed at the right at zero dphi_rf. We therefore have to
        #    # add it back (TO CHECK IF THIS COMPENSATION OF DPHI_RF IS ACTUALLY
        #    # THE SAME THAN THE ONE ADDED BY THE TRACKER IN THE PREVIOUS TURN):
        #    self.phi_corr = 0.5*np.pi*self.Vset_imag - alpha_sum \
        #        - self.rf.phi_rf[0, self.rf.counter[0]] + self.rf.dphi_rf[0]
        #elif self.OTFB_1.dphi_rf_opt == self.OTFB_2.dphi_rf_opt and self.OTFB_1.dphi_rf_opt == 1:
        #    # Removing phi_rf removes the rf phase with the correction dphi_rf
        #    # (applied by the tracker in the previous turn if phase loop present).
        #    # This is not a problem since the correction to replace this phase
        #    # as already computed at the right dphi_rf reference.
        self.phi_corr = 0.5*np.pi*self.Commissioning.Vset_imag - alpha_sum \
            - self.rf.phi_rf[0, self.rf.counter[0]]
        #
        #else:
        #    raise RuntimeError("Both SPSCavityFeedbacks (1 and 2) must use the same " +
        #                       "convention for the reference of the profile.bin_centers " +
        #                       "and rf_centers!")


class SPSOneTurnFeedback(object):

    r'''Voltage feedback around a travelling wave cavity with given amount of
    sections. The quantities of the LLRF system cover one turn with a coarse
    resolution.

    Parameters
    ----------
    RFStation : class
        An RFStation type class
    Beam : class
        A Beam type class
    Profile : class
        Beam profile object
    n_sections : int
        Number of sections in the cavities
    n_cavities : int
        Number of cavities of the same type
    V_part : float
        Voltage partition for the given n_cavities; in range (0,1)
    G_ff : float
        FF gain [1]; default is 1
    G_llrf : float
        LLRF gain [1]; default is 10
    G_tx : float
        Transmitter gain [A/V]; default is :math:`(50 \Omega)^{-1}`
    open_loop : int(bool)
        Open (0) or closed (1) feedback loop; default is 1
    deltaf0 : float
        Central (resonance) frequency offset :math:`\Delta f_0` in
        :math:`2 \Pi (\f + \Delta f_0)` [Hz]; default is 0

    Attributes
    ----------
    TWC : class
        A TravellingWaveCavity type class
    counter : int
        Counter of the current time step
    omega_c : float
        Carrier revolution frequency [1/s] at the current time step
    omega_r : const float
        Resonant revolution frequency [1/s] of the travelling wave cavities
    n_coarse : int
        Number of bins for the coarse grid (equals harmonic number)
    V_gen_coarse : complex array
        Generator voltage [V] of the present turn in (I,Q) coordinates
    V_gen_prev : complex array
        Generator voltage [V] of the previous turn in (I,Q) coordinates
    V_ind_beam_fine : complex array
        Beam-induced voltage [V] in (I,Q) coordinates on the fine grid
        used for tracking the beam
    V_ind_beam_coarse : complex array
        Beam-induced voltage [V] in (I,Q) coordinates on the coarse grid used
        tracking the LLRF
    V_ind_gen_coarse : complex array
        Generator-induced voltage [V] in (I,Q) coordinates on the coarse grid
        used tracking the LLRF
    V_tot_coarse : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates which is used
        for tracking the LLRF
    V_tot_fine : complex array
        Cavity voltage [V] at present turn in (I,Q) coordinates which is used
        for tracking the beam
    a_comb : float
        Recursion constant of the comb filter; :math:`a_{\mathsf{comb}}=15/16`
    n_mov_av_coarse : const int
        Number of points for moving average modelling cavity response;
        :math:`n_{\mathsf{mov.av.}} = \frac{f_r}{f_{\mathsf{bw,cav}}}`, where
        :math:`f_r` is the cavity resonant frequency of TWC_4 and TWC_5
    logger : logger
        Logger of the present class

    '''

    def __init__(self, RFStation, Beam, Profile, n_sections, n_cavities, # MYEDIT 2020.10.01 Original: n_cavities=2
                 V_part, G_ff=1, G_llrf=10, G_tx=0.5, a_comb=15/16,   # MYEDIT 2020.10.01 Original: V_part=4/9
                 Commissioning=CavityFeedbackCommissioning(),
                 deltaf0=0, # MYEDIT 2020.11.24: Added central (resonance) frequency offset
                 fillpattern=None, # MYEDIT 2020.02.02: Added fillpattern
                 dV_Hcav_opt='0',
                 power_clamp=False,
                 nollrf=False
                 ):

        # Set up logging
        self.logger = logging.getLogger(__class__.__name__)

        # Commissioning options
        self.open_loop = Commissioning.open_loop
        if self.open_loop == 0:
            self.logger.debug("Opening overall OTFB loop")
        elif self.open_loop == 1:
            self.logger.debug("Closing overall OTFB loop")
        self.open_FB = Commissioning.open_FB
        if self.open_FB == 0:
            self.logger.debug("Opening feedback of drive correction")
        elif self.open_FB == 1:
            self.logger.debug("Closing feedback of drive correction")
        self.open_drive = Commissioning.open_drive
        if self.open_drive == 0:
            self.logger.debug("Opening drive to generator")
        elif self.open_drive == 1:
            self.logger.debug("Closing drive to generator")
        self.open_FF = Commissioning.open_FF
        if self.open_FF == 0:
            self.logger.debug("Opening feed-forward on beam current")
        elif self.open_FF == 1:
            self.logger.debug("Closing feed-forward on beam current")

        # Read input
        self.rf = RFStation
        self.beam = Beam
        self.profile = Profile

        #####

        # Some added attributes:

        self.cavtype = f'{n_cavities}x{n_sections}sec' # MYEDIT 2020.10.01
        print(f'cavtype = {self.cavtype}') # MYEDIT 2020.10.01

        ## MYEDIT: 2020.12.09:
        # Keep the rf_centers (t_coarse) w.r.t. to phase reference of zero,
        # and define the t_fine form the profile correcting any offset phase
        # reference due to dphi_rf (if applicable). This is the correct
        # option: rf_centers should remain constant, since they are calculated
        # by the LLRF based on a fixed clock, and the profile should be shifted
        # internally in this module to correspond to its.
       #self.dphi_rf_opt = 0
        # Use the current profile with offset phase reference due to dphi_rf
        # (if applicable) to t_fine, and add the necessary correction to
        # rf_centers (t_coarse). This is simpler to implement, but in reality
        # rf_centers should be kept constant, since they are calculated by the
        # LLRF based on a fixed clock.
       #self.dphi_rf_opt = 1
        # Old: no 'if' option exists in the code whenever is should
        # be needed, so basically the old ways with no correction
        self.dphi_rf_opt = 2

        self.Vset_imag = Commissioning.Vset_imag
        self.redef_Zb = Commissioning.redef_Zb
        self.redef_Ib = Commissioning.redef_Ib
        self.redef_Vb = Commissioning.redef_Vb
        self.redef_Vbff = Commissioning.redef_Vbff

        self.add_ff_to_gen = Commissioning.add_ff_to_gen

        #self.Vindbeamsign = +1 # -1 Original, +1 for tests: the good one, permanently changed now

        self.use_gen_fine = Commissioning.use_gen_fine
        self.nollrf = nollrf

        self.dV_Hcav_opt = str(dV_Hcav_opt)

        self.outdir = Commissioning.outdir

        ######

        self.n_cavities = int(n_cavities)
        if self.n_cavities < 1:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_cavities has invalid value!")
        self.V_part = float(V_part)
        if self.V_part*(1 - self.V_part) < 0:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: V_part" +
                               " should be in range (0,1)!")
        # MYEDIT 2020.10.01:
        ##self.V_part_cav = self.V_part/self.n_cavities
        #print(f'V_part     = {self.V_part}')
        ##print(f'V_part_cav = {self.V_part_cav}')
        #print('')
        # MYEDIT 2020.11.24
        self.deltaf0 = deltaf0
        print(f'deltaf0 = {self.deltaf0}')

        # Gain settings
        self.G_ff = float(G_ff)
        self.G_llrf = float(G_llrf)
        self.G_tx = float(G_tx)

        # Length of arrays in LLRF
        # we need the fine either way
        self.n_fine   = int(self.profile.n_slices)
        print(f'self.n_fine = {self.n_fine}')
        self.n_coarse = int(self.rf.t_rev[0]/self.rf.t_rf[0, 0])
        print(f'self.n_coarse = {self.n_coarse}')

        self.fillpattern   = fillpattern
        self.nbs           = self.fillpattern[1]-self.fillpattern[0] # In the SPS, this has to be 5 for the FIR filter to work
        self.bunch_spacing = self.nbs * self.rf.t_rf[0,0] # notw we wont use T_s_coarse to ensure that it doesnt change

        if self.fillpattern[-1] + self.nbs == self.n_coarse:
        # if int(len(self.fillpattern) * self.nbs) == self.n_coarse: # Alterntive
            self.flag_macfull = True
            self.Ns = int(self.profile.n_slices // self.rf.harmonic[0,0])
        else:
            self.flag_macfull = False

        # 200 MHz travelling wave cavity (TWC) model
        if n_sections in [3, 4, 5]:

            if self.redef_Zb:
                self.TWC = eval( f'SPS{ str(n_sections) }Section200MHzTWC( {str(self.deltaf0)}, Zb_negative=True )' )
            else:
                # Original:
                self.TWC = eval( f'SPS{ str(n_sections) }Section200MHzTWC( {str(self.deltaf0)} )' )

            # TWC resonant frequency # MYEDIT 2020.12.14: Moved up here
            self.omega_r = self.TWC.omega_r

            if self.open_FF == 1 and not self.use_gen_fine: # Not yet implemented for fine generator

                self.nbs_FF = 5 # 5 # self.nbs # Having nbs_FF = nbs  (and not necessarily being exactly 5) will alow having the FIR filter to be applied at the same bunch spacing but the coefficients might not necessarily be the optimum and do not correspond the the coeffiecientes of the FIR filter in the real machine
                if self.nbs != self.nbs_FF:
                    # The position of the 1st bunch doesn't need to be at a multiple of 5
                    print("[!] Warning: bunch separation for FF computations is assumed to be 5, "+
                          f"but actual bunch separation is {self.nbs}!")

                #self.delay_FF_opt = 0 # MYEDIT: 2020.09.04
                self.delay_FF_opt = 2 # MYEDIT: 2021.02.04 New default

                # Feed-forward filter
                self.coeff_FF = getattr(sys.modules[__name__],
                    "feedforward_filter_TWC" + str(n_sections))
                #self.coeff_FF = np.flip(self.coeff_FF) # MYEDIT: 2020.09.04
                self.n_FF = len(self.coeff_FF)
                self.n_FF_delay = int(0.5*(self.n_FF - 1) +
                                      0.5*self.TWC.tau/self.rf.t_rf[0, 0]/self.nbs_FF)
                self.logger.debug("Feed-forward delay in samples %d",
                                  self.n_FF_delay)
                # Multiply gain by normalisation factors from filter and
                # beam-to generator current
                self.G_ff *= self.TWC.R_beam/(self.TWC.R_gen *
                                              np.sum(self.coeff_FF))

        else:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: argument" +
                               " n_sections has invalid value!")
        self.logger.debug("SPS OTFB cavities: %d, sections: %d, voltage" +
                          " partition %.2f, gain: %.2e", self.n_cavities,
                          n_sections, self.V_part, self.G_tx)

        # Moving average modelling cavity response
        # MYEDIT: 2020.12.14 Moved up here
        if self.use_gen_fine:
            self.n_mov_av_fine = int(round(self.TWC.tau/self.profile.bin_size))
            print(f'n_mov_av_fine = {self.n_mov_av_fine}')
        else:
            # It's still needed, but it'll be defiend differently below.
            # TODO: It is formally not correct (the correct is the one above,
            # and should be debugged)
            pass
        self.n_mov_av_coarse   = int(round(self.TWC.tau/self.rf.t_rf[0, 0]))  # VS HTIMKO 2021-02: self.n_mov_av = int(np.rint(self.TWC.tau/self.rf.t_rf[0, 0])) -- should be the same
        print(f'n_mov_av_coarse = {self.n_mov_av_coarse}')
        #
        self.logger.debug("Moving average over %d points", self.n_mov_av_coarse)
        # Initialise moving average
        if self.n_mov_av_coarse < 2:
            raise RuntimeError("ERROR in SPSOneTurnFeedback: profile has to" +
                               " have at least 12.5 ns resolution!")

        # Length of long arrays in LLRF (for charge/current signals)
        if self.use_gen_fine:
            self.n_fine_long = int(self.n_fine   + self.n_mov_av_fine)
        else:
            # We need either way
            self.n_fine_long = int(self.n_fine * (1 + self.n_mov_av_coarse/self.rf.harmonic[0,0])) # n_fine/rf.harmonic = slices per bucket -> n_mov_av_coarsee (i.e. a given no. of buckets) = no. of slices to add for long fine
        print(f'self.n_fine_long = {self.n_fine_long}')
        self.n_coarse_long   = int(self.n_coarse + self.n_mov_av_coarse)
        print(f'self.n_coarse_long = {self.n_coarse_long}')

        if not self.use_gen_fine:
            self.n_mov_av_fine = self.n_fine_long - self.n_fine
            # print(f'n_mov_av_fine = {self.n_mov_av_fine}')

        #print(self.n_fine, self.n_mov_av_coarse, self.rf.harmonic[0,0] self.n_mov_av_coarse/self.rf.harmonic[0,0])
        #quit()

        if self.open_FF == 1 and not self.use_gen_fine: # Not yet implemented for fine generator
            #print(f'fillpattern = {self.fillpattern}, shape = {self.fillpattern.shape}')
            # Instead of using [::5]
            #self.indices_coarseFF = np.arange(self.fillpattern[0], self.n_coarse, self.nbs_FF)
            #print(f'indices_coarseFF = {self.indices_coarseFF}, shape = {self.indices_coarseFF.shape}')
            #self.indices_coarseFF = np.arange(self.fillpattern[0], self.fillpattern[-1]+self.nbs_FF, self.nbs_FF) # = self.filledpattern
            #print(f'indices_coarseFF = {self.indices_coarseFF}, shape = {self.indices_coarseFF.shape}')
            self.indices_coarseFF = np.arange(self.fillpattern[0]%self.nbs_FF, self.n_coarse, self.nbs_FF)
            #print(f'indices_coarseFF = {self.indices_coarseFF}, shape = {self.indices_coarseFF.shape}')
            #self.indices_coarseFF = np.arange(self.fillpattern[0]%self.nbs_FF, self.fillpattern[-1]+self.nbs_FF, self.nbs_FF)
            #print(f'indices_coarseFF = {self.indices_coarseFF}, shape = {self.indices_coarseFF.shape}')
            #
            #self.n_coarseFF = int(self.n_coarse/self.nbs_FF)
            ##self.n_coarseFF = int(round(self.rf.t_rev[0]/self.rf.t_rf[0, 0]/self.nbs_FF))  ##### MYEDIT 2020.09.08
            self.n_coarseFF = len(self.indices_coarseFF)

        self.power_clamp_thresh = power_clamp
        self.I_gen_abs_max_coarse = False   # In physical current (per cav, the same for all) [A]
        self.Q_gen_abs_max_coarse = False   # In instantaneous coarse charge (total all cavs) [C]

        # Initialise turn-by-turn variables
        # MYEDIT: Also created and/or updated the time grid arrays. Added fillpattern
        self.update_variables()

        self.counter_pretrack = 0
        self.in_track_beam = False

        # if self.add_ff_to_gen:
        #     self.V_ind_beam_coarse_prev = np.zeros(self.n_coarse, dtype=complex)

        # Initialise bunch-by-bunch voltage array with LENGTH OF PROFILE
        if self.use_gen_fine:
            self.V_tot_fine = np.zeros(self.n_fine,   dtype=complex)
            self.logger.debug("Length of arrays on fine grid %d", self.n_fine)
        # Array to hold the bucket-by-bucket voltage with LENGTH OF LLRF
        self.V_tot_coarse   = np.zeros(self.n_coarse, dtype=complex)
        self.logger.debug("Length of arrays on coarse grid %d", self.n_coarse)
        # #
        # # self.V_tot_fine = polar_to_cartesian(
        # #     np.ones(self.n_fine)*self.V_part*self.rf.voltage[0, 0],
        # #     0.5*np.pi*self.Vset_imag -self.rf.phi_rf[0, 0])
        # self.V_tot_coarse = polar_to_cartesian(
        #     np.ones(self.n_coarse)*self.V_part*self.rf.voltage[0, 0],
        #     0.5*np.pi*self.Vset_imag -self.rf.phi_rf[0, 0])

        # Initialise comb filter
        self.a_comb = float(a_comb)
        if self.use_gen_fine:
            self.dV_comb_fine_prev = np.zeros(self.n_fine,   dtype=complex)
        self.dV_comb_coarse_prev   = np.zeros(self.n_coarse, dtype=complex)

        # Initialise cavity filter (moving average)
        if self.use_gen_fine:
            self.dV_mod_fine_prev = np.zeros(self.n_fine,   dtype=complex)

        if self.dV_Hcav_opt == '0':
            # OPT0
            self.dV_mod_coarse_prev   = np.zeros(self.n_coarse, dtype=complex)
        elif self.dV_Hcav_opt == '1':
            # OPT1 HTIMKO 2021-02: At this point the are the same in length (should it be long -check-, but we only need the last points)
            self.dV_Hcav_coarse_prev_long = np.zeros(self.n_coarse, dtype=complex)
        elif self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
            # OPT2 MY TEST:
            self.dV_mod_coarse_prev_long  = np.zeros(self.n_coarse, dtype=complex)

        # Initialise generator-induced voltage
        if self.use_gen_fine:
            self.Q_gen_fine_prev = np.zeros(self.n_mov_av_fine,   dtype=complex)
        self.Q_gen_coarse_prev   = np.zeros(self.n_mov_av_coarse, dtype=complex) # VS HTIMKO 2021-02: self.I_gen_prev = np.zeros(self.n_coarse, dtype=complex)

        self.logger.info("Class initialized")

        # Initialise feed-forward; sampled every nbs_FF = 5 buckets
        if self.open_FF == 1 and not self.use_gen_fine: # Not yet implemented for fine generator
            self.logger.debug("Feed-forward active")
            self.Q_beam_coarseFF_prev = np.zeros(self.n_coarseFF, dtype=complex)
            self.Q_beam_ff_coarseFF  = np.zeros(self.n_coarseFF, dtype=complex)
            # self.dV_ind_beam_ff_coarseFF = np.zeros(self.n_coarseFF, dtype=complex)
            #### self.dV_ind_beam_ff_coarseFF_del = np.zeros(self.n_coarseFF, dtype=complex) # Removed renaming of dV_ind_beam_ff_coarseFF after delay (kept the same name)

            self.Q_beam_coarseFF_prev_ext_testnbsFF_prev = np.zeros(self.n_coarseFF, dtype=complex)
            self.Q_beam_coarse_prev_ext_testnbsFF_prev   = np.zeros(self.n_coarse, dtype=complex)

    def beam_induced_voltage(self, lpf=False):
        """Calculates the beam-induced voltage

        Parameters
        ----------
        lpf : bool
            Apply low-pass filter for beam current calculation;
            default is False

        Attributes
        ----------
        Q_beam_coarse : complex array
            RF component of the beam charge [C] at the present time step,
            calculated in coarse grid
        Q_beam_fine : complex array
            RF component of the beam charge [C] at the present time step,
            calculated in fine grid
        V_ind_beam_coarse : complex array
            Induced voltage [V] from beam-cavity interaction on the
            coarse grid
        V_ind_beam_fine : complex array
            Induced voltage [V] from beam-cavity interaction on the fine
            grid
        """


        # Beam current from profile
        # MYEDIT: 2020.12.14: Time array corresponding to Q_beam_coarse will
        # be one that corresponds to profile.bin_centers, therefore make sure
        # that t_coarse has been defined with any possible reference offset
        # present in it (and thus in t_fine)

        if self.flag_macfull and self.counter >= 1: # 1 if the tracker is called after trackin cavityfeedback (which updates rf.counter 0 -> 1 in its first call)
            # Extended arrayel with the last bunches from previous turn to ensure
            # staying at steady-state at the beginning of the turn. The number
            # bunches needed depends on the TWC filling time (take 2x the number
            # to be safe):
            self.nbprev = int(2*np.ceil(self.TWC.tau/(self.rf.t_rf[0, 0]*self.nbs)))
            # print(f'self.nbprev = {self.nbprev}')
            profile_n_macroparticles_tmp = self.profile.n_macroparticles
            profile_bin_centers_tmp      = self.profile.bin_centers
            self.profile.n_macroparticles = np.concatenate( (self.profile.n_macroparticles[-int(self.nbprev*self.nbs*self.Ns):], self.profile.n_macroparticles) )
            self.profile.bin_centers      = np.concatenate( (self.profile.bin_centers[:int(self.nbprev*self.nbs*self.Ns)] - self.profile.bin_centers[int(self.nbprev*self.nbs*self.Ns)] + 0.5*self.profile.bin_size, self.profile.bin_centers) )
            # print(f'self.profile.n_macroparticles = {self.profile.n_macroparticles}, shape = {self.profile.n_macroparticles.shape}')
            # print(f'self.profile.n_macroparticles[int(self.nbprev*self.nbs*self.Ns)-1:int(self.nbprev*self.nbs*self.Ns)+1+1] = {self.profile.n_macroparticles[int(self.nbprev*self.nbs*self.Ns)-1:int(self.nbprev*self.nbs*self.Ns)+1+1]}')
            # print(f'self.profile.bin_centers = {self.profile.bin_centers}, shape = {self.profile.bin_centers.shape}')
            # print(f'self.profile.bin_centers[int(self.nbprev*self.nbs*self.Ns)-1:int(self.nbprev*self.nbs*self.Ns)+1+1] = {self.profile.bin_centers[int(self.nbprev*self.nbs*self.Ns)-1:int(self.nbprev*self.nbs*self.Ns)+1+1]}')
            n_coarse_ext   = np.copy(self.n_coarse) + int(self.nbprev*self.nbs)
            # print(f'n_coarse_ext = {n_coarse_ext}')

        else:

            n_coarse_ext   = np.copy(self.n_coarse)

        self.Q_beam_fine, self.Q_beam_coarse = \
            rf_beam_current(self.profile,
                            self.omega_c,self.rf.t_rev[self.counter],
                            lpf=lpf,
                            downsample={'Ts':     self.T_s_coarse,
                                        'points': n_coarse_ext})
        self.Q_beam_fine   += 0.+0j
        self.Q_beam_coarse += 0.+0j
        if not self.Vset_imag:
            # Rotate by -90 deg since Vset is at +I
            self.Q_beam_fine   *= np.exp(-0.5j*np.pi)
            self.Q_beam_coarse *= np.exp(-0.5j * np.pi)
        if self.redef_Ib:
            self.Q_beam_fine   *= -1
            self.Q_beam_coarse *= -1


        # Beam-induced voltage

        # if self.add_ff_to_gen:
        #     self.V_ind_beam_coarse_prev = np.copy(self.V_ind_beam_coarse)
        #     self.V_ind_beam_fine_prev   = np.copy(self.V_ind_beam_fine)
        #     self.V_ind_beam_noff_coarse_prev = np.copy(self.V_ind_beam_coarse)
        #     self.V_ind_beam_noff_fine_prev   = np.copy(self.V_ind_beam_fine)

        self.induced_voltage('beam_fine')
        self.induced_voltage('beam_coarse')
        if self.redef_Vb:
            # self.V_ind_beam_fine   *= -1
            # self.V_ind_beam_coarse *= -1
            self.V_ind_beam_fine   = np.conjugate(self.V_ind_beam_fine)
            self.V_ind_beam_coarse = np.conjugate(self.V_ind_beam_coarse)


        if self.flag_macfull and self.counter >= 1:
            # Restore profile bin_centers and n_macroparticles to the right one-turn window
            self.profile.n_macroparticles = np.copy(profile_n_macroparticles_tmp)
            self.profile.bin_centers      = np.copy(profile_bin_centers_tmp)
            # Extract only the Q_beam data corresponding to the present turn:
            self.Q_beam_coarse = self.Q_beam_coarse[int(self.nbprev*self.nbs):]
            self.Q_beam_fine   = self.Q_beam_fine[int(self.nbprev*self.nbs*self.Ns):]
            # # Extract only the V_ind_beam data corresponding to the present turn:
            # print(f'self.V_ind_beam_coarse = {self.V_ind_beam_coarse}, shape = {self.V_ind_beam_coarse.shape}')
            # print(f'self.V_ind_beam_fine = {self.V_ind_beam_fine}, shape = {self.V_ind_beam_fine.shape}')
            self.V_ind_beam_coarse = self.V_ind_beam_coarse[int(self.nbprev*self.nbs):]
            self.V_ind_beam_fine   = self.V_ind_beam_fine[int(self.nbprev*self.nbs*self.Ns):]
            # print(f'self.V_ind_beam_coarse = {self.V_ind_beam_coarse}, shape = {self.V_ind_beam_coarse.shape}')
            # print(f'self.V_ind_beam_fine = {self.V_ind_beam_fine}, shape = {self.V_ind_beam_fine.shape}')

        # Save the beam-induced voltage (before FF correction, in case it is active; for referene)
        self.V_ind_beam_noff_coarse = np.copy(self.V_ind_beam_coarse)
        self.V_ind_beam_noff_fine   = np.copy(self.V_ind_beam_fine)

        if self.open_FF == 1 and not self.use_gen_fine: # Not yet implemented for fine generator

            # print(f'{self.cavtype}')
            # print(f'self.fillpattern = {self.fillpattern}, shape = {self.fillpattern.shape}')
            # print(f'self.Q_beam_coarse = {self.Q_beam_coarse}, shape = {self.Q_beam_coarse.shape}, max(abs) = {np.max(np.abs(self.Q_beam_coarse))}')
            # for i in range(len(self.Q_beam_coarse)):
            #     if (i >= self.fillpattern[0]-5 and  i <= self.fillpattern[0]+5) or (i >= self.fillpattern[-1]-5 and i <= self.fillpattern[-1]+5):
            #         print(i, self.Q_beam_coarse[i])
            # pos_peaks_fine   = np.where(np.abs(self.Q_beam_fine)   > 0.50*np.max(np.abs(self.Q_beam_fine)))[0] # 0.50 generous, ~0.90 more reasonable?
            pos_peaks_coarse = np.where(np.abs(self.Q_beam_coarse) > 0.50*np.max(np.abs(self.Q_beam_coarse)))[0] # 0.50 generous, ~0.90 more reasonable?
            # Simplified, just see which samples are not zero. It will work even when the bunches (and their peaks) are different,
            # which can potentially set an unrealistic threshold (max) for the peaks for smaller bunches.
            # BUT when particles spill around the bucket, it yield error. Maybe leave the option above (slower?)
            # pos_peaks_coarse = np.nonzero(self.Q_beam_coarse)[0]
            # Ns = int(self.n_fine/self.n_coarse)
            # print(f'pos_peaks_fine = {pos_peaks_fine} -> /Ns = {pos_peaks_fine/Ns}')
            # print(f'pos_peaks_coarse = {pos_peaks_coarse}')
            if pos_peaks_coarse[0] != self.fillpattern[0]:
                print('Q_beam_coarse =')
                for ii in range(len(self.Q_beam_coarse)):
                    print(ii, self.Q_beam_coarse[ii], np.abs(self.Q_beam_coarse[ii]))
                print('pos_peaks_coarse =', end= ' ')
                for ii in pos_peaks_coarse:
                    print(ii, end=', ')
                print('')
                sys.exit(f'\n[!] ERROR: Coarse downsampling of beam current ({pos_peaks_coarse}) does not match the fill pattern ({self.fillpattern})!\n')
            #
            # print(f'self.indices_coarseFF = {self.indices_coarseFF}, shape = {self.indices_coarseFF.shape}')
            # print(f'self.Q_beam_coarse[self.indices_coarseFF] = {self.Q_beam_coarse[self.indices_coarseFF]}, shape = {self.Q_beam_coarse[self.indices_coarseFF].shape}, max(abs) = {np.max(np.abs(self.Q_beam_coarse[self.indices_coarseFF]))}')
            # print(f'self.Q_beam_coarseFF_prev =  = {self.Q_beam_coarseFF_prev}, shape = {self.Q_beam_coarseFF_prev.shape}, max(abs) = {np.max(np.abs(self.Q_beam_coarseFF_prev))}')
            # quit()

            # Calculate correction based on previous turn on coarse grid.
            # At first turn, Q_beam_coarseFF_prev is zero
            # print(f'Q_beam_coarseFF_prev = {self.Q_beam_coarseFF_prev}, shape = {self.Q_beam_coarseFF_prev.shape}')
            # print(f'Q_beam_coarse = {self.Q_beam_coarse}, shape = {self.Q_beam_coarse.shape}')
            # print(f'n_FF_delay = {self.n_FF_delay}')
            # print(f'self.nbs_FF = {self.nbs_FF}')
            # print(f'int(self.n_FF_delay*self.nbs_FF) = {int(self.n_FF_delay*self.nbs_FF)}')
            # print(f'Q_beam_coarse[:int(self.n_FF_delay*self.nbs_FF):self.nbs_FF] = {self.Q_beam_coarse[:int(self.n_FF_delay*self.nbs_FF):self.nbs_FF]}, shape = {self.Q_beam_coarse[:int(self.n_FF_delay*self.nbs_FF):self.nbs_FF].shape}')

            self.delay_FF_opt = 1

            if self.delay_FF_opt == 0:

                for ind in range(self.n_coarseFF): # + self.n_FF_delay):
                    self.Q_beam_ff_coarseFF[ind]      = self.coeff_FF[0] * self.Q_beam_coarseFF_prev[ind]
                    for k in range(self.n_FF):
                        self.Q_beam_ff_coarseFF[ind] += self.coeff_FF[k] * self.Q_beam_coarseFF_prev[ind-k]
                self.dV_ind_beam_ff_coarseFF = self.G_ff * self.n_cavities * self.matr_conv(self.Q_beam_ff_coarseFF, self.TWC.h_gen_coarse[self.indices_coarseFF])
                print(f'dV_ind_beam_ff_coarseFF = {self.dV_ind_beam_ff_coarseFF}, shape = {self.dV_ind_beam_ff_coarseFF.shape}')

                dV_ind_beam_ff_coarseFF_tmp = np.copy(self.dV_ind_beam_ff_coarseFF)

                # Delay:
                self.dV_ind_beam_ff_coarseFF = np.concatenate( (self.dV_ind_beam_ff_coarseFF[self.n_FF_delay:], np.zeros(self.n_FF_delay)) )

                if True and self.cavtype == '4x3sec': # and self.counter == 1:
                    fig, ax = plt.subplots(2,1)
                    ax[0].plot(np.abs(self.Q_beam_coarseFF_prev)/np.max(np.abs(self.Q_beam_coarseFF_prev))*np.max(np.abs(self.Q_beam_ff_coarseFF)), '+') # scaled to elf.Q_beam_ff_coarseFF for plotting
                    ax[0].plot(np.abs(self.Q_beam_ff_coarseFF), 'x')
                    ax[1].plot(np.abs(dV_ind_beam_ff_coarseFF_tmp), '+')
                    ax[1].plot(np.abs(self.dV_ind_beam_ff_coarseFF), 'x')
                    ax[0].set_ylabel('Q_beam_ff_coarseFF')
                    ax[1].set_ylabel('dV_ind_beam_ff_coarseFF(_tmp)')
                    ax[0].set_title(f'{self.cavtype}, turn {self.counter}')
                    plt.show()
                    plt.cla()
                    plt.clf()

            elif self.delay_FF_opt == 1:

                # print('')
                ### n_FF_extra = int(0.5*(self.n_FF - 1))
                ### n_FF_extra = self.n_FF_delay
                n_FF_extra = self.n_coarseFF #160 # 2*(self.n_FF_delay)

                # print(f'Q_beam_coarseFF_prev = {self.Q_beam_coarseFF_prev}, shape = {self.Q_beam_coarseFF_prev.shape}')
                # print(f'Q_beam_ff_coarseFF = {self.Q_beam_ff_coarseFF}, shape = {self.Q_beam_ff_coarseFF.shape}')
                # Q_beam_coarseFF_prev_ext = np.exp(-0.5j*np.pi.)*np.concatenate( (self.Q_beam_coarseFF_prev, self.Q_beam_coarse[:int(n_FF_extra*self.nbs_FF):self.nbs_FF]) ) / self.nbs_FF # Note that Qb might have already been multiplied by -1 if redef_Ib
                Q_beam_coarseFF_prev_ext = np.concatenate( (self.Q_beam_coarseFF_prev, self.Q_beam_coarse[:int(n_FF_extra*self.nbs_FF):self.nbs_FF]) ) # DO NOT DIVIDE BY self.nbs_FF
                # print(f'Q_beam_coarseFF_prev_ext = {Q_beam_coarseFF_prev_ext}, shape = {Q_beam_coarseFF_prev_ext.shape}')
                # testnbsFF = True
                testnbsFF = False
                if testnbsFF:
                    Q_beam_coarseFF_prev_ext_testnbsFF_wo = np.concatenate( (self.Q_beam_coarseFF_prev_ext_testnbsFF_prev, self.Q_beam_coarse[:int(n_FF_extra*self.nbs_FF):self.nbs_FF]) )
                    Q_beam_coarseFF_prev_ext_testnbsFF_w  = np.concatenate( (self.Q_beam_coarseFF_prev_ext_testnbsFF_prev, self.Q_beam_coarse[:int(n_FF_extra*self.nbs_FF):self.nbs_FF]) ) / self.nbs_FF
                    Q_beam_coarse_prev_ext_testnbsFF      = np.concatenate( (self.Q_beam_coarse_prev_ext_testnbsFF_prev,   self.Q_beam_coarse) )

                self.Q_beam_ff_coarseFF = np.zeros(self.n_coarseFF + n_FF_extra, dtype=complex)
                for ind in range(self.n_coarseFF + n_FF_extra):
                    self.Q_beam_ff_coarseFF[ind]      = self.coeff_FF[0] * Q_beam_coarseFF_prev_ext[ind]
                    for k in range(self.n_FF):
                        self.Q_beam_ff_coarseFF[ind] += self.coeff_FF[k] * Q_beam_coarseFF_prev_ext[ind-k]

                h_gen_coarseFF_ext = self.TWC.h_gen_coarse[self.indices_coarseFF]
                ### h_gen_coarseFF_ext = np.concatenate( (self.TWC.h_gen_coarse[self.indices_coarseFF], self.TWC.h_gen_coarse[self.indices_coarseFF[:n_FF_extra]]) )
                self.dV_ind_beam_ff_coarseFF = self.G_ff * self.n_cavities * self.matr_conv(self.Q_beam_ff_coarseFF, h_gen_coarseFF_ext)
                # print(f'dV_ind_beam_ff_coarseFF = {self.dV_ind_beam_ff_coarseFF}, shape = {self.dV_ind_beam_ff_coarseFF.shape}')
                if testnbsFF:
                    self.dV_ind_beam_ff_coarseFF_testnbsFF_wo = self.n_cavities * self.matr_conv(Q_beam_coarseFF_prev_ext_testnbsFF_wo, h_gen_coarseFF_ext)
                    self.dV_ind_beam_ff_coarseFF_testnbsFF_w  = self.n_cavities * self.matr_conv(Q_beam_coarseFF_prev_ext_testnbsFF_w,  h_gen_coarseFF_ext)
                    self.dV_ind_beam_ff_coarse_testnbsFF      = self.n_cavities * self.matr_conv(Q_beam_coarse_prev_ext_testnbsFF,   self.TWC.h_gen_coarse)
                    self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam = self.n_cavities * self.matr_conv(Q_beam_coarseFF_prev_ext_testnbsFF_wo, self.TWC.h_beam_coarse[self.indices_coarseFF])
                    self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam  = self.n_cavities * self.matr_conv(Q_beam_coarseFF_prev_ext_testnbsFF_w,  self.TWC.h_beam_coarse[self.indices_coarseFF])
                    self.dV_ind_beam_ff_coarse_testnbsFF_hbeam      = self.n_cavities * self.matr_conv(Q_beam_coarse_prev_ext_testnbsFF, self.TWC.h_beam_coarse)

                if testnbsFF:
                    fig, ax = plt.subplots(4, 2, sharex=True, sharey='row')
                    fig.set_size_inches(2 * 8.00, 4 * 2.00)

                    # Vff
                    # Qb (coarseFF, NO div. by nbsFF) convolved with hg (coarseFF) - no FIR
                    ax[0,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo))*5,  np.real(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo), '.-', markersize=1.0, lw=1.0)
                    ax[1,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo))*5,  np.imag(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo), '.-', markersize=1.0, lw=1.0)
                    ax[2,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo))*5,   np.abs(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo), '.-', markersize=1.0, lw=1.0)
                    ax[3,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo))*5, np.angle(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo), '.-', markersize=1.0, lw=1.0, label='coarseFF, NO nbsFF, no FIR')
                    # Qb (coarseFF, div. by nbsFF) convolved with hg (coarseFF) - no FIR
                    ax[0,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w))*5,  np.real(self.dV_ind_beam_ff_coarseFF_testnbsFF_w), '.-', markersize=1.0, lw=1.0)
                    ax[1,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w))*5,  np.imag(self.dV_ind_beam_ff_coarseFF_testnbsFF_w), '.-', markersize=1.0, lw=1.0)
                    ax[2,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w))*5,   np.abs(self.dV_ind_beam_ff_coarseFF_testnbsFF_w), '.-', markersize=1.0, lw=1.0)
                    ax[3,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w))*5, np.angle(self.dV_ind_beam_ff_coarseFF_testnbsFF_w), '.-', markersize=1.0, lw=1.0, label='coarseFF, w/ nbsFF, no FIR')
                    # Qb (coarse) convolved with hg (coarse) - no FIR
                    ax[0,0].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF)),  np.real(self.dV_ind_beam_ff_coarse_testnbsFF), '.-', markersize=1.0, lw=1.0)
                    ax[1,0].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF)),  np.imag(self.dV_ind_beam_ff_coarse_testnbsFF), '.-', markersize=1.0, lw=1.0)
                    ax[2,0].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF)),   np.abs(self.dV_ind_beam_ff_coarse_testnbsFF), '.-', markersize=1.0, lw=1.0)
                    ax[3,0].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF)), np.angle(self.dV_ind_beam_ff_coarse_testnbsFF), '.-', markersize=1.0, lw=1.0, label='coarse, no FIR')
                    # FIR: As it is (check if it was div. by nbsFF or not)
                    ax[0,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF))*5,  np.real(self.dV_ind_beam_ff_coarseFF), '.-', markersize=1.0, lw=1.0)
                    ax[1,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF))*5,  np.imag(self.dV_ind_beam_ff_coarseFF), '.-', markersize=1.0, lw=1.0)
                    ax[2,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF))*5,   np.abs(self.dV_ind_beam_ff_coarseFF), '.-', markersize=1.0, lw=1.0)
                    ax[3,0].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF))*5, np.angle(self.dV_ind_beam_ff_coarseFF), '.-', markersize=1.0, lw=1.0, label='Vff (coarseFF, (w/w/o?) nbsFF, FIR')
                    # Vb
                    # Qb (coarseFF, NO div. by nbsFF) convolved with hg (coarseFF) - no FIR
                    ax[0,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam))*5,  np.real(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[1,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam))*5,  np.imag(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[2,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam))*5,   np.abs(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[3,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam))*5, np.angle(self.dV_ind_beam_ff_coarseFF_testnbsFF_wo_hbeam), '.-', markersize=1.0, lw=1.0, label='coarseFF, NO nbsFF')
                    # Qb (coarseFF, div. by nbsFF) convolved with hg (coarseFF) - no FIR
                    ax[0,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam))*5,  np.real(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[1,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam))*5,  np.imag(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[2,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam))*5,   np.abs(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[3,1].plot( np.arange(len(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam))*5, np.angle(self.dV_ind_beam_ff_coarseFF_testnbsFF_w_hbeam), '.-', markersize=1.0, lw=1.0, label='coarseFF, w/ nbsFF')
                    # Qb (coarse) convolved with hg (coarse)
                    ax[0,1].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam)),  np.real(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[1,1].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam)),  np.imag(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[2,1].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam)),   np.abs(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam), '.-', markersize=1.0, lw=1.0)
                    ax[3,1].plot( np.arange(len(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam)), np.angle(self.dV_ind_beam_ff_coarse_testnbsFF_hbeam), '.-', markersize=1.0, lw=1.0, label='coarse')
                    # Vb computed before
                    ax[0,1].plot( np.arange(len(self.V_ind_beam_coarse)),  np.real(self.V_ind_beam_coarse), '.-', markersize=1.0, lw=1.0)
                    ax[1,1].plot( np.arange(len(self.V_ind_beam_coarse)),  np.imag(self.V_ind_beam_coarse), '.-', markersize=1.0, lw=1.0)
                    ax[2,1].plot( np.arange(len(self.V_ind_beam_coarse)),   np.abs(self.V_ind_beam_coarse), '.-', markersize=1.0, lw=1.0)
                    ax[3,1].plot( np.arange(len(self.V_ind_beam_coarse)), np.angle(self.V_ind_beam_coarse), '.-', markersize=1.0, lw=1.0, label='Vb')

                    for jsp in range(2):
                        for isp in range(4):
                            ax[isp,jsp].grid()
                            ax[isp,jsp].axvline(self.n_coarse, ls=':', color='gray')
                        for isp in range(2):
                            ax[isp,jsp].set_ylim( -ax[2,jsp].set_ylim()[1],  ax[2,jsp].set_ylim()[1])
                        ax[3,jsp].legend(loc=1)
                    ax[0,0].set_ylabel('Re')
                    ax[1,0].set_ylabel('Im')
                    ax[2,0].set_ylabel('Abs')
                    ax[3,0].set_ylabel('Ang')
                    ax[0,0].set_title('Vff')
                    ax[0,1].set_title('Vb')
                    fig.suptitle(f'{self.cavtype}, turn {self.counter}')
                    fig.tight_layout()
                    plt.show()
                    #
                    tocontinue = input('\nContinue? ')
                    if tocontinue in ['y', 'Y', '1', 'yes', 'Yes', 't', 'true', 'True']:
                        pass
                    else:
                        sys.exit('\nBYE!\n')

                if self.redef_Vbff:
                    self.dV_ind_beam_ff_coarseFF *= -1
                    # self.dV_ind_beam_ff_coarseFF = np.conjugate(self.dV_ind_beam_ff_coarseFF)

                dV_ind_beam_ff_coarseFF_tmp = np.copy(self.dV_ind_beam_ff_coarseFF) # just for plot, before applying delay

                # Delay: take the signal from n_FF_delay and until the end (note that, formally, n_FF_delay points were then computed from the beam charge at the current turn)
                self.dV_ind_beam_ff_coarseFF = self.dV_ind_beam_ff_coarseFF[self.n_FF_delay:self.n_FF_delay+self.n_coarseFF]
                # print(f'dV_ind_beam_ff_coarseFF = {self.dV_ind_beam_ff_coarseFF}, shape = {self.dV_ind_beam_ff_coarseFF.shape}')

                if self.counter == 0:
                    # In turn zero we must remove an unphysical correction at the beginning of the signal (it's using information of more than a turn ahead):
                    # print(f'np.abs(dV_ind_beam_ff_coarseFF[:self.n_FF_delay+1]) = {np.abs(self.dV_ind_beam_ff_coarseFF[:self.n_FF_delay+1])}, shape = {self.dV_ind_beam_ff_coarseFF[:self.n_FF_delay+1].shape}')
                    self.dV_ind_beam_ff_coarseFF[:self.n_FF_delay+1] = self.dV_ind_beam_ff_coarseFF[self.n_FF_delay+1:self.n_FF_delay+1+1] # Basically zeros...
                    ###self.dV_ind_beam_ff_coarseFF[:self.n_FF_delay+1] = np.zeros(self.n_FF_delay+1)

                # if True and self.cavtype == '4x3sec': # and self.counter == 1:
                if False: #and self.cavtype == '2x4sec': # and self.counter == 1:
                    fig, ax = plt.subplots(2,1)
                    ax[0].plot(np.abs(self.Q_beam_coarseFF_prev)/np.max(np.abs(self.Q_beam_coarseFF_prev))*np.max(np.abs(self.Q_beam_ff_coarseFF)), '+')
                    ax[0].plot(np.abs(Q_beam_coarseFF_prev_ext) /np.max(np.abs(Q_beam_coarseFF_prev_ext)) *np.max(np.abs(self.Q_beam_ff_coarseFF)), '.')
                    ax[0].plot(np.abs(self.Q_beam_ff_coarseFF), 'x')
                    ax[1].plot(np.abs(dV_ind_beam_ff_coarseFF_tmp), '+')
                    ax[1].plot(np.abs(self.dV_ind_beam_ff_coarseFF), 'x')
                    ax[0].set_ylabel('Q_beam_ff_coarseFF')
                    ax[1].set_ylabel('dV_ind_beam_ff_coarseFF(_tmp)')
                    ax[0].set_title(f'{self.cavtype}, turn {self.counter}')
                    plt.show()
                    plt.cla()
                    plt.clf()

                # Remove the extended part (in case of plotting this parameter)
                self.Q_beam_ff_coarseFF = self.Q_beam_ff_coarseFF[:-n_FF_extra]
                # print(f'Q_beam_ff_coarseFF = {self.Q_beam_ff_coarseFF}, shape = {self.Q_beam_ff_coarseFF.shape}')
                # print('')

            elif self.delay_FF_opt == 2:

                for ind in range(self.n_coarseFF):
                    self.Q_beam_ff_coarseFF[ind]      = self.coeff_FF[0] * self.Q_beam_coarseFF_prev[ind]
                    # if self.cavtype == '4x3sec' and (ind == 0 or ind == self.n_coarseFF-1):
                    #     print(ind, self.coeff_FF[0], 'x', np.abs(self.Q_beam_coarseFF_prev[ind]), '->', np.abs(self.Q_beam_ff_coarseFF[ind]))
                    for k in range(self.n_FF):
                        self.Q_beam_ff_coarseFF[ind] += self.coeff_FF[k] * self.Q_beam_coarseFF_prev[ind-k]
                        # if self.cavtype == '4x3sec' and (ind == 0 or ind == self.n_coarseFF-1):
                        #     if k < 3 or k >= self.n_FF-3: print('  +', k, self.coeff_FF[k], 'x', np.abs(self.Q_beam_coarseFF_prev[ind-k]), '->', np.abs(self.Q_beam_ff_coarseFF[ind]))
                        #     elif k == 3:                  print('...')
                self.dV_ind_beam_ff_coarseFF = self.G_ff * self.n_cavities * self.matr_conv(self.Q_beam_ff_coarseFF, self.TWC.h_gen_coarse[self.indices_coarseFF]) # h_gen[::5]

                dV_ind_beam_ff_coarseFF_tmp = np.copy(self.dV_ind_beam_ff_coarseFF)

                Q_beam_coarse_pres      = np.copy(self.Q_beam_coarse[self.indices_coarseFF])
                Q_beam_ff_coarseFF_pres = np.zeros(len(self.Q_beam_ff_coarseFF), dtype=complex)
                for ind in range(self.n_coarseFF):
                    Q_beam_ff_coarseFF_pres[ind]      = self.coeff_FF[0] * Q_beam_coarse_pres[ind]
                    # if self.cavtype == '4x3sec' and (ind == 0 or ind == self.n_coarseFF-1):
                    #     print(ind, self.coeff_FF[0], 'x', np.abs(Q_beam_coarse_pres[ind]), '->', np.abs(Q_beam_ff_coarseFF_pres[ind]))
                    for k in range(self.n_FF):
                        Q_beam_ff_coarseFF_pres[ind] += self.coeff_FF[k] * Q_beam_coarse_pres[ind-k]
                        # if self.cavtype == '4x3sec' and (ind == 0 or ind == self.n_coarseFF-1):
                        #     if k < 3 or k >= self.n_FF-3: print('  +', k, self.coeff_FF[k], 'x', np.abs(Q_beam_coarse_pres[ind-k]), '->', np.abs(Q_beam_ff_coarseFF_pres[ind]))
                        #     elif k == 3:                  print('...')

                dV_ind_beam_ff_coarseFF_pres = self.G_ff * self.n_cavities * self.matr_conv(Q_beam_ff_coarseFF_pres, self.TWC.h_gen_coarse[self.indices_coarseFF]) # h_gen[::5]

                # Delay:
                self.dV_ind_beam_ff_coarseFF = np.concatenate( (dV_ind_beam_ff_coarseFF_tmp[self.n_FF_delay:], dV_ind_beam_ff_coarseFF_pres[:self.n_FF_delay]) )

                if self.cavtype == '4x3sec': # self.counter == 2 and
                    fig, ax = plt.subplots(3,1)
                    fig.set_size_inches(1*12.00, 3*2.00)
                    ax[0].plot(np.abs(self.Q_beam_coarseFF_prev), '-', label='prev')
                    ax[0].plot(np.abs(Q_beam_coarse_pres), ':', label='pres')
                    ax[1].plot(np.abs(self.Q_beam_ff_coarseFF), '-', label='From prev')
                    ax[1].plot(np.abs(Q_beam_ff_coarseFF_pres), ':', label='From pres')
                    ax[2].plot(np.abs(dV_ind_beam_ff_coarseFF_tmp), '-', label='Pres correc')
                    ax[2].plot(np.abs(dV_ind_beam_ff_coarseFF_pres), ':', label='Futr correc')
                    ax[2].plot(np.abs(self.dV_ind_beam_ff_coarseFF), '-.', label='Pres + Futr')
                    ax[0].set_ylabel('Q_beam_coarseFF(_pres)')
                    ax[1].set_ylabel('Q_beam_ff_coarseFF(_pres)')
                    ax[2].set_ylabel('dV_ind_beam_ff_coarseFF(_pres)')
                    ax[0].legend(loc=9)
                    ax[1].legend(loc=9)
                    ax[2].legend(loc=9)
                    ax[0].set_title(f'{self.cavtype}, turn {self.counter}')
                    fig.savefig(f'{self.outdir}/plot_test_ff_{self.cavtype}_{self.counter}.png')
                    #plt.cla()
                    plt.close() #fig)

            # print(f'self.Q_beam_ff_coarseFF =  = {self.Q_beam_ff_coarseFF}, shape = {self.Q_beam_ff_coarseFF.shape}, max(abs) = {np.max(np.abs(self.Q_beam_ff_coarseFF))}')

            # Compensate for FIR filter delay

            # MYEDIT: 2020.09.04

            ##################################################################

            #   beam  | no beam
            # ABCDEFGH|ijklmnopq
            # del = 3

            # ABCDEFGH|ijklmn---  Opt 5 (basically the same than Opt 1...)
            # ABCDEFGH|ijklmnopq  Opt 1 i.e. no delay
            # ---DEFGH|ijklmnopq  Opt 3

            # DEFGHijk|lmnopqABC  Opt 2
            # DEFGHijk|lmnopq---  Opt 0 # DEFAULT (Correct?)

            # ---ABCDE|FGHijklmn  Opt 4

            ###

            # if(self.delay_FF_opt == 0):
            #     self.dV_ind_beam_ff_coarseFF = np.concatenate((self.dV_ind_beam_ff_coarseFF[self.n_FF_delay:],            np.zeros(self.n_FF_delay, dtype=np.complex)))

            # # elif(self.delay_FF_opt == 1):
            # #     self.dV_ind_beam_ff_coarseFF = np.concatenate((self.dV_ind_beam_ff_coarseFF[:self.n_FF_delay],            self.dV_ind_beam_ff_coarseFF[self.n_FF_delay:])) # = np.copy(self.dV_ind_beam_ff_coarseFF) = no delay

            # elif(self.delay_FF_opt == 2):
            #     # Seems to be the correct now, to make it circular
            #     self.dV_ind_beam_ff_coarseFF = np.concatenate((self.dV_ind_beam_ff_coarseFF[self.n_FF_delay:],            self.dV_ind_beam_ff_coarseFF[:self.n_FF_delay]))

            # # elif(self.delay_FF_opt == 3):
            # #     self.dV_ind_beam_ff_coarseFF = np.concatenate((np.zeros(self.n_FF_delay, dtype=np.complex), self.dV_ind_beam_ff_coarseFF[self.n_FF_delay:]))

            # # elif(self.delay_FF_opt == 4):
            # #     self.dV_ind_beam_ff_coarseFF = np.concatenate((np.zeros(self.n_FF_delay, dtype=np.complex), self.dV_ind_beam_ff_coarseFF[:-self.n_FF_delay]))

            # # elif(self.delay_FF_opt == 5):
            # #     self.dV_ind_beam_ff_coarseFF = np.concatenate((self.dV_ind_beam_ff_coarseFF[:-self.n_FF_delay],           np.zeros(self.n_FF_delay, dtype=np.complex)))

            # End of MYEDIT: 2020.09.04

            ##################################################################

            # Interpolate to finer grids
            # MYEDIT: 2020.12.14: Important that t_fine and t_coarse have the
            # same reference:
            self.dV_ind_beam_ff_coarse = np.interp(self.t_coarse,                          # self.rf_centers,
                                          self.t_coarse[self.indices_coarseFF],  # self.rf_centers[::5],
                                          self.dV_ind_beam_ff_coarseFF)
            self.dV_ind_beam_ff_fine = np.interp(self.t_fine,                             # self.profile.bin_centers,
                                        self.t_coarse[self.indices_coarseFF],    # self.rf_centers[::5],
                                        self.dV_ind_beam_ff_coarseFF)

            # Add to beam-induced voltage (opposite sign)
            #self.V_ind_beam_coarse_noff = np.copy(self.V_ind_beam_coarse)
            #self.V_ind_beam_fine_noff   = np.copy(self.V_ind_beam_fine)
            # self.V_ind_beam_coarse += -self.Vindbeamsign * self.n_cavities*np.copy(self.dV_ind_beam_ff_coarse)  # Original: positive, changed to negative for testing
            # self.V_ind_beam_fine   += -self.Vindbeamsign * self.n_cavities*np.copy(self.dV_ind_beam_ff_fine)    # Original: positive, changed to negative for testing
            #
            self.V_ind_beam_coarse -= np.copy(self.dV_ind_beam_ff_coarse)
            self.V_ind_beam_fine   -= np.copy(self.dV_ind_beam_ff_fine)
            #
            # self.V_ind_beam_coarse += np.copy(self.dV_ind_beam_ff_coarse)
            # self.V_ind_beam_fine   += np.copy(self.dV_ind_beam_ff_fine)
            #
            # self.V_ind_beam_coarse -= np.exp(0.5j*np.pi.)*np.copy(self.dV_ind_beam_ff_coarse)  # Changed adopted: using the negative now since we use the positive in Vindbeam now
            # self.V_ind_beam_fine   -= np.exp(0.5j*np.pi.)*np.copy(self.dV_ind_beam_ff_fine)    # Changed adopted: using the negative now since we use the positive in Vindbeam now
            #
            #
            # Has to be positive if using redef_Hb, redef_Ib, redef_Vb, redef_Vbff
            # self.V_ind_beam_coarse += np.copy(self.dV_ind_beam_ff_coarse)  # Changed adopted: using the negative now since we use the positive in Vindbeam now
            # self.V_ind_beam_fine   += np.copy(self.dV_ind_beam_ff_fine)    # Changed adopted: using the negative now since we use the positive in Vindbeam now

            # Update vector from previous turn
            self.Q_beam_coarseFF_prev = np.copy(self.Q_beam_coarse[self.indices_coarseFF]) # Q_beam_coarse[::5]
            if testnbsFF:
                self.Q_beam_coarseFF_prev_ext_testnbsFF_prev = np.copy(self.Q_beam_coarse[self.indices_coarseFF])
                self.Q_beam_coarse_prev_ext_testnbsFF_prev   = np.copy(self.Q_beam_coarse)

    def call_conv(self, signal, kernel):
        """Routine to call optimised C++ convolution"""

        # Make sure that the buffers are stored contiguously
        signal = np.ascontiguousarray(signal)
        kernel = np.ascontiguousarray(kernel)

        result = np.zeros(len(kernel) + len(signal) - 1)
        bm.convolve(signal, kernel, result)

        return result

    def generator_induced_voltage(self):
        r"""Calculates the generator-induced voltage. The transmitter model is
        a simple linear gain [C/V] converting voltage to charge.

        .. math:: I = G_{\mathsf{tx}}\,\frac{V}{R_{\mathsf{gen}}} \, ,

        where :math:`R_{\mathsf{gen}}` is the generator resistance,
        :py:attr:`llrf.impulse_response.TravellingWaveCavity.R_gen`

        Attributesg
        ----------
        Q_gen_coarse : complex array
            RF component of the generator charge [C] at the present time step
        V_ind_gen_coarse : complex array
            Induced voltage [V] from generator-cavity interaction

        """

        # Add correction to the drive already existing: V_gen_coarse is the full generator voltage (all cavities)
        # print(f'phi_0 = {self.phi_0} rad -> {self.phi_0/(2*np.pi)} Sc')
        # print(f'tau/Ts*2pi = {self.TWC.tau/self.T_s_coarse*(2.*np.pi)} rad ->  {self.TWC.tau/self.T_s_coarse} Sc')
        # print(f'phi_0 + tau/Ts*2pi = {self.phi_0 + self.TWC.tau/self.T_s_coarse*(2.*np.pi)} rad -> {self.phi_0/(2.*np.pi) + self.TWC.tau/self.T_s_coarse} Sc')
        # print(f'phi_0 - tau/Ts*2pi = {self.phi_0 - self.TWC.tau/self.T_s_coarse*(2.*np.pi)} rad -> {self.phi_0/(2.*np.pi) - self.TWC.tau/self.T_s_coarse} Sc')
        if self.use_gen_fine:
            self.dV_gen_fine = self.open_FB * modulator(self.dV_Hcav_fine,   self.omega_r, self.omega_c, self.T_s_fine  )
        self.dV_gen_coarse = self.open_FB * modulator(self.dV_Hcav_coarse, self.omega_r, self.omega_c, self.T_s_coarse, phi_0=-self.phi_0)
       #self.dV_gen_coarse = self.open_FB * modulator(self.dV_Hcav_coarse, self.omega_r, self.omega_c, self.T_s_coarse, phi_0=-self.phi_0+self.n_mov_av_coarse*(2.*np.pi)) #+self.TWC.tau/self.T_s_coarse*(2.*np.pi)))

        if self.use_gen_fine:
            self.V_gen_fine = self.open_drive * np.copy(self.V_set_fine)   + self.dV_gen_fine
        self.V_gen_coarse   = self.open_drive * np.copy(self.V_set_coarse) + self.dV_gen_coarse
        #print(f'mean(|V_gen_coarse|) = {np.mean(np.absolute(self.V_gen_coarse))/1e6:6.2f} MV') # MYEDIT 2020.10.02

        # Generator charge from voltage, transmitter model: Q_gen_coarse is the full generator charge (all cavities)

        if self.use_gen_fine:
            self.Q_gen_fine = self.G_tx*np.copy(self.V_gen_fine)  /self.TWC.R_gen*self.T_s_fine   #* 4/self.n_cavities
        self.Q_gen_coarse   = self.G_tx*np.copy(self.V_gen_coarse)/self.TWC.R_gen*self.T_s_coarse #* 4/self.n_cavities
        #print(f'mean(|Q_gen_coarse|) = {np.mean(np.absolute(self.Q_gen_coarse))/1e-9:6.2f} [nC]') # MYEDIT 2020.10.02
        # print(f'Q_gen_fine   = {self.Q_gen_fine},   shape = {self.Q_gen_fine.shape}')
        # print(f'Q_gen_coarse = {self.Q_gen_coarse},\n       abs^2 = {np.abs(self.Q_gen_coarse)**2}, shape = {self.Q_gen_coarse.shape}')

        ##########

        # CLAMP

        if self.use_gen_fine:
            if self.Q_gen_abs_max_fine:
                # print(f'Clamp threshold Q_gen_abs_max_fine = {self.Q_gen_abs_max_fine}')
                Q_gen_abs_fine, Q_gen_ang_fine = cartesian_to_polar(self.Q_gen_fine)
                indices_clamp_fine = np.where( Q_gen_abs_fine >= self.Q_gen_abs_max_fine)[0]
                # print(f'Clamping fine indices = {indices_clamp_fine}, shape = {indices_clamp_fine.shape}')
                Q_gen_abs_fine[ indices_clamp_fine ] = self.Q_gen_abs_max_fine
                # print(f'Q_gen_fine = {self.Q_gen_fine},\n       abs = {np.abs(self.Q_gen_fine)}, shape = {self.Q_gen_fine.shape}')
                self.Q_gen_fine = polar_to_cartesian( Q_gen_abs_fine, Q_gen_ang_fine)
                # print(f'Q_gen_fine = {self.Q_gen_fine},\n       abs = {np.abs(self.Q_gen_fine)}, shape = {self.Q_gen_fine.shape}')
            else:
                pass

        if self.Q_gen_abs_max_coarse:
            Q_gen_abs_coarse, Q_gen_ang_coarse = cartesian_to_polar(self.Q_gen_coarse)
            indices_clamp_coarse = np.where( Q_gen_abs_coarse >= self.Q_gen_abs_max_coarse)[0]
            if len(indices_clamp_coarse) > 0: print(f'> Clamping ({self.cavtype}) at turn {self.counter}): coarse indices = {indices_clamp_coarse}, shape = {indices_clamp_coarse.shape}')
            Q_gen_abs_coarse[ indices_clamp_coarse ] = self.Q_gen_abs_max_coarse
            self.Q_gen_coarse = polar_to_cartesian( Q_gen_abs_coarse, Q_gen_ang_coarse )
            if len(indices_clamp_coarse) > 0: print(f'Q_gen_coarse ({self.cavtype}) = {self.Q_gen_coarse},\nabs = {np.abs(self.Q_gen_coarse)}, shape = {self.Q_gen_coarse.shape}\n')
            #
            # if len(indices_clamp_coarse) > 0:
            #     fig, ax = plt.subplots(4, 1, sharex=True)
            #     ax[0].plot(np.real(self.Q_gen_coarse),  label='Q_gen_coarse')
            #     ax[1].plot(np.imag(self.Q_gen_coarse),  label='Q_gen_coarse')
            #     ax[2].plot(np.abs(self.Q_gen_coarse),   label='Q_gen_coarse')
            #     ax[3].plot(np.angle(self.Q_gen_coarse), label='Q_gen_coarse')
            #     ax[0].legend()
            #     ax[0].set_ylabel('Real')
            #     ax[1].set_ylabel('Imag')
            #     ax[2].set_ylabel('Abs')
            #     ax[3].set_ylabel('Angle')
            #     plt.show()
        else:
            # self.Q_gen_abs_max_coarse is always False if use_power_clamp was
            # False when update_variavbles was called before this module. This
            # is always done in pretracking, and in tracking with beam when
            # no power_clamp requested in SPSCavityFeedback
            pass

        #########3

        # quit()

        # Circular convolution: attach last points of previous turn
        if self.use_gen_fine:
            self.Q_gen_fine = np.concatenate((self.Q_gen_fine_prev,   self.Q_gen_fine))
        # if not self.in_track_beam or self.counter_pretrack >= 1:
        self.Q_gen_coarse   = np.concatenate((self.Q_gen_coarse_prev, self.Q_gen_coarse))
        # else:
        #     self.Q_gen_coarse   = np.concatenate((self.Q_gen_coarse[-self.n_mov_av_coarse:], self.Q_gen_coarse))
        # print(f'Q_gen_fine_prev   = {self.Q_gen_fine_prev},   shape = {self.Q_gen_fine_prev.shape}')
        # print(f'Q_gen_coarse_prev = {self.Q_gen_coarse_prev}, shape = {self.Q_gen_coarse_prev.shape}')
        # print(f'Q_gen_fine   = {self.Q_gen_fine},   shape = {self.Q_gen_fine.shape}')
        # print(f'Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse.shape}')

        # Generator-induced voltage
        if self.use_gen_fine:
            self.induced_voltage('gen_fine')
        self.induced_voltage('gen_coarse')
        # print(f'TWC.h_gen_coarse = {self.TWC.h_gen_coarse}, shape = {self.TWC.h_gen_coarse.shape}, type = {type(self.TWC.h_gen_coarse)}, dtype = {self.TWC.h_gen_coarse.dtype}')
        # print(f'Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse.shape}, type = {type(self.Q_gen_coarse)}, dtype = {self.Q_gen_coarse.dtype}')
        # print(f'V_ind_gen_coarse = {self.V_ind_gen_coarse}, shape = {self.V_ind_gen_coarse.shape}, type = {type(self.V_ind_gen_coarse)}, dtype = {self.V_ind_gen_coarse.dtype}')

        # - - -
        if self.add_ff_to_gen and self.in_track_beam:

            # # FULL DEVELOPMENT: It seems relatively OK, but the padding for the coarseFF to compensate deconvolution
            # # and n_mov_av to be able to add to Q_gen is tricky. Maybe just go for the simplified version: take th
            # # Q_beam_ff and transform to Q_gen_ff with G_ff and n_cavities
            # 
            # # # Deconvolution theory:
            # # filtered = np.convolve(signal, gauss, mode='same')        -> V =   conv(Q, h) = conv(h, Q)
            # # deconv,  _ = scipy.signal.deconvolve( filtered, gauss )   -> Q = deconv(V, h)
            # # #the deconvolution has n = len(signal) - len(gauss) + 1 points
            # # n = len(signal)-len(gauss)+1
            # # # so we need to expand it by
            # # s = (len(signal)-n)/2
            # # #on both sides.
            # # deconv_res = np.zeros(len(signal))
            # # deconv_res[s:len(signal)-s-1] = deconv
            # # deconv = deconv_res
            # # # now deconv contains the deconvolution
            # # # expanded to the original shape (filled with zeros)
            # 
            # # Voltage applied by the FF, which must come from extra current in generator:
            # # In beam: self.dV_ind_beam_ff_coarseFF = self.G_ff * self.n_cavities * self.matr_conv(self.Q_beam_ff_coarseFF, self.TWC.h_gen_coarse[self.indices_coarseFF])
            # # + delays + interpolation -> self.dV_ind_beam_ff_coarse
            # 
            # # j, jff = self.fillpattern[0], int(self.fillpattern[0]/5)
            # print(f'{self.cavtype}, counter = {self.counter}, first bunch')
            # print(f'Q_beam_ff_coarseFF  = {self.Q_beam_ff_coarseFF}, abs = {np.abs(self.Q_beam_ff_coarseFF)}, ang = {np.angle(self.Q_beam_ff_coarseFF)}, shape = {self.Q_beam_ff_coarseFF.shape}')
            # print(f'dV_ind_beam_ff_coarse = {self.dV_ind_beam_ff_coarse}, abs = {np.abs(self.dV_ind_beam_ff_coarse)}, ang = {np.angle(self.dV_ind_beam_ff_coarse)}, shape = {self.dV_ind_beam_ff_coarse.shape}')
            # h_gen_coarse_tmp = np.copy(self.TWC.h_gen_coarse[:self.n_mov_av_coarse:])
            # h_gen_coarse_tmp[0] = h_gen_coarse_tmp[1] # This first point triggers at half-height triggers instability
            # h_gen_coarseFF_tmp = np.copy(self.TWC.h_gen_coarse[:self.n_mov_av_coarse:self.nbs])
            # h_gen_coarseFF_tmp[0] = h_gen_coarse_tmp[1]
            # Q_gen_ff_coarse,  _   = scipy.signal.deconvolve( self.dV_ind_beam_ff_coarse, h_gen_coarse_tmp)
            # Q_gen_ff_coarseFF,  _ = scipy.signal.deconvolve( self.dV_ind_beam_ff_coarseFF, h_gen_coarseFF_tmp)
            # print(len(self.dV_ind_beam_ff_coarse), len(h_gen_coarse_tmp), len(Q_gen_ff_coarse))
            # print(len(self.dV_ind_beam_ff_coarseFF), len(h_gen_coarseFF_tmp), len(Q_gen_ff_coarseFF))
            # # Q_gen_ff_coarse will be len(dV_ind_beam_ff_coarse) - len(h_gen_coarse_tmp) + 1 = n_coarse - n_mov_av_coarse + 1 samples long
            # # Q_gen_ff_coarseFF will be len(dV_ind_beam_ff_coarseFF) - len(h_gen_coarseFF_tmp) + 1 = n_coarse/nbs - n_mov_av_coarse/nbs + 1 samples long
            # # then we need to pad zeros (n_mov_av_coarse-1)/2 and (n_mov_av_coarse/nbs-1)/2 on both sides to get the original one-turn length signals:
            # # n_coarse and n_coarseFF, respectively. Also, to be able to add to Q_gen, we need to add the extra n_mov_av_coarse and n_mov_av_coarse/nbs
            # # samples at the beginning of Q_gen (long array)
            # self.Q_gen_ff_coarse = np.zeros(self.n_coarse_long, dtype=complex)
            # self.Q_gen_ff_coarseFF = np.zeros(int(self.n_coarse_long/self.nbs), dtype=complex)
            # self.Q_gen_ff_coarse[ int(self.n_mov_av_coarse+0.5*(self.n_mov_av_coarse-1))+1 : int(self.n_mov_av_coarse+0.5*(self.n_mov_av_coarse-1))+1+len(Q_gen_ff_coarse) ] = Q_gen_ff_coarse[:]
            # self.Q_gen_ff_coarseFF[ int(self.n_mov_av_coarse/self.nbs+0.5*(self.n_mov_av_coarse/self.nbs-1))+1 : int(self.n_mov_av_coarse/self.nbs+0.5*(self.n_mov_av_coarse/self.nbs-1))+1+len(Q_gen_ff_coarseFF) ] = Q_gen_ff_coarseFF[:]
            # print(f'Q_gen_ff_coarse  = {self.Q_gen_ff_coarse}, abs = {np.abs(self.Q_gen_ff_coarse)}, ang = {np.angle(self.Q_gen_ff_coarse)}, shape = {self.Q_gen_ff_coarse.shape}')
            # print(f'Q_gen_ff_coarseFF  = {self.Q_gen_ff_coarseFF}, abs = {np.abs(self.Q_gen_ff_coarseFF)}, ang = {np.angle(self.Q_gen_ff_coarseFF)}, shape = {self.Q_gen_ff_coarseFF.shape}')
            # # At the end, this is just the beam current (times G_ff times n_cavities).
            # 
            # self.I_gen_noff_coarse = self.Q_gen_coarse / self.n_cavities / self.T_s_coarse
            # print(f'I_gen_noff_coarse  = {self.I_gen_noff_coarse}, abs = {np.abs(self.I_gen_noff_coarse)}, ang = {np.angle(self.I_gen_noff_coarse)}, shape = {self.I_gen_noff_coarse.shape}')
            # 
            # self.I_gen_ff_coarse   = self.Q_gen_ff_coarse   / self.n_cavities / self.T_s_coarse
            # self.I_gen_ff_coarseFF = self.Q_gen_ff_coarseFF / self.n_cavities / self.T_s_coarse
            # print(f'I_gen_ff_coarse  = {self.I_gen_ff_coarse}, abs = {np.abs(self.I_gen_ff_coarse)}, ang = {np.angle(self.I_gen_ff_coarse)}, shape = {self.I_gen_ff_coarseFF.shape}')
            # print(f'I_gen_ff_coarseFF  = {self.I_gen_ff_coarseFF}, abs = {np.abs(self.I_gen_ff_coarseFF)}, ang = {np.angle(self.I_gen_ff_coarseFF)}, shape = {self.I_gen_ff_coarseFF.shape}')
            # 
            # self.I_gen_coarseFF = np.copy(self.I_gen_noff_coarse[::self.nbs])
            # self.I_gen_coarseFF -= self.I_gen_ff_coarseFF # Opposite sign
            # # print(f'I_gen_coarse = {self.I_gen_coarse}, abs = {np.abs(self.I_gen_coarse)}, ang = {np.angle(self.I_gen_coarse)}, shape = {self.I_gen_coarse.shape}')
            # print(f'I_gen_coarse = {self.I_gen_coarseFF}, abs = {np.abs(self.I_gen_coarseFF)}, ang = {np.angle(self.I_gen_coarseFF)}, shape = {self.I_gen_coarseFF.shape}')
            # print('')
            # 
            # ncols = 2
            # nrows = 5
            # 
            # fig, ax = plt.subplots(nrows,ncols,sharex=True)
            # fig.set_size_inches(ncols*5.00, nrows*2.50)
            # 
            # # h_gen at coarseFF as used in the FF
            # ax[0,0].plot(self.indices_coarseFF, self.TWC.h_gen_coarse[self.indices_coarseFF].real, '+', label='Re(hg) coarseFF')
            # ax[0,0].plot(self.indices_coarseFF, self.TWC.h_gen_coarse[self.indices_coarseFF].imag, '+', label='Im(hg) coarseFF')
            # # h_gen at all coarse grid points, but only non-zero part, i.e. up to tau i.e. n_mov_ave points), for Q_gen_ff_coarse computation
            # ax[0,1].plot(h_gen_coarse_tmp.real, '+', label='Re(hg) coarse')
            # ax[0,1].plot(h_gen_coarse_tmp.imag, '+', label='Im(hg) coarse')
            # # h_gen at coarseFF, but only non-zero part, i.e. up to tau i.e. n_mov_ave points), for Q_gen_ff_coarseFF computation
            # ax[0,1].plot(np.arange(len(h_gen_coarseFF_tmp))*self.nbs, h_gen_coarseFF_tmp.real, '+', label='Re(hg) coarseFF')
            # ax[0,1].plot(np.arange(len(h_gen_coarseFF_tmp))*self.nbs, h_gen_coarseFF_tmp.imag, '+', label='Im(hg) coarseFF')
            # 
            # # FIR-procecessed beam current at coarseFF as used in the FF
            # ax[1,0].plot(self.indices_coarseFF, self.Q_beam_ff_coarseFF.real, '+', label='Re(Qbff) coarseFF')
            # ax[1,0].plot(self.indices_coarseFF, self.Q_beam_ff_coarseFF.imag, '+', label='Im(Qbff) coarseFF')
            # 
            # # FF voltage correction result (interpolated to coarse from coarseFF)
            # ax[2,0].plot(self.dV_ind_beam_ff_coarse.real, '+', label='Re(dVbff) coarse')
            # ax[2,0].plot(self.dV_ind_beam_ff_coarse.imag, '+', label='Im(dVbff) coarse')
            # 
            # # Generator current at coarse grid
            # ax[3,0].plot(self.Q_gen_coarse.real, '.', label='Re(Qgnoff) coarse')
            # ax[3,0].plot(self.Q_gen_coarse.imag, '.', label='Im(Qgnoff) coarse')
            # 
            # # Generator current corresponding from FF correction, at coarse and coarseFF
            # l0 = ax[1,1].plot(self.Q_gen_ff_coarse.real, '.', label='Re(Qgff) coarse')
            # l1 = ax[1,1].plot(self.Q_gen_ff_coarse.imag, '.', label='Im(Qgff) coarse')
            # l2 = ax[1,1].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs, self.Q_gen_ff_coarseFF.real, '.', label='Re(Qgff) coarseFF')
            # l3 = ax[1,1].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs, self.Q_gen_ff_coarseFF.imag, '.', label='Im(Qgff) coarseFF')
            # 
            # ax[1,1].plot(self.indices_coarseFF, self.Q_beam_ff_coarseFF.real * self.n_cavities * self.G_ff, '-', color = l2[0].get_color())
            # ax[1,1].plot(self.indices_coarseFF, self.Q_beam_ff_coarseFF.imag * self.n_cavities * self.G_ff, '-', color = l3[0].get_color())
            # 
            # # Physical current from generator current (i.e. no ff) at coarse grid
            # ax[4,0].plot(self.I_gen_noff_coarse.real, '.', label='Re(Ignoff) coarse')
            # ax[4,0].plot(self.I_gen_noff_coarse.imag, '.', label='Im(Ignoff) coarse')
            # 
            # # # Physical current from generator current correspoding to FF correction, at coarse and coarseFF
            # ## ax[2,1].plot(self.I_gen_ff_coarse.real, '.', label='Re(Igff) coarse')
            # ## ax[2,1].plot(self.I_gen_ff_coarse.imag, '.', label='Im(Igff) coarse')
            # ax[3,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs, self.I_gen_ff_coarseFF.real, '.', label='Re(Igff) coarseFF')
            # ax[3,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs, self.I_gen_ff_coarseFF.imag, '.', label='Im(Igff) coarseFF')
            # ax[4,1].plot(np.arange(len(self.I_gen_coarseFF)), self.I_gen_coarseFF.real, '.', label='Re(Igff), coarseFF')
            # ax[4,1].plot(np.arange(len(self.I_gen_coarseFF)), self.I_gen_coarseFF.imag, '.', label='Im(Igff), coarseFF')
            # 
            # ax[1,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
            # ax[1,1].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
            # ax[3,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
            # ax[3,1].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
            # ax[4,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
            # 
            # ax[1,0].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls=':', lw=1.0, color='grey')
            # ax[1,1].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls=':', lw=1.0, color='grey')
            # ax[3,0].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls=':', lw=1.0, color='grey')
            # ax[3,1].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls=':', lw=1.0, color='grey')
            # ax[4,0].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls=':', lw=1.0, color='grey')
            # 
            # for isp in range(nrows):
            #     for jsp in range(ncols):
            #         ax[isp,jsp].legend()
            #         # ax[isp,jsp].set_xlim(self.fillpattern[0]-25,self.fillpattern[0]+100+25)
            #         ax[isp,jsp].set_xlim(self.fillpattern[0]-200,self.fillpattern[0]+200)
            #         # ax[isp,jsp].set_xlim(0,700)
            # 
            # fig.tight_layout()
            # fig.savefig(f'{self.outdir}/plot_Qgff_{self.cavtype}_{self.counter}.png')
            # plt.cla()
            # plt.close(fig)
            # 
            # SIMPLIFIED:

            self.Q_gen_ff_coarseFF = self.G_ff * self.n_cavities * self.Q_beam_ff_coarseFF/2.
            self.Q_gen_ff_coarse   = np.interp(np.arange(len(self.Q_gen_ff_coarseFF)*self.nbs),
                                               np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs,
                                               self.Q_gen_ff_coarseFF/2.) / self.nbs  # <<----
            # print(f'len(self.Q_gen_ff_coarseFF) = {len(self.Q_gen_ff_coarseFF)}')
            # print(f'len(self.Q_gen_ff_coarse) = {len(self.Q_gen_ff_coarse)}')

            self.I_gen_ff_coarseFF = self.Q_gen_ff_coarseFF/self.T_s_coarse / self.n_cavities
            self.I_gen_ff_coarse   = self.Q_gen_ff_coarse  /self.T_s_coarse / self.n_cavities

            self.Q_gen_wff_coarse = np.copy(self.Q_gen_coarse)
            # print(f'len(self.Q_gen_coarse) = {len(self.Q_gen_coarse)}')

            diff_g_ff = self.n_mov_av_coarse-self.n_FF_delay*self.nbs
            # print(f'diff_g_ff = {diff_g_ff}')
            if diff_g_ff >= 0:
                # print(f'len(Q_gen_wff_coarse[...]) = {len(self.Q_gen_wff_coarse[diff_g_ff: diff_g_ff + len(self.Q_gen_ff_coarse)])}')
                self.Q_gen_wff_coarse[ diff_g_ff : diff_g_ff+len(self.Q_gen_ff_coarse) ] += self.Q_gen_ff_coarse * np.exp(-0.5j*np.pi)
            else:
                diff_g_ff = abs(diff_g_ff)
                # print(f'len(Q_gen_wff_coarse[...]) = {len(self.Q_gen_wff_coarse[:len(self.Q_gen_ff_coarse) - diff_g_ff])}')
                # print(f'len(Q_gen_ff_coarse[diff_g_ff:]) = {len(self.Q_gen_ff_coarse[diff_g_ff:])}')
                self.Q_gen_wff_coarse[ :len(self.Q_gen_ff_coarse)-diff_g_ff ] += self.Q_gen_ff_coarse[diff_g_ff:] * np.exp(-0.5j*np.pi)

            self.I_gen_wff_coarse = self.Q_gen_wff_coarse/self.T_s_coarse / self.n_cavities
            self.P_gen_wff_coarse = get_power_gen_I2(self.I_gen_wff_coarse, self.TWC.Z_0)

            I_gen_coarse = self.Q_gen_coarse/self.T_s_coarse / self.n_cavities
            P_gen_coarse = get_power_gen_I2(I_gen_coarse, self.TWC.Z_0)

            if True: #self.counter <=3 or self.counter >= self.rf.n_turns-2-1:

                ncols = 3
                nrows = 6
                fig, ax = plt.subplots(nrows,ncols,sharex=True)
                fig.set_size_inches(ncols*5.00, nrows*2.50)

                ax[0,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs, np.real(self.Q_beam_ff_coarseFF), '+', label='Re(Qbff) coarseFF')
                ax[0,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs, np.imag(self.Q_beam_ff_coarseFF), '+', label='Im(Qbff) coarseFF')
                ax[0,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs,  np.abs(self.Q_beam_ff_coarseFF), '+', label='abs(Qbff) coarseFF')
                ax[0,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')

                ax[1,0].plot(np.real(self.dV_ind_beam_ff_coarse), '+', label='Re(dVbff) coarse')
                ax[1,0].plot(np.imag(self.dV_ind_beam_ff_coarse), '+', label='Im(dVbff) coarse')
                ax[1,0].plot( np.abs(self.dV_ind_beam_ff_coarse), '+', label='Abs(dVbff) coarse')
                ax[1,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[1,0].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')

                ax[2,0].plot(np.real(self.Q_gen_coarse), '+', label='Re(Qg) coarse')
                ax[2,0].plot(np.imag(self.Q_gen_coarse), '+', label='Im(Qg) coarse')
                ax[2,0].plot( np.abs(self.Q_gen_coarse), '+', label='Abs(Qg) coarse')
                ax[2,0].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[2,1].plot(np.real(I_gen_coarse), '+', label='Re(Ig) coarse')
                ax[2,1].plot(np.imag(I_gen_coarse), '+', label='Im(Ig) coarse')
                ax[2,1].plot( np.abs(I_gen_coarse), '+', label='Abs(Ig) coarse')
                ax[2,1].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[2,2].plot(P_gen_coarse, '+', label='Pg coarse')
                ax[2,2].axvline(self.fillpattern[0]+self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                l0 = ax[3,0].plot(np.real(self.Q_gen_ff_coarse), '.', label='Re(Qgff) coarse')
                l1 = ax[3,0].plot(np.imag(self.Q_gen_ff_coarse), '.', label='Im(Qgff) coarse')
                l2 = ax[3,0].plot( np.abs(self.Q_gen_ff_coarse), '.', label='Abs(Qgff) coarse')
                l4 = ax[3,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs, np.real(self.Q_gen_ff_coarseFF), '.', label='Re(Qgff) coarseFF')
                l5 = ax[3,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs, np.imag(self.Q_gen_ff_coarseFF), '.', label='Im(Qgff) coarseFF')
                l6 = ax[3,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs,  np.abs(self.Q_gen_ff_coarseFF), '.', label='Abs(Qgff) coarseFF')
                ax[3,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[3,0].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[3,0].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[3,1].plot(np.real(self.I_gen_ff_coarse), '.', label='Re(Igff) coarse')
                ax[3,1].plot(np.imag(self.I_gen_ff_coarse), '.', label='Im(Igff) coarse')
                ax[3,1].plot( np.abs(self.I_gen_ff_coarse), '.', label='Abs(Igff) coarse')
                ax[3,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs, np.real(self.I_gen_ff_coarseFF), '.', label='Re(Igff) coarseFF')
                ax[3,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs, np.imag(self.I_gen_ff_coarseFF), '.', label='Im(Igff) coarseFF')
                ax[3,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs,  np.abs(self.I_gen_ff_coarseFF), '.', label='Abs(Igff) coarseFF')
                ax[3,1].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[3,1].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[3,1].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[4,0].plot(np.arange(len(self.Q_gen_ff_coarse))+self.n_mov_av_coarse-self.n_FF_delay*self.nbs, np.real(self.Q_gen_ff_coarse), '.', label='Re(Qgff) coarse')
                ax[4,0].plot(np.arange(len(self.Q_gen_ff_coarse))+self.n_mov_av_coarse-self.n_FF_delay*self.nbs, np.imag(self.Q_gen_ff_coarse), '.', label='Im(Qgff) coarse')
                ax[4,0].plot(np.arange(len(self.Q_gen_ff_coarse))+self.n_mov_av_coarse-self.n_FF_delay*self.nbs,  np.abs(self.Q_gen_ff_coarse), '.', label='Im(Qgff) coarse')
                ax[4,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs+self.n_mov_av_coarse/self.nbs*self.nbs-self.n_FF_delay*self.nbs, np.real(self.Q_gen_ff_coarseFF), '.', label='Re(Qgff) coarseFF')
                ax[4,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs+self.n_mov_av_coarse/self.nbs*self.nbs-self.n_FF_delay*self.nbs, np.imag(self.Q_gen_ff_coarseFF), '.', label='Im(Qgff) coarseFF')
                ax[4,0].plot(np.arange(len(self.Q_gen_ff_coarseFF))*self.nbs+self.n_mov_av_coarse/self.nbs*self.nbs-self.n_FF_delay*self.nbs,  np.abs(self.Q_gen_ff_coarseFF), '.', label='Abs(Qgff) coarseFF')
                ax[4,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[4,0].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[4,0].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[4,1].plot(np.arange(len(self.I_gen_ff_coarse))+self.n_mov_av_coarse-self.n_FF_delay*self.nbs, np.real(self.I_gen_ff_coarse), '.', label='Re(Igff) coarse')
                ax[4,1].plot(np.arange(len(self.I_gen_ff_coarse))+self.n_mov_av_coarse-self.n_FF_delay*self.nbs, np.imag(self.I_gen_ff_coarse), '.', label='Im(Igff) coarse')
                ax[4,1].plot(np.arange(len(self.I_gen_ff_coarse))+self.n_mov_av_coarse-self.n_FF_delay*self.nbs,  np.abs(self.I_gen_ff_coarse), '.', label='Abs(Igff) coarse')
                ax[4,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs+self.n_mov_av_coarse/self.nbs*self.nbs-self.n_FF_delay*self.nbs, np.real(self.I_gen_ff_coarseFF), '.', label='Re(Igff) coarseFF')
                ax[4,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs+self.n_mov_av_coarse/self.nbs*self.nbs-self.n_FF_delay*self.nbs, np.imag(self.I_gen_ff_coarseFF), '.', label='Im(Igff) coarseFF')
                ax[4,1].plot(np.arange(len(self.I_gen_ff_coarseFF))*self.nbs+self.n_mov_av_coarse/self.nbs*self.nbs-self.n_FF_delay*self.nbs,  np.abs(self.I_gen_ff_coarseFF), '.', label='Abs(Igff) coarseFF')
                ax[4,1].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[4,1].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[4,1].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[5,0].plot(np.real(self.Q_gen_wff_coarse), '.', label='Re(Qgwff) coarse')
                ax[5,0].plot(np.imag(self.Q_gen_wff_coarse), '.', label='Im(Qgwff) coarse')
                ax[5,0].plot( np.abs(self.Q_gen_wff_coarse), '.', label='Abs(Qgwff) coarse')
                # ax[5,0].plot(np.arange(len(self.Q_gen_wff_coarseFF))*self.nbs, np.real(self.Q_gen_wff_coarseFF), '.', label='Re(Qgwff) coarseFF')
                # ax[5,0].plot(np.arange(len(self.Q_gen_wff_coarseFF))*self.nbs, np.imag(self.Q_gen_wff_coarseFF), '.', label='Im(Qgwff) coarseFF')
                # ax[5,0].plot(np.arange(len(self.Q_gen_wff_coarseFF))*self.nbs,  np.abs(self.Q_gen_wff_coarseFF), '.', label='Abs(Qgwff) coarseFF')
                ax[5,0].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[5,0].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[5,0].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[5,1].plot(np.real(self.I_gen_wff_coarse), '.', label='Re(Igwff) coarse')
                ax[5,1].plot(np.imag(self.I_gen_wff_coarse), '.', label='Im(Igwff) coarse')
                ax[5,1].plot( np.abs(self.I_gen_wff_coarse), '.', label='Abs(Igwff) coarse')
                # ax[5,1].plot(np.arange(len(self.I_gen_wff_coarseFF))*self.nbs, np.real(self.I_gen_wff_coarseFF), '.', label='Re(Igwff) coarseFF')
                # ax[5,1].plot(np.arange(len(self.I_gen_wff_coarseFF))*self.nbs, np.imag(self.I_gen_wff_coarseFF), '.', label='Im(Igwff) coarseFF')
                # ax[5,1].plot(np.arange(len(self.I_gen_wff_coarseFF))*self.nbs,  np.abs(self.I_gen_wff_coarseFF), '.', label='Abs(Igwff) coarseFF')
                ax[5,1].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[5,1].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[5,1].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                ax[5,2].plot(self.P_gen_wff_coarse, '.', label='Pgwff coarse')
                ax[5,2].axvline(self.fillpattern[0], ls='-', lw=1.0, color='grey')
                ax[5,2].axvline(self.fillpattern[0] - self.n_FF_delay * self.nbs, ls='--', lw=1.0, color='grey')
                ax[5,2].axvline(self.fillpattern[0] + self.n_mov_av_coarse, ls='-.', lw=1.0, color='grey')

                for isp in range(nrows):
                    for jsp in range(ncols):
                      ax[isp, jsp].legend()
                      # ax[isp,jsp].set_xlim(self.fillpattern[0]-200,self.fillpattern[0]+200)
                      # ax[isp,jsp].set_xlim(-50,50)
                      ax[isp,jsp].set_xlim(self.fillpattern[0]-200,self.fillpattern[-1]+200)
                      ax[isp,jsp].axvline(0, ls='-', lw=1.0, color='black')
                      ax[isp,jsp].axvline(self.n_coarse, ls='-', lw=1.0, color='black')
                      ax[isp,jsp].axvline(self.n_coarse+self.n_mov_av_coarse, ls='-', lw=1.0, color='black')

                fig.tight_layout()
                fig.savefig(f'{self.outdir}/plot_Qgff_{self.cavtype}_{self.counter}.png')
                plt.cla()
                plt.close(fig)

                # if self.cavtype == '2x4sec' and self.counter == 2:
                #     quit()

        # - - -

        # Update memory of previous turn
        if self.use_gen_fine:
            self.Q_gen_fine_prev = self.Q_gen_fine[  -self.n_mov_av_fine:]
        self.Q_gen_coarse_prev   = self.Q_gen_coarse[-self.n_mov_av_coarse:]
        # print(f'Q_gen_fine_prev   = {self.Q_gen_fine_prev},   shape = {self.Q_gen_fine_prev.shape}')
        # print(f'Q_gen_coarse_prev = {self.Q_gen_coarse_prev}, shape = {self.Q_gen_coarse_prev.shape}')
        #print(f'Q_gen_coarse_prev = {self.Q_gen_coarse_prev}, shape = {self.Q_gen_coarse_prev.shape}')

        #     Q_gen_coarse_0 = self.Q_gen_coarse[:]
        #     self.Q_gen_coarse = np.concatenate((self.Q_gen_coarse_prev, Q_gen_coarse_0[:-self.n_mov_av_coarse]))
        #     #print(f'Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse}')
        #     # Update memory of previous turn
        #     self.Q_gen_coarse_prev = Q_gen_coarse_0[-self.n_mov_av_coarse:]
        #     #print(f'Q_gen_coarse_prev = {self.Q_gen_coarse_prev}, shape = {self.Q_gen_coarse_prev.shape}')

        #     # Generator-induced voltage
        #     self.induced_voltage('gen')


    def induced_voltage(self, name):
        r"""Generation of beam- or generator-induced voltage from the
        beam or
        generator current, at a given carrier frequency and turn. The
        induced
        voltage :math:`V(t)` is calculated from the impulse response matrix
        :math:`h(t)` as follows:

        .. math::
            \left( \begin{matrix} V_I(t) \\
            V_Q(t) \end{matrix} \right)
            = \left( \begin{matrix} h_s(t) & - h_c(t) \\
            h_c(t) & h_s(t) \end{matrix} \right)
            * \left( \begin{matrix} Q_I(t) \\
            Q_Q(t) \end{matrix} \right) \, ,

        where :math:`*` denotes convolution,
        :math:`h(t)*x(t) = \int d\tau h(\tau)x(t-\tau)`. If the carrier
        frequency is close to the cavity resonant frequency, :math:`h_c
        = 0`.

        :see also: :py:class:`llrf.impulse_response.TravellingWaveCavity`

        The impulse response is made to be the same length as the beam
        profile.

        """

        self.logger.debug("Matrix convolution for V_ind")

        if "beam" in name: # name is 'beam_fine' or 'beam_coarse'
            # Compute the beam-induced voltage on the fine grid
            # print(self.__getattribute__("Q_"+name+"_fine"))
            # print(cartesian_rotation(self.__getattribute__("Q_"+name+"_fine"), np.pi))
            # quit()
            V_ind = \
                self.matr_conv(getattr(self,     "Q_"+name), # or cartesian_rotation in I by np.pi instead of sign below?
                               getattr(self.TWC, "h_"+name))
            #self.V_ind_beam_fine = remove_noise_cartesian(self.V_ind_beam_fine, threshold=1e-3*self.rf.voltage[0,self.counter]) # i.e 1/1000th of V0 ~ 10 MV -> 1 kV
            # self.V_ind_beam_fine *= self.Vindbeamsign * self.n_cavities # Original: negative, changed to positive for testing
            setattr(self, "V_ind_"+name, +1 * self.n_cavities * V_ind ) #Changed adopted: using the positive now

        if "gen" in name: # name is 'gen_fine' or 'gen_coarse'
            # Compute the generator-induced voltage on the coarse grid
            # print(f'Q_{name} = {getattr(self,     "Q_"+name)}, shape = {getattr(self,     "Q_"+name).shape}')
            # print(f'h_{name} = {getattr(self.TWC, "h_"+name)}, shape = {getattr(self.TWC, "h_"+name).shape}')
            V_ind = \
                self.matr_conv(getattr(self,     "Q_"+name), # MEDIT: Originally added /self.n_cavities, but then it can be removed if the self.n_cavities is also removed from the following line
                               getattr(self.TWC, "h_"+name))
            #self.V_ind_gen_coarse = remove_noise_cartesian(self.V_ind_gen_coarse, threshold=1e-3*self.rf.voltage[0,self.counter]) # i.e 1/1000th of V0 ~ 10 MV -> 1 kV
            # Circular convolution
            n_fc        = 1*self.n_coarse        if 'coarse' in name else 1*self.n_fine
            n_mov_av_fc = 1*self.n_mov_av_coarse if 'coarse' in name else 1*self.n_mov_av_fine
            # print(f'V_ind_{name} = {V_ind}, shape = {V_ind.shape} ->')
            setattr(self, "V_ind_" + name, +1.0 * V_ind[ n_mov_av_fc : n_fc+n_mov_av_fc ] ) # MYEDIT: removed self.n_cavities -> 1.0, possible because self.n_cavities is alsore remove din the previous line
            # V_ind = 1*getattr(self, "V_ind_" + name)
            # print(f'V_ind_{name} = {V_ind}, shape = {V_ind.shape}')
            #print(f'max(|V_ind_gen_coarse|)  = {max(np.absolute(self.V_ind_gen_coarse))/1e6:6.2f} MV') # MYEDIT 2020.10.01


    def set_point(self, V_part_rampfact_fine_i=None, V_part_rampfact_coarse_i=None): # V_part_rampfact_coarse_i = (1,0) means full magnitude of V_part (1), and the exact phi_rf (+0)

        # Voltage set point of current turn (I,Q); depends on voltage partition
        # Sinusoidal voltage completely in Q

        # MYEDIT: 2020.12.09: Note that we are using the present phi_rf which
        # has been corrected in the previous turn due to beam phase loop (if
        # applicable).
        # MYEDIT: 2020.12.16: Change in polar_coordinates to force zero for
        # small cos() or sin().

        # if Vset_imag is True (=1), Vset is defined at +Q; else if Vset_imag is False, then Vset is defined at +I

        if self.use_gen_fine:

            if V_part_rampfact_fine_i is None:

                V_set = polar_to_cartesian(
                    self.V_part*self.rf.voltage[0, self.counter],
                    0.5*np.pi*self.Vset_imag - self.rf.phi_rf[0,self.counter] * self.Vset_imag) #,
                    #threshold=1.) # Anything below 1 V is treated as zero, as 1 V is O(-6) w.r.t. MV
                # Convert to array
                self.V_set_fine = V_set*np.ones(self.n_fine)

            else:

                # Made an array from the beginning:
                self.V_set_fine = polar_to_cartesian(
                    np.ones(self.n_fine) * self.V_part*V_part_rampfact_fine_i[0]*self.rf.voltage[0, self.counter],
                    0.5*np.pi*self.Vset_imag - (self.rf.phi_rf[0,self.counter]+V_part_rampfact_fine_i[1]))

        # print(f'V_set_fine = {self.V_set_fine}, shape = {self.V_set_fine.shape}')

        if V_part_rampfact_coarse_i is None:

            V_set = polar_to_cartesian(
                self.V_part*self.rf.voltage[0, self.counter],
                0.5*np.pi*self.Vset_imag - self.rf.phi_rf[0,self.counter]) #,
                #threshold=1.) # Anything below 1 V is treated as zero, as 1 V is O(-6) w.r.t. MV
            #self.V_set_coarse = remove_noise_cartesian(self.V_set_coarse, threshold=1e-3*self.rf.voltage[0,self.counter])
            # Convert to array
            self.V_set_coarse = V_set*np.ones(self.n_coarse)
            # print(f'mean(|V_set_coarse|) = {np.mean(np.absolute(self.V_set_coarse))/1e6:6.2f} MV') # MYEDIT 2020.10.01

        else:

            # Made an array from the beginning:
            self.V_set_coarse = polar_to_cartesian(
                np.ones(self.n_coarse)*self.V_part*V_part_rampfact_coarse_i[0]*self.rf.voltage[0, self.counter],
                0.5*np.pi*self.Vset_imag - (self.rf.phi_rf[0,self.counter]+V_part_rampfact_coarse_i[1]))

        # print(f'V_set_coarse = {self.V_set_coarse}, shape = {self.V_set_coarse.shape}')

    def llrf_model(self):
        """Models the LLRF part of the OTFB.

        Attributes
        ----------
        V_set_coarse : complex array
            Voltage set point [V] in (I,Q); :math:`V_{\mathsf{set}}`, amplitude
            proportional to voltage partition
        dV_Hcav_coarse : complex array
            Generator voltage [V] in (I,Q);
            :math:`dV_{\mathsf{gen}} = V_{\mathsf{set}} - V_{\mathsf{tot}}`
        """

        # VS HTIMKO 2021-02: V_set is zero for n_mov_av points, then it is linearly
        # ramped until reaching V_set at the end of the turn

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### Difference of set point and actual voltage:
        # if self.use_gen_fine:
        #     self.dV_err_fine = np.copy(self.V_set_fine)   - self.open_loop*np.copy(self.V_tot_fine)
        self.dV_err_coarse  = np.copy(self.V_set_coarse) - self.open_loop*np.copy(self.V_tot_coarse)

        # Closed-loop gain
        # if self.use_gen_fine:
        #     self.dV_err_gain_fine = self.G_llrf * np.copy(self.dV_err_fine)
        self.dV_err_gain_coarse   = self.G_llrf * np.copy(self.dV_err_coarse)
        self.logger.debug("Set voltage %.6f MV",     1e-6*np.mean(np.absolute(self.V_set_coarse)))
        self.logger.debug("Antenna voltage %.6f MV", 1e-6*np.mean(np.absolute(self.V_tot_coarse)))
        self.logger.debug("Voltage error %.6f MV",   1e-6*np.mean(np.absolute(self.dV_err_gain_coarse)))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### One-turn delay comb filter; memorise the value of the previous turn:

        # if self.use_gen_fine:
        #     self.dV_comb_fine = comb_filter(self.dV_comb_fine_prev,   self.dV_err_gain_fine,   self.a_comb)
        self.dV_comb_coarse   = comb_filter(self.dV_comb_coarse_prev, self.dV_err_gain_coarse, self.a_comb)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### Shift signals with the delay time (to make exactly one turn):

        # if self.use_gen_fine:
        #     self.dV_del_fine = np.concatenate((self.dV_comb_fine_prev[  -self.n_delay_fine:],
        #                                        self.dV_comb_fine[  :self.n_fine  -self.n_delay_fine]))
        if self.dV_Hcav_opt == '0':
            # OPT0, OPT2a, OPT2b
            self.dV_del_coarse  = np.concatenate((self.dV_comb_coarse_prev[-self.n_delay_coarse:],
                                                  self.dV_comb_coarse[:self.n_coarse-self.n_delay_coarse]))
        # elif self.dV_Hcav_opt == '1' or self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
        # # OPT1 HTIMKO 2021-02: Added a 'long' version instead (substitutes the one above)
        #     self.dV_del_coarse_long = np.concatenate((self.dV_comb_coarse_prev[-self.n_delay_coarse:],
        #                                               self.dV_comb_coarse[:self.n_coarse-self.n_delay_coarse+self.n_mov_av_coarse-1]))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### For comb filter, update memory of previous turn:

        # if self.use_gen_fine:
        #     self.dV_comb_fine_prev = np.copy(self.dV_comb_fine)
        # OPT0, OPT1, OPT2 (all)
        self.dV_comb_coarse_prev = np.copy(self.dV_comb_coarse)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### Modulate from omega_rf to omega_r:

        if self.in_track_beam:
            self.phi_0 = self.phi_mod_0*self.counter
            ## self.phi_0 = 0 if self.counter == 0 else self.phi_mod_0
        else:
            self.phi_0 = self.phi_mod_0*self.counter_pretrack
            # self.phi_0 = 0 #if self.counter_pretrack == 0 else self.phi_mod_0
            # print(self.counter_pretrack, self.counter, self.phi_0)

        # if self.use_gen_fine:
        #     self.dV_mod_fine = modulator(self.dV_del_fine,   self.omega_c, self.omega_r, self.T_s_fine)
        if self.dV_Hcav_opt == '0':
            self.dV_mod_coarse = modulator(self.dV_del_coarse, self.omega_c, self.omega_r, self.T_s_coarse, phi_0=self.phi_0)
        # elif self.dV_Hcav_opt == '1' or self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
        #     # OPT1 HTIMKO 2021-02 and OPT2
        #     self.dV_mod_coarse_long = modulator(self.dV_del_coarse_long, self.omega_c, self.omega_r, self.T_s_coarse)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### Memorize current modulated signal to use in next turn

        # if self.use_gen_fine:
        #     self.dV_mod_in_fine = np.copy(self.dV_mod_fine)
        if self.dV_Hcav_opt == '0':
            # OPT0
            self.dV_mod_in_coarse   = np.copy(self.dV_mod_coarse)
        # elif self.dV_Hcav_opt == '1':
        #     # OPT1 HTIMKO 2021-02: Variable no longer needed
        #     pass
        # elif self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
        #     # OPT2 MY TEST: Included to fully mirror the current implmentation:
        #     self.dV_mod_in_long_coarse = np.copy(self.dV_mod_coarse_long)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ### Cavity filter: CIRCULAR moving average over filling time

        # if self.use_gen_fine:
        #     self.dV_Hcav_fine = moving_average(self.dV_mod_fine,   self.n_mov_av_fine,   x_prev=self.dV_mod_fine_prev[  -self.n_mov_av_fine+1:])   # Changed to [-self.n_mov_av_fine:-1], but reset to [-self.n_mov_av_fine+1:]
        if self.dV_Hcav_opt == '0':
            # OPT0
            self.dV_Hcav_coarse_0      = moving_average(self.dV_mod_coarse,      self.n_mov_av_coarse, x_prev=self.dV_mod_coarse_prev[-self.n_mov_av_coarse+1:])
        # elif self.dV_Hcav_opt == '1':
        #     # OP1 HTIMKO 2021-02: Acts on the new 'long' version: note dV_Hcav_coarse_prev_long in prev:
        #     self.dV_Hcav_coarse_long_1 = moving_average(self.dV_mod_coarse_long, self.n_mov_av_coarse, x_prev=self.dV_Hcav_coarse_prev_long[-self.n_mov_av_coarse+1:])
        # elif self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
        #     # OPT2  MY TEST: Keeps a dV_mod_coarse_prev_long in prev (instead of dV_Hcav_coarse_prev_long):
        #     self.dV_Hcav_coarse_long_2 = moving_average(self.dV_mod_coarse_long, self.n_mov_av_coarse, x_prev=self.dV_mod_coarse_prev_long[-self.n_mov_av_coarse+1:])


        if self.dV_Hcav_opt == '0':
            # OPT0: The signal's lengths is alreadu n_coarse
            pass
        # elif self.dV_Hcav_opt == '1':
        #     # OP1 HTIMKO 2021-02: Extract the signal at the current turn only (n_coarse length)
        #     self.dV_Hcav_coarse_1  = self.dV_Hcav_coarse_long_1[:-self.n_mov_av_coarse+1]
        # elif self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
        #     # OPT2 MY TEST: Done as above
        #     self.dV_Hcav_coarse_2a = self.dV_Hcav_coarse_long_2[:-self.n_mov_av_coarse+1]
        #     if hasattr(self, 'dV_Hcav_coarse_2b'):
        #         self.dV_Hcav_coarse_2b = np.concatenate((self.dV_Hcav_coarse_long_2[-self.n_mov_av_coarse+1:], self.dV_Hcav_coarse_long_2[self.n_mov_av_coarse-1:-self.n_mov_av_coarse+1]))
        #     else:
        #         self.dV_Hcav_coarse_2b = np.copy(self.dV_Hcav_coarse_2a)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # if self.use_gen_fine:
        #     self.dV_mod_fine_prev = np.copy(self.dV_mod_in_fine)
        if self.dV_Hcav_opt == '0':
            # OPT0
            self.dV_mod_coarse_prev   = np.copy(self.dV_mod_in_coarse)
        # elif self.dV_Hcav_opt == '1':
        #     # OPT1 HTIMKO 2021-02: Instead of the lines above, this:
        #     self.dV_Hcav_coarse_prev_long = self.dV_Hcav_coarse_long_1[-self.n_mov_av_coarse+1:]
        # elif self.dV_Hcav_opt == '2a' or self.dV_Hcav_opt == '2b':
        #     # OPT2 MY TEST:
        #     self.dV_mod_coarse_prev_long  = self.dV_mod_in_long_coarse[-self.n_mov_av_coarse+1:]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Continue to next turn with the appropiate signal:

        if self.dV_Hcav_opt == '0':
            self.dV_Hcav_coarse = np.copy(self.dV_Hcav_coarse_0)
        elif self.dV_Hcav_opt == '1':
            self.dV_Hcav_coarse = np.copy(self.dV_Hcav_coarse_1)
        elif self.dV_Hcav_opt == '2a':
            self.dV_Hcav_coarse = np.copy(self.dV_Hcav_coarse_2a)
        elif self.dV_Hcav_opt == '2b':
            self.dV_Hcav_coarse = np.copy(self.dV_Hcav_coarse_2b)


    def matr_conv(self, I, h):
        """Convolution of beam current with impulse response; uses a complete
        matrix with off-diagonal elements."""

        use_res = 1

        # SCIPY routines:

        if use_res == 1:
            res1 = scipy.signal.fftconvolve(I, h, mode='full')[:I.shape[0]] # ORIGINAL: FAST even in fine but large numeric error in real part of V_ind = I conv h,
            # print(I, I.shape[0], I[int(0.50*self.n_coarse)], np.angle(I[int(0.50*self.n_coarse)], deg=True))
            # print(h, h.shape[0], h[int(0.50*self.n_coarse)], np.angle(h[int(0.50*self.n_coarse)], deg=True))
            # print(res1, res1.shape, res1[int(0.50*self.n_coarse)], np.angle(res1[int(0.50*self.n_coarse)], deg=True))
            # quit()

        elif use_res == 2:
            res2 = scipy.signal.convolve(I, h, mode='full', method='fft')[:I.shape[0]] # EQUIVALENT TO ORIGINAL: FAST but large numeric error in real part of V_ind = I conv h
            # # res12 = res1/res2
            # # print('conv:')
            # # print('1:  ', res1,  ', midnobeam =',  res1[int(0.5*self.n_coarse)], '->', np.abs( res1[int(0.5*self.n_coarse)]), 'V', np.angle( res1[int(0.5*self.n_coarse)], deg=True), 'deg')
            # # print('2:  ', res2,  ', midnobeam =',  res2[int(0.5*self.n_coarse)], '->', np.abs( res2[int(0.5*self.n_coarse)]), 'V', np.angle( res2[int(0.5*self.n_coarse)], deg=True), 'deg')
            # # print('1/2:', res12, ', midnobeam =', res12[int(0.5*self.n_coarse)], '->', np.abs(res12[int(0.5*self.n_coarse)]), 'V', np.angle(res12[int(0.5*self.n_coarse)], deg=True), 'deg')

        elif use_res == 3:
            res3 = scipy.signal.convolve(I, h, mode='full', method='direct')[:I.shape[0]]  # FAST/SLOW in coarse/fine, Formal definition, the SLOWEST

        # NUMPY routines: always uses FFT:

        elif use_res == 4:
            res4 = np.convolve(I, h, mode='full')[:I.shape[0]]  # FAST/SLOW in coarse/fine, it returns the proper convolution (mirrored arround center) in and array with len = len(I)+len(M)-1. We then take just part we are interested / ~half (note boundary effects might be visible)
            # print(f'res4 = {res4}, shape = {res4.shape}')
            # res5 = np.convolve(I, h, mode='same')[:I.shape[0]]  # FAST, but NOT APPLICABLE anyway since convolution is SQUEEZED into an array with len = max(I,M) (note boundary effects might be visible)
            # print(f'res5 = {res5}, shape = {res5.shape}')
            # res6 = np.ones(I.shape[0]) * np.convolve(I, h, mode='valid')[0] # Only the center point (in res4), converted into an array
            # print(f'res6 = {res6}, shape = {res6.shape}')
            # quit()

        if   use_res == 1: return res1
        elif use_res == 2: return res2
        elif use_res == 3: return res3
        elif use_res == 4: return res4
        else: sys.error('\n[!] ERROR: Wrong matr_conv option!\n')

    def track(self):
        """Turn-by-turn tracking method."""

        self.in_track_beam = True

        # Update turn-by-turn variables
        # Compute internally the Q_gen_abs_max_coarse corresponding to
        # power_clamp_thresh, if applicable
        self.update_variables(use_power_clamp=self.power_clamp_thresh)

        # Update the impulse response at present carrier frequency
        # MYEDIT: 2020.12.14: Independent of the phase references of t_coarse
        # and t_fine, since they'll be shifted to zero:
        if self.use_gen_fine:
            self.TWC.impulse_response_gen(self.omega_c, self.t_fine)
            self.TWC.h_gen_fine = np.copy(self.TWC.h_gen) + 0+0j
            delattr(self.TWC, 'h_gen')
        self.TWC.impulse_response_gen(self.omega_c, self.t_coarse)  # self.rf_centers)
        self.TWC.h_gen_coarse = np.copy(self.TWC.h_gen) + 0+0j
        delattr(self.TWC, 'h_gen')
        #
        self.TWC.impulse_response_beam(self.omega_c, self.t_fine, self.t_coarse)  # self.profile.bin_centers, self.rf_centers)
        self.TWC.h_beam_fine = np.copy(self.TWC.h_beam) + 0+0j
        delattr(self.TWC, 'h_beam')

        # On current measured (I,Q) voltage, apply LLRF model
        self.set_point()
        self.llrf_model()

        if self.add_ff_to_gen:
            # Beam-induced voltage from beam profile
            # print(self.cavtype, self.counter)
            self.beam_induced_voltage(lpf=False)
            # Generator-induced voltage from generator current
            self.generator_induced_voltage()

        else:
            # Original
            # Generator-induced voltage from generator current
            self.generator_induced_voltage()
            # Beam-induced voltage from beam profile
            # print(self.cavtype, self.counter)
            self.beam_induced_voltage(lpf=False)

        # Sum and generator- and beam-induced voltages for coarse grid
        self.V_tot_coarse = np.copy(self.V_ind_beam_coarse) + np.copy(self.V_ind_gen_coarse)

        # Sum and generator- and beam-induced voltages for fine grid. Note:
        # Obtain generator-induced voltage on the fine grid by interpolation
        # MYEDIT: 2020.12.14: Important that both t_fine and t_coarse have the
        # same phase reference:
        if self.use_gen_fine:
            self.V_tot_fine = np.copy(self.V_ind_beam_fine) + np.copy(self.V_ind_gen_fine)
        else:
            self.V_tot_fine = np.copy(self.V_ind_beam_fine) \
                + np.interp(self.t_fine,   # self.profile.bin_centers,
                            self.t_coarse, # self.rf_centers,
                            self.V_ind_gen_coarse)

    def track_nollrf(self, V_part_rampfact_fine_i=None, V_part_rampfact_coarse_i=None):
        """Initial tracking method, before injecting beam, w/o llrf and generator
        current therefore straight from generator (total-beam) voltage and Zg """

        # Update turn-by-turn variables
        # Compute internally the Q_gen_abs_max_coarse corresponding to
        # power_clamp_thresh, if applicable
        self.update_variables(use_power_clamp=self.power_clamp_thresh)

        # Generate Z_gen (instead of impulse response)
        self.TWC.impedance_gen(self.t_coarse)
        self.Z_gen_coarse = np.copy(self.TWC.Z_gen)
        delattr(self.TWC, 'Z_gen')
        print(f'self.Z_gen_coarse = {self.Z_gen_coarse}, shape = {self.Z_gen_coarse.shape}')

        # Update the impulse response at present carrier frequency (beam only)
        self.TWC.impulse_response_beam(self.omega_c, self.t_fine, self.t_coarse)  # self.profile.bin_centers, self.rf_centers)
        self.TWC.h_beam_fine = np.copy(self.TWC.h_beam) + 0+0j
        delattr(self.TWC, 'h_beam')

        # Update set-point (no LLRF)
        self.set_point(V_part_rampfact_fine_i, V_part_rampfact_coarse_i)

        # Beam-induced voltage from beam profile
        self.beam_induced_voltage(lpf=False)

        # Assuming perfect feedback compensation, the total voltage is equal to
        # the set point voltage
        self.V_tot_coarse = np.copy(self.V_set_coarse)
        self.V_tot_fine   = np.interp(self.t_fine,
                                      self.t_coarse,
                                      self.V_tot_coarse)

        # With beam, generator si the difference between the total and beam-induced voltage
        self.V_ind_gen_coarse = self.V_tot_coarse  - self.V_ind_beam_coarse

        # Generator current from generator voltage (i.e. diff total-beam voltage when beam)
        # Opt A:
        self.Q_gen_coarse = np.ifft( np.fft(self.V_ind_gen_coarse) / self.Z_gen_coarse )
        print(f'self.Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse.shape}')
        # Opt B:
        self.Q_gen_coarse = self.V_ind_gen_coarse / np.fft.ifft(1./self.Z_gen_coarse)
        print(f'self.Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse.shape}')


    def track_no_beam(self, V_part_rampfact_fine_i=None, V_part_rampfact_coarse_i=None):
        """Initial tracking method, before injecting beam."""
        ### TAKES AROUND 0.004 s (and 0.14 s w/ plot)

        # Update turn-by-turn variables
        # Power clamp does not apply in pretracking
        self.update_variables(use_power_clamp=False)

        # Update the impulse response at present carrier frequency
        # MYEDIT: 2020.12.4: Independent of the phase reference of t_coarse since
        # it will be shifted to zero:

        #print(f'self.TWC.h_gen = {self.TWC.h_gen}, shape = {self.TWC.h_gen.shape}')
        # self.TWC.impulse_response_gen(self.omega_c, self.t_coarse) #self.rf_centers)
        if self.use_gen_fine:
            self.TWC.impulse_response_gen(self.omega_c, self.t_fine)
            self.TWC.h_gen_fine = np.copy(self.TWC.h_gen) + 0+0j
            delattr(self.TWC, 'h_gen')
        self.TWC.impulse_response_gen(self.omega_c, self.t_coarse)
        self.TWC.h_gen_coarse = np.copy(self.TWC.h_gen) + 0+0j
        delattr(self.TWC, 'h_gen')

        # print(f'self.TWC.h_gen = {self.TWC.h_gen}, shape = {self.TWC.h_gen.shape}')
        # print(f'self.TWC.h_gen[int(0.5*self.n_coarse)] = {self.TWC.h_gen[int(0.5*self.n_coarse)]}')
        # print(f'self.TWC.h_gen[self.n_coarse-1] = {self.TWC.h_gen[self.n_coarse-1]}')

       #self.TWC.h_gen += 0+0j
        # print(f'self.TWC.h_gen = {self.TWC.h_gen}, shape = {self.TWC.h_gen.shape}')
        # print(f'self.TWC.h_gen[int(0.5*self.n_coarse)] = {self.TWC.h_gen[int(0.5*self.n_coarse)]}')
        # print(f'self.TWC.h_gen[self.n_coarse-1] = {self.TWC.h_gen[self.n_coarse-1]}')

        # On current measured (I,Q) voltage, apply LLRF model
        self.set_point(V_part_rampfact_fine_i, V_part_rampfact_coarse_i)
        self.llrf_model()

        self.counter_pretrack += 1 # Only update after llrf so that first turn has phi_mod_0 = 0

        # Generator-induced voltage from generator current (totals, divide by
        # n_cavities for single cavity)
        self.generator_induced_voltage()
        self.logger.debug("Total voltage to generator %.3e V",
                          np.mean(np.absolute(self.V_gen_coarse)))
        self.logger.debug("Total current from generator %.3e A",
                          np.mean(np.absolute(self.Q_gen_coarse))
                          / self.T_s_coarse)

        # Without beam, total voltage equals generator-induced voltage
        if self.use_gen_fine:
            self.V_tot_fine   = np.copy(self.V_ind_gen_fine)
        self.V_tot_coarse = np.copy(self.V_ind_gen_coarse)

        self.logger.debug(
            "Average generator voltage, last half of array %.3e V",
            np.mean(np.absolute(self.V_ind_gen_coarse[int(0.5*self.n_coarse):])))

    def track_no_beam_nollrf(self, V_part_rampfact_fine_i=None, V_part_rampfact_coarse_i=None):
        """Initial tracking method, before injecting beam, w/o llrf and generator
        current therefore straight from generator (total) voltage and Zg """

        # Update turn-by-turn variables
        # Power clamp does not apply in pretracking
        self.update_variables(use_power_clamp=False)

        # From TotalInducedVoltage
        n_fft = 300000
        # * freq = [0.00000000e+00 4.27508267e+04 8.55016533e+04 ... 6.41253850e+09 6.41258125e+09 6.41262400e+09]
        frequency_resolution = 42750.82666666667
        freq = np.linspace(0.0, n_fft*frequency_resolution, n_fft, endpoint=True)

        # Generate Z_gen (instead of impulse response)
        self.TWC.impedance_gen(omega_array=2*np.pi*freq) #self.t_fine)
        self.Z_gen_coarse = np.copy(self.TWC.Z_gen)
        delattr(self.TWC, 'Z_gen')
        print(f'self.Z_gen_coarse = {self.Z_gen_coarse}, shape = {self.Z_gen_coarse.shape}')

        fig, ax = plt.subplots()
        ax.plot(self.Z_gen_coarse, label='Z_g_coarse')
        plt.show()

        quit()

        # Update set-point (no LLRF)
        self.set_point(V_part_rampfact_fine_i, V_part_rampfact_coarse_i)
        print(f'self.V_set_coarse = {self.V_set_coarse}, shape = {self.V_set_coarse.shape}')

        # Assuming perfect feedback compensation, the total voltage is equal to
        # the set point voltage
        self.V_tot_coarse = np.copy(self.V_set_coarse)
        self.V_tot_fine   = np.interp(self.t_fine,
                                      self.t_coarse,
                                      self.V_tot_coarse)
        print(f'self.V_tot_coarse = {self.V_tot_coarse}, shape = {self.V_tot_coarse.shape}')
        print(f'self.V_tot_fine = {self.V_tot_fine}, shape = {self.V_tot_fine.shape}')

        # Without beam, total voltage comes entirely from the generator
        self.V_ind_gen_coarse = np.copy(self.V_tot_coarse)
        print(f'self.V_ind_gen_coarse = {self.V_ind_gen_coarse}, shape = {self.V_ind_gen_coarse.shape}')

        # Generator current from generatpr voltage (i.e. total voltage when no beam)
        # Opt A:
        self.Q_gen_coarse = np.fft.ifft( np.fft.fft(self.V_ind_gen_coarse) / self.Z_gen_coarse )
        print(f'self.Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse.shape}')
        # Opt B:
        self.Q_gen_coarse = self.V_ind_gen_coarse / np.fft.ifft(1./self.Z_gen_coarse)
        print(f'self.Q_gen_coarse = {self.Q_gen_coarse}, shape = {self.Q_gen_coarse.shape}')
        quit()


    def update_variables(self,use_power_clamp=False):
        '''Update counter and frequency-dependent variables in a given turn'''

        # Present time step
        self.counter = self.rf.counter[0]
        # Present carrier frequency: main RF frequency
        self.omega_c = self.rf.omega_rf[0,self.counter]

        # Present sampling time
        self.T_s_fine   = self.profile.bin_size # always needed even if not self.use_gen_fine
        self.T_s_coarse = self.rf.t_rf[0,self.counter]

        # Phase offset at the end of a 1-turn modulated/demodulated signal (for demoluated, multiply by -1 as c and r reversed)
        self.phi_mod_0 = (self.omega_c - self.omega_r)*self.T_s_coarse * (self.n_coarse-1)
        # print(self.phi_mod_0)
        # quit()

        # Power clamp:
        # Pgcav = 0.5 * Z0 * |Igcav|^2
        # Igcav = np.sqrt( 2 Pgcav / Z0 )
        # Qgtot = Qgcav * ncav
        # Qgtot = (Igcav * bs)/nbs * ncav
        # but bs = nbs * Ts
        # Qgtot = (Igcav * nbs * Ts)/nbs * ncav
        # Qgtot = Igcav*Ts * ncav
        # Igcav = Qgtot/Ts / ncav
        # Pgcav = 0.5 * Z0 * | Qgtot/Ts / ncav |^2

        # if self.use_gen_fine:
        # # Clamp in total current based on power clamp per cavity (fine)
        # if use_power_clamp:
        #     self.Q_gen_abs_max_fine   = np.sqrt(2 * self.power_clamp_thresh / self.TWC.Z_0) *  self.T_s_fine   * self.n_cavities
        # else:
        #     self.Q_gen_abs_max_fine   = False
        # # Clamp in total current based on power clamp per cavity (coarse)
        if use_power_clamp and not self.Q_gen_abs_max_coarse:
            # Calculate the clamp threshold in current (from the threshold in power)
            # only once (not changing):
            # # First try:
            # self.I_gen_abs_max_coarse = np.sqrt(2 * self.power_clamp_thresh / self.TWC.Z_0) * (self.nbs * self.T_s_coarse)  # In physical current (per cav, the same for all) [A]
            # self.Q_gen_abs_max_coarse = self.I_gen_abs_max_coarse/self.nbs * self.n_cavities                                # In instantaneous coarse charge (total all cavs) [C]
            self.I_gen_abs_max_coarse = np.sqrt(2 * self.power_clamp_thresh / self.TWC.Z_0)                           # In physical current [A]
            self.Q_gen_abs_max_coarse = self.I_gen_abs_max_coarse * self.bunch_spacing / self.nbs * self.n_cavities   # In instantaneous coarse charge [C]
            print(f'I_gen_abs_max_coarse = {self.I_gen_abs_max_coarse} A')
            print(f'Q_gen_abs_max_coarse = {self.Q_gen_abs_max_coarse/1e-9} nC')

        # MYEDIT: 2020.12.14: t_fine and t_coarse (rename of rf_centers)
        # definitions and possible phase reference corrections:
        self.t_fine = np.copy(self.profile.bin_centers)
        self.t_coarse = (np.arange(self.n_coarse) + 0.5) * self.T_s_coarse # renamed from self.rf_centers
        #
        if self.dphi_rf_opt == 0:
            # Correct t_fine to have phase reference zero since profile.bin_centers
            # might include an offset due to dphi_rf. t_coarse is already defined
            # w.r.t. phase reference zero:
            #self.t_fine += self.rf.dphi_rf[0]/2./np.pi*self.T_s_coarse
            # TODO
            pass
           #self.t_fine -= self.t_fine[0] #- 0.5*(self.t_fine[1]-self.t_fine[0])
        elif self.dphi_rf_opt == 1:
            # Correct t_coarse to have the same phase offset due to dphi_rf that
            # profile.bin_centers might have. t_fine is already defined with
            # this same offset:
            self.t_coarse -= self.rf.dphi_rf[0]/2./np.pi*self.T_s_coarse
        if self.open_FF == 1:
            self.t_coarseFF = self.t_coarse[self.indices_coarseFF]

        # In both cases (for a "V_gen_fine" and for "Q_gen_coarse_coarse", respectively):
        self.t_fine_long   = np.concatenate( (self.t_fine,   self.t_fine[-1]   + np.arange(1, self.n_fine_long   - self.n_fine   + 1) * self.T_s_fine  ) )
        self.t_coarse_long = np.concatenate( (self.t_coarse, self.t_coarse[-1] + np.arange(1, self.n_coarse_long - self.n_coarse + 1) * self.T_s_coarse) )

        # print(f't_fine = {self.t_fine}, shape = {self.t_fine.shape}')
        # print(f't_fine_long = {self.t_fine_long}, shape = {self.t_fine_long.shape}')
        # print(f't_coarse = {self.t_coarse}, shape = {self.t_coarse.shape}')
        # print(f't_coarse_long = {self.t_coarse_long}, shape = {self.t_coarse_long.shape}')
        #quit()

        # Present coarse grid
        #if self.dphi_rf_opt == 0: # MYEDIT: 2020.12.09
        #    # Keep the rf_centers w.r.t. to a profile starting at zero
        #    self.profile.cut_left    -= self.rf.dphi_rf[0]/2./np.pi* self.T_s_coarse
        #    self.profile.cut_right   -= self.rf.dphi_rf[0]/2./np.pi* self.T_s_coarse
        #    self.profile.edges       -= self.rf.dphi_rf[0]/2./np.pi* self.T_s_coarse
        #    self.profile.bin_centers -= self.rf.dphi_rf[0]/2./np.pi* self.T_s_coarse
        #    self.rf_centers = (np.arange(self.n_coarse) + 0.5) * self.T_s_coarse
        #else:

        # Check number of samples required per turn
        if self.use_gen_fine:
            n_fine = int(self.rf.t_rev[self.counter]/self.T_s_fine)
            if self.n_fine != n_fine:
                raise RuntimeError("Error in SPSOneTurnFeedback: changing number" +
                    " of fine samples. This option is not yet implemented!")
        n_coarse = int(self.rf.t_rev[self.counter]/self.T_s_coarse)
        if self.n_coarse != n_coarse:
            raise RuntimeError("Error in SPSOneTurnFeedback: changing number" +
                " of coarse samples. This option is not yet implemented!")

        # VS HTIMKO 2021-02: Added to match it
        # Present moving average window
        if self.use_gen_fine:
            self.n_mov_av_fine = int(np.rint(self.TWC.tau/self.T_s_fine  ))
        self.n_mov_av_coarse   = int(np.rint(self.TWC.tau/self.T_s_coarse))

        # Present delay time = This is the ONE TURN in the "one turn delay feedback (OTFB)", although
        # it is not exactly one turn, as we substract the TWC filling time thus counting only the
        # necessary time to make up one turn
        if self.use_gen_fine:
            self.n_delay_fine = self.n_fine   - self.n_mov_av_fine   # VS HTIMKO 2021-02: changing to new definition. Old: int((self.rf.t_rev[self.counter] - self.TWC.tau) / self.T_s_fine  )
        self.n_delay_coarse   = self.n_coarse - self.n_mov_av_coarse # VS HTIMKO 2021-02: changing to new definition. Old: int((self.rf.t_rev[self.counter] - self.TWC.tau) / self.T_s_coarse)



#    def pre_compute_semi_analytic_factor(self, time):
#        r""" Pre-computes factor for semi-analytic method, which is used to
#        compute the beam-induced voltage on the coarse grid.
#
#        Parameters
#        ----------
#        time : float array [s]
#            Time array at which to compute the beam-induced voltage
#
#        Attributes
#        ----------
#        profile_coarse : class
#            Beam profile with 20 bins per RF-bucket
#        semi_analytic_factor : complex array [:math:`\Omega\,s`]
#            Factor that is used to compute the beam-induced voltage
#        """
#
#        self.logger.info("Pre-computing semi-analytic factor")
#
#        n_slices_per_bucket = 20
#
#        n_buckets = int(np.round(
#            (self.profile.cut_right - self.profile.cut_left)
#            / self.rf.t_rf[0, 0]))
#
#        self.profile_coarse = Profile(self.beam, CutOptions=CutOptions(
#            cut_left=self.profile.cut_left,
#            cut_right=self.profile.cut_right,
#            n_slices=n_buckets*n_slices_per_bucket))
#
#        # pre-factor [Ohm s]
#
#        pre_factor = 2*self.TWC.R_beam / self.TWC.tau**2 / self.omega_r**3
#
#        # Matrix of time differences [1]
#        dt1 = np.zeros(shape=(len(time), self.profile_coarse.n_slices))
#
#        for i in range(len(time)):
#            dt1[i] = (time[i] - self.profile_coarse.bin_centers) * self.omega_r
#
##        dt2 = dt1 - self.TWC.tau * self.omega_r
#
##        phase1 = np.exp(-1j * dt1)
#        phase = np.exp(-1j * self.TWC.tau * self.TWC.omega_r)
#
##        diff1 = 2j - dt1 + self.TWC.tau * self.omega_r
#
##        diff2 = (2j - dt1 + self.TWC.tau * self.omega_r) * np.exp(-1j * dt1)
#
#        tmp = (-2j - dt1 + self.TWC.tau*self.omega_r
#               + (2j - dt1 + self.TWC.tau*self.omega_r) * np.exp(-1j * dt1))\
#            * np.sign(dt1) \
#            - ((2j - dt1 + self.TWC.tau * self.omega_r) * np.exp(-1j * dt1)
#               + (-2j - dt1 + self.TWC.tau * self.omega_r) * phase) \
#            * np.sign(dt1 - self.TWC.tau * self.omega_r) \
#            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)
#
##        tmp = (-2j - dt1 + self.TWC.tau*self.omega_r + diff2) * np.sign(dt1) \
##            - (diff2 + (-2j - dt1 + self.TWC.tau * self.omega_r) * phase) \
##                * np.sign(dt1 - self.TWC.tau * self.omega_r) \
##            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)
#
##        tmp = (diff1.conjugate() + diff2) * np.sign(dt1) \
##            - (diff2 + diff1.conjugate() * phase) \
##                * np.sign(dt1 - self.TWC.tau * self.omega_r) \
##            - (2 - 1j*dt1) * self.TWC.tau * self.TWC.omega_r * np.sign(dt1)
#
#        tmp *= pre_factor
#
#        self.semi_analytic_factor = np.diff(tmp)
#
#    def beam_induced_voltage_semi_analytic(self):
#        r"""Computes the beam-induced voltage in (I,Q) at the present carrier
#        frequency :math:`\omega_c` using the semi-analytic method. It requires
#        that pre_compute_semi_analytic_factor() was called previously.
#
#        Returns
#        -------
#        complex array [V]
#            Beam-induced voltage in (I,Q) at :math:`\omega_c`
#        """
#
#        # Update the coarse profile
#        self.profile_coarse.track()
#
#        # Slope of line segments [A/s]
#        kappa = self.beam.ratio*self.beam.Particle.charge*e \
#            * np.diff(self.profile_coarse.n_macroparticles) \
#            / self.profile_coarse.bin_size**2
#
#        return np.exp(1j*self.rf_centers*self.omega_c)\
#            * np.sum(self.semi_analytic_factor * kappa, axis=1)
#
#