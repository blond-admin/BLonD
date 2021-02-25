#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:31:00 2019

@author: medinamluis
"""

# General imports
# ----------------
from __future__ import division, print_function
import numpy as np
from copy import copy

from scipy.constants import c

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from blond.impedances.impedance_sources import TravelingWaveCavity

##########################

# Based on Joel's SPS_tools (in Joel_simulations directory)

# Definition of the total phase slip tau
# ---------------------------------------

def taum(f,fr,L,v,vg):
    return L/vg*(1.-vg/v)*2*np.pi*(f-fr)

def taup(f,fr,L,v,vg):
    return L/vg*(1.+vg/v)*2*np.pi*(f+fr)

# Definition of the cavity impedance
# ----------------------------------

def ReZrfm(f,fr,Z_0,rho,L,vg):
    return L*np.sqrt(rho*Z_0/2.) * np.sin(taum(f,fr,L,c,vg)/2.)/(taum(f,fr,L,c,vg)/2.) * np.heaviside( f, 0.5) # positive frequencies

def Zrfm(f,fr,Z_0,rho,L,vg):
    return ReZrfm(f,fr,Z_0,rho,L,vg)

def ReZrfp(f,fr,Z_0,rho,L,vg):
    return L*np.sqrt(rho*Z_0/2.) * np.sin(taup(f,fr,L,c,vg)/2.)/(taup(f,fr,L,c,vg)/2.) * np.heaviside(-f, 0.5) # negative frequencies

def Zrfp(f,fr,Z_0,rho,L,vg):
    return ReZrfp(f,fr,Z_0,rho,L,vg)

def ReZrf(f,fr,Z_0,rho,L,vg):
    return ReZrfm(f,fr,Z_0,rho,L,vg) + ReZrfp(f,fr,Z_0,rho,L,vg)

def Zrf(f,fr,Z_0,rho,L,vg):
    return Zrfm(f,fr,Z_0,rho,L,vg) + Zrfp(f,fr,Z_0,rho,L,vg)


# Definition of the impedance seen by the beam
# ---------------------------------------------

def ReZbm(f,fr,Z_0,rho,L,vg):
    return   ((rho*L**2)/8.) * (np.sin(taum(f,fr,L,c,vg)/2.)/(taum(f,fr,L,c,vg)/2.))**2 * np.heaviside( f, 0.5) # positive frequencies

def ReZbp(f,fr,Z_0,rho,L,vg):
    return   ((rho*L**2)/8.) * (np.sin(taup(f,fr,L,c,vg)/2.)/(taup(f,fr,L,c,vg)/2.))**2 * np.heaviside(-f, 0.5) # negative frequencies

def ReZb(f,fr,Z_0,rho,L,vg):
    return ReZbm(f,fr,Z_0,rho,L,vg) + ReZbp(f,fr,Z_0,rho,L,vg)

def ImZbm(f,fr,Z_0,rho,L,vg):
    return - ((rho*L**2)/8.) * 2* (taum(f,fr,L,c,vg)-np.sin(taum(f,fr,L,c,vg)))/(taum(f,fr,L,c,vg)**2) * np.heaviside( f, 0.5) # positive frequencies

def ImZbp(f,fr,Z_0,rho,L,vg):
    return - ((rho*L**2)/8.) * 2* (taup(f,fr,L,c,vg)-np.sin(taup(f,fr,L,c,vg)))/(taup(f,fr,L,c,vg)**2) * np.heaviside(-f, 0.5) # negative frequencies

def ImZb(f,fr,Z_0,rho,L,vg):
    return ImZbm(f,fr,Z_0,rho,L,vg) + ImZbp(f,fr,Z_0,rho,L,vg)

def Zbm(f,fr,Z_0,rho,L,vg):
    return ReZbm(f,fr,Z_0,rho,L,vg) + 1j* ImZbm(f,fr,Z_0,rho,L,vg) # The real (imaginary) part has a positive (negative) sign internally

def Zbp(f,fr,Z_0,rho,L,vg):
    return ReZbp(f,fr,Z_0,rho,L,vg) + 1j* ImZbp(f,fr,Z_0,rho,L,vg) # The real (imaginary) part has a positive (negative) sign internally

def Zb(f,fr,Z_0,rho,L,vg):
    return Zbm(f,fr,Z_0,rho,L,vg) + Zbp(f,fr,Z_0,rho,L,vg)  # The real (imaginary) part has a negative (postive) sign internally

def reduce_impedance_feedforward_feedback(profile, freqres, impedanceScenario, outdir=None, Gfb=10.0, gff=0.5): # Changed default from Gfb = 7.5 to 10. on 05.Aug.2020

    if type(Gfb) is not list:
        if(Gfb == 0.): Gfblist = [None, None]
        else:          Gfblist = [Gfb,  Gfb ]
    else:
        # It comes already in a list form:
        # For past/present imp: [4-sec cavs, 5-sec cavs]
        # For future       imp: [3-sec cavs, 4-sec cavs]
        Gfblist = copy(Gfb)
        pass
    
    if type(gff) is not list:
        if(gff == 0.): gfflist = [None, None]
        else:          gfflist = [gff,  gff ]
    else:
        # For past/present imp: [4-sec cavs, 5-sec cavs]
        # For future       imp: [3-sec cavs, 4-sec cavs]
        gfflist = copy(gff)
        pass
    
    ##gfb = # 1/(4.*50)*10. #4.*50. * 0.1
    #gfb = 1e-2 #1.5e-2 #0.86e6/2. # 1e-3 # None #1e-6 # with Zr
    
    #print(f'impedanceScenario = {impedanceScenario}')
    print(f'impedanceScenario.scenarioFileName = {impedanceScenario.scenarioFileName}')
    
    if('future' in impedanceScenario.scenarioFileName):
        Gfb3or5sec = Gfblist[0]
        Gfb4sec    = Gfblist[1]
        print('* Gfb3sec =', Gfb3or5sec, '(Gfb3or5sec)')
        print('* Gfb4sec =', Gfb4sec)
        gff3or5sec = gfflist[0]
        gff4sec    = gfflist[1]
        print('* gff3sec =', gff3or5sec, '(gff3or5sec)')
        print('* gff4sec =', gff4sec)
    else:
        Gfb3or5sec = Gfblist[1]
        Gfb4sec    = Gfblist[0]
        print('* Gfb5sec =', Gfb3or5sec, '(Gfb3or5sec)')
        print('* Gfb4sec =', Gfb4sec)
        gff3or5sec = gfflist[1]
        gff4sec    = gfflist[0]
        print('* gff5sec =', gff3or5sec, '(gff3or5sec)')
        print('* gff4sec =', gff4sec)
    print('')

    print('Running reduce_impedance_feedforward...\n')

    # Frequency array

    freq_res = freqres/10 #150e3
    n_fft    = int(1./(profile.bin_size)/freq_res)
    print('n_fft      =', n_fft)

    profile.beam_spectrum_freq_generation(n_fft)
    freq_array = profile.beam_spectrum_freq
    print('freq_array =', freq_array, len(freq_array)) # it will not be exactly equal than the frequency array to be computed in the profile
    print('')
    #quit()

    # Find the index of the main harmonic cavities in the impedance list using their source file names as keys
    # (the necessary attenuations (e.g. from FB) and corrections have already been performed upon loading):

    key3 = 'cavities/200MHz/3sections/TWC200_3sections_dome_MAIN.dat'
    key4 = 'cavities/200MHz/4sections/TWC200_4sections_dome_MAIN.dat'
    key5 = 'cavities/200MHz/5sections/TWC200_5sections_dome_MAIN.dat'

    elID_dict = {} # List of indices of the main cavities to be "erased" and replaced
    for i in range(len(impedanceScenario.SPSimpList)):
        filei = impedanceScenario.SPSimpList[i]['file']
        if  (i  < 15 or i == len(impedanceScenario.SPSimpList)-1): print(i, filei)
        elif(i == 15):                                            print('...')
        if(filei == key3): elID_dict['3sections'] = i # i = elIDkey3
        if(filei == key4): elID_dict['4sections'] = i # i = elIDkey4
        if(filei == key5): elID_dict['5sections'] = i # i = elIDkey5
    print('')
    print('Elements being replaced:')
    print('elID_dict =', elID_dict)
    print('')

    # Parameters for the cavities

    Z_0 = 50               # Characteristic impedance of the RF chain [Ohm]
    oneCellLength = 0.374  # Length of one cell [m] = 0.374 m
    rho = 27.1e3           # Series impedance of the cavity [Ohm/m^2]
    vg = 0.0946*c          # Group velocity [m/s]

    elID_list = []

    for key in elID_dict.keys():

        elID = elID_dict[key]
        elID_list.append(elID)
        print('elID =', elID, ' | key =', key, '|', impedanceScenario.table_impedance[elID]['file'])
        print('')

        if(key == '4sections'):

            nsections = 4
            ncavities = 2
            print('nsections = ', nsections)
            print('ncavities = ', ncavities)
            print('')

            L4sec = oneCellLength*(nsections*11 -0.5-0.5)  # Length of the 4-section 200MHz cavities (43 cells) [m]
            print('L4sec =', L4sec)
            print('')

            fr4sec    = impedanceScenario.table_impedance[elID]['fr'][0,0]             # Centre frequency of the 4-section 200MHz cavities [Hz]
            Rsh4sec   = impedanceScenario.table_impedance[elID]['Rsh'][0,0]/ncavities  # Shunt impedance [Ohm] # = (rho*L4sec**2)/8., the shunt impedance? [Ohm] -> works
            alpha4sec = impedanceScenario.table_impedance[elID]['alpha'][0,0]          # Time factor alpha [s] # = L4sec/vg* 0.5*( (1.-vg/c) + (1.+vg/c)) *2*np.pi, the daming time [s] (average of 1.-vg/c and 1.+vg/c):
            rho4sec   = 8.*Rsh4sec/L4sec**2
            print('fr4sec      =', fr4sec)
            print('Rsh4sec     =', Rsh4sec/1e6, 'MOhm')
            print('alpha4sec   =', alpha4sec/1e-6, 'us')
            print('rho4sec     =', rho4sec/1e3, 'kOhm/m2')
            print('rho4sec/rho =', rho4sec/rho)
            print('')

            # For comparison: creating TWCs with the parameters derived from the loaded input using BLonD's TWC object:
            twc4sec = TravelingWaveCavity(Rsh4sec, fr4sec, alpha4sec)
            twc4sec.imped_calc(freq_array)
            Ztwc4sec = twc4sec.impedance
            print('Ztwc4sec =', Ztwc4sec) # ~ Zb4sec
            print('')

            # Creating TWCs, characterised by their impedance (Zb); creating RF impedance (Zrf)

            params0 = [fr4sec, Z_0, rho4sec, L4sec, vg]

            ReZb4sec = ReZb(freq_array,*params0)
            Zb4sec   =   Zb(freq_array,*params0)
            print('ReZb4sec =', ReZb4sec, '\n', max(np.abs(np.real(ReZb4sec))), max(np.abs(np.imag(ReZb4sec))), max(np.abs(ReZb4sec)))
            print('Zb4sec   =', Zb4sec,   '\n', max(np.abs(np.real(Zb4sec))),   max(np.abs(np.imag(Zb4sec))),   max(np.abs(Zb4sec)))
            print('')
            ReZrf4sec = ReZrf(freq_array,*params0)
            Zrf4sec   =   Zrf(freq_array,*params0)
            print('ReZrf4sec =', ReZrf4sec, '\n', max(np.abs(ReZrf4sec)))
            print('Zrf4sec   =', Zrf4sec,   '\n', max(np.abs(Zrf4sec)))
            print('')

            Zrf4secnew = np.copy(Zb4sec)

            # FF reduction
            if(gff4sec is not None):
                hff4sec = gff4sec / (4.*Z_0)
                ReHff4sec = hff4sec * ReZrf4sec
                ImHff4sec = 0.
                Hff4sec   = ReHff4sec + 1j * ImHff4sec
                Zrf4secnew -= Hff4sec * Zrf4sec
                print('gff4sec =', gff4sec)
                print('hff4sec =', hff4sec)
                print('Hff4sec =', Hff4sec)
            else:
                print('No FF')
            print('Zrf4secnew =', Zrf4secnew, '\n', max(np.abs(np.real(Zrf4secnew))), max(np.abs(np.imag(Zrf4secnew))), max(np.abs(Zrf4secnew)))
            print('')

            # FB reduction (acting on top of the FF-reduction, if applicable)
            #if(gfb is not None):
            if(Gfb4sec is not None):
            #    gfb4sec = Gfb4sec * ((rho4sec*L4sec**2)/8.)
            #    hfb4sec = gfb4sec / ((rho4sec*L4sec**2)/8.)
            #    Hfb4sec = hfb4sec * ReZb4sec          # [1]
            #    #gfb4sec = Gfb4sec * (L4sec*np.sqrt(rho4sec*Z_0/2.))
            #    #hfb4sec = gfb4sec / (L4sec*np.sqrt(rho4sec*Z_0/2.))
            #    #Hfb4sec = hfb4sec * ReZrf4sec         # [1]
            #   #Zrf4secnew /= 1.  + Hfb4sec*Zrf4sec
            #   #Zrf4secnew /= Zb4sec + Hfb4sec*Zrf4sec
            #    Hfb4sec = Gfb4sec / (4.*Z_0)
            #    Zrf4secnew /= 1.  + Hfb4sec*Hff4sec*Zrf4sec
            #   #Zrf4secnew /= Zb4sec + Hfb4sec*Hff4sec*Zrf4sec
            #   #Zrf4secnew /= 1.  + Hfb4sec*Zrf4secnew
            #   #Zrf4secnew /= Zb4sec + Hfb4sec*Zrf4secnew
            #    print('Gfb4sec =', Gfb4sec)
            #    print('gfb4sec =', gfb4sec)
            #    print('hfb4sec =', hfb4sec)
            #    print('Hfb4sec =', Hfb4sec)
                Hfb4secZrf = Gfb4sec * ReZb4sec / ((rho4sec*L4sec**2)/8.)
                Zrf4secnew /= 1.  + Hfb4secZrf
                print('Hfb4secZrf =', Hfb4secZrf)
            else:
                print('No FB')
            print('Zrf4secnew =', Zrf4secnew, '\n', max(np.abs(np.real(Zrf4secnew))), max(np.abs(np.imag(Zrf4secnew))), max(np.abs(Zrf4secnew)))
            print('')

        elif(key == '3sections' or key == '5sections'):

            if(  key == '3sections'):
                nsections = 3
                ncavities = 4
            elif(key == '5sections'):
                nsections = 5
                ncavities = 2
            print('nsections = ', nsections)
            print('ncavities = ', ncavities)
            print('')

            L3or5sec = oneCellLength*(nsections*11 -0.5-0.5)  # Length of the (3/5)-section cavities (32/54 cells) [m]
            print('L3or5sec =', L3or5sec)
            print('')

            fr3or5sec    = impedanceScenario.table_impedance[elID]['fr'][0,0]             # Centre frequency of the 200MHz cavities [Hz]
            Rsh3or5sec   = impedanceScenario.table_impedance[elID]['Rsh'][0,0]/ncavities  # Shunt impedance [Ohm] # = (rho*L3or5sec**2)/8., the shunt impedance? [Ohm] -> works
            alpha3or5sec = impedanceScenario.table_impedance[elID]['alpha'][0,0]          # Time factor alpha [s] # = L3or5sec/vg* 0.5*( (1.-vg/c) + (1.+vg/c)) *2*np.pi, the daming time [s] (average of 1.-vg/c and 1.+vg/c):
            rho3or5sec   = 8.*Rsh3or5sec/L3or5sec**2
            print('fr3or5sec      =', fr3or5sec)
            print('Rsh3or5sec     =', Rsh3or5sec/1e6, 'MOhm')
            print('alpha3or5sec   =', alpha3or5sec/1e-6, 'us')
            print('rho3or5sec     =', rho3or5sec/1e3, 'kOhm/m2')
            print('rho3or5sec/rho =', rho3or5sec/rho)
            print('')

            # For comparison: creating TWCs with the parameters derived from the loaded input using BLonD's TWC object:
            twc3or5sec = TravelingWaveCavity(Rsh3or5sec, fr3or5sec, alpha3or5sec)
            twc3or5sec.imped_calc(freq_array)
            Ztwc3or5sec = twc3or5sec.impedance
            print('Ztwc3or5sec =', Ztwc3or5sec) # ~ Zb3or5sec
            print('')

            # Creating TWCs, characterised by their impedance (Zb); creating RF impedance (Zrf)

            params1 = [fr3or5sec, Z_0, rho3or5sec, L3or5sec, vg]

            ReZb3or5sec = ReZb(freq_array,*params1)
            Zb3or5sec   =   Zb(freq_array,*params1)
            print('ReZb3or5sec =', ReZb3or5sec, '\n', max(np.abs(np.real(ReZb3or5sec))), max(np.abs(np.imag(ReZb3or5sec))), max(np.abs(ReZb3or5sec)))
            print('Zb3or5sec   =', Zb3or5sec,   '\n', max(np.abs(np.real(Zb3or5sec))),   max(np.abs(np.imag(Zb3or5sec))),   max(np.abs(Zb3or5sec)))
            print('')
            ReZrf3or5sec = ReZrf(freq_array,*params1)
            Zrf3or5sec   =   Zrf(freq_array,*params1)
            print('ReZrf3or5sec =', ReZrf3or5sec, '\n', max(np.abs(ReZrf3or5sec)))
            print('Zrf3or5sec   =', Zrf3or5sec,   '\n', max(np.abs(Zrf3or5sec)))
            print('')

            Zrf3or5secnew = np.copy(Zb3or5sec)

            # FF reduction
            if(gff3or5sec is not None):
                hff3or5sec = gff3or5sec / (4.*Z_0)
                ReHff3or5sec = hff3or5sec * Zrf3or5sec
                ImHff3or5sec = 0.
                Hff3or5sec   = ReHff3or5sec + 1j * ImHff3or5sec
                Zrf3or5secnew -= Hff3or5sec * Zrf3or5sec
                print('gff3or5sec =', gff3or5sec)
                print('hff3or5sec =', hff3or5sec)
                print('Hff3or5sec =', Hff3or5sec)
            else:
                print('No FF')
            print('Zrf3or5secnew =', Zrf3or5secnew, '\n', max(np.abs(np.real(Zrf3or5secnew))), max(np.abs(np.imag(Zrf3or5secnew))), max(np.abs(Zrf3or5secnew)))
            print('')

            # FB reduction (acting on top of the FF-reduction, if applicable)
            #if(gfb is not None):
            if(Gfb3or5sec is not None):
            #    gfb3or5sec = Gfb3or5sec * ((rho3or5sec*L3or5sec**2)/8.)
            #    hfb3or5sec = gfb3or5sec / ((rho3or5sec*L3or5sec**2)/8.)
            #    Hfb3or5sec = hfb3or5sec * ReZb3or5sec          # [1]
            #    #gfb3or5sec = Gfb3or5sec * (L3or5sec*np.sqrt(rho3or5sec*Z_0/2.))
            #    #hfb3or5sec = gfb3or5sec / (L3or5sec*np.sqrt(rho3or5sec*Z_0/2.))
            #    #Hfb3or5sec = hfb3or5sec * ReZrf3or5sec         # [1]
            #   #Zrf3or5secnew /= 1.  + Hfb3or5sec*Zrf3or5sec
            #   #Zrf3or5secnew /= Zb3or5sec + Hfb3or5sec*Zrf3or5sec
            #    Hfb3or5sec = Gfb3or5sec / (4.*Z_0)
            #    Zrf3or5secnew /= 1.  + Hfb3or5sec*Hff3or5sec*Zrf3or5sec
            #   #Zrf3or5secnew /= Zb3or5sec + Hfb3or5sec*Hff3or5sec*Zrf3or5sec
            #   #Zrf3or5secnew /= 1.  + Hfb3or5sec*Zrf3or5secnew
            #   #Zrf3or5secnew /= Zb3or5sec + Hfb3or5sec*Zrf3or5secnew
            #    print('Gfb3or5sec =', Gfb3or5sec)
            #    print('gfb3or5sec =', gfb3or5sec)
            #    print('hfb3or5sec =', hfb3or5sec)
            #    print('Hfb3or5sec =', Hfb3or5sec)
                Hfb3or5secZrf = Gfb3or5sec * ReZb3or5sec / ((rho3or5sec*L3or5sec**2)/8.)
                Zrf3or5secnew /= 1.  + Hfb3or5secZrf
                print('Hfb3or5secZrf =', Hfb3or5secZrf)
            else:
                print('No FB')
            print('Zrf3or5secnew =', Zrf3or5secnew, '\n', max(np.abs(np.real(Zrf3or5secnew))), max(np.abs(np.imag(Zrf3or5secnew))), max(np.abs(Zrf3or5secnew)))
            print('')

    #quit()

    ################################

    # "Erase" the main harmonic already loaded and replace it by the new one.
    # Only the contributions in the range [0, 1 GHz] are set to zero (the frequency
    # as computed above might span higher, e.g 6.41 GHz if using the usual 44 kHz
    # from SPS input as frequency resolution

    for i in elID_list:
        impedanceScenario.damp_R_resonatorOrTWC(i, [0,1e9], R_factor=0.) # Deletes elIDkey4 and elIDkey5 if 'present' or 'past'
                                                                         # or      elIDkey3 and elIDkey4 if 'future'
    print('')

    ## Replace them with the a new object that is the sum of the new cavities:
    #if('present' in impedanceScenario.scenarioFileName or 'past' in impedanceScenario.scenarioFileName): # 'past' is similar to present with attenuations of -20.0/-20.0 instead of -15.0/None for the 200MHz/800MHz systems (if applicable)
    #    impedanceScenario.importInputTableFromList([freq_array,
    #                                                np.real(2*Zrfnew4 + 2*Zrfnew5),
    #                                                np.imag(2*Zrfnew4 + 2*Zrfnew5)],
    #                                                'cavities/totalReducedByFeedbackFeedforward')
    #elif('future' in impedanceScenario.scenarioFileName):
    #    impedanceScenario.importInputTableFromList([freq_array,
    #                                                np.real(2*Zrfnew4 + 4*Zrfnew3),
    #                                                np.imag(2*Zrfnew4 + 4*Zrfnew3)],
    #                                                'cavities/totalReducedByFeedbackFeedforward')
    #else:
    #    raise RuntimeError('You must use MODEL= \'present\' or \'future\' to use the reducesImpedanceFeedbackFeedforward function')

    if outdir is not None:

        impedanceScenario.importInputTableFromList([freq_array, np.real(2*Zrf4secnew + ncavities*Zrf3or5secnew), np.imag(2*Zrf4secnew + ncavities*Zrf3or5secnew)], 'cavities/200MHz_total_reducedFF')

        # plot
        myncols=3
        mynrows=3
        fig, ax = plt.subplots(nrows=mynrows, ncols=myncols) #, sharex=True,sharey=True) #) #2) #,sharex=True)
        fig.set_size_inches(1.5*4.0*myncols, 1.5*2.0*mynrows)

        for nrow in [0, 1]:

            if(  '3sections' in elID_dict.keys()): nsec = '3'
            elif('5sections' in elID_dict.keys()): nsec = '5'

            #print('nrow =', nrow)

            if(  nrow == 0):
                ZbA     = Zb4sec
               #ZtwcA   = Zb4sec
                ZrfA    = Zrf4sec
                ZrfnewA = Zrf4secnew
                nsecA   = '4'

            elif(nrow == 1):
                ZbA     = Zb3or5sec
               #ZtwcA   = Zb3or5sec
                ZrfA    = Zrf3or5sec
                ZrfnewA = Zrf3or5secnew
                nsecA   = nsec

            print('ZbA =', ZbA, max(np.abs(ZbA)))
            print('ZrfA =', ZrfA, max(np.abs(ZrfA)))
            print('ZrfnewA =', ZrfnewA, max(np.abs(ZrfnewA)))
            print('')

           #ax[nrow][0].plot(freq_array/1e6, np.real(ZtwcA)/1e6, '-y', label=r'Re($Z_{twc,'+nsecA+r'}$)', alpha=0.5)
           #ax[nrow][1].plot(freq_array/1e6, np.imag(ZtwcA)/1e6, '-y', label=r'Im($Z_{twc,'+nsecA+r'}$)', alpha=0.5)
           #ax[nrow][2].plot(freq_array/1e6, np.abs( ZtwcA)/1e6, '-y', label=r'|$Z_{twc,'+nsecA+r'}$|'  , alpha=0.5)

            ax[nrow][0].plot(freq_array/1e6, np.real(ZbA)/1e6, '--r', label=r'Re($Z_{b,'+nsecA+r'}$)', alpha=0.5) # = ReZb
            ax[nrow][1].plot(freq_array/1e6, np.imag(ZbA)/1e6, '--r', label=r'Im($Z_{b,'+nsecA+r'}$)', alpha=0.5) # = ImZb
            ax[nrow][2].plot(freq_array/1e6, np.abs( ZbA)/1e6, '--r', label=r'|$Z_{b,'+nsecA+r'}$|'  , alpha=0.5) # = |Zb|

            ax[nrow][0].plot(freq_array/1e6, np.real(ZrfA)/1e6, '-b', label=r'Re($Z_{rf,'+nsecA+r'}$)', alpha=0.8) # = ReZrf = Zrf
            ax[nrow][1].plot(freq_array/1e6, np.imag(ZrfA)/1e6, '-b', label=r'Im($Z_{rf,'+nsecA+r'}$)', alpha=0.8) # = zero by definition
            ax[nrow][2].plot(freq_array/1e6, np.abs( ZrfA)/1e6, '-b', label=r'|$Z_{rf,'+nsecA+r'}$|'  , alpha=0.8) # = |Zrf|

            ax[nrow][0].plot(freq_array/1e6, np.real(ZrfnewA)/1e6, ':k', label=r'Re($Z_{rf,'+nsecA+r'}^{new}$)', alpha=1.0)
            ax[nrow][1].plot(freq_array/1e6, np.imag(ZrfnewA)/1e6, ':k', label=r'Im($Z_{rf,'+nsecA+r'}^{new}$)', alpha=1.0)
            ax[nrow][2].plot(freq_array/1e6, np.abs( ZrfnewA)/1e6, ':k', label=r'|$Z_{rf,'+nsecA+r'}^{new}$|'  , alpha=1.0)

            #ax[nrow][2].plot(freq_array/1e6, 20*np.log10( np.abs(ZrfnewA)/np.abs(ZbA) )/100., '-g', label=r'20 log( |$Z_{rf,'+nsecA+r'}^{new}|/|Z_{b,'+nsecA+r'}$| ) /100'+f'\n(min = {min(20*np.log10( np.abs(ZrfnewA)/np.abs(ZbA) )):.4f})', alpha=1.0)
            ##ax[nrow][2].plot(freq_array/1e6, -0.3*np.ones(len(freq_array)), ':', color='#888888')

        for nrow in [2]:

            ZbA     = Zb4sec
           #ZtwcA   = Zb4sec
            ZrfA    = Zrf4sec
            ZrfnewA = Zrf4secnew
            nsecA   = '4'

            ZbB     = Zb3or5sec
           #ZtwcB   = Zb3or5sec
            ZrfB    = Zrf3or5sec
            ZrfnewB = Zrf3or5secnew
            nsecB   = nsec

           #ax[nrow][0].plot(freq_array/1e6, np.real(2*ZtwcA+ncavities*ZtwcB)/1e6, '-y', label=r'Re($2 Z_{twc,'+nsecB+r'} + '+str(ncavities)+r' Z_{twc,'+nsecB+r'}$)', alpha=0.5) # = ReZtwc
           #ax[nrow][1].plot(freq_array/1e6, np.imag(2*ZtwcA+ncavities*ZtwcB)/1e6, '-y', label=r'Im($2 Z_{twc,'+nsecB+r'} + '+str(ncavities)+r' Z_{twc,'+nsecB+r'}$)', alpha=0.5) # = ImZtwc
           #ax[nrow][2].plot(freq_array/1e6, np.abs( 2*ZtwcA+ncavities*ZtwcB)/1e6, '-y', label=r'|$2 Z_{twc,'+nsecB+r'} + '+str(ncavities)+r' Z_{twc,'+nsecB+r'}$|'  , alpha=0.5) # = |Ztwc|

            ax[nrow][0].plot(freq_array/1e6, np.real(2*ZbA + ncavities*ZbB)/1e6, '--r', label=r'Re($2 Z_{b,'+nsecA+r'} + '+str(ncavities)+r' Z_{b,'+nsecB+r'}$)', alpha=0.5) # = ReZb
            ax[nrow][1].plot(freq_array/1e6, np.imag(2*ZbA + ncavities*ZbB)/1e6, '--r', label=r'Im($2 Z_{b,'+nsecA+r'} + '+str(ncavities)+r' Z_{b,'+nsecB+r'}$)', alpha=0.5) # = ImZb
            ax[nrow][2].plot(freq_array/1e6, np.abs( 2*ZbA + ncavities*ZbB)/1e6, '--r', label=r'|$2 Z_{b,'+nsecA+r'} + '  +str(ncavities)+r' Z_{b,'+nsecB+r'}$|'  , alpha=0.5) # = |Zb|

            ax[nrow][0].plot(freq_array/1e6, np.real(2*ZrfA + ncavities*ZrfB)/1e6, '-b', label=r'Re($2 Z_{rf,'+nsecA+r'} + '+str(ncavities)+r' Z_{rf,'+nsecB+r'}$)', alpha=0.8) # = ReZrf = Zrf
            ax[nrow][1].plot(freq_array/1e6, np.imag(2*ZrfA + ncavities*ZrfB)/1e6, '-b', label=r'Im($2 Z_{rf,'+nsecA+r'} + '+str(ncavities)+r' Z_{rf,'+nsecB+r'}$)', alpha=0.8) # = zero by definition
            ax[nrow][2].plot(freq_array/1e6, np.abs( 2*ZrfA + ncavities*ZrfB)/1e6, '-b', label=r'|$2 Z_{rf,'+nsecA+r'} + '  +str(ncavities)+r' Z_{rf,'+nsecB+r'}$|'  , alpha=0.8) # = |Zrf|

            ax[nrow][0].plot(freq_array/1e6, np.real(2*ZrfnewA + ncavities*ZrfnewB)/1e6, ':k', label=r'Re($2 Z_{rf,'+nsecA+r'}^{new} + '+str(ncavities)+r' Z_{rf,'+nsecB+r'}^{new}$)', alpha=1.0)
            ax[nrow][1].plot(freq_array/1e6, np.imag(2*ZrfnewA + ncavities*ZrfnewB)/1e6, ':k', label=r'Im($2 Z_{rf,'+nsecA+r'}^{new} + '+str(ncavities)+r' Z_{rf,'+nsecB+r'}^{new}$)', alpha=1.0)
            ax[nrow][2].plot(freq_array/1e6, np.abs( 2*ZrfnewA + ncavities*ZrfnewB)/1e6, ':k', label=r'|$2 Z_{rf,'+nsecA+r'}^{new} + '  +str(ncavities)+r' Z_{rf,'+nsecB+r'}^{new}$|'  , alpha=1.0)

            ##ax[nrow][2].plot(freq_array/1e6, 10*np.log10(np.abs(2*ZrfnewA + ncavities*ZrfnewB)/np.abs(2*ZbA+ncavities*ZbB))/100., '-g', label=r'log|$2 Z_{rf,'+nsecB+r'}^{new} + '+str(ncavities)+r' Z_{rf,'+nsecB+r'}^{new}$|/|$2 Z_{b,'+nsecB+r'} + '+str(ncavities)+r' Z_{b,'+nsecB+r'}$|/10'  , alpha=1.0)
            ##ax[nrow][2].plot(freq_array/1e6, -0.3*np.ones(len(freq_array)), ':', color='#888888')


        for i in range(mynrows):
            ax[i][0].set_ylabel('Impedance (M$\Omega$)')
            for j in range(myncols):
                if(i==0):
                    ax[mynrows-1][j].set_xlabel('f (MHz)')
                    #if(j==2-1):
                ax[i][j].legend(frameon=False, loc='upper left') # loc='best')
                ax[i][j].set_xlim(180,220)
                ax[i][j].grid()

        fig.tight_layout()
        fig.savefig(outdir+'/plot_reduce_impedance_feedforward_feedback.pdf')
        print('Saving', outdir+'/plot_reduce_impedance_feedforward_feedback.pdf ...\n')
        fig.clf()

    #quit()

#def remove_TWCs(impedanceScenario):
#
#    # Find the index of the main harmonic cavities in the impedance list using their source file names as keys
#    # (the necessary attenuations (e.g. from FB) and corrections have already been performed upon loading):
#
#    key3 = 'cavities/200MHz/3sections/TWC200_3sections_dome_MAIN.dat'
#    key4 = 'cavities/200MHz/4sections/TWC200_4sections_dome_MAIN.dat'
#    key5 = 'cavities/200MHz/5sections/TWC200_5sections_dome_MAIN.dat'
#
#    elID_dict = {} # List of indices of the main cavities to be "erased" and replaced
#    for i in range(len(impedanceScenario.SPSimpList)):
#        filei = impedanceScenario.SPSimpList[i]['file']
#        print(i, filei)
#        if(filei == key3): elID_dict['3sections'] = i # i = elIDkey3
#        if(filei == key4): elID_dict['4sections'] = i # i = elIDkey4
#        if(filei == key5): elID_dict['5sections'] = i # i = elIDkey5
#    print('')
#    print('\nElements being removed:')
#    print('elID_dict =', elID_dict)
#    if(len(list(elID_dict.keys())) == 0):
#        print('No TWCs found to be deleted!\n')
#    print('')
#
#    elID_list = []
#
#    for key in elID_dict.keys():
#
#        elID = elID_dict[key]
#        elID_list.append(elID)
#        print('elID =', elID, ' | key =', key, '|', impedanceScenario.table_impedance[elID]['file'])
#        print('')
#
#
#    # "Erase" the main harmonic already loaded and replace it by the new one.
#    # Only the contributions in the range [0, 1GHz] are set to zero (the frequency
#    # as computed above might span higher, e.g 6.41 GHz if using the usual 44 KHz
#    # from SPS input as frequency resolution
#
#    for i in elID_list:
#        impedanceScenario.damp_R_resonatorOrTWC(i, [0,1e9], R_factor=0.) # Deletes elIDkey4 and elIDkey5 if 'present' or 'past'
#                                                                         # or      elIDkey3 and elIDkey4 if 'future'
#    print('')
