'''
Script to calculate the expected values from theory of a fun machine.
'''

import numpy as np

def alpha(tau, domega, R_gen):
    return 2 * (R_gen / (tau * domega)) * np.sin(domega * tau / 2)

def V_Ib(tau, domega, R_beam, I_Ib):
    return -(2 * R_beam / (tau * domega)**2) * (1 - np.cos(domega * tau)) * I_Ib

def V_Qb(tau, domega, R_beam, I_Ib):
    return (2 * R_beam / (tau * domega)) * (1 - np.sin(domega * tau)/(domega * tau)) * I_Ib

def power(I):
    return 25 * np.abs(I)**2


def theory_calc():
    # General parameters
    V_tot = 10e6                        # [V]
    V_part = 0.5127
    I_Ibeam = 2.750131212489839         # [A]

    # 3-section
    tau3 = 4.619e-7                     # [s]
    domega3 = 1849904.8                 # [rad/s]
    R_gen3 = 9850.907                   # [ohms]
    R_beam3 = 485201.87                 # [ohms]
    n_cav3 = 4

    V3 = V_tot * V_part
    a3 = alpha(tau3, domega3, R_gen3)

    V_Ib3 = V_Ib(tau3, domega3, R_beam3, n_cav3 * I_Ibeam)
    V_Qb3 = V_Qb(tau3, domega3, R_beam3, n_cav3 * I_Ibeam)

    V_b3 = V_Ib3 + 1j * V_Qb3
    Ig3_wob = 0 + 1j * V3 / a3
    Ig3_wb = -(V_Ib3 / a3) + 1j * (V3 - V_Qb3) / a3

    print('3-section:')
    print(f'\tV_beam = {V_b3}')
    print(f'\tI_gen (without beam) = {Ig3_wob/n_cav3}')
    print(f'\tI_gen (with beam) = {Ig3_wb/n_cav3}')
    print(f'\tPower (without beam) = {power(Ig3_wob/n_cav3)}')
    print(f'\tPower (with beam) = {power(Ig3_wb/n_cav3)}')


    # 4-section
    tau4 = 6.207e-7                     # [s]
    domega4 = 1849904.8                 # [rad/s]
    R_gen4 = 13237.1566                 # [ohms]
    R_beam4 = 876111.578                # [ohms]
    n_cav4 = 2

    V4 = V_tot * (1 - V_part)
    a4 = alpha(tau4, domega4, R_gen4)

    V_Ib4 = V_Ib(tau4, domega4, R_beam4, n_cav4 * I_Ibeam)
    V_Qb4 = V_Qb(tau4, domega4, R_beam4, n_cav4 * I_Ibeam)

    V_b4 = V_Ib4 + 1j * V_Qb4
    Ig4_wob = 0 + 1j * V4 / a4
    Ig4_wb = -(V_Ib4 / a4) + 1j * (V4 - V_Qb4) / a4

    print('4-section:')
    print(f'\tV_beam = {V_b4}')
    print(f'\tI_gen (without beam) = {Ig4_wob/n_cav4}')
    print(f'\tI_gen (with beam) = {Ig4_wb/n_cav4}')
    print(f'\tPower (without beam) = {power(Ig4_wob/n_cav4)}')
    print(f'\tPower (with beam) = {power(Ig4_wb/n_cav4)}')

    Ig3_wob = Ig3_wob / n_cav3
    Ig3_wb = Ig3_wb / n_cav3
    Ig4_wob = Ig4_wob / n_cav4
    Ig4_wb = Ig4_wb / n_cav4

    return V_b3, Ig3_wob, Ig3_wb, a3, V_b4, Ig4_wob, Ig4_wb, a4
