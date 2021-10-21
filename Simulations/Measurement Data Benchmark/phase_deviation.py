import numpy as np
import scipy.signal as sps
import blond.utils.bmath as bm


def get_peaks(pro, dt):
    ind = sps.argrelextrema(pro, np.greater)
    return pro[ind[0][1::2]], dt[ind[0][1::2]]

def beam_phase_multibunch(profile, rfstation, OTFB, N_b, bucketlength, bunch_pos):

    indlength = 0

    for i in range(len(profile.bin_centers)):
        if profile.bin_centers[i] > bucketlength:
            break
        indlength += 1

    phi_beam = np.zeros(N_b)

    for i in range(N_b):
        st_ind = bunch_pos[i] * indlength
        end_ind = (bunch_pos[i] + 1) * indlength
        coeff = bm.beam_phase(profile.bin_centers[st_ind:end_ind], profile.n_macroparticles[st_ind:end_ind],
                              0, rfstation.omega_rf[0,0], np.angle(OTFB.OTFB_1.V_ANT[bunch_pos[i]]),
                              profile.bin_size)

        phi_beam[i] = np.arctan2(1,1/coeff) + np.pi

    #import pdb; pdb.set_trace()

    return phi_beam