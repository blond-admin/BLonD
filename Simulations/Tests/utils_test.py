'''
Functions to test the OTFB.
'''
import matplotlib.pyplot as plt
import numpy as np
from Simulations.Tests.OTFB_full_machine_theory import theory_calc

from blond.llrf.new_SPS_OTFB import SPSCavityFeedback_new, CavityFeedbackCommissioning_new


def print_stats(SPSOneTurnFeedback, SPSOneTurnFeedback_new):
    print("The two one turn feedbacks:")
    print("\t\t\t\t Old \t\t\t\t\t New")
    print(f"n_delay: \t\t {SPSOneTurnFeedback.n_delay} \t\t\t\t\t {SPSOneTurnFeedback_new.n_delay}")
    print(f"n_delay_TWC: \t {SPSOneTurnFeedback.n_coarse - SPSOneTurnFeedback.n_delay} \t\t\t\t\t {SPSOneTurnFeedback_new.n_coarse - SPSOneTurnFeedback_new.n_delay}")
    print(f"n_mov_av: \t\t {SPSOneTurnFeedback.n_mov_av} \t\t\t\t\t {SPSOneTurnFeedback_new.n_mov_av}")


def update_signal_array(object, Array, attribute, h):
    Array[:h] = Array[-h:]
    Array[-h:] = getattr(object, attribute)
    return Array


def ripple_calculation(signal):
    max_val = np.max(signal)
    min_val = np.min(signal)
    mean_val = np.mean(signal)

    rel_max = np.abs(max_val - mean_val)/mean_val
    rel_min = np.abs(min_val - mean_val)/mean_val

    if rel_max > rel_min:
        return rel_max
    else:
        return rel_min


def make_step(tot_len, step_len, step_amp, step_pos = 0):
    signal = np.zeros(tot_len, dtype=complex)
    signal[step_pos: step_pos + step_len] = step_amp
    return signal


def eq_line(x, m, x0, y0):
    return m*(x - x0) + y0


def eq_parabola(x, p, x0, y0):
    return p*(x - x0)**2 + y0


def make_rise(rise, A):
    x = np.arange(rise)
    sig = np.zeros(rise, dtype=complex)

    a = 4 * A / (3 * rise)
    b = - A / 6
    p = 8 * A / (3 * rise**2)

    sig[:int(rise/4)] = eq_parabola(x[:int(rise/4)], p, 0, 0)
    sig[int(rise/4):int(3*rise/4)] = eq_line(x[int(rise/4):int(3*rise/4)], a, 0, b)
    sig[int(3*rise/4):] = eq_parabola(x[int(3*rise/4):], -p, rise, A)

    return sig


def make_decay(decay, A):
    x = np.arange(decay)
    sig = np.zeros(decay, dtype=complex)

    a = - 4 * A / (3 * decay)
    b = A + A / 6
    p = 8 * A / (3 * decay**2)

    sig[:int(decay / 4)] = eq_parabola(x[:int(decay/4)], -p, 0, A)
    sig[int(decay / 4):int(3 * decay / 4)] = eq_line(x[int(decay / 4):int(3 * decay / 4)], a, 0, b)
    sig[int(3 * decay / 4):] = eq_parabola(x[int(3 * decay / 4):], p, decay, 0)

    return sig


def make_smooth_step(tot_len, l, A, rise, decay, pos = 0):
    sig = np.zeros(tot_len, dtype=complex)

    sig[pos: pos + rise] = make_rise(rise, A)
    sig[pos + rise: pos + rise + l - decay] = A

    sig[pos + l - decay: pos + l] = make_decay(decay, A)

    return sig





def plot_everything(OTFB_new):
    x_a = np.array(range(2 * OTFB_new.n_coarse))


    plt.title("error and gain")
    plt.plot(x_a, np.abs(OTFB_new.DV_GEN), label="DV_GEN", color="r")
    plt.plot(x_a, np.abs(OTFB_new.V_SET), label="V_SET", color="b")
    plt.plot(x_a, np.abs(OTFB_new.V_ANT_START), label="V_ANT", color='g')
    plt.legend()
    plt.show()

    plt.title("Comb")
    plt.plot(x_a, np.abs(OTFB_new.DV_GEN), label="DV_GEN", color="r")
    plt.plot(x_a, np.abs(OTFB_new.DV_COMB_OUT), label="DV_COMB_OUT", color="b")
    plt.legend()
    plt.show()

    plt.title("one turn delay")
    plt.plot(x_a, np.abs(OTFB_new.DV_COMB_OUT), label="DV_COMB_OUT", color="r")
    plt.plot(x_a, np.abs(OTFB_new.DV_DELAYED), label="DV_DELAYED", color="b")
    plt.legend()
    plt.show()

    plt.title('mod to fr')
    plt.plot(x_a, OTFB_new.DV_DELAYED.real, label="DV_DELAYED", color="r")
    plt.plot(x_a, OTFB_new.DV_DELAYED.imag, label="DV_DELAYED", color="r", linestyle='dotted')
    plt.plot(x_a, OTFB_new.DV_MOD_FR.real, label="DV_MOD_FR", color="b")
    plt.plot(x_a, OTFB_new.DV_MOD_FR.imag, label="DV_MOD_FR", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title("mov avg")
    plt.plot(x_a, np.abs(OTFB_new.DV_MOV_AVG), label="DV_MOV_AVG", color="r")
    plt.plot(x_a, np.abs(OTFB_new.DV_MOD_FR), label="DV_MOD_FR", color="b")
    plt.legend()
    plt.show()

    plt.title('mod to frf')
    plt.plot(x_a, OTFB_new.DV_MOV_AVG.real, label="DV_DELAYED", color="r")
    plt.plot(x_a, OTFB_new.DV_MOV_AVG.imag, label="DV_DELAYED", color="r", linestyle='dotted')
    plt.plot(x_a, OTFB_new.DV_MOD_FRF.real, label="DV_MOD_FR", color="b")
    plt.plot(x_a, OTFB_new.DV_MOD_FRF.imag, label="DV_MOD_FR", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title("sum and gain")
    plt.plot(x_a, np.abs(OTFB_new.DV_MOD_FRF), label="DV_MOD_FRF", color="r")
    plt.plot(x_a, np.abs(OTFB_new.I_GEN), label="I_GEN", color="b")
    plt.legend()
    plt.show()

    plt.subplot(211)
    plt.title("gen response")
    plt.plot(x_a, np.abs(OTFB_new.V_IND_COARSE_GEN), label="V_IND_COARSE_GEN")
    plt.legend()
    plt.subplot(212)
    plt.plot(x_a, np.abs(OTFB_new.I_GEN), label="I_GEN")
    plt.legend()
    plt.show()


def plot_everything_real_imag(O):
    x_a = np.array(range(2 * O.n_coarse))

    plt.title("error and gain")
    plt.plot(x_a, O.DV_GEN.real, label="DV_GEN", color="r")
    plt.plot(x_a, O.DV_GEN.imag, label="DV_GEN", color="r", linestyle='dotted')
    plt.plot(x_a, O.V_SET.real, label="V_SET", color="b")
    plt.plot(x_a, O.V_SET.imag, label="V_SET", color="b", linestyle='dotted')
    plt.plot(x_a, O.V_ANT_START.real, label="V_ANT", color='g')
    plt.plot(x_a, O.V_ANT_START.imag, label="V_ANT", color='g', linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title("Comb")
    plt.plot(x_a, O.DV_GEN.real, label="DV_GEN", color="r")
    plt.plot(x_a, O.DV_GEN.imag, label="DV_GEN", color="r", linestyle='dotted')
    plt.plot(x_a, O.DV_COMB_OUT.real, label="DV_COMB_OUT", color="b")
    plt.plot(x_a, O.DV_COMB_OUT.imag, label="DV_COMB_OUT", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title("one turn delay")
    plt.plot(x_a, O.DV_COMB_OUT.real, label="DV_COMB_OUT", color="r")
    plt.plot(x_a, O.DV_COMB_OUT.imag, label="DV_COMB_OUT", color="r", linestyle='dotted')
    plt.plot(x_a, O.DV_DELAYED.real, label="DV_DELAYED", color="b")
    plt.plot(x_a, O.DV_DELAYED.imag, label="DV_DELAYED", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title('mod to fr')
    plt.plot(x_a, O.DV_DELAYED.real, label="DV_DELAYED", color="r")
    plt.plot(x_a, O.DV_DELAYED.imag, label="DV_DELAYED", color="r", linestyle='dotted')
    plt.plot(x_a, O.DV_MOD_FR.real, label="DV_MOD_FR", color="b")
    plt.plot(x_a, O.DV_MOD_FR.imag, label="DV_MOD_FR", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title("mov avg")
    plt.plot(x_a, O.DV_MOD_FR.real, label="DV_MOD_FR", color="r")
    plt.plot(x_a, O.DV_MOD_FR.imag, label="DV_MOD_FR", color="r", linestyle='dotted')
    plt.plot(x_a, O.DV_MOV_AVG.real, label="DV_MOV_AVG", color="b")
    plt.plot(x_a, O.DV_MOV_AVG.imag, label="DV_MOV_AVG", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.title("sum and gain")
    plt.plot(x_a, O.DV_MOD_FRF.real, label="DV_MOD_FRF", color="r")
    plt.plot(x_a, O.DV_MOD_FRF.imag, label="DV_MOD_FRF", color="r", linestyle='dotted')
    plt.plot(x_a, O.I_GEN.real, label="I_GEN", color="b")
    plt.plot(x_a, O.I_GEN.imag, label="I_GEN", color="b", linestyle='dotted')
    plt.legend()
    plt.show()

    plt.subplot(211)
    plt.title("gen response")
    plt.plot(x_a, O.V_IND_COARSE_GEN.real, label="V_IND_COARSE_GEN")
    plt.plot(x_a, O.V_IND_COARSE_GEN.imag, label="V_IND_COARSE_GEN", linestyle='dotted')
    plt.legend()
    plt.subplot(212)
    plt.plot(x_a, O.I_GEN.real, label="I_GEN")
    plt.plot(x_a, O.I_GEN.imag, label="I_GEN", linestyle='dotted')
    plt.legend()
    plt.show()


def plot_cont_everything(OTFBs, h0s):
    linestyles = ['-', 'dotted', '--', '-.', '-..', 'loosely dotted', 'loosely dashed', 'loosely dashdotted',
                  'loosely dashdotdotted']
    markers = ['x', 'D', 'o', '+']
    plt.title("error and gain")
    plt.vlines(4620, 0, np.max(np.abs(OTFBs[0].DV_GEN)), 'black', 'dotted')
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].DV_GEN), -h0s[i]), label="DV_GEN", color="r", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
        plt.plot(np.roll(np.abs(OTFBs[i].V_SET), -h0s[i]), label="V_SET", color="b", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
        plt.plot(np.roll(np.abs(OTFBs[i].V_ANT_START), -h0s[i]), label="V_ANT", color='g', linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.show()

    plt.title("Comb")
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].DV_GEN), -h0s[i]), label="DV_GEN", color="r", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
        plt.plot(np.roll(np.abs(OTFBs[i].DV_COMB_OUT), -h0s[i]), label="DV_COMB_OUT", color="b", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.show()

    plt.title("one turn delay")
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].DV_COMB_OUT), -h0s[i]), label="DV_COMB_OUT", color="r", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
        plt.plot(np.roll(np.abs(OTFBs[i].DV_DELAYED), -h0s[i]), label="DV_DELAYED", color="b", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.show()

    plt.title("mov avg")
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].DV_MOV_AVG), -h0s[i]), label="DV_MOV_AVG", color="r", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
        plt.plot(np.roll(np.abs(OTFBs[i].DV_MOD_FR), -h0s[i]), label="DV_MOD_FR", color="b", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.show()

    plt.title("sum and gain")
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].DV_MOD_FRF), -h0s[i]), label="DV_MOD_FRF", color="r", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
        plt.plot(np.roll(np.abs(OTFBs[i].I_GEN), -h0s[i]), label="I_GEN", color="b", linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.show()

    plt.subplot(211)
    plt.title("gen response")
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].V_IND_COARSE_GEN), -h0s[i]), label="V_IND_COARSE_GEN", color='r', linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.subplot(212)
    for i in range(len(OTFBs)):
        plt.plot(np.roll(np.abs(OTFBs[i].I_GEN), -h0s[i]), label="I_GEN", color='r', linestyle=linestyles[i], marker=markers[i], alpha=0.7)
    plt.legend()
    plt.show()


def rms_between_arrays(a1, a2, h0, l, PLOT = False):
    diff = np.abs(np.abs(a1) - np.abs(a2)) / np.max(np.abs(a1))
    diff_relevant = diff[h0:h0 + l]

    if PLOT:
        plt.subplot(211)
        plt.plot(np.abs(a1), label='a1')
        plt.plot(np.abs(a2), label='a2')
        plt.vlines(h0, np.min(np.abs(a1)), np.max(np.abs(a1)), 'black', 'dotted', 'start_error')
        plt.vlines(h0 + l, np.min(np.abs(a1)), np.max(np.abs(a1)), 'black', 'dotted', 'end_error')
        plt.legend()
        plt.subplot(212)
        plt.plot(diff_relevant)
        plt.show()

    return np.sum(diff_relevant) / len(diff_relevant)


def calc_rms_error_module(OTFBs, h0s, h0, l, PLOT = False):
    errors = np.zeros((6, len(OTFBs)-1))

    for i in range(len(OTFBs)-1):
        errors[0, i] = rms_between_arrays(np.roll(OTFBs[i].DV_GEN, -h0s[i]),
                                          np.roll(OTFBs[-1].DV_GEN, -h0s[-1]),
                                          h0, l, PLOT)

        errors[1, i] = rms_between_arrays(np.roll(OTFBs[i].DV_COMB_OUT, -h0s[i]),
                                          np.roll(OTFBs[-1].DV_COMB_OUT, -h0s[-1]),
                                          h0, l, PLOT)

        errors[2, i] = rms_between_arrays(np.roll(OTFBs[i].DV_DELAYED, -h0s[i]),
                                          np.roll(OTFBs[-1].DV_DELAYED, -h0s[-1]),
                                          h0, l, PLOT)

        errors[3, i] = rms_between_arrays(np.roll(OTFBs[i].DV_MOV_AVG, -h0s[i]),
                                          np.roll(OTFBs[-1].DV_MOV_AVG, -h0s[-1]),
                                          h0, l, PLOT)

        errors[4, i] = rms_between_arrays(np.roll(OTFBs[i].I_GEN, -h0s[i]),
                                          np.roll(OTFBs[-1].I_GEN, -h0s[-1]),
                                          h0, l, PLOT)

        errors[5, i] = rms_between_arrays(np.roll(OTFBs[i].V_IND_COARSE_GEN, -h0s[i]),
                                          np.roll(OTFBs[-1].V_IND_COARSE_GEN, -h0s[-1]),
                                          h0, l, PLOT)

    return errors


def plot_errors(errors):

    plt.title('DV_GEN')
    for i in range(np.shape(errors)[2]):
        plt.plot(errors[:, 0, i])
    plt.show()

    plt.title('DV_COMB_OUT')
    for i in range(np.shape(errors)[2]):
        plt.plot(errors[:, 1, i])
    plt.show()

    plt.title('DV_DELAYED')
    for i in range(np.shape(errors)[2]):
        plt.plot(errors[:, 2, i])
    plt.show()

    plt.title('DV_MOV_AVG')
    for i in range(np.shape(errors)[2]):
        plt.plot(errors[:, 3, i])
    plt.show()

    plt.title('I_GEN')
    for i in range(np.shape(errors)[2]):
        plt.plot(errors[:, 4, i])
    plt.show()

    plt.title('V_IND_COARSE_GEN')
    for i in range(np.shape(errors)[2]):
        plt.plot(errors[:, 5, i])
    plt.show()


def in_out_mov_avg_calc(Os, h0s, h0, l, PLOT = False):
    errors = np.zeros((2, len(Os)-1))

    for i in range(len(Os) - 1):
        errors[0, i] = rms_between_arrays(np.roll(Os[i].DV_MOD_FR, -h0s[i]),
                                          np.roll(Os[-1].DV_MOD_FR, -h0s[-1]),
                                          h0, l, PLOT = PLOT)

        errors[1, i] = rms_between_arrays(np.roll(Os[i].DV_MOV_AVG, -h0s[i]),
                                          np.roll(Os[-1].DV_MOV_AVG, -h0s[-1]),
                                          h0, l, PLOT = PLOT)

    return errors


def find_spike(ER, i, n, tol):
    if i < n - 1:
        avg = np.mean(ER[:i - 1])
    elif i < 1:
        return False
    else:
        avg = np.mean(ER[i - n - 1:i - 1])

    diff = np.abs(ER[i] - avg) * 100 / avg

    if diff > tol:
        print(f"Error {i} is {ER[i]} and the mean is {avg}")

    return diff > tol


def rms_between_arrays_plot(a1, a2, h0, l, title=''):
    diff = np.abs(np.abs(a1) - np.abs(a2))
    diff_relevant = diff[h0:h0 + l]

    plt.subplot(211)
    plt.title(title)
    plt.plot(np.abs(a1), label='a1')
    plt.plot(np.abs(a2), label='a2')
    plt.vlines(h0, np.min(np.abs(a1)), np.max(np.abs(a1)), 'black', 'dotted', 'start_error')
    plt.vlines(h0 + l, np.min(np.abs(a1)), np.max(np.abs(a1)), 'black', 'dotted', 'end_error')
    plt.legend()
    plt.subplot(212)
    plt.plot(diff_relevant)
    plt.show()


def plot_spikes(Os, h0s, h0, l, PLOT_ALL = True):
    for i in range(len(Os)-1):
        rms_between_arrays_plot(np.roll(Os[i].DV_GEN, -h0s[i]),
                                np.roll(Os[-1].DV_GEN, -h0s[-1]),
                                h0, l,
                                title='DV_GEN '+ str(h0s[i]))

        if PLOT_ALL:
            rms_between_arrays_plot(np.roll(Os[i].DV_COMB_OUT, -h0s[i]),
                                    np.roll(Os[-1].DV_COMB_OUT, -h0s[-1]),
                                    h0, l,
                                    title='DV_COMB_OUT '+ str(h0s[i]))

            rms_between_arrays_plot(np.roll(Os[i].DV_DELAYED, -h0s[i]),
                                    np.roll(Os[-1].DV_DELAYED, -h0s[-1]),
                                    h0, l,
                                    title='DV_DELAYED '+ str(h0s[i]))

            rms_between_arrays_plot(np.roll(Os[i].DV_MOV_AVG, -h0s[i]),
                                    np.roll(Os[-1].DV_MOV_AVG, -h0s[-1]),
                                    h0, l,
                                    title='DV_MOV_AVG '+ str(h0s[i]))

            rms_between_arrays_plot(np.roll(Os[i].I_GEN, -h0s[i]),
                                    np.roll(Os[-1].I_GEN, -h0s[-1]),
                                    h0, l,
                                    title='I_GEN '+ str(h0s[i]))

            rms_between_arrays_plot(np.roll(Os[i].V_IND_COARSE_GEN, -h0s[i]),
                                    np.roll(Os[-1].V_IND_COARSE_GEN, -h0s[-1]),
                                    h0, l,
                                    title='V_IND_COARSE_GEN '+ str(h0s[i]))


def difference_array(O1, O2, h1, h2, h0, l, SIG_NAME):
    a1 = getattr(O1, SIG_NAME)
    a2 = getattr(O2, SIG_NAME)

    a1 = np.roll(a1, -h1)
    a2 = np.roll(a2, -h2)

    relevant_difference = np.abs(np.abs(a1) - np.abs(a2))[h0:h0+l] / np.max(np.abs(a2))

    return relevant_difference


def plot_beam_test_comparison(O1, O2):

    plt.subplot(121)
    plt.title('Beam induced voltage')
    plt.plot(O1.V_IND_COARSE_BEAM.real, label='New OTFB real', color='r', alpha=0.7)
    plt.plot(O1.V_IND_COARSE_BEAM.imag, label='New OTFB imag', color='r', linestyle='dotted', alpha=0.7)
    plt.plot(O2.V_IND_COARSE_BEAM.real, label='Old OTFB real', color='b', alpha=0.7)
    plt.plot(O2.V_IND_COARSE_BEAM.imag, label='Old OTFB imag', color='b', linestyle='dotted', alpha=0.7)
    plt.legend()

    plt.subplot(122, polar=True)
    plt.title('Beam induced voltage, amplitude and phase')
    plt.polar(np.angle(O1.V_IND_COARSE_BEAM), np.abs(O1.V_IND_COARSE_BEAM), label='New OTFB', color='r', alpha=0.7)
    plt.polar(np.angle(O2.V_IND_COARSE_BEAM), np.abs(O2.V_IND_COARSE_BEAM), label='Old OTFB', color='b', alpha=0.7)
    plt.legend()

    plt.show()

    plt.subplot(121)
    plt.title('Antenna voltage')
    plt.plot(O1.V_ANT_START.real, label='New OTFB real', color='r', alpha=0.7)
    plt.plot(O1.V_ANT_START.imag, label='New OTFB imag', color='r', linestyle='dotted', alpha=0.7)
    plt.plot(O2.V_ANT.real, label='Old OTFB real', color='b', alpha=0.7)
    plt.plot(O2.V_ANT.imag, label='Old OTFB imag', color='b', linestyle='dotted', alpha=0.7)
    plt.legend()

    plt.subplot(122, polar=True)
    plt.title('Antenna voltage, amplitude and phase')
    plt.polar(np.angle(O1.V_ANT_START), np.abs(O1.V_ANT_START), label='New OTFB', color='r', alpha=0.7)
    plt.polar(np.angle(O2.V_ANT), np.abs(O2.V_ANT), label='Old OTFB', color='b', alpha=0.7)
    plt.legend()

    plt.show()

def plot_beam_test_full_machine(O):
    plt.subplot(121)
    plt.title('Beam induced voltage')
    plt.plot(O.V_IND_COARSE_BEAM.real, label='New OTFB real', color='r', alpha=0.7)
    plt.plot(O.V_IND_COARSE_BEAM.imag, label='New OTFB imag', color='r', linestyle='dotted', alpha=0.7)
    plt.legend()

    plt.subplot(122, polar=True)
    plt.title('Beam induced voltage, amplitude and phase')
    plt.polar(np.angle(O.V_IND_COARSE_BEAM), np.abs(O.V_IND_COARSE_BEAM), label='New OTFB', color='r', alpha=0.7)
    plt.legend()

    plt.show()

    plt.subplot(121)
    plt.title('Antenna voltage')
    plt.plot(O.V_ANT_START.real, label='New OTFB real', color='r', alpha=0.7)
    plt.plot(O.V_ANT_START.imag, label='New OTFB imag', color='r', linestyle='dotted', alpha=0.7)
    plt.legend()

    plt.subplot(122, polar=True)
    plt.title('Antenna voltage, amplitude and phase')
    plt.polar(np.angle(O.V_ANT_START), np.abs(O.V_ANT_START), label='New OTFB', color='r', alpha=0.7)
    plt.legend()

    plt.show()

    plt.title('RF beam current')
    plt.plot(O.I_COARSE_BEAM.real / O.T_s / 5, label='I_BEAM Real', color='r', alpha=0.7)
    plt.plot(O.I_COARSE_BEAM.imag / O.T_s / 5, label='I_BEAM Imag', color='r', alpha=0.7)
    plt.legend()
    plt.show()

def plot_IQ_full_machine(O, cav_type = 3, with_theory = False, norm_plot = False, with_beam = True):

    if with_theory:
        V_b3, Ig3_wob, Ig3_wb, a3, V_b4, Ig4_wob, Ig4_wb, a4 = theory_calc()

    plt.subplot(121, polar=True)
    plt.title(r'I/Q-plane voltages')
    if with_beam:
        plt.polar([np.mean(np.angle(O.V_IND_COARSE_BEAM)), 0], [np.mean(np.abs(O.V_IND_COARSE_BEAM)), 0],
                  label=r'$V_{\textrm{ind,beam}}$ Sim', marker='<', alpha=0.7, color='r', markevery=2, markersize=15)
        if with_theory:
            if cav_type == 3:
                plt.polar([np.angle(V_b3), 0], [np.abs(V_b3), 0], label=r'$V_{\textrm{ind,beam}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)
            else:
                plt.polar([np.angle(V_b4), 0], [np.abs(V_b4), 0], label=r'$V_{\textrm{ind,beam}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)

    plt.polar([np.mean(np.angle(O.V_SET)), 0], [np.mean(np.abs(O.V_SET)), 0],
              label=r'$V_{\textrm{set}}$ Sim', marker='<', alpha=0.7, color='g', markevery=2, markersize=15)
    plt.polar([np.mean(np.angle(O.V_IND_COARSE_GEN)), 0], [np.mean(np.abs(O.V_IND_COARSE_GEN)), 0],
              label=r'$V_{\textrm{ind,gen}}$ Sim', marker='<', alpha=0.7, color='c', markevery=2, markersize=15)
    if with_theory:
        if with_beam:
            if cav_type == 3:
                plt.polar([np.angle(O.n_cavities * a3 * Ig3_wb), 0], [np.abs(O.n_cavities * a3 * Ig3_wb), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)
            else:
                plt.polar([np.angle(O.n_cavities * a4 * Ig4_wb), 0], [np.abs(O.n_cavities * a4 * Ig4_wb), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)
        else:
            if cav_type == 3:
                plt.polar([np.angle(O.n_cavities * a3 * Ig3_wob), 0], [np.abs(O.n_cavities * a3 * Ig3_wob), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)
            else:
                plt.polar([np.angle(O.n_cavities * a4 * Ig4_wob), 0], [np.abs(O.n_cavities * a4 * Ig4_wob), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)

    plt.polar([np.mean(np.angle(O.V_ANT)), 0], [np.mean(np.abs(O.V_ANT)), 0],
              label=r'$V_{\textrm{ant}}$ Sim', marker='<', alpha=0.7, color='m', markevery=2, markersize=15)

    plt.legend()

    plt.subplot(122, polar=True)
    plt.title(r'I/Q-plane currents')
    plt.polar([np.mean(np.angle(O.I_GEN / O.T_s)), 0], [np.mean(np.abs(O.I_GEN / O.T_s)), 0],
              label=r'$I_{\textrm{gen}}$ Sim', marker='<', alpha=0.7, color='r', markevery=2, markersize=15)
    if with_beam:
        pass
        #plt.polar([np.mean(np.angle(O.I_COARSE_BEAM / O.T_s / 5)), 0], [np.max(np.abs(O.I_COARSE_BEAM / O.T_s / 5)), 0],
        #          label='I_COARSE_BEAM', marker='<', alpha=0.7, color='b', markevery=2, markersize=15)

    if with_theory:
        if with_beam:
            if cav_type == 3:
                plt.polar([np.angle(Ig3_wb), 0], [np.abs(Ig3_wb), 0], label=r'$I_{\textrm{gen}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)
            else:
                plt.polar([np.angle(Ig4_wb), 0], [np.abs(Ig4_wb), 0], label=r'$I_{\textrm{gen}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)
        else:
            if cav_type == 3:
                plt.polar([np.angle(Ig3_wob), 0], [np.abs(Ig3_wob), 0], label=r'$I_{\textrm{gen}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)
            else:
                plt.polar([np.angle(Ig4_wob), 0], [np.abs(Ig4_wob), 0], label=r'$I_{\textrm{gen}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)

    plt.suptitle(r'{0}-section'.format(cav_type))
    plt.legend()

    plt.show()

    if norm_plot:
        plt.subplot(121, polar=True)
        plt.title('I/Q-plane voltages')
        plt.polar(np.angle(O.V_IND_COARSE_BEAM), np.abs(O.V_IND_COARSE_BEAM)/np.max(np.abs(O.V_IND_COARSE_BEAM)), label='V_IND_COARSE_BEAM', marker='x',
                  alpha=0.7)
        plt.polar(np.angle(O.V_ANT_START), np.abs(O.V_ANT_START)/np.max(np.abs(O.V_ANT_START)), label='V_ANT_START', marker='x', alpha=0.7)
        plt.polar(np.angle(O.V_SET), np.abs(O.V_SET)/np.max(np.abs(O.V_SET)), label='V_SET', marker='x', alpha=0.7)
        plt.polar(np.angle(O.V_IND_COARSE_GEN), np.abs(O.V_IND_COARSE_GEN)/np.max(np.abs(O.V_IND_COARSE_GEN)), label='V_IND_COARSE_GEN', marker='x', alpha=0.7)
        plt.polar(np.angle(O.V_ANT), np.abs(O.V_ANT)/np.max(np.abs(O.V_ANT)), label='V_ANT', marker='x', alpha=0.7)
        plt.legend()

        plt.subplot(122, polar=True)
        plt.title('I/Q-plane currents')
        plt.polar(np.angle(O.I_GEN), np.abs(O.I_GEN)/np.max(np.abs(O.I_GEN)), label='I_GEN', marker='x', alpha=0.7)
        plt.polar(np.angle(O.I_COARSE_BEAM), np.abs(O.I_COARSE_BEAM)/np.max(np.abs(O.I_COARSE_BEAM)), label='I_COARSE_BEAM', marker='x', alpha=0.7)
        plt.legend()

        plt.show()


def plot_IQ_full_machine_v2(O, with_theory = False, with_beam = True):

    if with_theory:
        V_b3, Ig3_wob, Ig3_wb, a3, V_b4, Ig4_wob, Ig4_wb, a4 = theory_calc()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    if with_beam:
        ax.plot([np.mean(np.angle(O.OTFB_1.V_IND_COARSE_BEAM)), 0], [np.mean(np.abs(O.OTFB_1.V_IND_COARSE_BEAM)), 0],
                  label=r'$V_{\textrm{ind,beam}}$ Sim', marker='<', alpha=0.7, color='r', markevery=2, markersize=15)
        if with_theory:
            ax.plot([np.angle(V_b3), 0], [np.abs(V_b3), 0], label=r'$V_{\textrm{ind,beam}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)

    ax.plot([np.mean(np.angle(O.OTFB_1.V_SET)), 0], [np.mean(np.abs(O.OTFB_1.V_SET)), 0],
              label=r'$V_{\textrm{set}}$ Sim', marker='<', alpha=0.7, color='g', markevery=2, markersize=15)
    ax.plot([np.mean(np.angle(O.OTFB_1.V_IND_COARSE_GEN)), 0], [np.mean(np.abs(O.OTFB_1.V_IND_COARSE_GEN)), 0],
              label=r'$V_{\textrm{ind,gen}}$ Sim', marker='<', alpha=0.7, color='c', markevery=2, markersize=15)
    if with_theory:
        if with_beam:
            ax.plot([np.angle(O.OTFB_1.n_cavities * a3 * Ig3_wb), 0], [np.abs(O.OTFB_1.n_cavities * a3 * Ig3_wb), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)
        else:
            ax.plot([np.angle(O.OTFB_1.n_cavities * a3 * Ig3_wob), 0], [np.abs(O.OTFB_1.n_cavities * a3 * Ig3_wob), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)

    ax.plot([np.mean(np.angle(O.OTFB_1.V_ANT)), 0], [np.mean(np.abs(O.OTFB_1.V_ANT)), 0],
              label=r'$V_{\textrm{ant}}$ Sim', marker='<', alpha=0.7, color='m', markevery=2, markersize=15)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    fig.suptitle(r'I/Q-plane Voltages, 3-section')
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    if with_beam:
        ax.plot([np.mean(np.angle(O.OTFB_2.V_IND_COARSE_BEAM)), 0], [np.mean(np.abs(O.OTFB_2.V_IND_COARSE_BEAM)), 0],
                  label=r'$V_{\textrm{ind,beam}}$ Sim', marker='<', alpha=0.7, color='r', markevery=2, markersize=15)
        if with_theory:
            ax.plot([np.angle(V_b4), 0], [np.abs(V_b4), 0], label=r'$V_{\textrm{ind,beam}}$ Theory',
                          marker='>', alpha=0.7, color='olive', markevery=2, markersize=15)

    ax.plot([np.mean(np.angle(O.OTFB_2.V_SET)), 0], [np.mean(np.abs(O.OTFB_2.V_SET)), 0],
              label=r'$V_{\textrm{set}}$ Sim', marker='<', alpha=0.7, color='g', markevery=2, markersize=15)
    ax.plot([np.mean(np.angle(O.OTFB_2.V_IND_COARSE_GEN)), 0], [np.mean(np.abs(O.OTFB_2.V_IND_COARSE_GEN)), 0],
              label=r'$V_{\textrm{ind,gen}}$ Sim', marker='<', alpha=0.7, color='c', markevery=2, markersize=15)
    if with_theory:
        if with_beam:
            ax.plot([np.angle(O.OTFB_2.n_cavities * a4 * Ig4_wb), 0], [np.abs(O.OTFB_2.n_cavities * a4 * Ig4_wb), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)
        else:
            ax.plot([np.angle(O.OTFB_2.n_cavities * a4 * Ig4_wob), 0], [np.abs(O.OTFB_2.n_cavities * a4 * Ig4_wob), 0],
                          label=r'$V_{\textrm{ind,gen}}$ Theory', marker='>', alpha=0.7, color='purple', markevery=2, markersize=15)

    ax.plot([np.mean(np.angle(O.OTFB_2.V_ANT)), 0], [np.mean(np.abs(O.OTFB_2.V_ANT)), 0],
              label=r'$V_{\textrm{ant}}$ Sim', marker='<', alpha=0.7, color='m', markevery=2, markersize=15)

    fig.suptitle(r'I/Q-plane Voltages, 4-section')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot([np.mean(np.angle(O.OTFB_1.I_GEN / O.OTFB_1.T_s)), 0], [np.mean(np.abs(O.OTFB_1.I_GEN / O.OTFB_1.T_s)), 0],
              label=r'$I_{\textrm{gen}}$ Sim', marker='<', alpha=0.7, color='r', markevery=2, markersize=15)
    if with_theory:
        if with_beam:
            ax.plot([np.angle(Ig3_wb), 0], [np.abs(Ig3_wb), 0], label=r'$I_{\textrm{gen}}$ Theory',
                          marker='>', alpha=0.7, color='b', markevery=2, markersize=15)
        else:
            ax.plot([np.angle(Ig3_wob), 0], [np.abs(Ig3_wob), 0], label=r'$I_{\textrm{gen}}$ Theory',
                          marker='>', alpha=0.7, color='b', markevery=2, markersize=15)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    fig.suptitle(r'I/Q-plane Currents, 3-section')

    fig = plt.figure()
    ax = fig.add_subplot(122, polar=True)
    ax.plot([np.mean(np.angle(O.OTFB_2.I_GEN / O.OTFB_2.T_s)), 0], [np.mean(np.abs(O.OTFB_2.I_GEN / O.OTFB_2.T_s)), 0],
              label=r'$I_{\textrm{gen}}$ Sim', marker='<', alpha=0.7, color='r', markevery=2, markersize=15)
    if with_theory:
        if with_beam:
            ax.plot([np.angle(Ig4_wb), 0], [np.abs(Ig4_wb), 0], label=r'$I_{\textrm{gen}}$ Theory',
                      marker='>', alpha=0.7, color='b', markevery=2, markersize=15)
        else:
            ax.plot([np.angle(Ig4_wob), 0], [np.abs(Ig4_wob), 0], label=r'$I_{\textrm{gen}}$ Theory',
                      marker='>', alpha=0.7, color='b', markevery=2, markersize=15)

    fig.suptitle(r'I/Q-plane Currents, 4-section')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    plt.show()


def get_power_gen_I2(I_gen_per_cav, Z_0): # Main in use
    ''' RF generator power from generator current (physical, in [A]), for any f_r (and thus any tau) '''
    return 0.5 * Z_0 * np.abs(I_gen_per_cav)**2


def theoretical_signals(O, I_beam):
    domega = O.omega_c - O.omega_r
    tau = O.TWC.tau
    R_g = O.TWC.R_gen
    R_b = O.TWC.R_beam

    a_gI = 2 * (R_g / tau / domega) * np.sin(domega * tau / 2)
    a_gQ = 2 * (R_g / tau / domega) * np.sin(domega * tau / 2)
    a_bI = -2 * (R_b / ((tau**2) * (domega**2))) * (1 - np.cos(domega * tau))
    a_bQ = 2 * (R_b / tau / domega) * (1 - (1/tau/domega) * np.sin(domega * tau))

    print('coefficients:',a_gI, a_gQ, a_bI, a_bQ)

    I_g_no_beam = ((1/a_gI) * np.real(O.V_set) + (1/a_gQ) * np.imag(O.V_set) * 1j) / O.n_cavities

    I_g_with_beam = ((1/a_gI) * (np.real(O.V_set) - a_bI * np.real(I_beam)) + (1/a_gQ) * (np.imag(O.V_set) - a_bQ * np.real(I_beam)) * 1j) / O.n_cavities

    return I_g_no_beam, I_g_with_beam

def print_stats(CFB):
    print()
    print('For the 3-section:')
    print(f'\ttau = {CFB.OTFB_1.TWC.tau/(1e-9)} ns')
    print(f'\tR_gen = {CFB.OTFB_1.TWC.R_gen} ohms')
    print(f'\tR_beam = {CFB.OTFB_1.TWC.R_beam} ohms')
    print(f'\tf_r = {CFB.OTFB_1.omega_r/(2 * np.pi)}')
    print(f'\tf_c = {CFB.OTFB_1.omega_c/(2 * np.pi)}')

    print('For the 4-section:')
    print(f'\ttau = {CFB.OTFB_2.TWC.tau / (1e-9)} ns')
    print(f'\tR_gen = {CFB.OTFB_2.TWC.R_gen} ohms')
    print(f'\tR_beam = {CFB.OTFB_2.TWC.R_beam} ohms')
    print(f'\tf_r = {CFB.OTFB_2.omega_r/(2 * np.pi)}')
    print(f'\tf_c = {CFB.OTFB_2.omega_c/(2 * np.pi)}')
    print()

def bisection_method_G_tx(RFStation, Beam, Profile, N_t):
    V_part = 0.5172

    Commissioning = CavityFeedbackCommissioning_new(open_FF=True, debug=False)
    OTFB = SPSCavityFeedback_new(RFStation, Beam, Profile, post_LS2=True, V_part=V_part,
                                 Commissioning=Commissioning, G_tx=G_tx_ls)

    return 1