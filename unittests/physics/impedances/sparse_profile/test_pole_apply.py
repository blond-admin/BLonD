import time
import unittest

import numba
import numpy as np
import skrf as rf
from matplotlib import pyplot as plt
from numba import complex128, float64
from scipy.signal import fftconvolve

from blond.physics.impedances.refactor_me2 import apply_poles2


def get_poles(
    freqs: np.ndarray,
    Z: np.ndarray,
    n_pole: int,
    verbose=True,
    max_iterations: int | None = None,
):
    print("Create Network object (skrf's container)")
    freq = rf.Frequency.from_f(freqs, unit="Hz")
    ntwk = rf.Network(frequency=freq, s=Z.reshape(-1, 1, 1))

    print("Fit")
    vf = rf.VectorFitting(ntwk)
    if max_iterations is not None:
        vf.max_iterations = max_iterations
    print(f"start with {n_pole} complex pairs")
    vf.vector_fit(
        n_poles_real=0,
        n_poles_cmplx=n_pole,
        fit_constant=True,
        fit_proportional=True,
    )

    print("Done")
    print("s_k (complex)")
    poles = vf.poles
    print("A_k (complex)")
    residues = vf.residues

    print(f"{vf.proportional_coeff=}")
    print(f"{vf.constant_coeff=}")

    print("Check quality")
    vf.plot_s_db()  # overlay fit vs original
    # plt.show()
    rms_error = vf.get_rms_error()
    if verbose:
        print(f"{rms_error=}")

    print("vf.poles = ", vf.poles)
    print("vf.residues = ", vf.residues)
    print("vf.proportional_coeff = ", vf.proportional_coeff)
    print("vf.constant_coeff = ", vf.constant_coeff)

    return poles, residues, rms_error, vf.proportional_coeff, vf.constant_coeff


@numba.njit()
def apply_single_pole(
    profile: np.ndarray, dt: float, pole: complex, residue: complex, voltage
):
    # phasor[n] = profile[n] + exp(p * dt) * phasor[n-1]
    # V[n] = 2 * Re(r * phasor[n])
    n_bins = len(profile)
    phasor = 0.0 + 0.0j
    decay = np.exp(pole * dt)

    for i in range(n_bins):
        profile_i_ = profile[i]
        phasor = phasor * decay + 0.5 * profile_i_
        voltage[i] += 2 * np.real(residue * phasor)
        phasor += 0.5 * profile_i_


@numba.njit(
    float64[:](float64[:], float64, complex128[:], complex128[:]),
    fastmath=True,
)
def apply_poles(profile, dt, poles, residues):
    voltage = np.zeros(len(profile))
    for i in range(len(residues)):
        apply_single_pole(profile, dt, poles[i], residues[i], voltage)
    return voltage


def get_test_data():
    hist_y, edges = np.histogram(
        np.concatenate(
            (
                np.random.randn(int(1e5)) * 1e-9,
                (np.random.randn(int(1e5)) * 1e-9) + 1e-7,
            ),
        ),
        bins=4096,
    )
    centers = edges[:-1] + np.diff(edges[:2]) / 2
    centers_extended = np.linspace(-0.2e-7, 3.5e-7, 100 * len(centers))
    hist_y_extended = np.interp(centers_extended, centers, hist_y)
    centers_extended = np.array(centers_extended, dtype=float)
    hist_y_extended = np.array(hist_y_extended, dtype=float)
    return centers_extended, hist_y_extended


class TestPole(unittest.TestCase):
    def test_pole(self):
        from blond.legacy.blond2.impedances.impedance_sources import Resonators

        freq = np.linspace(0, 1e9, 10000)
        res = Resonators(
            R_S=[1, 1, 2], frequency_R=[1e8, 2e8, 3e8], Q=[10, 20, 10]
        )
        res.imped_calc(freq)
        Z = res.impedance
        poles, residues, rms_error, proportional_coeff, constant_coeff = (
            get_poles(freqs=freq, Z=Z, n_pole=3)
        )

        centers, hist_y = get_test_data()
        centers -= centers.min()

        dt = np.diff(centers[:2])[0]

        if len(residues.shape) == 2:
            assert residues.shape[0] == 1
            residues = residues[0, :]
        # apply_poles(hist_y, dt, poles, residues)
        voltage = np.zeros_like(hist_y, dtype=float)
        state = np.zeros(len(poles) + 1, dtype=complex)
        state[-1] -= dt
        apply_poles2(
            profile=hist_y,
            profile_dts=centers,
            poles=poles,
            residues=residues,
            states=state,
            voltage=voltage,
            voltage_threaded=np.zeros(
                (numba.get_num_threads(), len(hist_y)), dtype=float
            ),
            update_on_bin=np.zeros(1, dtype=np.int32),
            factor=1.0
        )
        print("masked")
        print("-" * 79)
        start = int(0.15 * len(hist_y))
        stop = int(0.25 * len(hist_y))
        sel = slice(start, stop)
        voltage = np.zeros_like(hist_y, dtype=float)
        state = np.zeros(len(poles) + 1, dtype=complex)
        mask = np.ones(len(hist_y), bool)
        mask[sel] = False
        voltage_masked = voltage[mask]
        apply_poles2(
            profile=hist_y[mask],
            profile_dts=centers[mask],
            poles=poles,
            residues=residues,
            states=state,
            voltage=voltage_masked,
            voltage_threaded=np.zeros(
                (numba.get_num_threads(), len(voltage_masked)), dtype=float
            ),
            update_on_bin=np.array([0, start], dtype=np.int32),
            factor=1.0

        )
        voltage[mask] = voltage_masked

        t0 = time.time()
        # voltage = apply_poles(hist_y, dt, poles, residues)
        t1 = time.time()
        print()
        t_ploish = t1 - t0
        print("pole-ish", t_ploish, "s")
        res.wake_calc(centers)
        kernel = res.wake

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(centers, hist_y)
        plt.subplot(3, 1, 2)
        plt.plot(centers, voltage, "-", label="new")
        t0 = time.time()
        wake_convolve = fftconvolve(hist_y, kernel)
        t1 = time.time()
        t_fftconvolve = t1 - t0
        print("fftconvolve", t_fftconvolve, "s")
        print("ratio", t_fftconvolve / t_ploish, "x")
        plt.plot(
            centers, wake_convolve[: len(centers)], "--", label="fftconvolve"
        )
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("voltage")
        plt.subplot(3, 1, 3)
        plt.plot(
            centers,
            voltage - wake_convolve[: len(centers)],
            "--",
            label="fftconvolve",
        )

        plt.show()


if __name__ == "__main__":
    main()
