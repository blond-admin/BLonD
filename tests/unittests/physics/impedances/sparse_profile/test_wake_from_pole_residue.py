import unittest

import numpy as np
import skrf as rf
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve

from blond import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.handle_results.helpers import callers_relative_path


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
    vf = rf.vectorFitting.VectorFitting(ntwk)
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

        poles_b = backend.array(poles)
        residues_b = backend.array(residues)
        centers_b = backend.array(centers)
        hist_y_b = backend.array(hist_y)
        n = len(hist_y)

        voltage = backend.zeros(n, dtype=float)
        state = backend.zeros(len(poles_b) + 1, dtype=complex)
        state[-1] -= dt
        backend.specials.wake_from_pole_residue(
            profile=hist_y_b,
            profile_dts=centers_b,
            poles=poles_b,
            residues=residues_b,
            is_counterrotating_beam=False,
            counterrotating_pole_signs=backend.ones_like(
                poles_b, dtype=backend.float
            ),
            states=state,
            voltage=voltage,
            voltage_threaded=backend.zeros(
                (backend.specials.get_max_threads(), n), dtype=float
            ),
            update_on_bin=backend.array(np.zeros(1, dtype=np.int32)),
            factor=1.0,
        )
        voltage_full_cpu = copy_to_cpu(voltage)
        print("masked")
        print("-" * 79)
        start = int(0.15 * n)
        stop = int(0.25 * n)
        sel = slice(start, stop)
        voltage = backend.zeros(n, dtype=float)
        state = backend.zeros(len(poles_b) + 1, dtype=complex)
        mask_np = np.ones(n, bool)
        mask_np[sel] = False
        mask_b = backend.array(mask_np)
        voltage_masked = voltage[mask_b]
        backend.specials.wake_from_pole_residue(
            profile=hist_y_b[mask_b],
            profile_dts=centers_b[mask_b],
            poles=poles_b,
            residues=residues_b,
            is_counterrotating_beam=False,
            counterrotating_pole_signs=backend.ones_like(
                poles_b, dtype=backend.float
            ),
            states=state,
            voltage=voltage_masked,
            voltage_threaded=backend.zeros(
                (backend.specials.get_max_threads(), len(voltage_masked)),
                dtype=float,
            ),
            update_on_bin=backend.array(np.array([0, start], dtype=np.int32)),
            factor=1.0,
        )
        voltage[mask_b] = voltage_masked

        res.wake_calc(centers)
        kernel = res.wake
        wake_convolve = fftconvolve(hist_y, kernel)

        ref = wake_convolve[: len(centers)]
        atol = 0.01 * np.max(np.abs(ref))
        np.testing.assert_allclose(
            voltage_full_cpu,
            ref,
            rtol=0.05,
            atol=atol,
            err_msg="non-masked wake_from_pole_residue must match fftconvolve reference",
        )
        voltage_cpu = copy_to_cpu(voltage)
        np.testing.assert_allclose(
            voltage_cpu[mask_np],
            ref[mask_np],
            rtol=0.05,
            atol=atol,
            err_msg="masked wake_from_pole_residue must match fftconvolve reference in unmasked region",
        )

        DEV_DRAW = False
        if DEV_DRAW:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(centers, hist_y)
            plt.subplot(3, 1, 2)
            plt.plot(centers, copy_to_cpu(voltage), "-", label="new")
            plt.plot(centers, ref, "--", label="fftconvolve")
            plt.legend()
            plt.xlabel("time")
            plt.ylabel("voltage")
            plt.subplot(3, 1, 3)
            plt.plot(
                centers, copy_to_cpu(voltage) - ref, "--", label="difference"
            )
            plt.show()

    def test_instable_pole(self):
        n = int(256)
        hist_y = backend.zeros(n, dtype=float)
        hist_y[1] = 1
        voltage = backend.zeros(n, dtype=float)
        centers = backend.array(np.linspace(0, 0.5e-6, n, dtype=float))

        poles = backend.array(
            np.array(
                [
                    -2623831.59946355 + 1.39099415e09j,
                    -3324980.21266537 + 1.37641478e09j,
                    -2627142.66632887 + 1.12342679e09j,
                    -4057802.76251799 + 1.36173266e09j,
                    -3332053.6956318 + 1.13800877e09j,
                    -4859598.51825629 + 1.34699927e09j,
                    -4069084.51675692 + 1.15269216e09j,
                    -5769309.76869008 + 1.33225090e09j,
                    -4876008.93418745 + 1.16742610e09j,
                    -5792260.22147667 + 1.18217408e09j,
                    -6839639.82479881 + 1.31752969e09j,
                    -6871261.85241204 + 1.19689324e09j,
                    -8150077.00435694 + 1.30291029e09j,
                    -8193661.34867393 + 1.21150693e09j,
                    -9814345.82669513 + 1.28856816e09j,
                    -9873890.22060673 + 1.22583392e09j,
                    -11873023.777707 + 1.27495791e09j,
                    -11943110.44842772 + 1.23940612e09j,
                    -13657970.95550037 + 1.26278082e09j,
                    -13693795.03125995 + 1.25153131e09j,
                ]
            )
        )
        residues = backend.array(
            np.array(
                [
                    -6.84660023e09 - 4.37025733e08j,
                    -1.28770839e10 + 2.69841169e09j,
                    -6.71349630e09 + 1.93609079e09j,
                    -2.31568303e10 + 8.46750710e09j,
                    -1.33672712e10 + 1.40672867e08j,
                    -4.31080554e10 + 2.14496055e10j,
                    -2.48388163e10 - 3.37893882e09j,
                    -8.56203549e10 + 5.57134298e10j,
                    -4.75666252e10 - 1.20580947e10j,
                    -9.76136408e10 - 3.73889757e10j,
                    -1.83602927e11 + 1.65101397e11j,
                    -2.19999294e11 - 1.27368736e11j,
                    -3.97378936e11 + 6.08181441e11j,
                    -5.30711943e11 - 5.36457411e11j,
                    -1.86501649e11 + 2.83061942e12j,
                    -7.46439606e11 - 2.88640952e12j,
                    1.29225109e13 + 7.08751464e12j,
                    1.23327280e13 - 9.52068717e12j,
                    -1.00567070e13 - 5.67636567e13j,
                    -3.42326842e12 + 5.91090417e13j,
                ]
            )
        )

        state = backend.zeros(len(poles) + 1, dtype=complex)
        backend.specials.wake_from_pole_residue(
            profile=hist_y,
            profile_dts=centers,
            poles=poles,
            residues=residues,
            is_counterrotating_beam=False,
            counterrotating_pole_signs=backend.ones_like(
                poles, dtype=backend.float
            ),
            states=state,
            voltage=voltage,
            voltage_threaded=backend.zeros(
                (backend.specials.get_max_threads(), n), dtype=float
            ),
            update_on_bin=backend.array(np.zeros(1, dtype=np.int32)),
            factor=1.0,
        )
        DEV_PLOT = False
        if DEV_PLOT:
            plt.subplot(2, 1, 1)
            plt.plot(copy_to_cpu(hist_y))
            plt.subplot(2, 1, 2)
            plt.plot(copy_to_cpu(voltage))
            plt.show()
        # pinned_values_helper(voltage, "voltage") # use this to generate
        filepath = callers_relative_path(
            "resources/voltage_pinned.txt", stacklevel=1
        )
        voltage_pinned = np.loadtxt(filepath)
        np.testing.assert_allclose(
            copy_to_cpu(voltage),
            voltage_pinned,
            rtol=1e-6 if backend.float == np.float32 else 1e-10,
        )


class TestWakeFromPoleResidueBranches(unittest.TestCase):
    """Targeted tests for branch coverage inside `wake_from_pole_residue`."""

    def test_bin_i_zero_uses_t_start(self):
        """First-bin branch: ``t_jump = profile_dts[0] - t_start``.

        Use two different ``states[-1]`` (a.k.a. ``t_start``) values and verify
        that the resulting voltage differs — i.e. the ``bin_i == 0`` branch
        actually consumes ``t_start`` instead of ignoring it.
        """
        n = 64
        hist_y = backend.zeros(n, dtype=float)
        hist_y[0] = 1.0
        centers = backend.array(np.linspace(0.0, 1e-9, n, dtype=float))
        bin_dt = float(centers[1] - centers[0])

        poles = backend.array(
            np.array([-1e8 + 2 * np.pi * 1e9j], dtype=complex)
        )
        residues = backend.array(np.array([1.0 + 0.5j], dtype=complex))

        def _run(t_start) -> np.ndarray:
            voltage = backend.zeros(n, dtype=float)
            state = backend.zeros(len(poles) + 1, dtype=complex)
            # Non-zero initial pole state so the `state *= exp(pole * t_jump)`
            # multiplication actually depends on `t_jump = dts[0] - t_start`.
            state[0] = 1.0 + 0.5j
            state[-1] = t_start
            backend.specials.wake_from_pole_residue(
                profile=hist_y,
                profile_dts=centers,
                poles=poles,
                residues=residues,
                is_counterrotating_beam=False,
                counterrotating_pole_signs=backend.ones_like(
                    poles, dtype=backend.float
                ),
                states=state,
                voltage=voltage,
                voltage_threaded=backend.zeros(
                    (backend.specials.get_max_threads(), n), dtype=float
                ),
                update_on_bin=backend.array(np.zeros(1, dtype=np.int32)),
                factor=1.0,
            )
            return voltage

        # Aligned t_start: t_jump = profile_dts[0] - (profile_dts[0] - bin_dt) = bin_dt
        voltage_aligned = _run(t_start=centers[0] - bin_dt)
        # Different t_start → different t_jump → different output
        voltage_offset = _run(t_start=centers[0] - 5 * bin_dt)
        self.assertFalse(
            np.allclose(
                copy_to_cpu(voltage_aligned), copy_to_cpu(voltage_offset)
            ),
            "bin_i==0 branch must consume t_start; outputs should differ.",
        )

    def test_multiple_update_on_bin_advances_index(self):
        """``i_update < len(update_on_bin)`` branch: piecewise calls match a single call.

        Splitting a profile at an internal index via ``update_on_bin = [0, k]``
        must produce the same induced voltage as a single contiguous call. This
        forces the inner branch ``i_update < len(update_on_bin)`` to fire on the
        first update so that ``update_on_bin_i`` advances to ``k``.
        """
        n = 32
        rng = np.random.default_rng(0)
        hist_y = backend.array(rng.standard_normal(n))
        centers = backend.array(np.linspace(0.0, 1e-9, n, dtype=float))

        poles = backend.array(
            np.array(
                [-1e8 + 2 * np.pi * 1e9j, -2e8 + 2 * np.pi * 1.5e9j],
                dtype=complex,
            )
        )
        residues = backend.array(
            np.array([1.0 + 0.5j, 0.7 - 0.2j], dtype=complex)
        )

        bin_dt = float(centers[1] - centers[0])

        # Single call with one update_on_bin entry
        v_single = backend.zeros(n, dtype=float)
        state_single = backend.zeros(len(poles) + 1, dtype=complex)
        state_single[-1] = centers[0] - bin_dt
        backend.specials.wake_from_pole_residue(
            profile=hist_y,
            profile_dts=centers,
            poles=poles,
            residues=residues,
            is_counterrotating_beam=False,
            counterrotating_pole_signs=backend.ones_like(
                poles, dtype=backend.float
            ),
            states=state_single,
            voltage=v_single,
            voltage_threaded=backend.zeros(
                (backend.specials.get_max_threads(), n), dtype=float
            ),
            update_on_bin=backend.array(np.zeros(1, dtype=np.int32)),
            factor=1.0,
        )

        # Split call with two update_on_bin entries — exercises the
        # `i_update < len(update_on_bin)` True branch.
        k = n // 2
        v_split = backend.zeros(n, dtype=float)
        state_split = backend.zeros(len(poles) + 1, dtype=complex)
        state_split[-1] = centers[0] - bin_dt
        backend.specials.wake_from_pole_residue(
            profile=hist_y,
            profile_dts=centers,
            poles=poles,
            residues=residues,
            is_counterrotating_beam=False,
            counterrotating_pole_signs=backend.ones_like(
                poles, dtype=backend.float
            ),
            states=state_split,
            voltage=v_split,
            voltage_threaded=backend.zeros(
                (backend.specials.get_max_threads(), n), dtype=float
            ),
            update_on_bin=backend.array(np.array([0, k], dtype=np.int32)),
            factor=1.0,
        )

        np.testing.assert_allclose(
            copy_to_cpu(v_split), copy_to_cpu(v_single), rtol=1e-10, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
