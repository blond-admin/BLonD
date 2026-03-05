import time
import unittest

import numba
import numpy as np
import skrf as rf
from matplotlib import pyplot as plt
from numba import complex128, float64
from scipy.signal import fftconvolve

from blond import backend
from blond.physics.impedances.induced_voltage_with_poles import apply_poles2
from blond.testing.helpers import pinned_values_helper


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
            factor=1.0,
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
            factor=1.0,
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

    def test_instable_pole(self):
        n = int(256)
        hist_y = np.zeros(n, float)
        voltage = np.zeros(n, float)
        hist_y[1] = 1
        centers = np.linspace(0, 0.5e-6, n, dtype=float)

        poles = np.array(
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
        residues = np.array(
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

        state = np.zeros((len(poles) + 1), complex)
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
            factor=1.0,
        )
        DEV_PLOT = False
        if DEV_PLOT:
            plt.subplot(2, 1, 1)
            plt.plot(hist_y)
            plt.subplot(2, 1, 2)
            plt.plot(voltage)
            plt.show()
        # pinned_values_helper(voltage, "voltage") # use this to generate
        voltage_pinned = [
            0.0,
            9148920350470.0,
            -14250767750131.273,
            3920924926629.75,
            8070029025660.5,
            -16410121880182.54,
            17461138493026.23,
            -10838131353252.188,
            -468154796170.3594,
            11448166186455.297,
            -17291958406489.658,
            15491135491903.43,
            -6911655559035.906,
            -4618447946014.766,
            14013062893254.844,
            -17168778067848.299,
            12757143082304.21,
            -2784761361690.172,
            -8316833325790.539,
            15660637337596.27,
            -16057331542775.729,
            9401680805408.21,
            1311815685480.9688,
            -11337274545914.617,
            16281526443396.46,
            -14029752517290.102,
            5648403021670.594,
            5118710506188.906,
            -13522886018843.793,
            15904146820619.32,
            -11283663030447.836,
            1763372393952.8984,
            8428753150585.547,
            -14814958669820.867,
            14633896721972.586,
            -8034116284325.164,
            -2020943356450.0078,
            11082369697735.781,
            -15184123473154.4,
            12579485818897.465,
            -4484716971016.133,
            -5487019829580.23,
            12938907722774.615,
            -14630178542206.291,
            9885778382618.379,
            -862571675533.7188,
            -8431107497660.109,
            13916572821076.652,
            -13234695361526.162,
            6759263765249.3125,
            2600589489070.8438,
            -10706806550521.117,
            14021175808972.46,
            -11148770239683.445,
            3423446621879.914,
            5709544906984.49,
            -12229087890989.658,
            13308157192602.203,
            -8539448064747.477,
            85264279336.15234,
            8301849259641.167,
            -12944830501554.887,
            11854713226040.791,
            -5582440000228.236,
            -3055060521358.3887,
            10240117974285.398,
            -12842373064618.979,
            9782730880778.225,
            -2478809908703.4023,
            -5810869034971.822,
            11440745511029.387,
            -11981265023488.932,
            7266366729043.182,
            565520547204.5898,
            -8040483164810.616,
            11889395669560.164,
            -10476961217659.32,
            4495321845499.174,
            3371390169215.7812,
            -9650605849826.418,
            11613899320303.852,
            -8462416411030.7,
            1650040612258.2285,
            5785787754328.402,
            -10578407967169.383,
            10667524162663.764,
            -6084755595469.057,
            -1091574771549.1992,
            7678703458901.118,
            -10801520826056.148,
            9147604175280.947,
            -3517257687940.455,
            -3561105671627.0884,
            8966252421164.434,
            -10360267992932.443,
            7198956762622.707,
            -940397724915.0,
            -5626153201793.54,
            9622718222734.934,
            -9343379050778.832,
            4981464843249.866,
            1486374397429.6138,
            -7194822806840.126,
            9659002292253.438,
            -7856105420862.971,
            2648646175031.639,
            3625434139534.286,
            -8202269220544.933,
            9110106070900.402,
            -6018720884855.276,
            354159422576.0713,
            5358203092580.417,
            -8619894922162.311,
            8052291433547.578,
            -3978054977127.301,
            -1753435298655.1616,
            6605070869880.39,
            -8474884860182.403,
            6605985132320.264,
            -1890153754616.8647,
            -3555159276419.435,
            7336817408916.557,
            -7837212278570.16,
            4905334031999.518,
            107127757242.54346,
            -4967368003082.122,
            7555754012573.277,
            -6790282985277.521,
            3077904699758.3994,
            1895301949374.9045,
            -5929442471481.069,
            7284434806812.636,
            -5430680427113.628,
            1252740139965.4346,
            3370684057256.1025,
            -6411739149881.44,
            6583074948897.265,
            -3881696023620.0405,
            -442103112060.9516,
            4462345925222.406,
            -6434986136312.904,
            5553793705735.775,
            -2277401486275.734,
            -1903800250508.8506,
            5144512554984.053,
            -6058339398794.824,
            4310074499234.6123,
            -733841377488.5497,
            -3062161828735.1025,
            5418337479062.67,
            -5349384631514.553,
            2955559493316.883,
            652028211125.8098,
            -3866021377199.8203,
            5299480549711.862,
            -4384702871978.716,
            1595332121979.9614,
            1793843585689.9976,
            -4289352494983.9863,
            4837477064971.186,
            -3268138595216.711,
            337845794741.1358,
            2632630643148.308,
            -4353077350015.4326,
            4123480448048.957,
            -2116275737831.7546,
            -731135701675.0709,
            3152270758196.495,
            -4114097653151.078,
            3256914815657.1084,
            -1024754556490.9181,
            -1559001517666.5835,
            3359781279535.432,
            -3630228308922.33,
            2319775990943.1763,
            -67150571187.5802,
            -2108246287660.4556,
            3268113018144.551,
            -2960641811338.936,
            1393814659978.222,
            688627931192.7747,
            -2359244402450.5503,
            2920430498455.714,
            -2194636605408.9128,
            571150493126.3987,
            1196620954198.8696,
            -2339697274618.138,
            2407021655873.364,
            -1439977454470.4026,
            -77483467621.3765,
            1456889793263.0366,
            -2116145302602.06,
            1823862157553.7566,
            -775409352032.5077,
            -521652551362.958,
            1490596057973.9844,
            -1742166362129.561,
            1230618913706.768,
            -246326074798.7576,
            -740730026978.3314,
            1305378145779.7856,
            -1252287042541.5554,
            679701779779.9761,
            97273907796.01282,
            -714957658521.6832,
            932358377203.4551,
            -723682863259.202,
            257169611346.6408,
            214674148137.05,
            -487307267210.7654,
            490284530510.3861,
            -292116962194.5194,
            36007381655.1582,
            147486901876.69708,
            -201007864750.7719,
            148286489956.02426,
            -57431719218.59968,
            -9372240601.449677,
            28108249486.455444,
            -13368986886.351044,
            -4538789941.804092,
            5242364298.621567,
            10453670385.01825,
            -27289538955.212463,
            30952932762.34439,
            -19127114063.692688,
            1165441020.11969,
            10953610242.477478,
            -11916766386.169983,
            5600337424.166199,
            -173859384.60786438,
            845980956.939148,
            -6247469046.209259,
            10665606797.20642,
            -9453561652.09329,
            2848824212.591263,
            4640317948.342529,
            -8227710978.969543,
            6611109349.498962,
            -2514894291.824051,
            -46044478.21047974,
            -800235141.4506226,
            3512849043.7341003,
            -4838260113.231747,
            2788628354.190796,
            1677676266.6348877,
            -5556360004.56778,
            6336237556.705429,
            -3967001014.389511,
            687208553.2884979,
            981985207.5304565,
            -291654146.6332092,
            -1322764368.5058823,
            1686699149.599823,
            145792106.32122803,
            -3080779714.809143,
        ]
        np.testing.assert_allclose(
            voltage,
            voltage_pinned,
            rtol=1e-6 if backend.float == np.float32 else 1e-12,
        )


if __name__ == "__main__":
    main()
