import subprocess
import unittest

import numpy as np


def clone_gitlab_repo(repo_url: str, destination_folder: str):
    try:
        # Run the git clone command
        print(f"Cloning {repo_url}")
        prc = subprocess.run(['git', 'clone', repo_url, destination_folder], check=True)
        assert prc.returncode == 0
        print(f"Repository cloned to {destination_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")


def rf_noise_repo_is_missing():
    try:
        from blond.interfaces.rf_noise_cpp.wrap_rf_noise import rf_noise, _local_path  # only possible after setUp
        return False, ""
    except FileNotFoundError as exc:
        return True, str(exc)


class RfNoise(unittest.TestCase):

    @unittest.skipIf(*rf_noise_repo_is_missing())
    def test_flat_noise(self):
        from blond.interfaces.rf_noise_cpp.wrap_rf_noise import rf_noise  # only possible after setUp
        ys = np.ones(10)
        xs = np.linspace(0.0, 1.0, len(ys))  # flat denisity within band
        N = 20000000
        f_low = 10 * np.ones(N)  # flat
        f_high = 20 * np.ones(N)  # flat
        results = np.empty(N)
        results = rf_noise(
            frequency_high=f_high,
            frequency_low=f_low,
            gain_x=xs,
            gain_y=ys,
            n_source=2048,
            n_pnt_min=8,
            r_seed=0,
            sampling_rate=11245.49,
            rms=1.0,
            phase_array=results,
        )
        freq = np.fft.rfftfreq(len(results), 1 / 11245.49)
        ampl = np.fft.rfft(results)
        threshold = 0.1 * np.max(np.abs(ampl))

        sel = np.abs(ampl) > threshold
        ampl_masked = ampl[sel]
        freq_masked = freq[sel]
        delta_x = 0.01
        DEV_DEBUG = False
        if DEV_DEBUG:
            from matplotlib import pyplot as plt
            plt.title("rf_noise_wrapper Python ctypes interface")

            plt.plot(freq, ampl)
            plt.plot(freq_masked, ampl_masked)
            plt.axvline(f_low[0] - delta_x, label="fLo", color="red")
            plt.axvline(f_high[0] + delta_x, label="fHi", color="red")
            plt.xlim(5, 25)
            plt.legend(loc="upper left")
            plt.show()
        self.assertTrue(np.all((freq_masked > (f_low[0] - delta_x)) & (freq_masked < (f_high[0] + delta_x))))

    def test_docstring_example(self):
        from blond.interfaces.rf_noise_cpp.wrap_rf_noise import rf_noise, _local_path  # only possible after setUp
        ys = np.loadtxt(_local_path / "lhc_spectrum.txt")
        xs = np.linspace(0.0, 1.0, len(ys))
        N = 20_000_000
        f_low = np.linspace(10, 100, N)
        f_high = np.linspace(20, 200, N)
        results = np.empty(N)
        results = rf_noise(
            frequency_high=f_high,
            frequency_low=f_low,
            gain_x=xs,
            gain_y=ys,
            n_source=2048,
            n_pnt_min=8,
            r_seed=0,
            sampling_rate=11245.49,
            rms=1.0,
            phase_array=results,
        )
        from matplotlib import pyplot as plt
        plt.title("rf_noise_wrapper Python ctypes interface")
        plt.specgram(results, Fs=11245.49, NFFT=int(20000000 / 1000), label="specgram(results)")
        xxx = np.linspace(0, 20000000 / 11245.49, 20000000)
        plt.plot(xxx, f_low, label="fLo", color="red")
        plt.plot(xxx, f_high, label="fHi", color="red")
        plt.ylim(0, 2 * f_high[-1])
        plt.legend(loc="upper left")
        # plt.show()


if __name__ == '__main__':
    unittest.main()
