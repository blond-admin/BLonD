import numpy as np
from ..utils import bmath as bm
from ..gpu.gpu_cache import get_gpuarray
from ..gpu.gpu_butils_wrap import gpu_diff, cugradient, gpu_copy_d2d, gpu_interp, d_multscalar

from ..beam.profile import Profile
from pycuda import gpuarray
from ..gpu import grid_size, block_size


try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing


class GpuProfile(Profile):

    # We change the arrays we want to keep also in the Gpu to CGA
    # To do that we need to define them like the ExampleClass in the
    # blond/gpu/cpu_gpu_array.py file

    # bin_centers

    @property
    def bin_centers(self):
        return self.bin_centers_obj.my_array

    @bin_centers.setter
    def bin_centers(self, value):
        self.bin_centers_obj.my_array = value

    @property
    def dev_bin_centers(self):
        return self.bin_centers_obj.dev_my_array

    @dev_bin_centers.setter
    def dev_bin_centers(self, value):
        self.bin_centers_obj.dev_my_array = value

    # n_macroparticles

    @property
    def n_macroparticles(self):
        return self.n_macroparticles_obj.my_array

    @n_macroparticles.setter
    def n_macroparticles(self, value):
        self.n_macroparticles_obj.my_array = value

    @property
    def dev_n_macroparticles(self):
        return self.n_macroparticles_obj.dev_my_array

    @dev_n_macroparticles.setter
    def dev_n_macroparticles(self, value):
        self.n_macroparticles_obj.dev_my_array = value

    # beam_spectrum

    @property
    def beam_spectrum(self):
        return self.beam_spectrum_obj.my_array

    @beam_spectrum.setter
    def beam_spectrum(self, value):
        self.beam_spectrum_obj.my_array = value

    @property
    def dev_beam_spectrum(self):
        return self.beam_spectrum_obj.dev_my_array

    @dev_beam_spectrum.setter
    def dev_beam_spectrum(self, value):
        self.beam_spectrum_obj.dev_my_array = value

    # beam_spectrum_freq

    @property
    def beam_spectrum_freq(self):
        return self.beam_spectrum_freq_obj.my_array

    @beam_spectrum_freq.setter
    def beam_spectrum_freq(self, value):
        self.beam_spectrum_freq_obj.my_array = value

    @property
    def dev_beam_spectrum_freq(self):
        return self.beam_spectrum_freq_obj.dev_my_array

    @dev_beam_spectrum_freq.setter
    def dev_beam_spectrum_freq(self, value):
        self.beam_spectrum_freq_obj.dev_my_array = value

    @timing.timeit(key='comp:histo')
    def _slice(self, reduce=True):
        """
        Gpu Equivalent for _slice
        """

        bm.slice(self.Beam.dev_dt, self.dev_n_macroparticles, self.cut_left, self.cut_right)
        self.n_macroparticles_obj.invalidate_cpu()

    @timing.timeit(key='serial:beam_spectrum_gen')
    def beam_spectrum_generation(self, n_sampling_fft):
        """
        Gpu Equivalent for beam_spectrum_generation
        """
        temp = bm.rfft(self.dev_n_macroparticles, n_sampling_fft)
        self.dev_beam_spectrum = temp

    def beam_profile_derivative(self, mode='gradient', caller_id=None):
        """
        Gpu Equivalent for beam_profile_derivative
        """
        x = self.bin_centers
        dist_centers = x[1] - x[0]
        if mode == 'filter1d':
            raise RuntimeError('filted1d mode is not supported in GPU.')
        elif mode == 'gradient':
            if caller_id:
                derivative = get_gpuarray(
                    (x.size, bm.precision.real_t, caller_id, 'der'), True)
            else:
                derivative = gpuarray.empty(x.size, dtype=bm.precision.real_t)
            cugradient(bm.precision.real_t(dist_centers), self.dev_n_macroparticles,
                       derivative, np.int32(x.size), block=block_size, grid=grid_size)
        elif mode == 'diff':
            if caller_id:
                derivative = get_gpuarray(
                    (x.size, bm.precision.real_t, caller_id, 'der'), True)
            else:
                derivative = gpuarray.empty(
                    self.dev_n_macroparticles.size - 1, bm.precision.real_t)
            gpu_diff(self.dev_n_macroparticles, derivative, dist_centers)
            diff_centers = get_gpuarray(
                (self.dev_bin_centers.size - 1, bm.precision.real_t, caller_id, 'dC'))
            gpu_copy_d2d(diff_centers, self.dev_bin_centers, slice=slice(0, -1))

            diff_centers = diff_centers + dist_centers / 2
            derivative = gpu_interp(self.dev_bin_centers, diff_centers, derivative)
        else:
            # ProfileDerivativeError
            raise RuntimeError('Option for derivative is not recognized.')

        return x, derivative

    def reduce_histo(self, dtype=np.uint32):
        """
        Gpu Equivalent for reduce_histo
        """
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        worker.sync()
        if self.Beam.is_splitted:
            with timing.timed_region('serial:conversion'):
                my_n_macroparticles = self.n_macroparticles.astype(
                    np.uint32, order='C')

            worker.allreduce(my_n_macroparticles, dtype=np.uint32, operator='custom_sum')

            with timing.timed_region('serial:conversion'):
                self.n_macroparticles = my_n_macroparticles.astype(dtype=bm.precision.real_t, order='C', copy=False)

    @timing.timeit(key='serial:scale_histo')
    def old_scale_histo(self):
        """
        Gpu Equivalent for scale_histo
        """
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        if self.Beam.is_splitted:
            self.n_macroparticles = self.n_macroparticles * worker.workers
            self.n_macroparticles_obj.invalidate_gpu()
    
    @timing.timeit(key='serial:scale_histo')
    def scale_histo(self):
        """
        Gpu Equivalent for scale_histo
        """
        if not bm.mpiMode():
            raise RuntimeError(
                'ERROR: Cannot use this routine unless in MPI Mode')

        from ..utils.mpi_config import worker
        if self.Beam.is_splitted:
            d_multscalar(self.dev_n_macroparticles, self.dev_n_macroparticles, bm.precision.real_t(worker.workers))
            self.n_macroparticles_obj.invalidate_cpu()