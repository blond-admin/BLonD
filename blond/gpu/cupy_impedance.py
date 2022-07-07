from __future__ import division, print_function
from builtins import range, object
import numpy as np
import cupy as cp
from scipy.constants import e
from ..utils import bmath as bm
from ..gpu.cupy_array import get_gpuarray
from ..gpu.cupy_butils_wrap import  gpu_interp

from ..impedances.impedance import TotalInducedVoltage, _InducedVoltage, \
    InducedVoltageFreq, InducedVoltageTime, \
    InductiveImpedance

try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing




class GpuTotalInducedVoltage(TotalInducedVoltage):

    @property
    def induced_voltage(self):
        return self.induced_voltage_obj.my_array

    @induced_voltage.setter
    def induced_voltage(self, value):
        self.induced_voltage_obj.my_array = value

    @property
    def dev_induced_voltage(self):
        return self.induced_voltage_obj.dev_my_array

    @dev_induced_voltage.setter
    def dev_induced_voltage(self, value):
        self.induced_voltage_obj.dev_my_array = value

    def induced_voltage_sum(self):
        """
        Method to sum all the induced voltages in one single array.
        """
        beam_spectrum_dict = {}
        self.dev_induced_voltage = get_gpuarray(self.profile.n_slices,bm.precision.real_t)
        self.dev_induced_voltage.fill(0)
        for induced_voltage_object in self.induced_voltage_list:
            induced_voltage_object.induced_voltage_generation(
                beam_spectrum_dict)
            if not hasattr(induced_voltage_object, 'dev_induced_voltage'):
                induced_voltage_object.dev_induced_voltage = cp.array(
                    induced_voltage_object.induced_voltage)
            assert induced_voltage_object.dev_induced_voltage.dtype == bm.precision.real_t
            self.dev_induced_voltage += induced_voltage_object.dev_induced_voltage[:self.profile.n_slices]

    def track(self):
        """
        GPU implementation of TotalInducedVoltage.track
        """

        self.induced_voltage_sum()
        bm.linear_interp_kick(self.beam.dev_dt, self.beam.dev_dE,
                              self.dev_induced_voltage,
                              self.profile.dev_bin_centers,
                              self.beam.Particle.charge,
                              acceleration_kick=0.)
        self.beam.dE_obj.invalidate_cpu()


class GpuInducedVoltage(_InducedVoltage):

    @property
    def mtw_memory(self):
        return self.mtw_memory_obj.my_array

    @mtw_memory.setter
    def mtw_memory(self, value):
        self.mtw_memory_obj.my_array = value

    @property
    def dev_mtw_memory(self):
        return self.mtw_memory_obj.dev_my_array

    @dev_mtw_memory.setter
    def dev_mtw_memory(self, value):
        self.mtw_memory_obj.dev_my_array = value

    # total_impedance

    @property
    def total_impedance(self):
        return self.total_impedance_obj.my_array

    @total_impedance.setter
    def total_impedance(self, value):
        self.total_impedance_obj.my_array = value

    @property
    def dev_total_impedance(self):
        return self.total_impedance_obj.dev_my_array

    @dev_total_impedance.setter
    def dev_total_impedance(self, value):
        self.total_impedance_obj.dev_my_array = value

    def induced_voltage_1turn(self, beam_spectrum_dict={}):
        """
        GPU implementation of induced_voltage_1turn
        """

        if self.n_fft not in beam_spectrum_dict:
            self.profile.beam_spectrum_generation(self.n_fft)
            beam_spectrum_dict[self.n_fft] = self.profile.dev_beam_spectrum
        beam_spectrum = beam_spectrum_dict[self.n_fft]
        #with timing.timed_region('serial:indVolt1Turn'):
        inp = get_gpuarray(beam_spectrum.size, bm.precision.complex_t)
        inp = self.dev_total_impedance * beam_spectrum
        my_res = bm.irfft(inp, caller_id=id(self))
        self.dev_induced_voltage = get_gpuarray(self.n_induced_voltage, bm.precision.real_t)
        a = bm.precision.real_t(-self.beam.Particle.charge * e * self.beam.ratio)
        self.dev_induced_voltage = a * my_res

    def induced_voltage_mtw(self, beam_spectrum_dict={}):
        """
        GPU implementation of induced_voltage_mtw
        """

        self.shift_trev()
        self.induced_voltage_1turn(beam_spectrum_dict)

        slice = slice(self.n_induced_voltage - self.front_wake_buffer, self.dev_induced_voltage.size)
        self.dev_induced_voltage[slice].fill(0)
                      
        self.dev_mtw_memory += self.dev_induced_voltage[:self.n_induced_voltage]

        self.dev_induced_voltage = get_gpuarray(self.n_induced_voltage, bm.precision.real_t)
        self.dev_induced_voltage = self.dev_mtw_memory[:self.n_induced_voltage]

    #@timing.timeit(key='serial:shift_trev_freq')
    def shift_trev_freq(self):
        """
        GPU implementation of shift_trev_freq
        """
        t_rev = self.RFParams.t_rev[self.RFParams.counter[0]]
        dev_induced_voltage_f = bm.rfft(self.dev_mtw_memory, self.n_mtw_fft)
        dev_induced_voltage_f *= cp.exp(self.dev_omegaj_mtw * t_rev)

        self.dev_mtw_memory = get_gpuarray(self.n_mtw_memory, bm.precision.real_t)
        dummy = bm.irfft(dev_induced_voltage_f, caller_id=id(self))
        self.dev_mtw_memory = dummy[:self.n_mtw_memory]
        slice = slice(-int(self.buffer_size), None, None)
        self.dev_mtw_memory[slice].fill(0)
                      

    #@timing.timeit(key='serial:shift_trev_time')
    def shift_trev_time(self):
        """
        GPU implementation of shift_trev_time
        """
        t_rev = self.RFParams.t_rev[self.RFParams.counter[0]]
        inc_dev_time_mtw = get_gpuarray(self.dev_time_mtw.size, bm.precision.real_t)
        inc_dev_time_mtw = self.dev_time_mtw
        inc_dev_time_mtw += t_rev
        self.dev_mtw_memory = gpu_interp(inc_dev_time_mtw,
                                         self.dev_time_mtw, self.dev_mtw_memory,
                                         left=0, right=0, caller_id=id(self))


class GpuInducedVoltageFreq(GpuInducedVoltage, InducedVoltageFreq):
    pass


class GpuInducedVoltageTime(GpuInducedVoltage, InducedVoltageTime):

    def sum_wakes(self, time_array):
        self.total_wake = np.zeros(time_array.shape, dtype=bm.precision.real_t)
        for wake_object in self.wake_source_list:
            wake_object.wake_calc(time_array)
            self.total_wake += wake_object.wake

        dev_total_wake = cp.array(self.total_wake.astype(bm.precision.real_t))
        self.dev_total_impedance = bm.rfft(self.total_wake, self.n_fft, caller_id=id(self))


class GpuInductiveImpedance(GpuInducedVoltage, InductiveImpedance):

    #@timing.timeit(key='serial:InductiveImped')
    def induced_voltage_1turn(self, beam_spectrum_dict={}):
        """
        GPU implementation for induced_voltage_1turn
        """

        index = self.RFParams.counter[0]

        sv = - (self.beam.Particle.charge * e / (2 * np.pi) *
                self.beam.ratio * self.Z_over_n[index] *
                self.RFParams.t_rev[index] / self.profile.bin_size)

        induced_voltage = self.profile.beam_profile_derivative(
            self.deriv_mode, caller_id=id(self))[1]
        induced_voltage = sv * induced_voltage

        self.dev_induced_voltage = get_gpuarray(self.n_induced_voltage,bm.precision.real_t)
        self.dev_induced_voltage = induced_voltage[:self.n_induced_voltage]
