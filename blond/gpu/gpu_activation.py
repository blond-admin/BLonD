from ..beam.profile import Profile
from ..input_parameters.rf_parameters import RFStation
from ..llrf.beam_feedback import BeamFeedback
from ..trackers.tracker import RingAndRFTracker
from ..utils import bmath as bm
from ..beam import beam
from ..impedances.impedance import TotalInducedVoltage, _InducedVoltage, InducedVoltageTime, InducedVoltageFreq, \
    InductiveImpedance, InducedVoltageResonator
import numpy as np


# Beam
def use_gpu_beam(self):
    # There has to be a previous call to bm.use_gpu() to enable gpu mode
    from ..gpu import gpu_beam as gb
    from ..gpu.cupy_array import CGA

    if self.__class__ == gb.GpuBeam:
        return

    self.dE_obj = CGA(self.dE)

    self.dt_obj = CGA(self.dt)

    self.id_obj = CGA(self.id, dtype2=bm.precision.real_t)
    self.__class__ = gb.GpuBeam


beam.Beam.use_gpu = use_gpu_beam


# Profile
def use_gpu_profile(self):
    # There has to be a previous call to bm.use_gpu() to enable gpu mode

    from ..gpu.cupy_array import CGA
    from ..gpu.cupy_profile import GpuProfile
    if self.__class__ == GpuProfile:
        return
    old_slice = self._slice

    # bin_centers to gpu
    self.bin_centers_obj = CGA(self.bin_centers)

    # n_macroparticles to gpu
    self.n_macroparticles_obj = CGA(self.n_macroparticles, dtype2=np.int32)
    # self.n_macroparticles_obj = CGA(self.n_macroparticles)

    # beam_spectrum to gpu
    self.beam_spectrum_obj = CGA(self.beam_spectrum)

    # beam_spectrum_freq to gpu
    self.beam_spectrum_freq_obj = CGA(self.beam_spectrum_freq)
    self.__class__ = GpuProfile

    for i in range(len(self.operations)):
        if self.operations[i] == old_slice:
            self.operations[i] = self._slice

    self.dev_n_macroparticles


Profile.use_gpu = use_gpu_profile


# Tracker
def use_gpu_tracker(self):
    from ..gpu.cupy_tracker import GpuRingAndRFTracker
    from ..gpu.cupy_array import CGA

    self.rf_voltage_obj = CGA(np.array([]))
    self.__class__ = GpuRingAndRFTracker
    if self.profile:
        self.profile.use_gpu()
    if self.totalInducedVoltage:
        self.totalInducedVoltage.use_gpu()
    if self.beam:
        self.beam.use_gpu()
    if self.beamFB:
        self.beamFB.use_gpu()
    if self.rf_params:
        self.rf_params.use_gpu()
    self.dev_phi_modulation = self.rf_params.dev_phi_modulation


RingAndRFTracker.use_gpu = use_gpu_tracker


# TotalInducedVoltage
def use_gpu_total_induced_voltage(self):
    from ..gpu.cupy_array import CGA
    from ..gpu.cupy_impedance import GpuTotalInducedVoltage

    if self.__class__ == GpuTotalInducedVoltage:
        return

    self.induced_voltage_obj = CGA(self.induced_voltage)
    self.__class__ = GpuTotalInducedVoltage

    for obj in self.induced_voltage_list:
        obj.use_gpu()


TotalInducedVoltage.use_gpu = use_gpu_total_induced_voltage


# _InducedVoltage
def use_gpu_induced_voltage(self, child=False, new_class=None):
    import cupy as cp
    from ..gpu.cupy_array import CGA
    from ..gpu.cupy_impedance import (GpuInducedVoltage, GpuInducedVoltageFreq,
                                     GpuInducedVoltageTime, GpuInductiveImpedance)

    if (self.__class__ in [GpuInducedVoltage, GpuInducedVoltageFreq,
                           GpuInducedVoltageTime, GpuInductiveImpedance]):
        return
    if not hasattr(self, 'mtw_memory'):
        self.mtw_memory = None
    if not hasattr(self, 'total_impedance'):
        self.total_impedance = None

    self.mtw_memory_obj = CGA(self.mtw_memory)
    self.total_impedance_obj = CGA(self.total_impedance)

    if not child:
        self.__class__ = GpuInducedVoltage
    else:
        self.__class__ = new_class

    if self.multi_turn_wake:
        self.induced_voltage_generation = self.induced_voltage_mtw
        if self.mtw_mode == 'freq':
            self.shift_trev = self.shift_trev_freq
        else:
            self.shift_trev = self.shift_trev_time
    else:
        self.induced_voltage_generation = self.induced_voltage_1turn

    if hasattr(self, "time_mtw"):
        self.dev_time_mtw = cp.array(self.time_mtw)


_InducedVoltage.use_gpu = use_gpu_induced_voltage


# InducedVoltageTime
def use_gpu_induced_voltage_time(self):
    from ..gpu.cupy_impedance import GpuInducedVoltageTime
    super(InducedVoltageTime, self).use_gpu(child=True, new_class=GpuInducedVoltageTime)


InducedVoltageTime.use_gpu = use_gpu_induced_voltage_time


# InducedVoltageFreq
def use_gpu_induced_voltage_freq(self):
    from ..gpu.cupy_impedance import GpuInducedVoltageFreq
    super(InducedVoltageFreq, self).use_gpu(child=True, new_class=GpuInducedVoltageFreq)


InducedVoltageFreq.use_gpu = use_gpu_induced_voltage_freq


# InductiveImpedance
def use_gpu_inductive_impedance(self):
    from ..gpu.cupy_impedance import GpuInductiveImpedance
    super(InductiveImpedance, self).use_gpu(child=True, new_class=GpuInductiveImpedance)


InductiveImpedance.use_gpu = use_gpu_inductive_impedance


# InducedVoltageResonator
def use_gpu_inductive_voltage_resonator(self):
    pass


InducedVoltageResonator.use_gpu = use_gpu_inductive_voltage_resonator


# RF_parameters
def use_gpu_rf_station(self):
    from ..gpu.cupy_array import CGA
    from ..gpu.gpu_rf_parameters import GpuRFStation
    from pycuda import gpuarray
    if self.__class__ == GpuRFStation:
        return

    if not self.phi_modulation:
        self.dev_phi_modulation = None
    else:
        self.dev_phi_modulation = (gpuarray.to_gpu(self.phi_modulation[0]), gpuarray.to_gpu(self.phi_modulation[1]))

    if (self.phi_noise is None):
        self.dev_phi_noise = None
    else:
        self.dev_phi_noise = gpuarray.to_gpu(self.phi_noise.flatten())

    # voltage to gpu
    self.voltage_obj = CGA(self.voltage)

    # omega_rf to gpu
    self.omega_rf_obj = CGA(self.omega_rf)

    # phi_rf to gpu
    self.phi_rf_obj = CGA(self.phi_rf)

    # omega_rf_d to gpu
    self.omega_rf_d_obj = CGA(self.omega_rf_d)
    # assert self.omega_rf_d.dtype == bm.precision.real_t

    # harmonic to gpu
    self.harmonic_obj = CGA(self.harmonic)

    # dphi_rf to gpu
    self.dphi_rf_obj = CGA(self.dphi_rf)
    self.__class__ = GpuRFStation


RFStation.use_gpu = use_gpu_rf_station

# BeamFeedback
def use_gpu_beamfeedback(self):
    from ..gpu.cupy_beamfeedback import GpuBeamFeedback
    self.__class__ = GpuBeamFeedback


BeamFeedback.use_gpu = use_gpu_beamfeedback
