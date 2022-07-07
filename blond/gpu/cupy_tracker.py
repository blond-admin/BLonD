from __future__ import division, print_function
import numpy as np
from ..utils import bmath as bm
from ..gpu.cupy_array import get_gpuarray
from ..gpu.cupy_butils_wrap import  first_kernel_tracker, second_kernel_tracker, \
    cavityFB_case,  gpu_rf_voltage_calc_mem_ops

try:
    from pyprof import timing
except ImportError:
    from ..utils import profile_mock as timing



from ..trackers.tracker import RingAndRFTracker


class GpuRingAndRFTracker(RingAndRFTracker):

    @property
    def rf_voltage(self):
        return self.rf_voltage_obj.my_array

    @rf_voltage.setter
    def rf_voltage(self, value):
        self.rf_voltage_obj.my_array = value

    @property
    def dev_rf_voltage(self):
        return self.rf_voltage_obj.dev_my_array

    @dev_rf_voltage.setter
    def dev_rf_voltage(self, value):
        self.rf_voltage_obj.dev_my_array = value

    def pre_track(self):
        """
        Gpu Equivalent for pre_track
        """
        turn = self.counter[0]
        if self.phi_noise is not None:
            with timing.timed_region('serial:pretrack_phirf'):
                if self.noiseFB is not None:
                    first_kernel_tracker(self.rf_params.dev_phi_rf,
                                         self.noiseFB.x, self.dev_phi_noise,
                                         self.rf_params.dev_phi_rf.shape[0],
                                         turn, limit = self.rf_params.n_rf)
                else:
                    first_kernel_tracker(self.rf_params.dev_phi_rf, 1.0,
                                         self.rf_params.dev_phi_noise,
                                         self.rf_params.dev_phi_rf.shape[0],
                                         turn, limit = self.rf_params.n_rf)
                    # self.phi_rf[:, turn] += \
                    #     self.phi_noise[:, turn]

        # Add phase modulation directly to the cavity RF phase
        if self.phi_modulation is not None:
            with timing.timed_region('serial:pretrack_phimodulation'):
                second_kernel_tracker(self.dev_phi_rf,
                                      self.dev_phi_modulation[0],
                                      self.dev_phi_modulation[1],
                                      self.dev_phi_rf.shape[0], turn)
            # Determine phase loop correction on RF phase and frequency

        if self.beamFB is not None and turn >= self.beamFB.delay:
            self.beamFB.track()
        if self.periodicity:
            RuntimeError('periodicity feature is not supported in GPU.')
        else:
            if self.rf_params.empty is False:
                if self.interpolation:
                    self.rf_voltage_calculation()

    def track_only(self):
        """
        Gpu Equivalent for track_only
        """
        turn = self.counter[0]

        if self.periodicity:

            pass

        else:
            if self.rf_params.empty is False:

                if self.interpolation:
                    with timing.timed_region('comp:LIKick'):
                        self.dev_total_voltage = get_gpuarray(self.dev_rf_voltage.size, bm.precision.real_t)
                        if self.totalInducedVoltage is not None:
                            self.dev_total_voltage = self.dev_rf_voltage + self.totalInducedVoltage.dev_induced_voltage
                        else:
                            self.dev_total_voltage = self.dev_rf_voltage
                        bm.linear_interp_kick(self.beam.dev_dt, self.beam.dev_dE,
                                              self.dev_total_voltage,
                                              self.profile.dev_bin_centers,
                                              self.beam.Particle.charge,
                                              self.acceleration_kick[turn],
                                              )
                        #self.beam.dE_obj.invalidate_cpu()
                        # bm.LIKick_n_drift(dev_voltage=self.dev_total_voltage,
                        #                   dev_bin_centers=self.profile.dev_bin_centers,
                        #                   charge=self.beam.Particle.charge,
                        #                   acceleration_kick=self.acceleration_kick[turn],
                        #                   T0=self.t_rev[turn + 1],
                        #                   length_ratio=self.length_ratio,
                        #                   eta0=self.eta_0[turn + 1],
                        #                   beta=self.rf_params.beta[turn+1],
                        #                   energy=self.rf_params.energy[turn+1],
                        #                   beam=self.beam)

                else:
                    self.kick(turn)
            self.drift(turn + 1)

        # Updating the beam synchronous momentum etc.
        self.beam.beta = self.rf_params.beta[turn + 1]
        self.beam.gamma = self.rf_params.gamma[turn + 1]
        self.beam.energy = self.rf_params.energy[turn + 1]
        self.beam.momentum = self.rf_params.momentum[turn + 1]

        # Increment by one the turn counter
        self.counter[0] += 1

    def track(self):
        self.pre_track()
        self.track_only()

    #@timing.timeit(key='serial:RFVCalc')
    def rf_voltage_calculation(self):
        """
        Gpu Equivalent for rf_voltage_calculation
        """

        dev_voltages = get_gpuarray(self.rf_params.n_rf, bm.precision.real_t)
        dev_omega_rf = get_gpuarray(self.rf_params.n_rf, bm.precision.real_t)
        dev_phi_rf = get_gpuarray(self.rf_params.n_rf, bm.precision.real_t)
        n_turns = self.rf_params.n_turns + 1

        # sz = self.n_rf
        my_end = self.rf_params.dev_voltage.size
        gpu_rf_voltage_calc_mem_ops(dev_voltages, dev_omega_rf, dev_phi_rf,
                                    self.rf_params.dev_voltage, self.rf_params.dev_omega_rf,
                                    self.rf_params.dev_phi_rf,
                                    np.int32(self.counter[0]),
                                    np.int32(my_end), np.int32(n_turns),
                                    block=(32, 1, 1), grid=(1, 1, 1))

        self.dev_rf_voltage = get_gpuarray(self.profile.dev_bin_centers.size, bm.precision.real_t)

        # TODO: test with multiple harmonics, think about 800 MHz OTFB

        if self.cavityFB:
            cavityFB_case(self.dev_rf_voltage, dev_voltages, dev_omega_rf,
                          dev_phi_rf, self.profile.dev_bin_centers,
                          self.cavityFB.V_corr, self.cavityFB.phi_corr)
            # self.rf_voltage = voltages[0] * self.cavityFB.V_corr * \
            #     bm.sin(omega_rf[0]*self.profile.bin_centers +
            #             phi_rf[0] + self.cavityFB.phi_corr)
            bm.rf_volt_comp(dev_voltages, dev_omega_rf, dev_phi_rf,
                            self.profile.dev_bin_centers, self.dev_rf_voltage, f_rf=1)
        else:
            bm.rf_volt_comp(dev_voltages, dev_omega_rf, dev_phi_rf,
                            self.profile.dev_bin_centers, self.dev_rf_voltage)

    #@timing.timeit(key='comp:kick')
    def kick(self, index):
        """
        Gpu Equivalent for kick
        """

        dev_voltage = get_gpuarray(self.rf_params.n_rf, bm.precision.real_t)
        dev_omega_rf = get_gpuarray(self.rf_params.n_rf, bm.precision.real_t)
        dev_phi_rf = get_gpuarray(self.rf_params.n_rf, bm.precision.real_t)

        my_end = self.rf_params.dev_voltage.size

        dev_voltage[:] = self.rf_params.dev_voltage[index:my_end:self.rf_params.n_turns + 1]
        dev_omega_rf[:] = self.rf_params.dev_omega_rf[index:my_end:self.rf_params.n_turns + 1]
        dev_phi_rf[:] = self.rf_params.dev_phi_rf[index:my_end:self.rf_params.n_turns + 1]

        bm.kick(self.beam.dev_dt, self.beam.dev_dE, dev_voltage, dev_omega_rf,
                dev_phi_rf, self.charge, self.n_rf,
                self.acceleration_kick[index])
        #self.beam.dE_obj.invalidate_cpu()

    #@timing.timeit(key='comp:drift')
    def drift(self, index):
        """
        Gpu Equivalent for drift
        """
        bm.drift(self.beam.dev_dt, self.beam.dev_dE, self.solver, self.t_rev[index],
                 self.length_ratio, self.alpha_order, self.eta_0[index],
                 self.eta_1[index], self.eta_2[index], self.alpha_0[index],
                 self.alpha_1[index], self.alpha_2[index],
                 self.rf_params.beta[index], self.rf_params.energy[index])
        #self.beam.dt_obj.invalidate_cpu()
