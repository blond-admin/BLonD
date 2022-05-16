from ..input_parameters.rf_parameters import RFStation


class GpuRFStation(RFStation):

    ##  voltage


    @property
    def voltage(self):
        return self.voltage_obj.my_array

    @voltage.setter
    def voltage(self, value):
        self.voltage_obj.my_array = value


    @property
    def dev_voltage(self):
        return self.voltage_obj.dev_my_array


    @dev_voltage.setter
    def dev_voltage(self, value):
        self.voltage_obj.dev_my_array = value

    ##  phi_rf


    @property
    def phi_rf(self):
        return self.phi_rf_obj.my_array

    @phi_rf.setter
    def phi_rf(self, value):
        self.phi_rf_obj.my_array = value


    @property
    def dev_phi_rf(self):
        return self.phi_rf_obj.dev_my_array


    @dev_phi_rf.setter
    def dev_phi_rf(self, value):
        self.phi_rf_obj.dev_my_array = value

    ##  omega_rf


    @property
    def omega_rf(self):
        return self.omega_rf_obj.my_array

    @omega_rf.setter
    def omega_rf(self, value):
        self.omega_rf_obj.my_array = value


    @property
    def dev_omega_rf(self):
        return self.omega_rf_obj.dev_my_array


    @dev_omega_rf.setter
    def dev_omega_rf(self, value):
        self.omega_rf_obj.dev_my_array = value

    ##  omega_rf_d


    @property
    def omega_rf_d(self):
        return self.omega_rf_d_obj.my_array

    @omega_rf_d.setter
    def omega_rf_d(self, value):
        self.omega_rf_d_obj.my_array = value


    @property
    def dev_omega_rf_d(self):
        return self.omega_rf_d_obj.dev_my_array


    @dev_omega_rf_d.setter
    def dev_omega_rf_d(self, value):
        self.omega_rf_d_obj.dev_my_array = value

    ##  harmonic


    @property
    def harmonic(self):
        return self.harmonic_obj.my_array

    @harmonic.setter
    def harmonic(self, value):
        self.harmonic_obj.my_array = value


    @property
    def dev_harmonic(self):
        return self.harmonic_obj.dev_my_array


    @dev_harmonic.setter
    def dev_harmonic(self, value):
        self.harmonic_obj.dev_my_array = value

    ##  dphi_rf


    @property
    def dphi_rf(self):
        return self.dphi_rf_obj.my_array

    @dphi_rf.setter
    def dphi_rf(self, value):
        self.dphi_rf_obj.my_array = value


    @property
    def dev_dphi_rf(self):
        return self.dphi_rf_obj.dev_my_array


    @dev_dphi_rf.setter
    def dev_dphi_rf(self, value):
        self.dphi_rf_obj.dev_my_array = value
