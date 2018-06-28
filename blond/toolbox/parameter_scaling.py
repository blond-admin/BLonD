# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Scaling of longitudinal beam and machine parameters, with user interface.**

:Authors: **Konstantinos Iliakis**, **Helga Timko**
'''

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import QButtonGroup, QHBoxLayout, QGroupBox
from scipy import integrate
from scipy.constants import m_p, e, c

import numpy as np


class ParameterScaling(object):
    
    @property
    def phi_b(self):
        
        return self.omega_rf * self.tau / 2.

        
    @property
    def delta_b(self):
        
        return self.dE_b / (self.beta_sq * self.energy)
    
    
    @property
    def dE_b(self):
        
        return np.sqrt(self.beta_sq * self.energy * self.voltage * (1 - 
                       np.cos(self.phi_b)) / (np.pi * self.harmonic * self.eta_0))


    @property
    def integral(self):
        
        return integrate.quad(lambda x: np.sqrt(2.*(np.cos(x) - 
                              np.cos(self.phi_b))), 0, self.phi_b)[0]

                              
    @property
    def emittance(self):
                          
        return 4.*self.energy * self.omega_s0 * self.beta_sq * \
                         self.integral / (self.omega_rf ** 2 * self.eta_0)


    def relativistic_quantities(self):

        self.momentum = np.sqrt(self.energy ** 2 - self.mass ** 2)
        self.tb1.append("    Synchronous momentum: " 
                               + np.str(self.momentum) + " eV")

        self.kinetic_energy = self.energy - self.mass
        self.tb1.append("    Synchronous kinetic energy: " 
                               + np.str(self.kinetic_energy) + " eV")

        self.gamma = self.energy / self.mass
        self.tb1.append("    Synchronous relativistic gamma: " 
                               + np.str(self.gamma) + "")

        self.beta = np.sqrt(1. - 1. / self.gamma ** 2)
        self.tb1.append("    Synchronous relativistic beta: " 
                               + np.str(self.beta) + "")

        self.beta_sq = self.beta ** 2
        self.tb1.append("    Synchronous relativistic beta squared: " 
                               + np.str(self.beta_sq) + "\n")
           
    
    def frequencies(self):
        
        self.t_rev = self.circumference / (self.beta * c)
        self.tb1.append("    Revolution period: " 
                               + np.str(self.t_rev * 1.e6) + " us")

        self.f_rev = 1. / self.t_rev
        self.tb1.append("    Revolution frequency: " 
                               + np.str(self.f_rev) + " Hz")

        self.omega_rev = 2.*np.pi * self.f_rev
        self.tb1.append("        Angular revolution frequency: " 
                               + np.str(self.omega_rev) + " 1/s")
        
        self.f_RF = self.harmonic * self.f_rev           
        self.tb1.append("    RF frequency: " 
                               + np.str(self.f_RF * 1.e-6) + " MHz")

        self.omega_rf = 2.*np.pi * self.f_RF           
        self.tb1.append("        Angular RF frequency: " 
                               + np.str(self.omega_rf) + " 1/s\n")
    
    
    def tune(self):
    
        self.eta_0 = np.fabs(1. / self.gamma_t ** 2 - 1. / self.gamma ** 2)
        self.tb1.append("    Slippage factor (zeroth order): " 
                               + np.str(self.eta_0) + "")

        self.Q_s0 = np.sqrt(self.harmonic * self.voltage * self.eta_0 / 
                           (2.*np.pi * self.beta_sq * self.energy))
        self.tb1.append("    Central synchrotron tune: " 
                               + np.str(self.Q_s0) + "")

        self.f_s0 = self.Q_s0 * self.f_rev
        self.tb1.append("    Central synchrotron frequency: " 
                               + np.str(self.f_s0) + "")

        self.omega_s0 = 2.*np.pi * self.f_s0           
        self.tb1.append("        Angular synchrotron frequency: " 
                               + np.str(self.omega_s0) + " 1/s\n")


    def bucket_parameters(self):
        
        self.tb1.append("Bucket parameters assuming single RF, stationary case, and no intensity effects.\n")

        self.bucket_area = 8.*np.sqrt(2.*self.beta_sq * self.energy * self.voltage / 
                           (np.pi * self.harmonic * self.eta_0)) / self.omega_rf
        self.tb1.append("    Bucket area: " 
                               + np.str(self.bucket_area) + " eVs")

        self.dt_max = 0.5 * self.t_rev / self.harmonic
        self.tb1.append("    Half of bucket length: " 
                               + np.str(self.dt_max * 1.e9) + " ns")
        
        self.dE_max = np.sqrt(2.*self.beta ** 2 * self.energy * self.voltage / 
                              (np.pi * self.eta_0 * self.harmonic))
        self.tb1.append("    Half of bucket height: " 
                               + np.str(self.dE_max * 1.e-6) + " MeV")
       
        self.delta_max = self.dE_max / (self.beta_sq * self.energy)
        self.tb1.append("        In relative momentum offset: " 
                               + np.str(self.delta_max) + "\n")
        
        
    def emittance_from_bunch_length(self, four_sigma_bunch_length):
        
        self.tau = four_sigma_bunch_length
        if self.tau >= 2.*self.dt_max:
            self.tb1.append("Chosen bunch length too large for this bucket. Aborting!")
            raise RuntimeError("Chosen bunch length too large for this bucket. Aborting!")
        self.tb1.append("Calculating emittance of 4-sigma bunch length: " 
                               + np.str(self.tau * 1.e9) + " ns")

        self.tb1.append("    Emittance contour in phase: " 
                               + np.str(self.phi_b) + " rad")
        self.tb1.append("    Emittance contour in relative momentum: " 
                               + np.str(self.delta_b) + "")
        self.tb1.append("    Emittance contour in energy offset: " 
                               + np.str(self.dE_b * 1.e-6) + " MeV")
        self.tb1.append("    Longitudinal emittance is: " 
                               + np.str(self.emittance) + " eVs\n")


    def bunch_length_from_emittance(self, emittance):
        
        self.emittance_aim = emittance
        
        if self.emittance_aim >= self.bucket_area:
            self.tb1.append("Chosen emittance too large for this bucket. Aborting!")
            raise RuntimeError("Chosen emittance too large for this bucket. Aborting!")
        self.tb1.append("Calculating 4-sigma bunch length for an emittance of " 
                               + np.str(self.emittance_aim) + " eVs")

        # Make a guess, iterate to get closer
        self.tau = self.dt_max / 2. 
        while (np.fabs((self.emittance - self.emittance_aim)
                       / self.emittance_aim) > 0.001):   
            self.tau *= np.sqrt(self.emittance_aim / self.emittance)

        self.tb1.append("    Bunch length is: " 
                               + np.str(self.tau * 1.e9) + " ns")
        self.tb1.append("    Corresponding matched rms relative momentum offset: " 
                               + np.str(self.delta_b) + "")
        self.tb1.append("    Emittance contour in phase: " 
                               + np.str(self.phi_b) + " rad")
    
    
    
    def setupUi(self, mainWindow):
        
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(586, 611)
        mainWindow.setWindowOpacity(1.0)
        mainWindow.setFixedSize(mainWindow.size())
#         mainWindow.setModal(True)
        
        self.lbMachine = QtWidgets.QLabel(mainWindow)
        self.lbMachine.setGeometry(QtCore.QRect(20, 20, 70, 17))
        self.lbMachine.setMinimumSize(QtCore.QSize(70, 0))
        self.lbMachine.setMaximumSize(QtCore.QSize(16777215, 17))
        self.lbMachine.setObjectName("lbMachine")
        self.lbEnergy = QtWidgets.QLabel(mainWindow)
        self.lbEnergy.setGeometry(QtCore.QRect(20, 80, 70, 17))
        self.lbEnergy.setObjectName("lbEnergy")
        self.leCustom = QtWidgets.QLineEdit(mainWindow)
        self.leCustom.setEnabled(True)
        self.leCustom.setGeometry(QtCore.QRect(125, 100, 71, 25))
        self.leCustom.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);")
        self.leCustom.hide()
        self.leCustom.setText("")
        self.leCustom.setObjectName("leCustom")
        self.lbEV1 = QtWidgets.QLabel(mainWindow)
        self.lbEV1.setEnabled(True)
        self.lbEV1.setGeometry(QtCore.QRect(200, 100, 30, 25))
        self.lbEV1.setObjectName("lbEV1")
        self.lbEV1.hide()
        self.lbVoltage = QtWidgets.QLabel(mainWindow)
        self.lbVoltage.setGeometry(QtCore.QRect(20, 160, 70, 25))
        self.lbVoltage.setObjectName("lbVoltage")
        self.lbEV2 = QtWidgets.QLabel(mainWindow)
        self.lbEV2.setGeometry(QtCore.QRect(150, 160, 31, 25))
        self.lbEV2.setObjectName("lbEV2")
        self.leVoltage = QtWidgets.QLineEdit(mainWindow)
        self.leVoltage.setGeometry(QtCore.QRect(80, 155, 70, 25))
        self.leVoltage.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);")
        self.leVoltage.setText("")
        self.leVoltage.setObjectName("leVoltage")
        self.lbOptional = QtWidgets.QLabel(mainWindow)
        self.lbOptional.setGeometry(QtCore.QRect(20, 230, 70, 17))
        self.lbOptional.setObjectName("lbOptional")
        
        self.leEmittance = QtWidgets.QLineEdit(mainWindow)
        self.leEmittance.setGeometry(QtCore.QRect(130, 270, 70, 25))
        self.leEmittance.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);")
        self.leEmittance.setText("")
        self.leEmittance.setObjectName("leEmittance")
        self.lbEVS1 = QtWidgets.QLabel(mainWindow)
        self.lbEVS1.setGeometry(QtCore.QRect(200, 275, 41, 25))
        self.lbEVS1.setObjectName("lbEVS1")
        self.lbEVS2 = QtWidgets.QLabel(mainWindow)
        self.lbEVS2.setGeometry(QtCore.QRect(330, 275, 41, 25))
        self.lbEVS2.setObjectName("lbEVS2")
        self.leBunchLength = QtWidgets.QLineEdit(mainWindow)
        self.leBunchLength.setGeometry(QtCore.QRect(260, 270, 70, 25))
        self.leBunchLength.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);")
        self.leBunchLength.setText("")
        self.leBunchLength.setObjectName("leBunchLength")
        
        self.pbSubmit = QtWidgets.QPushButton(mainWindow)
        self.pbSubmit.setGeometry(QtCore.QRect(230, 320, 101, 27))
        self.pbSubmit.setObjectName("pbSumbit")
        self.tb1 = QtWidgets.QTextBrowser(mainWindow)
        self.tb1.setGeometry(QtCore.QRect(10, 350, 561, 241))
        self.tb1.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.tb1.setObjectName("tb1")
        self.cbMachine = QtWidgets.QComboBox(mainWindow)
        self.cbMachine.setGeometry(QtCore.QRect(20, 40, 83, 25))
        self.cbMachine.setEditable(False)
        self.cbMachine.setObjectName("cbMachine")
        self.cbMachine.addItem("")
        self.cbMachine.addItem("")
        self.cbMachine.addItem("")
        self.cbMachine.addItem("")
        self.cbMachine.addItem("")
        self.cbEnergy = QtWidgets.QComboBox(mainWindow)
        self.cbEnergy.setGeometry(QtCore.QRect(20, 100, 100, 25))
        self.cbEnergy.setObjectName("cbEnergy")
        self.cbEnergy.addItem("")
        self.cbEnergy.addItem("")
        self.cbEnergy.addItem("")
        
        self.rbBunchLength = QtWidgets.QRadioButton(mainWindow)
        self.rbBunchLength.setGeometry(QtCore.QRect(260, 250, 140, 22))
        self.rbBunchLength.setObjectName("rbBunchLength")
        
        self.rbEmittance = QtWidgets.QRadioButton(mainWindow)
        self.rbEmittance.setGeometry(QtCore.QRect(130, 250, 100, 22))
        self.rbEmittance.setObjectName("rbEmittance")
        
        self.rbNoOption = QtWidgets.QRadioButton(mainWindow)
        self.rbNoOption.setGeometry(QtCore.QRect(20, 250, 100, 22))
        self.rbNoOption.setObjectName('rbNoOption')
        self.rbNoOption.setChecked(True)
#         self.rbLayout = QHBoxLayout()
#         self.rbLayout.addWidget(self.rbBunchLength)
#         self.rbLayout.addWidget(self.rbEmittance)
#         self.rbLayout.addWidget(self.rbNoOptioin)
#         self.rbLayout.setGeometry(QtCore.QRect(10, 10, 10, 10))
#         self.rbLayout.addStretch(1)
#         
#         self.rbGroupBox = QGroupBox()
#         self.rbGroupBox.setLayout(self.rbLayout)
#         self.rbGroupBox.setGeometry(QtCore.QRect(10, 10, 10, 10))
#         
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        self.addactions(mainWindow)
        
    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Parameter Scaling"))
        self.lbMachine.setText(_translate("mainWindow", "Machine"))
        self.lbEnergy.setText(_translate("mainWindow", "Energy"))
        self.lbEV1.setText(_translate("mainWindow", "[eV]"))
        self.lbVoltage.setText(_translate("mainWindow", "Voltage"))
        self.lbEV2.setText(_translate("mainWindow", "[V]"))
        self.lbOptional.setText(_translate("mainWindow", "Optional"))
        self.rbEmittance.setText(_translate("mainWindow", "Emittance"))
        self.lbEVS1.setText(_translate("mainWindow", "[eVs]"))
        self.lbEVS2.setText(_translate("mainWindow", "[s]"))
        self.rbBunchLength.setText(_translate("mainWindow", "Bunch Length"))
        self.rbNoOption.setText(_translate("mainWindow", "No Options"))

        self.pbSubmit.setText(_translate("mainWindow", "Submit"))
        self.cbEnergy.setCurrentText(_translate("mainWindow", "LHC"))
        self.cbMachine.setItemText(0, _translate("mainWindow", "LHC"))
        self.cbMachine.setItemText(1, _translate("mainWindow", "PSB"))
        self.cbMachine.setItemText(2, _translate("mainWindow", "SPS, Q20"))
        self.cbMachine.setItemText(3, _translate("mainWindow", "SPS, Q26"))
        self.cbMachine.setItemText(4, _translate("mainWindow", "CPS"))
        self.cbEnergy.setItemText(0, _translate("mainWindow", "Flat bottom"))
        self.cbEnergy.setItemText(1, _translate("mainWindow", "Flat top"))
        self.cbEnergy.setItemText(2, _translate("mainWindow", "Custom"))
    
    
    def addactions(self, mainWindow):
        self.pbSubmit.clicked.connect(self.pbHandler)
        self.cbEnergy.activated[str].connect(self.cbEnergyHandler)
    
    def pbHandler(self):
#         self.tb1.append('pbSubmit was clicked!')
        
        self.machine = str(self.cbMachine.currentText())
#         self.tb1.append('machine is ' + self.machine)
        
        self.energy_type = self.cbEnergy.currentText()
        if(self.energy_type == 'Custom'):
            self.custom_energy = self.leCustom.text()
            try:
                self.energy = np.double(self.custom_energy)
            except ValueError:
                self.tb1.append("Energy not recognized!")
                return
                # raise RuntimeError('Energy not recognized. Aborting!')
#             self.tb1.append('custom_energy is ' + self.custom_energy)
            
#         self.tb1.append('energy_type is ' + self.energy_type)
        
        
        self.voltage = self.leVoltage.text()
#         self.tb1.append('voltage is ' + self.voltage)
        
        self.emittance_target = self.leEmittance.text()
#         self.tb1.append('emittance_target is ' + self.emittance_target)
        
        self.bunch_length_target = self.leBunchLength.text()
#         self.tb1.append('bunch_length_target is ' + self.bunch_length_target)
        
        
        self.tb1.append("Input -- chosen machine: " + 
                               np.str(self.machine) + "\n")

        # Machine-dependent parameters [SI-units] -----------------------------
        gamma_ts = {'PSB': 4.0767, 'CPS': np.sqrt(37.2), 'SPS, Q20': 18.,
                   'SPS, Q26': 22.83, 'LHC': 55.759505}
        harmonics = {'PSB': 1, 'CPS': 21, 'SPS, Q20': 4620, 'SPS, Q26': 4620,
                     'LHC': 35640}
        circumferences = {'PSB': 2 * np.pi * 25, 'CPS': 2 * np.pi * 100.,
                          'SPS, Q20': 2 * np.pi * 1100.009,
                          'SPS, Q26': 2 * np.pi * 1100.009, 'LHC': 26658.883}
        energies_fb = {'PSB': (50.e6 + m_p * c ** 2 / e), 'CPS': 1.4e9, 'SPS, Q20': 25.94e9,
                       'SPS-Q26': 25.94e9, 'LHC': 450.e9}
        energies_ft = {'PSB': (1.4e9 + m_p * c ** 2 / e), 'CPS': 25.92e9, 'SPS, Q20': 450.e9,
                       'SPS, Q26': 450.e9, 'LHC': 6.5e12}
        # Machine-dependent parameters [SI-units] -----------------------------
              
        self.gamma_t = gamma_ts[self.machine]
        self.alpha = 1. / (self.gamma_t) ** 2
        self.tb1.append("    * with relativistic gamma at transition: " 
                               + np.str(self.gamma_t) + "")
        self.tb1.append("    * with momentum compaction factor: " 
                               + np.str(self.alpha) + "")
        
        self.harmonic = harmonics[self.machine]
        self.tb1.append("    * with main harmonic: " 
                               + np.str(self.harmonic) + "")
        
        self.circumference = circumferences[self.machine]
        self.tb1.append("    * and machine circumference: " 
                               + np.str(self.circumference) + " m\n")
        
        if self.energy_type == 'Flat bottom':
            self.energy = energies_fb[self.machine]
        elif self.energy_type == 'Flat top':
            self.energy = energies_ft[self.machine]
        
        self.tb1.append("Input -- synchronous total energy: " 
                               + np.str(self.energy * 1.e-6) + " MeV")

        try:
            self.voltage = np.double(self.voltage)
        except ValueError:
            self.tb1.append("Voltage not recognised!")
            return
            # raise RuntimeError('Voltage not recognised. Aborting!')
        self.tb1.append("Input -- RF voltage: " 
                               + np.str(self.voltage * 1.e-6) + " MV")

        self.mass = m_p * c ** 2 / e
        self.tb1.append("Input -- particle mass: " 
                               + np.str(self.mass * 1.e-6) + " MeV\n")
                
        # Derived quantities --------------------------------------------------
        self.relativistic_quantities()
        self.frequencies()
        self.tune()
        self.bucket_parameters()
        
        if self.rbEmittance.isChecked():
            try:
                self.emittance_target = np.double(self.emittance_target)
            except ValueError:
                self.tb1.append("Target emittance not recognised!")
                return 
                # raise RuntimeError('Target emittance not recognised. Aborting!')
            self.bunch_length_from_emittance(self.emittance_target)
        elif self.rbBunchLength.isChecked():
            try:
                self.bunch_length_target = np.double(self.bunch_length_target)
            except ValueError:
                self.tb1.append("Target bunch length not recognised!")
                return 
                #raise RuntimeError('Target bunch length not recognised. Aborting!')
            self.emittance_from_bunch_length(self.bunch_length_target)

    
    def cbEnergyHandler(self, text):
#         self.tb1.append('cbEnergy changed to ' + text)
        if text == 'Custom':
            self.leCustom.show()
            self.lbEV1.show()
        else:
            self.leCustom.hide()
            self.lbEV1.hide()
            

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = ParameterScaling()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
    
    
