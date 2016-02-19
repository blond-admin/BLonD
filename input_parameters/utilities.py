
# Copyright 2015 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Module gathering and processing all RF parameters used in the simulation.**

:Authors: **Helga Timko**
'''

from __future__ import division
import numpy as np
from scipy.constants import m_p, e, c
from scipy import integrate
import Tkinter as tk



class ParameterScaling(object):
    '''
    Determines longitudinal parameters to construct simulation input or simply
    estimate different parameters. Assumes a single RF system.
    '''
    
    def __init__(self):
        
        
        # Display dialogue window
        self.master = tk.Tk()
        self.dialogue_window()
        self.master.mainloop()
        

    @property
    def phi_b(self):
        
        return self.omega_RF*self.tau/2.

        
    @property
    def delta_b(self):
        
        return self.dE_b/(self.beta_sq*self.energy)
    
    
    @property
    def dE_b(self):
        
        return np.sqrt(self.beta_sq*self.energy*self.voltage*(1 - 
                       np.cos(self.phi_b))/(np.pi*self.harmonic*self.eta_0))


    @property
    def integral(self):
        
        return integrate.quad(lambda x: np.sqrt(2.*(np.cos(x) - 
                              np.cos(self.phi_b))), 0, self.phi_b)[0]

                              
    @property
    def emittance(self):
                          
        return 4.*self.energy*self.omega_s0*self.beta_sq* \
                         self.integral/(self.omega_RF**2*self.eta_0)


    def relativistic_quantities(self):

        self.momentum = np.sqrt(self.energy**2 - self.mass**2)
        self.textwindow.insert('end', "    Synchronous momentum: " 
                               + np.str(self.momentum) + " eV\n")

        self.kinetic_energy = self.energy - self.mass
        self.textwindow.insert('end', "    Synchronous kinetic energy: " 
                               + np.str(self.kinetic_energy) + " eV\n")

        self.gamma = self.energy/self.mass
        self.textwindow.insert('end', "    Synchronous relativistic gamma: " 
                               + np.str(self.gamma) + "\n")

        self.beta = np.sqrt(1. - 1./self.gamma**2)
        self.textwindow.insert('end', "    Synchronous relativistic beta: " 
                               + np.str(self.beta) + "\n")

        self.beta_sq = self.beta**2
        self.textwindow.insert('end', "    Synchronous relativistic beta squared: " 
                               + np.str(self.beta_sq) + "\n\n")
           
    
    def frequencies(self):
        
        self.t_rev = self.circumference/(self.beta*c)
        self.textwindow.insert('end', "    Revolution period: " 
                               + np.str(self.t_rev*1.e6) + " us\n")

        self.f_rev = 1./self.t_rev
        self.textwindow.insert('end', "    Revolution frequency: " 
                               + np.str(self.f_rev) + " Hz\n")

        self.omega_rev = 2.*np.pi*self.f_rev
        self.textwindow.insert('end', "        Angular revolution frequency: " 
                               + np.str(self.omega_rev) + " 1/s\n")
        
        self.f_RF = self.harmonic*self.f_rev           
        self.textwindow.insert('end', "    RF frequency: " 
                               + np.str(self.f_RF*1.e-6) + " MHz\n")

        self.omega_RF = 2.*np.pi*self.f_RF           
        self.textwindow.insert('end', "        Angular RF frequency: " 
                               + np.str(self.omega_RF) + " 1/s\n\n")
    
    
    def tune(self):
    
        self.eta_0 = np.fabs(1./self.gamma_t**2 - 1./self.gamma**2)
        self.textwindow.insert('end', "    Slippage factor (zeroth order): " 
                               + np.str(self.eta_0) + "\n")

        self.Q_s0 = np.sqrt(self.harmonic*self.voltage*self.eta_0/
                           (2.*np.pi*self.beta_sq*self.energy))
        self.textwindow.insert('end', "    Central synchrotron tune: " 
                               + np.str(self.Q_s0) + "\n")

        self.f_s0 = self.Q_s0*self.f_rev
        self.textwindow.insert('end', "    Central synchrotron frequency: " 
                               + np.str(self.f_s0) + "\n")

        self.omega_s0 = 2.*np.pi*self.f_s0           
        self.textwindow.insert('end', "        Angular synchrotron frequency: " 
                               + np.str(self.omega_s0) + " 1/s\n\n")


    def bucket_parameters(self):
        
        self.textwindow.insert('end', "Bucket parameters assuming single RF, stationary case, and no intensity effects.\n")

        self.bucket_area = 8.*np.sqrt(2.*self.beta_sq*self.energy*self.voltage/
                           (np.pi*self.harmonic*self.eta_0))/self.omega_RF
        self.textwindow.insert('end', "    Bucket area: " 
                               + np.str(self.bucket_area) + " eVs\n")

        self.dt_max = 0.5*self.t_rev/self.harmonic
        self.textwindow.insert('end', "    Half of bucket length: " 
                               + np.str(self.dt_max*1.e9) + " ns\n")
        
        self.dE_max = np.sqrt(2.*self.beta**2*self.energy*self.voltage/
                              (np.pi*self.eta_0*self.harmonic))
        self.textwindow.insert('end', "    Half of bucket height: " 
                               + np.str(self.dE_max*1.e-6) + " MeV\n")
       
        self.delta_max = self.dE_max/(self.beta_sq*self.energy)
        self.textwindow.insert('end', "        In relative momentum offset: " 
                               + np.str(self.delta_max) + "\n\n")
        
        
    def emittance_from_bunch_length(self, four_sigma_bunch_length):
        
        self.tau = four_sigma_bunch_length
        if self.tau >= 2.*self.dt_max:
            self.textwindow.insert('end', "Chosen bunch length too large for this bucket. Aborting!")
            raise RuntimeError("Chosen bunch length too large for this bucket. Aborting!")
        self.textwindow.insert('end', "Calculating emittance of 4-sigma bunch length: " 
                               + np.str(self.tau*1.e9) + " ns\n")

        self.textwindow.insert('end', "    Emittance contour in phase: " 
                               + np.str(self.phi_b) + " rad\n")
        self.textwindow.insert('end', "    Emittance contour in relative momentum: " 
                               + np.str(self.delta_b) + "\n")
        self.textwindow.insert('end', "    Emittance contour in energy offset: " 
                               + np.str(self.dE_b*1.e-6) + " MeV\n")
        self.textwindow.insert('end', "    Longitudinal emittance is: " 
                               + np.str(self.emittance) + " eVs\n\n")


    def bunch_length_from_emittance(self, emittance):
        
        self.emittance_aim = emittance
        
        if self.emittance_aim >= self.bucket_area:
            self.textwindow.insert('end', "Chosen emittance too large for this bucket. Aborting!")
            raise RuntimeError("Chosen emittance too large for this bucket. Aborting!")
        self.textwindow.insert('end', "Calculating 4-sigma bunch length for an emittance of " 
                               + np.str(self.emittance_aim) + " eVs\n")

        # Make a guess, iterate to get closer
        self.tau = self.dt_max/2. 
        while (np.fabs((self.emittance - self.emittance_aim)
                       /self.emittance_aim) > 0.001):   
            self.tau *= np.sqrt(self.emittance_aim/self.emittance)

        self.textwindow.insert('end', "    Bunch length is: " 
                               + np.str(self.tau*1.e9) + " ns\n")
        self.textwindow.insert('end', "    Corresponding matched rms relative momentum offset: " 
                               + np.str(self.delta_b) + "\n")
        self.textwindow.insert('end', "    Emittance contour in phase: " 
                               + np.str(self.phi_b) + " rad\n")  


    def dialogue_window(self):
        """Set up password entry dialogue box"""
        
        # Define pop-up frame -------------------------------------------------
        tk.Frame(self.master)
        self.master.geometry('600x800')
        self.master.title('Parameter Scaling')
        
        # Choose machine ------------------------------------------------------
        self.machine = tk.StringVar()
        self.machine.set('LHC') # Default
        tk.Label(self.master, text = """Machine""", justify = tk.LEFT, 
                 anchor = tk.W, font=(16)).place(x = 20, y = 20, width = 100, 
                                                 height = 20)
        machines = [("PSB",'PSB'), ("CPS",'CPS'), ("SPS, Q20",'SPS-Q20'), 
                    ("SPS, Q26", 'SPSQ26'), ("LHC", 'LHC')]
        dx = 0
        for txt, val in machines:
            tk.Radiobutton(self.master, text=txt, variable=self.machine, 
                           value=val, padx = 10, anchor = tk.W).place(x = 20 + 
                           dx, y = 50, width = 100, height = 20)
            dx += 100
        
        # Choose energy -------------------------------------------------------
        self.energy_type = tk.StringVar() 
        self.energy_type.set('flat_bottom') # Default
        tk.Label(self.master, text = """Energy""", justify = tk.LEFT, 
                 anchor = tk.W, font=(16)).place(x = 20, y = 100, width = 100, 
                                                 height = 20) 
        energy_type = [("Flat bottom",'flat_bottom'), ("Flat top",'flat_top'), 
                       ("Custom",'custom')]
        dx = 0
        for txt, val in energy_type:
            tk.Radiobutton(self.master, text=txt, variable=self.energy_type, 
                           value=val, padx = 10, anchor = tk.W).place(x = 20 + 
                           dx, y = 130, width = 100, height = 20)
            dx += 100
        self.custom_energy = tk.Entry(self.master, bd = 5)
        self.custom_energy.place(x = 20 + dx, y = 125, width = 100, 
                                 height = 30)
        dx += 100
        tk.Label(self.master, text = "[eV]", justify = tk.LEFT, 
                 anchor = tk.W).place(x = 20 + dx, y = 130, width = 100, 
                                      height = 20)
                 
        # Choose voltage ------------------------------------------------------
        tk.Label(self.master, text = """Voltage""", justify = tk.LEFT, 
                 anchor = tk.W, font=(16)).place(x = 20, y = 180, width = 100, 
                                                 height = 20) 
        self.voltage = tk.Entry(self.master, bd = 5)
        self.voltage.place(x = 120, y = 175, width = 100, height = 30)
        tk.Label(self.master, text = "[V]", justify = tk.LEFT, 
                 anchor = tk.W).place(x = 220, y = 180, width = 100, 
                                      height = 20)

        # Optional calculation ------------------------------------------------
        tk.Label(self.master, text = """Optional""", justify = tk.LEFT, 
                 anchor = tk.W, font=(16)).place(x = 20, y = 230, width = 100, 
                                                 height = 20) 
        self.switch = tk.IntVar()
        self.switch.set(0) # Default: no optional calculations
        
        tk.Radiobutton(self.master, text="Emittance", variable=self.switch, 
                       value=1, padx = 10, anchor = tk.W).place(x = 20, 
                       y = 260, width = 100, height = 20)        
        self.emittance_target = tk.Entry(self.master, bd = 5)
        self.emittance_target.place(x = 120, y = 255, width = 100, height = 30)
        tk.Label(self.master, text = "[eVs]", justify = tk.LEFT, 
                 anchor = tk.W).place(x = 220, y = 260, width = 100, 
                                      height = 20)
        
        tk.Radiobutton(self.master, text="Bunch length", variable=self.switch, 
                       value=2, padx = 10, anchor = tk.W).place(x = 290, 
                       y = 260, width = 120, height = 20)        
        self.bunch_length_target = tk.Entry(self.master, bd = 5)
        self.bunch_length_target.place(x = 410, y = 255, width = 100, 
                                       height = 30)
        tk.Label(self.master, text = "[s]", justify = tk.LEFT, 
                 anchor = tk.W).place(x = 510, y = 260, width = 100, 
                                      height = 20)
        
        # Submit button -------------------------------------------------------
        button = tk.Button(self.master, text = "Submit", width = 10, 
                           font = (16), justify = tk.CENTER, 
                           command = self.callback)
        button.place(x = 250, y = 320, width = 100, height = 40) 
  
        # Action at pressing enter
        self.master.bind('<Return>', self.callback)
               
        # Scrollbar for output on screen --------------------------------------
        self.scrollbar = tk.Scrollbar(self.master)
        self.textwindow = tk.Text(self.master, height=4, width=50)
        self.scrollbar.place(x = 580, y = 400, width = 10, height = 380)
        self.textwindow.place(x = 10, y = 400, width = 560, height = 380)
        self.scrollbar.config(command=self.textwindow.yview)
        self.textwindow.config(yscrollcommand=self.scrollbar.set)
        

    def callback(self, *args):
        """Get values from dialogue window and execute calculations"""
        
        self.machine = self.machine.get()
        self.energy_type = self.energy_type.get()
        if self.energy_type == 'custom':
            self.custom_energy = self.custom_energy.get()
        self.voltage = self.voltage.get()
        self.switch = self.switch.get()
        self.emittance_target = self.emittance_target.get()
        self.bunch_length_target = self.bunch_length_target.get()

        self.textwindow.insert('end', "Input -- chosen machine: " + 
                               np.str(self.machine) + "\n")

        # Machine-dependent parameters [SI-units] -----------------------------
        gamma_ts = {'PSB': 4.0767, 'CPS': np.sqrt(37.2), 'SPS-Q20': 18., 
                   'SPS-Q26': 22.83, 'LHC': 55.759505}
        harmonics = {'PSB': 1, 'CPS': 21, 'SPS-Q20': 4620, 'SPS-Q26': 4620,
                     'LHC': 35640}
        circumferences = {'PSB': 2*np.pi*25, 'CPS': 2*np.pi*100., 
                          'SPS-Q20': 2*np.pi*1100.009, 
                          'SPS-Q26': 2*np.pi*1100.009, 'LHC': 26658.883}
        energies_fb = {'PSB': (50.e6+m_p*c**2/e), 'CPS': 1.4e9, 'SPS-Q20': 25.94e9, 
                       'SPS-Q26': 25.94e9, 'LHC': 450.e9}
        energies_ft = {'PSB': (1.4e9+m_p*c**2/e), 'CPS': 25.92e9, 'SPS-Q20': 450.e9, 
                       'SPS-Q26': 450.e9, 'LHC': 6.5e12}
        # Machine-dependent parameters [SI-units] -----------------------------
              
        self.gamma_t = gamma_ts[self.machine]
        self.alpha = 1./(self.gamma_t)**2
        self.textwindow.insert('end', "    * with relativistic gamma at transition: " 
                               + np.str(self.gamma_t) + "\n")
        self.textwindow.insert('end', "    * with momentum compaction factor: " 
                               + np.str(self.alpha) + "\n")
        
        self.harmonic = harmonics[self.machine]
        self.textwindow.insert('end', "    * with main harmonic: " 
                               + np.str(self.harmonic) + "\n")
        
        self.circumference = circumferences[self.machine]
        self.textwindow.insert('end', "    * and machine circumference: " 
                               + np.str(self.circumference) + " m\n")
        
        if self.energy_type == 'flat_bottom':
            self.energy = energies_fb[self.machine]
        elif self.energy_type == 'flat_top':
            self.energy = energies_ft[self.machine]
        else:
            try:
                self.energy = np.double(self.custom_energy)
            except ValueError:
                self.textwindow.insert('end', "Energy not recognised. Aborting!")
                raise RuntimeError('Energy not recognised. Aborting!')
        self.textwindow.insert('end', "Input -- synchronous total energy: " 
                               + np.str(self.energy*1.e-6) + " MeV\n")

        try:
            self.voltage = np.double(self.voltage)
        except ValueError:
            self.textwindow.insert('end', "Voltage not recognised. Aborting!")
            raise RuntimeError('Voltage not recognised. Aborting!')
        self.textwindow.insert('end', "Input -- RF voltage: " 
                               + np.str(self.voltage*1.e-6) + " MV\n")

        self.mass = m_p*c**2/e
        self.textwindow.insert('end', "Input -- particle mass: " 
                               + np.str(self.mass*1.e-6) + " MeV\n\n")
                
        # Derived quantities --------------------------------------------------
        self.relativistic_quantities()
        self.frequencies()
        self.tune()
        self.bucket_parameters()
        
        if self.switch == 1:
            try:
                self.emittance_target = np.double(self.emittance_target)
            except ValueError:
                self.textwindow.insert('end', "Target emittance not recognised. Aborting!")
                raise RuntimeError('Target emittance not recognised. Aborting!')
            self.bunch_length_from_emittance(self.emittance_target)
        elif self.switch == 2:
            try:
                self.bunch_length_target = np.double(self.bunch_length_target)
            except ValueError:
                self.textwindow.insert('end', "Target bunch length not recognised. Aborting!")
                raise RuntimeError('Target bunch length not recognised. Aborting!')
            self.emittance_from_bunch_length(self.bunch_length_target)

        #self.master.destroy()


       
        