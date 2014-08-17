'''
**Module to save beam statistics in h5 file**

:Authors: **Kevin Li**, **Michael Schenk**, **Danilo Quartullo**
'''

import h5py as hp
import numpy as np


class BunchMonitor(object):
    
    ''' Class able to save bunch data into h5 file. The user can choose to save
        just longitudinal statistics, or just transverse statistics, or both.
        If in the constructor a Slices object is passed, that means that one
        wants to save the gaussian-fit bunch length as well (obviously the 
        Slices object has to have the fit_option set to 'gaussian').
    '''
    
    def __init__(self, filename, n_steps, statistics = "All", slices = None):
        
        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        self.i_steps = 0
        self.statistics = statistics
        self.slices = slices
        self.h5file.create_group('Bunch')

    
    def track(self, bunch):
        
        if self.statistics == "All":
            bunch.longit_statistics()
            bunch.transv_statistics()
        elif self.statistics == "Longitudinal":
            bunch.longit_statistics()
        else:
            bunch.transv_statistics()
        
        if not self.i_steps:
            n_steps = self.n_steps
            self.create_data(self.h5file['Bunch'], (n_steps,))
            self.write_data(bunch, self.h5file['Bunch'], self.i_steps)
        else:
            self.write_data(bunch, self.h5file['Bunch'], self.i_steps)

        self.i_steps += 1

    
    def create_data(self, h5group, dims):
        
        h5group.create_dataset("n_macroparticles", dims, compression="gzip", compression_opts=9)
        
        if self.statistics != "Longitudinal": 
            # Transverse statistics
            h5group.create_dataset("mean_x",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_xp",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_y",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_yp",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_z",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_delta",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_x",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_y",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_z",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_delta", dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("epsn_x",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("epsn_y",   dims, compression="gzip", compression_opts=9)
        
        if self.statistics == "All" or self.statistics == "Longitudinal": 
            # Longitudinal statistics
            h5group.create_dataset("mean_theta",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_dE",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_theta",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_dE", dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("epsn_rms_l",   dims, compression="gzip", compression_opts=9)
            if self.slices:
                h5group.create_dataset("bunch_length_gauss_theta", dims, compression="gzip", compression_opts=9)

    
    def write_data(self, bunch, h5group, i_steps):
        
        h5group["n_macroparticles"][i_steps] = bunch.n_macroparticles
        
        if self.statistics != "Longitudinal": 
            # Transverse statistics
            h5group["mean_x"][i_steps]   = bunch.mean_x
            h5group["mean_xp"][i_steps]  = bunch.mean_xp
            h5group["mean_y"][i_steps]   = bunch.mean_y
            h5group["mean_yp"][i_steps]  = bunch.mean_yp
            h5group["mean_z"][i_steps]   = bunch.mean_z
            h5group["mean_delta"][i_steps]  = bunch.mean_delta
            h5group["sigma_x"][i_steps]  = bunch.sigma_x
            h5group["sigma_y"][i_steps]  = bunch.sigma_y
            h5group["sigma_z"][i_steps]  = bunch.sigma_z
            h5group["sigma_delta"][i_steps] = bunch.sigma_delta
            h5group["epsn_x"][i_steps]   = bunch.epsn_x
            h5group["epsn_y"][i_steps]   = bunch.epsn_y
        
        if self.statistics == "All" or self.statistics == "Longitudinal": 
            # Longitudinal statistics
            h5group["mean_theta"][i_steps]   = bunch.mean_theta
            h5group["mean_dE"][i_steps]  = bunch.mean_dE
            h5group["sigma_theta"][i_steps]  = bunch.sigma_theta
            h5group["sigma_dE"][i_steps] = bunch.sigma_dE
            h5group["epsn_rms_l"][i_steps]   = bunch.epsn_rms_l
            if self.slices:
                h5group["bunch_length_gauss_theta"][i_steps] = self.slices.bl_gauss
            
        
    def close(self):
        self.h5file.close()



class SlicesMonitor(object):

    ''' Class able to save slices data into h5 file. The user can choose for 
        now to save the longitudinal statistics only in theta and dE coordinates
        together with the bunch profile. The last is always saved, the former
        are saved only if the statistics_option is set to 'on' in the Slices
        object.
    '''
    def __init__(self, filename, n_steps, slices):
        
        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_steps = n_steps
        self.i_steps = 0
        self.slices = slices
        self.h5file.create_group('Slices')

    
    def track(self, bunch):
        
        if not self.i_steps:
            n_steps = self.n_steps
            n_slices = self.slices.n_slices
            self.create_data(self.h5file['Slices'], (n_slices, n_steps))
            self.write_data(self.slices, self.h5file['Slices'], self.i_steps)
        else:
            self.write_data(self.slices, self.h5file['Slices'], self.i_steps)

        self.i_steps += 1

    
    def create_data(self, h5group, dims):
        
        h5group.create_dataset("n_macroparticles", dims, compression="gzip", compression_opts=9)
        
        if self.slices.statistics_option == 'on':
            
            h5group.create_dataset("mean_theta",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_dE",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_theta",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_dE", dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("eps_rms_l",   dims, compression="gzip", compression_opts=9)
            

    def write_data(self, bunch, h5group, i_steps):
        
        h5group["n_macroparticles"][:, i_steps] = self.slices.n_macroparticles
        
        if self.slices.statistics_option == 'on':
            
            h5group["mean_theta"][:, i_steps] = self.slices.mean_theta
            h5group["mean_dE"][:, i_steps] = self.slices.mean_dE
            h5group["sigma_theta"][:, i_steps] = self.slices.sigma_theta
            h5group["sigma_dE"][:, i_steps] = self.slices.sigma_dE
            h5group["eps_rms_l"][:, i_steps] = self.slices.eps_rms_l
            
            
    def close(self):
        self.h5file.close()



