'''
**Module to save beam statistics in h5 files**

:Authors: **Danilo Quartullo**, **Kevin Li**
'''

import h5py as hp


class BunchMonitor(object):
    
    ''' Class able to save bunch data into h5 file.
        If in the constructor a Slices object is passed, that means that one
        wants to save the gaussian-fit bunch length as well (obviously the 
        Slices object has to have the fit_option set to 'gaussian').
    '''
    
    def __init__(self, filename, n_turns, slices = None, 
                 PhaseLoop = None, LHCNoiseFB = None):
        
        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_turns = n_turns
        self.i_turn = 0
        self.slices = slices
        self.PL = PhaseLoop
        self.LHCNoiseFB = LHCNoiseFB
        self.h5file.create_group('Bunch')

    
    def track(self, bunch):
        
        bunch.statistics()
        
        if not self.i_turn:
            n_turns = self.n_turns
            self.create_data(self.h5file['Bunch'], (n_turns,))
            self.write_data(bunch, self.h5file['Bunch'], self.i_turn)
        else:
            self.write_data(bunch, self.h5file['Bunch'], self.i_turn)

        self.i_turn += 1

    
    def create_data(self, h5group, dims):
        
        h5group.create_dataset("n_macroparticles_alive", dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_theta",   dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("mean_dE",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_theta",  dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("sigma_dE", dims, compression="gzip", compression_opts=9)
        h5group.create_dataset("epsn_rms_l",   dims, compression="gzip", compression_opts=9)
        if self.slices:
            h5group.create_dataset("bunch_length_gauss_theta", dims, compression="gzip", compression_opts=9)
        if self.PL:
            h5group.create_dataset("PL_phase_corr", dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("PL_omegaRF_corr", dims, compression="gzip", compression_opts=9)
        if self.LHCNoiseFB:
            h5group.create_dataset("LHC_noise_scaling", dims, compression="gzip", compression_opts=9)
                

    def write_data(self, bunch, h5group, i_turn):
        
        h5group["n_macroparticles_alive"][i_turn] = bunch.n_macroparticles_alive
        h5group["mean_theta"][i_turn]   = bunch.mean_theta
        h5group["mean_dE"][i_turn]  = bunch.mean_dE
        h5group["sigma_theta"][i_turn]  = bunch.sigma_theta
        h5group["sigma_dE"][i_turn] = bunch.sigma_dE
        h5group["epsn_rms_l"][i_turn]   = bunch.epsn_rms_l
        if self.slices:
            h5group["bunch_length_gauss_theta"][i_turn] = self.slices.bl_gauss
        if self.PL:
            h5group["PL_phase_corr"][i_turn] = self.PL.dphi
            h5group["PL_omegaRF_corr"][i_turn] = self.PL.domega_RF_next
        if self.LHCNoiseFB:
            h5group["LHC_noise_scaling"][i_turn] = self.LHCNoiseFB.x

        
    def close(self):
        self.h5file.close()



class SlicesMonitor(object):

    ''' Class able to save slices data into h5 file. The user can save the 
        statistics in theta and dE coordinates together with the bunch profile.
        The last is always saved, the former are saved only if the 
        statistics_option is set to 'on' in the Slices object.
    '''
    
    def __init__(self, filename, n_turns, slices):
        
        self.h5file = hp.File(filename + '.h5', 'w')
        self.n_turns = n_turns
        self.i_turn = 0
        self.slices = slices
        self.h5file.create_group('Slices')

    
    def track(self, bunch):
        
        if not self.i_turn:
            n_turns = self.n_turns
            n_slices = self.slices.n_slices
            self.create_data(self.h5file['Slices'], (n_slices, n_turns))
            self.write_data(self.slices, self.h5file['Slices'], self.i_turn)
        else:
            self.write_data(self.slices, self.h5file['Slices'], self.i_turn)

        self.i_turn += 1

    
    def create_data(self, h5group, dims):
        
        h5group.create_dataset("n_macroparticles", dims, compression="gzip", compression_opts=9)
        
        if self.slices.statistics_option == 'on':
            
            h5group.create_dataset("mean_theta",   dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("mean_dE",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_theta",  dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("sigma_dE", dims, compression="gzip", compression_opts=9)
            h5group.create_dataset("eps_rms_l",   dims, compression="gzip", compression_opts=9)
            

    def write_data(self, bunch, h5group, i_turn):
        
        h5group["n_macroparticles"][:, i_turn] = self.slices.n_macroparticles
        
        if self.slices.statistics_option == 'on':
            
            h5group["mean_theta"][:, i_turn] = self.slices.mean_theta
            h5group["mean_dE"][:, i_turn] = self.slices.mean_dE
            h5group["sigma_theta"][:, i_turn] = self.slices.sigma_theta
            h5group["sigma_dE"][:, i_turn] = self.slices.sigma_dE
            h5group["eps_rms_l"][:, i_turn] = self.slices.eps_rms_l
            
            
    def close(self):
        self.h5file.close()



