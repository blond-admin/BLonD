[![Build Status](https://travis-ci.org/blond-admin/BLonD.svg?branch=master)](https://travis-ci.org/blond-admin/BLonD)
[![Coverage Status](https://coveralls.io/repos/github/blond-admin/BLonD/badge.svg?branch=master)](https://coveralls.io/github/blond-admin/BLonD?branch=master)

Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3), 
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities 
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.

CODE NAME
=========

BLonD (Beam Longitudinal Dynamics)


DESCRIPTION
===========

CERN code for the simulation of 
longitudinal beam dynamics in synchrotrons.


LINKS
=====

Repository:
	https://github.com/dquartul/BLonD

Documentation:
	http://blond-admin.github.io/BLonD/

Project website:
	http://blond.web.cern.ch


CURRENT DEVELOPERS
==================

Simon Albright (simon.albright (at) cern.ch)
Theodoros Argyropoulos (theodoros.argyropoulos (at) cern.ch)
Konstantinos Iliakis (konstantinos.iliakis (at) cern.ch)
Ivan Karpov (ivan.karpov (at) cern.ch)
Alexandre Lasheen (alexandre.lasheen (at) cern.ch)
Danilo Quartullo (danilo.quartullo (at) cern.ch)
Joel Repond (joel.repond (at) cern.ch)
Helga Timko (Helga.Timko (at) cern.ch)

PREVIOUS DEVELOPERS
===================

Juan Esteban Muller

STRUCTURE
==========

1) the folder __TEST_CASES contains several main files which
   show how to use the principal features of the code; 
2) the __doc folder contains the source files for the documentation on-line; 
3) the various packages which constitute the code;
4) setup_cpp.py is needed to compile all the C++ files present in the project; 
   this file should be run once before launching any simulation. 
   The compiler C++ GCC (at least version 4.8) is necessary.
5) WARNINGS.txt contains useful information related to code usage.


VERSION CONTENTS 
================
2017-03-28
v1.19.0 - Several files have been rearranged and simplified

2017-02-10
v1.18.0 - Fixed an important bug in linear_interp_kick.cpp: before the
          acceleration kick was not applied if rf_kick_interp==TRUE in 
          RingAndRFSection
        
v1.17.0 - Numerical synchrotron frequency distribution added (TC12)
        - Possibility to compute multi-turn wake with acceleration
          (in impedance.py)
        - fixed a bug in the periodicity routine (in tracker.py)

2016-10-24
v1.16.0 - MuSiC algorithm introduced, TC11 added, minor bugs fixed

2016-07-29
v1.15.1 - several upgrades and bug fixes

2016-06-23
v1.14.5 - RF modulation file added in llrf folder
        - documentation on-line for PSB phase loop added
        - setup_cython.py removed because not used

2016-06-21
v1.14.4 - PSB phase loop and rf noise fixed

2016-06-09
v1.14.3 - multi-turn wake revised 
        - PSB phase noise for emittance blow up
        - added support for OpenMP for the cpp functions
        - minor bugs fixed

2016-04-14
v1.14.2 - bug in bigaussian function

2016-04-14
v1.14.1 - cleaning and small bug fixes

2016-03-16
v1.14.0 - fixed bug in periodicity
        - small fixes in impedance module
        - potential well when # max >2

2016-03-10
v1.13.3 - small fixes

2016-02-29
v1.13.2 - bug in periodicity routine fixed
        - added end to end bunch length in bunch generation from matched
          distribution density

2016-02-25
v1.13.1 - Important bux fixed for the below-transition case

2016-02-19
v1.13.0 - Minor updates and bug fixes

2015-9-16
v1.12.0  -  Periodicity added
         -  Updated documentation
         -  Test case 8 added
         -  Multi-turn wake added
         -  Multi RF separatric and phi_s added
         -  Toolbox added

2015-5-13
v1.11.2 - Input paths have been corrected in the test cases main files. 

2015-5-11
v1.11.1 - Fixed two bugs inside the preprocess_ramp routine. 
		- the General_parameters class has been reviewed to take innto account the output from
		  preprocess_ramp
		- a new test case showing how to use the preprocess routines has been added in the
		  _TEST_CASES folder

2015-4-23
v1.11.0 - New features included: multibunch and phase loop
        - The phase-space coordinate theta has changed to time to allow better treatment of phase loop
        - Linear constant space interpolation routine optimised
        - Smoother histogram routine introduced as an option

2015-4-20
v1.10.5 - Test cases reorganized for benchmark between old and new versions

2015-4-07
v1.10.4 - Fixed two bugs in the EXAMPLE_MAIN_FILES

2015-1-15
v1.10.3 - Fixed two bugs in potential_well-cut and beam_profile_derivative routines

2015-1-15
v1.10.2 - Added the n_sampling option to the synchrotron_frequency_tracker.frequency_calculation method

2015-1-13
v1.10.1 - Two bugs fixed

2015-1-13
v1.10.0 - Synchrotron frequency distribution method
	    - Synchrotron frequency tracker method
	    - Reorganized functions between trackers.utilities and beams.distributions
	    - Changing the returns of the matched distribution generation methods
	    - Adding new filtering method (Chebyshev type II) for the line density (mainly for space charge)
	    - preprocess.py and feedbacks.py have been improved
	    - Added plot_parameters.py and improved plot_beams.py and plot_llrf.py
	    - Added the "LHC_ramp" testcase folder
	    - Added the TC 7 and 8 plus the correspondent data files.
	    - Documentation about trackers and llrt have been improved
	    
2014-11-1
v1.9.1 - fixed two bugs in plots routines
	   - implemented Juan's changes on bunch generation routine
	   - implemented Helga's changes on RF noise generation for multi-RF
	   - folder _EXAMPLE_MAIN_FILES has been rearranged	

2014-10-30
v1.9.0 - setup_cpp.py improved
       - now all the tracker (kick, kick_accelleration and drift) has been 
         translated in C++ with autovectorization
       - merged completely with Helga's branch
       - removed linear map in the tracker method
       - removed constant_charge and constant_space routines in the slices method
       - code cleaned
       - added license and warnings files
       - updated documentation
       
2014-10-24
v1.8.0 - Fixed important bug in the slices class related to the histogram
         routine and added an important warning for the user in the README.md
         file

2014-10-22
v1.7.1 - setup_cpp.py has been improved: now it is more user-friendly and the 
         false error which appeared before in the console is gone

2014-10-18
v1.7.0 - Tracker.kick method has been optimised via the CERN VDT math library
	   - Code cleaning and rearrangement
	   - Fixed a bug in the calc_phi_s method
	   - New example file added

2014-10-18
v1.6.1 - Fixed bugs regarding ctypes in slices.py 

2014-10-17
v1.6.0 - New simplified setup files for cython and pure c++ routines
         in Windows and Linux environments
       - preprocess.py added
       - example file _LHC_BUP added
       - fixed two important bugs in slices.py and rf_parameters.py
       - modifications to plot routines

2014-10-14
v1.5.1 - Removed cython calls
	   - Updated c++ setup file

2014-10-13
v1.5.0 - Phase loop implemented and tested
	   - New feedback features in plotting and monitoring
	   - Removed average values (beta etc.) from input_parameters
	   - Plotting: formatting setup is separate, files made "uniform"
	   - Calculated synchronous phase now has the same array length as momentum

2014-10-13
v1.4.0 - The name of the code has been changed from PyLongitudinal to BLond and 
         a new repository has been created especially for this code.
	   - The mpi package has been deleted because not used.
	   - The cython setup file together with the corresponding package has been
	     deleted; now the optimisation routines are coded in C++ and the user
	     must run the setup_cpp_routines.py file.
	   - The histogram.cpp file has been created to make the code faster.

2014-09-16
v1.3.1 - Fixed an important bug in the matched_from_distribution_density method.


2014-09-16
v1.3.0 - Bunch generation from distribution function implemented, input is 
         distribution type, emittance and bunch length.
	   - Bunch generation from line density implemented through Abel transform,
	     input is line density type and bunch length.
	   - Option in beam spectrum calculation in order to compute the 
	     frequency_array only with respect to the sampling and time step.
       - Functions in the induced voltage objects to reprocess the wake/impedance
         sources according to a new slicing.
       - Corrected InputArray object in longitudinal_impedance.
       - Added returns in the induced_voltage_generation in order to return 
         induced_voltage after the frame of the slicing.
       - Initialisation to 0 instead of empty for arrays.
       - Initial guess for gaussian fit corrected.
       - n_macroparticles converted to int in beam class (solve some warnings).
       - Some changes in the plot functions.


2014-08-17
v1.2.3 - The monitor package has been revisited and cleaned, the SlicesMonitor
		 class has been tested.
	   - The warnings derived from the compute_statistics method have been
	     disabled since the NaN entities don't create problems for the plots,
	     arithmetic operations and prints.
	   - The main file Wake_impedance has been corrected and revisited; in it
	     there is an example of how to use the SlicesMonitor class together 
	     with an example of how the option seed is able to generate a clone of 
	     a random generated beam.
	   - Found a very easy way to set MPI on Windows systems (read point 3 in
	     the description of the code structure) so now the user can test
	     the main script in the mpi package on Windows as well.


2014-08-15
v1.2.2 - RNG seeds for numpy.random fixed in RF_noise.py and 
       	 longitudinal_distributions.py
       - updated example files
       - full documentation of RF_noise (see llrf.rst)
       - small improvement of profile plot 
       - change bunchmonitor.dump to bunchmonitor.track
       - MPI parallelized tracker (optional)

2014-08-15
v1.2.1 Elimination of the useless .pyd files from the various packages and of 
	   the _doc/build directory; now the user has to build herself to see the
	   documentation.
	   Fixed a bug in the setup.py file that caused the undesired elimination
	   of all the html and h5 files present in the structure of the code.


2014-08-14
v1.2.0 Reorganisation of the slices module
  		- the different coordinates type is redone, a function to convert values
  		  from one coordinates type to another is included to reduce the code 
  		  length
  		- constant_space is now the reference (and it is constant frame also)
  		- Gaussian fit inside the slice module (and the track method updates the
  		  bunch length in Beams class)
  	   Reorganisation of the longitudinal_impedance module
  		- all calculations are now done in tau
    	- the impedance coming from the impedance table is assumed to be 0 for 
    	  higher frequencies
    	- the wake_calc in InputTable assumes that the wake begins at t=0
    	- the precalculation is always done in InducedVoltageTime unless you use 
    	  constant_charge slicing
   	  	- the input parameters have been changed and the constructor 
   	  	  reorganised accordingly
   		- the number of sampling points for the fft is now a power of 2
   	   PEP8 corrections in the slices and longitudinal_impedance modules
   	   and renaming of some classes and variables.
   	   Corrected cut_left and cut_right calculation for n_sigma (divided by 2).
       Documentation has been improved in the slices and longitudinal_impedance 
       modules.
  	   The monitors module and the various plots modules have been revised
   	   according to these reorganisations; the same is true for the main files
   	   present in the EXAMPLE folder.
   	   Elimination of the developers' personal folders.


2014-08-13
v1.1.2 PEP8 changes:
        - changed LLRF module name to llrf
        - changed GeneralParameters.T0 to t_rev plus correcting the calculation
          method
        - changed RFSectionParameters.sno to section_index
       Put section_number as an option in RFSectionParameters input plus 
       changing its name to section_index.
       Optimization of the longitudinal_tracker.RingAndRFSection object
  	   Applied little changes to documentation.
       Added a method that chooses the solver to be 'simple' or 'full' wrt the 
       input order.
  	   Changed general_parameters to GeneralParameters as an input for
  	   RFSectionParameters.
  	   Changed rf_params to RFSectionParameters as an input for RingAndRFSection
       Secured the cases where the user input momentum compaction with higher 
       orders than 2.


2014-07-23
v1.1.1 Plotting routines now separated into different files:
       beams.plot_beams.py -> phase space, beam statistics plots
       beams.plot_slices.py -> profile, slice statistics
       impedances.plot_impedance.py -> induced voltage, wakes, impedances
       LLRF.plot_llrf.py -> noise in time and frequency domain


2014-07-22
v1.1   Added method in 'longitudinal_impedance' to calculate through the 
       derivative of the bunch profile the induced voltage derived from 
       inductive impedance.
       Added in 'longitudinal_plots' the feature to delete the 'fig' folder 
       for Linux users together with a method for plotting the evolution of
       the position of the bunch.
       Two new example main files showing the use of intensity effects methods
       have been included in the corresponding folder.
       The doc folder has been updated to take into account all the packages
       of the tree.
       Several bugs, mostly in the 'longitudinal_impedance' script, have been
       fixed.


2014-07-17
v1.0   Longitudinal tracker tested. Works for acceleration and multiple
       RF sections.
       Beams and slices ready for transverse features to be added.
       Basic statistics, monitors, and plotting in longitudinal plane.
       Longitudinal bunch generation options. 
       Simple versions of separatrix/Hamiltonian.
       Longitudinal impedance calculations in frequency and time domain.
       RF noise.


