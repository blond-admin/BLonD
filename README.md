PYHEADTAIL LONGITUDINAL
==========

Longitudinal version of the CERN PyHeadTail code for the simulation of 
multi-particle beam dynamics with collective effects.

Documentation: http://like2000.github.io/PyHEADTAIL/

The structure is as follows:

1) the folder __EXAMPLE_MAIN_FILES contains several main files which
   show how to use the principal features of the code; for additional examples
   have a look at the code developers' personal folders present 
   in the corresponding git branches; 
2) the __doc folder contains the source files for the documentation; 
   to have an output for example in html format, type make html into the console 
   from the folder itself, then go to build, html and open the index file;
   note that you need Latex and dvipng (if not present in the Latex 
   distribution) to be able to see displayed all the math formulas;
   the latest docs should be uploaded to the "gh-pages" branch
3) the various packages which constitute the code together with a beta-version
   package named mpi in which the longitudinal tracker method is parallelised 
   in order to save computational time when the main file RFnoise_mpi is launched.
   If you are using a Windows system download either OpenMPI or MS-MPI together
   with mpi4py from the following link: http://www.lfd.uci.edu/~gohlke/pythonlibs/ 
   which is very useful in general to get easely a lot of Python extension 
   packages for Windows 32-64 bit without building sources.
4) a setup file needed to compile the cython files present in the 
   cython_functions package; this file should be run before launching any 
   simulation; from the console window type "python setup.py cleanall 
   build_ext --inplace".


VERSION CONTENTS
==========

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

