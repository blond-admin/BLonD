PYHEADTAIL LONGITUDINAL
==========

Longitudinal version of the CERN PyHeadTail code for the simulation of 
multi-particle beam dynamics with collective effects.

The structure is as follows:

1) the folder __EXAMPLE_MAIN_FILES contains several main_files which
   show how to use the principal features of the code; for additional examples
   have a look at the code developers' personal folders present 
   in the corresponding git branches; 
2) the __doc folder contains the source files for the documentation; 
   to have an output for example in html format, type make html into the console 
   from the folder itself, then go to build, html and open the index file;
   note that you need Latex and dvipng (if not present in the Latex 
   distribution) to be able to see displayed all the math formulas;
3) the various packages which constitute the code;
4) a setup file needed to compile the cython files present in the 
   cython_functions package; this file should be run before launching any 
   simulation; from the console window type "python setup.py cleanall 
   build_ext --inplace".


VERSION CONTENTS
==========

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

