PYHEADTAIL LONGITUDINAL v1.0
==========

Longitudinal version of the CERN PyHeadTail code for the simulation of multi-particle 
beam dynamics with collective effects.

The structure is as follows:

1) for example main files, see __EXAMPLE_MAIN_FILES; contains examples for using
   the longitudinal package with acceleration, several RF stations, etc.
2) 5 folders reserved for the current members of the "longitudinal team" for
   their main files, input and output data	
3) the doc folder contains the documentation (type make html into the console 
   from the folder itself, then go to build, html and open the index file)
4) the various packages which are the basis of the simulation code
5) this README.md file
6) a setup file to compile the various cython files present in the 
   cython_functions package; this file should be run before launching any 
   simulation; from the console window type "python setup.py cleanall 
   build_ext --inplace"
