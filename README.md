PYHEADTAIL LONGITUDINAL
==========

Longitudinal version of the CERN HeadTail code for simulation of multi-particle 
beam dynamics with collective effects.

The structure is the following:

1) 5 folders reserved for the current members of the "longitudinal team" for
   their main files, input and output data	
2) a folder called doc for the documentation (type make html into the console 
   from the folder itself, then go to build, html and open the index file)
3) the various packages which are the basis of the simulation code
4) this README.md file
5) a setup file to compile the various cython files present in the 
   cython_functions package; this file should be run before launching any 
   simulation
