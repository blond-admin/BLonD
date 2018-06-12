.. image:: https://travis-ci.org/blond-admin/BLonD.svg?branch=master
    :target: https://travis-ci.org/blond-admin/BLonD
.. image:: https://ci.appveyor.com/api/projects/status/m3p1lq18s3ex6q3u/branch/master?svg=true
    :target: https://ci.appveyor.com/project/blond-admin/blond/branch/master
.. image:: https://coveralls.io/repos/github/blond-admin/BLonD/badge.svg?branch=master
    :target: https://coveralls.io/github/blond-admin/BLonD?branch=master


Copyright 2016 CERN. This software is distributed under the terms of the
GNU General Public Licence version 3 (GPL Version 3), copied verbatim in
the file LICENCE.txt. In applying this licence, CERN does not waive the
privileges and immunities granted to it by virtue of its status as an
Intergovernmental Organization or submit itself to any jurisdiction.

CODE NAME
=========

BLonD (Beam Longitudinal Dynamics)

DESCRIPTION
===========

CERN code for the simulation of longitudinal beam dynamics in
synchrotrons.

LINKS
=====

Repository: https://github.com/dquartul/BLonD

Documentation: http://blond-admin.github.io/BLonD/

Project website: http://blond.web.cern.ch

INSTALL
=======


Requirements
------------

1. A gcc compiler with C++11 support (version greater than 4.8.4).  

2. An Anaconda distribution (Python 3 recommended).

3. That's all!


Install Steps
-------------


* The easy way: 
    .. code-block:: bash

        $ pip install blond


* If this fails try this:

  1. Clone the repository from github or download and extract the zip from here_.

  2. Navigate to the downloaded BLonD directory and run:

    .. code-block:: bash

        $ python setup.py install


* If it still fails, navigate to the BLonD directory and run:

  1.  
    .. code-block:: bash
      
       $ python setup.py compile

  2. Then you have to use the PYTHONPATH variable or some other mechanism to point to the BLonD installation.


* In the extremely rare occassion that it continues to fail, you can submit an issue and we will handle it ASAP. 


Confirm proper installation
---------------------------

* Run the unittests with pytest (may need to be installed first with pip install pytest):
    .. code-block:: bash

        $ pytest -v unittests

* Try to run some of the main files found in the examples:
    .. code-block:: bash

        $ python __EXAMPLES/main_files/EX_01_Acceleration.py
        $ python __EXAMPLES/main_files/EX_02_Main_long_ps_booster.py
        $ etc..

CURRENT DEVELOPERS
==================

* Simon Albright (simon.albright (at) cern.ch)
* Theodoros Argyropoulos (theodoros.argyropoulos (at) cern.ch)
* Konstantinos Iliakis (konstantinos.iliakis (at) cern.ch)
* Ivan Karpov (ivan.karpov (at) cern.ch)
* Alexandre Lasheen (alexandre.lasheen (at) cern.ch)
* Danilo Quartullo (danilo.quartullo (at) cern.ch)
* Joel Repond (joel.repond (at) cern.ch)
* Helga Timko (Helga.Timko (at) cern.ch)

PREVIOUS DEVELOPERS
===================

Juan Esteban Muller

STRUCTURE
=========

* the folder \__TEST_CASES contains several main files which show how to use the principal features of the code;
* the \__doc folder contains the source files for the documentation on-line;
* the various packages which constitute the code;
* setup_cpp.py is needed to compile all the C++ files present in the project; this file should be run once before launching any simulation. The compiler C++ GCC (at least version 4.8) is necessary.
* WARNINGS.txt contains useful information related to code usage.

VERSION CONTENTS
================

+ 2017-03-28 v1.19.0 - Several files have been rearranged and simplified

+ 2017-02-10 v1.18.0 - Fixed an important bug in linear_interp_kick.cpp: before the acceleration kick was not applied if rf_kick_interp==TRUE in RingAndRFSection

+ v1.17.0 - Numerical synchrotron frequency distribution added (TC12) - Possibility to compute multi-turn wake with acceleration (inimpedance.py) - fixed a bug in the periodicity routine (in tracker.py)

+ 2016-10-24 v1.16.0 - MuSiC algorithm introduced, TC11 added, minor bugs fixed

+ 2016-07-29 v1.15.1 - several upgrades and bug fixes

+ 2016-06-23 v1.14.5 - RF modulation file added in llrf folder - documentation on-line for PSB phase loop added - setup_cython.py removed because not used

+ 2016-06-21 v1.14.4 -


.. _here: https://github.com/blond-admin/BLonD/archive/master.zip