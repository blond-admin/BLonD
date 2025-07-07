/**
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/


BLonD common header file
@Author: Konstantinos Iliakis
@Date: 12.12.2023
*/
#pragma once

#include <complex>
#include "sin.h"
#include "cos.h"
#include "exp.h"

#ifdef USEFLOAT

typedef float real_t;
#define FAST_SIN vdt::fast_sinf
#define FAST_COS vdt::fast_cosf
#define FAST_EXP vdt::fast_expf

#else

typedef double real_t;
#define FAST_SIN vdt::fast_sin
#define FAST_COS vdt::fast_cos
#define FAST_EXP vdt::fast_exp

#endif

typedef std::complex<real_t> complex_t;