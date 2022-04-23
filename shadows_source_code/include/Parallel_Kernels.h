///////////////////////////////////////////////////////////////
//  
//      Parrallel_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Parallel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __PARALLEL_KERNELS_H_
#define __PARALLEL_KERNELS_H_

extern "C" void initFilter(void); // doesnt actually do anything, just a placeholder for reference




#endif
