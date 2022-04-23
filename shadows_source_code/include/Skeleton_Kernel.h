///////////////////////////////////////////////////////////////
//  
//      Skeleton_Kernel.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Skeleton_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __Skeleton_KERNELS_H_
#define __Skeleton_KERNELS_H_

extern "C" void SkeletonKernel(unsigned char hostInput[IM_ROWS][IM_COLS], unsigned char hostOutput[IM_ROWS][IM_COLS],
                             int imageWidth, int imageHeight);




#endif
