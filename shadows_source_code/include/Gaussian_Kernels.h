///////////////////////////////////////////////////////////////
//  
//      Gaussian_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Gaussian_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __GAUSSIAN_KERNELS_H_
#define __GAUSSIAN_KERNELS_H_

extern "C" void GaussianBlur(unsigned char hostGrayImage[IM_ROWS][IM_COLS], unsigned char hostBlurImage[IM_ROWS][IM_COLS],
                             int imageWidth, int imageHeight);




#endif
