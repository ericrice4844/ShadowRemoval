///////////////////////////////////////////////////////////////
//  
//      Sobel_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Sobel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __SOBEL_KERNELS_H_
#define __SOBEL_KERNELS_H_


extern "C" void SobelFilter(unsigned char hostImage[IM_ROWS][IM_COLS],     unsigned char hostMagImage[IM_ROWS][IM_COLS], 
                            unsigned char hostDirXImage[IM_ROWS][IM_COLS], unsigned char hostDirYImage[IM_ROWS][IM_COLS],
                            int imageWidth, int imageHeight);


#endif
