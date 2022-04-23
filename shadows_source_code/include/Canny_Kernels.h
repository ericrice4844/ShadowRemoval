///////////////////////////////////////////////////////////////
//  
//      Canny_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Sobel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __Canny_KERNELS_H_
#define __Canny_KERNELS_H_


extern "C" void CannyDet(unsigned char hostDirXImage[IM_ROWS][IM_COLS], unsigned char hostDirYImage[IM_ROWS][IM_COLS],
                         unsigned char hostOutputImage[IM_ROWS][IM_COLS],
                         int imageWidth, int imageHeight);


#endif
