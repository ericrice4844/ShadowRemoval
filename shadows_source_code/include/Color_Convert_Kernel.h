///////////////////////////////////////////////////////////////
//  
//      Color_Convert_Kernel.h
//      Constaints headers for each color convert kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Parallel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __COLOR_CONVERT_KERNEL_H_
#define __COLOR_CONVERT_KERNEL_H_



extern "C" void convertRGBtoGrayscale_CUDA(unsigned char rgbImage[IM_ROWS][IM_COLS*IM_CHAN], 
                           unsigned char grayImage[IM_ROWS][IM_COLS], 
                           int width, int height, int channels);
                           
          

#endif                 
                           
