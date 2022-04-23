///////////////////////////////////////////////////////////////
//  
//      Canny_Master_Call.h
//      This should call each of the functions from 
//      Gaussian_Kernels, Sobel_Kernels, Canny_Kernels
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"


// This file includes all headers for the Sobel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __CANNY_MASTER_CALL_H_
#define __CANNY_MASTER_CALL_H_


extern "C" void CannyMasterCall(unsigned char hostGrayImage[IM_ROWS][IM_COLS], unsigned char hostCannyImage[IM_ROWS][IM_COLS],
                             int imageWidth, int imageHeight);


#endif
