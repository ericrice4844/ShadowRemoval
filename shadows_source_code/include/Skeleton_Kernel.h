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
#include <opencv2/opencv.hpp>


// This file includes all headers for the Skeleton_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __Skeleton_KERNELS_H_
#define __Skeleton_KERNELS_H_

void SkeletonKernel(unsigned char* hostInput, unsigned char* hostOutput, int imageWidth, int imageHeight);
void SkeletonKernel(cv::Mat& input, cv::Mat& output);




#endif
