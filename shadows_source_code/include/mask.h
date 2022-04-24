///////////////////////////////////////////////////////////////
//  
//      mask.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////
#pragma once
#include "Constants.h"
#include <opencv2/opencv.hpp>

// This file includes all headers for the Sobel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __MASK_H_
#define __MASK_H_


void mask_diff_gpu(unsigned char* hostImage1, unsigned char* hostImage2, unsigned char* hostDiff, int radius, int imageWidth, int imageHeight);
void mask_diff_gpu(cv::Mat& m1, cv::Mat& m2, cv::Mat& diff, const int radius);
int mask_diff_test();

#endif
