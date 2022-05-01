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


#ifndef __FRAME_PROPERTIES_H_
#define __FRAME_PROPERTIES_H_


float frame_avg_atten_gpu(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg);
float frame_avg_atten_gpu(unsigned char* host_hsvFrame, unsigned char* host_hsvBg, unsigned char* host_fg, int imageWidth = IM_COLS, int imageHeight = IM_ROWS, int imageChannels = 3);
int frame_avg_atten_test();

#endif
