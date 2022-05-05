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
#include <opencv2/opencv.hpp>
// This file includes all headers for the Parallel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __COLOR_CONVERT_KERNEL_H_
#define __COLOR_CONVERT_KERNEL_H_



void convertRGBToGrayAndHSV(unsigned char* rgbImage, 
                                       unsigned char* grayImage, 
						               unsigned char* hsvImage,
                                       int width, 
                                       int height, 
                                       int channels);

void convertRGBToGrayAndHSV(const cv::Mat& input, cv::Mat& output1, cv::Mat& output2);
#endif                 
                           
