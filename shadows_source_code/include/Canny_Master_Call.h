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
#include <opencv2/opencv.hpp>


// This file includes all headers for the Sobel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __CANNY_MASTER_CALL_H_
#define __CANNY_MASTER_CALL_H_


void CannyMasterCall(unsigned char* hostGrayImage, unsigned char* hostCannyImage, int imageWidth, int imageHeight);
void CannyMasterCall(const cv::Mat& input, cv::Mat& output);


#endif
