#pragma once


// This file includes all headers for the Parallel_Kernels.cu functions
// Provides access to main cpp code.


#ifndef __PARALLEL_KERNELS_H_
#define __PARALLEL_KERNELS_H_

#include <opencv2/opencv.hpp>
#include "opencv2/core/utils/logger.hpp"
using namespace cv;


extern "C" void convertRGBtoGrayscale(uchar rgbImage[3120][4160*3], uchar grayImage[3120][4160], int width, int height, int channels);
extern "C" void initFilter(void);




#endif