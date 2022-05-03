// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utils/logger.hpp"
#include "../include/LrTextureShadRem.h"
#include "mask.h"
using namespace cv;
using namespace std::chrono;


int main(int argc, char** argv) {
	int use_cuda = 0;
	if (argc > 1) {
		char* cuda_arg = argv[1];
		use_cuda = atoi(cuda_arg);
		if (use_cuda) {
			std::cout << "Implementation Specified: GPU" << std::endl;
		}
		else {
			std::cout << "Implementation Specified: CPU" << std::endl;
		}
	}
	else {
		std::cout << "Implementation Unspecified: Default to CPU" << std::endl;
	}
	cv::setNumThreads(1);
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

	std::cout << "\nBeginning Image Processing\n\n";

	// load frame, background and foreground 3120x4160
	cv::Mat frame = cv::imread("../shadows_source_code/samples/BackYard/frame2.jpg");
	cv::Mat bg    = cv::imread("../shadows_source_code/samples/BackYard/bg2.jpg");
	cv::Mat blank = cv::imread("../shadows_source_code/samples/BackYard/blank2.bmp", IMREAD_GRAYSCALE);
	cv::Mat fg    = cv::imread("../shadows_source_code/samples/BackYard/fg2.bmp", IMREAD_GRAYSCALE);

    
    // 288x384
	//cv::Mat frame = cv::imread("../shadows_source_code/samples/frame1.bmp");
	//cv::Mat bg = cv::imread("../shadows_source_code/samples/bg1.bmp");
	//cv::Mat blank = cv::imread("../shadows_source_code/samples/blank1.bmp", IMREAD_GRAYSCALE);
	//cv::Mat fg = cv::imread("../shadows_source_code/samples/fg1.bmp", IMREAD_GRAYSCALE);

	std::cout << "Images Loaded \n\n";
	//cv::Mat frame_show = frame.clone();
	//cv::imshow("pre", frame_show);
	//cv::resizeWindow("pre", 500, 500);
	//
	// create shadow removers
	LrTextureShadRem lrTex;
	lrTex.use_cuda = use_cuda;

	// matrices to store the masks after shadow removal
	cv::Mat lrTexMask;

	// remove shadows
	auto start = high_resolution_clock::now();
	lrTex.removeShadows(frame, blank, bg, lrTexMask);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << "\nImage size: " << frame.size;
	std::cout << "\nTotal Time: " << duration.count() / 1e6 << " seconds\n";

	// show results
	//cv::imshow("frame", frame);
	//cv::resizeWindow("frame", 500, 500);
	
	//cv::imshow("bg", bg);
	//cv::resizeWindow("bg", 500, 500);
	
	//cv::imshow("fg", fg);
	//cv::resizeWindow("fg", 500, 500);
	
	cv::imshow("lrTex", lrTexMask);
	cv::resizeWindow("lrTex", 500, 500);
	
	if (use_cuda){
	    cv::imwrite("Final/lrTexMask-CUDA.jpg", lrTexMask);
    } else{
	    cv::imwrite("Final/lrTexMask-CPU.jpg", lrTexMask);
    }

	cv::waitKey();
	
	return 0;
}
