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
using namespace cv;
using namespace std::chrono;



int main() {

	cv::setNumThreads(1);
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

	std::cout << "\nBeginning Image Processing\n\n";

	// load frame, background and foreground
	cv::Mat frame = cv::imread("../shadows_source_code/samples/frame.bmp");
	cv::Mat bg    = cv::imread("../shadows_source_code/samples/bg.bmp");
	cv::Mat blank = cv::imread("../shadows_source_code/samples/blank.bmp", IMREAD_GRAYSCALE);
	cv::Mat fg    = cv::imread("../shadows_source_code/samples/fg.bmp", IMREAD_GRAYSCALE);


	
	// create shadow removers
	LrTextureShadRem lrTex;

	// matrices to store the masks after shadow removal
	cv::Mat lrTexMask;

	// remove shadows
	auto start = high_resolution_clock::now();
	lrTex.removeShadows(frame, blank, bg, lrTexMask);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	// show results

	cv::imshow("frame", frame);
	cv::imshow("bg", bg);
	cv::imshow("fg", fg);
	cv::imshow("lrTex", lrTexMask);


	std::cout << "\n\nDone!\n";
	std::cout << "\nImage size: " << frame.size;
	std::cout << "\nTotal Time: " << duration.count() / 1e6 << " seconds\n";

	cv::waitKey();
	
	return 0;
}
