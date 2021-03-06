// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "../include/ConnComp.h"
using namespace cv;

ConnComp::ConnComp() {
}

ConnComp::~ConnComp() {
}

void ConnComp::draw(cv::Mat& dst, const cv::Scalar& color, bool filled) const {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	cv::drawContours(dst, contours, -1, color, (filled ? FILLED : 1));
}

void ConnComp::verticalProjection(std::map<int, int>& verticalProjection) const {
	verticalProjection.clear();

	for (int i = 0; i < (int) pixels.size(); ++i) {
		++verticalProjection[pixels[i].x];
	}
}
