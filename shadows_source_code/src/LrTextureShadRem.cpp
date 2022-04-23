// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)
#include <stdexcept>
#include "../include/LrTextureShadRem.h"
using namespace cv;
using namespace std::chrono;

const std::vector<cv::Mat> LrTextureShadRem::skeletonKernels = getSkeletonKernels();

// ##################################################################################################
// ###   LrTextureShadRem()   ###
// constructor
LrTextureShadRem::LrTextureShadRem(const LrTextureShadRemParams& params) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	this->params = params;

	frameCount = 0;
	avgAtten = 0;
	avgSat = 0;
	avgPerim = 0;
}

LrTextureShadRem::~LrTextureShadRem() {
}




// ##################################################################################################
// ###   convertCv2Arr_1Chan()   ###
// main function call to identify shadows in an image. Tons of calls in here
void LrTextureShadRem::convertCv2Arr_1Chan(const cv::Mat& frame, unsigned char frameChar[][IM_COLS])
{
	for (int ii = 0; ii < IM_ROWS; ii++)
	{
		for (int jj = 0; jj < IM_COLS; jj++)
		{
			frameChar[ii][jj] = (unsigned char)(frame.at<uchar>(ii, jj));
		}
	}
}

// ##################################################################################################
// ###   convertCv2Arr3Channel()   ###
// main function call to identify shadows in an image. Tons of calls in here
void LrTextureShadRem::convertCv2Arr_3Chan(const cv::Mat& frame, unsigned char frameChar[][IM_COLS*IM_CHAN])
{
	for (int ii = 0; ii < IM_ROWS; ii++)
	{
		for (int jj = 0; jj < IM_COLS; jj++)
		{
			for (int kk = 0; kk < IM_CHAN; kk++)
			{
				frameChar[ii][IM_CHAN*jj+kk] = (unsigned char)(frame.at<Vec3b>(ii, jj)[kk]);
			}
		}
	}
}

// ##################################################################################################
// ###   removeShadows()   ###
// main function call to identify shadows in an image. Tons of calls in here
void LrTextureShadRem::removeShadows(const cv::Mat& frame, const cv::Mat& fgMask, const cv::Mat& bg, cv::Mat& srMask) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	ConnCompGroup fg(fgMask);
	fg.mask.copyTo(srMask);

	// TODO Make color convert parallel

	cv::Mat grayFrame, grayBg, hsvFrame, hsvBg;

    // Check that image size is expected size
	int imRows = frame.rows;
	int imCols = frame.cols;
	int imChan = frame.channels();
	
    if ( imRows != IM_ROWS ) 
    {
           printf("input rows: %d vs %d expected\n", imRows, IM_ROWS);
        throw std::invalid_argument( "\nImage Rows do not match expected values.\n Update path to image or values in Constants.h\n\n;" );
    }
    if ( imCols != IM_COLS ) 
    {
           printf("input cols: %d vs %d expected\n", imCols, IM_COLS);
        throw std::invalid_argument( "\nImage Columns do not match expected values.\n Update path to image or values in Constants.h\n\n;" );
    }
    if ( imChan != IM_CHAN ) 
    {
           printf("input channels: %d vs %d expected\n", imChan, IM_CHAN);
        throw std::invalid_argument( "\nImage Channels do not match expected values.\n Update path to image or values in Constants.h\n\n;" );
    }
	




	// ##############################################################################
	// parallel convert color to gray
	
	auto frameChar = new unsigned char[IM_ROWS][IM_COLS * IM_CHAN]{};
	auto frameGrayChar = new unsigned char[IM_ROWS][IM_COLS]{};
	
	
	// Use this to go from OpenCV Mat to unsigned char array (3 channel)
	convertCv2Arr_3Chan(frame, frameChar);
	
	auto startT3 = high_resolution_clock::now();
	convertRGBtoGrayscale_CUDA(frameChar, frameGrayChar, IM_COLS, IM_ROWS, IM_CHAN);
	
	auto stopT3 = high_resolution_clock::now();
	auto grayscaleTimeParallel = duration_cast<microseconds>(stopT3 - startT3);
	
	
	// ##############################################################################

	cv::cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
	cv::cvtColor(bg, grayBg, COLOR_BGR2GRAY);
	cv::cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
	cv::cvtColor(bg, hsvBg, COLOR_BGR2HSV);
	


	// TODO Make frame stats parallel
	// calculate global frame properties
	auto startT = high_resolution_clock::now();

	avgAtten = ((avgAtten * frameCount) + frameAvgAttenuation(hsvFrame, hsvBg, fg.mask)) / (frameCount + 1);
	avgSat = ((avgSat * frameCount) + frameAvgSaturation(hsvFrame, fg.mask)) / (frameCount + 1);
	avgPerim = ((avgPerim * frameCount) + fgAvgPerim(fg)) / (frameCount + 1);
	++frameCount;

	auto stopT = high_resolution_clock::now();
	auto durationFrameProps = duration_cast<microseconds>(stopT - startT);
	//std::cout << "Frame Prop Calcs: " << durationFrameProps.count() / 1e6 << " seconds\n\n";


	// find candidate shadow pixels
	getCandidateShadows(hsvFrame, hsvBg, fg.mask, candidateShadows);
	
	//cv::imshow("candidateShadows", candidateShadows);
	//cv::waitKey();
	// TODO Make edge finding parallel
	startT = high_resolution_clock::now();
	//std::cout << "Starting Edge diff:\n";
	// split using foreground edges
	auto start2 = high_resolution_clock::now();
	getEdgeDiff(grayFrame, grayBg, fg, candidateShadows, cannyFrame, cannyBg, cannyDiffWithBorders, borders, cannyDiff);
	auto stop2 = high_resolution_clock::now();
	auto deltaT = duration_cast<microseconds>(stop2 - start2);
	//std::cout << "  getEdgeDiff(): " << deltaT.count() / 1e6 << " seconds\n";

	start2 = high_resolution_clock::now();
	cv::Mat splitCandidateShadowsMask;
	maskDiff(candidateShadows, cannyDiff, splitCandidateShadowsMask, params.splitRadius);
	stop2 = high_resolution_clock::now();
	deltaT = duration_cast<microseconds>(stop2 - start2);
	//std::cout << "  maskDiff(): " << deltaT.count() / 1e6 << " seconds\n";

	//cv::imshow("cannyDiff", cannyDiff);
	//cv::imshow("splitCandidateShadowsMask", splitCandidateShadowsMask);
	//cv::waitKey();


	// connected components are candidate shadow regions
	start2 = high_resolution_clock::now();
	splitCandidateShadows.update(splitCandidateShadowsMask, false, false);
	stop2 = high_resolution_clock::now();
	deltaT = duration_cast<microseconds>(stop2 - start2);
	//std::cout << "  splitCandidateShadows.update(): " << deltaT.count() / 1e6 << " seconds\n";

	stopT = high_resolution_clock::now();
	auto durationDetector = duration_cast<microseconds>(stopT - startT);
	//std::cout << "Edge Detector: " << durationDetector.count() / 1e6 << " seconds\n";


	// TODO Post processing parallel
	startT = high_resolution_clock::now();

	shadows.create(grayFrame.size(), CV_8U);
	shadows.setTo(cv::Scalar(0));

	// classify regions with high correlation as shadows
	for (int sr = 0; sr < (int) splitCandidateShadows.comps.size(); ++sr) {
		ConnComp& cc = splitCandidateShadows.comps[sr];

		float regionCorr = getGradDirCorr(grayFrame, cc, grayBg);
		if (regionCorr > gradCorrThresh) {
			cc.draw(shadows, 255);
		}
	}

	bool cleanShadows = (avgAtten < params.avgAttenThresh ? params.cleanShadows : false);
	if (cleanShadows || params.fillShadows || params.minShadowPerim > 0) {
		if (cleanShadows) {
			cv::morphologyEx(shadows, shadows, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
		}

		postShadows.update(shadows, false, params.fillShadows, params.minShadowPerim);
		postShadows.mask.copyTo(shadows);
	}


	stopT = high_resolution_clock::now();
	auto durationPost = duration_cast<microseconds>(stopT - startT);
	//cv::imshow("shadows", shadows);
	//cv::waitKey();

	 
	//std::cout << "Post Processing: "  << durationPost.count()       / 1e6 << " seconds\n";

	srMask.setTo(0, shadows);
	if (params.cleanSrMask || params.fillSrMask) {
		postSrMask.update(srMask, params.cleanSrMask, params.fillSrMask);
		postSrMask.mask.copyTo(srMask);
	}
}


// ##################################################################################################
// ###   frameAvgAttenuation()   ###
// Gets frame attenuation stats
float LrTextureShadRem::frameAvgAttenuation(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	float avgAtten = 0;
	int count = 0;
	for (int y = 0; y < hsvFrame.rows; ++y) {
		const uchar* fgPtr = fg.ptr(y);
		const uchar* framePtr = hsvFrame.ptr(y);
		const uchar* bgPtr = hsvBg.ptr(y);

		for (int x = 0; x < hsvFrame.cols; ++x) {
			if (fgPtr[x] > 0) {
				float atten = (float) (10 + bgPtr[x * 3 + 2]) / (10 + framePtr[x * 3 + 2]);
				bool vIsShadow = (atten > 1 && atten < 5);

				int hDiff = abs(framePtr[x * 3] - bgPtr[x * 3]);
				if (hDiff > 90) {
					hDiff = 180 - hDiff;
				}
				bool hIsShadow = (hDiff < 4);

				if (vIsShadow && hIsShadow) {
					avgAtten += atten;
					++count;
				}
			}
		}
	}

	if (count > 0) {
		avgAtten /= count;
	}

	return avgAtten;
}

// ##################################################################################################
// ###   frameAvgSaturation()   ###
// Gets frame saturation stats
float LrTextureShadRem::frameAvgSaturation(const cv::Mat& hsvFrame, const cv::Mat& fg) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	float avgSat = 0;
	int count = 0;
	int vSum = 0;
	for (int y = 0; y < hsvFrame.rows; ++y) {
		const uchar* fgPtr = fg.ptr(y);
		const uchar* framePtr = hsvFrame.ptr(y);

		for (int x = 0; x < hsvFrame.cols; ++x) {
			if (fgPtr[x] > 0) {
				avgSat += framePtr[x * 3 + 1] * framePtr[x * 3 + 2];
				vSum += framePtr[x * 3 + 2];
				++count;
			}
		}
	}

	if (count > 0) {
		avgSat /= vSum;
	}

	return avgSat;
}


// ##################################################################################################
// ###   fgAvgPerim()   ###
// Gets frame perimeter stats
float LrTextureShadRem::fgAvgPerim(const ConnCompGroup& fg) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	float avgPerim = 0;

	for (int i = 0; i < (int) fg.comps.size(); ++i) {
		avgPerim += fg.comps[i].contours[0].size();
	}

	if (fg.comps.size() > 0) {
		avgPerim /= fg.comps.size();
	}

	return avgPerim;
}


// ##################################################################################################
// ###   maskDiff()   ###
// Does masking of image. TODO Need to look at more
void LrTextureShadRem::maskDiff(cv::Mat& m1, cv::Mat& m2, cv::Mat& diff, const int m2Radius) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	diff.create(m1.size(), CV_8U);
	

	for (int y = 0; y < m1.rows; ++y) {
		uchar* m1Ptr = m1.ptr(y);
		uchar** m2Ptrs = new uchar*[2 * m2Radius+1];
		int count = 0;
		for (int y2 = y - m2Radius; y2 <= y + m2Radius; ++y2) {
			if (y2 < 0 || y2 >= m1.rows) {
				m2Ptrs[count] = NULL;
			}
			else {
				m2Ptrs[count] = m2.ptr(y2);
			}

			++count;
		}
		uchar* diffPtr = diff.ptr(y);

		for (int x = 0; x < m1.cols; ++x) {
			bool isInBg = false;
			for (int i = 0; i < count && !isInBg; ++i) {
				if (m2Ptrs[i]) {
					for (int x2 = x - m2Radius; x2 <= x + m2Radius && !isInBg; ++x2) {
						if (x2 >= 0 && x2 < m1.cols) {
							if (m2Ptrs[i][x2] > 0) {
								isInBg = true;
							}
						}
					}
				}
			}

			if (m1Ptr[x] > 0 && !isInBg) {
				diffPtr[x] = 255;
			}
			else {
				diffPtr[x] = 0;
			}
		}
	}
}



// ##################################################################################################
// ###   getSkeleton()   ###
// This is the slowest function (about 60 seconds on large image)
void LrTextureShadRem::getSkeleton(const cv::Mat& mask, cv::Mat& skeleton) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	cv::Mat tmpMask = mask.clone();
	tmpMask.copyTo(skeleton);
	
    int iterCount = 0;
    int maxIter   = 50;
	bool changed = true;
	while (changed && iterCount < maxIter) {
		changed = false;
		iterCount++;

		for (int k = 0; k < (int) skeletonKernels.size(); ++k) {
			for (int y = 1; y < tmpMask.rows - 1; ++y) {
				uchar* skeletonPtr = skeleton.ptr(y);

				for (int x = 1; x < tmpMask.cols - 1; ++x) {
					bool allMatch = true;
					for (int dy = -1; dy <= 1 && allMatch; ++dy) {
						uchar* maskPtr = tmpMask.ptr(y + dy);
						const uchar* kernelPtr = skeletonKernels[k].ptr(1 + dy);

						for (int dx = -1; dx <= 1 && allMatch; ++dx) {
							uchar maskVal = maskPtr[x + dx];
							uchar kernelVal = kernelPtr[1 + dx];

							if (kernelVal != 127 && maskVal != kernelVal) {
								allMatch = false;
							}
						}
					}

					if (allMatch && skeletonPtr[x] > 0) {
						skeletonPtr[x] = 0;
						changed = true;
						break;
					}
				}
			}

			if (changed) {
				skeleton.copyTo(tmpMask);
			}
		}
	}
}



// ##################################################################################################
// ###   getSkeletonKernels()   ###
// Creates kernel for skeleton function above
std::vector<cv::Mat> LrTextureShadRem::getSkeletonKernels() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	std::vector<cv::Mat> skeletonKernels(8);

	skeletonKernels[0].create(3, 3, CV_8U);
	skeletonKernels[0].at<uchar>(0, 0) = 0;
	skeletonKernels[0].at<uchar>(0, 1) = 0;
	skeletonKernels[0].at<uchar>(0, 2) = 0;
	skeletonKernels[0].at<uchar>(1, 0) = 127;
	skeletonKernels[0].at<uchar>(1, 1) = 255;
	skeletonKernels[0].at<uchar>(1, 2) = 127;
	skeletonKernels[0].at<uchar>(2, 0) = 255;
	skeletonKernels[0].at<uchar>(2, 1) = 255;
	skeletonKernels[0].at<uchar>(2, 2) = 255;

	skeletonKernels[2] = skeletonKernels[0].t();
	cv::flip(skeletonKernels[0], skeletonKernels[4], 0);
	cv::flip(skeletonKernels[2], skeletonKernels[6], 1);

	skeletonKernels[1].create(3, 3, CV_8U);
	skeletonKernels[1].at<uchar>(0, 0) = 127;
	skeletonKernels[1].at<uchar>(0, 1) = 0;
	skeletonKernels[1].at<uchar>(0, 2) = 0;
	skeletonKernels[1].at<uchar>(1, 0) = 255;
	skeletonKernels[1].at<uchar>(1, 1) = 255;
	skeletonKernels[1].at<uchar>(1, 2) = 0;
	skeletonKernels[1].at<uchar>(2, 0) = 127;
	skeletonKernels[1].at<uchar>(2, 1) = 255;
	skeletonKernels[1].at<uchar>(2, 2) = 127;

	cv::flip(skeletonKernels[1], skeletonKernels[3], 1);
	cv::flip(skeletonKernels[3], skeletonKernels[5], 0);
	cv::flip(skeletonKernels[5], skeletonKernels[7], 1);
	
	//for (int ii = 0; ii<8; ii++)
	//    std::cout << "Skeleton " << ii << "\n" << skeletonKernels[ii] << "\n\n\n";

	return skeletonKernels;
}


// ##################################################################################################
// ###   getCandidateShadows()   ###
// TODO - describe
void LrTextureShadRem::getCandidateShadows(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg, cv::Mat& hsvMask) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

	float vThreshLower = (avgAtten < params.avgAttenThresh ? params.vThreshLowerLowAtten : params.vThreshLowerHighAtten);
	float vThreshUpper = (avgAtten < params.avgAttenThresh ? params.vThreshUpperLowAtten : params.vThreshUpperHighAtten);
	float hThresh = (avgSat < params.avgSatThresh ? params.hThreshLowSat : params.hThreshHighSat);
	float sThresh = (avgSat < params.avgSatThresh ? params.sThreshLowSat : params.sThreshHighSat);

	hsvMask.create(hsvFrame.size(), CV_8U);
	hsvMask.setTo(cv::Scalar(0));
	for (int y = 0; y < hsvFrame.rows; ++y) {
		const uchar* hsvFramePtr = hsvFrame.ptr(y);
		const uchar* hsvBgPtr = hsvBg.ptr(y);
		const uchar* fgPtr = fg.ptr(y);
		uchar* hsvMaskPtr = hsvMask.ptr(y);

		for (int x = 0; x < hsvFrame.cols; ++x) {
			if (fgPtr[x] > 0) {
				float vRatio = (float) hsvFramePtr[x * 3 + 2] / hsvBgPtr[x * 3 + 2];
				bool vIsShadow = (vRatio > vThreshLower && vRatio < vThreshUpper);

				int hDiff = abs(hsvFramePtr[x * 3] - hsvBgPtr[x * 3]);
				if (hDiff > 90) {
					hDiff = 180 - hDiff;
				}
				bool hIsShadow = (hDiff < hThresh);

				int sDiff = hsvFramePtr[x * 3 + 1] - hsvBgPtr[x * 3 + 1];
				bool sIsShadow = (sDiff < sThresh);

				if (vIsShadow && hIsShadow && sIsShadow) {
					hsvMaskPtr[x] = 255;
				}
			}
		}
	}
}


// ##################################################################################################
// ###   getEdgeDiff()   ###
// TODO - describe
void LrTextureShadRem::getEdgeDiff(const cv::Mat& grayFrame, const cv::Mat& grayBg, const ConnCompGroup& fg,
		const cv::Mat& candidateShadows, cv::Mat& cannyFrame, cv::Mat& cannyBg, cv::Mat& cannyDiffWithBorders,
		cv::Mat& borders, cv::Mat& cannyDiff) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	cv::Mat invCandidateShadows(candidateShadows.size(), CV_8U, cv::Scalar(255));
	invCandidateShadows.setTo(0, candidateShadows);

	auto startT = high_resolution_clock::now();
	cv::Canny(grayFrame, cannyFrame, params.cannyThresh1, params.cannyThresh2, params.cannyApertureSize, params.cannyL2Grad);
	cannyFrame.setTo(0, invCandidateShadows);

	auto stopT = high_resolution_clock::now();
	auto deltaT = duration_cast<microseconds>(stopT - startT);
	
	
	
	// Convert openCV Mat to array of uchar 
	// small sample image: 384  x 288  x 3
	// large sample image: 3120 x 4160 x 3
	auto frameGrayChar = new unsigned char[IM_ROWS][IM_COLS]{};
	auto frameGrayBlurChar = new unsigned char[IM_ROWS][IM_COLS]{};
	auto frameSobelEdgeMagChar = new unsigned char[IM_ROWS][IM_COLS]{};
	auto frameSobelEdgeXDirChar = new unsigned char[IM_ROWS][IM_COLS]{};
	auto frameSobelEdgeYDirChar = new unsigned char[IM_ROWS][IM_COLS]{};
	auto frameCannyChar = new unsigned char[IM_ROWS][IM_COLS]{};
	auto frameCannyChar2 = new unsigned char[IM_ROWS][IM_COLS]{};
	
	// Convert gray CV Mat to char array
	convertCv2Arr_1Chan(grayFrame, frameGrayChar);
	
	// ##############################################################################
	// Test Blur Kernel here
	auto startT4 = high_resolution_clock::now();
	//GaussianBlur(frameGrayChar, frameGrayBlurChar, IM_COLS, IM_ROWS);
	
	auto stopT4 = high_resolution_clock::now();
	auto gaussBlurTimeParallel = duration_cast<microseconds>(stopT4 - startT4);
	
	// ##############################################################################
	// Test Sobel Kernel here
	startT4 = high_resolution_clock::now();
	//SobelFilter(frameGrayBlurChar, frameSobelEdgeMagChar, frameSobelEdgeXDirChar, frameSobelEdgeYDirChar, IM_COLS, IM_ROWS);
	
	stopT4 = high_resolution_clock::now();
	auto sobelFilterTimeParallel = duration_cast<microseconds>(stopT4 - startT4);
	
	// ##############################################################################
	// Test Canny Kernel here
	startT4 = high_resolution_clock::now();
	//CannyDet(frameSobelEdgeXDirChar, frameSobelEdgeYDirChar, frameCannyChar, IM_COLS, IM_ROWS);
	
	stopT4 = high_resolution_clock::now();
	auto cannyFilterTimeParallel = duration_cast<microseconds>(stopT4 - startT4);
	
	
	// ##############################################################################
	// Fast Canny Here - contains Gauss+Sobel+Canny
	auto startTMaster = high_resolution_clock::now();
	CannyMasterCall(frameGrayChar, frameCannyChar2, IM_COLS, IM_ROWS);
	auto stopTMaster = high_resolution_clock::now();
	auto cannyFilterTimeParallel2 = duration_cast<microseconds>(stopTMaster - startTMaster);
	
	// ##############################################################################
	
	// Use this to go from unsigned char array to OpenCV Mat 
	//cv::Mat tempBlur(IM_ROWS, IM_COLS, CV_8UC1, frameGrayBlurChar);
	//cv::Mat tempSobelMag(IM_ROWS, IM_COLS, CV_8UC1, frameSobelEdgeMagChar);
	//cv::Mat tempSobelXDir(IM_ROWS, IM_COLS, CV_8UC1, frameSobelEdgeXDirChar);
	//cv::Mat tempSobelYDir(IM_ROWS, IM_COLS, CV_8UC1, frameSobelEdgeYDirChar);
	//cv::Mat tempCanny(IM_ROWS, IM_COLS, CV_8UC1, frameCannyChar);
	cv::Mat tempCanny2(IM_ROWS, IM_COLS, CV_8UC1, frameCannyChar2);

	
    std::cout << "****************************************************" << "\n\n";
	//std::cout << "  gaussBlurTimeParallel     : " << gaussBlurTimeParallel.count() / 1e3 << " msec\n";
	//std::cout << "  sobelFilterTimeParallel   : " << sobelFilterTimeParallel.count() / 1e3 << " msec\n";
	//std::cout << "  cannyFilterTimeParallel   : " << cannyFilterTimeParallel.count() / 1e3 << " msec\n";
	//std::cout << "  Canny Total Parallel Time : " << (gaussBlurTimeParallel.count()+sobelFilterTimeParallel.count()+cannyFilterTimeParallel.count()) / 1e3 << " msec\n";
	
	std::cout << "\n  Fast Canny Total Parallel Time   : " << cannyFilterTimeParallel2.count() / 1e3 << " msec\n";
	
	
	std::cout << "\n  Open CVCanny Time: " << deltaT.count() / 1e3 << " msec\n";
	
	
	std::cout << "\n  speedup: x" << float(deltaT.count()) / cannyFilterTimeParallel2.count() << "\n\n";
    std::cout << "****************************************************" << "\n\n";
	
	//cv::imshow("cannyFrame", cannyFrame);
	//cv::waitKey();
	cv::imwrite("cannyImages/cannyCV.jpg", cannyFrame);
	//cv::imwrite("cannyImages/cannyCuda-Slow.jpg", tempCanny);
	cv::imwrite("cannyImages/cannyCuda-Fast.jpg", tempCanny2);

	startT = high_resolution_clock::now();
	cv::Canny(grayBg, cannyBg, params.cannyThresh1, params.cannyThresh2, params.cannyApertureSize, params.cannyL2Grad);
	cannyBg.setTo(0, invCandidateShadows);

	stopT = high_resolution_clock::now(); 
	deltaT = duration_cast<microseconds>(stopT - startT);
	//std::cout << "   Canny2 IM time: " << deltaT.count() / 1e6 << " seconds\n";

	int edgeDiffRadius = (avgPerim > params.avgPerimThresh ? params.edgeDiffRadius : 0);

	startT = high_resolution_clock::now();
	maskDiff(cannyFrame, cannyBg, cannyDiffWithBorders, edgeDiffRadius);
	stopT = high_resolution_clock::now();
	deltaT = duration_cast<microseconds>(stopT - startT);
	//std::cout << "   maskDiff IM time: " << deltaT.count() / 1e6 << " seconds\n";


	startT = high_resolution_clock::now();
	int borderDiffRadius = (avgAtten < params.avgAttenThresh ? params.borderDiffRadius : 2);
	borderDiffRadius = (avgPerim > params.avgPerimThresh ? borderDiffRadius : 0);
	borders.create(fg.mask.size(), CV_8U);
	borders.setTo(cv::Scalar(0));
	fg.draw(borders, cv::Scalar(255, 255, 255), false);
	if (borderDiffRadius > 0) {
		cv::dilate(borders, borders, cv::Mat(borderDiffRadius, borderDiffRadius, CV_8U, cv::Scalar(255)));
	}

	stopT = high_resolution_clock::now();
	deltaT = duration_cast<microseconds>(stopT - startT);
	//std::cout << "   dilate IM time: " << deltaT.count() / 1e6 << " seconds\n";



    
	//startT = high_resolution_clock::now();
	int splitIncrement = (avgAtten < params.avgAttenThresh ? params.splitIncrement : 2);
	splitIncrement = (avgPerim > params.avgPerimThresh ? splitIncrement : 0);
	cannyDiffWithBorders.copyTo(cannyDiff);
	cannyDiff.setTo(0, borders);
	if (splitIncrement > 0) {
		cv::dilate(cannyDiff, cannyDiff, cv::Mat(splitIncrement, splitIncrement, CV_8U, cv::Scalar(255)));
		
		
		
        
	    // ##############################################################################
	    // Test skeleton Kernel here
	
        auto frameCannyDiffChar = new unsigned char[IM_ROWS][IM_COLS]{};
        auto frameSkeletonChar = new unsigned char[IM_ROWS][IM_COLS]{};
        
	    convertCv2Arr_1Chan(cannyDiff, frameCannyDiffChar);
	    
        cv::Mat myCannyDiff(IM_ROWS, IM_COLS, CV_8UC1, frameCannyDiffChar);
        
        // Run it
	    auto startTpar = high_resolution_clock::now();
	
        SkeletonKernel(frameCannyDiffChar, frameSkeletonChar, IM_COLS, IM_ROWS);
        
        auto stopTpar = high_resolution_clock::now();
        auto deltaTSkeleton = duration_cast<microseconds>(stopTpar - startTpar);
        
        
	    cv::Mat mySkeleton(IM_ROWS, IM_COLS, CV_8UC1, frameSkeletonChar);
        
	    cv::imwrite("skeletonImages/skeletonParallel.jpg", mySkeleton);
        //cv::imshow("mySkeleton", mySkeleton);
		
		
	    // ##############################################################################
		// Serial version
	    startT = high_resolution_clock::now();
	    
		getSkeleton(cannyDiff, cannyDiff);
	    mySkeleton.copyTo(cannyDiff);
	    
		stopT = high_resolution_clock::now();
	    deltaT = duration_cast<microseconds>(stopT - startT);
	
	
	    // ##############################################################################
	    
	    
        std::cout << "****************************************************" << "\n\n";
        std::cout << "   SkeletonKernel-Parallel: " << deltaTSkeleton.count() / 1e3 << " msec\n\n";
        
	    std::cout << "   SkeletonKernel-Serial  : " << deltaT.count() / 1e3 << " msec\n";
	    std::cout << "   speedup : x" << float(deltaT.count())/deltaTSkeleton.count() << "\n\n";
	
	
        std::cout << "****************************************************" << "\n\n";
	    cv::imwrite("skeletonImages/skeletonSerial.jpg", cannyDiff);
	
        //cv::imshow("mainSkeleton", cannyDiff);
        //cv::waitKey();
	}
}



// ##################################################################################################
// ###   getGradDirCorr()   ###
// TODO - describe
float LrTextureShadRem::getGradDirCorr(const cv::Mat& grayFrame, const ConnComp& cc, const cv::Mat& grayBg) {
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	std::vector<int> deltas(params.gradScales);
	deltas[0] = 1;
	for (int i = 1; i < params.gradScales; ++i) {
		deltas[i] = int(pow(2, i));
	}

	gradCorrThresh = (
			avgAtten < params.avgAttenThresh ? params.gradCorrThreshLowAtten : params.gradCorrThreshHighAtten);

	int minCorrPoints = (avgPerim > params.avgPerimThresh ? params.minCorrPoints : 3);
	int round = 1;
	int nGradDirs = -1;
	int nSimilarGradDirs = 0;
	while (nGradDirs < minCorrPoints && round <= params.maxCorrRounds) {
		nGradDirs = 0;
		nSimilarGradDirs = 0;

		for (int i = 0; i < (int) cc.pixels.size(); ++i) {
			int x = cc.pixels[i].x;
			int y = cc.pixels[i].y;

			if (x < grayFrame.cols - 1 && y < grayFrame.rows - 1) {
				int boxMaskX = x - cc.box.x;
				int boxMaskY = y - cc.box.y;

				float frMaxGradMag = -1;
				float frMaxGradDir;
				float bgMaxGradMag = -1;
				float bgMaxGradDir;

				for (int d = 0; d < (int) deltas.size(); ++d) {
					// avoid pixels in the border of the component
					int boxMaskX1 = boxMaskX - params.corrBorder;
					int boxMaskX2 = boxMaskX + params.corrBorder + deltas[d];
					int boxMaskY1 = boxMaskY - params.corrBorder;
					int boxMaskY2 = boxMaskY + params.corrBorder + deltas[d];
					if (boxMaskX1 >= 0 && boxMaskX2 < cc.box.width && boxMaskY1 >= 0 && boxMaskY2 < cc.box.height) {
						const uchar* ccMaskU = cc.boxMask.ptr(boxMaskY1) + boxMaskX;
						const uchar* ccMaskD = cc.boxMask.ptr(boxMaskY2) + boxMaskX;
						const uchar* ccMaskL = cc.boxMask.ptr(boxMaskY) + (boxMaskX1);
						const uchar* ccMaskR = cc.boxMask.ptr(boxMaskY) + (boxMaskX2);

						if ((*ccMaskU > 0) && (*ccMaskD > 0) && (*ccMaskL > 0) && (*ccMaskR > 0)) {
							// search scale with largest gradient

							// calculate gradients
							float frVal = grayFrame.ptr(y)[x];
							float frNextXVal = grayFrame.ptr(y)[x + deltas[d]];
							float frNextYVal = grayFrame.ptr(y + deltas[d])[x];
							float bgVal = grayBg.ptr(y)[x];
							float bgNextXVal = grayBg.ptr(y)[x + deltas[d]];
							float bgNextYVal = grayBg.ptr(y + deltas[d])[x];
							float frGx = frNextXVal - frVal;
							float frGy = frVal - frNextYVal;
							float bgGx = bgNextXVal - bgVal;
							float bgGy = bgVal - bgNextYVal;
							float frMagnitude = sqrt(frGx * frGx + frGy * frGy);
							float bgMagnitude = sqrt(bgGx * bgGx + bgGy * bgGy);

							if (frMagnitude + bgMagnitude > frMaxGradMag + bgMaxGradMag) {
								frMaxGradMag = frMagnitude;
								bgMaxGradMag = bgMagnitude;
								frMaxGradDir = atan2(frGy, frGx);
								bgMaxGradDir = atan2(bgGy, bgGx);
							}
						}
					}
				}

				// use gradient if magnitude is big enough
				float magThresh = params.gradMagThresh / pow(2, round);
				if (frMaxGradMag > magThresh || bgMaxGradMag > magThresh) {
					if (bgMaxGradMag / frMaxGradMag > params.gradAttenThresh) {
						float angle = abs(frMaxGradDir - bgMaxGradDir);
						angle = std::min(angle, (float) (2 * CV_PI - angle));

						// add to gradient directions sums
						if (angle < params.gradDistThresh) {
							++nSimilarGradDirs;
						}
					}

					++nGradDirs;
				}
			}
		}

		++round;
	}

	if (nGradDirs < minCorrPoints || nGradDirs == 0) {
		return 0;
	}

	return (float) nSimilarGradDirs / nGradDirs;
}
