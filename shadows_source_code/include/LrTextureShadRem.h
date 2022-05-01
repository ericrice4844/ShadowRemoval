// Copyright (C) 2011 NICTA (www.nicta.com.au)
// Copyright (C) 2011 Andres Sanin
//
// This file is provided without any warranty of fitness for any purpose.
// You can redistribute this file and/or modify it under the terms of
// the GNU General Public License (GPL) as published by the
// Free Software Foundation, either version 3 of the License
// or (at your option) any later version.
// (see http://www.opensource.org/licenses for more info)

#ifndef LRTEXTURESHADREM_H_
#define LRTEXTURESHADREM_H_


#include "Constants.h"

#include <opencv2/opencv.hpp>
#include "LrTextureShadRemParams.h"
#include "ConnCompGroup.h"
#include "opencv2/core/utils/logger.hpp"
#include <vector>

#include "Parallel_Kernels.h"
#include "Color_Convert_Kernel.h"
#include "Gaussian_Kernels.h"
#include "Sobel_Kernels.h"
#include "Canny_Kernels.h"
#include "Canny_Master_Call.h"
#include "Skeleton_Kernel.h"

/*
*/

/**
 * Implemented from:
 *    Improved shadow removal for robust person tracking in surveillance scenarios
 *    Sanin et al. (ICPR 2010)
 *
 * Extended to split candidate shadow regions using foreground edges
 */
class LrTextureShadRem {

	public:
		cv::Mat candidateShadows;
		cv::Mat cannyFrame;
		cv::Mat cannyBg;
		cv::Mat cannyDiffWithBorders;
		cv::Mat borders;
		cv::Mat cannyDiff;
		ConnCompGroup splitCandidateShadows;

		cv::Mat distances;
		cv::Mat avgDistances;
		cv::Mat shadows;

		float gradCorrThresh;

		LrTextureShadRem(const LrTextureShadRemParams& params = LrTextureShadRemParams());
		virtual ~LrTextureShadRem();

		void removeShadows(const cv::Mat& frame, const cv::Mat& fg, const cv::Mat& bg, cv::Mat& srMask);
		
		
		// Functions to convert from CV to array. One for 3 channel and one for 1 channel images
		void convertCv2Arr_3Chan(const cv::Mat& frame, unsigned char frameChar[IM_ROWS][IM_COLS*IM_CHAN]);
		void convertCv2Arr_1Chan(const cv::Mat& frame, unsigned char frameChar[IM_ROWS][IM_COLS]);


	private:
		static const std::vector<cv::Mat> skeletonKernels;

		LrTextureShadRemParams params;

		int frameCount;
		float avgAtten;
		float avgSat;
		float avgPerim;

		ConnCompGroup postShadows;
		ConnCompGroup postSrMask;

		cv::Mat frHist;
		cv::Mat bgHist;

		static float frameAvgAttenuation(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg);
		static float frameAvgSaturation(const cv::Mat& hsvFrame, const cv::Mat& fg);
		static float fgAvgPerim(const ConnCompGroup& fg);
		static void maskDiff(cv::Mat& m1, cv::Mat& m2, cv::Mat& diff, int m2Radius);
		static void getSkeleton(const cv::Mat& mask, cv::Mat& skeleton);
		static std::vector<cv::Mat> getSkeletonKernels();

		void getCandidateShadows(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg, cv::Mat& hsvMask);
		void getEdgeDiff(const cv::Mat& grayFrame, const cv::Mat& grayBg, const ConnCompGroup& fg,
				const cv::Mat& candidateShadows, cv::Mat& cannyFrame, cv::Mat& cannyBg, cv::Mat& cannyDiffWithBorders,
				cv::Mat& borders, cv::Mat& cannyDiff);
		float getGradDirCorr(const cv::Mat& grayFrame, const ConnComp& cc, const cv::Mat& grayBg);
};

#endif /* LRTEXTURESHADREM_H_ */
