///////////////////////////////////////////////////////////////
//  
//      FrameProperties_Kernel.cu
//      Parallelized version of Average Attenuation Function
//      using reduction methods
//      
//
///////////////////////////////////////////////////////////////

#include "FrameProperties.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cooperative_groups.h>

using namespace std::chrono;
using std::to_string;
using namespace cv;

// ##################################################################################################
// ###   frameAvgAttenuation_Kernel()   ### 
// Calculates the frame average attenuation
__shared__ float block_atten;
__shared__ int block_count;
__global__ void frame_avg_atten_glbl (
							unsigned char* d_hsvFrame,
							unsigned char* d_hsvBg,
							unsigned char* d_fg,
							int width, int height,
							float* avg_atten,
							int* count
							) {
	
	// X (col) and Y (row) Thread index within the Grid (Global)
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
    
	// 2D indexing turned into 1D indexing for Block
	int idx = width * row + col;
	if (idx == 0) {
		*avg_atten = 0;
		*count = 0.0f;
	}
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		block_count = 0;
		block_atten = 0.0f;
	}
	__syncthreads();
    // Check if in image boundary
	if (col < width && row < height) {
		// Looking at HSV Frame vs HSV Background ratio
		float atten = (float)(10.0f + d_hsvBg[idx * 3 + 2]) / (10 + d_hsvFrame[idx * 3 + 2]);

		// V = value/lightness
		unsigned char is_shadow = (atten > 1 && atten < 5);

		// H = Hue
		int hDiff = abs(d_hsvFrame[idx * 3] - d_hsvBg[idx * 3]);
		//if (hDiff > 90) {
		//	hDiff = 180 - hDiff;
		//}
		// Lower intensities are darker
		is_shadow &= (hDiff < 4) || (hDiff > 176);
		is_shadow &= (d_fg[idx] > 0);
		atten *= is_shadow;
		atomicAdd(&block_atten, atten);
		atomicAdd(&block_count, is_shadow);
	}
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		atomicAdd(avg_atten, block_atten);
		atomicAdd(count, block_count);
	}
}

// ##################################################################################################
// ###   frameAvgAttenuation_Kernel()   ### 
// Calculates the frame average attenuation
float frame_avg_atten_dbg(
	unsigned char* d_hsvFrame,
	unsigned char* d_hsvBg,
	unsigned char* d_fg,
	int width, int height
) {

	// 2D indexing turned into 1D indexing for Block
	int count = 0;
	float avg_atten = 0.0f;
	// Check if in image boundary
	for(int idx=0;idx<(width*height);idx++) {
		if (d_fg[idx] > 0) {
			// Looking at HSV Frame vs HSV Background ratio
			float atten = (float)(10.0f + d_hsvBg[idx * 3 + 2]) / (10.0f + d_hsvFrame[idx * 3 + 2]);

			// V = value/lightness
			unsigned char is_shadow = (atten > 1 && atten < 5);

			// H = Hue
			int hDiff = abs(d_hsvFrame[idx * 3] - d_hsvBg[idx * 3]);
			if (hDiff > 90) {
				hDiff = 180 - hDiff;
			}
			// Lower intensities are darker
			is_shadow &= (hDiff < 4);
			//is_shadow &= (d_fg[idx] > 0);
			//atten *= is_shadow;
			if (is_shadow) {
				avg_atten += atten;
				count++;
			}
		}
	}
	avg_atten /= ((float) count);
	return avg_atten;
}

// ##################################################################################################
// ###   frameAvgAttenuation()   ###
// Gets frame attenuation stats
float frame_avg_atten_ser(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg) {
	float avgAtten = 0;
	int count = 0;
	//Iterate through all rows in hsvFrame
	for (int y = 0; y < hsvFrame.rows; ++y) {
		//Grab row pointers for all inputs
		const uint8_t* fgPtr = fg.ptr(y);
		const uint8_t* framePtr = hsvFrame.ptr(y);
		const uint8_t* bgPtr = hsvBg.ptr(y);
		//Iterate through all columns
		for (int x = 0; x < hsvFrame.cols; ++x) {
			//If FG[pixel] > 0
			if (fgPtr[x] > 0) {
				float atten = (float)(10 + bgPtr[x * 3 + 2]) / (10 + framePtr[x * 3 + 2]);
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
// ###   FrameProperties()    ###
// This function sets up the device memory, calls the kernels, and retrieves the output from the device

float frame_avg_atten_gpu (
								unsigned char* host_hsvFrame,
								unsigned char* host_hsvBg,
								unsigned char* host_fg,
								int imageWidth,
								int imageHeight,
								int imageChannels
								) {
	cudaError_t result = cudaSuccess;
	cudaEvent_t start, stop;
	float T0, T1, T2;
	uint8_t* dev_hsv_frame;
	uint8_t* dev_hsv_bg;
	uint8_t* dev_fg;
	float*	dev_avg_atten;
	float	host_avg_atten;
	int* dev_count;
	int host_count;
	if (TIMER_MODE) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	if (DEBUG_MODE) std::cout << "    frame_avg_atten_gpu::malloc\n";
	result = cudaMalloc((void**)&dev_hsv_frame, imageChannels * imageWidth * imageHeight * sizeof(unsigned char));
	result = cudaMalloc((void**)&dev_hsv_bg, imageChannels * imageWidth * imageHeight * sizeof(unsigned char));
	result = cudaMalloc((void**)&dev_fg, imageChannels * imageWidth * imageHeight * sizeof(unsigned char));
	result = cudaMalloc((void**)&dev_avg_atten, sizeof(float));
	result = cudaMalloc((void**)&dev_count, sizeof(int));

	if (cudaSuccess == result) {
		//Copy input data to GPU
		cudaMemcpy(dev_hsv_frame, host_hsvFrame, imageChannels * imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_hsv_bg, host_hsvBg, imageChannels * imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_fg, host_fg, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

		if (TIMER_MODE) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&T0, start, stop);
			cudaEventRecord(start);
		}

		// ##################################################################################################
		// Do the computation on the GPU
		// ##################################################################################################
		int blockSize = 32;
		int grid_width = (((imageWidth - 1) / blockSize) + 1);
		int grid_height = (((imageHeight - 1) / blockSize) + 1);
		dim3 DimGrid(grid_width, grid_height, 1);
		dim3 DimBlock(blockSize, blockSize, 1);
		if (DEBUG_MODE) std::cout << "    frame_avg_atten_gpu::kernel\n";
		frame_avg_atten_glbl<<<DimGrid, DimBlock>>>(dev_hsv_frame, dev_hsv_bg, dev_fg, imageWidth, imageHeight, dev_avg_atten, dev_count);
		if (TIMER_MODE) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&T1, start, stop);
			cudaEventRecord(start);
		}
		//Copy result to host
		if (DEBUG_MODE) std::cout << "    frame_avg_atten_gpu::cudaMemcpy\n";
		cudaMemcpy(&host_avg_atten, dev_avg_atten, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&host_count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
		host_avg_atten /= host_count;
		//Free the memory
		if (DEBUG_MODE) std::cout << "    frame_avg_atten_gpu::cudaFree\n";
		cudaFree(dev_hsv_frame);
		cudaFree(dev_hsv_bg);
		cudaFree(dev_fg);
		cudaFree(dev_avg_atten);
		if (TIMER_MODE) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&T2, start, stop);

			printf("****************************************\n");
			printf("  \Avg Attenuation TIMING \n");
			printf("  \tMemSets  : %f msec\n", T0);
			printf("  \tKernel   : %f msec\n", T1);
			printf("  \tCleanup  : %f msec\n", T2);
			printf("  \tTotal    : %f msec\n", (T0 + T1 + T2));
			printf("****************************************\n\n");
		}
	}
	return host_avg_atten;
}

float frame_avg_atten_gpu(const cv::Mat& hsvFrame, const cv::Mat& hsvBg, const cv::Mat& fg) {
	int image_size = hsvFrame.total();
	int width = hsvFrame.cols;
	int height = hsvFrame.rows;
	unsigned char* host_frame = hsvFrame.data;
	unsigned char* host_bg = hsvBg.data;
	unsigned char* host_fg = fg.data;
	return frame_avg_atten_gpu(host_frame, host_bg, host_fg, width, height);
}

typedef struct pixel_rgb_s { uchar r, g, b; } rgb_pixel;
// ##################################################################################################
// ###   frame_avg_atten_test()    ###
// Tests the GPU against the serial implementation
int frame_avg_atten_test() {
	std::cout << "Testing frame_avg_atten function" << std::endl;
	//Create 2D arrays for GPU Implementation
	const long image_size = IM_ROWS * IM_COLS * 3;
	//Create cv::Mats for openCV implementation
	cv::Mat ctrlFrame(IM_ROWS, IM_COLS, CV_8UC3);
	cv::Mat ctrlBg(IM_ROWS, IM_COLS, CV_8UC3);
	cv::Mat ctrlFg(IM_ROWS, IM_COLS, CV_8UC1);

	//Create random array
	std::cout << "Creating Random Arrays" << std::endl;
	for (int i = 0; i < IM_ROWS; i++) {
		for (int j = 0; j < IM_COLS; j++) {
			uint8_t frame_pixel_r = rand() % 256;
			uint8_t frame_pixel_g = rand() % 256;
			uint8_t frame_pixel_b = rand() % 256;
			uint8_t bg_pixel_r = rand() % 256;
			uint8_t bg_pixel_g = rand() % 256;
			uint8_t bg_pixel_b = rand() % 256;
			cv::Point3_<uchar> frame_pixel(frame_pixel_r, frame_pixel_g, frame_pixel_b);
			cv::Point3_<uchar> bg_pixel(bg_pixel_r, bg_pixel_g, bg_pixel_b);
			ctrlFrame.at<cv::Point3_<uchar>>(i, j) = frame_pixel;
			ctrlBg.at<cv::Point3_<uchar>>(i, j) = bg_pixel;
			ctrlFg.at<uchar>(i, j) = rand() % 2;
		}
	}
	//Cycle through a small number of radii and compare results
	const int num_avg = 8;
	auto ser_avg = 0, gpu_avg = 0;
	float ser_result, gpu_result;
	for (int avg = 0; avg < num_avg; avg++) {
		auto start = high_resolution_clock::now();
		ser_result = frame_avg_atten_ser(ctrlFrame, ctrlBg, ctrlFg);
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		ser_avg += duration.count();

		start = high_resolution_clock::now();
		gpu_result = frame_avg_atten_gpu(ctrlFrame, ctrlBg, ctrlFg);
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		gpu_avg += duration.count();
	}
	gpu_avg /= num_avg;
	ser_avg /= num_avg;
	std::cout << "Mask Serial Time: " << ser_avg / 1e6 << " seconds\n";
	std::cout << "Mask GPU Time: " << gpu_avg / 1e6 << " seconds\n";
	int error = abs(gpu_result - ser_result);
	if (error < 0.01) {
		std::cout << "Test Passed!" << std::endl;
	}
	return 0;
}
