
#include "mask.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cooperative_groups.h>

using namespace std::chrono;
using std::to_string;
using namespace cv;

__global__ void mask_diff_glbl(uint8_t* m1, uint8_t* m2, uint8_t* diff,
								int width, int height, int m2Radius) {

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	if (col < width && row < height) {
		int is_bg = 0;
		int start_col = col - m2Radius;
		int start_row = row - m2Radius;
		for (int i = 0; i <= (2 * m2Radius); i++) {
			for (int j = 0; j <= (2 * m2Radius); j++) {
				int cur_col = start_col + j;
				int cur_row = start_row + i;
				if (cur_col >= 0 && cur_col < width && cur_row >= 0 && cur_row < height) {
					is_bg = is_bg + m2[(cur_row * width) + cur_col]);
				}
			}
		}
		diff[col + (row * width)] = ((m1[col + (row * width)] > 0) && (is_bg == 0)) * 255;
	}
}

__global__ void mask_diff_shrd(uint8_t* m1, uint8_t* m2, uint8_t* diff, int width, int height, int m2Radius) {
	__shared__ uint8_t local_m2[20][20];
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int lcl_col = 2 + threadIdx.x;
	int lcl_row = 2 + threadIdx.y;
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			int cur_col = threadIdx.x + (x*blockDim.x);
			int cur_row = threadIdx.y + (y * blockDim.y);
			if (cur_row < 20 && cur_col < 20) {
				int glbl_row = row - 2 + (y * blockDim.y);
				int glbl_col = col - 2 + (x * blockDim.x);
				if (glbl_col >= 0 && glbl_col < width && glbl_row >= 0 && glbl_row < height)
					local_m2[cur_row][cur_col] = m2[((glbl_row)*width) + glbl_col];
				else
					local_m2[cur_row][cur_col] = 0;
			}
		}
	}
	__syncthreads();
	if (col < width && row < height) {
		int is_bg = 0;
		for (int i = -m2Radius; i <= m2Radius; i++) {
			for (int j = -m2Radius; j <= m2Radius; j++) {
				int cur_col = lcl_col + j;
				int cur_row = lcl_row + i;
				atomicAdd(&is_bg,local_m2[cur_row][cur_col]);
			}
		}
		diff[col + (row * width)] = ((m1[(col + (row * width))] > 0) && (is_bg == 0)) * 255;
	}
}


void mask_diff_dbg(uint8_t* m1, uint8_t* m2, uint8_t* diff, int width, int height, int m2Radius) {
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int is_bg = 0;
			int start_col = col - m2Radius;
			int start_row = row - m2Radius;
			for (int i = 0; i <= (2 * m2Radius); i++) {
				for (int j = 0; j <= (2 * m2Radius); j++) {
					int cur_col = start_col + j;
					int cur_row = start_row + i;
					if (cur_col >= 0 && cur_col < width && cur_row >= 0 && cur_row < height)
						is_bg += m2[(cur_row * width) + cur_col];
				}
				diff[col+(row*width)] = (m1[col+(row * width)] > 0 && is_bg==0) * 255;
			}
		}
	}
}


void maskDiff(cv::Mat &m1, cv::Mat &m2, cv::Mat &diff, const int m2Radius) {
	diff.create(m1.size(), CV_8U);

	for (int y = 0; y < m1.rows; ++y) {
		uchar* m1Ptr = m1.ptr(y);
		uchar** m2Ptrs = new uchar * [2 * m2Radius + 1];
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
	// ###   mask_diff_gpu()    ###
	// This function sets up the device memory, calls the kernel, and retrieves the output from the device
	// currently hardcoded to a specific image size 
void mask_diff_gpu(unsigned char* hostImage1, unsigned char* hostImage2, unsigned char* hostDiff, int radius, int imageWidth= IM_COLS, int imageHeight= IM_ROWS) {
	cudaError_t result = cudaSuccess;
	cudaEvent_t start, stop;
	float T0, T1, T2;
	uint8_t* deviceImage1;
	uint8_t* deviceImage2;
	uint8_t* deviceDiff;
	if (TIMER_MODE) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	if (DEBUG_MODE) std::cout << "    mask_diff_gpu::malloc\n";
	result = cudaMalloc((void**)&deviceImage1, imageWidth * imageHeight * sizeof(unsigned char));
	result = cudaMalloc((void**)&deviceImage2, imageWidth * imageHeight * sizeof(unsigned char));
	result = cudaMalloc((void**)&deviceDiff, imageWidth * imageHeight * sizeof(unsigned char));
	if (cudaSuccess == result) {
		//Copy input data to GPU
		cudaMemcpy(deviceImage1, hostImage1, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceImage2, hostImage2, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

		if (TIMER_MODE) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&T0, start, stop);
			cudaEventRecord(start);
		}

		// ##################################################################################################
		// Do the computation on the GPU
		// ##################################################################################################
		int blockSize = 16;
		int grid_width = (((imageWidth - 1)/blockSize) + 1);
		int grid_height = (((imageHeight - 1)/blockSize) + 1);
		dim3 DimGrid(grid_width, grid_height, 1);
		dim3 DimBlock(blockSize, blockSize, 1);
		if (DEBUG_MODE) std::cout << "    mask_diff_gpu::kernel\n";
		mask_diff_glbl <<<DimGrid, DimBlock>>> (deviceImage1, deviceImage2, deviceDiff, imageWidth, imageHeight, radius);
		if (TIMER_MODE) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&T1, start, stop);
			cudaEventRecord(start);
		}
		//Copy result to host
		if (DEBUG_MODE) std::cout << "    mask_diff_gpu::cudaMemcpy\n";
		cudaMemcpy(hostDiff, deviceDiff, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//Free the memory
		if (DEBUG_MODE) std::cout << "    mask_diff_gpu::cudaFree\n";
		cudaFree(deviceImage1);
		cudaFree(deviceImage2);
		cudaFree(deviceDiff);
		if (TIMER_MODE) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&T2, start, stop);

			printf("****************************************\n");
			printf("  \tMask Diff TIMING \n");
			printf("  \tMemSets  : %f msec\n", T0);
			printf("  \tKernel   : %f msec\n", T1);
			printf("  \tCleanup  : %f msec\n", T2);
			printf("  \tTotal    : %f msec\n", (T0 + T1 + T2));
			printf("****************************************\n\n");
		}
	}
}

void mask_diff_gpu(cv::Mat& m1, cv::Mat& m2, cv::Mat& diff, const int radius) {
	int image_size = m1.total();
	int width = m1.cols;
	int height = m1.rows;
	unsigned char* host_m1 = m1.data;
	unsigned char* host_m2 = m2.data;
	if (0 == diff.total()) {
		diff.create(height, width, CV_8UC1);
	}
	mask_diff_gpu(host_m1, host_m2, diff.data, radius, width, height);
}

// ##################################################################################################
// ###   mask_diff_test()    ###
// Tests the GPU against the serial implementation
int mask_diff_test() {
	std::cout << "Testing mask_diff function" << std::endl;
	//Create 2D arrays for GPU Implementation
	const long image_size = IM_ROWS * IM_COLS;
	unsigned char* test_image1;
	test_image1 = new unsigned char[image_size]();
	unsigned char* test_image2;
	test_image2 = new unsigned char[image_size]();
	unsigned char* test_diff;
	test_diff = new unsigned char[image_size]();
	//Create cv::Mats for openCV implementation
	cv::Mat test_mat1(IM_ROWS, IM_COLS, CV_8UC1);
	cv::Mat test_mat2(IM_ROWS, IM_COLS, CV_8UC1);
	cv::Mat test_diff_mat(IM_ROWS, IM_COLS, CV_8UC1);
	cv::Mat test_diff2_mat(IM_ROWS, IM_COLS, CV_8UC1);

	//Create random array
	std::cout << "Creating Random Array" << std::endl;
	for (int i = 0; i < IM_ROWS; i++) {
		for (int j = 0; j < IM_COLS; j++) {
			test_image1[(i * IM_COLS) + j] = rand() % 2;
			test_image2[(i * IM_COLS) + j] = rand() % 2;
			test_mat1.at<uchar>(i, j) = test_image1[(i * IM_COLS) + j];
			test_mat2.at<uchar>(i, j) = test_image2[(i * IM_COLS) + j];
		}
	}
	//Cycle through a small number of radii and compare results
	int error = 0;
	const int num_avg = 8;
	for (int rad = 0; rad < 3; rad++) {
		std::cout << "Radius: " << rad << std::endl;
		auto ser_avg=0, gpu_avg=0;
		for (int avg = 0; avg < num_avg; avg++) {
			auto start = high_resolution_clock::now();
			maskDiff(test_mat1, test_mat2, test_diff_mat, rad);
			auto stop = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(stop - start);
			ser_avg += duration.count();
			//std::cout << "Mask Serial Time: " << duration.count() / 1e6 << " seconds\n";

			start = high_resolution_clock::now();
			mask_diff_gpu(test_mat1, test_mat2, test_diff2_mat, rad);
			stop = high_resolution_clock::now();
			duration = duration_cast<microseconds>(stop - start);
			gpu_avg += duration.count();
			//std::cout << "Mask GPU Time: " << duration.count() / 1e6 << " seconds\n";
		}
		gpu_avg /= num_avg;
		ser_avg /= num_avg;
		std::cout << "Mask Serial Time: " << ser_avg / 1e6 << " seconds\n";
		std::cout << "Mask GPU Time: " << gpu_avg / 1e6 << " seconds\n";
		error = 0;
		for (int i = 0; i < IM_ROWS; i++) {
			for (int j = 0; j < IM_COLS; j++) {
				unsigned char gpu_val, ser_val;
				ser_val = test_diff_mat.at<uchar>(i, j);
				gpu_val = test_diff2_mat.at<uchar>(i, j);
				//gpu_val = test_diff[(i * IM_COLS) + j];
				if (gpu_val != ser_val) {
					std::cout << "Error: Mismatch between serial and GPU Implementation @ [" << to_string(i) <<"," << to_string(j) << "]" << std::endl;
					std::cout << "Pixels M1: " << to_string(test_image1[(i* IM_COLS)+j]) << std::endl << "M2: " << std::endl;
					for (int x = -rad; x <= rad; x++) {
						for (int y = -rad; y <= rad; y++) {
							std::cout << to_string(test_image2[i + x, j + y]) << "\t";
						}
						std::cout << std::endl;
					}
					//std::cout << "Serial Pixels M1: " << to_string(test_mat1.at<uchar>(i, j)) << "\tM2: " << to_string(test_mat2.at<uchar>(i, j)) << std::endl;
					std::cout << "Serial Result: " << to_string(ser_val) << std::endl;
					//std::cout << "GPU Pixels M1: " << to_string(test_image1[(i * IM_COLS) + j]) << "\tM2: " << to_string(test_image2[(i * IM_COLS) + j]) << std::endl;
					std::cout << "GPU Result: " << to_string(gpu_val) << std::endl;

					error++;
					if (error > 5) {
						break;
					}
				}
			}
			if (error > 5) {
				break;
			}
		}
	}
	if (0 == error) {
		std::cout << "Test Passed!" << std::endl;
	}
	return error;
}