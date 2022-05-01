///////////////////////////////////////////////////////////////
//  
//      Color_Converter.cu
//      Constaints kernel functions for each of the color converters
//      
//
///////////////////////////////////////////////////////////////
#include "Color_Convert_Kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>




// ##################################################################################################
// ###   kernel_convertRGBtoGrayscale_GLOBAL()   ### 
// Basic global memory RGB to grayscale
__global__ void 
kernel_convertRGBtoGrayscale_GLOBAL(unsigned char* rgbImage, unsigned char* grayImage, int width, int height, int channels)
{
	// Row X and Column Y pixel positions
	int x = threadIdx.x + blockIdx.x * blockDim.x; // x for column
	int y = threadIdx.y + blockIdx.y * blockDim.y; // y for row

	// Only convert pixels in image bounds
	if (x < width && y < height)
	{
		// Pixel Addresses
		int grayAddr = y * width + x; 		    // Gray
		int colorAddr = channels * grayAddr; 	// Color RGB

		// Convert color to grayscale with selected conversion factors
		grayImage[grayAddr] = rgbImage[colorAddr] * 0.21f +   // red
						      rgbImage[colorAddr + 1] * 0.71f +   // green
						      rgbImage[colorAddr + 2] * 0.07f;    // blue
	}
}




// ##################################################################################################
// ###   convertRGBtoGrayscale()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
void convertRGBtoGrayscale_CUDA(unsigned char* hostRgbImage, unsigned char* hostGrayImage, 
                           int imageWidth, int imageHeight, int channels)
{
	
	if (DEBUG_MODE)
	    std::cout << "  COLOR::Parallel Start\n\n";

	unsigned char* deviceRgbImage;
	unsigned char* deviceGrayImage;

	// Allocate device RGB image
	if (DEBUG_MODE)
    	std::cout << "    COLOR::cudaMalloc RGB\n";
	if (cudaMalloc((void**)&deviceRgbImage, imageWidth * imageHeight * channels * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!";
		return;
	}


	// Allocate device Gray image
	if (DEBUG_MODE)
	    std::cout << "    COLOR::cudaMalloc Gray\n";
	if (cudaMalloc((void**)&deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceRgbImage);
		std::cout << "        Error!";
		return;
	}

	// copy RGB image
	if (DEBUG_MODE)
	    std::cout << "    COLOR::cudaMemcpy RGB\n";
	if (cudaMemcpy(deviceRgbImage, hostRgbImage, imageWidth * imageHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceRgbImage);
		cudaFree(deviceGrayImage);
		std::cout << "        Error!";
		return;

	}

	// Do the computation on the GPU
	int blockSize = 32;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	if (DEBUG_MODE)
	    std::cout << "    COLOR::kernel_convertRGBtoGrayscale\n";
	
	// Global Mem
	kernel_convertRGBtoGrayscale_GLOBAL <<<DimGrid,DimBlock>>>(deviceRgbImage, deviceGrayImage, imageWidth, imageHeight, channels);
	
	// Shared mem opt
    //kernel_convertRGBtoGrayscale_SHARED1<<<DimGrid,DimBlock>>>(deviceRgbImage, deviceGrayImage, imageWidth, imageHeight);
    

	if (DEBUG_MODE)
	    std::cout << "    COLOR::cudaMemcpy Gray\n";
	if (cudaMemcpy(hostGrayImage, deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceRgbImage);
		cudaFree(deviceGrayImage);
		return;

	}

	if (DEBUG_MODE)
	    std::cout << "    COLOR::cudaFree\n";
	cudaFree(deviceGrayImage);
	cudaFree(deviceRgbImage);

}


void convertRGBtoGrayscale_CUDA(cv::Mat& input, cv::Mat& output) {

	int image_size = input.total();
	int width = input.cols;
	int height = input.rows;
	unsigned char* host_input = input.data;
	if (0 == output.total()) {
		output.create(height, width, CV_8UC1);
	}
	convertRGBtoGrayscale_CUDA(host_input, output.data, width, height, 3);
}