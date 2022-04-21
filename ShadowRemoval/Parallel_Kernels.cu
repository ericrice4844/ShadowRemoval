


#include "Parallel_Kernels.h"

#include <stdio.h>
#include <stdlib.h>


using namespace cv;

// ##################################################################################################
// ###   kernel_convertRGBtoGrayscale_GLOBAL()   ### 
// Basic global memory RGB to grayscale
__global__ void 
kernel_convertRGBtoGrayscale_GLOBAL(uchar* rgbImage, uchar* grayImage, int width, int height, int channels)
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
extern "C" void convertRGBtoGrayscale(uchar hostRgbImage[3120][4160*3], uchar hostGrayImage[3120][4160], int imageWidth, int imageHeight, int channels)
{
	
	std::cout << "  Parallel Start\n\n";

	uchar* deviceRgbImage;
	uchar* deviceGrayImage;

	// Allocate device RGB image
	std::cout << "    cudaMalloc RGB\n";
	if (cudaMalloc((void**)&deviceRgbImage, imageWidth * imageHeight * channels * sizeof(uchar)) != cudaSuccess)
	{
		std::cout << "        Error!";
		return;
	}


	// Allocate device Gray image
	std::cout << "    cudaMalloc Gray\n";
	if (cudaMalloc((void**)&deviceGrayImage, imageWidth * imageHeight * sizeof(uchar)) != cudaSuccess)
	{
		cudaFree(deviceRgbImage);
		std::cout << "        Error!";
		return;
	}

	// copy RGB image
	std::cout << "    cudaMemcpy RGB\n";
	if (cudaMemcpy(deviceRgbImage, hostRgbImage, imageWidth * imageHeight * channels * sizeof(uchar), cudaMemcpyHostToDevice) != cudaSuccess)
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
	std::cout << "    kernel_convertRGBtoGrayscale\n";
	kernel_convertRGBtoGrayscale_GLOBAL <<<DimGrid,DimBlock>>>(deviceRgbImage, deviceGrayImage, imageWidth, imageHeight, channels);

	std::cout << "    cudaMemcpy Gray\n";
	if (cudaMemcpy(hostGrayImage, deviceGrayImage, imageWidth * imageHeight * sizeof(uchar), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceRgbImage);
		cudaFree(deviceGrayImage);
		return;

	}

	std::cout << "    cudaFree\n";
	cudaFree(deviceGrayImage);
	cudaFree(deviceRgbImage);

}



// ##################################################################################################
// ###   initFilter()   ###
// place holder wrapper
extern "C" void initFilter(void)
{
	std::cout << "HERE\n\n";;
	//cv::imshow(rgbImage);
	//cv::waitKey(0);

}
