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
#include <cuda_profiler_api.h>

// ##################################################################################################
// ###   convertRGBToGrayAndHSV()   ### 
// Global implementation of the RGB to Grayscale and RGB to HSV color conversions
__global__ void 
convertRGBToGrayAndHSV_global(unsigned char* rgbImage, 
                       unsigned char* grayImage,
                       unsigned char* hsvImage,
                       int width, 
                       int height, 
                       int channels)
{
	// Row X and Column Y pixel positions
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;

	// Only convert pixels in the image bounds
	if (Col < width && Row < height)
	{
		// Pixel Addresses
		int grayAddr = Row * width + Col;
		int colorAddr = channels * grayAddr;

		float R = rgbImage[colorAddr];
		float G = rgbImage[colorAddr + 1];
		float B = rgbImage[colorAddr + 2];

		// Convert color to grayscale with selected conversion factors
		grayImage[grayAddr] = R * 0.21f +   // Red
						      G * 0.71f +   // Green
						      B * 0.07f;    // Blue 

		// Initialize Max and Min values used for the HSV calculations
		float Cmax = 0;
		float Cmin = 0;

		// Normalize RGB values between 0 - 1
		R = R/255.0;
		G = G/255.0;
		B = B/255.0;

		// Determine the largest and smallest value between RGB
		if ((R > G) && (R > B) ){
			Cmax = R;
			if (G >= B){
				Cmin = B;
			}
			else{
				Cmin = G;
			}
		}
		else if ((G > R) && (G > B) ){
			Cmax = G;
			if (R >= B){
				Cmin = B;
			}
			else{
				Cmin = R;
			}
		}
		else if ((B > R) && (B > G) ){
			Cmax = B;
			if (R >= G){
				Cmin = G;
			}
			else{
				Cmin = R;
			}
		}
		else if ((R == G) && (R > B) ){
			Cmax = R;
			Cmin = B;
		}
		else if ((R == B) && (R > G) ){
			Cmax = R;
			Cmin = G;
		}
		else if ((G == B) && (G > R) ){
			Cmax = G;
			Cmin = R;
		}
		else{
			Cmax = R;
			Cmin = R;
		}
		
		// Initialize delta and H,S, and V
		float delta = Cmax - Cmin;
		float H = 0;
		float S = 0;
		float V = Cmax;

		// Compute the Hue
		if (delta == 0){
			H = 0;
		}
		else if (R == Cmax){
			H = ((60.0*((G-B)/delta)));
		}
		else if (G == Cmax){
			H = (60.0*((B-R)/delta)+120);
		}
		else {
			H = (60.0*((R-G)/delta)+240);
		}

		// Compute the Saturation
		if (V == 0){
			S = 0;
		}
		else {
			S = (delta/Cmax);
		}

		// Wrtie the HSV results back to memory
		hsvImage[colorAddr + 0] = H/2;
		hsvImage[colorAddr + 1] = S*255;
		hsvImage[colorAddr + 2] = V*255;
	}
}

// ##################################################################################################
// ###   convertRGBToGrayAndHSV()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
void convertRGBToGrayAndHSV(unsigned char* hostRgbImage, 
                                       unsigned char* hostGrayImage, 
                                       unsigned char* hostHsvImage,
                                       int imageWidth, 
                                       int imageHeight, 
                                       int channels)
{

	// Initialize timing variables
	cudaProfilerStart();
	cudaEvent_t start0, start1, start1B, start2, start3, start4, stop0, stop1, stop1B, stop2, stop3, stop4;
	float T0, T1, T1B, T2, T3, T4;

	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);
	cudaEventRecord(start0);

	unsigned char* deviceRgbImage;
	unsigned char* deviceGrayImage;
    unsigned char* deviceHsvImage;

	// Start recording the time for device memory allocation 
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);

    // Allocate device RGB image memory
	if (cudaMalloc((void**)&deviceRgbImage, imageWidth * imageHeight * channels * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!";
		return;
	}

	// Allocate device Gray image memory
	if (cudaMalloc((void**)&deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceRgbImage);
		std::cout << "        Error!";
		return;
	}

	// Allocate device HSV image memory
	if (cudaMalloc((void**)&deviceHsvImage, imageWidth * imageHeight * channels * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceRgbImage);
		std::cout << "        Error!";
		return;
	}

	// Stop recording the time for device memory allocation 
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&T1, start1, stop1);

	// Start recording the time it takes to copy the RGB image from the host to the device
	cudaEventCreate(&start1B);
	cudaEventCreate(&stop1B);
	cudaEventRecord(start1B);

	// copy RGB image to the device global memory
	if (cudaMemcpy(deviceRgbImage, hostRgbImage, imageWidth * imageHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceRgbImage);
		cudaFree(deviceGrayImage);
		cudaFree(deviceHsvImage);
		std::cout << "        Error!";
		return;
	}

	// Stop recording the time it takes to copy the RGB image from the host to the device
	cudaEventRecord(stop1B);
	cudaEventSynchronize(stop1B);
	cudaEventElapsedTime(&T1B, start1B, stop1B);

	// Define the grid and block size
	int blockSize = 32;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);

	// Start recording the color conversion main kernel
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2);

	// Call the kernel that does the Grayscale and HSV conversion
	convertRGBToGrayAndHSV_global <<<DimGrid,DimBlock>>>(deviceRgbImage, 
                                                  deviceGrayImage, 
                                                  deviceHsvImage, 
                                                  imageWidth, 
                                                  imageHeight, 
                                                  channels);

	// Stop recording the color conversion main kernel
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&T2, start2, stop2);
    
	// Start recording the time it copy the images from global memory back to the host
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3);	

	if (cudaMemcpy(hostGrayImage, deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceRgbImage);
		cudaFree(deviceGrayImage);
        cudaFree(deviceHsvImage);
		return;
	}

	if (cudaMemcpy(hostHsvImage, deviceHsvImage, imageWidth * imageHeight * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceRgbImage);
        cudaFree(deviceGrayImage);
		cudaFree(deviceHsvImage);
		return;
	}

	// Stop recording the time it copy the images from global memory back to the host
	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&T3, start3, stop3);	

	// Start recording the time it takes to free up memory
	cudaEventCreate(&start4);
	cudaEventCreate(&stop4);
	cudaEventRecord(start4);

	// Free up memory that is no longer needed
	cudaFree(deviceGrayImage);
	cudaFree(deviceRgbImage);
	cudaFree(deviceHsvImage);

	// Stop recording the time it takes to free up memory
	cudaEventRecord(stop4);
	cudaEventSynchronize(stop4);
	cudaEventElapsedTime(&T4, start4, stop4);

	// Stop recording the time for the entire function call
	cudaEventRecord(stop0);
	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&T0, start0, stop0);

	// Write timing results
	printf("\n");
	printf("========= Gray Timing Details Start =========\n");
	printf("\n");
	printf(" \tTotal Function Time          : %f msec\n", T0);
	printf(" \tAllocate device memory time  : %f msec\n", T1);
	printf(" \tHost to device RGB copy time : %f msec\n", T1B);
	printf(" \tColor conversion kernel time : %f msec\n", T2);
	printf(" \tDevice to host copy time     : %f msec\n", T3);
	printf(" \tMemory free up time          : %f msec\n", T4);
	printf("\n");
	printf("========= Gray Timing Details End =========\n");
	printf("\n");
	cudaProfilerStop();
}

void convertRGBToGrayAndHSV(const cv::Mat& input, cv::Mat& output1, cv::Mat& output2){
	int image_size = input.total();
	int width = input.cols;
	int height = input.rows;
	unsigned char* host_input = input.data;
	if (0 == output1.total()){
		output1.create(height, width, CV_8UC1);
	}
	if (0 == output2.total()){
		output2.create(height, width, CV_8UC3);
	}
	convertRGBToGrayAndHSV(host_input, output1.data, output2.data, width, height, 3);
}
