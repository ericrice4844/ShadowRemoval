///////////////////////////////////////////////////////////////
//  
//      Gaussian_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////

#include "Skeleton_Kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;




__constant__ unsigned char s_Kernel[NUM_SKELETONS][SKELETON_WIDTH][SKELETON_WIDTH];



// ##################################################################################################
// ###   kernel_getSkeleton_GLOBAL()   ### 
// Global version of skeleton kernel
__global__ void
kernel_getSkeleton_GLOBAL(unsigned char* in, unsigned char* out, int width, int height)
{
	// Index variable setup
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	int idx = width*row + col;
	
	
	// first need to clone input into output matrix 
	if ((row < height) && (col < width))
	    out[idx]=in[idx];
	    
    // make sure copy is done
	__syncthreads();
	
	

    int iterCount = 0;
    int maxIter = 100;
	// check the boundary condition
    if( col > 0 && row > 0 && col < width-1 && row < height-1) 
	{
	    // Continue to iterate if changes were found
	    bool changed = true;
	    while (changed && iterCount < maxIter) 
	    {
		    changed = false;
		    iterCount++;
            
            // iterate over all skeleton kernels
		    for (int k = 0; k < NUM_SKELETONS; k++) 
		    {
			    bool allMatch = true;
			    
			    // Check for matching kernel values to image input (see s_Kernel)
			    for (int dRow = -1; dRow <= 1 && allMatch; dRow++) 
			    {

				    for (int dCol = -1; dCol <= 1 && allMatch; dCol++) 
				    {
					    int maskVal = in[(row+dRow)*width + col + dCol];
					    int kernelVal = s_Kernel[k][1+dCol][1 + dRow];

					    if (kernelVal != 127 && maskVal != kernelVal) 
						    allMatch = false;
				    }
				    
			    }
			    __syncthreads();

                // if match, change
			    if (allMatch && out[idx] > 0) 
			    {
			        //printf("Idx %d - Changed from %d to 0 \n", idx, out[idx]);
				    out[idx] = 0;
				    in[idx] = out[idx];
				    
				    changed = true;
				    // need to run through again
			    }
		    }
	    }
	}
}







// ##################################################################################################
// ###   SkeletonParallel()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
void SkeletonKernel(unsigned char* hostInput, unsigned char* hostOutput, 
                             int imageWidth, int imageHeight)
{

    cudaProfilerStart();

    // Timing Variables	
    cudaEvent_t start, stop;
    float T0, T1, T2;

    if (TIMER_MODE)
    {	
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
	    cudaEventRecord(start);
	}
	
    // ##################################################################################
	
	if (DEBUG_MODE)
	    std::cout << "  SKELETON::Parallel Start\n\n";
	    
	    
	    
	// Skeleton kernels
    unsigned char Skeleton_Kernel[NUM_SKELETONS][SKELETON_WIDTH][SKELETON_WIDTH] = {{{  0,   0,   0}, {127, 255, 127}, { 255, 255, 255}},
                                                                                    {{127,   0,   0}, {255, 255,   0}, {127, 255, 127}},
                                                                                    {{  0, 127, 255}, {  0, 255, 255}, {  0, 127, 255}},
                                                                                    {{  0,   0, 127}, {  0, 255, 255}, {127, 255, 127}},
                                                                                    {{255, 255, 255}, {127, 255, 127}, {  0,   0,   0}},
                                                                                    {{127, 255, 127}, {  0, 255, 255}, {  0,   0, 127}},
                                                                                    {{255, 127,   0}, {255, 255,   0}, {255, 127,   0}},
                                                                                    {{127, 255, 127}, {255, 255,   0}, {127,   0,   0}}};
                                
                                
	// Copy to constant memory
	cudaMemcpyToSymbol(s_Kernel, &Skeleton_Kernel, NUM_SKELETONS * SKELETON_WIDTH * SKELETON_WIDTH * sizeof(unsigned char));
	

	unsigned char* deviceInput;
	unsigned char* deviceOutput;

	// Allocate device image
	if (DEBUG_MODE)
	    std::cout << "    SKELETON::cudaMalloc Output\n";
	if (cudaMalloc((void**)&deviceOutput, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		return;
	}


	// Allocate device Gray image
	if (DEBUG_MODE)
	    std::cout << "    SKELETON::cudaMalloc Input\n";
	if (cudaMalloc((void**)&deviceInput, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceOutput);
		std::cout << "        Error!\n";
		return;
	}

	// copy  image
	if (DEBUG_MODE)
	    std::cout << "    SKELETON::cudaMemcpy INPUT\n";
	if (cudaMemcpy(deviceInput, hostInput, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceOutput);
		cudaFree(deviceInput);
		std::cout << "        Error!\n";
		return;
	}

	
	
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T0, start, stop);
        
        
	    cudaEventRecord(start);
	}

// ##################################################################################################
	// Do the computation on the GPU
	if (DEBUG_MODE)
	    std::cout << "    SKELETON::kernel\n";
	    
    // Global version
	int blockSize = BLOCK_DIM;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	
	kernel_getSkeleton_GLOBAL <<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, imageWidth, imageHeight);
	
// ##################################################################################################
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T1, start, stop);
           
	    cudaEventRecord(start);
    }

	if (DEBUG_MODE)
	    std::cout << "    SKELETON::cudaMemcpy Output\n";
	if (cudaMemcpy(hostOutput, deviceInput, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceOutput);
		cudaFree(deviceInput);
		return;

	}

	if (DEBUG_MODE)
	    std::cout << "    SKELETON::cudaFree\n";
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
		
	
	
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T2, start, stop);
        
        
	    printf("\n****************************************\n");
	    printf("  \tSKELETON TIMING \n");
	    printf("  \tMemSets  : %f msec\n", T0 );
	    printf("  \tKernel   : %f msec\n", T1 );
	    printf("  \tCleanup  : %f msec\n", T2 );
	
	    printf("  \tTotal    : %f msec\n", (T0+T1+T2) );
	    printf("****************************************\n\n");
	}
	cudaProfilerStop();

}

void SkeletonKernel(cv::Mat& input, cv::Mat& output) {

	int image_size = input.total();
	int width = input.cols;
	int height = input.rows;
	unsigned char* host_input = input.data;
	if (0 == output.total()) {
		output.create(height, width, CV_8UC1);
	}
	SkeletonKernel(host_input, output.data, width, height);
}
