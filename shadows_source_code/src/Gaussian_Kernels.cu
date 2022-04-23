///////////////////////////////////////////////////////////////
//  
//      Gaussian_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////

extern "C" {
#include "Gaussian_Kernels.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__constant__ float c_Kernel[GAUSS_WIN_WIDTH];





// ##################################################################################################
// ###   kernel_GaussianBlur_GLOBAL()   ### 
// Global version of gaussian blurring kernel
__global__ void 
kernel_GaussianBlur_GLOBAL_BASIC(unsigned char* in, unsigned char* out, int width, int height)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	// write the boundary condition
	if ((row < height) && (col < width))
	{
		int pixelValue = 0;
		int pixelCount = 0;
		// write the nested for loop and its body (only to find current pixel's blurred value)
		for (int blurRow = -GAUSS_WIN_WIDTH; blurRow < GAUSS_WIN_WIDTH+1; blurRow++)
		{
			for  (int blurCol = -GAUSS_WIN_WIDTH; blurCol < GAUSS_WIN_WIDTH+1; blurCol++)
			{
				// Check for corner/edge cases
				if ((row + blurRow) > -1 && (row + blurRow) < height && (col + blurCol) > -1 && (col + blurCol) < width) 
				{
					pixelValue += (int)in[(row + blurRow) * width + (col + blurCol)];
					pixelCount++;
				}
			}
		}
		__syncthreads();
		// write our new pixel output value
		out[row*width + col] = pixelValue/pixelCount;
	}
}

// ##################################################################################################
// ###   kernel_gaussianSeparablePassX()   ### 
// Shared, separated version of gaussian blurring kernel - X Direction

__global__ void 
kernel_gaussianSeparablePassX(unsigned char* input, float* out, int width, int height)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    
    // Shared memory is main block plus halo in X direction (columns)
    __shared__ unsigned char ds_In[ROWS_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCK_DIM];
    
    // Set up indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    //Offset to the left halo edge
    int col = (blockIdx.x * RESULT_STEPS - HALO_STEPS) * BLOCK_DIM + tx;
    int row = blockIdx.y * ROWS_Y + ty;

    // update starting point for input/output data 
    input += row*width + col;
    out += row*width + col;

    //Load main data to shared memory
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
        ds_In[ty][tx + i * BLOCK_DIM] = input[i * BLOCK_DIM];


    //Load left halo to shared memory
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++)
        ds_In[ty][tx + i * BLOCK_DIM] = (col >= -i * BLOCK_DIM) ? input[i * BLOCK_DIM] : 0;


    //Load right halo to shared memory
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++)
        ds_In[ty][tx + i * BLOCK_DIM] = (width - col > i * BLOCK_DIM) ? input[i * BLOCK_DIM] : 0;


    //Compute and store results
    cg::sync(cta); // sync the thread block threads
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
    {
        float sum = 0;

        #pragma unroll
        for (int j = -GAUSS_WIN_RADI; j <= GAUSS_WIN_RADI; j++)
            sum += c_Kernel[GAUSS_WIN_RADI - j] * ds_In[ty][tx + i * BLOCK_DIM + j];

        out[i * BLOCK_DIM] = sum;
    }
}





// ##################################################################################################
// ###   kernel_gaussianSeparablePassY()   ### 
// Shared, separated version of gaussian blurring kernel - Y direction

__global__ void 
kernel_gaussianSeparablePassY(float *input, unsigned char* out, int width, int height)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    
    // Shared memory is main block plus halo in Y direction (rows)
    __shared__ float ds_In[BLOCK_DIM][(RESULT_STEPS + 2 * HALO_STEPS) * COLUMNS_Y + 1];

    // Set up indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    //Offset to the upper halo edge
    int col = blockIdx.x * BLOCK_DIM + tx;
    int row = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * COLUMNS_Y + ty;
    
    input += row * width + col;
    out += row * width + col;


    // Load main data to shared memory
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
        ds_In[tx][ty + i * COLUMNS_Y] = input[i * COLUMNS_Y * width];


    // Load Upper halo to shared memory
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++)
        ds_In[tx][ty + i * COLUMNS_Y] = (row >= -i * COLUMNS_Y) ? input[i * COLUMNS_Y * width] : 0;


    //Load Lower halo to shared memory
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++)
        ds_In[tx][ty + i * COLUMNS_Y]= (height - row > i * COLUMNS_Y) ? input[i * COLUMNS_Y * width] : 0;
        

    //Compute and store results
    cg::sync(cta); // sync thread block
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
    {
        float sum = 0;
        
        #pragma unroll
        for (int j = -GAUSS_WIN_RADI; j <= GAUSS_WIN_RADI; j++)
            sum += c_Kernel[GAUSS_WIN_RADI - j] * ds_In[tx][ty + i * COLUMNS_Y + j];

        out[i * COLUMNS_Y * width] = (unsigned char)sum;
    }
}



// ##################################################################################################
// ###   generateGaussiankernels()   ### 
// Helper function to pre-generate blurring kernel
void generateGaussiankernel(float sigma, float* Gkernel){
	
	float sumY=0;
	
	for(int a=-GAUSS_WIN_RADI;a<=GAUSS_WIN_RADI;++a)
	{
		Gkernel[a+GAUSS_WIN_RADI]=exp(-a*a/(2*sigma*sigma));
		sumY+=Gkernel[a+GAUSS_WIN_RADI];
	}
	
	for(int a=0;a<GAUSS_WIN_WIDTH;++a)
	{
		Gkernel[a]/=sumY;
	}
}



// ##################################################################################################
// ###   GaussianBlur()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
extern "C" void GaussianBlur(unsigned char hostGrayImage[IM_ROWS][IM_COLS], unsigned char hostBlurImage[IM_ROWS][IM_COLS], 
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
    // FOR OPTIMZIED VERSION
    
    float gKernel[GAUSS_WIN_WIDTH];
    
    // Create Gaussian kernels in x and y directions
	generateGaussiankernel(0.8, gKernel);	
	
	// Copy to constant memory
	cudaMemcpyToSymbol(c_Kernel, gKernel, GAUSS_WIN_WIDTH * sizeof(float));
	
    // ##################################################################################
	
	if (DEBUG_MODE)
	    std::cout << "  GAUSS::Parallel Start\n\n";

	unsigned char* deviceGrayImage;
	float* deviceBlurImage;

	// Allocate device image
	if (DEBUG_MODE)
	    std::cout << "    GAUSS::cudaMalloc BLUR\n";
	if (cudaMalloc((void**)&deviceBlurImage, imageWidth * imageHeight * sizeof(float)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		return;
	}


	// Allocate device Gray image
	if (DEBUG_MODE)
	    std::cout << "    GAUSS::cudaMalloc Gray\n";
	if (cudaMalloc((void**)&deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceBlurImage);
		std::cout << "        Error!\n";
		return;
	}

	// copy RGB image
	if (DEBUG_MODE)
	    std::cout << "    GAUSS::cudaMemcpy Gray\n";
	if (cudaMemcpy(deviceGrayImage, hostGrayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceBlurImage);
		cudaFree(deviceGrayImage);
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
	    std::cout << "    GAUSS::kernel_GaussianBlur\n";
	    
    // Global version (slow)
	int blockSize = BLOCK_DIM;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	//kernel_GaussianBlur_GLOBAL_BASIC <<<DimGrid,DimBlock>>>(deviceGrayImage, deviceBlurImage, imageWidth, imageHeight);
								
    
    
    // Separable, shmem version (fast)
    dim3 blocksRow(imageWidth / (RESULT_STEPS * BLOCK_DIM), imageHeight / ROWS_Y);
    dim3 threadsRow(BLOCK_DIM, ROWS_Y);

	kernel_gaussianSeparablePassX<<<blocksRow, threadsRow>>>(deviceGrayImage, deviceBlurImage, imageWidth, imageHeight);
							
							
								
	dim3 blocksCol(imageWidth / BLOCK_DIM, imageHeight / (RESULT_STEPS * COLUMNS_Y));
    dim3 threadsCol(BLOCK_DIM, COLUMNS_Y);	
    	
	kernel_gaussianSeparablePassY<<<blocksCol, threadsCol>>>(deviceBlurImage, deviceGrayImage, imageWidth, imageHeight);
	
// ##################################################################################################
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T1, start, stop);
           
	    cudaEventRecord(start);
    }

	if (DEBUG_MODE)
	    std::cout << "    GAUSS::cudaMemcpy Gray\n";
	if (cudaMemcpy(hostBlurImage, deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceBlurImage);
		cudaFree(deviceGrayImage);
		return;

	}

	if (DEBUG_MODE)
	    std::cout << "    GAUSS::cudaFree\n";
	cudaFree(deviceGrayImage);
	cudaFree(deviceBlurImage);
		
	
	
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T2, start, stop);
        
        
	    printf("\n****************************************\n");
	    printf("  \tGAUSS TIMING \n");
	    printf("  \tMemSets  : %f msec\n", T0 );
	    printf("  \tKernel   : %f msec\n", T1 );
	    printf("  \tCleanup  : %f msec\n", T2 );
	
	    printf("  \tTotal    : %f msec\n", (T0+T1+T2) );
	    printf("****************************************\n\n");
	}
	cudaProfilerStop();

}
