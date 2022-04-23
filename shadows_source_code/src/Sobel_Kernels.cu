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

#define SOBEL_RADII 1
__constant__ float c_KernelSobel1[3];
__constant__ float c_KernelSobel2[3];


// ##################################################################################################
// ###   kernel_SobelFilter_GLOBAL()   ### 
// Global version of gaussian blurring kernel
// 
// Inputs: const unsigned char* in  : the original image being evaluated
//               unsigned char* out : the image being created using the sobel filter
//                        int width : the width of the image
//                        int height : the height of the image

__global__ void 
kernel_SobelFilter_GLOBAL(unsigned char* in, unsigned char* mag_out, unsigned char* x_out, unsigned char* y_out, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    
		
    // Apply the convolution kernels
    //        [-1  0  1]          [-1 -2 -1 ]
    //   Kx = [-2  0  2]     Ky = [ 0  0  0 ]
    //        [-1  0  1]          [ 1  2  1 ]
    int Kx[3][3] = {{-1,  0,  1}, {-2,  0,  2}, {-1,  0,  1}};
    int Ky[3][3] = {{-1, -2, -1}, { 0,  0,  0}, {-1, -2, -1}};
    
    if( x > 0 && y > 0 && x < width-1 && y < height-1) 
    {
        for (int sobY = -SOBEL_DIM; sobY < SOBEL_DIM+1; sobY++)
		{
			for  (int sobX = -SOBEL_DIM; sobX < SOBEL_DIM+1; sobX++)
			{
				// Check for corner/edge cases
				if ((y + sobY) > -1 && (y + sobY) < height && (x + sobX) > -1 && (x + sobX) < width) 
				{
					dx += Kx[sobY][sobX] * in[(y + sobY) * width + (x + sobX)];
					dy += Ky[sobY][sobX] * in[(y + sobY) * width + (x + sobX)];
				}
			}
		}
		__syncthreads();
		
        // Gradient Magnitude
        mag_out[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
        
        // Get gradient in X & Y
        x_out[y*width + x] = (unsigned char)dx;
        y_out[y*width + x] = (unsigned char)dy;
        
    }
}

// ##################################################################################################
// ###   kernel_SobelFilter_GATHER()   ### 
// Gather version of gaussian blurring kernel
// 
// Inputs: const unsigned char* in  : the original image being evaluated
//               unsigned char* out : the image being created using the sobel filter
//                        int width : the width of the image
//                        int height : the height of the image

__global__ void 
kernel_SobelFilter_GATHER(unsigned char* in, unsigned char* mag_out, unsigned char* x_out, unsigned char* y_out, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) 
    {
        // Apply the convolution kernels
        //        [-1  0  1]          [-1 -2 -1 ]
        //   Kx = [-2  0  2]     Ky = [ 0  0  0 ]
        //        [-1  0  1]          [ 1  2  1 ]
        dx = (-1* in[(y-1)*width + (x-1)]) + (-2*in[y*width+(x-1)]) + (-1*in[(y+1)*width+(x-1)]) +
             (    in[(y-1)*width + (x+1)]) + ( 2*in[y*width+(x+1)]) + (   in[(y+1)*width+(x+1)]);
             
        dy = (-1* in[(y-1)*width + (x-1)]) + (-2*in[(y-1)*width+x]) + (-1*in[(y-1)*width+(x+1)]) +
             (    in[(y+1)*width + (x-1)]) + ( 2*in[(y+1)*width+x]) + (   in[(y+1)*width+(x+1)]);
             
        // Gradient Magnitude
        mag_out[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
        
        // Get gradient in X & Y
        x_out[y*width + x] = (unsigned char)dx;
        y_out[y*width + x] = (unsigned char)dy;
        
    }
}



// ##################################################################################################
// ###   kernel_sobelSeparablePassX()   ### 
// Shared, separated version of gaussian blurring kernel - X Direction

__global__ void 
kernel_sobelSeparablePassX(unsigned char* input, unsigned char* out, int width, int height)
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
        for (int j = -SOBEL_RADII; j <= SOBEL_RADII; j++)
        {   
            #pragma unroll
            for (int j = -SOBEL_RADII; j <= SOBEL_RADII; j++)
                sum += c_KernelSobel1[SOBEL_RADII - j] * c_KernelSobel2[SOBEL_RADII - j] * ds_In[ty][tx + i * BLOCK_DIM + j];
        }

        out[i * BLOCK_DIM] = (unsigned char)sum;
    }
}





// ##################################################################################################
// ###   kernel_sobelSeparablePassY()   ### 
// Shared, separated version of gaussian blurring kernel - Y direction

__global__ void 
kernel_sobelSeparablePassY(unsigned char *input, unsigned char* out, int width, int height)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    
    // Shared memory is main block plus halo in Y direction (rows)
    __shared__ unsigned char ds_In[BLOCK_DIM][(RESULT_STEPS + 2 * HALO_STEPS) * COLUMNS_Y + 1];

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
        for (int j = -SOBEL_RADII; j <= SOBEL_RADII; j++)
        {   
            #pragma unroll
            for (int j = -SOBEL_RADII; j <= SOBEL_RADII; j++)
                sum += c_KernelSobel2[SOBEL_RADII - j] * c_KernelSobel1[SOBEL_RADII - j] * ds_In[tx][ty + i * COLUMNS_Y + j];
        }

        out[i * COLUMNS_Y * width] = (unsigned char)sum;
    }
}


// ##################################################################################################
// ###   kernel_SobelFilter_MagCalc()   ### 
// Calculate magnitude
__global__ void 
kernel_SobelFilter_MagCalc(unsigned char* inX, unsigned char* inY, unsigned char* mag_out, int width, int height) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = width*y + x;
	
	float dx = inX[idx];
	float dy = inY[idx];
    if( x > 0 && y > 0 && x < width-1 && y < height-1) 
    {
             
        // Gradient Magnitude
        mag_out[y*width + x] = (unsigned char)sqrt( (dx*dx) + (dy*dy) );
        
    }
}





// ##################################################################################################
// ###   SobelFilter()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
extern "C" void SobelFilter(unsigned char hostImage[IM_ROWS][IM_COLS], unsigned char hostMagImage[IM_ROWS][IM_COLS], 
                            unsigned char hostDirXImage[IM_ROWS][IM_COLS], unsigned char hostDirYImage[IM_ROWS][IM_COLS],
                            int imageWidth, int imageHeight)
{

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
    
    float k1[3] = {-1, 0, 1};
    float k2[3] = { 1, 2, 1};
	
	// Copy to constant memory
	//cudaMemcpyToSymbol(c_KernelSobel1, k1, 3 * sizeof(float));
	//cudaMemcpyToSymbol(c_KernelSobel2, k2, 3 * sizeof(float));
	
    // ##################################################################################
    
    
	
	if (DEBUG_MODE)
	    std::cout << "  SOBEL::Parallel Start\n\n";

	unsigned char* deviceImage;
	unsigned char* deviceMagImage;
	unsigned char* deviceGradDirXImage;
	unsigned char* deviceGradDirYImage;
	
	
	

	// Allocate device magnitude image
	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMalloc EDGE\n";
	if (cudaMalloc((void**)&deviceMagImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!";
		return;
	}


	// Allocate device gradient direction image
	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMalloc EDGE\n";
	if (cudaMalloc((void**)&deviceGradDirXImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceMagImage);
		return;
	}


	// Allocate device gradient direction image
	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMalloc EDGE\n";
	if (cudaMalloc((void**)&deviceGradDirYImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceMagImage);
		cudaFree(deviceGradDirXImage);
		return;
	}

	// Allocate device image
	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMalloc nominal\n";
	if (cudaMalloc((void**)&deviceImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceMagImage);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		std::cout << "        Error!";
		return;
	}

	// copy image
	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMemcpy Gray\n";
	if (cudaMemcpy(deviceImage, hostImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceMagImage);
		cudaFree(deviceImage);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		std::cout << "        Error!";
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
	int blockSize = 32;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	if (DEBUG_MODE)
	    std::cout << "    SOBEL::kernel_sobelFilter\n";
	kernel_SobelFilter_GLOBAL <<<DimGrid,DimBlock>>>(deviceImage, deviceMagImage, deviceGradDirXImage, deviceGradDirYImage, imageWidth, imageHeight);

    
	//kernel_SobelFilter_GATHER <<<DimGrid,DimBlock>>>(deviceImage, deviceMagImage, deviceGradDirXImage, deviceGradDirYImage, imageWidth, imageHeight);
    
    
    /*
    // Separable, shmem version (fast)
    dim3 blocksRow(imageWidth / (RESULT_STEPS * BLOCK_DIM), imageHeight / ROWS_Y);
    dim3 threadsRow(BLOCK_DIM, ROWS_Y);

	kernel_sobelSeparablePassX<<<blocksRow, threadsRow>>>(deviceImage, deviceGradDirXImage, imageWidth, imageHeight);
							
							
								
	dim3 blocksCol(imageWidth / BLOCK_DIM, imageHeight / (RESULT_STEPS * COLUMNS_Y));
    dim3 threadsCol(BLOCK_DIM, COLUMNS_Y);	
    	
	kernel_sobelSeparablePassY<<<blocksCol, threadsCol>>>(deviceImage, deviceGradDirYImage, imageWidth, imageHeight);
	
	
	kernel_SobelFilter_MagCalc<<<DimGrid,DimBlock>>>(deviceGradDirXImage, deviceGradDirYImage, deviceMagImage, imageWidth, imageHeight);
	*/
	
// ##################################################################################################
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T1, start, stop);
        
        
	    cudaEventRecord(start);
    }

	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMemcpy Magnitude\n";
	if (cudaMemcpy(hostMagImage, deviceMagImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceMagImage);
		cudaFree(deviceImage);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		return;
	}

	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMemcpy Direction\n";
	if (cudaMemcpy(hostDirXImage, deviceGradDirXImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceMagImage);
		cudaFree(deviceImage);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		return;
	}

	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaMemcpy Direction\n";
	if (cudaMemcpy(hostDirYImage, deviceGradDirYImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceMagImage);
		cudaFree(deviceImage);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		return;
	}

	if (DEBUG_MODE)
	    std::cout << "    SOBEL::cudaFree\n";
	cudaFree(deviceImage);
	cudaFree(deviceMagImage);
	cudaFree(deviceGradDirXImage);
	cudaFree(deviceGradDirYImage);
		
		
	
    if (TIMER_MODE)
    {
	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T2, start, stop);
        
        
	    printf("****************************************\n");
	    printf("  \tSOBEL TIMING \n");
	    printf("  \tMemSets  : %f msec\n", T0 );
	    printf("  \tKernel   : %f msec\n", T1 );
	    printf("  \tCleanup  : %f msec\n", T2 );
	
	    printf("  \tTotal    : %f msec\n", (T0+T1+T2) );
	
	    printf("****************************************\n\n");
    }

}
