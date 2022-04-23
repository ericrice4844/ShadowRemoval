///////////////////////////////////////////////////////////////
//  
//      Canny_Master_Call.cu
//      This should call each of the functions from 
//      Gaussian_Kernels, Sobel_Kernels, Canny_Kernels
//      
//
///////////////////////////////////////////////////////////////

extern "C" {
#include "Canny_Master_Call.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


#define WEAK -1
#define STRONG -2

__constant__ float c_Kernel_MASTER[GAUSS_WIN_WIDTH];


// ##################################################################################################
// ###   kernel_GaussianBlur_GLOBAL_MASTER()   ### 
// Global version of gaussian blurring kernel
__global__ void 
kernel_GaussianBlur_GLOBAL_MASTER(unsigned char* in, unsigned char* out, int width, int height)
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
// ###   kernel_gaussianSeparablePassX_MASTER()   ### 
// Shared, separated version of gaussian blurring kernel - X Direction

__global__ void 
kernel_gaussianSeparablePassX_MASTER(unsigned char* input, float* out, int width, int height)
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
        for (int j = -GAUSS_WIN_RADI; j <= GAUSS_WIN_RADI; j++)
            sum += c_Kernel_MASTER[GAUSS_WIN_RADI - j] * ds_In[ty][tx + i * BLOCK_DIM + j];

        out[i * BLOCK_DIM] = sum;
    }
}





// ##################################################################################################
// ###   kernel_gaussianSeparablePassY_MASTER()   ### 
// Shared, separated version of gaussian blurring kernel - Y direction

__global__ void 
kernel_gaussianSeparablePassY_MASTER(float *input, unsigned char* out, int width, int height)
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
            sum += c_Kernel_MASTER[GAUSS_WIN_RADI - j] * ds_In[tx][ty + i * COLUMNS_Y + j];

        out[i * COLUMNS_Y * width] = (unsigned char)sum;
    }
}

// ##################################################################################################
// ###   generateGaussiankernels()   ### 
// Helper function to pre-generate blurring kernel
void generateGaussiankernel_MASTER(float sigma, float* Gkernel){
	
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
// ###   kernel_SobelFilter_ALL()   ### 
// Global version of gaussian blurring kernel
// 
// Inputs: const unsigned char* in  : the original image being evaluated
//               unsigned char* out : the image being created using the sobel filter
//                        int width : the width of the image
//                        int height : the height of the image

__global__ void 
kernel_SobelFilter_And_CannyGrad_ALL(unsigned char* in, float* gradStrength, unsigned char* gradDirection, int width, int height) {

    // Indexing setup
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int row = ty + blockIdx.y * blockDim.y; // height
	int col = tx + blockIdx.x * blockDim.x; // width
		
	int idx = width*row + col;
    
    float dx, dy;
    
    
    //get magnitude while loading to reuse space
    float dir, r2d;
    r2d = 180.0 / 3.141592653589793;
    
    
    if( row > 0 && col > 0 && row < height-1 && col < width-1) 
    {
        // Apply the convolution kernels for sobel filter
        //        [-1  0  1]          [-1 -2 -1 ]
        //   Kx = [-2  0  2]     Ky = [ 0  0  0 ]
        //        [-1  0  1]          [ 1  2  1 ]
        dx = (-1* in[(row-1)*width + (col-1)]) + (-2*in[row*width + (col-1)]) + (-1*in[(row+1)*width+(col-1)]) +
             (    in[(row-1)*width + (col+1)]) + ( 2*in[row*width + (col+1)]) + (   in[(row+1)*width+(col+1)]);
             
        dy = (-1* in[(row-1)*width + (col-1)]) + (-2*in[(row-1)*width + col]) + (-1*in[(row-1)*width+(col+1)]) +
             (    in[(row+1)*width + (col-1)]) + ( 2*in[(row+1)*width + col]) + (   in[(row+1)*width+(col+1)]);
             
                
	
	    // Calculate Direction of gradient (to nearest 45 deg, 0-180 deg)
	    dir = 45.0-floor(abs(atan2(dy, dx))*r2d/45.0);
	    dir =  180.0 ? 0.0 : dir;
	
	
        // Set output	 
	    //Calculate magnitude   
	    gradStrength[idx]=sqrt(dx*dx+dy*dy);
	    gradDirection[idx]=(unsigned char)dir;
        
    }
	else if (row < height && col < width)
	{
	    gradStrength[idx] = 0.0;
	    gradDirection[idx] = 0;
    }
    
}


// ##################################################################################################
// ###   kernel_CannyNonMaxSuppression_GLOBAL()   ### 
// Calculates the gradient stength and direction

__global__ void 
kernel_CannyNonMaxSuppression_GLOBAL_MASTER(float* gradStrength, unsigned char* gradDirection, 
                                     int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col;
	
	bool a;
        
    
    // Check if in image boundary with 1 pixel buffer
    if (row < height-1 && row > 0 && col < width-1 && col > 0)
    {  
        
	    // Get direction of comparison based on gradient direction
	    // 4 direction vectors used to represent orientation and cover all 8 neighboring pixels.
        //      gradDirection={	0->left to right
		//                      45 -> top left to bottom right
        //                      90 -> top to down
        //                      135 -> top right to bottom left
	    int f=gradDirection[idx];
	    int dirX = (f==135 ? -1 : (f==90 ? 0 : 1));
	    int dirY = (f==135 || f==45 ? -1 : (f==0 ? 0 : 1));

	    // Is pixel a maximum? 
	    // High chance of bank conflict
	    a = (gradStrength[idx] > max(gradStrength[idx+dirX+dirY*width], gradStrength[idx-dirX-dirY*width]));
        
        
        __syncthreads();  
        //Suppress gradient in all nonmaximum pixels
        gradStrength[idx] *= a;
	
	}
	else if (row < height && col < width)
	{
	    gradStrength[idx] = 0.0;
    }
    
}


// ##################################################################################################
// ###   kernel_CannyThresholding_GLOBAL()   ### 
// Gets the strong  and weak edge pixels

__global__ void 
kernel_CannyThresholding_GLOBAL_MASTER(float* gradStrength, int width, int height,
                                float minThresh, float maxThresh)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col;
    
    // Check if in image boundary with 1 pixel buffer
    if (row < height-1 && row > 0 && col < width-1 && col > 0)
    {  
        float str = gradStrength[idx];
        
        if(str>maxThresh)
            str=STRONG;     // strong edge
        else if(str>minThresh)	
            str=WEAK;     // weak edge
        else if(str>0)				
            str=0;      // not edge
            
        
        gradStrength[idx] = str;
    }
    // Assign edge of image strength of 0.0
	else if (row < height && col < width)
	{
	    gradStrength[idx] = 0.0;
    }  
}


// ##################################################################################################
// ###   kernel_CannyHysteresis_GLOBAL()   ### 
// Calculates the gradient stength and direction

__global__ void 
kernel_CannyHysteresis_GLOBAL_MASTER(float* gradStrength,
                              int width, int height)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col; 
    
    // Check if in image boundary with 1 pixel buffer
    if (row < height-1 && row > 0 && col < width-1 && col > 0)
    {  
        float strength = gradStrength[idx];
        
        if (strength == WEAK)
        {
            
            // unroll this
            #pragma unroll
            for (int ii=-1; ii<2; ii++)
            {
                #pragma unroll
                for (int jj=-1; jj<2; jj++)
                {
                    if (gradStrength[idx+ii+jj*width]==STRONG)
                        strength=STRONG;
                }
            }
            
        }       
        
        __syncthreads();
        gradStrength[idx] = strength;
    }
    // Assign edge of image strength of 0.0
	else if (row < height && col < width)
	{
	    gradStrength[idx] = 0.0;
    }

    __syncthreads();    
}



// ##################################################################################################
// ###   kernel_CannyHysteresis_SHARED()   ### 
// 
#define HYST_SHARED_BLOCK_DIM 8
__global__ void 
kernel_CannyHysteresis_SHARED_MASTER(float* gradStrength,
                              int width, int height)
{

    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    
    // Shared memory is main block plus halo in X&Y direction
    __shared__ float ds_GS[HYST_SHARED_BLOCK_DIM+2*HALO_STEPS][HYST_SHARED_BLOCK_DIM+2*HALO_STEPS];
    
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col; 
	
	
	// Load main data
	#pragma unroll
	for (int ii = 0; ii < HYST_SHARED_BLOCK_DIM; ii++)
    {
    	#pragma unroll
        for (int jj = 0; jj < HYST_SHARED_BLOCK_DIM; jj++)
        {
            if (row+ii < height && col+jj < width )
                ds_GS[ty][tx] = gradStrength[width*(row+ii) + col+jj];
            else
                ds_GS[ty][tx] = 0.0;
        }
        
    }
    
    // load left, right, top, bottom halos
    // top (-HALO_STEPS)
    #pragma unroll
    for (int ii=0; ii<HYST_SHARED_BLOCK_DIM; ii++)
    {
        if (row-HALO_STEPS > 0  && col+ii < width)
            ds_GS[0][tx] = gradStrength[width*(row-HALO_STEPS) + col+ii];
        else
            ds_GS[0][tx] = 0.0;
    }
    
    
    // load left, right, top, bottom halos
    // bottom (+BLOCK_DIM+HALO_STEPS)
    #pragma unroll
    for (int ii=0; ii<HYST_SHARED_BLOCK_DIM; ii++)
    {
        if (row+HYST_SHARED_BLOCK_DIM+HALO_STEPS < height && col+ii < width)
            ds_GS[HYST_SHARED_BLOCK_DIM+HALO_STEPS][tx] = gradStrength[width*(row+HYST_SHARED_BLOCK_DIM+HALO_STEPS) + col+ii];
        else
            ds_GS[HYST_SHARED_BLOCK_DIM+HALO_STEPS][tx] = 0.0;
    }
        
        
    // left (-HALO_STEPS_
    #pragma unroll
    for (int ii=0; ii<HYST_SHARED_BLOCK_DIM; ii++)
    {
        if (col-HALO_STEPS > 0  && row+ii < height)
            ds_GS[ty][0] = gradStrength[width*(row+ii) + col-HALO_STEPS];
        else
            ds_GS[ty][0] = 0.0;
    }
    
    
    // right
    #pragma unroll
    for (int ii=0; ii<HYST_SHARED_BLOCK_DIM; ii++)
    {
        if (col+HYST_SHARED_BLOCK_DIM+HALO_STEPS < width && row+ii < height)
            ds_GS[ty][HYST_SHARED_BLOCK_DIM+HALO_STEPS] = gradStrength[width*(row+ii) + col+HYST_SHARED_BLOCK_DIM+HALO_STEPS];
        else
            ds_GS[ty][HYST_SHARED_BLOCK_DIM+HALO_STEPS] = 0.0;
    }
    
    
    cg::sync(cta); // sync the thread block threads
	
    // Check if in image boundary with 1 pixel buffer
    //if (row < height-1 && row > 0 && col < width-1 && col > 0)
    if (row < height && col < width)
    {  
        float strength = ds_GS[tx][ty];
        
        if (strength == WEAK)
        {
            
            // unroll this
            #pragma unroll
            for (int ii=-1; ii<2; ii++)
            {
                #pragma unroll
                for (int jj=-1; jj<2; jj++)
                {
                    if (ds_GS[tx+ii][ty+jj]==STRONG)
                        strength=STRONG;
                }
            }     
        }   
             
        
        
        cg::sync(cta); // sync the thread block threads
        gradStrength[idx] = strength;
    }
    // Assign edge of image strength of 0.0
	else if (row < height && col < width)
	{
	    gradStrength[idx] = 0.0;
    }

    __syncthreads();    
    
     
}



// ##################################################################################################
// ###   kernel_CannyBlockConverter()   ### 
// Converts output image
__global__ void 
kernel_CannyBlockConverter_MASTER(float* gradStrength,
						unsigned char *outputImage,
					    int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
	int idx = width*row + col;
	
	
	////Load global data into shared
	//Load center
	__syncthreads();
    if (row < height && col < width)
    {
    	outputImage[idx]=(unsigned char)(255*(gradStrength[idx]==STRONG));
	}

}


// ##################################################################################################
// ###   kernel_Float2Char()   ### 
// Converts output image
__global__ void 
kernel_Float2Char(float* gradStrength,
						unsigned char *outputImage,
					    int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
	int idx = width*row + col;
	
	
	////Load global data into shared
	//Load center
	__syncthreads();
    if (row < height && col < width)
    {
    	outputImage[idx]=(unsigned char)gradStrength[idx];
	}

}



// ##################################################################################################
// ##################################################################################################
// ##################################################################################################





// ##################################################################################################
// ###   GaussianBlur()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
extern "C" void CannyMasterCall(unsigned char hostGrayImage[IM_ROWS][IM_COLS], unsigned char hostCannyImage[IM_ROWS][IM_COLS], 
                             int imageWidth, int imageHeight)
{



    // Timing Variables	
    cudaEvent_t start, stop;
    float T0, T1, T2, T3, T4, T5, T6, T7;

    if (TIMER_MODE)
    {	
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
	    cudaEventRecord(start);
	}

    // ##################################################################################
    // FOR OPTIMZIED gaussian
    
    float gKernel[GAUSS_WIN_WIDTH];
    
    // Create Gaussian kernels in x and y directions
	generateGaussiankernel_MASTER(0.8, gKernel);	
	
	// Copy to constant memory
	cudaMemcpyToSymbol(c_Kernel_MASTER, gKernel, GAUSS_WIN_WIDTH * sizeof(float));
	
    // ##################################################################################
    
	if (DEBUG_MODE)
	    std::cout << "  CANNY_MASTER::Parallel Start\n\n";

	unsigned char* deviceGrayCharArr;
	unsigned char* deviceCannyCharArr;
	
	float* deviceFloatArr;
	
	
	float minThreshold = 72.0;
	float maxThreshold = 94.0;
	int thresholdIdterations = 2;

	// Allocate device image - character array 1
	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::cudaMalloc BLUR\n";
	if (cudaMalloc((void**)&deviceCannyCharArr, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!";
		return;
	}


	// Allocate device image - character array 2
	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::cudaMalloc Gray\n";
	if (cudaMalloc((void**)&deviceGrayCharArr, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceCannyCharArr);
		std::cout << "        Error!";
		return;
	}
	
	
	// Allocate device image - float array
	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::cudaMalloc Gray\n";
	if (cudaMalloc((void**)&deviceFloatArr, imageWidth * imageHeight * sizeof(float)) != cudaSuccess)
	{
		cudaFree(deviceCannyCharArr);
		std::cout << "        Error!";
		return;
	}
	
	

	// copy initial grayscale image
	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::cudaMemcpy Gray\n";
	if (cudaMemcpy(deviceGrayCharArr, hostGrayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceCannyCharArr);
		cudaFree(deviceGrayCharArr);
		cudaFree(deviceFloatArr);
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
	
	// General grid and block setup
	int blockSize = BLOCK_DIM;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	
	
	// ###########################################
	// Gaussian blur algorithm
	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::kernel_GaussianBlur\n";
	
	//kernel_GaussianBlur_GLOBAL_MASTER<<<DimGrid,DimBlock>>>(deviceGrayCharArr, deviceCannyCharArr, imageWidth, imageHeight);
	
	
    // Separable, shmem version (fast)
    dim3 blocksRow(imageWidth / (RESULT_STEPS * BLOCK_DIM), imageHeight / ROWS_Y);
    dim3 threadsRow(BLOCK_DIM, ROWS_Y);

	kernel_gaussianSeparablePassX_MASTER <<<blocksRow, threadsRow>>>(deviceGrayCharArr, deviceFloatArr, imageWidth, imageHeight);
							
								
	dim3 blocksCol(imageWidth / BLOCK_DIM, imageHeight / (RESULT_STEPS * COLUMNS_Y));
    dim3 threadsCol(BLOCK_DIM, COLUMNS_Y);	
    	
	kernel_gaussianSeparablePassY_MASTER <<<blocksCol, threadsCol>>>(deviceFloatArr, deviceCannyCharArr, imageWidth, imageHeight);
	
	
	
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T1, start, stop);
           
	    cudaEventRecord(start);
    }

	
	
	// ###########################################
    // Sobel and canny calcs 
	kernel_SobelFilter_And_CannyGrad_ALL <<<DimGrid,DimBlock>>>(deviceCannyCharArr, deviceFloatArr, deviceGrayCharArr, imageWidth, imageHeight);
	
	
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T2, start, stop);
           
	    cudaEventRecord(start);
    }
    
	// ###########################################
    // Canny non-max suppression
	kernel_CannyNonMaxSuppression_GLOBAL_MASTER <<<DimGrid,DimBlock>>>(deviceFloatArr, deviceGrayCharArr, imageWidth, imageHeight);
	
	
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T3, start, stop);
           
	    cudaEventRecord(start);
    }
	// ###########################################
    // Canny thresholding
	for (int idx = 0; idx < thresholdIdterations; idx++)
    	kernel_CannyThresholding_GLOBAL_MASTER <<<DimGrid,DimBlock>>>(deviceFloatArr, imageWidth, imageHeight, minThreshold, maxThreshold);
    
    
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T4, start, stop);
           
	    cudaEventRecord(start);
    }
	// ###########################################
    // Canny hysteresis
	kernel_CannyHysteresis_GLOBAL_MASTER <<<DimGrid,DimBlock>>>(deviceFloatArr, imageWidth, imageHeight);
	
	
	//int blockSizeHyst = HYST_SHARED_BLOCK_DIM;
	//dim3 DimGridHyst((imageWidth - 1) / blockSizeHyst + 1, (imageHeight - 1) / blockSizeHyst + 1, 1);
	//dim3 DimBlockHyst(blockSizeHyst, blockSizeHyst, 1);
	//kernel_CannyHysteresis_SHARED_MASTER <<<DimGridHyst,DimBlockHyst>>>(deviceFloatArr, imageWidth, imageHeight);
	
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	{
	    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; // add
        cudaProfilerStop();
        return;
    }
    
    
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T5, start, stop);
           
	    cudaEventRecord(start);
    }
    
	// ###########################################
    // convert back to uchar from float
	kernel_CannyBlockConverter_MASTER <<<DimGrid,DimBlock>>>(deviceFloatArr, deviceCannyCharArr, imageWidth, imageHeight);
	
	
	
    
	
	// TESTING ONLY 
	//kernel_Float2Char<<<DimGrid,DimBlock>>>(deviceFloatArr, deviceCannyCharArr, imageWidth, imageHeight);
	
	
    if (TIMER_MODE)
    {	
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T6, start, stop);
           
	    cudaEventRecord(start);
    }

// ##################################################################################################
	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::cudaMemcpy Gray\n";
	if (cudaMemcpy(hostCannyImage, deviceCannyCharArr, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!";
		cudaFree(deviceCannyCharArr);
		cudaFree(deviceGrayCharArr);
		cudaFree(deviceFloatArr);
		return;

	}

	if (DEBUG_MODE)
	    std::cout << "    CANNY_MASTER::cudaFree\n";
	cudaFree(deviceGrayCharArr);
	cudaFree(deviceCannyCharArr);
	cudaFree(deviceFloatArr);
		
	
	
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T7, start, stop);
        
        
	    printf("\n****************************************\n");
	    printf("  \tCANNY_MASTER TIMING \n");
	    printf("  \tMemSets      : %f msec\n", T0 );
	    printf("  \tgaussian     : %f msec\n", T1 );
	    printf("  \tsobel+grad   : %f msec\n", T2 );
	    printf("  \tnonMax       : %f msec\n", T3 );
	    printf("  \tthreshold    : %f msec\n", T4 );
	    printf("  \thystersis    : %f msec\n", T5 );
	    printf("  \tconverter    : %f msec\n", T6 );
	    printf("  \tCleanup      : %f msec\n", T7 );
	
	    printf("  \tTotal        : %f msec\n", (T0+T1+T2+T3+T4+T5+T6+T7) );
	    printf("****************************************\n\n");
	}

}









