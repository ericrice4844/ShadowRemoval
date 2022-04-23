///////////////////////////////////////////////////////////////
//  
//      Canny_Kernels.cu
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////

extern "C" {
#include "Canny_Kernels.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define WEAK -1
#define STRONG -2
#define CANNY_SMEM_SIZE 34

// ##################################################################################################
// ###   kernel_CannyGradientCalcs_GLOBAL()   ### 
// Calculates the gradient stength and direction

__global__ void 
kernel_CannyGradientCalcs_GLOBAL(unsigned char* gradX, unsigned char* gradY,
                       int width, int height,
                       float* gradStrength, unsigned char* gradDirection)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col;
    
    // Check if in image boundary
    if (col < width && row < height)
    {
    
	    //get magnitude while loading to reuse space
	    float strengthX, strengthY, strengthMag, dir, r2d;
	    r2d = 180.0 / 3.141592653589793;

	    //Get gradients
	    strengthX=gradX[idx];
	    strengthY=gradY[idx];

	    //Calculate magnitude
	    strengthMag=sqrt(strengthX*strengthX+strengthY*strengthY);
	
	    // Calculate Direction of gradient (to nearest 45 deg, 0-180 deg)
	    dir = 45.0-floor(abs(atan2((float)strengthY, (float)strengthX))*r2d/45.0);
	    dir =  180.0 ? 0 : dir;
	
	
        // Set output
	    __syncthreads();
	    gradStrength[idx]=strengthMag;
	    gradDirection[idx]=(unsigned char)dir;
	}
}

// ##################################################################################################
// ###   kernel_CannyNonMaxSuppression_GLOBAL()   ### 
// Calculates the gradient stength and direction

__global__ void 
kernel_CannyNonMaxSuppression_GLOBAL(float* gradStrengthOut, float* gradStrength, unsigned char* gradDirection, 
                                     int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col;
        
    __syncthreads();  
    
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
	    bool a = (gradStrength[idx] > max(gradStrength[idx+dirX+dirY*width], gradStrength[idx-dirX-dirY*width]));
	
	    //Suppress gradient in all nonmaximum pixels
	    __syncthreads();
	    gradStrengthOut[idx] = gradStrength[idx] * a;
	}
	else if (row < height && col < width)
	{
	    gradStrengthOut[idx] = 0.0;
    }
    
    __syncthreads();  

}


// ##################################################################################################
// ###   kernel_CannyThresholding_GLOBAL()   ### 
// Gets the strong  and weak edge pixels

__global__ void 
kernel_CannyThresholding_GLOBAL(float* gradStrength, float* gradStrengthOut,
                                int width, int height,
                                float minThresh, float maxThresh)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col;
    
    __syncthreads();    
    
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
        
        
        //if (str==STRONG)
        //    printf("STRONG %d\n", idx);
            
        gradStrengthOut[idx] = str;
    }
    // Assign edge of image strength of 0.0
	else if (row < height && col < width)
	{
	    gradStrengthOut[idx] = 0.0;
    }

    __syncthreads();    
}


// ##################################################################################################
// ###   kernel_CannyHysteresis_GLOBAL()   ### 
// Calculates the gradient stength and direction

__global__ void 
kernel_CannyHysteresis_GLOBAL(float* gradStrength, float* gradStrengthOut,
                              int width, int height)
{

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockDim.x*blockIdx.x + tx;
	int row = blockDim.y*blockIdx.y + ty;
    
	int idx = width*row + col;
    
    __syncthreads();    
    
    // Check if in image boundary with 1 pixel buffer
    if (row < height-1 && row > 0 && col < width-1 && col > 0)
    {  
        float strength = gradStrength[idx];
        
        if (strength == WEAK)
        {
            int ii = -1; 
            int jj = -1; 
            while (ii<2 && strength > STRONG)
            {
                while (jj<2 && strength > STRONG)
                {
                    if (gradStrength[idx+ii+jj*width]==STRONG)
                    {
                        strength = STRONG;
                    }
                    jj++;
                }
                ii++;
            }
        }        
        gradStrengthOut[idx] = strength;
    }
    // Assign edge of image strength of 0.0
	else if (row < height && col < width)
	{
	    gradStrengthOut[idx] = 0.0;
    }

    __syncthreads();    
}





// ##################################################################################################
// ###   kernel_CannyBlockConverter()   ### 
// Converts output image
__global__ void 
kernel_CannyBlockConverter(float* gradStrength,
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
// ###   CannyDet()    ###
// This function sets up the device memory, calls the kernel, and retrieves the output from the device
// currently hardcoded to a specific image size 
extern "C" void CannyDet(unsigned char hostDirXImage[IM_ROWS][IM_COLS], unsigned char hostDirYImage[IM_ROWS][IM_COLS],
                         unsigned char hostOutputImage[IM_ROWS][IM_COLS],
                         int imageWidth, int imageHeight)
{
    // Timing Variables	
    cudaEvent_t start, stop;
    float T0, T1, T2, T3, T4, T5, T6, T7;
    
	    
    // Do Timing
    if (TIMER_MODE)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
	    cudaEventRecord(start);
    }
	
	if (DEBUG_MODE)
	    std::cout << "  CANNY::Parallel Start\n\n";
	    
	float minThreshold = 25.0;
	float maxThreshold = 35.0;
	int thresholdIdterations = 4;

	unsigned char* deviceImage;
	unsigned char* deviceGradDirXImage;
	unsigned char* deviceGradDirYImage;
    unsigned char* deviceGradientDirection;
	float* deviceGradientStrength;
	float* deviceGradientStrengthOut;
	
	
	
	if (DEBUG_MODE)
	    printf("Image - Height = %d,  Width = %d \n", imageHeight, imageWidth);

	// Allocate device magnitude image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMalloc deviceGradientDirection\n";
	if (cudaMalloc((void**)&deviceGradientDirection, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		return;
	}
	

	// Allocate device magnitude image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMalloc deviceGradientStrength\n";
	if (cudaMalloc((void**)&deviceGradientStrength, imageWidth * imageHeight * sizeof(float)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceGradientDirection);
		return;
	}
	

	// Allocate device magnitude image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMalloc deviceGradientStrengthOut\n";
	if (cudaMalloc((void**)&deviceGradientStrengthOut, imageWidth * imageHeight * sizeof(float)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceGradientDirection);
		cudaFree(deviceGradientStrength);
		return;
	}


	// Allocate device gradient direction image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMalloc deviceGradDirXImage\n";
	if (cudaMalloc((void**)&deviceGradDirXImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceGradientDirection);
		cudaFree(deviceGradientStrength);
		cudaFree(deviceGradientStrengthOut);
		return;
	}


	// Allocate device gradient direction image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMalloc deviceGradDirYImage\n";
	if (cudaMalloc((void**)&deviceGradDirYImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceGradientDirection);
		cudaFree(deviceGradientStrength);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradientStrengthOut);
		return;
	}

	// Allocate device image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMalloc deviceImage\n";
	if (cudaMalloc((void**)&deviceImage, imageWidth * imageHeight * sizeof(unsigned char)) != cudaSuccess)
	{
		cudaFree(deviceGradientDirection);
		cudaFree(deviceImage);
		cudaFree(deviceGradientStrength);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		cudaFree(deviceGradientStrengthOut);
		std::cout << "        Error!\n";
		return;
	}

	// copy image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMemcpy deviceGradDirXImage\n";
	if (cudaMemcpy(deviceGradDirXImage, hostDirXImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceGradientDirection);
		cudaFree(deviceImage);
		cudaFree(deviceGradientStrength);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		cudaFree(deviceGradientStrengthOut);
		std::cout << "        Error!\n";
		return;

	}

	// copy image
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMemcpy deviceGradDirYImage\n";
	if (cudaMemcpy(deviceGradDirYImage, hostDirYImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(deviceGradientDirection);
		cudaFree(deviceImage);
		cudaFree(deviceGradientStrength);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		cudaFree(deviceGradientStrengthOut);
		std::cout << "        Error!\n";
		return;

	}
	
	
    // Do Timing
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T0, start, stop);
    }
	
	
    
    // ##################################################################################################################################
	// Do the computation on the GPU

	
	if (DEBUG_MODE)
	    std::cout << "    CANNY::kernel\n";
	int blockSize = 32;
	dim3 DimGrid((imageWidth - 1) / blockSize + 1, (imageHeight - 1) / blockSize + 1, 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	
	
    if (TIMER_MODE)
    	cudaEventRecord(start);
	if (DEBUG_MODE)
        std::cout << "    CANNY::kernel_CannyGradients_GLOBAL\n";
	kernel_CannyGradientCalcs_GLOBAL<<<DimGrid,DimBlock>>>(deviceGradDirXImage, deviceGradDirYImage, imageWidth, imageHeight, 
	                                                    deviceGradientStrength, deviceGradientDirection);


    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
     
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T1, start, stop);
        
        cudaEventRecord(start);
	}
	
	
	if (DEBUG_MODE)                          
        std::cout << "    CANNY::kernel_CannyNonMaxSuppression_GLOBAL\n";
    kernel_CannyNonMaxSuppression_GLOBAL<<<DimGrid,DimBlock>>>(deviceGradientStrengthOut, deviceGradientStrength, deviceGradientDirection,
                                                        imageWidth, imageHeight);


    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
     
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T2, start, stop);
    
    	cudaEventRecord(start);
	}
	
    for (int tIts = 0; tIts < thresholdIdterations; tIts++)
    {
	    if (DEBUG_MODE)
            std::cout << "    CANNY::kernel_CannyThresholding_GLOBAL\n";
        kernel_CannyThresholding_GLOBAL<<<DimGrid,DimBlock>>>(deviceGradientStrength, deviceGradientStrengthOut, 
                                                        imageWidth, imageHeight, minThreshold, maxThreshold);
        
    	
	    // copy image
	    if (DEBUG_MODE)
	        std::cout << "    CANNY::cudaMemcpy deviceGradientStrengthOut\n";
		if (cudaMemcpy(deviceGradientStrength, deviceGradientStrengthOut, imageWidth*imageHeight*sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess)
	    {
		    cudaFree(deviceGradientDirection);
		    cudaFree(deviceImage);
		    cudaFree(deviceGradientStrength);
		    cudaFree(deviceGradDirXImage);
		    cudaFree(deviceGradDirYImage);
		    cudaFree(deviceGradientStrengthOut);
		    std::cout << "        Error!\n";
		    return;
	    }
    }
        if (TIMER_MODE)
        {
            cudaEventRecord(stop);
         
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&T3, start, stop);
        
        	cudaEventRecord(start);
    	}
	if (DEBUG_MODE)                          
        std::cout << "    CANNY::kernel_CannyHysteresis_GLOBAL\n";
    kernel_CannyHysteresis_GLOBAL<<<DimGrid,DimBlock>>>(deviceGradientStrength, deviceGradientStrengthOut, imageWidth, imageHeight);
    
    
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T5, start, stop);
        
	    cudaEventRecord(start);
    }
    
    
	if (DEBUG_MODE)
        std::cout << "    CANNY::kernel_CannyBlockConverter\n";
    kernel_CannyBlockConverter<<<DimGrid,DimBlock>>>(deviceGradientStrengthOut, deviceImage, imageWidth, imageHeight);
    
    
    
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T6, start, stop);
        
	    cudaEventRecord(start);
    }
    // ##################################################################################################################################
	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaMemcpy hostOutputImage\n";
	if (cudaMemcpy(hostOutputImage, deviceImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		std::cout << "        Error!\n";
		cudaFree(deviceGradientDirection);
		cudaFree(deviceGradientStrength);
		cudaFree(deviceImage);
		cudaFree(deviceGradDirXImage);
		cudaFree(deviceGradDirYImage);
		cudaFree(deviceGradientStrengthOut);
		return;
	}

	if (DEBUG_MODE)
	    std::cout << "    CANNY::cudaFree\n";
	cudaFree(deviceGradientDirection);
	cudaFree(deviceGradientStrength);
	cudaFree(deviceImage);
	cudaFree(deviceGradDirXImage);
	cudaFree(deviceGradDirYImage);
	cudaFree(deviceGradientStrengthOut);
	
    
    if (TIMER_MODE)
    {
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&T7, start, stop);
	
	
	    printf("****************************************\n");
	    printf("  \tCANNY TIMING \n");
	    printf("  \tMemSets  : %f msec\n", T0  );
	    printf("  \tGradCalcs: %f msec\n", T1  );
	    printf("  \tSuppress : %f msec\n", T2  );
	    printf("  \tThresh   : %f msec\n", T3  );
	    printf("  \tHyster   : %f msec\n", T5  );
	    printf("  \tConvert  : %f msec\n", T6  );
	    printf("  \tCleanup  : %f msec\n", T7  );
	
	    printf("  \tTotal    : %f msec\n", (T0+T1+T2+T3+T5+T6+T7));
	
	    printf("****************************************\n\n");
    }

}
