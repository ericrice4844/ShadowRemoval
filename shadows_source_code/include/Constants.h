///////////////////////////////////////////////////////////////
//  
//      Constants.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////

#include<cuda_profiler_api.h>

#ifndef __CONSTANTS_H_
#define __CONSTANTS_H_

#define DEBUG_MODE false

#define TIMER_MODE false

// Define the image size here
// Big image
#define IM_ROWS 3120
#define IM_COLS 4160

// Small image
//#define IM_ROWS 288
//#define IM_COLS 384


#define IM_CHAN 3


#define BLOCK_DIM 32


// Gaussian blur params
#define GAUSS_WIN_RADI	2
#define GAUSS_WIN_WIDTH	(GAUSS_WIN_RADI*2+1)


#define COLUMNS_Y       2
#define ROWS_Y          4
#define RESULT_STEPS    2
#define HALO_STEPS      1

// Sobel filter parameters
#define SOBEL_DIM 1

// Skeleton parameters
#define NUM_SKELETONS 8
#define SKELETON_WIDTH 3

#endif
