///////////////////////////////////////////////////////////////
//  
//      Parrallel_Kernels.h
//      Constaints headers for each parallel kernel so that cpp
//      code can access the functions.
//      
//
///////////////////////////////////////////////////////////////

extern "C" {
#include "Parallel_Kernels.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// ##################################################################################################
// ###   initFilter()   ###
// place holder wrapper
extern "C" void initFilter(void)
{
	std::cout << "HERE\n\n";;
	//cv::imshow(rgbImage);
	//cv::waitKey(0);

}
