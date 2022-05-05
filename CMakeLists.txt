cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(shadowRemover LANGUAGES CXX CUDA)
 
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED PATHS "/home/u5/ericrice4844/opencv/build")

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
set(COMPUTE_CAPABILITY 60)
# set(CUDA_SEPARABLE_COMPILATION TRUE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture;sm_${COMPUTE_CAPABILITY}; ";)
# list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Set warning levels for host compilation
if (UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra")
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-function,-Wno-unused-parameter")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif ()

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif()
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else ()
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++-3.6 clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        set(CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" )
    endif ()
endif ()



# Add debugging to CUDA NVCC flags (for NVidia's NSight tools)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")



# Set the correct linker options on UNIX platforms
# -lpthread inserted by ali akoglu  for compatiblity with elgato
if (UNIX)
    set(LINK_LIBRARIES "-lpthread -lm")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else ()
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else ()
            message(FATAL_ERROR "clock_gettime() not found")
        endif ()
    endif ()
endif ()



# Include Directories with libraries and source code
include_directories(${PROJECT_SOURCE_DIR}/shadows_source_code/include)
include_directories(${PROJECT_SOURCE_DIR}/shadows_source_code/src)
include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})


## Build parallel kernels
add_library(Parallel_Kernels STATIC  
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Skeleton_Kernel.cu 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Canny_Master_Call.cu 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Gaussian_Kernels.cu 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Sobel_Kernels.cu   
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Canny_Kernels.cu   
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Parallel_Kernels.cu   
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/Color_Convert_Kernel.cu 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/mask.cu
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/FrameProperties.cu 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Skeleton_Kernel.h  
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Canny_Master_Call.h   
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Gaussian_Kernels.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Sobel_Kernels.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Canny_Kernels.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Parallel_Kernels.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Color_Convert_Kernel.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/mask.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/FrameProperties.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Constants.h
)
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(Parallel_Kernels PUBLIC cxx_std_11)

 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( Parallel_Kernels
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
                 
                 
                       
# Build the main executable
add_executable(shadowRemover ${PROJECT_SOURCE_DIR}/shadows_source_code/src/main.cpp
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/ConnComp.cpp	      
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/LrTextureShadRemParams.cpp  
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/ConnCompGroup.cpp 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/src/LrTextureShadRem.cpp  
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/ConnComp.h	 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/LrTextureShadRem.h
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/ConnCompGroup.h  
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/LrTextureShadRemParams.h	 
    ${PROJECT_SOURCE_DIR}/shadows_source_code/include/Constants.h)
    
# Link the parallel kernels and required libraries
target_link_libraries(shadowRemover Parallel_Kernels ${OpenCV_LIBS} ${CUDA_LIBRARIES})


 

 
 
 
