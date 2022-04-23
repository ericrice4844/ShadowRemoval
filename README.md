
##########################################################################
# ShadowRemoval
ECE569 Shadow Removal Code

# Eric Rice, Tanner Bryant, Jonathan Herrera, Charles Hoskins




##########################################################################
Requires openCV 4.5.5 @ "/home/u5/ericrice4844/opencv/build")
Requires cmake 3.5.1  @ "/home/u5/ericrice4844/CMAKE_FILES/cmake-3.5.1-Linux-x86_64")

    
##########################################################################
Source code is in 'shadows_source_code'
    *.h   files in the 'include' directory
    *.cpp files in the 'src' directory
    *.cu  files in the 'src' directory
    
To add a new parallel kernel which can be called from cpp:
    Follow the templates in Parallel_Kernels.h and Parallel_Kernels.cu
    

##########################################################################
Build instructions:

1.) Add cmake 3.5.1 path to ~/.bashrc file
    export PATH="/home/u5/ericrice4844/CMAKE_FILES/cmake-3.5.1-Linux-x86_64/bin:$PATH"
    
2.) Copy/Clone repo from source
    cp /home/u5/ericrice4844/ece569/ECE569_ShadowRemoval ~/ECE569_ShadowRemoval
    < or >
    git clone /home/u5/ericrice4844/ece569/ECE569_ShadowRemoval ~/ECE569_ShadowRemoval
    
3.) Load CUDA module
    module load cuda11/11.0

4.) Create "build" directory
    mkdir ~/ECE569_ShadowRemoval/build
    cd ~/ECE569_ShadowRemoval/build

5.) Prep and build commands:
    CC=gcc cmake3 ..
    make


##########################################################################
Execute Instructions:

In ~/ECE569_ShadowRemoval/build, execute ./shadowRemover




