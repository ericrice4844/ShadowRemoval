
##########################################################################
# ShadowRemoval
ECE569 Shadow Removal Code

# Eric Rice, Tanner Bryant, Jonathan Herrera, Charles Hoskins




##########################################################################
Requires openCV 4.5.5 @ "/home/u5/ericrice4844/opencv/build") [path in CMakeLists.txt by default]
Requires cmake 3.5.1  @ "/home/u5/ericrice4844/CMAKE_FILES/cmake-3.5.1-Linux-x86_64")

    
##########################################################################
Source code is in 'shadows_source_code'
    *.h   files in the 'include' directory
    *.cpp files in the 'src' directory
    *.cu  files in the 'src' directory
    

##########################################################################
Build instructions:

0.) Create main directory for build
    mkdir ~/ECE569_Spring22_ShadowRemoval_Group2
    
1.) Copy/unpack build to home directory
    a.) Copy files to main directory
        cp -r /home/u5/ericrice4844/ece569/Shadow_Removal_Final/ShadowRemoval-feature-integration/* ~/ECE569_Spring22_ShadowRemoval_Group2
    - or - 
    b.) Extract projec files to main directory
        unzip project.zip ~/ECE569_Spring22_ShadowRemoval_Group2
    
2.) Add cmake 3.5.1 path to ~/.bashrc file
    export PATH="/home/u5/ericrice4844/CMAKE_FILES/cmake-3.5.1-Linux-x86_64/bin:$PATH"
        
3.) Load CUDA module
    module load cuda11/11.0

4.) Create "build" directory
    rm ~/ECE569_Spring22_ShadowRemoval_Group2/build
    mkdir ~/ECE569_Spring22_ShadowRemoval_Group2/build
    cd ~/ECE569_Spring22_ShadowRemoval_Group2/build

5.) Prep and build commands:
    CC=gcc cmake3 ..
    make
    
6.) Setup image output directories in "build" directory
    mkdir Canny MaskDiff Skeleton Colors Final
    


##########################################################################
Execute Instructions:

In ~/ECE569_ShadowRemoval/build, execute the commands for:

1.) Serial Version:
    ./shadowRemover 0
    
2.) Parallel Version:
    ./shadowRemover 1



##########################################################################
** NOTE ** 
* All output images in following directories: Canny  MaskDiff  Skeleton  Colors  Final
* Appended with "-CPU" are serial, and "-CUDA" and parallel.




