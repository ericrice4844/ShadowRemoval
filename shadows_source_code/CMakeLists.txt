cmake_minimum_required(VERSION 3.4)
project(directory_test)
 
#Bring the headers into the project
include_directories(include)

 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")
 
add_executable(shadowRemover ${SOURCES})