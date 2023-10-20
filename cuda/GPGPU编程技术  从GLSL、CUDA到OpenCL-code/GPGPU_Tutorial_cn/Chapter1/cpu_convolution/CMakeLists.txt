CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#include directories (-I)
INCLUDE_DIRECTORIES(.)

#link libraries (-l)
#LINK_LIBRARIES()

#library directories (-L)
#LINK_DIRECTORIES()

#project name
PROJECT (convolution)

#chosse compiler
SET(CMAKE_CXX_COMPILER g++)

#flags
ADD_DEFINITIONS(-O3)

#echos
MESSAGE(STATUS "This is BINARY dir " ${HELLO_BINARY_DIR})
MESSAGE(STATUS "This is SOURCE dir "${HELLO_SOURCE_DIR})

#target library file and source files for it
#ADD_LIBRARY()

#target executable file and source files for it
ADD_EXECUTABLE(convolution convolution.cpp CTimer.h CTimer.cpp CSystem_float.h CSystem_float.cpp)