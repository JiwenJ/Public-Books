CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#include directories (-I)
INCLUDE_DIRECTORIES(/opt/local/include/GL)

#link libraries (-l)
#LINK_LIBRARIES(CReader GLEW GL GLU)

#library directories (-L)
LINK_DIRECTORIES(/System/Library/Frameworks/OpenGL.framework/Libraries /opt/local/lib /Users/deyuanqiu/Data/Workspace/GPGPU_Tutorial/CReader)

#project name
PROJECT (glsl_uniform)

#chosse compiler
SET(CMAKE_CXX_COMPILER g++)

#flags
ADD_DEFINITIONS(-Wall -framework GLUT -framework OpenGL)

#echos
MESSAGE(STATUS "This is BINARY dir " ${HELLO_BINARY_DIR})
MESSAGE(STATUS "This is SOURCE dir "${HELLO_SOURCE_DIR})

#target library file and source files for it
#ADD_LIBRARY()

#target executable file and source files for it
ADD_EXECUTABLE(glsl_uniform glsl_uniform.cpp)

target_link_libraries(glsl_uniform CReader GLEW GL GLU)