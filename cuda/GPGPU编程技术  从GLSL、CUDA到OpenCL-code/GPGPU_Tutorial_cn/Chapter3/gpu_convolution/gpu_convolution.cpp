/*
 * @brief The First Example: GLSL-accelerated Discrete Convolution
 * @author Deyuan Qiu
 * @date June 3, 2009
 * @file gpu_convolution.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <glew.h>
#include <GLUT/glut.h>
#include "CReader.h"
#include "CTimer.h"

#define WIDTH	1024	//data block width
#define HEIGHT	1024	//data block height
#define MASK_RADIUS	2	//Mask radius

using namespace std;

void initGLSL(void);
void initFBO(unsigned unWidth, unsigned unHeight);
void initGLUT(int argc, char** argv);
void createTextures (void);
void setupTexture(const GLuint texID);
void performComputation(void);
void transferFromTexture(float* data);
void transferToTexture(float* data, GLuint texID);

// texture identifiers
GLuint yTexID;
GLuint xTexID;

// GLSL vars
GLuint glslProgram;
GLuint fragmentShader;
GLint outParam, inParam, radiusParam;

// FBO identifier
GLuint fb;

// handle to offscreen "window", providing a valid GL environment.
GLuint glutWindowHandle;

// struct for GL texture (texture format, float format etc)
struct structTextureParameters {
	GLenum texTarget;
	GLenum texInternalFormat;
	GLenum texFormat;
	char* shader_source;
}textureParameters;

// global vars
float* pfInput;			//input data
float fRadius = (float)MASK_RADIUS;
unsigned unWidth = (unsigned)WIDTH;
unsigned unHeight = (unsigned)HEIGHT;
unsigned unSize = unWidth * unHeight;

int main(int argc, char **argv) {
	// create test data
	unsigned unNoData = 4 * unSize;		//total number of Data
	pfInput = new float[unNoData];
	float* pfOutput = new float[unNoData];
	for (unsigned i = 0; i < unNoData; i++)	pfInput[i] = i;

	// create variables for GL
	textureParameters.texTarget			= GL_TEXTURE_RECTANGLE_ARB;
	textureParameters.texInternalFormat	= GL_RGBA32F_ARB;
	textureParameters.texFormat			= GL_RGBA;
	CReader reader;

    // init glut and glew
    initGLUT(argc, argv);
    glewInit();
    // init framebuffer
    initFBO(unWidth, unHeight);
    // create textures for vectors
    createTextures();
    // clean the texture buffer (for security reasons)
    textureParameters.shader_source = reader.textFileRead("clean.frag");
    initGLSL();
    performComputation();
    // perform computation
    textureParameters.shader_source = reader.textFileRead("convolution.frag");
    initGLSL();
    performComputation();

    // get GPU results
    transferFromTexture (pfOutput);

    // clean up
    glDetachShader(glslProgram, fragmentShader);
    glDeleteShader(fragmentShader);
    glDeleteProgram(glslProgram);
    glDeleteFramebuffersEXT(1,&fb);
    glDeleteTextures(1,&yTexID);
    glDeleteTextures (1,&xTexID);
    glutDestroyWindow (glutWindowHandle);

    // exit
    delete pfInput;
    delete pfOutput;
    return EXIT_SUCCESS;
}

/**
 * Set up GLUT. The window is created for a valid GL environment.
 */
void initGLUT(int argc, char **argv) {
    glutInit ( &argc, argv );
    glutWindowHandle = glutCreateWindow("GPGPU Tutorial");
}

/**
 * Off-screen Rendering.
 */
void initFBO(unsigned unWidth, unsigned unHeight) {
    // create FBO (off-screen framebuffer)
    glGenFramebuffersEXT(1, &fb);
    // bind offscreen framebuffer (that is, skip the window-specific render target)
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
    // viewport for 1:1 pixel=texture mapping
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, unWidth, 0.0, unHeight);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, unWidth, unHeight);
}

/**
 * Set up the GLSL runtime and creates shader.
 */
void initGLSL(void) {
    // create program object
    glslProgram = glCreateProgram();
    // create shader object (fragment shader)
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER_ARB);
    // set source for shader
    const GLchar* source = textureParameters.shader_source;
    glShaderSource(fragmentShader, 1, &source, NULL);
    // compile shader
    glCompileShader(fragmentShader);

    // attach shader to program
    glAttachShader (glslProgram, fragmentShader);
    // link into full program, use fixed function vertex shader.
    // you can also link a pass-through vertex shader.
    glLinkProgram(glslProgram);

    // Get location of the uniform variable
    radiusParam = glGetUniformLocation(glslProgram, "fRadius");
}

/**
 * create textures and set proper viewport etc.
 */
void createTextures (void) {
    // create textures.
    // y is write-only; x is just read-only.
    glGenTextures (1, &yTexID);
    glGenTextures (1, &xTexID);
    // set up textures
    setupTexture (yTexID);
    setupTexture (xTexID);
    transferToTexture(pfInput,xTexID);
    // set texenv mode
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}

/**
 * Sets up a floating point texture with the NEAREST filtering.
 */
void setupTexture (const GLuint texID) {
    // make active and bind
    glBindTexture(textureParameters.texTarget,texID);
    // turn off filtering and wrap modes
    glTexParameteri(textureParameters.texTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(textureParameters.texTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(textureParameters.texTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(textureParameters.texTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // define texture with floating point format
    glTexImage2D(textureParameters.texTarget,0,textureParameters.texInternalFormat,unWidth,unHeight,0,textureParameters.texFormat,GL_FLOAT,0);
}

void performComputation(void) {
    // attach output texture to FBO
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, textureParameters.texTarget, yTexID, 0);

    // enable GLSL program
    glUseProgram(glslProgram);
    // enable the read-only texture x
    glActiveTexture(GL_TEXTURE0);
    // enable mask radius
    glUniform1f(radiusParam,fRadius);
    // Synchronize for the timing reason.
	glFinish();

	CTimer timer;
	long lTime = 0.0;
	timer.reset();

	// set render destination
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	// Hit all texels in quad.
	glPolygonMode(GL_FRONT, GL_FILL);

	// render quad with unnormalized texcoords
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex2f(0.0, 0.0);
	glTexCoord2f(unWidth, 0.0);
	glVertex2f(unWidth, 0.0);
	glTexCoord2f(unWidth, unHeight);
	glVertex2f(unWidth, unHeight);
	glTexCoord2f(0.0, unHeight);
	glVertex2f(0.0, unHeight);
	glEnd();
	glFinish();
	lTime = timer.getTime();
	cout<<"Time elapsed: "<<lTime<<" ms."<<endl;
}

/**
 * Transfers data from currently texture to host memory.
 */
void transferFromTexture(float* data) {
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    glReadPixels(0, 0, unWidth, unHeight,textureParameters.texFormat,GL_FLOAT,data);
}

/**
 * Transfers data to texture. Notice the difference between ATI and NVIDIA.
 */
void transferToTexture (float* data, GLuint texID) {
    // version (a): HW-accelerated on NVIDIA
    glBindTexture(textureParameters.texTarget, texID);
    glTexSubImage2D(textureParameters.texTarget,0,0,0,unWidth,unHeight,textureParameters.texFormat,GL_FLOAT,data);
    // version (b): HW-accelerated on ATI
//	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, textureParameters.texTarget, texID, 0);
//	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
//	glRasterPos2i(0,0);
//	glDrawPixels(unWidth,unHeight,textureParameters.texFormat,GL_FLOAT,data);
//	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, textureParameters.texTarget, 0, 0);
}
