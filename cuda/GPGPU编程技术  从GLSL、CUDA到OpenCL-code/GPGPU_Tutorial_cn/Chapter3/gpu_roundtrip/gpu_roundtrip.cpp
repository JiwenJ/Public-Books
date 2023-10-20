/*
 * @brief OpenGL texture memory roundtrip test.
 * @author Deyuan Qiu
 * @date June 3, 2009
 * @file gpu_roundtrip.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <glew.h>
#include <GLUT/glut.h>

#define WIDTH	2	//data block width
#define HEIGHT	3	//data block height

using namespace std;

int main(int argc, char **argv) {
	int nWidth = (int)WIDTH;
	int nHeight = (int)HEIGHT;
	int nSize = nWidth * nHeight;

	// create test data
	float* pfInput = new float[4* nSize];
	float* pfOutput = new float[4* nSize];
	for (int i = 0; i < nSize * 4; i++)	pfInput[i] = i + 1.2345;

	// set up glut to get valid GL context and get extension entry points
	glutInit(&argc, argv);
	glutCreateWindow("GPGPU Tutorial");
	glewInit();

	// create FBO and bind it
	GLuint fb;
	glGenFramebuffersEXT(1, &fb);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);

	// create texture and bind it
	GLuint tex;
	glGenTextures(1, &tex);
//	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	// set texture parameters
//	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
//	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// attach texture to the FBO
//	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, tex, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex, 0);

	// define texture with floating point format
//	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, nWidth, nHeight, 0, GL_RGBA, GL_FLOAT, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA_FLOAT32_ATI, nWidth, nHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	// transfer data to texture
//	glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, nWidth, nHeight, GL_RGBA, GL_FLOAT, pfInput);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nWidth, nHeight, GL_RGBA, GL_FLOAT, pfInput);

	// and read back
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glReadPixels(0, 0, nWidth, nHeight, GL_RGBA, GL_FLOAT, pfOutput);

	// print and check results
	bool bCmp = true;
	for (int i = 0; i < nSize * 4; i++){
		cout<<i<<":\t"<<pfInput[i]<<'\t'<<pfOutput[i]<<endl;
		if(pfInput[i] != pfOutput[i])	bCmp = false;
	}
	if(bCmp)	cout<<"Round trip complete!"<<endl;
	else		cout<<"Raund trip failed!"<<endl;

	// clean up
	delete pfInput;
	delete pfOutput;
	glDeleteFramebuffersEXT(1, &fb);
	glDeleteTextures(1, &tex);
	return 0;
}
