/*
 * @brief The minimum OpenGL application: 2nd version
 * @author Deyuan Qiu
 * @date May 8, 2009
 * @file minimum_shader.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <glew.h>
#include <GLUT/glut.h>
#include "CReader.h"

GLuint v,f,p;
float lpos[4] = {1,0.5,1,0};
float a = 0;

void changeSize(int w, int h) {
	// Prevent a divide by zero, when window is too short
	if(h == 0)	h = 1;
	float ratio = 1.0* w / h;

	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Set the viewport to be the entire window
    glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,1,1000);
	glMatrixMode(GL_MODELVIEW);
}

void renderScene(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	gluLookAt(0.0,0.0,5.0,
		      0.0,0.0,-1.0,
			  0.0f,1.0f,0.0f);
	glLightfv(GL_LIGHT0, GL_POSITION, lpos);
	glRotatef(a,0,1,1);
	glutSolidTeapot(1);
	a+=0.1;
	glutSwapBuffers();
}

void setShaders() {
	char *vs = NULL,*fs = NULL;
	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	CReader reader;
	vs = reader.textFileRead("passthrough.vert");
	fs = reader.textFileRead("passthrough.frag");

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv,NULL);
	glShaderSource(f, 1, &ff,NULL);

	free(vs);free(fs);
	glCompileShader(v);
	glCompileShader(f);

	p = glCreateProgram();
	glAttachShader(p,v);
	glAttachShader(p,f);
	glLinkProgram(p);
	glUseProgram(p);
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(320,320);
	glutCreateWindow("GPGPU Tutorial");
	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);
	glutReshapeFunc(changeSize);
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0,0.0,0.0,1.0);
	glColor3f(1.0,1.0,1.0);
	glEnable(GL_CULL_FACE);
	glewInit();

	setShaders();

	glutMainLoop();

	return 0;
}
