/*
 * @brief The minimum OpenGL application
 * @author Deyuan Qiu
 * @date May 8, 2009
 * @file minimum_opengl.cpp
 */

#include <stdio.h>
#include <stdlib.h>
#include <glew.h>
#include <GLUT/glut.h>

GLuint v,f,p;
float lpos[4] = {1,0.5,1,0};

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

float a = 0;

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
//	glEnable(GL_CULL_FACE);
	glewInit();

	glutMainLoop();

	return 0;
}
