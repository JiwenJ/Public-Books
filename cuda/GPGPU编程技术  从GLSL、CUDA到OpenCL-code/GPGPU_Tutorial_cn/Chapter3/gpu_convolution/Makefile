LIBDIRS = -L/System/Library/Frameworks/OpenGL.framework/Libraries
LIBDIRS += -L/opt/local/lib
LIBDIRS += -L.
LIBDIRS += -L../../lib
INCDIRS = -I/opt/local/include/GL
INCDIRS += -I../../CReader
INCDIRS += -I../../CTimer
FRAMEWORK = -framework GLUT
FRAMEWORK += -framework OpenGL 
LIBS = -lGLEW -lGL -lGLU -lCReader -lCTimer

PRG = gpu_convolution
CC = g++
FLAG = -O3

${PRG}: ${PRG}.cpp
	${CC} ${FRAMEWORK} ${FLAG} -o ${PRG} ${PRG}.cpp ${LIBDIRS} ${INCDIRS} ${LIBS} 

clean:
	rm ${PRG}