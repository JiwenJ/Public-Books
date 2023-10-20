LIBDIRS = -L/System/Library/Frameworks/OpenGL.framework/Libraries
LIBDIRS += -L/opt/local/lib
INCDIRS = -I/opt/local/include/GL
INCDIRS += -I/Developer/GPU\ Computing/C/common/inc/GL/
FRAMEWORK = -framework GLUT
FRAMEWORK += -framework OpenGL 
LIBS = -lGLEW -lGL -lGLU

PRG = minimum_opengl
CC = g++
FLAG = -O3

${PRG}:${PRG}.cpp
	${CC} ${FRAMEWORK} ${FLAG} -o ${PRG} ${LIBS} ${PRG}.cpp ${LIBDIRS} ${INCDIRS}

clean:
	rm ${PRG}