LIBDIRS = -L/System/Library/Frameworks/OpenGL.framework/Libraries
LIBDIRS += -L/opt/local/lib
LIBDIRS += -L.
LIBDIRS += -L../../lib
INCDIRS = -I/opt/local/include/GL
INCDIRS += -I../../CReader
FRAMEWORK = -framework GLUT
FRAMEWORK += -framework OpenGL 
LIBS = -lGLEW -lGL -lGLU -lCReader

PRG = minimum_shader
CC = g++
FLAG = -Wall

${PRG}: ${PRG}.cpp
	${CC} ${FRAMEWORK} ${FLAG} -o ${PRG} ${PRG}.cpp ${LIBDIRS} ${INCDIRS} ${LIBS} 

clean:
	rm ${PRG}