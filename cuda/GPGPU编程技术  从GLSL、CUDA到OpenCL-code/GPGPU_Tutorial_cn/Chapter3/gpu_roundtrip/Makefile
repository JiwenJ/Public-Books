LIBDIRS = -L/System/Library/Frameworks/OpenGL.framework/Libraries
LIBDIRS += -L/opt/local/lib
INCDIRS = -I/opt/local/include/GL
FRAMEWORK = -framework GLUT
FRAMEWORK += -framework OpenGL 
LIBS = -lGLEW -lGL -lGLU

PRG = gpu_roundtrip
CC = g++
FLAG = -Wall

${PRG}: ${PRG}.cpp
	${CC} ${FRAMEWORK} ${FLAG} -o ${PRG} ${LIBS} ${PRG}.cpp ${LIBDIRS} ${INCDIRS}

clean:
	rm ${PRG}