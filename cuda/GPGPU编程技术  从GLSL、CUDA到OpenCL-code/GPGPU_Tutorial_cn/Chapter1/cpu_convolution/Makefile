LIBDIRS += -L.
LIBDIRS += -L../../lib
LIBS = -lCTimer -lCSystem
INCDIRS += -I../../CTimer
INCDIRS += -I../../CSystem
OS = ${shell uname -s}

PRG = convolution
CC = g++
FLAG = -O3

${PRG}:${PRG}.cpp
ifeq (${OS}, Darwin)
	@echo Building with ${OS}.
	${CC} ${FLAG} ${LIBDIRS} ${INCDIRS} -o ${PRG} ${PRG}.cpp ${LIBS} 
else
	@echo Operating system not supported.
endif

clean:
	rm ${PRG}