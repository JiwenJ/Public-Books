LIBDIRS += -L.
LIBDIRS += -L../../lib
INCDIRS = -I../../CTimer
LIBS = -lCTimer

PRG = cpu_quadratic_sum
CC = g++
FLAG = -O3

${PRG}:${PRG}.cpp
	${CC} ${FLAG} ${LIBDIRS} ${INCDIRS} -o ${PRG} ${PRG}.cpp ${LIBS}
clean:
	rm ${PRG}