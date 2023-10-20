PRG = CReader
CC = g++
FLAG = -O3 -arch i386

${PRG}.a:${PRG}.h ${PRG}.cpp
	@${CC} -c ${PRG}.cpp
	@ar -rcs lib${PRG}.a ${PRG}.o
	@rm *.o

clean:
	@rm *.a
	
install:
	@mv lib${PRG}.a ../lib