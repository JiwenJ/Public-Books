INCDIRS = 	-I/usr/local/cuda/include \
			-I/usr/local/boost_1_39_0

PRG = GPUWorker
CC = g++-4.0
FLAG = -O3

${PRG}.a:${PRG}.h ${PRG}.cpp
	@${CC} ${CFLAGS} -c GPUWorker.cpp ${INCDIRS}
	@ar -rcs lib${PRG}.a ${PRG}.o
	@rm *.o

clean:
	@rm *.a
	
install:
	@mv lib${PRG}.a ../lib