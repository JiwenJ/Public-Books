PRG = CSystem
CC = g++
FLAG = -O3 -arch i386

${PRG}.a:${PRG}.h ${PRG}.cpp
	@${CC} -c ${PRG}.h
	@ar -rcs lib${PRG}.a ${PRG}.h.gch
	@rm *.h.gch

clean:
	@rm *.a
	
install:
	@mv lib${PRG}.a ../lib