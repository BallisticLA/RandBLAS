CFLAGS=-Wall -g -O3
LIBS= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

sjlt: sjlt.cpp
	g++ sjlt.cpp -o sjlt.o $(CFLAGS) $(LIBS) -DMKL_ILP64 -m64 -I"${MKLROOT}/include" -fopenmp

clean:
	rm -f sjlt.so
