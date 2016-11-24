all:
	mpicc -Wall -g -O0 src/mpi/m-dist.c -o m-dist
	mpicxx -std=c++11 -o mat StrassenSerial.cpp
	mpicxx -std=c++11 -o dist-mat src/mpi/V07.cc

run:
	mpirun -np 7 ./dist-mat 4 0
	