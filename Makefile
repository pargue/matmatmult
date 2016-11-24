all:
	mpicxx -std=c++11 -o m-dist -Wall -g -O3 src/mpi/m-dist.cc
	mpicxx -std=c++11 -o mat StrassenSerial.cpp

run:
	mpirun -np 7 ./dist-mat 4 0
	