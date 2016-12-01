all:
	mpicxx -std=c++11 -o m-dist -Wall -g -O3 src/mpi/m-dist.cc
	mpicxx -std=c++11 -o mat StrassenSerial.cpp
	mpicxx -std=c++11 -c cudam-dist.cc -o cudam-dist.o
	nvcc -arch=sm_20 -c cudamatrix.cu -o cudamatrix.o
	mpicxx cudam-dist.o cudamatrix.o -lcudart -L/apps/CUDA/cuda-5.0/lib64/ -o program

run:
	mpirun -np 7 ./dist-mat 4 0

runc:
	mpirun -np 7 ./program 4 1