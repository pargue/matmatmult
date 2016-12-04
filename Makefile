all: m-dist mat program m-p

m-dist:
	mpicxx -std=c++11 -o m-dist -Wall -g -O3 src/mpi/m-dist.cc
mat:
	mpicxx -std=c++11 -o mat StrassenSerial.cpp
program:
	mpicxx -std=c++11 -c cudam-dist.cc -o cudam-dist.o
	nvcc -arch=sm_20 -c cudamatrix.cu -o cudamatrix.o
	mpicxx cudam-dist.o cudamatrix.o -lcudart -L/apps/CUDA/cuda-5.0/lib64/ -o program
m-p: src/mpi/petsc-mmm.c
	mpicc -Wall -DPETSC_USE_LOG -I/usr/lib/petscdir/3.4.2/include -o $@ $^  -lpetsc

run:
	mpirun -np 7 ./dist-mat 4 0

runc:
	mpirun -np 7 ./program 4 1