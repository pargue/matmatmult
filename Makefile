all:
	mpicc -Wall -g -O0 src/mpi/m-dist.c -o m-dist

run:
	mpirun -np 7 ./m-dist
	