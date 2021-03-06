/************************************************************
Known issues:
This program only works on matrices smaller than or equal to 
256x256. 1024x1024 will cause segmentation faults and 512x512
simply causes the program to almost crash and return times of
0 for each kernel call. 
The matrices must be square and all matrices must be the same
size.
*/

#include <stdio.h>
#include <stdlib.h>
#define MATSIZE 128
#define THREADS_PER_BLOCK 32 
 

 
//serial matrix multiplication kernel
__global__ void smultiply(int* g_a, int* g_b, int* g_c)
{
  int x, y, z;

  for (x = 0; x < MATSIZE; ++ x)
  {
    for (y = 0; y < MATSIZE; ++ y)
    {
      for (z = 0; z < MATSIZE; ++ z)
      {
	g_c[(x * MATSIZE) + y] += g_a[(x * MATSIZE) + z] * g_b[(z * MATSIZE) + y];
      }
    }
  }
}

//parallel matrix multiplication kernel
__global__ void pmultiply(double* g_a, double* g_b, double* g_d , int dim)
{
  int z;
  double sum = 0.0;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  for (z = 0; z < dim; ++ z)
  {
    sum += g_a[x * dim + z] * g_b[y + z * dim];
  }
    
  g_d[(x * dim) + y] = sum;
}

extern "C" void Cudamultiply(double* a, double* b, double* d, int Dim)
{
//  double *d;
  int i;
  double *g_a, *g_b, *g_d;
  int g_size = Dim * Dim * sizeof(double);
  cudaEvent_t start, stop;
  float time; 

//  d=new double[Dim*Dim];

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //used for timing the Cuda run

  cudaMalloc(&g_a, g_size); //allocate memory on Cuda device
  cudaMemcpy(g_a, a, g_size, cudaMemcpyHostToDevice);
  //copy matrix A onto the Cuda device  

  cudaMalloc(&g_b, g_size);
  cudaMemcpy(g_b, b, g_size, cudaMemcpyHostToDevice);
  
//  cudaMalloc(&g_c, g_size);
//  cudaMemcpy(g_c, c, g_size, cudaMemcpyHostToDevice);

  dim3 dimGrid((Dim / THREADS_PER_BLOCK), (Dim / THREADS_PER_BLOCK));
  //create the needed number of grids
  dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  //create the needed number of threads in each grid

  //serial Cuda kernel call
  //cudaEventRecord(start, 0);
  //smultiply<<<1,1>>>(g_a, g_b, g_c);
  //cudaEventRecord(stop, 0);  

  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&time, start, stop);
  //get run time

//  cudaMemcpy(c, g_c, g_size, cudaMemcpyDeviceToHost);
  //copy results back to host device
//  cudaFree(g_c);  
  //free up unused user allocated memory on Cuda device

  printf("Time = %f milliseconds\n", time);
  
  //create a second answer matrix to use
  //This is not done until now so that memory on the
  //Cuda device is not wasted.
  cudaMalloc(&g_d, g_size);
  cudaMemcpy(g_d, d, g_size, cudaMemcpyHostToDevice);

  //parallel Cuda kernel call
  cudaEventRecord(start, 0);
  pmultiply<<<dimGrid,threads>>>(g_a, g_b, g_d,Dim);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Time = %f milliseconds\n", time);

  cudaMemcpy(d, g_d, g_size, cudaMemcpyDeviceToHost);

  cudaFree(g_a);
  cudaFree(g_b);
  cudaFree(g_d);
  //free up all unused user allocated memory on Cuda device

  //printf("\n");  

  /*The next 2 for loops print out the values of both
  answer matrices. This can be used to ensure that both
  kernel calls are producing the same results, and that
  the results are correct. This section can be commented 
  out when the user only wants the timing of a run.*/

//  for (i = 1; i <= Dim * Dim; ++ i)
//  {
//      if (i % Dim == 0)
//      {
////        printf("%f ", c[i-1]);
//	printf("\n");
//      }
//   
//      else
////        printf("%f ", c[i-1]);
//  }

 // printf("\n");

//  for (i = 1; i <= Dim * Dim; ++ i)
//  {
//    if ( i % Dim == 0)
//    {
//     printf("%f ", d[i-1]);
//     printf("\n");
//    }
//
//    else
//     printf("%f ", d[i-1]);
//  }
//  delete d;
}
