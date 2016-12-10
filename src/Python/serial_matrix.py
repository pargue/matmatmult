#!/usr/bin/env python3

"""
AUTHOR: Cody Kankel
PROG: Serial_matrix.py
DESC: This Python 3 script is a serial version of parallel_matrix.py.
MPI4py is still required for this serial version as the MPI time is 
still being used to calculate the run time of Strassen algorithm.
This serial version needs to be ran with only 1 proc/rank, and it
will multiply two matricies together, both the same matrix as specified
on the cmd line while executing this script. Example:
    mpirun -np 1 serial_matrix.py csv/4096_4096.csv
"""


import numpy, sys, csv, math
from mpi4py import MPI

def main():
    """Main will read and initialize 2 matrices, print them,
    generate the correct dot product of them, and send them off to the
    strassen function to be calulated there. It is up to the user of this
    program as of right now to verify the strassen method is working."""
 
    if len(sys.argv) != 2:
        sys.exit(2)
 
        
    matrix_A = get_matrix(str(sys.argv[1]))
    matrix_B = get_matrix(str(sys.argv[1]))
        
        
    #print('Matrix A is:')
    #print(matrix_A )
    #print('Matrix B is:')
    #print(matrix_B)
    #print('-'.center(40,'-'))
    #print('The correct product of these matrices is:')
    #print(numpy.dot(matrix_A, matrix_B))
    
    if matrix_A.shape != matrix_B.shape:
        print('Error: Matrix A and Matrix B are not the same size Matrix.')
        sys.exit()
        
    a_subSize = int(get_dim(matrix_A)/2)
    b_subSize = int(get_dim(matrix_B)/2)
    if a_subSize != b_subSize:
        print("error")
        sys.exit()
    startTime = MPI.Wtime()
        
    matrix_C = strassen(matrix_A, matrix_B, a_subSize)
    
    # Leaving MPI in the serial version solely for the use of the MPI time.
    runTime = MPI.Wtime() - startTime
        
    print("The time to calculate strassen function in parallel is:\n", runTime)

    sys.exit()

def get_matrix(fileName):
    """Function to open a specified file and read the contents using numpy's loadtxt call.
    Function will return a numpy matrix (formatted). 'fileName' argument MUST be a string"""
    
    with open(fileName, 'r') as file_ob:
        reader = csv.reader(file_ob)
        temp_list = list(reader)
        temp_list = temp_list[0]
        temp_list = list(map(int, temp_list))
        matr_len = len(temp_list)
        new_shape = int(math.sqrt(matr_len))
        matrix = numpy.asarray(temp_list)
        matrix = matrix.reshape(new_shape, new_shape)
        return matrix
    
def strassen(A, B, subSize):
    """Function to perform the strassen algorithm on 2 numpy matricies specified as
    A and B. The function will return the dot product of these two matricies
    as a numpy.array matrix."""
    
            
        # Rank 0 is the master, so it will prepare everything to be parallelized
    a_11 = A[0:subSize, 0:subSize]
    a_12 = A[0:subSize, subSize:]
    a_21 = A[subSize:, 0:subSize]
    a_22 = A[subSize:, subSize:]
        
    b_11 = B[0:subSize, 0:subSize]
    b_12 = B[0:subSize, subSize:]
    b_21 = B[subSize:, 0:subSize]
    b_22 = B[subSize:, subSize:]
    
    
    m1 = (a_11 + a_22).dot((b_11 + b_22))
    m2 = ((a_21 + a_22).dot(b_11))
    m3 = a_11.dot(b_12 - b_22)
    m4 = a_22.dot(b_21 - b_11)
    m5 = (a_11 + a_12).dot(b_22)
    m6 = (a_21 - a_11).dot(b_11 + b_12)
    m7 = (a_12 - a_22).dot((b_21 + b_22))
        
        

    C11 = m1 + m4 - m5 + m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 -m2 + m3 + m6
        

    # making final matrix from each piece
    C = numpy.bmat([[C11, C12], [C21, C22]])
    return C


def get_dim(matrix):
    """Function to get the dim of a matrix and return. Assumes the matricies are
    already square. Returns an integer for the dim of the matrix"""
    return int((str(matrix.shape).split(',')[0].replace('(','')))

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
