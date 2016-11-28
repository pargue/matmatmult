#!/usr/bin/env python3

import numpy, sys, csv, math
from mpi4py import MPI

# MPI calls necesarry for all ranks
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main():
    """Main will read and initialize 2 matrices, print them,
    generate the correct dot product of them, and send them off to the
    strassen function to be calulated there. It is up to the user of this
    program as of right now to verify the strassen method is working."""
 
    if len(sys.argv) != 2:
        sys.exit(2)
 
    if rank == 0:
        
        matrix_A = get_matrix(str(sys.argv[1]))
        matrix_B = get_matrix(str(sys.argv[1]))
        
        print('Matrix A is:')
        print(matrix_A )
        print('Matrix B is:')
        print(matrix_B)
        print('-'.center(40,'-'))
        print('The correct product of these matrices is:')
        print(numpy.dot(matrix_A, matrix_B))
    
        if matrix_A.shape != matrix_B.shape:
            print('Error: Matrix A and Matrix B are not the same size Matrix.')
            sys.exit()
        
        a_subSize = int(get_dim(matrix_A)/2)
        b_subSize = int(get_dim(matrix_B)/2)
        if a_subSize != b_subSize:
            print("error")
            sys.exit()
        a_subSize = comm.bcast(a_subSize, root=0)

        
    else:
        # Dumbie vars so other ranks can get into strassen function
        a_subSize = None
        a_subSize = comm.bcast(a_subSize, root=0)
        matrix_A = numpy.empty([2,2]) 
        matrix_B = numpy.empty([2,2])
        
    matrix_C = strassen(matrix_A, matrix_B, a_subSize)
    
    if rank == 0:
        print("Matrix C, after strassen is:")
        print(matrix_C)

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
    A and B. The function will (hopefully eventually) return the dot product of these two matricies
    as a numpy.array matrix."""
    
    
    if rank == 0:

        # Rank 0 is the master, so it will prepare everything to be parallelized
        a_11 = A[0:subSize, 0:subSize]
        a_11 = numpy.ascontiguousarray(a_11)
        a_12 = A[0:subSize, subSize:]
        a_12 = numpy.ascontiguousarray(a_12)
        a_21 = A[subSize:, 0:subSize]
        a_21 = numpy.ascontiguousarray(a_21)
        a_22 = A[subSize:, subSize:]
        a_22 = numpy.ascontiguousarray(a_22)
        
        b_11 = B[0:subSize, 0:subSize]
        b_11 = numpy.ascontiguousarray(b_11)
        b_12 = B[0:subSize, subSize:]
        b_12 = numpy.ascontiguousarray(b_12)
        b_21 = B[subSize:, 0:subSize]
        b_21 = numpy.ascontiguousarray(b_21)
        b_22 = B[subSize:, subSize:]
        b_22 = numpy.ascontiguousarray(b_22)
    
        # Setting up rank 1 for calculating m2
        comm.Send(a_21, dest=1, tag=11)
        comm.Send(a_22, dest=1, tag=12)
        comm.Send(b_11, dest=1, tag=13)

        # Setting up rank 2 for calculating m1
        comm.Send(a_11, dest=2, tag=14)
        comm.Send(a_22, dest=2, tag=15)
        comm.Send(b_22, dest=2, tag=16)
        comm.Send(b_11, dest=2, tag=17)
        
        # Setting up rank 3 for calculating m4
        comm.Send(a_22, dest=3, tag=18)
        comm.Send(b_11, dest=3, tag=19)
        comm.Send(b_21, dest=3, tag=20)

        # Setting up rank 4 for calculating m5
        comm.Send(a_11, dest=4, tag=21)
        comm.Send(a_12, dest=4, tag=22)
        comm.Send(b_22, dest=4, tag=23)

        # Setting up rank 5 for calculating m6
        comm.Send(a_11, dest=5, tag=24)
        comm.Send(a_21, dest=5, tag=25)
        comm.Send(b_11, dest=5, tag=26)
        comm.Send(b_12, dest=5, tag=27)

        # Setting up rank 6 for calculating m7
        comm.Send(a_12, dest=6, tag=28)
        comm.Send(a_22, dest=6, tag=29)
        comm.Send(b_21, dest=6, tag=30)
        comm.Send(b_22, dest=6, tag=31)

        # rank 0 will now calculate m3
        m3 = a_11.dot(b_12 - b_22)
        
        # rank 0 will send m3 to rank 5 to calculate c22
        comm.Send(m3, dest=5, tag=32)

        # rank 0 will receive m5 from 4 for C12
        m5 = numpy.arange((subSize * subSize))
        comm.Recv(m5, source=4, tag=36)
        m5 = m5.reshape(subSize, subSize)

        # rank 0 will now calculate C12
        C12 = m3 + m5
        
        #receiving the rest of C from the other ranks
        C11 = numpy.arange((subSize * subSize))
        comm.Recv(C11, source=2, tag=42)
        C21 = numpy.arange((subSize * subSize))
        comm.Recv(C21, source=3, tag=40)
        C22 = numpy.arange((subSize * subSize))
        comm.Recv(C22, source=5, tag=41)
        
        C11 = C11.reshape(subSize, subSize)
        C21 = C21.reshape(subSize, subSize)
        C22 = C22.reshape(subSize, subSize)
        # making empty matrix
        C = numpy.bmat([[C11, C12], [C21, C22]])
        return C
        
    if rank == 1:
        
        a_21 = numpy.arange((subSize * subSize))
        a_22 = numpy.arange((subSize * subSize))
        b_11 = numpy.arange((subSize * subSize))
        comm.Recv(a_21, source=0, tag=11)
        comm.Recv(a_22, source=0, tag=12)
        comm.Recv(b_11, source=0, tag=13)
        
        a_21 = a_21.reshape(subSize, subSize)
        a_22 = a_22.reshape(subSize, subSize)
        b_11 = b_11.reshape(subSize, subSize)
        # using numpy's matrix multiplier to calculate m2
        m2 = ((a_21 + a_22).dot(b_11))
        m2 = numpy.ascontiguousarray(m2)
    
        # sending m2 other ranks to calculate portions of matrix C
        comm.Send(m2, dest=5, tag=34)
        comm.Send(m2, dest=3, tag=35)
        return None
    
    if rank == 2:
        
        a_11 = numpy.arange((subSize * subSize))
        a_22 = numpy.arange((subSize * subSize))
        b_11 = numpy.arange((subSize * subSize))
        b_22 = numpy.arange((subSize * subSize))
        comm.Recv(a_11, source=0, tag=14)
        comm.Recv(a_22, source=0, tag=15)
        comm.Recv(b_11, source=0, tag=16)
        comm.Recv(b_22, source=0, tag=17)
        a_11 = a_11.reshape(subSize, subSize)
        a_22 = a_22.reshape(subSize, subSize)
        b_11 = b_11.reshape(subSize, subSize)
        b_22 = b_22.reshape(subSize, subSize)


        # using numpy's matrix multiplier to calculate m1        
        m1 = (a_11 + a_22).dot((b_11 + b_22))
        m1 = numpy.ascontiguousarray(m1)
        # sending m1 rank 5 to calulate portions of prodcut matrix C
        comm.Send(m1, dest=5, tag=33)
        
        m4 = numpy.arange((subSize * subSize))
        comm.Recv(m4, source=3, tag=36)
        m5 = numpy.arange((subSize * subSize))
        comm.Recv(m5, source=4, tag=38)
        m7 = numpy.arange((subSize * subSize))
        comm.Recv(m7, source=6, tag=39)
        m4 = m4.reshape(subSize, subSize)
        m5 = m5.reshape(subSize, subSize)
        m7 = m7.reshape(subSize, subSize)
        
        #calculating C11
        C11 = m1 + m4 - m5 + m7
        C11 = numpy.ascontiguousarray(C11)
        comm.Send(C11, dest=0, tag=42)
        return None
        
    if rank == 3:
        
        a_22 = numpy.arange((subSize * subSize))
        b_11 = numpy.arange((subSize * subSize))
        b_21 = numpy.arange((subSize * subSize))
        comm.Recv(a_22, source=0, tag=18)
        comm.Recv(b_11, source=0, tag=19)
        comm.Recv(b_21, source=0, tag=20)
        a_22 = a_22.reshape(subSize, subSize)
        b_11 = b_11.reshape(subSize, subSize)
        b_21 = b_21.reshape(subSize, subSize)
        
        # Using numpy's matrix multiplier to calculate m4
        m4 = a_22.dot(b_21 - b_11)
        m4 = numpy.ascontiguousarray(m4)
        # Sending m4 to rank 2
        comm.Send(m4, dest=2, tag=36)
        
        #receiving 2 from rank 1
        m2 = numpy.arange((subSize * subSize))
        comm.Recv(m2, source=1, tag=35)
        m2 = m2.reshape(subSize, subSize)

        C21 = m2 + m4
        C21 = numpy.ascontiguousarray(C21)
        comm.Send(C21, dest=0, tag=40)
        return None
    
    if rank == 4:
        
        a_11 = numpy.arange((subSize * subSize))
        a_12 = numpy.arange((subSize * subSize))
        b_22 = numpy.arange((subSize * subSize))
        comm.Recv(a_11, source=0, tag=21)
        comm.Recv(a_12, source=0, tag=22)
        comm.Recv(b_22, source=0, tag=23)
        a_11 = a_11.reshape(subSize, subSize)
        a_12 = a_12.reshape(subSize, subSize)
        b_22 = b_22.reshape(subSize, subSize)
        
        m5 = (a_11 + a_12).dot(b_22)
        m5 = numpy.ascontiguousarray(m5)

        # Sending m5 to ranks to calculate portions of C
        comm.Send(m5, dest=0, tag=36)
        comm.Send(m5, dest=2, tag=38)
        return None
        
    if rank == 5:
        
        a_11 = numpy.arange((subSize * subSize))
        a_21 = numpy.arange((subSize * subSize))
        b_11 = numpy.arange((subSize * subSize))
        b_12 = numpy.arange((subSize * subSize))
        comm.Recv(a_11, source=0, tag=24)
        comm.Recv(a_21, source=0, tag=25)
        comm.Recv(b_11, source=0, tag=26)
        comm.Recv(b_12, source=0, tag=27)
        a_11 = a_11.reshape(subSize, subSize)
        a_21 = a_21.reshape(subSize, subSize)
        b_11 = b_11.reshape(subSize, subSize)
        b_12 = b_12.reshape(subSize, subSize)

        m6 = (a_21 - a_11).dot(b_11 + b_12)
        
        # receiving m3, m1, m2 to calculate c22
        m3 = numpy.arange((subSize * subSize))
        comm.Recv(m3, source=0, tag=32)
        m1 = numpy.arange((subSize * subSize))
        comm.Recv(m1, source=2 , tag=33)
        m2 = numpy.arange((subSize * subSize))
        comm.Recv(m2, source=1, tag=34)
        
        m3 = m3.reshape(subSize, subSize)
        m1 = m1.reshape(subSize, subSize)
        m2 = m2.reshape(subSize, subSize)
        
        #calculate C22
        C22 = m1 -m2 + m3 + m6
        C22 = numpy.ascontiguousarray(C22)

        comm.Send(C22, dest=0, tag=41)
        return None
         
    if rank == 6:
        
        a_12 = numpy.arange((subSize * subSize))
        a_22 = numpy.arange((subSize * subSize))
        b_21 = numpy.arange((subSize * subSize))
        b_22 = numpy.arange((subSize * subSize))
        comm.Recv(a_12, source=0, tag=28)
        comm.Recv(a_22, source=0, tag=29)
        comm.Recv(b_21, source=0, tag=30)
        comm.Recv(b_22, source=0, tag=31)
        a_12 = a_12.reshape(subSize, subSize)
        a_22 = a_22.reshape(subSize, subSize)
        b_21 = b_21.reshape(subSize, subSize)
        b_22 = b_22.reshape(subSize, subSize)
        
        m7 = (a_12 - a_22).dot((b_21 + b_22))
        m7 = numpy.ascontiguousarray(m7)
        
        comm.Send(m7, dest=2, tag=39)
        return None


def get_dim(matrix):
    """Function to get the dim of a matrix and return. Assumes the matricies are
    already square. Returns an integer for the dim of the matrix"""
    return int((str(matrix.shape).split(',')[0].replace('(','')))

# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
