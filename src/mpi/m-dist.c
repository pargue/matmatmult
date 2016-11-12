/*
 * Recursive matrix mult by strassen's method and MPI.
 * adapted from https://sites.google.com/site/algorithms2013/home/recursive-matrix-operations/strassen-matrix-multiplication
 * 2013-Feb-15 Fri 11:47 by moshahmed/at/gmail.
 *
 */


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MFMT "%4.2g"
#define MAT(NAME) #NAME, NAME
#define LEN(A) (sizeof(A)/sizeof(A[0]))
typedef double elem;

typedef elem **mat;

#define MORDEREXP 2
#define N (1<<MORDEREXP)   // 2^M
typedef struct { int ra, rb, ca, cb; } corners; // for tracking rows and columns.

// set A[a] = k
void set(mat A, corners a, elem k)
{
  int i,j;
  for(i=a.ra;i<a.rb;i++)
    for(j=a.ca;j<a.cb;j++)
      A[i][j] = k;
}

int randm(mat A, elem x, elem y)
{
    int i,j;
    for (i = 0; i<N; i++)
    {
        for (j = 0; j<N; j++)
        {
            A[i][j] = (elem) (x + (y-x) * (rand()/(elem)RAND_MAX));;
        }
    }
    return 0;
}

int printm(char* name, mat A)
{
    int i,j;
    printf("mat %s = \n", name);
    for (i = 0; i<N; i++)
    {
        for (j = 0; j<N; j++)
        {
            printf(MFMT " ", A[i][j]);
        }
        printf("\n");
    }
    return 0;
}

mat allocm(int rows, int cols) {
    elem *data = (elem *)malloc(rows*cols*sizeof(elem));
    if (NULL == data)
        return NULL;
    mat array = (mat)malloc(rows*sizeof(elem*));
    if (NULL == array)
    {
        free(data);
        return NULL;
    }
    int i;
    for (i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

void freem(mat m)
{
    if (NULL == m)
        return;
    free(m[0]);
    m[0] = NULL;
    free(m);
    m = NULL;
}

// C[c] = A[a] + B[b]
void add(mat A, mat B, mat C, corners a, corners b, corners c)
{
    int rd = a.rb - a.ra;
    int cd = a.cb - a.ca;
    int i,j;
    for(i = 0; i<rd; i++ ){
        for(j = 0; j<cd; j++ ){
            C[i+c.ra][j+c.ca] = A[i+a.ra][j+a.ca] + B[i+b.ra][j+b.ca];
        }
    }
}

// C[c] = A[a] - B[b]
void  sub(mat A, mat B, mat C, corners a, corners b, corners c)
{
    int rd = a.rb - a.ra;
    int cd = a.cb - a.ca;
    int i,j;
    for(i = 0; i<rd; i++ ){
        for(j = 0; j<cd; j++ ){
            C[i+c.ra][j+c.ca] = A[i+a.ra][j+a.ca] - B[i+b.ra][j+b.ca];
        }
    }
}

// Return 1/4 of the matrix: top/bottom , left/right.
void find_corner(corners a, int i, int j, corners *b)
 {
    int rm = a.ra + (a.rb - a.ra)/2 ;
    int cm = a.ca + (a.cb - a.ca)/2 ;
    *b = a;
    if (i==0)  b->rb = rm;     // top rows
    else       b->ra = rm;     // bot rows
    if (j==0)  b->cb = cm;     // left cols
    else       b->ca = cm;     // right cols
}

// Multiply: A[a] * B[b] => C[c], recursively.
void mul(mat A, mat B, mat C, corners a, corners b, corners c)
{
    corners aii[2][2], bii[2][2], cii[2][2], p;
    mat P[7], S = NULL, T = NULL;
    int i, j, m, n, k;

    int my_rank, comm_size;
    int msgtag = 0;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Check: A[m n] * B[n k] = C[m k]
    m = a.rb - a.ra; assert(m == (c.rb-c.ra));
    n = a.cb - a.ca; assert(n == (b.rb-b.ra));
    k = b.cb - b.ca; assert(k == (c.cb-c.ca));
    assert(m > 0);

    if (1 == n)
    {
        if (0 == my_rank)
            C[c.ra][c.ca] += A[a.ra][a.ca] * B[b.ra][b.ca];
        return;
    }

    // Create the 12 smaller matrix indexes:
    //  A00 A01   B00 B01   C00 C01
    //  A10 A11   B10 B11   C10 C11
    for(i=0;i<2;i++)
    {
        for(j=0;j<2;j++)
        {
            find_corner(a, i, j, &aii[i][j]);
            find_corner(b, i, j, &bii[i][j]);
            find_corner(c, i, j, &cii[i][j]);
        }
    }

    p.ra = p.ca = 0;
    p.rb = p.cb = m/2;

    for(i=0; i < LEN(P); i++)
    {
        if (0 == my_rank || i == my_rank)
        {
            P[i] = NULL;
            P[i] = allocm(N, N);
            if (NULL == P[i])
                goto freemats;
            set(P[i], p, 0);
        }
    }
    S = allocm(N, N);
    if (NULL == S)
        goto freemats;
    T = allocm(N, N);
    if (NULL == T)
        goto freemats;

    #define ST0 set(S,p,0); set(T,p,0)

    switch(my_rank)
    {
    case 0:
        // (A00 + A11) * (B00+B11) = S * T = P0
        ST0;
        add( A, A, S, aii[0][0], aii[1][1], p);
        add( B, B, T, bii[0][0], bii[1][1], p);
        mul( S, T, P[0], p, p, p);
        for (i=1; i<LEN(P); i++)
        MPI_Recv(&(P[i]), N*N, MPI_DOUBLE, MPI_ANY_SOURCE, msgtag, MPI_COMM_WORLD, &status);
        break;

    case 1:
        // (A10 + A11) * B00 = S * B00 = P1
        ST0;
        add( A, A, S, aii[1][0], aii[1][1], p);
        mul( S, B, P[1], p, bii[0][0], p);
        MPI_Send(&(P[1]), N*N, MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
        break;

    case 2:
        // A00 * (B01 - B11) = A00 * T = P2
        ST0;
        sub( B, B, T, bii[0][1], bii[1][1], p);
        mul( A, T, P[2], aii[0][0], p, p);
        MPI_Send(&(P[2]), N*N, MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
        break;

    case 3:
        // A11 * (B10 - B00) = A11 * T = P3
        ST0;
        sub(B, B, T, bii[1][0], bii[0][0], p);
        mul(A, T, P[3], aii[1][1], p, p);
        MPI_Send(&(P[3]), N*N, MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
        break;

    case 4:
        // (A00 + A01) * B11 = S * B11 = P4
        ST0;
        add(A, A, S, aii[0][0], aii[0][1], p);
        mul(S, B, P[4], p, bii[1][1], p);
        MPI_Send(&(P[4]), N*N, MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
        break;

    case 5:
        // (A10 - A00) * (B00 + B01) = S * T = P5
        ST0;
        sub(A, A, S, aii[1][0], aii[0][0], p);
        add(B, B, T, bii[0][0], bii[0][1], p);
        mul(S, T, P[5], p, p, p);
        MPI_Send(&(P[5]), N*N, MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
        break;

    case 6:
        // (A01 - A11) * (B10 + B11) = S * T = P6
        ST0;
        sub(A, A, S, aii[0][1], aii[1][1], p);
        add(B, B, T, bii[1][0], bii[1][1], p);
        mul(S, T, P[6], p, p, p);
        MPI_Send(&(P[6]), N*N, MPI_DOUBLE, 0, msgtag, MPI_COMM_WORLD);
        break;
    }

    if (0 == my_rank)
    {
        // P0 + P3 - P4 + P6 = S - P4 + P6 = T + P6 = C00
        add(P[0], P[3], S, p, p, p);
        sub(S, P[4], T, p, p, p);
        add(T, P[6], C, p, p, cii[0][0]);

        // P2 + P4 = C01
        add(P[2], P[4], C, p, p, cii[0][1]);

        // P1 + P3 = C10
        add(P[1], P[3], C, p, p, cii[1][0]);

        // P0 + P2 - P1 + P5 = S - P1 + P5 = T + P5 = C11
        add(P[0], P[2], S, p, p, p);
        sub(S, P[1], T, p, p, p);
        add(T, P[5], C, p, p, cii[1][1]);
    }

freemats:
    for(i=0; i < LEN(P); i++)
    {
        freem(P[i]);
    }
    freem(S);
    freem(T);

}


int main(int argc, char **argv){

    int my_rank,comm_size;

    mat A = NULL;
    mat B = NULL;
    mat C = NULL;
    A = allocm(N,N);
    if (NULL == A)
        goto freemats;
    B = allocm(N,N);
    if (NULL == B)
        goto freemats;
    C = allocm(N,N);
    if (NULL == C)
        goto freemats;

    corners ai = {0,N,0,N};
    corners bi = {0,N,0,N};
    corners ci = {0,N,0,N};

    srand(time(0));

    if(MPI_Init(&argc,&argv) != MPI_SUCCESS){
        printf("MPI-INIT Failed\n");
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if(0 == my_rank)
    {
        randm(A, 0, 10);
        randm(B, 0, 10);
        printm(MAT(A));
        printm(MAT(B));
    }
    MPI_Bcast(&(A[0][0]), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(B[0][0]), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (0 == my_rank)
    {
        set(C,ci,0);
    }
    // add(A,B,C, ai, bi, ci);
    mul(A,B,C, ai, bi, ci);

    if(0 == my_rank)
    {
//        printm(MAT(C), ci, "C");
        printm(MAT(C));
    }
    else
    {
        printf("my_rank %d\n", my_rank);
        printm(MAT(A));
        printm(MAT(B));
    }

freemats:
    freem(C);
    freem(B);
    freem(A);

    MPI_Finalize();

    return 0;
}
