#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
using namespace std;

int GetNextInt(ifstream &fin, char input[]);
void GetMatrixInts(ifstream &fin, vector<vector<int>>, char file[], int dim);

int main()
{
    char firstInput[] = "Matrix1.txt";        // source file containing 1st matrix to multiply
    char secondInput[] = "Matrix2.txt";       // source file containing 2nd matrix to multiply
    vector<vector<int>> firstMatrix;          // first matrix to multiply
    int firstDim;                             // number of rows/cols in first matrix
    vector<vector<int>> secondMatrix;         // second matrix to multiply
    int secondDim;                            // number of rows/cols in second matrix
    vector<vector<int>> resultMatrix;         // matrix holding result of multiplication       
    time_t start;                             // initial time when program starts
    time_t end;                               // time after output image is created
    time_t elapsedTime;                       // time for image processing

    // Open input files
    ifstream fin1(firstInput);
    ifstream fin2(secondInput);

    // Verify files opened correctly
    if (fin1.fail())
    {
        cerr << "Couldn't open " << firstInput << endl;
        return 1;
    }
    if (fin2.fail())
    {
        cerr << "Couldn't open " << secondInput << endl;
        return 1;
    }

    // Fill first input matrix from file
    GetMatrixInts(fin1, firstMatrix, firstInput, firstDim);

    // Fill second input matrix from file
    GetMatrixInts(fin2, secondMatrix, secondInput, secondDim);

    // Get dimensions of first and second matrix
    firstDim = GetNextInt(fin1, firstInput);
    secondDim = GetNextInt(fin2, firstInput);

    // Verify that the two matrices have the same dimensions
    if (firstDim != secondDim)
    {
        cerr << "Input matrices are not the same size!";
        return 1;
    }


 


    fin1.close();
    fin2.close();
    return 0;
}

// Read the next integer from file input
int GetNextInt(ifstream &fin, char input[])
{
   int value;
   fin >> value;         
   if (fin.eof())
   {
       cerr << "Error in length of " << input << endl;
       exit(EXIT_FAILURE);
   }
   return value;
}



// Fill a matrix with integers from file input
void GetMatrixInts(ifstream &fin, int **matrix, char file[], int dim)
{
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
            matrix[i][j] = GetNextInt(fin, file);
}