/*
 * petsc-mmm.c
 * Copyright (c) 2016 pargue
 *  Created on: Nov 7, 2016
 *      Author: arguepr
 */


#include <petscmat.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscsys.h>
#include <petscviewer.h>

static char help[] = "Matrix Multiplication\n";

struct mats {
    Mat A;
    Mat B;
    Mat C;
    PetscInt nx, ny;
};

PetscErrorCode pm__MatMatMult(struct mats *sm)
{
    MatMatMult(sm->A, sm->B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(sm->C));
    return 0;
}

int main(int argc, char **argv)
{
    int rank;
#if defined(PETSC_USE_LOG)
    PetscLogEvent  MAT_GENERATE;
#endif

    PetscInt size = 4;
    PetscBool isViewMat = 0;

    PetscInitialize(&argc, &argv, NULL, help);
    PetscOptionsGetInt(NULL, "-size", &size, NULL);
    PetscOptionsGetBool(NULL, "-view", &isViewMat, NULL);
    struct mats my_mats;

    // make mats
#if defined(PETSC_USE_LOG)
    PetscLogEventRegister("Generate Matrix",PETSC_VIEWER_CLASSID,&MAT_GENERATE);
    PetscLogEventBegin(MAT_GENERATE,0,0,0,0);
#endif

    my_mats.nx = my_mats.ny = size;
    MatCreate(PETSC_COMM_WORLD,&my_mats.A);
    MatSetSizes(my_mats.A,PETSC_DECIDE,PETSC_DECIDE,my_mats.nx,my_mats.ny);
    MatSetType(my_mats.A,MATMPIDENSE);
    MatSetUp(my_mats.A);

    MatCreate(PETSC_COMM_WORLD,&my_mats.B);
    MatSetSizes(my_mats.B,PETSC_DECIDE,PETSC_DECIDE,my_mats.nx,my_mats.ny);
    MatSetType(my_mats.B,MATMPIAIJ);    // the other matrix can't be dense for parallel mult
    MatSetUp(my_mats.B);

    /* assemble the matrix */
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank==0)
    {
        int i,j;
        for (i=0; i<my_mats.nx; i++) {
            for (j=0; j<my_mats.ny; j++) {
                MatSetValue(my_mats.A,i,j,1, INSERT_VALUES);
                MatSetValue(my_mats.B,i,j,1, INSERT_VALUES);
            }
        }
    }
    MatAssemblyBegin(my_mats.A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(my_mats.A,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(my_mats.B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(my_mats.B,MAT_FINAL_ASSEMBLY);
    PetscLogEventEnd(MAT_GENERATE,0,0,0,0);

    if (isViewMat)
    {
        MatView(my_mats.A, PETSC_VIEWER_DEFAULT);
        MatView(my_mats.B, PETSC_VIEWER_DEFAULT);
    }

    pm__MatMatMult(&my_mats);
    if (isViewMat)
    {
        MatView(my_mats.C, PETSC_VIEWER_DEFAULT);
    }

    MatDestroy(&my_mats.A);
    MatDestroy(&my_mats.B);
    MatDestroy(&my_mats.C);

    PetscFinalize();
    return 0;
}
