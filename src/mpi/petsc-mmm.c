/*
 * petsc-mmm.c
 * Copyright (c) 2016 pargue
 *  Created on: Nov 7, 2016
 *      Author: arguepr
 */


#include <petscmat.h>
#include <petscksp.h>
#include <petscsys.h>
#include <petscviewer.h>
#include <petsctime.h>

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
    PetscScalar    *vals;
    PetscScalar    *zvals;
    PetscInt       *idxn;
    PetscLogDouble v1,v2,elapsed_time;
    int rank;
#if defined(PETSC_USE_LOG)
    PetscLogEvent  MAT_GENERATE,MAT_MULT;
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
        PetscMalloc(my_mats.nx * sizeof(vals[0]), &vals);
        PetscMalloc(my_mats.nx * sizeof(idxn[0]), &idxn);
        PetscMalloc(my_mats.nx * sizeof(vals[0]), &zvals);
        int i;
//        int j;
        int x;
        for (x=0; x<my_mats.nx; x++) {
            vals[x] = 1.0;
            zvals[x] = 0.0;
            idxn[x]=x;
        }
        for (i=0; i<my_mats.ny; i++) {
//            MatSetValues(Mat mat,PetscInt m,const PetscInt idxm[],PetscInt n,const PetscInt idxn[],const PetscScalar v[],InsertMode addv)
            MatSetValues(my_mats.A /*mat*/, 1/*m*/, &i /*idxm[]*/, my_mats.nx /*n*/, idxn, vals /*v[]*/, INSERT_VALUES);
            MatSetValues(my_mats.B /*mat*/, 1/*m*/, &i /*idxm[]*/, my_mats.nx /*n*/, idxn, zvals /*v[]*/, INSERT_VALUES);
        }

    }
    MatAssemblyBegin(my_mats.A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(my_mats.A,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(my_mats.B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(my_mats.B,MAT_FINAL_ASSEMBLY);
#if defined(PETSC_USE_LOG)
    PetscLogEventEnd(MAT_GENERATE,0,0,0,0);
#endif

    if (isViewMat)
    {
        MatView(my_mats.A, PETSC_VIEWER_DEFAULT);
        MatView(my_mats.B, PETSC_VIEWER_DEFAULT);
    }

#if defined(PETSC_USE_LOG)
    PetscLogEventRegister("Matrix Mult",PETSC_VIEWER_CLASSID,&MAT_MULT);
#endif

    int idx;
    for (idx=0; idx<10; idx++) {
#if defined(PETSC_USE_LOG)
        PetscLogEventBegin(MAT_MULT,0,0,0,0);
#endif
        PetscGetCPUTime(&v1);
        pm__MatMatMult(&my_mats);
        PetscGetCPUTime(&v2);
        elapsed_time = v2 - v1;
        PetscPrintf(PETSC_COMM_WORLD,"run %d elsapsed_time=%f\n",idx,elapsed_time);
#if defined(PETSC_USE_LOG)
        PetscLogEventEnd(MAT_MULT,0,0,0,0);
#endif
    }
    if (isViewMat)
    {
        MatView(my_mats.C, PETSC_VIEWER_DEFAULT);
    }
    MatDestroy(&my_mats.A);
    MatDestroy(&my_mats.B);
    MatDestroy(&my_mats.C);

    if (rank==0) {
        PetscFree(vals);
        PetscFree(zvals);
        PetscFree(idxn);
    }

    PetscFinalize();
    return 0;
}
