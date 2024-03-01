/**
 *
 *
 *
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "mttkrp.h"

/* Array initialization. */
static void init_array(int nr, int nq, int np,
                       DATA_TYPE POLYBENCH_2D(A, NP, NP, np, np),
                       DATA_TYPE POLYBENCH_2D(B, NP, NP, np, np),
                       DATA_TYPE POLYBENCH_2D(C, NP, NP, np, np),
                       DATA_TYPE POLYBENCH_3D(T, NR, NQ, NP, nr, nq, np)) {
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) T[i][j][k] = ((DATA_TYPE)i * j + k) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) A[i][j] = ((DATA_TYPE)i * j) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) B[i][j] = ((DATA_TYPE)i * j) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) C[i][j] = ((DATA_TYPE)i * j) / np;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nr, int nq, int np,
                        DATA_TYPE POLYBENCH_2D(A, NP, NP, np, np)) {
  int i, j, k;

  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if (i % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_doitgen(int nr, int nq, int np,
                           DATA_TYPE POLYBENCH_2D(A, NP, NP, np, np),
                           DATA_TYPE POLYBENCH_2D(B, NP, NP, np, np),
                           DATA_TYPE POLYBENCH_2D(C, NP, NP, np, np),
                           DATA_TYPE POLYBENCH_3D(T, NP, NP, NP, np, np, np)) {
  int i, j, k, r;

#pragma scop
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      for (k = 0; k < np; k++)
        for (r = 0; r < np; r++) A[i][r] += T[i][j][k] * B[j][r] * C[k][r];
#pragma endscop
}

int main(int argc, char** argv) {
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(T, DATA_TYPE, NR, NQ, NP, np, np, np);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NP, NP, np, np);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NP, NP, np, np);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NP, NP, np, np);

  /* Initialize array(s). */
  init_array(np, np, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(T));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen(np, np, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
                 POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(T));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(np, nq, np, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(T);

  return 0;
}
