/**
 * conv-2d
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
#include "conv-2d.h"

/* Array initialization. */
static void init_array(int nc, int nh, int nw, int noh, int now, int nkh,
                       int nkw,
                       DATA_TYPE POLYBENCH_3D(I, NC, NH, NW, nc, nh, nw),
                       DATA_TYPE POLYBENCH_3D(O, NC, NOH, NOW, nc, noh, now),
                       DATA_TYPE POLYBENCH_3D(K, NC, NKH, NKW, nc, nkh, nkw)) {
  int c, h, w;

  for (c = 0; c < nc; c++)
    for (h = 0; h < nh; h++)
      for (w = 0; w < nw; w++) {
        I[c][h][w] = ((DATA_TYPE)c * nh * nw + h * nw + w);
      }
  for (c = 0; c < nc; c++)
    for (h = 0; h < noh; h++)
      for (w = 0; w < now; w++) {
        O[c][h][w] = ((DATA_TYPE)0);
      }
  for (c = 0; c < nc; c++)
    for (h = 0; h < nkh; h++)
      for (w = 0; w < nkw; w++) {
        K[c][h][w] = ((DATA_TYPE)c * nkh * nkw + h * nkw + w);
      }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nc, int noh, int now,
                        DATA_TYPE POLYBENCH_3D(O, NC, NOH, NOW, nc, noh, now)) {
  int h, w, c;

  for (c = 0; c < nc; c++) {
    for (h = 0; h < noh; h++) {
      for (w = 0; w < now; w++) {
        fprintf(stderr, DATA_PRINTF_MODIFIER, O[c][h][w]);
        if ((c * noh * now + h * now + w) % 20 == 0) fprintf(stderr, "\n");
      }
    }
  }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_conv2d(int nc, int nh, int nw, int noh, int now, int nkh,
                          int nkw,
                          DATA_TYPE POLYBENCH_3D(I, NC, NH, NW, nc, nh, nw),
                          DATA_TYPE POLYBENCH_3D(O, NC, NOH, NOW, nc, nh, nw),
                          DATA_TYPE POLYBENCH_3D(K, NC, NKH, NKW, nc, nkh,
                                                 nkw)) {
  int c, h, w, kh, kw;

// assume padding = 0 and stride = 1
// batch size = 1
#pragma scop
  for (c = 0; c < nc; c++)
    for (h = 0; h < nh - nkh + 1; h++)
      for (w = 0; w < nw - nkw + 1; w++)
        for (kh = 0; kh < nkh; kh++)
          for (kw = 0; kw < nkw; kw++) {
            DATA_TYPE tmp = I[c][h + kh][w + kw] * K[c][kh][kw];
            O[c][h][w] = O[c][h][w] + tmp;
          }

#pragma endscop
}

int main(int argc, char **argv) {
  /* Retrieve problem size. */
  int nc = NC;
  int nh = NH;
  int nw = NW;
  int noh = NOH;
  int now = NOW;
  int nkh = NKH;
  int nkw = NKW;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(K, DATA_TYPE, NC, NKH, NKW, nc, nkh, nkw);
  POLYBENCH_3D_ARRAY_DECL(I, DATA_TYPE, NC, NH, NW, nc, nh, nw);
  POLYBENCH_3D_ARRAY_DECL(O, DATA_TYPE, NC, NOH, NOW, nc, noh, now);

  /* Initialize array(s). */
  init_array(nc, nh, nw, noh, now, nkh, nkw, POLYBENCH_ARRAY(I),
             POLYBENCH_ARRAY(O), POLYBENCH_ARRAY(K));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_conv2d(nc, nh, nw, noh, now, nkh, nkw, POLYBENCH_ARRAY(I),
                POLYBENCH_ARRAY(O), POLYBENCH_ARRAY(K));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nc, noh, now, POLYBENCH_ARRAY(O)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(I);
  POLYBENCH_FREE_ARRAY(O);
  POLYBENCH_FREE_ARRAY(K);

  return 0;
}
