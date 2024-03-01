/**
 * conv-2d.h: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#ifndef CONV2D_H
#define CONV2D_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && \
    !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(NC) && !defined(NH) && !defined(NW) && !defined(NKW) && \
    !defined(NKH) && !defined(NOH) && !defined(NOW)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define NC 224
#define NH 224
#define NW 224
#define NKH 3
#define NKW 3
#define NOH 222
#define NOW 222
#endif

#ifdef SMALL_DATASET
#define NC 224
#define NH 224
#define NW 224
#define NKH 3
#define NKW 3
#define NOH 222
#define NOW 222
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define NC 224
#define NH 224
#define NW 224
#define NKH 3
#define NKW 3
#define NOH 222
#define NOW 222
#endif

#ifdef LARGE_DATASET
#define NC 224
#define NH 224
#define NW 224
#define NKH 3
#define NKW 3
#define NOH 222
#define NOW 222
#endif

#ifdef EXTRALARGE_DATASET
#define NC 224
#define NH 224
#define NW 224
#define NKH 3
#define NKW 3
#define NOH 222
#define NOW 222
#endif
#endif /* !N */

#define _PB_NI POLYBENCH_LOOP_BOUND(NI, ni)
#define _PB_NJ POLYBENCH_LOOP_BOUND(NJ, nj)
#define _PB_NK POLYBENCH_LOOP_BOUND(NK, nk)

#ifndef DATA_TYPE
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

#endif /* !GEMM */
