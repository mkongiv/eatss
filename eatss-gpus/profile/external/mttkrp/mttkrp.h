/**
 *
 *
 *
 */
#ifndef MTTKRP_H
#define MTTKRP_H

/* Default to STANDARD_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && \
    !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#define STANDARD_DATASET
#endif

/* Do not define anything if the user manually defines the size. */
#if !defined(NQ) && !defined(NR) && !defined(NP)
/* Define the possible dataset sizes. */
#ifdef MINI_DATASET
#define NQ 10
#define NR 10
#define NP 10
#endif

#ifdef SMALL_DATASET
#define NQ 128
#define NR 128
#define NP 128
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
#define NQ 384
#define NR 384
#define NP 384
#endif

#ifdef LARGE_DATASET
#define NQ 384
#define NR 384
#define NP 384
#endif

#ifdef EXTRALARGE_DATASET
#define NQ 384
#define NR 384
#define NP 384
#endif
#endif /* !N */

#define _PB_NQ POLYBENCH_LOOP_BOUND(NQ, nq)
#define _PB_NR POLYBENCH_LOOP_BOUND(NR, nr)
#define _PB_NP POLYBENCH_LOOP_BOUND(NP, np)

#ifndef DATA_TYPE
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

#endif /* !DOITGEN */
