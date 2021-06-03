/*
Copyright 2018 Embedded Microprocessor Benchmark Consortium (EEMBC)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original Author: Shay Gal-on
*/

#include "coremark.h"
/*
Topic: Description
        Matrix manipulation benchmark

        This very simple algorithm forms the basis of many more complex
algorithms.

        The tight inner loop is the focus of many optimizations (compiler as
well as hardware based) and is thus relevant for embedded processing.

        The total available data space will be divided to 3 parts:
        NxN Matrix A - initialized with small values (upper 3/4 of the bits all
zero). NxN Matrix B - initialized with medium values (upper half of the bits all
zero). NxN Matrix C - used for the result.

        The actual values for A and B must be derived based on input that is not
available at compile time.
*/
ee_s16 matrix_test(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B, MATDAT val);
ee_s16 matrix_sum(ee_u32 N, MATRES *C, MATDAT clipval);
void   matrix_mul_const(ee_u32 N, MATRES *C, MATDAT *A, MATDAT val);
void   matrix_mul_vect(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B);
void   matrix_mul_matrix(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B);
void   matrix_mul_matrix_bitextract(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B);
void   matrix_add_const(ee_u32 N, MATDAT *A, MATDAT val);

#define matrix_test_next(x)      (x + 1)
#define matrix_clip(x, y)        ((y) ? (x)&0x0ff : (x)&0x0ffff)
#define matrix_big(x)            (0xf000 | (x))
#define bit_extract(x, from, to) (((x) >> (from)) & (~(0xffffffff << (to))))

#if CORE_DEBUG
void
printmat(MATDAT *A, ee_u32 N, char *name)
{
    ee_u32 i, j;
    ee_printf("Matrix %s [%dx%d]:\n", name, N, N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (j != 0)
                ee_printf(",");
            ee_printf("%d", A[i * N + j]);
        }
        ee_printf("\n");
    }
}
void
printmatC(MATRES *C, ee_u32 N, char *name)
{
    ee_u32 i, j;
    ee_printf("Matrix %s [%dx%d]:\n", name, N, N);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            if (j != 0)
                ee_printf(",");
            ee_printf("%d", C[i * N + j]);
        }
        ee_printf("\n");
    }
}
#endif
/* Function: core_bench_matrix
        Benchmark function

        Iterate <matrix_test> N times,
        changing the matrix values slightly by a constant amount each time.
*/
ee_u16
core_bench_matrix(mat_params *p, ee_s16 seed, ee_u16 crc)
{
    ee_u32  N   = p->N;
    MATRES *C   = p->C;
    MATDAT *A   = p->A;
    MATDAT *B   = p->B;
    MATDAT  val = (MATDAT)seed;

    crc = crc16(matrix_test(N, C, A, B, val), crc);

    return crc;
}

/* Function: matrix_test
        Perform matrix manipulation.

        Parameters:
        N - Dimensions of the matrix.
        C - memory for result matrix.
        A - input matrix
        B - operator matrix (not changed during operations)

        Returns:
        A CRC value that captures all results calculated in the function.
        In particular, crc of the value calculated on the result matrix
        after each step by <matrix_sum>.

        Operation:

        1 - Add a constant value to all elements of a matrix.
        2 - Multiply a matrix by a constant.
        3 - Multiply a matrix by a vector.
        4 - Multiply a matrix by a matrix.
        5 - Add a constant value to all elements of a matrix.

        After the last step, matrix A is back to original contents.
*/
ee_s16
matrix_test(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B, MATDAT val)
{
    ee_u16 crc     = 0;
    MATDAT clipval = matrix_big(val);

    matrix_add_const(N, A, val); /* make sure data changes  */
#if CORE_DEBUG
    printmat(A, N, "matrix_add_const");
#endif
    matrix_mul_const(N, C, A, val);
    crc = crc16(matrix_sum(N, C, clipval), crc);
#if CORE_DEBUG
    printmatC(C, N, "matrix_mul_const");
#endif
    matrix_mul_vect(N, C, A, B);
    crc = crc16(matrix_sum(N, C, clipval), crc);
#if CORE_DEBUG
    printmatC(C, N, "matrix_mul_vect");
#endif
    matrix_mul_matrix(N, C, A, B);
    crc = crc16(matrix_sum(N, C, clipval), crc);
#if CORE_DEBUG
    printmatC(C, N, "matrix_mul_matrix");
#endif
    matrix_mul_matrix_bitextract(N, C, A, B);
    crc = crc16(matrix_sum(N, C, clipval), crc);
#if CORE_DEBUG
    printmatC(C, N, "matrix_mul_matrix_bitextract");
#endif

    matrix_add_const(N, A, -val); /* return matrix to initial value */
    return crc;
}

/* Function : matrix_init
        Initialize the memory block for matrix benchmarking.

        Parameters:
        blksize - Size of memory to be initialized.
        memblk - Pointer to memory block.
        seed - Actual values chosen depend on the seed parameter.
        p - pointers to <mat_params> containing initialized matrixes.

        Returns:
        Matrix dimensions.

        Note:
        The seed parameter MUST be supplied from a source that cannot be
   determined at compile time
*/
ee_u32
core_init_matrix(ee_u32 blksize, void *memblk, ee_s32 seed, mat_params *p)
{
    ee_u32  N = 0;
    MATDAT *A;
    MATDAT *B;
    ee_s32  order = 1;
    MATDAT  val;
    ee_u32  i = 0, j = 0;
    if (seed == 0)
        seed = 1;
    while (j < blksize)
    {
        i++;
        j = i * i * 2 * 4;
    }
    N = i - 1;
    A = (MATDAT *)align_mem(memblk);
    B = A + N * N;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            seed         = ((order * seed) % 65536);
            val          = (seed + order);
            val          = matrix_clip(val, 0);
            B[i * N + j] = val;
            val          = (val + order);
            val          = matrix_clip(val, 1);
            A[i * N + j] = val;
            order++;
        }
    }

    p->A = A;
    p->B = B;
    p->C = (MATRES *)align_mem(B + N * N);
    p->N = N;
#if CORE_DEBUG
    printmat(A, N, "A");
    printmat(B, N, "B");
#endif
    return N;
}

/* Function: matrix_sum
        Calculate a function that depends on the values of elements in the
   matrix.

        For each element, accumulate into a temporary variable.

        As long as this value is under the parameter clipval,
        add 1 to the result if the element is bigger then the previous.

        Otherwise, reset the accumulator and add 10 to the result.
*/
ee_s16
matrix_sum(ee_u32 N, MATRES *C, MATDAT clipval)
{
    MATRES tmp = 0, prev = 0, cur = 0;
    ee_s16 ret = 0;
    ee_u32 i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            cur = C[i * N + j];
            tmp += cur;
            if (tmp > clipval)
            {
                ret += 10;
                tmp = 0;
            }
            else
            {
                ret += (cur > prev) ? 1 : 0;
            }
            prev = cur;
        }
    }
    return ret;
}

/* Function: matrix_mul_const
        Multiply a matrix by a constant.
        This could be used as a scaler for instance.
*/
void
matrix_mul_const(ee_u32 N, MATRES *C, MATDAT *A, MATDAT val)
{
#ifdef RVV
    ee_u32 M = N*N;
    int actually_vl = 0;

    while(M > 0)
    {
        asm(
            "vsetvli        %[actually_vl], %[M], e16, m4\n"
            "vle16.v        v8, (%[A])\n"
            "vwmul.vx       v16, v8, %[val]\n"
            "vse32.v        v16, (%[C])\n"
            : [actually_vl]"+r"(actually_vl)
            : [A]"r"(A), [C]"r"(C), [val]"r"(val), [M]"r"(M)
            :"a0"
        );

        A += actually_vl;
        C += actually_vl;
        M -= actually_vl;
   }
#else
    ee_u32 i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            C[i * N + j] = (MATRES)A[i * N + j] * (MATRES)val;
        }
    }
#endif
}

/* Function: matrix_add_const
        Add a constant value to all elements of a matrix.
*/
void
matrix_add_const(ee_u32 N, MATDAT *A, MATDAT val)
{
#ifdef RVV
    ee_u32 M = N*N;
    int actually_vl = 0;
    while(M > 0)
    {
        asm(
            "vsetvli        %[actually_vl], %[M], e16, m8\n"
            "vle16.v        v8, (%[A])\n"
            "vadd.vx        v16, v8, %[val]\n"
            "vse16.v        v16, (%[A])\n"
            : [actually_vl]"+r"(actually_vl)
            : [A]"r"(A), [val]"r"(val), [M]"r"(M)
            :"a0"
            );

        A += actually_vl;
        M -= actually_vl;
    }
#else
    ee_u32 i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] += val;
        }
    }
#endif
}

/* Function: matrix_mul_vect
        Multiply a matrix by a vector.
        This is common in many simple filters (e.g. fir where a vector of
   coefficients is applied to the matrix.)
*/
void
matrix_mul_vect(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B)
{
#ifdef RVV
    while(N--)
    {
        asm(
            "li         a0, 9\n"
            "vsetvli    t0, a0, e16, m2\n"
            "vle16.v    v8, (%[A])\n"
            "vle16.v    v10, (%[B])\n"
            "vwmul.vv   v12, v8, v10\n"
            "vsetvli    t0, a0, e32, m4\n"
            "vmv.v.i    v16, 0\n"
            "vredsum.vs v20, v12, v16\n"
            "li         a0, 1\n"
            "vsetvli    t0, a0, e32, m1\n"
            "vse32.v    v20, (%[C])\n"
            :
            : [A]"r"(A), [C]"r"(C), [B]"r"(B)
            :"a0", "t0"
        );

        A += 9;
        C += 1;
    }
#else
    ee_u32 i, j;
    for (i = 0; i < N; i++)
    {
        C[i] = 0;
        for (j = 0; j < N; j++)
        {
            C[i] += (MATRES)A[i * N + j] * (MATRES)B[j];
        }
    }
#endif
}

/* Function: matrix_mul_matrix
        Multiply a matrix by a matrix.
        Basic code is used in many algorithms, mostly with minor changes such as
   scaling.
*/
void
matrix_mul_matrix(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B)
{
#ifdef RVV
    ee_u32 Max = N;
    ee_u32 M;
    while(N--) {
        M = Max;
        while(M--) {
        asm (
            "li         a0, 9\n"
            "li         a1, 18\n"
            "vsetvli    t0, a0, e16, m2\n"
            "vle16.v    v8, (%[A])\n"
            "vlse16.v   v10, (%[B]), a1\n"
            "vwmul.vv   v12, v8, v10\n"
            "vsetvli    t0, a0, e32, m4\n"
            "vmv.v.i    v16, 0\n"
            "vredsum.vs v20, v12, v16\n"
            "li         a0, 1\n"
            "vsetvli    t0, a0, e32, m1\n"
            "vse32.v    v20, (%[C])\n"
            :
            : [A]"r"(A), [C]"r"(C), [B]"r"(B)
            :"a0", "t0", "a1"
            );
            C += 1;
            B += 1;
        }
        A += 9;
        B -= 9;
    }
#else
    ee_u32 i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            C[i * N + j] = 0;
            for (k = 0; k < N; k++)
            {
                C[i * N + j] += (MATRES)A[i * N + k] * (MATRES)B[k * N + j];
            }
        }
    }
#endif
}

/* Function: matrix_mul_matrix_bitextract
        Multiply a matrix by a matrix, and extract some bits from the result.
        Basic code is used in many algorithms, mostly with minor changes such as
   scaling.
*/
void
matrix_mul_matrix_bitextract(ee_u32 N, MATRES *C, MATDAT *A, MATDAT *B)
{
#ifdef RVV
    ee_u32 Max = N;
    ee_u32 M;
    ee_u32 NN = N*N;
    MATRES tmp[NN];
    MATRES *tmp_p = tmp;
    int actually_vl = 0;

    while(N--)
    {
        M = Max;
        while(M--)
        {
            asm (
                "li         a0, 9\n"
                "li         a1, 18\n"
                "vsetvli    t0, a0, e16, m2\n"
                "vle16.v    v8, (%[A])\n"
                "vlse16.v   v10, (%[B]), a1\n"
                "vwmul.vv   v12, v8, v10\n"
                "vsetvli    t0, a0, e32, m4\n"
                "vmv.v.i    v16, 0\n"
                "vredsum.vs v20, v12, v16\n"
                "li         a0, 1\n"
                "vsetvli    t0, a0, e32, m1\n"
                "vse32.v    v20, (%[tmp])\n"
                :
                : [A]"r"(A), [tmp]"r"(tmp_p), [B]"r"(B)
                :"a0", "t0", "a1", "a2"
            );

            tmp_p += 1;
            B += 1;
        }

        A += 9;
        B -= 9;
    }

    tmp_p -= 81;

    while(NN > 0) {
        asm (
                "vsetvli    %[actually_vl], %[NN], e32, m8\n"
                "vle32.v    v8, (%[tmp])\n"
                "vsra.vi    v16, v8, 2\n"
                "vand.vi    v24, v16, 0xf\n"
                "vsra.vi    v16, v8, 5\n"
                "li         a2, 0x7f\n"
                "vand.vx    v8, v16, a2\n"
                "vmul.vv    v16, v8, v24\n"
                "vse32.v    v16, (%[C])\n"
                : [actually_vl]"+r"(actually_vl)
                : [C]"r"(C), [tmp]"r"(tmp_p), [NN]"r"(NN)
                : "a2"
            );

            tmp_p += actually_vl;
            C += actually_vl;
            NN -= actually_vl;
    }
#else
    ee_u32 i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            C[i * N + j] = 0;
            for (k = 0; k < N; k++)
            {
                MATRES tmp = (MATRES)A[i * N + k] * (MATRES)B[k * N + j];
                C[i * N + j] += bit_extract(tmp, 2, 4) * bit_extract(tmp, 5, 7);
            }
        }
    }
#endif
}
