#include <cmath>
#include <cstdio>
#include <limits>

#include "nem_math.h"

// Helper macro to verify that the zmm implementation of fxn matches the scalar ref
// std::reffxn, and that the different vector sizes and scalar impl all produce identical results.
// Assumes there are: float[16] xArr and refArr, with xArr holding the inputs to test;
// __m512 lowerBound and upperBound, holding the min and max allowed ratio between the results
// of fxn and reffxn.
#define TEST_VECTOR_FUNCTION(fxn, reffxn)                                                          \
    do                                                                                             \
    {                                                                                              \
        for (int i = 0; i < 16; ++i)                                                               \
        {                                                                                          \
            refArr[i] = std::reffxn(xArr[i]);                                                      \
        }                                                                                          \
                                                                                                   \
        __m512 x = _mm512_loadu_ps(xArr);                                                          \
        __m512 y = fxn(x);                                                                         \
        __m512 ref = _mm512_loadu_ps(refArr);                                                      \
        __m512 lower = _mm512_mul_ps(ref, lowerBound);                                             \
        __m512 upper = _mm512_mul_ps(ref, upperBound);                                             \
        __m512 lo = _mm512_min_ps(lower, upper);                                                   \
        __m512 hi = _mm512_max_ps(lower, upper);                                                   \
        __mmask16 pass =                                                                           \
            _mm512_cmp_ps_mask(lo, y, _CMP_LE_OS) & _mm512_cmp_ps_mask(y, hi, _CMP_LE_OS);         \
        if (pass != 65535)                                                                         \
        {                                                                                          \
            printf("Inaccurate result from " #fxn " in vector from %.10f", xArr[0]);               \
            exit(1);                                                                               \
        }                                                                                          \
                                                                                                   \
        {                                                                                          \
            __m256 x0 = _mm512_extractf32x8_ps(x, 0);                                              \
            __m256 x1 = _mm512_extractf32x8_ps(x, 1);                                              \
            __m256 y0 = fxn(x0);                                                                   \
            __m256 y1 = fxn(x1);                                                                   \
            __m512 yTest = _mm512_broadcast_f32x8(y0);                                             \
            yTest = _mm512_insertf32x8(yTest, y1, 1);                                              \
            __mmask16 pass1 = _mm512_cmp_ps_mask(y, yTest, _CMP_EQ_OS);                            \
            if (pass1 != 65535)                                                                    \
            {                                                                                      \
                printf("Inconsistent ymm result from " #fxn " in vector from %.10f", xArr[0]);     \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
        {                                                                                          \
            __m128 x0 = _mm512_extractf32x4_ps(x, 0);                                              \
            __m128 x1 = _mm512_extractf32x4_ps(x, 1);                                              \
            __m128 x2 = _mm512_extractf32x4_ps(x, 2);                                              \
            __m128 x3 = _mm512_extractf32x4_ps(x, 3);                                              \
            __m128 y0 = fxn(x0);                                                                   \
            __m128 y1 = fxn(x1);                                                                   \
            __m128 y2 = fxn(x2);                                                                   \
            __m128 y3 = fxn(x3);                                                                   \
            __m512 yTest = _mm512_broadcast_f32x4(y0);                                             \
            yTest = _mm512_insertf32x4(yTest, y1, 1);                                              \
            yTest = _mm512_insertf32x4(yTest, y2, 2);                                              \
            yTest = _mm512_insertf32x4(yTest, y3, 3);                                              \
            __mmask16 pass1 = _mm512_cmp_ps_mask(y, yTest, _CMP_EQ_OS);                            \
            if (pass1 != 65535)                                                                    \
            {                                                                                      \
                printf("Inconsistent xmm result from " #fxn " in vector from %.10f", xArr[0]);     \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
        {                                                                                          \
            float x0 = _mm512_cvtss_f32(x);                                                        \
            float yTest = fxn(x0);                                                                 \
            if (yTest != _mm512_cvtss_f32(y))                                                      \
            {                                                                                      \
                printf("Inconsistent scalar result from " #fxn " for input %.10f", x0);            \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
    } while (0);

// Helper macro to verify that the SinCos implementation matches the Sin and Cos implementations.
#define TEST_SINCOS()                                                                              \
    do                                                                                             \
    {                                                                                              \
        __m512 x = _mm512_loadu_ps(xArr);                                                          \
        __m512 sRef = Sinf(x);                                                                     \
        __m512 cRef = Cosf(x);                                                                     \
        __m512 s, c;                                                                               \
        SinCosf(s, c, x);                                                                          \
        __mmask16 pass1 =                                                                          \
            _mm512_cmp_ps_mask(s, sRef, _CMP_EQ_OS) & _mm512_cmp_ps_mask(c, cRef, _CMP_EQ_OS);     \
        if (pass1 != 65535)                                                                        \
        {                                                                                          \
            printf("SinCos didn't match sin and cos in vector from %.10f", xArr[0]);               \
            exit(1);                                                                               \
        }                                                                                          \
        {                                                                                          \
            __m256 x0 = _mm512_extractf32x8_ps(x, 0);                                              \
            __m256 x1 = _mm512_extractf32x8_ps(x, 1);                                              \
            __m256 s0, c0, s1, c1;                                                                 \
            SinCosf(s0, c0, x0);                                                                   \
            SinCosf(s1, c1, x1);                                                                   \
            __m512 sTest = _mm512_broadcast_f32x8(s0);                                             \
            __m512 cTest = _mm512_broadcast_f32x8(c0);                                             \
            sTest = _mm512_insertf32x8(sTest, s1, 1);                                              \
            cTest = _mm512_insertf32x8(cTest, c1, 1);                                              \
            __mmask16 pass1 = _mm512_cmp_ps_mask(s, sTest, _CMP_EQ_OS) &                           \
                              _mm512_cmp_ps_mask(c, cTest, _CMP_EQ_OS);                            \
            if (pass1 != 65535)                                                                    \
            {                                                                                      \
                printf("Inconsistent ymm result from SinCos in vector from %.10f", xArr[0]);       \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
        {                                                                                          \
            __m128 x0 = _mm512_extractf32x4_ps(x, 0);                                              \
            __m128 x1 = _mm512_extractf32x4_ps(x, 1);                                              \
            __m128 x2 = _mm512_extractf32x4_ps(x, 2);                                              \
            __m128 x3 = _mm512_extractf32x4_ps(x, 3);                                              \
            __m128 s0, c0, s1, c1, s2, c2, s3, c3;                                                 \
            SinCosf(s0, c0, x0);                                                                   \
            SinCosf(s1, c1, x1);                                                                   \
            SinCosf(s2, c2, x2);                                                                   \
            SinCosf(s3, c3, x3);                                                                   \
            __m512 sTest = _mm512_broadcast_f32x4(s0);                                             \
            __m512 cTest = _mm512_broadcast_f32x4(c0);                                             \
            sTest = _mm512_insertf32x4(sTest, s1, 1);                                              \
            cTest = _mm512_insertf32x4(cTest, c1, 1);                                              \
            sTest = _mm512_insertf32x4(sTest, s2, 2);                                              \
            cTest = _mm512_insertf32x4(cTest, c2, 2);                                              \
            sTest = _mm512_insertf32x4(sTest, s3, 3);                                              \
            cTest = _mm512_insertf32x4(cTest, c3, 3);                                              \
            __mmask16 pass1 = _mm512_cmp_ps_mask(s, sTest, _CMP_EQ_OS) &                           \
                              _mm512_cmp_ps_mask(c, cTest, _CMP_EQ_OS);                            \
            if (pass1 != 65535)                                                                    \
            {                                                                                      \
                printf("Inconsistent xmm result from SinCos in vector from %.10f", xArr[0]);       \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
        {                                                                                          \
            float x0 = _mm512_cvtss_f32(x);                                                        \
            float s0, c0;                                                                          \
            SinCosf(s0, c0, x0);                                                                   \
            if (s0 != _mm512_cvtss_f32(s) || c0 != _mm512_cvtss_f32(c))                            \
            {                                                                                      \
                printf("Inconsistent scalar result from SinCos on input %.10f", x0);               \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
    } while (0);

void TestAccuracy()
{
    alignas(64) float xArr[16];
    alignas(64) float refArr[16];
    constexpr float kStepSize = 0x1.0p-23f;
    static_assert(kStepSize <= 0x1.0p-4f,
                  "kStepSize must be <= 1/16 since there are 16 floats in a zmm");
    constexpr float kRelErrTol = 0x2.0p-23f;
    const __m512 lowerBound = _mm512_set1_ps(1.0f - kRelErrTol);
    const __m512 upperBound = _mm512_set1_ps(1.0f + kRelErrTol);
    __m512 maxRelErr = _mm512_setzero_ps();
    __m512 minRelErr = _mm512_setzero_ps();
    printf("Starting trig function test\n");
    for (float exp = 0x1.0p20f; exp >= 0x1.0p-125f; exp *= 0.5f)
    {
        for (float sign = 1.0f; sign >= -1.0f; sign -= 2.0f)
        {
            for (float mant = 1.0f; mant < 2.0f; mant += (16.0f * kStepSize))
            {
                for (int i = 0; i < 16; ++i)
                {
                    xArr[i] = sign * exp * (mant + i * kStepSize);
                }
                TEST_VECTOR_FUNCTION(Sinf, sinf);
                TEST_VECTOR_FUNCTION(Cosf, cosf);
                TEST_SINCOS();
                TEST_VECTOR_FUNCTION(Tanf, tanf);
            }
        }
        if (exp == 0x1p-10f)
        {
            printf("All trig functions pass above %.10f\n", exp);
        }
    }
    printf("Finished testing trig functions!\n");
}

void Debug()
{
    for (float xStart = 0.5f; xStart > -1.0f; xStart -= 0x1p-20f)
    {
        alignas(64) float xArr[16];
        alignas(64) float refArr[16];
        __m512 x = _mm512_set1_ps(xStart);
        x = _mm512_castsi512_ps(_mm512_sub_epi32(
            _mm512_castps_si512(x),
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)));
        _mm512_storeu_ps(xArr, x);
        for (U32 i = 0; i < 16; ++i)
        {
            refArr[i] = std::asinf(xArr[i]);
        }
        __m512 ref = _mm512_loadu_ps(refArr);
        __m512 y = Asinf(x);
        __m512 relErrS = _mm512_div_ps(_mm512_sub_ps(y, ref), ref);
        __m512 absErrS = _mm512_sub_ps(y, ref);
        if (_mm512_reduce_max_ps(_mm512_abs_ps(relErrS)) > 2e-7f)
            printf("Breakpoint me\n");
    }
}

void main()
{
    //TestAccuracy();
    Debug();
}
