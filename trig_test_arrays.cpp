#include <cmath>
#include <cstdio>
#include <limits>

#include "nem_math.h"

#define TEST_STATIC_ARRAY_LEN(fxn, len)                                                            \
    do                                                                                             \
    {                                                                                              \
        __m256 nans = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));                          \
        for (U32 i = 0; i < 64; i += 8)                                                            \
        {                                                                                          \
            _mm256_storeu_ps(arrTestY + i, nans);                                                  \
        }                                                                                          \
        fxn<(len)>(arrTestY, arrTestX);                                                            \
        for (U32 j = 0; j < (len); ++j)                                                            \
        {                                                                                          \
            if (arrTestY[j] != arrTestRef[j])                                                      \
            {                                                                                      \
                printf("Static array " #fxn " incorrect on iter %d, idx %d, with input %.10f",     \
                       (len),                                                                      \
                       j,                                                                          \
                       arrTestX[j]);                                                               \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
        if constexpr ((len) < 64)                                                                  \
        {                                                                                          \
            if (arrTestY[(len)] == arrTestY[(len)])                                                \
            {                                                                                      \
                printf("Static array " #fxn " overran on iter %d", (len));                         \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
    } while (0);

#define TEST_STATIC_ARRAY(fxn)                                                                     \
    do                                                                                             \
    {                                                                                              \
        for (U32 i = 0; i < 64; i += 8)                                                            \
        {                                                                                          \
            _mm256_storeu_ps(arrTestRef + i, fxn(_mm256_loadu_ps(arrTestX + i)));                  \
        }                                                                                          \
        TEST_STATIC_ARRAY_LEN(fxn, 0)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 1)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 2)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 3)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 4)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 5)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 6)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 7)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 8)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 9)                                                              \
        TEST_STATIC_ARRAY_LEN(fxn, 10)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 11)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 12)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 13)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 14)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 15)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 16)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 17)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 18)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 19)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 20)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 21)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 22)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 23)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 24)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 25)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 26)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 27)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 28)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 29)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 30)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 31)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 32)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 33)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 34)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 35)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 36)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 37)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 38)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 39)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 40)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 41)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 42)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 43)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 44)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 45)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 46)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 47)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 48)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 49)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 50)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 51)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 52)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 53)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 54)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 55)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 56)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 57)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 58)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 59)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 60)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 61)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 62)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 63)                                                             \
        TEST_STATIC_ARRAY_LEN(fxn, 64)                                                             \
    } while (0);

#define TEST_DYNAMIC_ARRAY(fxn)                                                                    \
    do                                                                                             \
    {                                                                                              \
        for (U32 i = 0; i < 64; i += 8)                                                            \
        {                                                                                          \
            _mm256_storeu_ps(arrTestRef + i, fxn(_mm256_loadu_ps(arrTestX + i)));                  \
        }                                                                                          \
        for (U32 i = 0; i < 64; ++i)                                                               \
        {                                                                                          \
            __m256 nans = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));                      \
            for (U32 i = 0; i < 64; i += 8)                                                        \
            {                                                                                      \
                _mm256_storeu_ps(arrTestY + i, nans);                                              \
            }                                                                                      \
            fxn(arrTestY, arrTestX, i);                                                            \
            for (U32 j = 0; j < i; ++j)                                                            \
            {                                                                                      \
                if (arrTestY[j] != arrTestRef[j])                                                  \
                {                                                                                  \
                    printf("Dynamic array " #fxn                                                   \
                           " incorrect on iter %d, idx %d, with input %.10f",                      \
                           i,                                                                      \
                           j,                                                                      \
                           arrTestX[j]);                                                           \
                    exit(1);                                                                       \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if (i < 63)                                                                                \
        {                                                                                          \
            if (arrTestY[i + 1] == arrTestY[i + 1])                                                \
            {                                                                                      \
                printf("Dynamic array " #fxn " overran on iter %d", i);                            \
                exit(1);                                                                           \
            }                                                                                      \
        }                                                                                          \
    } while (0);

void TestStaticArrays()
{
    alignas(64) float arrTestX[64];
    alignas(64) float arrTestY[64];
    alignas(64) float arrTestRef[64];
    float sign = 1.0f;
    float exp = 0.25f;
    float mant = 1.0f;
    U32 i = 0;
    do
    {
        arrTestX[i] = sign * exp * mant;
        ++i;
        mant += 0.25f;
        if ((i & 4) == 0)
        {
            exp *= 2.0f;
            mant = 1.0f;
        }
        if ((i & 32) == 0)
        {
            exp = 0.25f;
            sign *= -1.0f;
        }
        ++i;
    } while (i < 64);
    TEST_STATIC_ARRAY(Sinf);
    TEST_STATIC_ARRAY(Cosf);
    TEST_STATIC_ARRAY(Tanf);
    printf("Finished testing static arrays\n");
}

void TestDynamicArrays()
{
    alignas(64) float arrTestX[64];
    alignas(64) float arrTestY[64];
    alignas(64) float arrTestRef[64];
    float sign = 1.0f;
    float exp = 0.25f;
    float mant = 1.0f;
    U32 i = 0;
    do
    {
        arrTestX[i] = sign * exp * mant;
        ++i;
        mant += 0.25f;
        if ((i & 4) == 0)
        {
            exp *= 2.0f;
            mant = 1.0f;
        }
        if ((i & 32) == 0)
        {
            exp = 0.25f;
            sign *= -1.0f;
        }
        ++i;
    } while (i < 64);
    TEST_DYNAMIC_ARRAY(Sinf);
    TEST_DYNAMIC_ARRAY(Cosf);
    TEST_DYNAMIC_ARRAY(Tanf);
    printf("Finished testing dynamic arrays\n");
}

void main()
{
    TestStaticArrays();
    TestDynamicArrays();
}
