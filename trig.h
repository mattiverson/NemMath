#pragma once

#include "common.h"

// Single precision, scalar.
static inline float Sin(float x);
static inline float Cos(float x);
static inline float Tan(float x);
static inline void SinCos(float& s, float& c, float x);
static inline float Asin(float x);
static inline float Acos(float x);
static inline float Atan(float x);
static inline float Atan2(float y, float x);

// Double precision, scalar.
static inline double Sin(double x);
static inline double Cos(double x);
static inline double Tan(double x);
static inline void SinCos(double& s, double& c, double x);
static inline double Asin(double x);
static inline double Acos(double x);
static inline double Atan(double x);
static inline double Atan2(double y, double x);

// Single precision, statically-sized arrays.
template <U64 len> static inline void Sin(float* __restrict out, const float* __restrict x);
template <U64 len> static inline void Cos(float* __restrict out, const float* __restrict x);
template <U64 len> static inline void Tan(float* __restrict out, const float* __restrict x);
template <U64 len> static inline void SinCos(float* __restrict s, float* __restrict c, const float* __restrict x);
template <U64 len> static inline void Asin(float* __restrict out, const float* __restrict x);
template <U64 len> static inline void Acos(float* __restrict out, const float* __restrict x);
template <U64 len> static inline void Atan(float* __restrict out, const float* __restrict x);
template <U64 len>
static inline void Atan2(float* __restrict out, const float* __restrict y, const float* __restrict x);

// Double precision, statically-sized arrays.
template <U64 len> static inline void Sin(double* __restrict out, const double* __restrict x);
template <U64 len> static inline void Cos(double* __restrict out, const double* __restrict x);
template <U64 len> static inline void Tan(double* __restrict out, const double* __restrict x);
template <U64 len> static inline void SinCos(double* __restrict s, double* __restrict c, const double* __restrict x);
template <U64 len> static inline void Asin(double* __restrict out, const double* __restrict x);
template <U64 len> static inline void Acos(double* __restrict out, const double* __restrict x);
template <U64 len> static inline void Atan(double* __restrict out, const double* __restrict x);
template <U64 len>
static inline void Atan2(double* __restrict out, const double* __restrict y, const double* __restrict x);

// Single precision, dynamically-sized arrays.
static inline void Sin(float* __restrict out, const float* __restrict x, U64 len);
static inline void Cos(float* __restrict out, const float* __restrict x, U64 len);
static inline void Tan(float* __restrict out, const float* __restrict x, U64 len);
static inline void SinCos(float* __restrict s, float* __restrict c, const float* __restrict x, U64 len);
static inline void Asin(float* __restrict out, const float* __restrict x, U64 len);
static inline void Acos(float* __restrict out, const float* __restrict x, U64 len);
static inline void Atan(float* __restrict out, const float* __restrict x, U64 len);
static inline void Atan2(float* __restrict out, const float* __restrict y, const float* __restrict x, U64 len);

// Double precision, dynamically-sized arrays.
static inline void Sin(double* __restrict out, const double* __restrict x, U64 len);
static inline void Cos(double* __restrict out, const double* __restrict x, U64 len);
static inline void Tan(double* __restrict out, const double* __restrict x, U64 len);
static inline void SinCos(double* __restrict s, double* __restrict c, const double* __restrict x, U64 len);
static inline void Asin(double* __restrict out, const double* __restrict x, U64 len);
static inline void Acos(double* __restrict out, const double* __restrict x, U64 len);
static inline void Atan(double* __restrict out, const double* __restrict x, U64 len);
static inline void Atan2(double* __restrict out, const double* __restrict y, const double* __restrict x, U64 len);

#ifdef __AVX2__
// Single precision, AVX2 SIMD for x86-64 CPUs.
static inline __m128 __vectorcall Sin(__m128 x);
static inline __m128 __vectorcall Cos(__m128 x);
static inline __m128 __vectorcall Tan(__m128 x);
static inline void __vectorcall SinCos(__m128& s, __m128& c, __m128 x);
static inline __m128 __vectorcall Asin(__m128 x);
static inline __m128 __vectorcall Acos(__m128 x);
static inline __m128 __vectorcall Atan(__m128 x);
static inline __m128 __vectorcall Atan2(__m128 y, __m128 x);

static inline __m256 __vectorcall Sin(__m256 x);
static inline __m256 __vectorcall Cos(__m256 x);
static inline __m256 __vectorcall Tan(__m256 x);
static inline void __vectorcall SinCos(__m256& s, __m256& c, __m256 x);
static inline __m256 __vectorcall Asin(__m256 x);
static inline __m256 __vectorcall Acos(__m256 x);
static inline __m256 __vectorcall Atan(__m256 x);
static inline __m256 __vectorcall Atan2(__m256 y, __m256 x);

// Double precision, AVX2 SIMD for x86-64 CPUs.
static inline __m128d __vectorcall Sin(__m128d x);
static inline __m128d __vectorcall Cos(__m128d x);
static inline __m128d __vectorcall Tan(__m128d x);
static inline void __vectorcall SinCos(__m128d& s, __m128d& c, __m128d x);
static inline __m128d __vectorcall Asin(__m128d x);
static inline __m128d __vectorcall Acos(__m128d x);
static inline __m128d __vectorcall Atan(__m128d x);
static inline __m128d __vectorcall Atan2(__m128d y, __m128d x);

static inline __m256d __vectorcall Sin(__m256d x);
static inline __m256d __vectorcall Cos(__m256d x);
static inline __m256d __vectorcall Tan(__m256d x);
static inline void __vectorcall SinCos(__m256d& s, __m256d& c, __m256d x);
static inline __m256d __vectorcall Asin(__m256d x);
static inline __m256d __vectorcall Acos(__m256d x);
static inline __m256d __vectorcall Atan(__m256d x);
static inline __m256d __vectorcall Atan2(__m256d y, __m256d x);
#endif // __AVX2__

#ifdef __AVX2__
// Single precision, scalar. Currently only supported for x86 CPUs with AVX2.
static inline float Sin(float x)
{
    __m128 X = _mm_set_ss(x);
    __m128 cycle = _mm_fmadd_ss(X, _mm_set_ss(2.0f * kInvPif), _mm_set_ss(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ss(cycle, _mm_set_ss(0x1.8p+23));
    const __m128 cosMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_srli_epi32(cycleI, 1), 31));
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPif), cycle, X);
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPi2f), cycle, X);
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPi3f), cycle, X);
    const __m128 x2 = _mm_mul_ss(X, X);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(X, _mm_set_ss(1.0f), cosMask));

    __m128 res = _mm_blendv_ps(_mm_set_ss(2.67107816e-06f), _mm_set_ss(2.43944651e-05f), cosMask);
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(-1.98358801e-04f), _mm_set_ss(-1.38866133e-03f), cosMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(8.33333470e-03f), _mm_set_ss(4.16666158e-02f), cosMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(-1.66666672e-01f), _mm_set_ss(-5.00000000e-01f), cosMask));
    res = _mm_fmadd_ss(res, _mm_mul_ss(b, x2), b);
    return _mm_cvtss_f32(res);
}

static inline float Cos(float x)
{
    __m128 X = _mm_set_ss(x);
    __m128 cycle = _mm_fmadd_ss(X, _mm_set_ss(2.0f * kInvPif), _mm_set_ss(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ss(cycle, _mm_set_ss(0x1.8p+23));
    const __m128 sinMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(cycleI, _mm_srli_epi32(cycleI, 1)), 31));

    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPif), cycle, X);
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPi2f), cycle, X);
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPi3f), cycle, X);
    const __m128 x2 = _mm_mul_ss(X, X);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(_mm_set_ss(1.0f), X, sinMask));

    __m128 res = _mm_blendv_ps(_mm_set_ss(2.43944651e-05f), _mm_set_ss(2.67107816e-06f), sinMask);
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(-1.38866133e-03f), _mm_set_ss(-1.98358801e-04f), sinMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(4.16666158e-02f), _mm_set_ss(8.33333470e-03f), sinMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(-5.00000000e-01f), _mm_set_ss(-1.66666672e-01f), sinMask));
    res = _mm_fmadd_ss(res, _mm_mul_ss(b, x2), b);
    return _mm_cvtss_f32(res);
}

static inline void SinCos(float& s, float& c, float x)
{
    __m128 X = _mm_unpacklo_ps(_mm_set_ss(x), _mm_set_ss(x));
    __m128 cycle = _mm_fmadd_ps(X, _mm_set1_ps(2.0f * kInvPif), _mm_set1_ps(0x1.8p+23));
    const __m128i cycleI = _mm_add_epi32(_mm_castps_si128(cycle), _mm_setr_epi32(0, 1, 0, 1));
    cycle = _mm_sub_ps(cycle, _mm_set1_ps(0x1.8p+23));
    const __m128 cosMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_srli_epi32(cycleI, 1), 31));

    X = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPif), cycle, X);
    X = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi2f), cycle, X);
    X = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi3f), cycle, X);
    const __m128 x2 = _mm_mul_ps(X, X);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(X, _mm_set1_ps(1.0f), cosMask));

    __m128 res = _mm_blendv_ps(_mm_set1_ps(2.67107816e-06f), _mm_set1_ps(2.43944651e-05f), cosMask);
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(-1.98358801e-04f), _mm_set1_ps(-1.38866133e-03f), cosMask));
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(8.33333470e-03f), _mm_set1_ps(4.16666158e-02f), cosMask));
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(-1.66666672e-01f), _mm_set1_ps(-5.00000000e-01f), cosMask));
    res = _mm_fmadd_ps(res, _mm_mul_ps(b, x2), b);
    s = _mm_cvtss_f32(res);
    c = _mm_cvtss_f32(_mm_shuffle_ps(res, res, 177));
}

static inline float Tan(float x)
{
    float s, c;
    SinCos(s, c, x);
    return s / c;
}

// Single-precision, x86 SIMD using AVX2 for 128-bit and 256-bit vectors.

static inline __m128 __vectorcall Sin(__m128 x)
{
    __m128 cycle = _mm_fmadd_ps(x, _mm_set1_ps(2.0f * kInvPif), _mm_set1_ps(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ps(cycle, _mm_set1_ps(0x1.8p+23));
    const __m128 cosMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_srli_epi32(cycleI, 1), 31));

    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPif), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi3f), cycle, x);
    const __m128 x2 = _mm_mul_ps(x, x);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(x, _mm_set1_ps(1.0f), cosMask));

    __m128 res = _mm_blendv_ps(_mm_set1_ps(2.67107816e-06f), _mm_set1_ps(2.43944651e-05f), cosMask);
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(-1.98358801e-04f), _mm_set1_ps(-1.38866133e-03f), cosMask));
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(8.33333470e-03f), _mm_set1_ps(4.16666158e-02f), cosMask));
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(-1.66666672e-01f), _mm_set1_ps(-5.00000000e-01f), cosMask));
    res = _mm_fmadd_ps(res, _mm_mul_ps(b, x2), b);
    return res;
}

static inline __m128 __vectorcall Cos(__m128 x)
{
    __m128 cycle = _mm_fmadd_ps(x, _mm_set1_ps(2.0f * kInvPif), _mm_set1_ps(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ps(cycle, _mm_set1_ps(0x1.8p+23));
    const __m128 sinMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(cycleI, _mm_srli_epi32(cycleI, 1)), 31));

    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPif), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi3f), cycle, x);
    const __m128 x2 = _mm_mul_ps(x, x);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(_mm_set1_ps(1.0f), x, sinMask));

    __m128 res = _mm_blendv_ps(_mm_set1_ps(2.43944651e-05f), _mm_set1_ps(2.67107816e-06f), sinMask);
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(-1.38866133e-03f), _mm_set1_ps(-1.98358801e-04f), sinMask));
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(4.16666158e-02f), _mm_set1_ps(8.33333470e-03f), sinMask));
    res = _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(-5.00000000e-01f), _mm_set1_ps(-1.66666672e-01f), sinMask));
    res = _mm_fmadd_ps(res, _mm_mul_ps(b, x2), b);
    return res;
}

static inline void __vectorcall SinCos(__m128& s, __m128& c, __m128 x)
{
    __m128 cycle = _mm_fmadd_ps(x, _mm_set1_ps(2.0f * kInvPif), _mm_set1_ps(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ps(cycle, _mm_set1_ps(0x1.8p+23));
    const __m128 cosMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 sinSignMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(cycleI, _mm_srli_epi32(cycleI, 1)), 31));
    const __m128 cosSignMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_srli_epi32(cycleI, 1), 31));

    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPif), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi3f), cycle, x);
    const __m128 x2 = _mm_mul_ps(x, x);
    __m128 bs = _mm_xor_ps(sinSignMask, x);
    __m128 bc = _mm_xor_ps(cosSignMask, _mm_set1_ps(1.0f));

    __m128 s1 = _mm_set1_ps(2.67107816e-06f);
    s1 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(-1.98358801e-04f));
    s1 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(8.33333470e-03f));
    s1 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(-1.66666672e-01f));
    __m128 c1 = _mm_set1_ps(2.43944651e-05f);
    c1 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(-1.38866133e-03f));
    c1 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(4.16666158e-02f));
    c1 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(-5.00000000e-01f));
    s1 = _mm_fmadd_ps(s1, _mm_mul_ps(bs, x2), bs);
    c1 = _mm_fmadd_ps(c1, _mm_mul_ps(bc, x2), bc);
    s = _mm_blendv_ps(s1, c1, cosMask);
    c = _mm_blendv_ps(c1, s1, cosMask);
}

static inline __m128 __vectorcall Tan(__m128 x)
{
    __m128 s, c;
    SinCos(s, c, x);
    return _mm_div_ps(s, c);
}

static inline __m256 __vectorcall Sin(__m256 x)
{
    __m256 cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPif), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256 cosMask = _mm256_castsi256_ps(_mm256_slli_epi32(cycleI, 31));
    const __m256 signMask = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(cycleI, 1), 31));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPif), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi3f), cycle, x);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 b = _mm256_xor_ps(signMask, _mm256_blendv_ps(x, _mm256_set1_ps(1.0f), cosMask));

    __m256 res = _mm256_blendv_ps(_mm256_set1_ps(2.67107816e-06f), _mm256_set1_ps(2.43944651e-05f), cosMask);
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(-1.98358801e-04f), _mm256_set1_ps(-1.38866133e-03f), cosMask));
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(8.33333470e-03f), _mm256_set1_ps(4.16666158e-02f), cosMask));
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(-1.66666672e-01f), _mm256_set1_ps(-5.00000000e-01f), cosMask));
    res = _mm256_fmadd_ps(res, _mm256_mul_ps(b, x2), b);
    return res;
}

static inline __m256 __vectorcall Cos(__m256 x)
{
    __m256 cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPif), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256 sinMask = _mm256_castsi256_ps(_mm256_slli_epi32(cycleI, 31));
    const __m256 signMask =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_xor_si256(cycleI, _mm256_srli_epi32(cycleI, 1)), 31));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPif), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi3f), cycle, x);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 b = _mm256_xor_ps(signMask, _mm256_blendv_ps(_mm256_set1_ps(1.0f), x, sinMask));

    __m256 res = _mm256_blendv_ps(_mm256_set1_ps(2.43944651e-05f), _mm256_set1_ps(2.67107816e-06f), sinMask);
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(-1.38866133e-03f), _mm256_set1_ps(-1.98358801e-04f), sinMask));
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(4.16666158e-02f), _mm256_set1_ps(8.33333470e-03f), sinMask));
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(-5.00000000e-01f), _mm256_set1_ps(-1.66666672e-01f), sinMask));
    res = _mm256_fmadd_ps(res, _mm256_mul_ps(b, x2), b);
    return res;
}

static inline void __vectorcall SinCos(__m256& s, __m256& c, __m256 x)
{
    __m256 cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPif), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256 cosMask = _mm256_castsi256_ps(_mm256_slli_epi32(cycleI, 31));
    const __m256 sinSignMask =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_xor_si256(cycleI, _mm256_srli_epi32(cycleI, 1)), 31));
    const __m256 cosSignMask = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(cycleI, 1), 31));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPif), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi3f), cycle, x);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 bs = _mm256_xor_ps(sinSignMask, x);
    __m256 bc = _mm256_xor_ps(cosSignMask, _mm256_set1_ps(1.0f));

    __m256 s1 = _mm256_set1_ps(2.67107816e-06f);
    __m256 c1 = _mm256_set1_ps(2.43944651e-05f);
    s1 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(-1.98358801e-04f));
    c1 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(-1.38866133e-03f));
    s1 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(8.33333470e-03f));
    c1 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(4.16666158e-02f));
    s1 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(-1.66666672e-01f));
    c1 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(-5.00000000e-01f));
    s1 = _mm256_fmadd_ps(s1, _mm256_mul_ps(bs, x2), bs);
    c1 = _mm256_fmadd_ps(c1, _mm256_mul_ps(bc, x2), bc);
    s = _mm256_blendv_ps(s1, c1, cosMask);
    c = _mm256_blendv_ps(c1, s1, cosMask);
}

static inline __m256 __vectorcall Tan(__m256 x)
{
    __m256 s, c;
    SinCos(s, c, x);
    return _mm256_div_ps(s, c);
}

static inline __m256 __vectorcall Asin(__m256 x)
{
    const __m256 kSignMask = _mm256_set1_ps(-0.0f);
    const __m256 xAbs = _mm256_andnot_ps(kSignMask, x);
    __m256 xOuter = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), xAbs, _mm256_set1_ps(2.0f));
    const __m256 outerMask = _mm256_cmp_ps(xAbs, _mm256_set1_ps(0.5f), _CMP_GE_OQ);
    const __m256 signBit = _mm256_and_ps(x, kSignMask);
    const __m256 flippedSignBit = _mm256_xor_ps(signBit, kSignMask);
    const __m256 p = _mm256_blendv_ps(_mm256_mul_ps(x, x), xOuter, outerMask);
    const __m256 lastMul = _mm256_blendv_ps(x, _mm256_xor_ps(flippedSignBit, _mm256_sqrt_ps(xOuter)), outerMask);
    __m256 yPoly = _mm256_blendv_ps(_mm256_set1_ps(kAsinf[4]), _mm256_set1_ps(kAcosf[4]), outerMask);
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[3]), _mm256_set1_ps(kAcosf[3]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[2]), _mm256_set1_ps(kAcosf[2]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[1]), _mm256_set1_ps(kAcosf[1]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[0]), _mm256_set1_ps(kAcosf[0]), outerMask));
    //
    // can bring this back to clean things up a bit for the edges of the acos
    yPoly = _mm256_mul_ps(yPoly, p);
    const __m256 lastAdd =
        _mm256_blendv_ps(x, _mm256_add_ps(_mm256_xor_ps(signBit, _mm256_set1_ps(0.5f * kPif)), lastMul), outerMask);
    return _mm256_fmadd_ps(yPoly, lastMul, lastAdd);
}

static inline __m256 __vectorcall Acos(__m256 x)
{
    const __m256 kSignMask = _mm256_set1_ps(-0.0f);
    const __m256 xAbs = _mm256_andnot_ps(kSignMask, x);
    __m256 xOuter = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), xAbs, _mm256_set1_ps(2.0f));
    const __m256 outerMask = _mm256_cmp_ps(xAbs, _mm256_set1_ps(0.5f), _CMP_GE_OQ);
    const __m256 signBit = _mm256_and_ps(x, kSignMask);
    const __m256 p = _mm256_blendv_ps(_mm256_mul_ps(x, x), xOuter, outerMask);
    const __m256 lastMul = _mm256_blendv_ps(x, _mm256_xor_ps(signBit, _mm256_sqrt_ps(xOuter)), outerMask);
    __m256 yPoly = _mm256_blendv_ps(_mm256_set1_ps(kAsinf[4]), _mm256_set1_ps(kAcosf[4]), outerMask);
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[3]), _mm256_set1_ps(kAcosf[3]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[2]), _mm256_set1_ps(kAcosf[2]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[1]), _mm256_set1_ps(kAcosf[1]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(kAsinf[0]), _mm256_set1_ps(kAcosf[0]), outerMask));
    const __m256 lastP = _mm256_xor_ps(p, _mm256_andnot_ps(outerMask, kSignMask));
    yPoly = _mm256_fmadd_ps(yPoly, lastP, _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_setzero_ps(), outerMask));
    const __m256 lastAdd = _mm256_blendv_ps(
        _mm256_set1_ps(0.5f * kPif),
        _mm256_add_ps(lastMul,
                      _mm256_and_ps(_mm256_set1_ps(kPif),
                                    _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(signBit), 31)))), outerMask);
    //return _mm256_fmadd_ps(yPoly, lastMul, lastAdd);
    return _mm256_fmadd_ps(yPoly, lastMul, lastAdd);
}

//static inline __m256 __vectorcall Atan_1(__m256 x)
//{
//    const __m256 kSignMask = _mm256_set1_ps(-0.0f);
//    const __m256 sign = _mm256_and_ps(x, kSignMask);
//    x = _mm256_andnot_ps(x, kSignMask);
//    const __m256 one = _mm256_set1_ps(1.0f);
//    const __m256 farMask = _mm256_cmp_ps(x, one, _CMP_GE_OQ);
//    x = _mm256_blendv_ps(x, _mm256_div_ps(one, x), farMask);
//    const __m256 x2 = _mm256_mul_ps(x, x);
//    __m256 y = _mm256_set1_ps(-1.472577981985523e-03f);
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(9.392884421686176e-03f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(-2.814183884379418e-02f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(5.464641870671016e-02f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(-8.188949902975562e-02f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(1.086826130424285e-01f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(-1.424497268154095e-01f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(1.999615551552361e-01f));
//    y = _mm256_fmadd_ps(y, x2, _mm256_set1_ps(-3.333316439068992e-01f));
//    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x2, x), x);

//    const __m256 ySign = _mm256_blendv_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f), farMask);
//    const __m256 base = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(0.5f * kPif), farMask);
//    const __m256 base2 = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(0.5f * kPi2f), farMask);
//    const __m256 r = _mm256_add_ps(_mm256_fmadd_ps(ySign, y, base2), base);
//    const __m256 res = _mm256_xor_ps(r, sign);
//    return res;
//}

static inline __m256 __vectorcall Atan(__m256 x)
{
    const __m256 kSignMask = _mm256_set1_ps(-0.0f);
    const __m256 sign = _mm256_and_ps(kSignMask, x);
    x = _mm256_andnot_ps(kSignMask, x);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 rx = _mm256_div_ps(one, x);
    const __m256 farMask = _mm256_cmp_ps(x, one, _CMP_GE_OQ);
    const __m256 farSign = _mm256_and_ps(kSignMask, farMask);
    x = _mm256_blendv_ps(x, rx, farMask);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 num = _mm256_set1_ps(1.70646477e-02f);
    num = _mm256_fmadd_ps(num, x2, _mm256_set1_ps(3.95497262e-01f));
    num = _mm256_fmadd_ps(num, x2, _mm256_set1_ps(1.27574670e+00f));
    num = _mm256_fmadd_ps(num, _mm256_mul_ps(x2, x), x);
    __m256 den = _mm256_set1_ps(8.18777159e-02f);
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(7.31906652e-01f));
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(1.60907602e+00f));
    den = _mm256_fmadd_ps(den, x2, _mm256_set1_ps(1.0f));
    const __m256 y = _mm256_div_ps(num, den);

    const __m256 ySign = _mm256_xor_ps(_mm256_set1_ps(1.0f), farSign);
    const __m256 base = _mm256_and_ps(_mm256_set1_ps(0.5f * kPif), farMask);
    const __m256 base2 = _mm256_and_ps(_mm256_set1_ps(0.5f * kPi2f), farMask);
    const __m256 r = _mm256_add_ps(_mm256_fmadd_ps(ySign, y, base2), base);
    const __m256 res = _mm256_xor_ps(r, sign);
    return res;
}

static inline __m256 __vectorcall Atan2(__m256 y, __m256 x)
{
    const __m256 kSignMask = _mm256_set1_ps(-0.0f);
    const __m256 absY = _mm256_andnot_ps(kSignMask, y);
    const __m256 absX = _mm256_andnot_ps(kSignMask, x);
    const __m256 farMask = _mm256_cmp_ps(absY, absX, _CMP_GE_OQ);
    __m256 num = _mm256_blendv_ps(absY, x, farMask);
    __m256 den = _mm256_blendv_ps(x, absY, farMask);
    const __m256 p = _mm256_div_ps(num, den);
    const __m256 signX = _mm256_and_ps(kSignMask, x);
    const __m256 signY = _mm256_and_ps(kSignMask, y);
    const __m256 signFar = _mm256_and_ps(kSignMask, farMask);
    const __m256 notZerosMask = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_cmpgt_epi32(_mm256_castps_si256(absX), _mm256_setzero_si256()),
        _mm256_cmpgt_epi32(_mm256_castps_si256(absY), _mm256_setzero_si256())));
    const __m256 p2 = _mm256_mul_ps(p, p);
    num = _mm256_set1_ps(1.70646477e-02f);
    num = _mm256_fmadd_ps(num, p2, _mm256_set1_ps(3.95497262e-01f));
    num = _mm256_fmadd_ps(num, p2, _mm256_set1_ps(1.27574670e+00f));
    num = _mm256_fmadd_ps(num, _mm256_mul_ps(p2, p), p);
    den = _mm256_set1_ps(8.18777159e-02f);
    den = _mm256_fmadd_ps(den, p2, _mm256_set1_ps(7.31906652e-01f));
    den = _mm256_fmadd_ps(den, p2, _mm256_set1_ps(1.60907602e+00f));
    den = _mm256_fmadd_ps(den, p2, _mm256_set1_ps(1.0f));
    __m256 res = _mm256_div_ps(num, den);

    const __m256 resSign = _mm256_or_ps(_mm256_set1_ps(1.0f), signFar);
    const __m256 signXMask = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(signX), 31));
    __m256 base = _mm256_and_ps(_mm256_set1_ps(kPif), signXMask);
    __m256 base2 = _mm256_and_ps(_mm256_set1_ps(kPi2f), signXMask);
    base = _mm256_blendv_ps(base, _mm256_set1_ps(0.5f * kPif), signFar);
    base2 = _mm256_blendv_ps(base2, _mm256_set1_ps(0.5f * kPi2f), signFar);
    res = _mm256_add_ps(_mm256_fmadd_ps(resSign, res, base2), base);
    res = _mm256_xor_ps(res, signY);
    res = _mm256_and_ps(res, notZerosMask);
    return res;
}

static inline __m256d __vectorcall Sin(__m256d x)
{
    __m256d cycle = _mm256_fmadd_pd(x, _mm256_set1_pd(2.0 * kInvPid), _mm256_set1_pd(0x1.8p+52));
    const __m256i cycleI = _mm256_castpd_si256(cycle);
    cycle = _mm256_sub_pd(cycle, _mm256_set1_pd(0x1.8p+52));
    const __m256d cosMask = _mm256_castsi256_pd(_mm256_slli_epi64(cycleI, 63));
    const __m256d signMask = _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_srli_epi64(cycleI, 1), 63));

    x = _mm256_fnmadd_pd(_mm256_set1_pd(0.5 * kPid), cycle, x);
    x = _mm256_fnmadd_pd(_mm256_set1_pd(0.5 * kPi2d), cycle, x);
    x = _mm256_fnmadd_pd(_mm256_set1_pd(0.5 * kPi3d), cycle, x);
    const __m256d x2 = _mm256_mul_pd(x, x);
    __m256d b = _mm256_xor_pd(signMask, _mm256_blendv_pd(x, _mm256_set1_pd(1.0), cosMask));

    __m256d res = _mm256_blendv_pd(_mm256_set1_pd(2.67107816e-06f), _mm256_set1_pd(2.43944651e-05f), cosMask);
    res = _mm256_fmadd_pd(res, x2, _mm256_blendv_pd(_mm256_set1_pd(-1.98358801e-04f), _mm256_set1_pd(-1.38866133e-03f), cosMask));
    res = _mm256_fmadd_pd(res, x2, _mm256_blendv_pd(_mm256_set1_pd(8.33333470e-03f), _mm256_set1_pd(4.16666158e-02f), cosMask));
    res = _mm256_fmadd_pd(res, x2, _mm256_blendv_pd(_mm256_set1_pd(-1.66666672e-01f), _mm256_set1_pd(-5.00000000e-01f), cosMask));
    res = _mm256_fmadd_pd(res, _mm256_mul_ps(b, x2), b);
    return res;
}

static inline __m256d __vectorcall Cos(__m256d x)
{
    __m256d cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPid), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256d sinMask = _mm256_castsi256_ps(_mm256_slli_epi64(cycleI, 63));
    const __m256d signMask =
        _mm256_castsi256_ps(_mm256_slli_epi64(_mm256_xor_si256(cycleI, _mm256_srli_epi64(cycleI, 1)), 63));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5 * kPid), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5 * kPi2d), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5 * kPi3d), cycle, x);
    const __m256d x2 = _mm256_mul_ps(x, x);
    __m256d b = _mm256_xor_ps(signMask, _mm256_blendv_ps(_mm256_set1_ps(1.0f), x, sinMask));

    __m256d res = _mm256_blendv_ps(_mm256_set1_ps(2.43944651e-05f), _mm256_set1_ps(2.67107816e-06f), sinMask);
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(-1.38866133e-03f), _mm256_set1_ps(-1.98358801e-04f), sinMask));
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(4.16666158e-02f), _mm256_set1_ps(8.33333470e-03f), sinMask));
    res = _mm256_fmadd_ps(res, x2, _mm256_blendv_ps(_mm256_set1_ps(-5.00000000e-01f), _mm256_set1_ps(-1.66666672e-01f), sinMask));
    res = _mm256_fmadd_ps(res, _mm256_mul_ps(b, x2), b);
    return res;
}

// Template definitions for single precision, statically-sized arrays.
template <U64 len> void Sin(float* __restrict out, const float* __restrict x)
{
    if constexpr (len >= 8)
    {
        constexpr U64 kNumFull = len << 3;
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pOutLast = out + len - 8;
        for (U64 i = 0; i < kNumFull; ++i)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 y = Sin(rx);
            _mm256_storeu_ps(out, y);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 32);
        }
        if constexpr (len % 8 != 0)
        {
            x = x + (len % 8) - 8;
            out = out + (len % 8) - 8;
            __m256 rx = _mm256_loadu_ps(pLast);
            __m256 y = Sin(rx);
            _mm256_storeu_ps(pOutLast, y);
        }
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float yTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 y = Sin(rx);
        _mm256_storeu_ps(out, yTemp);
        for (U64 i = 0; i < len; ++i)
            out[i] = yTemp[i];
    }
}

template <U64 len> void Cos(float* __restrict out, const float* __restrict x)
{
    if constexpr (len >= 8)
    {
        constexpr U64 kNumFull = len << 3;
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pOutLast = out + len - 8;
        for (U64 i = 0; i < kNumFull; ++i)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 y = Cos(rx);
            _mm256_storeu_ps(out, y);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 32);
        }
        if constexpr (len % 8 != 0)
        {
            x = x + (len % 8) - 8;
            out = out + (len % 8) - 8;
            __m256 rx = _mm256_loadu_ps(pLast);
            __m256 y = Cos(rx);
            _mm256_storeu_ps(pOutLast, y);
        }
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float yTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 y = Cos(rx);
        _mm256_storeu_ps(out, yTemp);
        for (U64 i = 0; i < len; ++i)
            out[i] = yTemp[i];
    }
}

template <U64 len> void Tan(float* __restrict out, const float* __restrict x)
{
    if constexpr (len >= 8)
    {
        constexpr U64 kNumFull = len << 3;
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pOutLast = out + len - 8;
        for (U64 i = 0; i < kNumFull; ++i)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 y = Tan(rx);
            _mm256_storeu_ps(out, y);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 32);
        }
        if constexpr (len % 8 != 0)
        {
            x = x + (len % 8) - 8;
            out = out + (len % 8) - 8;
            __m256 rx = _mm256_loadu_ps(pLast);
            __m256 y = Tan(rx);
            _mm256_storeu_ps(pOutLast, y);
        }
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float yTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 y = Tan(rx);
        _mm256_storeu_ps(out, yTemp);
        for (U64 i = 0; i < len; ++i)
            out[i] = yTemp[i];
    }
}

template <U64 len> static inline void SinCos(float* __restrict s, float* __restrict c, const float* __restrict x)
{
    if constexpr (len >= 8)
    {
        constexpr U64 kNumFull = len << 3;
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pSLast = s + len - 8;
        float* const __restrict pCLast = c + len - 8;
        for (U64 i = 0; i < kNumFull; ++i)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 rs, rc;
            SinCos(rs, rc, rx);
            _mm256_storeu_ps(s, rs);
            _mm256_storeu_ps(c, rc);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            s = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(s) + 32);
            c = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(c) + 32);
        }
        if constexpr (len % 8 != 0)
        {
            __m256 rx = _mm256_loadu_ps(pLast);
            __m256 rs, rc;
            SinCos(rs, rc, rx);
            _mm256_storeu_ps(pSLast, rs);
            _mm256_storeu_ps(pCLast, rc);
        }
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float sTemp[8];
        alignas(32) float cTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 rs, rc;
        SinCos(rs, rc, rx);
        _mm256_storeu_ps(sTemp, rs);
        _mm256_storeu_ps(cTemp, rc);
        for (U64 i = 0; i < len; ++i)
        {
            s[i] = sTemp[i];
            c[i] = cTemp[i];
        }
    }
}

// Single precision, dynamically sized arrays.

void Sin(float* __restrict out, const float* __restrict x, const U64 len)
{
    if (len >= 8)
    {
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pOutLast = out + len - 8;
        while (x < pLast)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 y = Sin(rx);
            _mm256_storeu_ps(out, y);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 32);
        }
        __m256 rx = _mm256_loadu_ps(pLast);
        __m256 y = Sin(rx);
        _mm256_storeu_ps(pOutLast, y);
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float yTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 y = Sin(rx);
        _mm256_storeu_ps(yTemp, y);
        for (U64 i = 0; i < len; ++i)
            out[i] = yTemp[i];
    }
}

void Cos(float* __restrict out, const float* __restrict x, const U64 len)
{
    if (len >= 8)
    {
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pOutLast = out + len - 8;
        while (x < pLast)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 y = Cos(rx);
            _mm256_storeu_ps(out, y);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 32);
        }
        __m256 rx = _mm256_loadu_ps(pLast);
        __m256 y = Cos(rx);
        _mm256_storeu_ps(pOutLast, y);
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float yTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 y = Cos(rx);
        _mm256_storeu_ps(yTemp, y);
        for (U64 i = 0; i < len; ++i)
            out[i] = yTemp[i];
    }
}

void Tan(float* __restrict out, const float* __restrict x, const U64 len)
{
    if (len >= 8)
    {
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pOutLast = out + len - 8;
        while (x < pLast)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 y = Tan(rx);
            _mm256_storeu_ps(out, y);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 32);
        }
        __m256 rx = _mm256_loadu_ps(pLast);
        __m256 y = Tan(rx);
        _mm256_storeu_ps(pOutLast, y);
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float yTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 y = Tan(rx);
        _mm256_storeu_ps(yTemp, y);
        for (U64 i = 0; i < len; ++i)
            out[i] = yTemp[i];
    }
}

void SinCos(float* __restrict s, float* __restrict c, const float* __restrict x, const U64 len)
{
    if (len >= 8)
    {
        const float* const __restrict pLast = x + len - 8;
        float* const __restrict pSLast = s + len - 8;
        float* const __restrict pCLast = c + len - 8;
        while (x < pLast)
        {
            __m256 rx = _mm256_loadu_ps(x);
            __m256 rs, rc;
            SinCos(rs, rc, rx);
            _mm256_storeu_ps(s, rs);
            _mm256_storeu_ps(c, rc);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 32);
            s = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(s) + 32);
            c = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(c) + 32);
        }
        __m256 rx = _mm256_loadu_ps(pLast);
        __m256 rs, rc;
        SinCos(rs, rc, rx);
        _mm256_storeu_ps(pSLast, rs);
        _mm256_storeu_ps(pCLast, rc);
    }
    else
    {
        alignas(32) float xTemp[8] = {};
        alignas(32) float sTemp[8];
        alignas(32) float cTemp[8];
        for (U64 i = 0; i < len; ++i)
            xTemp[i] = x[i];
        __m256 rx = _mm256_load_ps(xTemp);
        __m256 rs, rc;
        SinCos(rs, rc, rx);
        _mm256_storeu_ps(sTemp, rs);
        _mm256_storeu_ps(cTemp, rc);
        for (U64 i = 0; i < len; ++i)
        {
            s[i] = sTemp[i];
            c[i] = cTemp[i];
        }
    }
}
#endif // __AVX2__

#ifdef NO_AVX512_PLS
#ifdef __AVX512F__
// Single-precision, x86 SIMD using AVX512 for 512-bit vectors.

static inline __m512 __vectorcall Sin(__m512 x)
{
    __m512 cycle = _mm512_fmadd_ps(x, _mm512_set1_ps(2.0f * kInvPif), _mm512_set1_ps(0x1.8p+23));
    const __m512i cycleI = _mm512_castps_si512(cycle);
    const __m512i lastBitMask = _mm512_set1_epi32(1);
    cycle = _mm512_sub_ps(cycle, _mm512_set1_ps(0x1.8p+23));
    const __mmask16 cosMask = _mm512_test_epi32_mask(cycleI, lastBitMask);
    const __m512 signMask = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(cycleI, 1), 31));

    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPif), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi3f), cycle, x);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 b = _mm512_xor_ps(signMask, _mm512_mask_blend_ps(cosMask, x, _mm512_set1_ps(1.0f)));

    __m512 res = _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(2.67107816e-06f), _mm512_set1_ps(2.43944651e-05f));
    res = _mm512_fmadd_ps(res, x2, _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(-1.98358801e-04f), _mm512_set1_ps(-1.38866133e-03f)));
    res = _mm512_fmadd_ps(res, x2, _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(8.33333470e-03f), _mm512_set1_ps(4.16666158e-02f)));
    res = _mm512_fmadd_ps(res, x2, _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(-1.66666672e-01f), _mm512_set1_ps(-5.00000000e-01f)));
    res = _mm512_fmadd_ps(res, _mm512_mul_ps(b, x2), b);
    return res;
}

static inline __m512 __vectorcall Cos(__m512 x)
{
    __m512 cycle = _mm512_fmadd_ps(x, _mm512_set1_ps(2.0f * kInvPif), _mm512_set1_ps(0x1.8p+23));
    const __m512i cycleI = _mm512_castps_si512(cycle);
    const __m512i lastBitMask = _mm512_set1_epi32(1);
    cycle = _mm512_sub_ps(cycle, _mm512_set1_ps(0x1.8p+23));
    const __mmask16 sinMask = _mm512_test_epi32_mask(cycleI, lastBitMask);
    const __m512 signMask =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_xor_epi32(cycleI, _mm512_srli_epi32(cycleI, 1)), 31));

    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPif), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi3f), cycle, x);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 b = _mm512_xor_ps(signMask, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(1.0f), x));

    __m512 res = _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(2.43944651e-05f), _mm512_set1_ps(2.67107816e-06f));
    res = _mm512_fmadd_ps(res, x2, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(-1.38866133e-03f), _mm512_set1_ps(-1.98358801e-04f)));
    res = _mm512_fmadd_ps(res, x2, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(4.16666158e-02f), _mm512_set1_ps(8.33333470e-03f)));
    res = _mm512_fmadd_ps(res, x2, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(-5.00000000e-01f), _mm512_set1_ps(-1.66666672e-01f)));
    res = _mm512_fmadd_ps(res, _mm512_mul_ps(b, x2), b);
    return res;
}

static inline void __vectorcall SinCos(__m512& s, __m512& c, __m512 x)
{
    __m512 cycle = _mm512_fmadd_ps(x, _mm512_set1_ps(2.0f * kInvPif), _mm512_set1_ps(0x1.8p+23));
    const __m512i cycleI = _mm512_castps_si512(cycle);
    const __m512i lastBitMask = _mm512_set1_epi32(1);
    cycle = _mm512_sub_ps(cycle, _mm512_set1_ps(0x1.8p+23));
    const __mmask16 cosMask = _mm512_test_epi32_mask(cycleI, lastBitMask);
    const __m512 sinSignMask =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_xor_epi32(cycleI, _mm512_srli_epi32(cycleI, 1)), 31));
    const __m512 cosSignMask = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(cycleI, 1), 31));

    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPif), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi3f), cycle, x);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 bs = _mm512_xor_ps(sinSignMask, x);
    __m512 bc = _mm512_xor_ps(cosSignMask, _mm512_set1_ps(1.0f));

    __m512 s1 = _mm512_set1_ps(2.67107816e-06f);
    __m512 c1 = _mm512_set1_ps(2.43944651e-05f);
    s1 = _mm512_fmadd_ps(s1, x2, _mm512_set1_ps(-1.98358801e-04f));
    c1 = _mm512_fmadd_ps(c1, x2, _mm512_set1_ps(-1.38866133e-03f));
    s1 = _mm512_fmadd_ps(s1, x2, _mm512_set1_ps(8.33333470e-03f));
    c1 = _mm512_fmadd_ps(c1, x2, _mm512_set1_ps(4.16666158e-02f));
    s1 = _mm512_fmadd_ps(s1, x2, _mm512_set1_ps(-1.66666672e-01f));
    c1 = _mm512_fmadd_ps(c1, x2, _mm512_set1_ps(-5.00000000e-01f));
    s1 = _mm512_fmadd_ps(s1, _mm512_mul_ps(bs, x2), bs);
    c1 = _mm512_fmadd_ps(c1, _mm512_mul_ps(bc, x2), bc);
    s = _mm512_mask_blend_ps(cosMask, s1, c1);
    c = _mm512_mask_blend_ps(cosMask, c1, s1);
}

static inline __m512 __vectorcall Tan(__m512 x)
{
    __m512 s, c;
    SinCos(s, c, x);
    return _mm512_div_ps(s, c);
}

static inline __m512 __vectorcall Asin(__m512 x)
{
    const __m512 xAbs = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(
        _mm512_castps_si512(x), _mm512_setzero_si512(), _mm512_set1_epi32(0x8000'0000U), 0x50));
    __m512 xOuter = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), xAbs, _mm512_set1_ps(2.0f));
    const __mmask16 outerMask = _mm512_cmp_ps_mask(_mm512_set1_ps(0.5f), xAbs, _CMP_LT_OQ);
    const __m512 signBit = _mm512_and_ps(x, _mm512_set1_ps(-0.0f));
    const __m512 flippedSignBit = _mm512_xor_ps(signBit, _mm512_set1_ps(-0.0f));
    const __m512 p = _mm512_mask_blend_ps(outerMask, _mm512_mul_ps(x, x), xOuter);
    const __m512 lastMul = _mm512_mask_blend_ps(outerMask, x, _mm512_xor_ps(flippedSignBit, _mm512_sqrt_ps(xOuter)));
    __m512 yPoly = _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[4]), _mm512_set1_ps(kAcosf[4]));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[3]), _mm512_set1_ps(kAcosf[3])));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[2]), _mm512_set1_ps(kAcosf[2])));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[1]), _mm512_set1_ps(kAcosf[1])));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[0]), _mm512_set1_ps(kAcosf[0])));
    //
    // can bring this back to clean things up a bit for the edges of the acos
    yPoly = _mm512_mul_ps(yPoly, p);
    const __m512 lastAdd =
        _mm512_mask_blend_ps(outerMask, x, _mm512_add_ps(_mm512_xor_ps(signBit, _mm512_set1_ps(0.5f * kPif)), lastMul));
    return _mm512_fmadd_ps(yPoly, lastMul, lastAdd);
}

static inline __m512 __vectorcall Acos(__m512 x)
{
    const __m512 xAbs = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x);
    __m512 xOuter = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), xAbs, _mm512_set1_ps(2.0f));
    const __mmask16 outerMask = _mm512_cmp_ps_mask(_mm512_set1_ps(0.5f), xAbs, _CMP_LT_OQ);
    const __m512 signBit = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(_mm512_castps_si512(x), 31), 31));
    //const __mmask16 signMask = _mm512_movepi32_mask(_mm512_castps_si512(signBit));
    //x = _mm512_mask_blend_ps(
    //    outerMask, xAbs, _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), xAbs, _mm512_set1_ps(2.0f)));
    const __m512 p = _mm512_mask_blend_ps(outerMask, _mm512_mul_ps(x, x), xOuter);
    const __m512 lastMul = _mm512_mask_blend_ps(outerMask, x, _mm512_xor_ps(signBit, _mm512_sqrt_ps(xOuter)));
    __m512 yPoly = _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[4]), _mm512_set1_ps(kAcosf[4]));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[3]), _mm512_set1_ps(kAcosf[3])));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[2]), _mm512_set1_ps(kAcosf[2])));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[1]), _mm512_set1_ps(kAcosf[1])));
    yPoly = _mm512_fmadd_ps(
        yPoly, p, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[0]), _mm512_set1_ps(kAcosf[0])));
    //yPoly = _mm512_fmadd_ps(yPoly, p, _mm512_set1_ps(1.0f));
    //return _mm512_mul_ps(yPoly, lastMul);
    const __m512 lastP = _mm512_mask_xor_ps(p, _mm512_knot(outerMask), _mm512_set1_ps(-0.0f), p);
    yPoly = _mm512_fmadd_ps(yPoly, lastP, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(-1.0f), _mm512_setzero_ps()));
    const __m512 lastAdd = _mm512_mask_blend_ps(
        outerMask,
        _mm512_set1_ps(0.5f * kPif),
        _mm512_add_ps(lastMul,
                      _mm512_and_ps(_mm512_set1_ps(kPif),
                                    _mm512_castsi512_ps(_mm512_srai_epi32(_mm512_castps_si512(signBit), 31)))));
    return _mm512_fmadd_ps(yPoly, lastMul, lastAdd);
}

static inline __m512 __vectorcall Atanf_Dumb1(__m512 x)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 sign = _mm512_and_ps(x, _mm512_set1_ps(-0.0f));
    x = _mm512_abs_ps(x);
    const __mmask16 farMask = _mm512_cmp_ps_mask(x, one, _CMP_GE_OQ);
    x = _mm512_mask_blend_ps(farMask, x, _mm512_div_ps(one, x));
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 y = _mm512_set1_ps(-1.472577981985523e-03f);
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(9.392884421686176e-03f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(-2.814183884379418e-02f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(5.464641870671016e-02f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(-8.188949902975562e-02f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(1.086826130424285e-01f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(-1.424497268154095e-01f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(1.999615551552361e-01f));
    y = _mm512_fmadd_ps(y, x2, _mm512_set1_ps(-3.333316439068992e-01f));
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(x2, x), x);

    const __m512 ySign = _mm512_mask_blend_ps(farMask, _mm512_set1_ps(1.0f), _mm512_set1_ps(-1.0f));
    const __m512 base = _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPif));
    const __m512 base2 = _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPi2f));
    const __m512 r = _mm512_add_ps(_mm512_fmadd_ps(ySign, y, base2), base);
    const __m512 res = _mm512_xor_ps(r, sign);
    return res;
}

static inline __m512 __vectorcall Atanf_Dumb2(__m512 x)
{
    const __m512 sign = _mm512_and_ps(x, _mm512_set1_ps(-0.0f));
    x = _mm512_xor_ps(x, sign);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 rx = _mm512_div_ps(one, x);
    const __mmask16 farMask = _mm512_cmp_ps_mask(x, one, _CMP_GE_OQ);
    x = _mm512_mask_blend_ps(farMask, x, rx);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 num = _mm512_set1_ps(1.70646477e-02f);
    num = _mm512_fmadd_ps(num, x2, _mm512_set1_ps(3.95497262e-01f));
    num = _mm512_fmadd_ps(num, x2, _mm512_set1_ps(1.27574670e+00f));
    num = _mm512_fmadd_ps(num, _mm512_mul_ps(x2, x), x);
    __m512 den = _mm512_set1_ps(8.18777159e-02f);
    den = _mm512_fmadd_ps(den, x2, _mm512_set1_ps(7.31906652e-01f));
    den = _mm512_fmadd_ps(den, x2, _mm512_set1_ps(1.60907602e+00f));
    den = _mm512_fmadd_ps(den, x2, _mm512_set1_ps(1.0f));
    const __m512 y = _mm512_div_ps(num, den);

    const __m512 ySign = _mm512_mask_blend_ps(farMask, _mm512_set1_ps(1.0f), _mm512_set1_ps(-1.0f));
    const __m512 base = _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPif));
    const __m512 base2 = _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPi2f));
    const __m512 r = _mm512_add_ps(_mm512_fmadd_ps(ySign, y, base2), base);
    const __m512 res = _mm512_xor_ps(r, sign);
    return res;
}
#endif // __AVX512F__


#ifdef __AVX512F__
// Template definitions for single precision, statically-sized arrays.

#define STATIC_LENGTH_IMPL(fxn)                                                                                        \
    template <U64 len> void fxn(float* __restrict out, const float* __restrict x)                                      \
    {                                                                                                                  \
        if constexpr (len <= 16)                                                                                       \
        {                                                                                                              \
            __mmask16 mask = (1U << len) - 1U;                                                                         \
            __m512 X = _mm512_setzero_ps();                                                                            \
            X = _mm512_mask_loadu_ps(X, mask, x);                                                                      \
            __m512 y = fxn(X);                                                                                         \
            _mm512_mask_storeu_ps(out, mask, y);                                                                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            constexpr U32 nFull = len >> 4;                                                                            \
            constexpr U32 nLast = len & 15;                                                                            \
            for (U32 i = 0; i < nFull; ++i)                                                                            \
            {                                                                                                          \
                __m512 X = _mm512_loadu_ps(x);                                                                         \
                __m512 y = fxn(X);                                                                                     \
                _mm512_storeu_ps(out, y);                                                                              \
                x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 64);                               \
                out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 64);                                 \
            }                                                                                                          \
            if constexpr (nLast)                                                                                       \
            {                                                                                                          \
                __mmask16 mask = (1U << nLast) - 1U;                                                                   \
                __m512 X = _mm512_setzero_ps();                                                                        \
                X = _mm512_mask_loadu_ps(X, mask, x);                                                                  \
                __m512 y = fxn(X);                                                                                     \
                _mm512_mask_storeu_ps(out, mask, y);                                                                   \
            }                                                                                                          \
        }                                                                                                              \
    }

STATIC_LENGTH_IMPL(Sinf)
STATIC_LENGTH_IMPL(Cosf)
STATIC_LENGTH_IMPL(Tanf)
STATIC_LENGTH_IMPL(Asinf)
STATIC_LENGTH_IMPL(Acosf)

template <U64 len> static inline void SinCos(float* __restrict s, float* __restrict c, const float* __restrict x)
{
    if constexpr (len <= 16)
    {
        __mmask16 mask = (1U << len) - 1U;
        __m512 X = _mm512_setzero_ps();
        X = _mm512_mask_loadu_ps(X, mask, x);
        __m512 S, C;
        SinCos(S, C, X);
        _mm512_mask_storeu_ps(s, mask, S);
        _mm512_mask_storeu_ps(c, mask, C);
    }
    else
    {
        constexpr U64 nFull = len >> 4;
        constexpr U64 nLast = len & 15;
        for (U64 i = 0; i < nFull; ++i)
        {
            __m512 X = _mm512_loadu_ps(x);
            __m512 S, C;
            SinCos(S, C, X);
            _mm512_storeu_ps(s, S);
            _mm512_storeu_ps(c, C);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 64);
            s = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(s) + 64);
            c = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(c) + 64);
        }
        if constexpr (nLast)
        {
            __mmask16 mask = (1U << nLast) - 1U;
            __m512 X = _mm512_setzero_ps();
            X = _mm512_mask_loadu_ps(X, mask, x);
            __m512 S, C;
            SinCos(S, C, X);
            _mm512_mask_storeu_ps(s, mask, S);
            _mm512_mask_storeu_ps(c, mask, C);
        }
    }
}

// Single precision, dynamically sized arrays.

#define DYNAMIC_LENGTH_IMPL(fxn)                                                                                       \
    void fxn(float* __restrict out, const float* __restrict x, const U64 len)                                          \
    {                                                                                                                  \
        if (len <= 16)                                                                                                 \
        {                                                                                                              \
            __mmask16 mask = (1U << len) - 1U;                                                                         \
            __m512 X = _mm512_setzero_ps();                                                                            \
            X = _mm512_mask_loadu_ps(X, mask, x);                                                                      \
            __m512 y = fxn(X);                                                                                         \
            _mm512_mask_storeu_ps(out, mask, y);                                                                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            U64 nFull = len >> 4;                                                                                      \
            U64 nLast = len & 15;                                                                                      \
            const float* __restrict pIn = x;                                                                           \
            float* __restrict pOut = out;                                                                              \
            for (U64 i = 0; i < nFull; ++i)                                                                            \
            {                                                                                                          \
                __m512 X = _mm512_loadu_ps(pIn);                                                                       \
                __m512 y = fxn(X);                                                                                     \
                _mm512_storeu_ps(pOut, y);                                                                             \
                pIn = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pIn) + 64);                           \
                pOut = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(pOut) + 64);                               \
            }                                                                                                          \
            if (nLast)                                                                                                 \
            {                                                                                                          \
                __mmask16 mask = (1U << nLast) - 1U;                                                                   \
                __m512 X = _mm512_setzero_ps();                                                                        \
                X = _mm512_mask_loadu_ps(X, mask, pIn);                                                                \
                __m512 y = fxn(X);                                                                                     \
                _mm512_mask_storeu_ps(pOut, mask, y);                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    }

DYNAMIC_LENGTH_IMPL(Sinf);
DYNAMIC_LENGTH_IMPL(Cosf);
DYNAMIC_LENGTH_IMPL(Tanf);
DYNAMIC_LENGTH_IMPL(Asinf);
DYNAMIC_LENGTH_IMPL(Acosf);

static inline void SinCos(float* __restrict s, float* __restrict c, const float* __restrict x, U64 len)
{
    if (len <= 16)
    {
        __mmask16 mask = (1U << len) - 1U;
        __m512 X = _mm512_setzero_ps();
        X = _mm512_mask_loadu_ps(X, mask, x);
        __m512 S, C;
        SinCos(S, C, X);
        _mm512_mask_storeu_ps(s, mask, S);
        _mm512_mask_storeu_ps(c, mask, C);
    }
    else
    {
        U64 nFull = len >> 4;
        U64 nLast = len & 15;
        for (U64 i = 0; i < nFull; ++i)
        {
            __m512 X = _mm512_loadu_ps(x);
            __m512 S, C;
            SinCos(S, C, X);
            _mm512_storeu_ps(s, S);
            _mm512_storeu_ps(c, C);
            x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 64);
            s = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(s) + 64);
            c = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(c) + 64);
        }
        if (nLast)
        {
            __mmask16 mask = (1U << nLast) - 1U;
            __m512 X = _mm512_setzero_ps();
            X = _mm512_mask_loadu_ps(X, mask, x);
            __m512 S, C;
            SinCos(S, C, X);
            _mm512_mask_storeu_ps(s, mask, S);
            _mm512_mask_storeu_ps(c, mask, C);
        }
    }
}
#endif // __AVX512F__
#endif // NO_AVX512_PLS