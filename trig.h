#pragma once

#include <immintrin.h>
#include <cstdint>
#include "common.h"

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

// Single precision, scalar.
float Sinf(float x);
float Cosf(float x);
float Tanf(float x);
void SinCosf(float& s, float& c, float x);
float Asinf(float x);
float Acosf(float x);
float Atanf(float x);
float Atan2f(float y, float x);

// Double precision, scalar.
double Sind(double x);
double Cosd(double x);
double Tand(double x);
void SinCosd(double& s, double& c, double x);
double Asind(double x);
double Acosd(double x);
double Atand(double x);
double Atan2d(double y, double x);

// Single precision, statically-sized arrays.
template <U64 len> void Sinf(float* __restrict out, const float* __restrict x);
template <U64 len> void Cosf(float* __restrict out, const float* __restrict x);
template <U64 len> void Tanf(float* __restrict out, const float* __restrict x);
template <U64 len>
void SinCosf(float* __restrict s, float* __restrict c, const float* __restrict x);
template <U64 len> void Asinf(float* __restrict out, const float* __restrict x);
template <U64 len> void Acosf(float* __restrict out, const float* __restrict x);
template <U64 len> void Atanf(float* __restrict out, const float* __restrict x);
template <U64 len>
void Atan2f(float* __restrict out, const float* __restrict y, const float* __restrict x);

// Double precision, statically-sized arrays.
template <U64 len> void Sind(double* __restrict out, const double* __restrict x);
template <U64 len> void Cosd(double* __restrict out, const double* __restrict x);
template <U64 len> void Tand(double* __restrict out, const double* __restrict x);
template <U64 len>
void SinCosd(double* __restrict s, double* __restrict c, const double* __restrict x);
template <U64 len> void Asind(double* __restrict out, const double* __restrict x);
template <U64 len> void Acosd(double* __restrict out, const double* __restrict x);
template <U64 len> void Atand(double* __restrict out, const double* __restrict x);
template <U64 len>
void Atan2d(double* __restrict out, const double* __restrict y, const double* __restrict x);

// Single precision, dynamically-sized arrays.
void Sinf(float* __restrict out, const float* __restrict x, U64 len);
void Cosf(float* __restrict out, const float* __restrict x, U64 len);
void Tanf(float* __restrict out, const float* __restrict x, U64 len);
void SinCosf(float* __restrict s, float* __restrict c, const float* __restrict x, U64 len);
void Asinf(float* __restrict out, const float* __restrict x, U64 len);
void Acosf(float* __restrict out, const float* __restrict x, U64 len);
void Atanf(float* __restrict out, const float* __restrict x, U64 len);
void Atan2f(float* __restrict out, const float* __restrict y, const float* __restrict x, U64 len);

// Double precision, dynamically-sized arrays.
void Sind(double* __restrict out, const double* __restrict x, U64 len);
void Cosd(double* __restrict out, const double* __restrict x, U64 len);
void Tand(double* __restrict out, const double* __restrict x, U64 len);
void SinCosd(double* __restrict s, double* __restrict c, const double* __restrict x, U64 len);
void Asind(double* __restrict out, const double* __restrict x, U64 len);
void Acosd(double* __restrict out, const double* __restrict x, U64 len);
void Atand(double* __restrict out, const double* __restrict x, U64 len);
void Atan2d(double* __restrict out,
            const double* __restrict y,
            const double* __restrict x,
            U64 len);

#ifdef __AVX2__
// Single precision, AVX2 SIMD for x86-64 CPUs.
__m128 __vectorcall Sinf(__m128 x);
__m128 __vectorcall Cosf(__m128 x);
__m128 __vectorcall Tanf(__m128 x);
void __vectorcall SinCosf(__m128& s, __m128& c, __m128 x);
__m128 __vectorcall Asinf(__m128 x);
__m128 __vectorcall Acosf(__m128 x);
__m128 __vectorcall Atanf(__m128 x);
__m128 __vectorcall Atan2f(__m128 y, __m128 x);

__m256 __vectorcall Sinf(__m256 x);
__m256 __vectorcall Cosf(__m256 x);
__m256 __vectorcall Tanf(__m256 x);
void __vectorcall SinCosf(__m256& s, __m256& c, __m256 x);
__m256 __vectorcall Asinf(__m256 x);
__m256 __vectorcall Acosf(__m256 x);
__m256 __vectorcall Atanf(__m256 x);
__m256 __vectorcall Atan2f(__m256 y, __m256 x);

// Double precision, AVX2 SIMD for x86-64 CPUs.
__m128d __vectorcall Sind(__m128d x);
__m128d __vectorcall Cosd(__m128d x);
__m128d __vectorcall Tand(__m128d x);
void __vectorcall SinCosd(__m128d& s, __m128d& c, __m128d x);
__m128d __vectorcall Asind(__m128d x);
__m128d __vectorcall Acosd(__m128d x);
__m128d __vectorcall Atand(__m128d x);
__m128d __vectorcall Atan2d(__m128d y, __m128d x);

__m256d __vectorcall Sind(__m256d x);
__m256d __vectorcall Cosd(__m256d x);
__m256d __vectorcall Tand(__m256d x);
void __vectorcall SinCosd(__m256d& s, __m256d& c, __m256d x);
__m256d __vectorcall Asind(__m256d x);
__m256d __vectorcall Acosd(__m256d x);
__m256d __vectorcall Atand(__m256d x);
__m256d __vectorcall Atan2d(__m256d y, __m256d x);
#endif // __AVX2__

#ifdef __AVX512F__
// Single precision, AVX512 SIMD for x86-64 CPUs.
__m512 __vectorcall Sinf(__m512 x);
__m512 __vectorcall Cosf(__m512 x);
__m512 __vectorcall Tanf(__m512 x);
void __vectorcall SinCosf(__m512& s, __m512& c, __m512 x);
__m512 __vectorcall Asinf(__m512 x);
__m512 __vectorcall Acosf(__m512 x);
__m512 __vectorcall Atanf(__m512 x);
__m512 __vectorcall Atan2f(__m512 y, __m512 x);

// Double precision, AVX512 SIMD for x86-64 CPUs.
__m512d __vectorcall Sind(__m512d x);
__m512d __vectorcall Cosd(__m512d x);
__m512d __vectorcall Tand(__m512d x);
void __vectorcall SinCosd(__m512d& s, __m512d& c, __m512d x);
__m512d __vectorcall Asind(__m512d x);
__m512d __vectorcall Acosd(__m512d x);
__m512d __vectorcall Atand(__m512d x);
__m512d __vectorcall Atan2d(__m512d y, __m512d x);
#endif // __AVX512F__


// Single-precision coefficients for sin and cos. Derived from Chebyshev interpolants over the interval (-pi/4, pi/4).
static constexpr U32 kNumSinf = 4;
static constexpr float kSinf[kNumSinf]{
    8.33333470e-03f, -1.98358801e-04f, 2.67107816e-06f
};
static constexpr float kCosf[kNumSinf]{
    -5.00000000e-01f, 4.16666158e-02f, -1.38866133e-03f, 2.43944651e-05f
};

#ifdef __AVX2__
// Single precision, scalar. Currently only supported for x86 CPUs with AVX2.
float Sinf(float x)
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

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m128 res = _mm_blendv_ps(_mm_set_ss(kSinf[3]), _mm_set_ss(kCosf[3]), cosMask);
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(kSinf[2]), _mm_set_ss(kCosf[2]), cosMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(kSinf[1]), _mm_set_ss(kCosf[1]), cosMask));
    res = _mm_fmadd_ss(
        res, x2, _mm_blendv_ps(_mm_set_ss(-1.66666672e-01f), _mm_set_ss(kCosf[0]), cosMask));
    res = _mm_fmadd_ss(res, _mm_mul_ss(b, x2), b);
    return _mm_cvtss_f32(res);
}

float Cosf(float x)
{
    __m128 X = _mm_set_ss(x);
    __m128 cycle = _mm_fmadd_ss(X, _mm_set_ss(2.0f * kInvPif), _mm_set_ss(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ss(cycle, _mm_set_ss(0x1.8p+23));
    const __m128 sinMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask =
        _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(cycleI, _mm_srli_epi32(cycleI, 1)), 31));

    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPif), cycle, X);
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPi2f), cycle, X);
    X = _mm_fnmadd_ss(_mm_set_ss(0.5f * kPi3f), cycle, X);
    const __m128 x2 = _mm_mul_ss(X, X);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(_mm_set_ss(1.0f), X, sinMask));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m128 res = _mm_blendv_ps(_mm_set_ss(kCosf[3]), _mm_set_ss(kSinf[3]), sinMask);
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(kCosf[2]), _mm_set_ss(kSinf[2]), sinMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(kCosf[1]), _mm_set_ss(kSinf[1]), sinMask));
    res = _mm_fmadd_ss(res, x2, _mm_blendv_ps(_mm_set_ss(kCosf[0]), _mm_set_ss(kSinf[0]), sinMask));
    res = _mm_fmadd_ss(res, _mm_mul_ss(b, x2), b);
    return _mm_cvtss_f32(res);
}

void SinCosf(float& s, float& c, float x)
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

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m128 res = _mm_blendv_ps(_mm_set1_ps(kSinf[3]), _mm_set1_ps(kCosf[3]), cosMask);
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kSinf[2]), _mm_set1_ps(kCosf[2]), cosMask));
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kSinf[1]), _mm_set1_ps(kCosf[1]), cosMask));
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kSinf[0]), _mm_set1_ps(kCosf[0]), cosMask));
    res = _mm_fmadd_ps(res, _mm_mul_ps(b, x2), b);
    s = _mm_cvtss_f32(res);
    c = _mm_cvtss_f32(_mm_shuffle_ps(res, res, 177));
}

float Tanf(float x)
{
    float s, c;
    SinCosf(s, c, x);
    return s / c;
}

// Single-precision, x86 SIMD using AVX2 for 128-bit and 256-bit vectors.

__m128 __vectorcall Sinf(__m128 x)
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

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m128 res = _mm_blendv_ps(_mm_set1_ps(kSinf[3]), _mm_set1_ps(kCosf[3]), cosMask);
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kSinf[2]), _mm_set1_ps(kCosf[2]), cosMask));
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kSinf[1]), _mm_set1_ps(kCosf[1]), cosMask));
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kSinf[0]), _mm_set1_ps(kCosf[0]), cosMask));
    res = _mm_fmadd_ps(res, _mm_mul_ps(b, x2), b);
    return res;
}

__m128 __vectorcall Cosf(__m128 x)
{
    __m128 cycle = _mm_fmadd_ps(x, _mm_set1_ps(2.0f * kInvPif), _mm_set1_ps(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ps(cycle, _mm_set1_ps(0x1.8p+23));
    const __m128 sinMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 signMask =
        _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(cycleI, _mm_srli_epi32(cycleI, 1)), 31));

    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPif), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi3f), cycle, x);
    const __m128 x2 = _mm_mul_ps(x, x);
    __m128 b = _mm_xor_ps(signMask, _mm_blendv_ps(_mm_set1_ps(1.0f), x, sinMask));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m128 res = _mm_blendv_ps(_mm_set1_ps(kCosf[3]), _mm_set1_ps(kSinf[3]), sinMask);
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kCosf[2]), _mm_set1_ps(kSinf[2]), sinMask));
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kCosf[1]), _mm_set1_ps(kSinf[1]), sinMask));
    res =
        _mm_fmadd_ps(res, x2, _mm_blendv_ps(_mm_set1_ps(kCosf[0]), _mm_set1_ps(kSinf[0]), sinMask));
    res = _mm_fmadd_ps(res, _mm_mul_ps(b, x2), b);
    return res;
}

void __vectorcall SinCosf(__m128& s, __m128& c, __m128 x)
{
    __m128 cycle = _mm_fmadd_ps(x, _mm_set1_ps(2.0f * kInvPif), _mm_set1_ps(0x1.8p+23));
    const __m128i cycleI = _mm_castps_si128(cycle);
    cycle = _mm_sub_ps(cycle, _mm_set1_ps(0x1.8p+23));
    const __m128 cosMask = _mm_castsi128_ps(_mm_slli_epi32(cycleI, 31));
    const __m128 sinSignMask =
        _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(cycleI, _mm_srli_epi32(cycleI, 1)), 31));
    const __m128 cosSignMask = _mm_castsi128_ps(_mm_slli_epi32(_mm_srli_epi32(cycleI, 1), 31));

    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPif), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm_fnmadd_ps(_mm_set1_ps(0.5f * kPi3f), cycle, x);
    const __m128 x2 = _mm_mul_ps(x, x);
    __m128 bs = _mm_xor_ps(sinSignMask, x);
    __m128 bc = _mm_xor_ps(cosSignMask, _mm_set1_ps(1.0f));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m128 s1 = _mm_set1_ps(kSinf[3]);
    __m128 c1 = _mm_set1_ps(kCosf[3]);
    s1 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(kSinf[2]));
    c1 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(kCosf[2]));
    s1 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(kSinf[1]));
    c1 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(kCosf[1]));
    s1 = _mm_fmadd_ps(s1, x2, _mm_set1_ps(kSinf[0]));
    c1 = _mm_fmadd_ps(c1, x2, _mm_set1_ps(kCosf[0]));
    s1 = _mm_fmadd_ps(s1, _mm_mul_ps(bs, x2), bs);
    c1 = _mm_fmadd_ps(c1, _mm_mul_ps(bc, x2), bc);
    s = _mm_blendv_ps(s1, c1, cosMask);
    c = _mm_blendv_ps(c1, s1, cosMask);
}

__m128 __vectorcall Tanf(__m128 x)
{
    __m128 s, c;
    SinCosf(s, c, x);
    return _mm_div_ps(s, c);
}

__m256 __vectorcall Sinf(__m256 x)
{
    __m256 cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPif), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256 cosMask = _mm256_castsi256_ps(_mm256_slli_epi32(cycleI, 31));
    const __m256 signMask =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(cycleI, 1), 31));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPif), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi3f), cycle, x);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 b = _mm256_xor_ps(signMask, _mm256_blendv_ps(x, _mm256_set1_ps(1.0f), cosMask));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m256 res = _mm256_blendv_ps(_mm256_set1_ps(kSinf[3]), _mm256_set1_ps(kCosf[3]), cosMask);
    res = _mm256_fmadd_ps(
        res, x2, _mm256_blendv_ps(_mm256_set1_ps(kSinf[2]), _mm256_set1_ps(kCosf[2]), cosMask));
    res = _mm256_fmadd_ps(
        res, x2, _mm256_blendv_ps(_mm256_set1_ps(kSinf[1]), _mm256_set1_ps(kCosf[1]), cosMask));
    res = _mm256_fmadd_ps(
        res, x2, _mm256_blendv_ps(_mm256_set1_ps(kSinf[0]), _mm256_set1_ps(kCosf[0]), cosMask));
    res = _mm256_fmadd_ps(res, _mm256_mul_ps(b, x2), b);
    return res;
}

__m256 __vectorcall Cosf(__m256 x)
{
    __m256 cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPif), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256 sinMask = _mm256_castsi256_ps(_mm256_slli_epi32(cycleI, 31));
    const __m256 signMask = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_xor_si256(cycleI, _mm256_srli_epi32(cycleI, 1)), 31));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPif), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi3f), cycle, x);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 b = _mm256_xor_ps(signMask, _mm256_blendv_ps(_mm256_set1_ps(1.0f), x, sinMask));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m256 res = _mm256_blendv_ps(_mm256_set1_ps(kCosf[3]), _mm256_set1_ps(kSinf[3]), sinMask);
    res = _mm256_fmadd_ps(
        res, x2, _mm256_blendv_ps(_mm256_set1_ps(kCosf[2]), _mm256_set1_ps(kSinf[2]), sinMask));
    res = _mm256_fmadd_ps(
        res, x2, _mm256_blendv_ps(_mm256_set1_ps(kCosf[1]), _mm256_set1_ps(kSinf[1]), sinMask));
    res = _mm256_fmadd_ps(
        res, x2, _mm256_blendv_ps(_mm256_set1_ps(kCosf[0]), _mm256_set1_ps(kSinf[0]), sinMask));
    res = _mm256_fmadd_ps(res, _mm256_mul_ps(b, x2), b);
    return res;
}

void __vectorcall SinCosf(__m256& s, __m256& c, __m256 x)
{
    __m256 cycle = _mm256_fmadd_ps(x, _mm256_set1_ps(2.0f * kInvPif), _mm256_set1_ps(0x1.8p+23));
    const __m256i cycleI = _mm256_castps_si256(cycle);
    cycle = _mm256_sub_ps(cycle, _mm256_set1_ps(0x1.8p+23));
    const __m256 cosMask = _mm256_castsi256_ps(_mm256_slli_epi32(cycleI, 31));
    const __m256 sinSignMask = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_xor_si256(cycleI, _mm256_srli_epi32(cycleI, 1)), 31));
    const __m256 cosSignMask =
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(cycleI, 1), 31));

    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPif), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm256_fnmadd_ps(_mm256_set1_ps(0.5f * kPi3f), cycle, x);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 bs = _mm256_xor_ps(sinSignMask, x);
    __m256 bc = _mm256_xor_ps(cosSignMask, _mm256_set1_ps(1.0f));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m256 s1 = _mm256_set1_ps(kSinf[3]);
    __m256 c1 = _mm256_set1_ps(kCosf[3]);
    s1 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(kSinf[2]));
    c1 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(kCosf[2]));
    s1 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(kSinf[1]));
    c1 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(kCosf[1]));
    s1 = _mm256_fmadd_ps(s1, x2, _mm256_set1_ps(kSinf[0]));
    c1 = _mm256_fmadd_ps(c1, x2, _mm256_set1_ps(kCosf[0]));
    s1 = _mm256_fmadd_ps(s1, _mm256_mul_ps(bs, x2), bs);
    c1 = _mm256_fmadd_ps(c1, _mm256_mul_ps(bc, x2), bc);
    s = _mm256_blendv_ps(s1, c1, cosMask);
    c = _mm256_blendv_ps(c1, s1, cosMask);
}

__m256 __vectorcall Tanf(__m256 x)
{
    __m256 s, c;
    SinCosf(s, c, x);
    return _mm256_div_ps(s, c);
}
#endif // __AVX2__

#ifdef __AVX512F__
// Single-precision, x86 SIMD using AVX512 for 512-bit vectors.

__m512 __vectorcall Sinf(__m512 x)
{
    __m512 cycle = _mm512_fmadd_ps(x, _mm512_set1_ps(2.0f * kInvPif), _mm512_set1_ps(0x1.8p+23));
    const __m512i cycleI = _mm512_castps_si512(cycle);
    const __m512i lastBitMask = _mm512_set1_epi32(1);
    cycle = _mm512_sub_ps(cycle, _mm512_set1_ps(0x1.8p+23));
    const __mmask16 cosMask = _mm512_test_epi32_mask(cycleI, lastBitMask);
    const __m512 signMask =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(cycleI, 1), 31));

    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPif), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi3f), cycle, x);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 b = _mm512_xor_ps(signMask, _mm512_mask_blend_ps(cosMask, x, _mm512_set1_ps(1.0f)));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m512 res = _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(kSinf[3]), _mm512_set1_ps(kCosf[3]));
    res = _mm512_fmadd_ps(
        res, x2, _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(kSinf[2]), _mm512_set1_ps(kCosf[2])));
    res = _mm512_fmadd_ps(
        res, x2, _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(kSinf[1]), _mm512_set1_ps(kCosf[1])));
    res = _mm512_fmadd_ps(
        res, x2, _mm512_mask_blend_ps(cosMask, _mm512_set1_ps(kSinf[0]), _mm512_set1_ps(kCosf[0])));
    res = _mm512_fmadd_ps(res, _mm512_mul_ps(b, x2), b);
    return res;
}

__m512 __vectorcall Cosf(__m512 x)
{
    __m512 cycle = _mm512_fmadd_ps(x, _mm512_set1_ps(2.0f * kInvPif), _mm512_set1_ps(0x1.8p+23));
    const __m512i cycleI = _mm512_castps_si512(cycle);
    const __m512i lastBitMask = _mm512_set1_epi32(1);
    cycle = _mm512_sub_ps(cycle, _mm512_set1_ps(0x1.8p+23));
    const __mmask16 sinMask = _mm512_test_epi32_mask(cycleI, lastBitMask);
    const __m512 signMask = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_xor_epi32(cycleI, _mm512_srli_epi32(cycleI, 1)), 31));

    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPif), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi3f), cycle, x);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 b = _mm512_xor_ps(signMask, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(1.0f), x));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m512 res = _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(kCosf[3]), _mm512_set1_ps(kSinf[3]));
    res = _mm512_fmadd_ps(
        res, x2, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(kCosf[2]), _mm512_set1_ps(kSinf[2])));
    res = _mm512_fmadd_ps(
        res, x2, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(kCosf[1]), _mm512_set1_ps(kSinf[1])));
    res = _mm512_fmadd_ps(
        res, x2, _mm512_mask_blend_ps(sinMask, _mm512_set1_ps(kCosf[0]), _mm512_set1_ps(kSinf[0])));
    res = _mm512_fmadd_ps(res, _mm512_mul_ps(b, x2), b);
    return res;
}

void __vectorcall SinCosf(__m512& s, __m512& c, __m512 x)
{
    __m512 cycle = _mm512_fmadd_ps(x, _mm512_set1_ps(2.0f * kInvPif), _mm512_set1_ps(0x1.8p+23));
    const __m512i cycleI = _mm512_castps_si512(cycle);
    const __m512i lastBitMask = _mm512_set1_epi32(1);
    cycle = _mm512_sub_ps(cycle, _mm512_set1_ps(0x1.8p+23));
    const __mmask16 cosMask = _mm512_test_epi32_mask(cycleI, lastBitMask);
    const __m512 sinSignMask = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_xor_epi32(cycleI, _mm512_srli_epi32(cycleI, 1)), 31));
    const __m512 cosSignMask =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(cycleI, 1), 31));

    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPif), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi2f), cycle, x);
    x = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f * kPi3f), cycle, x);
    const __m512 x2 = _mm512_mul_ps(x, x);
    __m512 bs = _mm512_xor_ps(sinSignMask, x);
    __m512 bc = _mm512_xor_ps(cosSignMask, _mm512_set1_ps(1.0f));

    static_assert(kNumSinf == 4, "Trig implementation expects 4 coefficients.");
    __m512 s1 = _mm512_set1_ps(kSinf[3]);
    __m512 c1 = _mm512_set1_ps(kCosf[3]);
    s1 = _mm512_fmadd_ps(s1, x2, _mm512_set1_ps(kSinf[2]));
    c1 = _mm512_fmadd_ps(c1, x2, _mm512_set1_ps(kCosf[2]));
    s1 = _mm512_fmadd_ps(s1, x2, _mm512_set1_ps(kSinf[1]));
    c1 = _mm512_fmadd_ps(c1, x2, _mm512_set1_ps(kCosf[1]));
    s1 = _mm512_fmadd_ps(s1, x2, _mm512_set1_ps(kSinf[0]));
    c1 = _mm512_fmadd_ps(c1, x2, _mm512_set1_ps(kCosf[0]));
    s1 = _mm512_fmadd_ps(s1, _mm512_mul_ps(bs, x2), bs);
    c1 = _mm512_fmadd_ps(c1, _mm512_mul_ps(bc, x2), bc);
    s = _mm512_mask_blend_ps(cosMask, s1, c1);
    c = _mm512_mask_blend_ps(cosMask, c1, s1);
}

__m512 __vectorcall Tanf(__m512 x)
{
    __m512 s, c;
    SinCosf(s, c, x);
    return _mm512_div_ps(s, c);
}

__m512 __vectorcall Asinf(__m512 x)
{
    const __m512 xAbs = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(
        _mm512_castps_si512(x), _mm512_setzero_si512(), _mm512_set1_epi32(0x8000'0000U), 0x50));
    __m512 xOuter = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), xAbs, _mm512_set1_ps(2.0f));
    const __mmask16 outerMask = _mm512_cmp_ps_mask(_mm512_set1_ps(0.5f), xAbs, _CMP_LT_OQ);
    const __m512 signBit = _mm512_and_ps(x, _mm512_set1_ps(-0.0f));
    const __m512 flippedSignBit = _mm512_xor_ps(signBit, _mm512_set1_ps(-0.0f));
    const __m512 p = _mm512_mask_blend_ps(outerMask, _mm512_mul_ps(x, x), xOuter);
    const __m512 lastMul =
        _mm512_mask_blend_ps(outerMask, x, _mm512_xor_ps(flippedSignBit, _mm512_sqrt_ps(xOuter)));
    __m512 yPoly =
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[4]), _mm512_set1_ps(kAcosf[4]));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[3]), _mm512_set1_ps(kAcosf[3])));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[2]), _mm512_set1_ps(kAcosf[2])));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[1]), _mm512_set1_ps(kAcosf[1])));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[0]), _mm512_set1_ps(kAcosf[0])));
    //
    // can bring this back to clean things up a bit for the edges of the acos
    yPoly = _mm512_mul_ps(yPoly, p);
    const __m512 lastAdd = _mm512_mask_blend_ps(
        outerMask, x, _mm512_add_ps(_mm512_xor_ps(signBit, _mm512_set1_ps(0.5f * kPif)), lastMul));
    return _mm512_fmadd_ps(yPoly, lastMul, lastAdd);
}

__m512 __vectorcall Acosf(__m512 x)
{
    const __m512 xAbs = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x);
    __m512 xOuter = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), xAbs, _mm512_set1_ps(2.0f));
    const __mmask16 outerMask = _mm512_cmp_ps_mask(_mm512_set1_ps(0.5f), xAbs, _CMP_LT_OQ);
    const __m512 signBit =
        _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(_mm512_castps_si512(x), 31), 31));
    //const __mmask16 signMask = _mm512_movepi32_mask(_mm512_castps_si512(signBit));
    //x = _mm512_mask_blend_ps(
    //    outerMask, xAbs, _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), xAbs, _mm512_set1_ps(2.0f)));
    const __m512 p = _mm512_mask_blend_ps(outerMask, _mm512_mul_ps(x, x), xOuter);
    const __m512 lastMul =
        _mm512_mask_blend_ps(outerMask, x, _mm512_xor_ps(signBit, _mm512_sqrt_ps(xOuter)));
    __m512 yPoly =
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[4]), _mm512_set1_ps(kAcosf[4]));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[3]), _mm512_set1_ps(kAcosf[3])));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[2]), _mm512_set1_ps(kAcosf[2])));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[1]), _mm512_set1_ps(kAcosf[1])));
    yPoly = _mm512_fmadd_ps(
        yPoly,
        p,
        _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(kAsinf[0]), _mm512_set1_ps(kAcosf[0])));
    //yPoly = _mm512_fmadd_ps(yPoly, p, _mm512_set1_ps(1.0f));
    //return _mm512_mul_ps(yPoly, lastMul);
    const __m512 lastP = _mm512_mask_xor_ps(p, _mm512_knot(outerMask), _mm512_set1_ps(-0.0f), p);
    yPoly = _mm512_fmadd_ps(
        yPoly, lastP, _mm512_mask_blend_ps(outerMask, _mm512_set1_ps(-1.0f), _mm512_setzero_ps()));
    const __m512 lastAdd =
        _mm512_mask_blend_ps(outerMask,
                             _mm512_set1_ps(0.5f * kPif),
                             _mm512_add_ps(lastMul,
                                           _mm512_and_ps(_mm512_set1_ps(kPif),
                                                         _mm512_castsi512_ps(_mm512_srai_epi32(
                                                             _mm512_castps_si512(signBit), 31)))));
    return _mm512_fmadd_ps(yPoly, lastMul, lastAdd);
}

__m512 __vectorcall Atanf_Dumb1(__m512 x)
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
    const __m512 base =
        _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPif));
    const __m512 base2 =
        _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPi2f));
    const __m512 r = _mm512_add_ps(_mm512_fmadd_ps(ySign, y, base2), base);
    const __m512 res = _mm512_xor_ps(r, sign);
    return res;
}

__m512 __vectorcall Atanf_Dumb2(__m512 x)
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
    const __m512 base =
        _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPif));
    const __m512 base2 =
        _mm512_mask_blend_ps(farMask, _mm512_setzero_ps(), _mm512_set1_ps(0.5f * kPi2f));
    const __m512 r = _mm512_add_ps(_mm512_fmadd_ps(ySign, y, base2), base);
    const __m512 res = _mm512_xor_ps(r, sign);
    return res;
}
#endif // __AVX512F__


#ifdef  __AVX512F__
// Template definitions for single precision, statically-sized arrays.

#define STATIC_LENGTH_IMPL(fxn)                                                                    \
    template <U64 len> void fxn(float* __restrict out, const float* __restrict x)                  \
    {                                                                                              \
        if constexpr (len <= 16)                                                                   \
        {                                                                                          \
            __mmask16 mask = (1U << len) - 1U;                                                     \
            __m512 X = _mm512_setzero_ps();                                                        \
            X = _mm512_mask_loadu_ps(X, mask, x);                                                  \
            __m512 y = fxn(X);                                                                     \
            _mm512_mask_storeu_ps(out, mask, y);                                                   \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            constexpr U32 nFull = len >> 4;                                                        \
            constexpr U32 nLast = len & 15;                                                        \
            for (U32 i = 0; i < nFull; ++i)                                                        \
            {                                                                                      \
                __m512 X = _mm512_loadu_ps(x);                                                     \
                __m512 y = fxn(X);                                                                 \
                _mm512_storeu_ps(out, y);                                                          \
                x = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(x) + 64);           \
                out = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(out) + 64);             \
            }                                                                                      \
            if constexpr (nLast)                                                                   \
            {                                                                                      \
                __mmask16 mask = (1U << nLast) - 1U;                                               \
                __m512 X = _mm512_setzero_ps();                                                    \
                X = _mm512_mask_loadu_ps(X, mask, x);                                              \
                __m512 y = fxn(X);                                                                 \
                _mm512_mask_storeu_ps(out, mask, y);                                               \
            }                                                                                      \
        }                                                                                          \
    }

STATIC_LENGTH_IMPL(Sinf)
STATIC_LENGTH_IMPL(Cosf)
STATIC_LENGTH_IMPL(Tanf)
STATIC_LENGTH_IMPL(Asinf)
STATIC_LENGTH_IMPL(Acosf)

template <U64 len> void SinCosf(float* __restrict s, float* __restrict c, const float* __restrict x)
{
    if constexpr (len <= 16)
    {
        __mmask16 mask = (1U << len) - 1U;
        __m512 X = _mm512_setzero_ps();
        X = _mm512_mask_loadu_ps(X, mask, x);
        __m512 S, C;
        SinCosf(S, C, X);
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
            SinCosf(S, C, X);
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
            SinCosf(S, C, X);
            _mm512_mask_storeu_ps(s, mask, S);
            _mm512_mask_storeu_ps(c, mask, C);
        }
    }
}

// Single precision, dynamically sized arrays.

#define DYNAMIC_LENGTH_IMPL(fxn)                                                                   \
    void fxn(float* __restrict out, const float* __restrict x, const U64 len)                      \
    {                                                                                              \
        if (len <= 16)                                                                             \
        {                                                                                          \
            __mmask16 mask = (1U << len) - 1U;                                                     \
            __m512 X = _mm512_setzero_ps();                                                        \
            X = _mm512_mask_loadu_ps(X, mask, x);                                                  \
            __m512 y = fxn(X);                                                                     \
            _mm512_mask_storeu_ps(out, mask, y);                                                   \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            U64 nFull = len >> 4;                                                                  \
            U64 nLast = len & 15;                                                                  \
            const float* __restrict pIn = x;                                                       \
            float* __restrict pOut = out;                                                          \
            for (U64 i = 0; i < nFull; ++i)                                                        \
            {                                                                                      \
                __m512 X = _mm512_loadu_ps(pIn);                                                   \
                __m512 y = fxn(X);                                                                 \
                _mm512_storeu_ps(pOut, y);                                                         \
                pIn = reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(pIn) + 64);       \
                pOut = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(pOut) + 64);           \
            }                                                                                      \
            if (nLast)                                                                             \
            {                                                                                      \
                __mmask16 mask = (1U << nLast) - 1U;                                               \
                __m512 X = _mm512_setzero_ps();                                                    \
                X = _mm512_mask_loadu_ps(X, mask, pIn);                                            \
                __m512 y = fxn(X);                                                                 \
                _mm512_mask_storeu_ps(pOut, mask, y);                                              \
            }                                                                                      \
        }                                                                                          \
    }

DYNAMIC_LENGTH_IMPL(Sinf);
DYNAMIC_LENGTH_IMPL(Cosf);
DYNAMIC_LENGTH_IMPL(Tanf);
DYNAMIC_LENGTH_IMPL(Asinf);
DYNAMIC_LENGTH_IMPL(Acosf);

void SinCosf(float* __restrict s, float* __restrict c, const float* __restrict x, U64 len)
{
    if (len <= 16)
    {
        __mmask16 mask = (1U << len) - 1U;
        __m512 X = _mm512_setzero_ps();
        X = _mm512_mask_loadu_ps(X, mask, x);
        __m512 S, C;
        SinCosf(S, C, X);
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
            SinCosf(S, C, X);
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
            SinCosf(S, C, X);
            _mm512_mask_storeu_ps(s, mask, S);
            _mm512_mask_storeu_ps(c, mask, C);
        }
    }
}
#endif // __AVX512F__
