#include <cstring>
#include <limits>
#include <cmath>
#include <random>

#include "common.h"
#include "nem_math.h"

float U32BitsAsFloat(U32 x)
{
    float d;
    memcpy(&d, &x, 4);
    return d;
}

__m256 GetUlp(__m256 x)
{
    constexpr float kMinNorm = std::numeric_limits<float>::min();
    __m256 ulp = _mm256_and_ps(
        _mm256_sub_ps(x, _mm256_xor_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x00000001)))), _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff'ffff)));
    return _mm256_max_ps(ulp, _mm256_set1_ps(kMinNorm));
}

__m512 GetUlp(__m512 x)
{
    constexpr float kMinNorm = std::numeric_limits<float>::min();
    __m512 ulp = _mm512_abs_ps(
        _mm512_sub_ps(x, _mm512_xor_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x00000001)))));
    return _mm512_max_ps(ulp, _mm512_set1_ps(kMinNorm));
}

float GetUlp(float x) { return _mm512_cvtss_f32(GetUlp(_mm512_castps128_ps512(_mm_set_ss(x)))); }

float Min(float a, float b) { return a < b ? a : b; }

float CalcAccuracy()
{
    __m512 maxAbsErr = _mm512_setzero_ps();
    __m512 minAbsErr = _mm512_setzero_ps();
    __m512 sumSqrAbsErr = _mm512_setzero_ps();
    __m512 maxRelErr = _mm512_setzero_ps();
    __m512 minRelErr = _mm512_setzero_ps();
    __m512 sumSqrRelErr = _mm512_setzero_ps();
    __m512 maxUlpErr = _mm512_setzero_ps();
    __m512 minUlpErr = _mm512_setzero_ps();
    __m512 sumSqrUlpErr = _mm512_setzero_ps();

    //static constexpr float a = 0.5f + 0x1p-24f;
    //static constexpr float b = 1.0f;
    static constexpr float a = 1.0f;
    static constexpr float b = 16.0f;
    //const float stepSize = Min(GetUlp(a), GetUlp(b));
    const float stepSize = 0x1p-23f;
    const U32 nSteps = _mm_cvtss_i32(_mm_div_round_ss(
        _mm_set_ss(b - a), _mm_set_ss(stepSize), _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
    for (U32 i = 0; i < nSteps; i += 16)
    {
        // build the next floats to test, and evaluate ref and impl
        alignas(64) float x[16];
        alignas(64) double ref[16];
        for (U32 j = 0; j < 16; ++j)
        {
            U32 k = i + j;
            x[j] = a + stepSize * float(k);
            ref[j] = std::log2(double(x[j]));
        }
        const __m512 xm = _mm512_loadu_ps(x);
        const __m512 ym = Log2f(xm);
        const __mmask16 outOfBounds = _mm512_cmp_ps_mask(xm, _mm512_set1_ps(b), _CMP_GT_OQ);

        // compute abs error in double precision
        const __m512d yLo = _mm512_cvtps_pd(_mm512_castps512_ps256(ym));
        const __m512d yHi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ym, 1));
        const __m512d refLo = _mm512_loadu_pd(ref);
        const __m512d refHi = _mm512_loadu_pd(ref + 8);
        const __m512d absErrLo = _mm512_sub_pd(yLo, refLo);
        const __m512d absErrHi = _mm512_sub_pd(yHi, refHi);
        __m512 absErr = _mm512_broadcast_f32x8(_mm512_cvtpd_ps(absErrLo));
        absErr = _mm512_insertf32x8(absErr, _mm512_cvtpd_ps(absErrHi), 1);
        absErr = _mm512_mask_and_ps(absErr, outOfBounds, absErr, _mm512_setzero_ps());

        // compute absolute and relative error
        __m512 ulp = GetUlp(ym);
        __m512 relErr = _mm512_div_ps(
            absErr,
            _mm512_abs_ps(_mm512_fmadd_ps(_mm512_set1_ps(stepSize), _mm512_set1_ps(0x1p-30f), ym)));
        __m512 ulpErr = _mm512_div_ps(absErr, ulp);
        maxAbsErr = _mm512_max_ps(maxAbsErr, absErr);
        minAbsErr = _mm512_min_ps(minAbsErr, absErr);
        sumSqrAbsErr = _mm512_fmadd_ps(absErr, absErr, sumSqrAbsErr);
        maxRelErr = _mm512_max_ps(maxRelErr, relErr);
        minRelErr = _mm512_min_ps(minRelErr, relErr);
        sumSqrRelErr = _mm512_fmadd_ps(relErr, relErr, sumSqrRelErr);
        maxUlpErr = _mm512_max_ps(maxUlpErr, ulpErr);
        minUlpErr = _mm512_min_ps(minUlpErr, ulpErr);
        sumSqrUlpErr = _mm512_fmadd_ps(ulpErr, ulpErr, sumSqrUlpErr);

        //maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_f32x4(maxUlpErr, maxUlpErr, 177));
        //maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_f32x4(maxUlpErr, maxUlpErr, 78));
        //maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_ps(maxUlpErr, maxUlpErr, 177));
        //maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_ps(maxUlpErr, maxUlpErr, 78));
        //minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_f32x4(minUlpErr, minUlpErr, 177));
        //minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_f32x4(minUlpErr, minUlpErr, 78));
        //minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_ps(minUlpErr, minUlpErr, 177));
        //minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_ps(minUlpErr, minUlpErr, 78));
        //if (_mm512_cvtss_f32(minUlpErr) < -10.0f)
        //{
        //    printf("Breakpoint me\n");
        //}
    }

    // reduce errors from vectors of 16 to single values
    maxAbsErr = _mm512_max_ps(maxAbsErr, _mm512_shuffle_f32x4(maxAbsErr, maxAbsErr, 177));
    maxAbsErr = _mm512_max_ps(maxAbsErr, _mm512_shuffle_f32x4(maxAbsErr, maxAbsErr, 78));
    maxAbsErr = _mm512_max_ps(maxAbsErr, _mm512_shuffle_ps(maxAbsErr, maxAbsErr, 177));
    maxAbsErr = _mm512_max_ps(maxAbsErr, _mm512_shuffle_ps(maxAbsErr, maxAbsErr, 78));
    minAbsErr = _mm512_min_ps(minAbsErr, _mm512_shuffle_f32x4(minAbsErr, minAbsErr, 177));
    minAbsErr = _mm512_min_ps(minAbsErr, _mm512_shuffle_f32x4(minAbsErr, minAbsErr, 78));
    minAbsErr = _mm512_min_ps(minAbsErr, _mm512_shuffle_ps(minAbsErr, minAbsErr, 177));
    minAbsErr = _mm512_min_ps(minAbsErr, _mm512_shuffle_ps(minAbsErr, minAbsErr, 78));
    sumSqrAbsErr =
        _mm512_add_ps(sumSqrAbsErr, _mm512_shuffle_f32x4(sumSqrAbsErr, sumSqrAbsErr, 177));
    sumSqrAbsErr =
        _mm512_add_ps(sumSqrAbsErr, _mm512_shuffle_f32x4(sumSqrAbsErr, sumSqrAbsErr, 78));
    sumSqrAbsErr = _mm512_add_ps(sumSqrAbsErr, _mm512_shuffle_ps(sumSqrAbsErr, sumSqrAbsErr, 177));
    sumSqrAbsErr = _mm512_add_ps(sumSqrAbsErr, _mm512_shuffle_ps(sumSqrAbsErr, sumSqrAbsErr, 78));
    sumSqrAbsErr = _mm512_sqrt_ps(_mm512_div_ps(sumSqrAbsErr, _mm512_set1_ps(float(nSteps))));
    maxRelErr = _mm512_max_ps(maxRelErr, _mm512_shuffle_f32x4(maxRelErr, maxRelErr, 177));
    maxRelErr = _mm512_max_ps(maxRelErr, _mm512_shuffle_f32x4(maxRelErr, maxRelErr, 78));
    maxRelErr = _mm512_max_ps(maxRelErr, _mm512_shuffle_ps(maxRelErr, maxRelErr, 177));
    maxRelErr = _mm512_max_ps(maxRelErr, _mm512_shuffle_ps(maxRelErr, maxRelErr, 78));
    minRelErr = _mm512_min_ps(minRelErr, _mm512_shuffle_f32x4(minRelErr, minRelErr, 177));
    minRelErr = _mm512_min_ps(minRelErr, _mm512_shuffle_f32x4(minRelErr, minRelErr, 78));
    minRelErr = _mm512_min_ps(minRelErr, _mm512_shuffle_ps(minRelErr, minRelErr, 177));
    minRelErr = _mm512_min_ps(minRelErr, _mm512_shuffle_ps(minRelErr, minRelErr, 78));
    sumSqrRelErr =
        _mm512_add_ps(sumSqrRelErr, _mm512_shuffle_f32x4(sumSqrRelErr, sumSqrRelErr, 177));
    sumSqrRelErr =
        _mm512_add_ps(sumSqrRelErr, _mm512_shuffle_f32x4(sumSqrRelErr, sumSqrRelErr, 78));
    sumSqrRelErr = _mm512_add_ps(sumSqrRelErr, _mm512_shuffle_ps(sumSqrRelErr, sumSqrRelErr, 177));
    sumSqrRelErr = _mm512_add_ps(sumSqrRelErr, _mm512_shuffle_ps(sumSqrRelErr, sumSqrRelErr, 78));
    sumSqrRelErr = _mm512_sqrt_ps(_mm512_div_ps(sumSqrRelErr, _mm512_set1_ps(float(nSteps))));
    maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_f32x4(maxUlpErr, maxUlpErr, 177));
    maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_f32x4(maxUlpErr, maxUlpErr, 78));
    maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_ps(maxUlpErr, maxUlpErr, 177));
    maxUlpErr = _mm512_max_ps(maxUlpErr, _mm512_shuffle_ps(maxUlpErr, maxUlpErr, 78));
    minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_f32x4(minUlpErr, minUlpErr, 177));
    minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_f32x4(minUlpErr, minUlpErr, 78));
    minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_ps(minUlpErr, minUlpErr, 177));
    minUlpErr = _mm512_min_ps(minUlpErr, _mm512_shuffle_ps(minUlpErr, minUlpErr, 78));
    sumSqrUlpErr =
        _mm512_add_ps(sumSqrUlpErr, _mm512_shuffle_f32x4(sumSqrUlpErr, sumSqrUlpErr, 177));
    sumSqrUlpErr =
        _mm512_add_ps(sumSqrUlpErr, _mm512_shuffle_f32x4(sumSqrUlpErr, sumSqrUlpErr, 78));
    sumSqrUlpErr = _mm512_add_ps(sumSqrUlpErr, _mm512_shuffle_ps(sumSqrUlpErr, sumSqrUlpErr, 177));
    sumSqrUlpErr = _mm512_add_ps(sumSqrUlpErr, _mm512_shuffle_ps(sumSqrUlpErr, sumSqrUlpErr, 78));
    sumSqrUlpErr = _mm512_sqrt_ps(_mm512_div_ps(sumSqrUlpErr, _mm512_set1_ps(float(nSteps))));
    printf("Error profile for [%f, %f): \n", a, b);
    printf("    Abs err: (%g, %g); RMS %g\n",
           _mm512_cvtss_f32(minAbsErr),
           _mm512_cvtss_f32(maxAbsErr),
           _mm512_cvtss_f32(sumSqrAbsErr));
    printf("    Rel err: (%g, %g); RMS %g\n",
           _mm512_cvtss_f32(minRelErr),
           _mm512_cvtss_f32(maxRelErr),
           _mm512_cvtss_f32(sumSqrRelErr));
    printf("    Ulp err: (%g, %g); RMS %g\n",
           _mm512_cvtss_f32(minUlpErr),
           _mm512_cvtss_f32(maxUlpErr),
           _mm512_cvtss_f32(sumSqrUlpErr));
    return _mm512_cvtss_f32(sumSqrUlpErr);
}

void WriteToFile()
{
    static constexpr float a = 1.0f;
    static constexpr float b = 2.0f;
    constexpr int N = (1 << 15) + 1;
    constexpr int NAlloc = (N + 15) & -16;
    constexpr float step = (b - a) / float(N - 1);
    float* x;
    float* y;
    double* ref;
    float* reff;
    float* stlErr;
    float* absErr;
    float* relErr;
    x = (float*)_mm_malloc(NAlloc * sizeof(float) * 8, 64);
    y = x + NAlloc;
    ref = (double*)(y + NAlloc);
    reff = (float*)(ref + NAlloc);
    stlErr = reff + NAlloc;
    absErr = stlErr + NAlloc;
    relErr = absErr + NAlloc;
    for (int i = 0; i < N; ++i)
    {
        x[i] = a + step * float(i);
        ref[i] = std::log2(double{ x[i] });
        reff[i] = std::log2f(x[i]);
        stlErr[i] = float(double(reff[i]) - ref[i]) / GetUlp(reff[i]);
    }
    for (int i = 0; i < N; i += 16)
    {
        const __m512 xm = _mm512_loadu_ps(x + i);
        const __m512 ym = Log2f(xm);
        const __m512d yLo = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ym, 0));
        const __m512d yHi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ym, 1));
        const __m512d deltaLo = _mm512_sub_pd(yLo, _mm512_load_pd(ref + i));
        const __m512d deltaHi = _mm512_sub_pd(yHi, _mm512_load_pd(ref + i + 8));
        _mm256_storeu_ps(absErr + i, _mm512_cvtpd_ps(deltaLo));
        _mm256_storeu_ps(absErr + i + 8, _mm512_cvtpd_ps(deltaHi));
        const __m512 absErrM = _mm512_load_ps(absErr + i);
        __m512 ulp = GetUlp(ym);
        __m512 relErrM = _mm512_div_ps(absErrM, ulp);
        _mm512_storeu_ps(relErr + i, relErrM);
        _mm512_storeu_ps(y + i, ym);
    }
    FILE* outFile = fopen("AccuracyData.bin", "wb");
    fwrite(&N, sizeof(int), 1, outFile);
    fwrite(x, sizeof(float), N, outFile);
    fwrite(y, sizeof(float), N, outFile);
    fwrite(ref, sizeof(double), N, outFile);
    fwrite(reff, sizeof(float), N, outFile);
    fwrite(stlErr, sizeof(float), N, outFile);
    fwrite(absErr, sizeof(float), N, outFile);
    fwrite(relErr, sizeof(float), N, outFile);
    fclose(outFile);
}

__m512 __vectorcall Log2f(__m512 x, float* coefs)
{
    __m512i exp = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0xff80'0000));
    __m512i mant = _mm512_ternarylogic_epi32(_mm512_set1_epi32(0xff80'0000),
                                             _mm512_castps_si512(x),
                                             _mm512_set1_epi32(0x3f80'0000),
                                             0xac);
    exp = _mm512_srai_epi32(exp, 23);
    const __m512i bad = _mm512_sub_epi32(exp, _mm512_set1_epi32(1));
    __m512 base = _mm512_or_ps(_mm512_set1_ps(0x1.8p+23f), _mm512_castsi512_ps(exp));
    base = _mm512_sub_ps(base, _mm512_set1_ps(0x1.8p+23f + 127.0f));
    x = _mm512_sub_ps(_mm512_castsi512_ps(mant), _mm512_set1_ps(1.0f));

    __m512 res = _mm512_set1_ps(coefs[0]);
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[1]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[2]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[3]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[4]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[5]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[6]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[7]));
    res = _mm512_fmadd_ps(res, x, _mm512_set1_ps(coefs[8]));
    __m512 lastMul = _mm512_castsi512_ps(
        _mm512_and_epi32(_mm512_srai_epi32(bad, 31), _mm512_set1_epi32(0x7fc0'0000)));
    lastMul = _mm512_add_ps(lastMul, x);
    res = _mm512_fmadd_ps(res, lastMul, base);
    return res;
}

static __m512i TernBlend(const __m512i mask, const __m512i a, const __m512i b)
{
    return _mm512_ternarylogic_epi32(mask, a, b, 0xac);
}

static __m512 TernBlend(const __m512i mask, const __m512 a, const __m512 b)
{
    return _mm512_castsi512_ps(
        _mm512_ternarylogic_epi32(mask, _mm512_castps_si512(a), _mm512_castps_si512(b), 0xac));
}

bool UpdateMaxError(__m512& r_argMax,
                    __m512& r_lowerBound,
                    __m512& r_upperBound,
                    float& r_maxErr,
                    float* coefs,
                    const float a,
                    const float b,
                    const float stepSize,
                    const U32 nSteps,
                    const double* ref)
{
    __m512 argMax = _mm512_setzero_ps();
    __m512i argMaxIdx = _mm512_setzero_si512();
    __m512 maxUlpErr = _mm512_setzero_ps();
    __m512 idx = _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i im = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512 prevMaxErr = _mm512_set1_ps(r_maxErr);
    for (U32 j = 0; j < nSteps; j += 16)
    {
        // build the next floats to test, and evaluate ref and impl
        const __m512 xm = _mm512_fmadd_ps(_mm512_set1_ps(stepSize), idx, _mm512_set1_ps(a));
        const __m512 ym = Log2f(xm, coefs);
        const __mmask16 outOfBounds = _mm512_cmp_ps_mask(xm, _mm512_set1_ps(b), _CMP_GT_OQ);

        // compute abs error in double precision
        const __m512d yLo = _mm512_cvtps_pd(_mm512_castps512_ps256(ym));
        const __m512d yHi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(ym, 1));
        const __m512d refLo = _mm512_loadu_pd(ref + j);
        const __m512d refHi = _mm512_loadu_pd(ref + j + 8);
        const __m512d absErrLo = _mm512_sub_pd(yLo, refLo);
        const __m512d absErrHi = _mm512_sub_pd(yHi, refHi);
        __m512 absErr = _mm512_broadcast_f32x8(_mm512_cvtpd_ps(absErrLo));
        absErr = _mm512_insertf32x8(absErr, _mm512_cvtpd_ps(absErrHi), 1);
        absErr = _mm512_mask_and_ps(absErr, outOfBounds, absErr, _mm512_setzero_ps());

        // compute absolute value of ulp error
        __m512 ulp = GetUlp(ym);
        __m512 ulpErr = _mm512_abs_ps(_mm512_div_ps(absErr, ulp));
        const __m512i isMax =
            _mm512_srai_epi32(_mm512_castps_si512(_mm512_sub_ps(maxUlpErr, ulpErr)), 31);
        maxUlpErr = TernBlend(isMax, maxUlpErr, ulpErr);
        argMax = TernBlend(isMax, argMax, xm);
        argMaxIdx = TernBlend(isMax, argMaxIdx, im);

        // Check if error exceeds previous best -- if it does, give up.
        if (_mm512_cmp_ps_mask(maxUlpErr, prevMaxErr, _CMP_GT_OQ))
        {
            return false;
        }

        idx = _mm512_add_ps(idx, _mm512_set1_ps(16.0f));
        im = _mm512_add_epi32(im, _mm512_set1_epi32(16));
    }

    // Record new max error; recompute upper/lower bounds for argmax cases.
    r_maxErr = _mm512_reduce_max_ps(maxUlpErr) * (1.0f - 0x1p-20f);
    const __m512d argMaxRefLo = _mm512_i32gather_pd(_mm512_castsi512_si256(argMaxIdx), ref, 8);
    const __m512d argMaxRefHi =
        _mm512_i32gather_pd(_mm512_extracti32x8_epi32(argMaxIdx, 1), ref, 8);
    const __m512d argMaxUlpLo = _mm512_cvtps_pd(GetUlp(_mm512_cvtpd_ps(argMaxRefLo)));
    const __m512d argMaxUlpHi = _mm512_cvtps_pd(GetUlp(_mm512_cvtpd_ps(argMaxRefHi)));
    const __m512d lbLo = _mm512_fnmadd_pd(_mm512_set1_pd(double(r_maxErr)), argMaxUlpLo, argMaxRefLo);
    const __m512d lbHi = _mm512_fnmadd_pd(_mm512_set1_pd(double(r_maxErr)), argMaxUlpHi, argMaxRefHi);
    const __m512d ubLo = _mm512_fmadd_pd(_mm512_set1_pd(double(r_maxErr)), argMaxUlpLo, argMaxRefLo);
    const __m512d ubHi = _mm512_fmadd_pd(_mm512_set1_pd(double(r_maxErr)), argMaxUlpHi, argMaxRefHi);
    __m512 lowerBound = _mm512_broadcast_f32x8(_mm512_cvtpd_ps(lbLo));
    lowerBound = _mm512_insertf32x8(lowerBound, _mm512_cvtpd_ps(lbHi), 1);
    __m512 upperBound = _mm512_broadcast_f32x8(_mm512_cvtpd_ps(ubLo));
    upperBound = _mm512_insertf32x8(upperBound, _mm512_cvtpd_ps(ubHi), 1);
    r_argMax = argMax;
    r_lowerBound = lowerBound;
    r_upperBound = upperBound;
    return true;
}

void OptimizeCoefs(U32 seed = 0)
{
    // The coefficients that we'll try to optimize.
    float coefs[16]{
                     5.41353924e-03f,
                     -3.32472920e-02f,
                     9.59496498e-02f,
                     -1.80703834e-01f,
                     2.66505808e-01f,
                     -3.55522752e-01f,
                     4.80219364e-01f,
                     -7.21309245e-01f,
                     1.44269466e+00f,
    };
    static constexpr U32 nCoefs = 9;

    static constexpr float a = 0.5f;
    static constexpr float b = 2.0f;
    //const float stepSize = 1.0f * Min(GetUlp(a), GetUlp(b));
    const float stepSize = 0x1p-24f;
    U32 nSteps = _mm_cvtss_i32(_mm_div_round_ss(
        _mm_set_ss(b - a), _mm_set_ss(stepSize), _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
    {
        const __m128 calcB =
            _mm_fmadd_ss(_mm_set_ss(stepSize), _mm_set_ss(float(nSteps)), _mm_set_ss(a));
        const __m128 deltaB = _mm_sub_ss(calcB, _mm_set_ss(b));
        const __m128 extraStepsFloat = _mm_div_ss(deltaB, _mm_set_ss(stepSize));
        const U32 nExtraSteps = _mm_cvtss_i32(extraStepsFloat);
        nSteps -= nExtraSteps;
    }
    nSteps = (nSteps + 15) & U32(-16);
    double* ref = reinterpret_cast<double*>(_mm_malloc(nSteps * sizeof(double), 4096));
    for (U32 i = 0; i < nSteps; ++i)
    {
        float x = a + stepSize * float(i);
        ref[i] = std::log2(double(x));
    }

    __m512 argMax, lowerBound, upperBound;
    float maxErr = 1e10f;

    // Compute max ulp error from starting coefs, and 16 of the worst cases (worst case for each lane).
    if (!UpdateMaxError(
            argMax, lowerBound, upperBound, maxErr, coefs, a, b, stepSize, nSteps, ref))
    {
        puts("Failed to find initial max error!");
        fflush(stdout);
        exit(1);
    }
    else
    {
        printf("Starting max error is %.8e\n", maxErr);
    }

    // Now that we've found the worst-case error, try random coef perturbations until we improve
    // the worst case. Once we do, check the rest of the floats.
    //std::mt19937 gen;
    //std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // "Random" values to use for generating new coefs to try.
    constexpr U32 kHash = 0x9e3779b9;
    constexpr U32 kNumHash = 16;
    static_assert(kNumHash >= nCoefs, "L");
    alignas(64) U32 hash[kNumHash];
    hash[0] = _mm_crc32_u32(12345 + seed, kHash);
    hash[1] = _mm_crc32_u32(54321 + seed, kHash);
    hash[2] = _mm_crc32_u32(67890 + seed, kHash);
    hash[3] = _mm_crc32_u32(98765 + seed, kHash);
    for (U32 i = 4; i < kNumHash; i += 4)
    {
        hash[i + 0] = _mm_crc32_u32(hash[i + 0 - 4], kHash);
        hash[i + 1] = _mm_crc32_u32(hash[i + 1 - 4], kHash);
        hash[i + 2] = _mm_crc32_u32(hash[i + 2 - 4], kHash);
        hash[i + 3] = _mm_crc32_u32(hash[i + 3 - 4], kHash);
    }

    bool changedCoef = true;
    while (true)
    {
        changedCoef = false;
        float deltaSize = 1.0f;
        U32 shiftIdx = 0;

        while (shiftIdx < 23 && deltaSize > 0x1p-23f)
        {
            printf("Searching w/ coef shift %u\n", shiftIdx);
            constexpr U32 kMaxAttempts = 10'000'000;
            constexpr U32 kMaxFullAttempts = 10'000;
            U32 numAttempts = 0;
            U32 numFullAttempts = 0;
            do
            {
                ++numAttempts;
                // Generate new coefs.
                alignas(64) float newCoefs[kNumHash];
                //for (U32 j = 0; j < nCoefs; ++j)
                //{
                    //newCoefs[j] = coefs[j] + deltaSize * dist(gen);
                //}
                static_assert(kNumHash == 16, "Only implemented for 1 vec");
                __m512i coi = _mm512_load_si512(hash);
                __m512 co = _mm512_xor_ps(_mm512_castsi512_ps(_mm512_and_si512(coi, _mm512_set1_epi32(0x8000'0000))), _mm512_cvtepi32_ps(_mm512_srlv_epi32(_mm512_and_si512(coi, _mm512_set1_epi32(0x007f'ffff)), _mm512_set1_epi32(shiftIdx))));
                co = _mm512_fmadd_ps(co, _mm512_set1_ps(0x1p-23f), _mm512_set1_ps(1.0f));
				co = _mm512_mul_ps(co, _mm512_load_ps(coefs));
                _mm512_store_ps(newCoefs, co);
                for (U32 j = 0; j < nCoefs; ++j)
                {
                    hash[j] = _mm_crc32_u32(hash[j], kHash);
                }

                // Check if they pass the current worst-case error cases.
                const __m512 worstResults = Log2f(argMax, newCoefs);
                {
                    const __mmask16 failLo =
                        _mm512_cmp_ps_mask(worstResults, lowerBound, _CMP_LE_OQ);
                    const __mmask16 failHi =
                        _mm512_cmp_ps_mask(worstResults, upperBound, _CMP_GE_OQ);
                    if (failLo || failHi)
                    {
                        continue;
                    }
                }
                ++numFullAttempts;
                const bool newRecord = UpdateMaxError(
                    argMax, lowerBound, upperBound, maxErr, newCoefs, a, b, stepSize, nSteps, ref);
                if (newRecord)
                {
                    printf("Found better max error: %1.9f\n", maxErr);
                    memcpy(coefs, newCoefs, nCoefs * sizeof(float));
                    FILE* outFile = fopen("FinalCoefs.txt", "w");
                    for (U32 i = 0; i < nCoefs; ++i)
                    {
                        fprintf(outFile, "%1.8ef\n", coefs[i]);
                    }
                    fclose(outFile);
                    changedCoef = true;
                }
            } while (numAttempts < kMaxAttempts && numFullAttempts < kMaxFullAttempts);

            if (numAttempts == kMaxAttempts)
            {
                printf("Finished %u short attempts, %u went full\n", kMaxAttempts, numFullAttempts);
            }
            else
            {
                printf("Finished %u full attempts out of %u attempts\n", kMaxFullAttempts, numAttempts);
            }
            ++shiftIdx;
            deltaSize *= 0.5f;
        }
    }

    printf("Final max error: %1.8ef\n", maxErr);

    FILE* outFile = fopen("FinalCoefs.txt", "w");
    printf("Final coefs:\n");
    for (U32 i = 0; i < nCoefs; ++i)
    {
        printf("%1.8ef, ", coefs[i]);
        fprintf(outFile, "%1.8ef\n", coefs[i]);
    }
    fclose(outFile);
    _mm_free(ref);
}

void CheckSpecialCases()
{
    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
    alignas(64) float xArr[16] = { -kInf, -1.0f, kNan, 0.0f, 1e-10f, 1.0f - 0x1p-24f, 1.0f, 2.0f - 0x1p-23f, 2.0f, 1e20f, kInf};
    alignas(64) float ref[16];
    for (U32 i = 0; i < 16; ++i)
    {
        ref[i] = std::log2f(xArr[i]);
    }
    __m512 x = _mm512_loadu_ps(xArr);
    __m512 y = Log2f(x);
    printf("Breakpoint me");
}

int main()
{
    //CalcAccuracy();
    OptimizeCoefs();
    //WriteToFile();
    //CheckSpecialCases();
}
