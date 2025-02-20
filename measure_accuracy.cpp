#include "common.h"
#include "nem_math.h"

#include <cstring>
#include <cmath>
#include <random>

#define TEST_FXN Acos
#define REF_FXN acos
#define REF_FXN_FLOAT acosf
static constexpr float a = -1.0f;
static constexpr float b =  1.0f;
static constexpr float kStepSize = 0x1p-25f;

float CalcAccuracy()
{
    __m256 maxAbsErr = _mm256_setzero_ps();
    __m256 minAbsErr = _mm256_setzero_ps();
    __m256 sumSqrAbsErr = _mm256_setzero_ps();
    __m256 maxRelErr = _mm256_setzero_ps();
    __m256 minRelErr = _mm256_setzero_ps();
    __m256 sumSqrRelErr = _mm256_setzero_ps();
    __m256 maxUlpErr = _mm256_setzero_ps();
    __m256 minUlpErr = _mm256_setzero_ps();
    __m256 sumSqrUlpErr = _mm256_setzero_ps();

    const U32 nSteps = _mm_cvt_ss2si(_mm_div_ps(
        _mm_set_ss(b - a), _mm_set_ss(kStepSize)));
    for (U32 i = 0; i < nSteps; i += 8)
    {
        // build the next floats to test, and evaluate ref and impl
        alignas(64) float x[8];
        alignas(64) double ref[8];
        for (U32 j = 0; j < 8; ++j)
        {
            U32 k = i + j;
            x[j] = a + kStepSize * float(k);
            ref[j] = REF_FXN(double(x[j]));
        }
        const __m256 xm = _mm256_loadu_ps(x);
        const __m256 ym = TEST_FXN(xm);
        const __m256 outOfBounds = _mm256_cmp_ps(xm, _mm256_set1_ps(b), _CMP_GT_OQ);

        // compute abs error in double precision
        const __m256d yLo = _mm256_cvtps_pd(_mm256_castps256_ps128(ym));
        const __m256d yHi = _mm256_cvtps_pd(_mm256_extractf128_ps(ym, 1));
        const __m256d refLo = _mm256_loadu_pd(ref);
        const __m256d refHi = _mm256_loadu_pd(ref + 4);
        const __m256d absErrLo = _mm256_sub_pd(yLo, refLo);
        const __m256d absErrHi = _mm256_sub_pd(yHi, refHi);
        __m256 absErr = _mm256_setr_m128(_mm256_cvtpd_ps(absErrLo), _mm256_cvtpd_ps(absErrHi));
        absErr = _mm256_andnot_ps(outOfBounds, absErr);
        if (ReduceMax(absErr) > 1e-3f)
        {
            printf("Dammit\n");
        }

        // compute absolute and relative error
        __m256 ulp = GetUlp(ym);
        __m256 relErr = _mm256_div_ps(
            absErr,
            Abs(_mm256_fmadd_ps(_mm256_set1_ps(kStepSize), _mm256_set1_ps(0x1p-30f), ym)));
        __m256 ulpErr = _mm256_div_ps(absErr, ulp);
        maxAbsErr = _mm256_max_ps(maxAbsErr, absErr);
        minAbsErr = _mm256_min_ps(minAbsErr, absErr);
        sumSqrAbsErr = _mm256_fmadd_ps(absErr, absErr, sumSqrAbsErr);
        maxRelErr = _mm256_max_ps(maxRelErr, relErr);
        minRelErr = _mm256_min_ps(minRelErr, relErr);
        sumSqrRelErr = _mm256_fmadd_ps(relErr, relErr, sumSqrRelErr);
        maxUlpErr = _mm256_max_ps(maxUlpErr, ulpErr);
        minUlpErr = _mm256_min_ps(minUlpErr, ulpErr);
        sumSqrUlpErr = _mm256_fmadd_ps(ulpErr, ulpErr, sumSqrUlpErr);
    }

    // reduce errors down to single values
    float rmsAbsErr = sqrt(ReduceAdd(sumSqrAbsErr) / float(nSteps));
    float rmsRelErr = sqrt(ReduceAdd(sumSqrRelErr) / float(nSteps));
    float rmsUlpErr = sqrt(ReduceAdd(sumSqrUlpErr) / float(nSteps));
    printf("Error profile for [%f, %f): \n", a, b);
    printf("    Abs err: (%g, %g); RMS %g\n",
           ReduceMin(minAbsErr),
           ReduceMax(maxAbsErr),
           rmsAbsErr);
    printf("    Rel err: (%g, %g); RMS %g\n",
           ReduceMin(minRelErr),
           ReduceMax(maxRelErr),
           rmsRelErr);
    printf("    Ulp err: (%g, %g); RMS %g\n",
           ReduceMin(minUlpErr),
           ReduceMax(maxUlpErr),
           rmsUlpErr);
    return rmsUlpErr;
}

void WriteToFile()
{
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
        ref[i] = REF_FXN(double(x[i]));
        reff[i] = REF_FXN_FLOAT(x[i]);
        stlErr[i] = float(double(reff[i]) - ref[i]) / GetUlp(reff[i]);
    }
    for (int i = 0; i < N; i += 8)
    {
        const __m256 xm = _mm256_loadu_ps(x + i);
        const __m256 ym = TEST_FXN(xm);
        const __m256d yLo = _mm256_cvtps_pd(_mm256_extractf128_ps(ym, 0));
        const __m256d yHi = _mm256_cvtps_pd(_mm256_extractf128_ps(ym, 1));
        const __m256d deltaLo = _mm256_sub_pd(yLo, _mm256_load_pd(ref + i));
        const __m256d deltaHi = _mm256_sub_pd(yHi, _mm256_load_pd(ref + i + 4));
        _mm_storeu_ps(absErr + i, _mm256_cvtpd_ps(deltaLo));
        _mm_storeu_ps(absErr + i + 4, _mm256_cvtpd_ps(deltaHi));
        const __m256 absErrM = _mm256_load_ps(absErr + i);
        __m256 ulp = GetUlp(ym);
        __m256 relErrM = _mm256_div_ps(absErrM, ulp);
        _mm256_storeu_ps(relErr + i, relErrM);
        _mm256_storeu_ps(y + i, ym);
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

void RunNear(float near, float stride = kStepSize)
{
    float xArr[8];
    double refArr[8];
    for (U32 i = 0; i < 8; ++i)
    {
        xArr[i] = near + float(I32(i) - 4) * stride;
        refArr[i] = REF_FXN(double(xArr[i]));
    }
    __m256 x = _mm256_loadu_ps(xArr);
    __m256 y = TEST_FXN(x);
    const __m256d yLo = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 0));
    const __m256d yHi = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1));
    const __m256d deltaLo = _mm256_sub_pd(yLo, _mm256_load_pd(refArr));
    const __m256d deltaHi = _mm256_sub_pd(yHi, _mm256_load_pd(refArr + 4));

    const __m256 absErr = _mm256_setr_m128(_mm256_cvtpd_ps(deltaLo), _mm256_cvtpd_ps(deltaHi));
    __m256 ulp = GetUlp(y);
    __m256 relErr = _mm256_div_ps(absErr, ulp);
}

static inline __m256 __vectorcall TEST_FXN(const __m256 x, const float* coefs)
{
    const __m256 kSignMask = _mm256_set1_ps(-0.0f);
    const __m256 xAbs = _mm256_andnot_ps(kSignMask, x);
    __m256 xOuter = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), xAbs, _mm256_set1_ps(2.0f));
    const __m256 outerMask = _mm256_cmp_ps(xAbs, _mm256_set1_ps(0.5f), _CMP_GE_OQ);
    const __m256 signBit = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_srli_epi32(_mm256_castps_si256(x), 31), 31));
    const __m256 p = _mm256_blendv_ps(_mm256_mul_ps(x, x), xOuter, outerMask);
    const __m256 lastMul = _mm256_blendv_ps(x, _mm256_xor_ps(signBit, _mm256_sqrt_ps(xOuter)), outerMask);
    __m256 yPoly = _mm256_blendv_ps(_mm256_set1_ps(coefs[4]), _mm256_set1_ps(coefs[5+4]), outerMask);
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(coefs[3]), _mm256_set1_ps(coefs[5+3]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(coefs[2]), _mm256_set1_ps(coefs[5+2]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(coefs[1]), _mm256_set1_ps(coefs[5+1]), outerMask));
    yPoly = _mm256_fmadd_ps(
        yPoly, p, _mm256_blendv_ps(_mm256_set1_ps(coefs[0]), _mm256_set1_ps(coefs[5+0]), outerMask));
    const __m256 lastP = _mm256_xor_ps(p, _mm256_and_ps(outerMask, kSignMask));
    yPoly = _mm256_fmadd_ps(yPoly, lastP, _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_setzero_ps(), outerMask));
    const __m256 lastAdd = _mm256_blendv_ps(
        _mm256_set1_ps(0.5f * kPif),
        _mm256_add_ps(lastMul,
                      _mm256_and_ps(_mm256_set1_ps(kPif),
                                    _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(signBit), 31)))), outerMask);
    return _mm256_fmadd_ps(yPoly, lastMul, lastAdd);
}

constexpr U32 kWidth = 16;
constexpr U32 kNumVecs = kWidth / 8;
static_assert(kWidth % 8 == 0, "Width must be multiple of vector width (8)");

bool UpdateMaxError(__m256(& r_argMax)[2 * kNumVecs],
                    __m256(& r_lowerBound)[2 * kNumVecs],
                    __m256(& r_upperBound)[2 * kNumVecs],
                    float& r_maxErr,
                    U64& r_numFullEvals,
                    const float* coefs,
                    const float a,
                    const float b,
                    const float stepSize,
                    const U32 nSteps,
                    const double* ref)
{

    __m256i im = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    alignas(32) U32 argMaxIdx[2*kWidth];
    alignas(32) float argMax[2*kWidth];
    alignas(32) float maxUlpErr[2*kWidth];
    for (U32 i = 0; i < 2*kWidth; ++i)
    {
        argMaxIdx[i] = 0;
        argMax[i] = 0.0f;
        maxUlpErr[i] = 0.0f;
    }

    // Loop through the full range to test, tracking the max error in each of `kWidth` different sections of the domain.
    for (U32 j = 0; j < nSteps;)
    {
        const U32 lane = (j * kWidth) / nSteps;
        // Build the next floats to test, and evaluate ref and impl.
        const __m256 xm = _mm256_fmadd_ps(_mm256_set1_ps(stepSize), _mm256_cvtepi32_ps(im), _mm256_set1_ps(a));
        const __m256 ym = TEST_FXN(xm, coefs);
        const __m256 outOfBounds = _mm256_cmp_ps(xm, _mm256_set1_ps(b), _CMP_GT_OQ);

        // Compute abs error in double precision.
        const __m256d yLo = _mm256_cvtps_pd(_mm256_castps256_ps128(ym));
        const __m256d yHi = _mm256_cvtps_pd(_mm256_extractf128_ps(ym, 1));
        const __m256d refLo = _mm256_loadu_pd(ref + j);
        const __m256d refHi = _mm256_loadu_pd(ref + j + 4);
        const __m256d absErrLo = _mm256_sub_pd(yLo, refLo);
        const __m256d absErrHi = _mm256_sub_pd(yHi, refHi);
        __m256 absErr = _mm256_setr_m128(_mm256_cvtpd_ps(absErrLo), _mm256_cvtpd_ps(absErrHi));
        absErr = _mm256_andnot_ps(outOfBounds, absErr);

        // Update min/max ulp error.
        __m256 ulp = GetUlp(ym);
        __m256 ulpErr = _mm256_div_ps(absErr, ulp);
        float loErr = ReduceMin(ulpErr);
        float hiErr = ReduceMax(ulpErr);
        if (loErr < maxUlpErr[lane])
        {
            maxUlpErr[lane] = loErr;
            const __m256 isLo = _mm256_cmp_ps(ulpErr, _mm256_set1_ps(loErr), _CMP_EQ_OQ);
            const __m256 loArgMax = _mm256_blendv_ps(_mm256_set1_ps(-kInf), xm, isLo);
            const __m256i loArgMaxIdx = _mm256_blendv_epi8(_mm256_setzero_si256(), im, _mm256_castps_si256(isLo));
            argMax[lane] = ReduceMax(loArgMax);
            argMaxIdx[lane] = U32(ReduceMaxEpi32(loArgMaxIdx));
        }
        if (hiErr > maxUlpErr[lane + kWidth])
        {
            maxUlpErr[lane + kWidth] = hiErr;
            const __m256 isHi = _mm256_cmp_ps(ulpErr, _mm256_set1_ps(hiErr), _CMP_EQ_OQ);
            const __m256 hiArgMax = _mm256_blendv_ps(_mm256_set1_ps(-kInf), xm, isHi);
            const __m256i hiArgMaxIdx = _mm256_blendv_epi8(_mm256_setzero_si256(), im, _mm256_castps_si256(isHi));
            argMax[lane + kWidth] = ReduceMax(hiArgMax);
            argMaxIdx[lane + kWidth] = U32(ReduceMaxEpi32(hiArgMaxIdx));
        }
        im = _mm256_add_epi32(im, _mm256_set1_epi32(8));
        j += 8;

        // Check if error exceeds previous best -- if it does, quit early.
        if (loErr <= -r_maxErr || hiErr >= r_maxErr)
        {
            r_numFullEvals += j;
            return false;
        }
    }

    // Record new max error; recompute upper/lower bounds for argmax cases.

    __m256 loErr = _mm256_loadu_ps(maxUlpErr);
    __m256 hiErr = _mm256_loadu_ps(maxUlpErr + kWidth);
    for (U32 i = 1; i < kNumVecs; ++i)
    {
        loErr = _mm256_min_ps(loErr, _mm256_loadu_ps(maxUlpErr + 8*i));
        hiErr = _mm256_max_ps(hiErr, _mm256_loadu_ps(maxUlpErr + 8*i + kWidth));
    }
    r_maxErr = Max(ReduceMax(hiErr), -ReduceMin(loErr));

    // Expand the tolerance by slightly less than half an ulp; this will make anything with slack v.s. the max-error
    // case round one ulp farther away, so we can narrow our search to solutions whose error is strictly less than
    // the tolerance.
    const double maxTol = double(r_maxErr) + 0.49999;
    
    for (U32 i = 0; i < 2*kNumVecs; ++i)
    {
        const __m256d argMaxRefLo = _mm256_i32gather_pd(ref, _mm_loadu_si128(reinterpret_cast<__m128i*>(argMaxIdx + 8*i)), 8);
        const __m256d argMaxRefHi =
            _mm256_i32gather_pd(ref, _mm_loadu_si128(reinterpret_cast<__m128i*>(argMaxIdx + 8*i + 4)), 8);
        const __m256d argMaxUlpLo = _mm256_cvtps_pd(GetUlp(_mm256_cvtpd_ps(argMaxRefLo)));
        const __m256d argMaxUlpHi = _mm256_cvtps_pd(GetUlp(_mm256_cvtpd_ps(argMaxRefHi)));
        const __m256d lbLo =
            _mm256_fnmadd_pd(_mm256_set1_pd(maxTol), argMaxUlpLo, argMaxRefLo);
        const __m256d lbHi =
            _mm256_fnmadd_pd(_mm256_set1_pd(maxTol), argMaxUlpHi, argMaxRefHi);
        const __m256d ubLo =
            _mm256_fmadd_pd(_mm256_set1_pd(maxTol), argMaxUlpLo, argMaxRefLo);
        const __m256d ubHi =
            _mm256_fmadd_pd(_mm256_set1_pd(maxTol), argMaxUlpHi, argMaxRefHi);
        __m256 lowerBound = _mm256_setr_m128(_mm256_cvtpd_ps(lbLo), _mm256_cvtpd_ps(lbHi));
        __m256 upperBound = _mm256_setr_m128(_mm256_cvtpd_ps(ubLo), _mm256_cvtpd_ps(ubHi));
        r_argMax[i] = _mm256_load_ps(argMax + 8*i);
        r_lowerBound[i] = lowerBound;
        r_upperBound[i] = upperBound;
    }
    r_numFullEvals += nSteps;
    return true;
}

void OptimizeCoefs(U32 seed = 0)
{
    // The coefficients that we'll try to optimize.
    static constexpr U32 kCoefBufLen = 16;
    alignas(32) float coefs[kCoefBufLen]{
        -5.93803041e-02f,
            1.46333009e-01f,
            -1.90639019e-01f,
            2.10152701e-01f,
            -2.40228832e-01f,
            2.88294435e-01f,
            -3.60665351e-01f,
            4.80903596e-01f,
            -7.21347809e-01f,
            1.44269502e+00f
    };
    static constexpr U32 nCoefs = 10;

    // Compute the minimum number of steps needed at the given step size to cover the full interval.
    U32 nSteps = _mm_cvt_ss2si(_mm_div_ss(
        _mm_set_ss(b - a), _mm_set_ss(kStepSize)));
    const __m128 calcB =
        _mm_fmadd_ss(_mm_set_ss(kStepSize), _mm_set_ss(float(nSteps)), _mm_set_ss(a));
    const __m128 deltaB = _mm_sub_ss(calcB, _mm_set_ss(b));
    __m128 extraStepsFloat = _mm_div_ss(deltaB, _mm_set_ss(kStepSize));
    extraStepsFloat = _mm_max_ps(extraStepsFloat, _mm_setzero_ps());
    const U32 nExtraSteps = _mm_cvt_ss2si(extraStepsFloat);
    nSteps -= nExtraSteps;
    nSteps = (nSteps + 15) & U32(-16);

    // Evaluate the ref in double-precision at every test point.
    double* ref = reinterpret_cast<double*>(_mm_malloc(nSteps * sizeof(double), 4096));
    for (U32 i = 0; i < nSteps; ++i)
    {
        float x = a + kStepSize * float(i);
        ref[i] = REF_FXN(double(x));
    }

    __m256 argMax[2*kNumVecs]; 
    __m256 lowerBound[2*kNumVecs];
    __m256 upperBound[2*kNumVecs];
    float maxErr = 1e10f;
    U64 numFullEvals = 0;

    // Compute max ulp error from starting coefs, and 16 of the worst cases (worst case for each lane).
    if (!UpdateMaxError(argMax, lowerBound, upperBound, maxErr, numFullEvals, coefs, a, b, kStepSize, nSteps, ref))
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

    // "Random" values to use for generating new coefs to try.
    constexpr U32 kHashConst = 0x9e3779b9;
    const __m128i hashConst = _mm_set1_epi32(kHashConst);
    alignas(32) U32 hash[kCoefBufLen];
    for (U32 i = 0; i < kCoefBufLen; ++i)
    {
        hash[i] = seed + i;
    }
    for (U32 i = 0; i < kCoefBufLen; i += 4)
    {
        __m128i* p = reinterpret_cast<__m128i*>(hash + i);
        __m128i x = _mm_loadu_si128(p);
        x = _mm_aesenc_si128(x, hashConst);
        _mm_storeu_si128(p, x);
    }

    bool changedCoef = true;
    while (true)
    {
        changedCoef = false;
        I32 shiftIdx = -1;

        while (shiftIdx < 23)
        {
            printf("Searching w/ coef shift %d\n", shiftIdx);
            constexpr U32 kMaxAttempts = 10'000'000;
            constexpr U32 kMaxFullAttempts = 10'000;
            U32 numAttempts = 0;
            U32 numFullAttempts = 0;
            numFullEvals = 0;
            do
            {
                ++numAttempts;
                const I32 shift = Max(shiftIdx, 0);
                const float scale = (shiftIdx == -1) ? 0x1p-22f : 0x1p-23f;
                // Generate new coefs.
                alignas(32) float newCoefs[kCoefBufLen];
                static_assert(kCoefBufLen % 8 == 0, "Only implemented for multiple of 8");
                for (U32 i = 0; i < kCoefBufLen; i += 8)
                {
                    __m256i h = _mm256_loadu_si256(reinterpret_cast<__m256i *>(hash + i));
                    __m256 co = _mm256_xor_ps(
                        _mm256_castsi256_ps(_mm256_and_si256(h, _mm256_set1_epi32(0x8000'0000))),
                        _mm256_cvtepi32_ps(
                            _mm256_srlv_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x007f'ffff)),
                                              _mm256_set1_epi32(shift))));
                    co = _mm256_fmadd_ps(co, _mm256_set1_ps(scale), _mm256_set1_ps(1.0f));
                    co = _mm256_mul_ps(co, _mm256_load_ps(coefs + i));
                    _mm256_store_ps(newCoefs + i, co);

                }

                for (U32 i = 0; i < kCoefBufLen; i += 4)
                {
                    __m128i* p = reinterpret_cast<__m128i*>(hash + i);
                    __m128i x = _mm_loadu_si128(p);
                    x = _mm_aesenc_si128(x, hashConst);
                    _mm_storeu_si128(p, x);
                }

                // Check if they pass the current worst-case error cases.
                __m256 failMask = _mm256_setzero_ps();
                for (U32 i = 0; i < 2*kNumVecs; ++i)
                {
                    const __m256 worstResults = TEST_FXN(argMax[i], newCoefs);
                    const __m256 failLo = _mm256_cmp_ps(worstResults, lowerBound[i], _CMP_LE_OQ);
                    const __m256 failHi = _mm256_cmp_ps(worstResults, upperBound[i], _CMP_GE_OQ);
                    failMask = _mm256_or_ps(failLo, _mm256_or_ps(failHi, failMask));
                }
                if (_mm256_movemask_ps(failMask))
                {
                    continue;
                }

                ++numFullAttempts;
                const bool newRecord = UpdateMaxError(
                    argMax, lowerBound, upperBound, maxErr, numFullEvals, newCoefs, a, b, kStepSize, nSteps, ref);
                if (newRecord)
                {
                    printf("Found better max error: %.9f\n", maxErr);
                    memcpy(coefs, newCoefs, nCoefs * sizeof(float));
                    FILE* outFile = fopen("FinalCoefs.txt", "w");
                    for (U32 i = 0; i < nCoefs; ++i)
                    {
                        fprintf(outFile, "%.8ef\n", coefs[i]);
                    }
                    fclose(outFile);
                    changedCoef = true;
                }
            } while (numAttempts < kMaxAttempts && numFullAttempts < kMaxFullAttempts);

            if (numAttempts == kMaxAttempts)
            {
                printf("Finished %u short attempts, %u went full, %lu full evals\n",
                       numAttempts,
                       numFullAttempts,
                       numFullEvals);
            }
            else
            {
                printf("Finished %u full attempts out of %u attempts, %lu full evals\n",
                       numFullAttempts,
                       numAttempts,
                       numFullEvals);
            }
            ++shiftIdx;
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

void ReportAccuracyAtan2(float rangeStart, float rangeEnd, float step)
{ 
    __m256 maxAbsErr = _mm256_setzero_ps(); 
    __m256 minAbsErr = _mm256_setzero_ps(); 
    __m256 sumSqrAbsErr = _mm256_setzero_ps(); 
    __m256 maxRelErr = _mm256_setzero_ps(); 
    __m256 minRelErr = _mm256_setzero_ps(); 
    __m256 sumSqrRelErr = _mm256_setzero_ps(); 
    __m256 maxUlpErr = _mm256_setzero_ps(); 
    __m256 minUlpErr = _mm256_setzero_ps(); 
    __m256 sumSqrUlpErr = _mm256_setzero_ps(); 
 
    const float a = float(rangeStart); 
    const float b = float(rangeEnd); 
    const float stepSize = float(step); 
    const U32 nSteps = _mm_cvtss_i32(_mm_div_ss(_mm_set_ss(b - a), _mm_set_ss(stepSize))); 
    for (U32 i = 0; i < nSteps; ++i) 
    { 
        float x = a + stepSize * float(i);
        for (U32 j = 0; j < nSteps; j += 8)
        {
            alignas(32) float y[8];
            alignas(64) double ref[8];
            for (U32 jj = 0; jj < 8; ++jj)
            {
                U32 k = j + jj;
                y[jj] = a + stepSize * float(k);
                ref[jj] = atan2(double(y[jj]), double(x));
            }
            const __m256 xm = _mm256_set1_ps(x);
            const __m256 ym = _mm256_loadu_ps(y);
            const __m256 zm = Atan2(ym, xm);
            const __m256 outOfBounds = _mm256_cmp_ps(ym, _mm256_set1_ps(b), _CMP_GT_OQ);
 
            /* compute abs error in double precision */ 
            const __m256d zLo = _mm256_cvtps_pd(_mm256_castps256_ps128(zm));
            const __m256d zHi = _mm256_cvtps_pd(_mm256_extractf128_ps(zm, 1));
            const __m256d refLo = _mm256_loadu_pd(ref);
            const __m256d refHi = _mm256_loadu_pd(ref + 4);
            const __m256d absErrLo = _mm256_sub_pd(zLo, refLo);
            const __m256d absErrHi = _mm256_sub_pd(zHi, refHi);
            __m256 absErr = _mm256_setr_m128(_mm256_cvtpd_ps(absErrLo), _mm256_cvtpd_ps(absErrHi));
            absErr = _mm256_andnot_ps(outOfBounds, absErr);
 
            /* compute absolute and relative error */
            __m256 ulp = GetUlp(zm);
            __m256 relErr = _mm256_div_ps(
                absErr,
                Abs(_mm256_fmadd_ps(_mm256_set1_ps(stepSize), _mm256_set1_ps(0x1p-30f), zm)));
            __m256 ulpErr = _mm256_div_ps(absErr, ulp);
            maxAbsErr = _mm256_max_ps(maxAbsErr, absErr);
            minAbsErr = _mm256_min_ps(minAbsErr, absErr);
            sumSqrAbsErr = _mm256_fmadd_ps(absErr, absErr, sumSqrAbsErr);
            maxRelErr = _mm256_max_ps(maxRelErr, relErr);
            minRelErr = _mm256_min_ps(minRelErr, relErr);
            sumSqrRelErr = _mm256_fmadd_ps(relErr, relErr, sumSqrRelErr);
            maxUlpErr = _mm256_max_ps(maxUlpErr, ulpErr);
            minUlpErr = _mm256_min_ps(minUlpErr, ulpErr);
            sumSqrUlpErr = _mm256_fmadd_ps(ulpErr, ulpErr, sumSqrUlpErr);
            if (_mm256_movemask_ps(_mm256_cmp_ps(ulpErr, _mm256_setzero_ps(), _CMP_UNORD_Q)))
            {
                printf("NaN Error\n");
            }
        }
    } 
 
        /* reduce errors from vectors to single values, and print results */ 
        float rmsAbsErr = sqrt(ReduceAdd(sumSqrAbsErr) / float(nSteps*nSteps)); 
        float rmsRelErr = sqrt(ReduceAdd(sumSqrRelErr) / float(nSteps*nSteps)); 
        float rmsUlpErr = sqrt(ReduceAdd(sumSqrUlpErr) / float(nSteps*nSteps)); 
        printf("Error profile for Atan2 in [%f, %f)^2: \n", a, b); 
        printf("    Abs err: [% 9.3g, % 9.3g]; RMS % 9.3g\n", 
               ReduceMin(minAbsErr), 
               ReduceMax(maxAbsErr), 
               rmsAbsErr); 
        printf("    Rel err: [% 9.3g, % 9.3g]; RMS % 9.3g\n", 
               ReduceMin(minRelErr), 
               ReduceMax(maxRelErr), 
               rmsRelErr); 
        printf("    Ulp err: [% 9.3g, % 9.3g]; RMS % 9.3g\n\n\n", 
               ReduceMin(minUlpErr), 
               ReduceMax(maxUlpErr), 
               rmsUlpErr); 
}

void CheckSpecialCases()
{
    alignas(64) float xArr[16] = { -kInf, -kMax, -kMin, -0.0f, 0.0f, kMin, kMax, kInf, kNan };
    alignas(64) float yArr[16];
    alignas(64) float refArr[16];
    for (U32 i = 0; i < 16; ++i)
    {
        refArr[i] = REF_FXN_FLOAT(xArr[i]);
    }
    __m256 x = _mm256_loadu_ps(xArr);
    __m256 y = TEST_FXN(x);
    _mm256_storeu_ps(yArr, y);
    __m256 ref = _mm256_loadu_ps(refArr);
    __m256i matchLo = _mm256_cmpeq_epi32(_mm256_castps_si256(y), _mm256_castps_si256(ref));
    U32 diffLo = (~_mm256_movemask_ps(_mm256_castsi256_ps(matchLo))) & 255;
    x = _mm256_loadu_ps(xArr + 8);
    y = TEST_FXN(x);
    _mm256_storeu_ps(yArr + 8, y);
    ref = _mm256_loadu_ps(refArr + 8);
    __m256i matchHi = _mm256_cmpeq_epi32(_mm256_castps_si256(y), _mm256_castps_si256(ref));
    U32 diffHi = (~_mm256_movemask_ps(_mm256_castsi256_ps(matchHi))) & 255;
    const U32 diffMask = diffLo | (diffHi << 8);
    if (diffMask)
    {
        printf("Breakpoint me");
    }
}

double foo1(double x, double& m, double& d2, double& e)
{
    double d = __builtin_floor((0.5 * kInvPid) * x);
    x = Fma(-2.0 * kPi3d, d, Fma(-2.0 * kPi2d, d, Fma(-2.0 * kPid, d, x)));
    m = x;
    d = __builtin_floor((0.5 * kInvPid) * x);
    d2 = d;
    x = Fma(-2.0 * kPid, d, x);
    e = x;
    x = (x == (2.0 * kPid)) ? 0.0 : ((x < 0.0) ? 0.0 : x);
    return x;
}

double foo2(double x, double& m, double& d2, double& e)
{
    double d = __builtin_floor((0.5 * kInvPid) * x);
    x = Fma(-2.0 * kPi3d, d, Fma(-2.0 * kPi2d, d, Fma(-2.0 * kPid, d, x)));
    m = x;
    d = __builtin_floor((0.5 * kInvPid) * x);
    d2 = d;
    x = Fma(-2.0 * kPi2d, d, Fma(-2.0 * kPid, d, x));
    e = x;
    x = (x == (2.0 * kPid)) ? 0.0 : (x < 0.0) ? 0.0 : x;
    return x;
}

double GetUlp(double x)
{
    const __m128d a = _mm_set_sd(x);
    const __m128d b = _mm_castsi128_pd(_mm_xor_si128(_mm_castpd_si128(a), _mm_set1_epi64x(1)));
    return _mm_cvtsd_f64(_mm_andnot_pd(_mm_set_sd(-0.0), _mm_sub_sd(a, b)));
}

int main()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    //CalcAccuracy();
    //OptimizeCoefs();
    WriteToFile();
    //ReportAccuracyAtan2(-1.0f, 1.0f, 0x1p-12f);
    //RunNear(0.75f, 2.0 * kStepSize);
    //CheckSpecialCases();
}
