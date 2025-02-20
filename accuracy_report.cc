#include "common.h"
#include "nem_math.h"
#include <cstring>
#include <limits>
#include <cmath>
#include <random>
#define REPORT_ACCURACY(testFxn, refFxn, rangeStart, rangeEnd, step) \
do \
{ \
    __m256 maxAbsErr = _mm256_setzero_ps(); \
    __m256 minAbsErr = _mm256_setzero_ps(); \
    __m256 sumSqrAbsErr = _mm256_setzero_ps(); \
    __m256 maxRelErr = _mm256_setzero_ps(); \
    __m256 minRelErr = _mm256_setzero_ps(); \
    __m256 sumSqrRelErr = _mm256_setzero_ps(); \
    __m256 maxUlpErr = _mm256_setzero_ps(); \
    __m256 minUlpErr = _mm256_setzero_ps(); \
    __m256 sumSqrUlpErr = _mm256_setzero_ps(); \
 \
    constexpr float a = float(rangeStart); \
    constexpr float b = float(rangeEnd); \
    constexpr float stepSize = float(step); \
    const U32 nSteps = _mm_cvtss_i32(_mm_div_ss( \
        _mm_set_ss(b - a), _mm_set_ss(stepSize))); \
    for (U32 i = 0; i < nSteps; i += 8) \
    { \
        /* build the next floats to test, and evaluate ref and impl */ \
        alignas(64) float x[8]; \
        alignas(64) double ref[8]; \
        for (U32 j = 0; j < 8; ++j) \
        { \
            U32 k = i + j; \
            x[j] = a + stepSize * float(k); \
            ref[j] = refFxn(double(x[j])); \
        } \
        const __m256 xm = _mm256_loadu_ps(x); \
        const __m256 ym = testFxn(xm); \
        const __m256 outOfBounds = _mm256_cmp_ps(xm, _mm256_set1_ps(b), _CMP_GT_OQ); \
 \
        /* compute abs error in double precision */ \
        const __m256d yLo = _mm256_cvtps_pd(_mm256_castps256_ps128(ym)); \
        const __m256d yHi = _mm256_cvtps_pd(_mm256_extractf128_ps(ym, 1)); \
        const __m256d refLo = _mm256_loadu_pd(ref); \
        const __m256d refHi = _mm256_loadu_pd(ref + 4); \
        const __m256d absErrLo = _mm256_sub_pd(yLo, refLo); \
        const __m256d absErrHi = _mm256_sub_pd(yHi, refHi); \
        __m256 absErr = _mm256_setr_m128(_mm256_cvtpd_ps(absErrLo), _mm256_cvtpd_ps(absErrHi)); \
        absErr = _mm256_andnot_ps(outOfBounds, absErr); \
 \
        /* compute absolute and relative error */ \
        __m256 ulp = GetUlp(ym); \
        __m256 relErr = _mm256_div_ps( \
            absErr, \
            Abs(_mm256_fmadd_ps(_mm256_set1_ps(stepSize), _mm256_set1_ps(0x1p-30f), ym))); \
        __m256 ulpErr = _mm256_div_ps(absErr, ulp); \
        maxAbsErr = _mm256_max_ps(maxAbsErr, absErr); \
        minAbsErr = _mm256_min_ps(minAbsErr, absErr); \
        sumSqrAbsErr = _mm256_fmadd_ps(absErr, absErr, sumSqrAbsErr); \
        maxRelErr = _mm256_max_ps(maxRelErr, relErr); \
        minRelErr = _mm256_min_ps(minRelErr, relErr); \
        sumSqrRelErr = _mm256_fmadd_ps(relErr, relErr, sumSqrRelErr); \
        maxUlpErr = _mm256_max_ps(maxUlpErr, ulpErr); \
        minUlpErr = _mm256_min_ps(minUlpErr, ulpErr); \
        sumSqrUlpErr = _mm256_fmadd_ps(ulpErr, ulpErr, sumSqrUlpErr); \
    } \
 \
    /* reduce errors from vectors of 16 to single values, and print results */ \
    float rmsAbsErr = sqrt(ReduceAdd(sumSqrAbsErr) / float(nSteps)); \
    float rmsRelErr = sqrt(ReduceAdd(sumSqrRelErr) / float(nSteps)); \
    float rmsUlpErr = sqrt(ReduceAdd(sumSqrUlpErr) / float(nSteps)); \
    printf("Error profile for " #testFxn " in [%f, %f): \n", a, b); \
    printf("    Abs err: [% 9.3g, % 9.3g]; RMS % 9.3g\n", \
           ReduceMin(minAbsErr), \
           ReduceMax(maxAbsErr), \
           rmsAbsErr); \
    printf("    Rel err: [% 9.3g, % 9.3g]; RMS % 9.3g\n", \
           ReduceMin(minRelErr), \
           ReduceMax(maxRelErr), \
           rmsRelErr); \
    printf("    Ulp err: [% 9.3g, % 9.3g]; RMS % 9.3g\n\n\n", \
           ReduceMin(minUlpErr), \
           ReduceMax(maxUlpErr), \
           rmsUlpErr); \
} while (0)
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
int main()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    REPORT_ACCURACY(Sin, sin, 0.0f, 2.0f, 0x1p-25f);
    REPORT_ACCURACY(Cos, cos, 0.0f, 2.0f, 0x1p-25f);
    REPORT_ACCURACY(Tan, tan, 0.0f, 2.0f, 0x1p-25f);
    REPORT_ACCURACY(Asin, asin, -1.0f, 1.0f, 0x1p-24f);
    REPORT_ACCURACY(Acos, acos, -1.0f, 1.0f, 0x1p-24f);
    REPORT_ACCURACY(Atan, atan, -4.0f, -1.0f, 0x1p-23f);
    REPORT_ACCURACY(Atan, atan, -1.0f, 1.0f, 0x1p-24f);
    REPORT_ACCURACY(Atan, atan, 1.0f, 4.0f, 0x1p-23f);
    REPORT_ACCURACY(Log2, log2, 0.25f, 1.0f, 0x1p-25f);
    REPORT_ACCURACY(Log2, log2, 1.0f, 4.0f, 0x1p-23f);
    REPORT_ACCURACY(Exp2, exp2, -2.0f, 0.0f, 0x1p-25f);
    REPORT_ACCURACY(Exp2, exp2, 0.0f, 2.0f, 0x1p-25f);
    ReportAccuracyAtan2(-1.0f, 1.0f, 0x1p-12f);
    ReportAccuracyAtan2(-4096.0f, 4096.0f, 1.0f);
}
