#include <chrono>
#include <cmath>
#include <immintrin.h>

#include "nem_math.h"

struct
{
    std::chrono::steady_clock::time_point start = {};
    double min = std::numeric_limits<double>::infinity();
    int idx = 0;
    void reset() { min = std::numeric_limits<double>::infinity(); }
} timekeeper;

void tic() { timekeeper.start = std::chrono::steady_clock::now(); }

void toc()
{
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    double elapsed = double((stop - timekeeper.start).count());
    if (elapsed >= timekeeper.min)
        return;
    timekeeper.min = elapsed;
    if (elapsed > 1'000'000'000.0)
    {
        printf("%.2f s\n", elapsed / 1'000'000'000.0);
    }
    else if (elapsed > 1'000'000.0)
    {
        printf("%.2f ms\n", elapsed / 1'000'000.0);
    }
    else if (elapsed > 1'000.0)
    {
        printf("%.2f us\n", elapsed / 1'000.0);
    }
    else
    {
        printf("%.0f ns\n", elapsed);
    }
}

void ProfileThroughput()
{
    printf("Profiling throughput:\n");
    static constexpr float a = 0.5f + 0x1p-24f;
    static constexpr float b = 1.0f;
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
    float* x = reinterpret_cast<float*>(_mm_malloc(nSteps * sizeof(float), 64));
    for (U32 i = 0; i < nSteps; ++i)
    {
        x[i] = a + stepSize * float(i);
    }

    // Ref impl
    int reps = 64;
    printf("Ref impl:\n");
    __m512 accum = _mm512_setzero_ps();
    do
    {
        tic();
        for (U32 i = 0; i < nSteps; i += 16)
        {
            alignas(64) float outBuffer[16];
            for (U32 j = 0; j < 16; ++j)
            {
                U32 k = i + j;
                outBuffer[j] = std::atanf(x[k]);
            }
            accum = _mm512_xor_ps(accum, _mm512_loadu_ps(outBuffer));
        }
        toc();
    } while (--reps);
    printf("Ref check: %a\n", _mm512_reduce_max_ps(accum));
    timekeeper.reset();

    // My impl
    reps = 256;
    printf("My impl:\n");
    accum = _mm512_setzero_ps();
    do
    {
        tic();
        for (U32 i = 0; i < nSteps; i += 16)
        {
            __m512 X = _mm512_loadu_ps(x + i);
            __m512 Y = Atanf_Dumb1(X);
            accum = _mm512_xor_ps(accum, Y);
        }
        toc();
    } while (--reps);
    printf("My check: %a\n", _mm512_reduce_max_ps(accum));
    timekeeper.reset();

    // My impl
    reps = 256;
    printf("My impl:\n");
    accum = _mm512_setzero_ps();
    do
    {
        tic();
        for (U32 i = 0; i < nSteps; i += 16)
        {
            __m512 X = _mm512_loadu_ps(x + i);
            __m512 Y = Atanf_Dumb2(X);
            accum = _mm512_xor_ps(accum, Y);
        }
        toc();
    } while (--reps);
    printf("My check: %a\n", _mm512_reduce_max_ps(accum));
    timekeeper.reset();
    _mm_free(x);
}

void ProfileSinLatency()
{
    printf("Profiling sin latency:\n");
    constexpr int N = 16;
    constexpr float step = 1.0f / N;
    float* x = reinterpret_cast<float*>(_mm_malloc(N * sizeof(float), 64));
    float* y = reinterpret_cast<float*>(_mm_malloc(N * sizeof(float), 64));
    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f + float(i) * step;
        //x[i] *= 0x1.0p25f;
    }

    // Ref sin
    int reps = 1024;
    printf("Sin Ref impl:\n");
    do
    {
        tic();
        //RefSinDep(x);
        toc();
    } while (--reps);
    if (!bool(x[0]))
        printf("%f\n", x[1]);
    timekeeper.min = std::numeric_limits<double>::infinity();

    // My sin
    reps = 1024;
    printf("My sin impl:\n");
    do
    {
        tic();
        //MySinDep(y);
        toc();
    } while (--reps);
    if (!bool(y[0]))
        printf("%f\n", y[1]);
    timekeeper.min = std::numeric_limits<double>::infinity();
    _mm_free(x);
    _mm_free(y);
}

void CountToABillion()
{
    tic();
    U32 sum = 0, i = 0;
    do
    {
        sum += i;
    } while (++i);
    printf("Sum is %llu\n", sum);
    toc();
}

int main()
{
    ProfileThroughput();
    //CountToABillion();
}
