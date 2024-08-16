//#ifndef NEM_POWLOG_ALLOW_IMPL
//#error "Do not include this file directly, include powlog.h instead."
//#endif
#pragma once

#include "common.h"

// Single precision, AVX2 SIMD for x86-64 CPUs.
#ifdef __AVX2__
__m128 __vectorcall Exp2f(__m128 x);
__m256 __vectorcall Exp2f(__m256 x);
__m128 __vectorcall Expf(__m128 x);
__m256 __vectorcall Expf(__m256 x);
__m128 __vectorcall Exp10f(__m128 x);
__m256 __vectorcall Exp10f(__m256 x);
__m128 __vectorcall Powf(__m128 x, __m128 y);
__m256 __vectorcall Powf(__m256 x, __m256 y);
__m128 __vectorcall Log2f(__m128 x);
__m256 __vectorcall Log2f(__m256 x);
__m128 __vectorcall Logf(__m128 x);
__m256 __vectorcall Logf(__m256 x);
__m128 __vectorcall Log10f(__m128 x);
__m256 __vectorcall Log10f(__m256 x);
__m128 __vectorcall LogBasef(__m128 b, __m128 x);
__m256 __vectorcall LogBasef(__m256 b, __m256 x);
#endif // __AVX2__

#ifdef __AVX512F__
__m512 __vectorcall Exp2f(__m512 x);
__m512 __vectorcall Expf(__m512 x);
__m512 __vectorcall Exp10f(__m512 x);
__m512 __vectorcall Powf(__m512 x, __m512 y);
__m512 __vectorcall Log2f(__m512 x);
__m512 __vectorcall Logf(__m512 x);
__m512 __vectorcall Log10f(__m512 x);
__m512 __vectorcall LogBasef(__m512 b, __m512 x);
#endif // __AVX512F__

#ifdef __AVX2__
__m128 __vectorcall Exp2f(__m128 x)
{
    const __m128 round = _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m128 frac = _mm_sub_ps(x, round);
    const __m128i roundi = _mm_cvtps_epi32(round);
    __m128i expi = _mm_min_epi32(_mm_max_epi32(roundi, _mm_set1_epi32(-127)), _mm_set1_epi32(128));
    expi = _mm_add_epi32(_mm_slli_epi32(expi, 23), _mm_set1_epi32(0x3f80'0000));
    __m128 exp = _mm_castsi128_ps(expi);
    __m128 mant = _mm_set1_ps(1.33978436e-03f);
    mant = _mm_fmadd_ps(mant, frac, _mm_set1_ps(9.67518147e-03f));
    mant = _mm_fmadd_ps(mant, frac, _mm_set1_ps(5.55042699e-02f));
    mant = _mm_fmadd_ps(mant, frac, _mm_set1_ps(2.40221620e-01f));
    mant = _mm_fmadd_ps(mant, frac, _mm_set1_ps(6.93147004e-01f));
    mant = _mm_fmadd_ps(mant, frac, _mm_set1_ps(1.0f));
    __m128 res = _mm_mul_ps(exp, mant);
    const __m128 oob = _mm_castsi128_ps(_mm_cmpeq_epi32(roundi, _mm_set1_epi32(0x8000'0000)));
    const __m128 isNan = _mm_cmp_ps(x, x, _CMP_UNORD_Q);
    __m128i oobFixupi = _mm_srai_epi32(_mm_castps_si128(x), 31);
    oobFixupi = _mm_blendv_epi8(_mm_set1_epi32(0x7f80'0000), _mm_setzero_si128(), oobFixupi);
    __m128 oobFixup = _mm_blendv_ps(_mm_castsi128_ps(oobFixupi), x, isNan);
    res = _mm_blendv_ps(res, oobFixup, oob);
    return res;
}

__m256 __vectorcall Exp2f(__m256 x)
{
    const __m256 round = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 frac = _mm256_sub_ps(x, round);
    const __m256i roundi = _mm256_cvtps_epi32(round);
    __m256i expi = _mm256_min_epi32(_mm256_max_epi32(roundi, _mm256_set1_epi32(-127)), _mm256_set1_epi32(128));
    expi = _mm256_add_epi32(_mm256_slli_epi32(expi, 23), _mm256_set1_epi32(0x3f80'0000));
    __m256 exp = _mm256_castsi256_ps(expi);
    __m256 mant = _mm256_set1_ps(1.33978436e-03f);
    mant = _mm256_fmadd_ps(mant, frac, _mm256_set1_ps(9.67518147e-03f));
    mant = _mm256_fmadd_ps(mant, frac, _mm256_set1_ps(5.55042699e-02f));
    mant = _mm256_fmadd_ps(mant, frac, _mm256_set1_ps(2.40221620e-01f));
    mant = _mm256_fmadd_ps(mant, frac, _mm256_set1_ps(6.93147004e-01f));
    mant = _mm256_fmadd_ps(mant, frac, _mm256_set1_ps(1.0f));
    __m256 res = _mm256_mul_ps(exp, mant);
    const __m256 oob = _mm256_castsi256_ps(_mm256_cmpeq_epi32(roundi, _mm256_set1_epi32(0x8000'0000)));
    const __m256 isNan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    __m256i oobFixupi = _mm256_srai_epi32(_mm256_castps_si256(x), 31);
    oobFixupi = _mm256_blendv_epi8(_mm256_set1_epi32(0x7f80'0000), _mm256_setzero_si256(), oobFixupi);
    __m256 oobFixup = _mm256_blendv_ps(_mm256_castsi256_ps(oobFixupi), x, isNan);
    res = _mm256_blendv_ps(res, oobFixup, oob);
    return res;
}
#endif // __AVX2__

#ifdef __AVX512F__
__m512 __vectorcall Exp2f(__m512 x)
{
    const __m512 round = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT);
    const __m512 frac = _mm512_sub_ps(x, round);
    const __m512i roundi = _mm512_cvtps_epi32(round);
    __m512i expi = _mm512_min_epi32(_mm512_max_epi32(roundi, _mm512_set1_epi32(-127)), _mm512_set1_epi32(128));
    expi = _mm512_add_epi32(_mm512_slli_epi32(expi, 23), _mm512_set1_epi32(0x3f80'0000));
    const __m512 exp = _mm512_castsi512_ps(expi);
    __m512 mant = _mm512_set1_ps(1.33978436e-03f);
    mant = _mm512_fmadd_ps(mant, frac, _mm512_set1_ps(9.67518147e-03f));
    mant = _mm512_fmadd_ps(mant, frac, _mm512_set1_ps(5.55042699e-02f));
    mant = _mm512_fmadd_ps(mant, frac, _mm512_set1_ps(2.40221620e-01f));
    mant = _mm512_fmadd_ps(mant, frac, _mm512_set1_ps(6.93147004e-01f));
	mant = _mm512_fmadd_ps(mant, frac, _mm512_set1_ps(1.0f));
    __m512 res = _mm512_mul_ps(exp, mant);
    const __mmask16 oob = _mm512_cmpeq_epi32_mask(roundi, _mm512_set1_epi32(0x8000'0000));
    const __mmask16 isNan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
	__m512i oobFixupi = _mm512_srai_epi32(_mm512_castps_si512(x), 31);
	oobFixupi = _mm512_ternarylogic_epi32(oobFixupi, _mm512_set1_epi32(0x7f80'0000), _mm512_setzero_si512(), 0xac);
	__m512 oobFixup = _mm512_mask_blend_ps(isNan, _mm512_castsi512_ps(oobFixupi), x);
	res = _mm512_mask_blend_ps(oob, res, oobFixup);
    return res;
}

__m512 __vectorcall Expf(__m512 x) { return Exp2f(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f))); }

__m512 __vectorcall Exp10f(__m512 x) { return Exp2f(_mm512_mul_ps(x, _mm512_set1_ps(3.321928095f))); }

__m512 __vectorcall Powf(__m512 x, __m512 y) { return Exp2f(_mm512_mul_ps(Log2f(x), y)); }

__m512 __vectorcall Log2f(__m512 x)
{
    __m512i exp = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0xff80'0000));
    __m512i mant = _mm512_ternarylogic_epi32(
        _mm512_set1_epi32(0xff80'0000), _mm512_castps_si512(x), _mm512_set1_epi32(0x3f80'0000), 0xac);
    exp = _mm512_srai_epi32(exp, 23);
    const __mmask16 fixup = _mm512_cmple_epi32_mask(exp, _mm512_setzero_si512()) |
                            _mm512_cmpeq_epi32_mask(exp, _mm512_set1_epi32(255));
    const __mmask16 setNegInf = _mm512_cmpeq_epi32_mask(exp, _mm512_setzero_si512());
    const __mmask16 setNan = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    //const __m512i bad = _mm512_sub_epi32(exp, _mm512_set1_epi32(1));
    __m512 base = _mm512_or_ps(_mm512_set1_ps(0x1.8p+23f), _mm512_castsi512_ps(exp));
    base = _mm512_sub_ps(base, _mm512_set1_ps(0x1.8p+23f + 127.0f));
    const __m512 p = _mm512_sub_ps(_mm512_castsi512_ps(mant), _mm512_set1_ps(1.0f));
    
    __m512 res = _mm512_set1_ps(5.41353924e-03f);
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(-3.32472920e-02f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(9.59496498e-02f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(-1.80703834e-01f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(2.66505808e-01f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(-3.55522752e-01f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(4.80219364e-01f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(-7.21309245e-01f));
    res = _mm512_fmadd_ps(res, p, _mm512_set1_ps(1.44269466e+00f));
    __m512 lastMul = _mm512_mask_blend_ps(fixup, p, x);
    lastMul = _mm512_mask_blend_ps(setNegInf, lastMul, _mm512_set1_ps(kNegInf));
    lastMul = _mm512_mask_blend_ps(setNan, lastMul, _mm512_set1_ps(kNan));
    //__m512 lastMul = _mm512_castsi512_ps(
    //    _mm512_and_epi32(_mm512_srai_epi32(bad, 31), _mm512_set1_epi32(0x7fc0'0000)));
    //lastMul = _mm512_add_ps(lastMul, x);
    res = _mm512_fmadd_ps(res, lastMul, base);
    return res;
}

#endif // __AVX512F__
