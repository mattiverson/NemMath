#pragma once

#include <immintrin.h>
#include <cstdint>

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;
using I8 = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;

static constexpr float kNan = float(__builtin_nan("0"));
static constexpr float kInf = float(__builtin_inf());
static constexpr float kNegInf = -kInf;

// Single-precision floats that sum to pi.
static constexpr float kPif = 3.1415927f;
static constexpr float kPi2f = -8.7422777e-08f;
static constexpr float kPi3f = -3.4302490e-15f;
static constexpr float kPi4f = 2.1125998e-23f;
static constexpr float kPi5f = 1.6956855e-31f;

// Double-precision floats that sum to pi.
static constexpr double kPid = 3.14159265358979323;
static constexpr double kPi2d = 1.224646799147353e-16;
static constexpr double kPi3d = -2.994769809718340e-33;
static constexpr double kPi4d = 1.112454220863365e-49;
static constexpr double kPi5d = 5.672231979640316e-66;

// Single-precision floats that sum to 1.0 / pi.
static constexpr float kInvPif = 0.31830987f;
static constexpr float kInvPi2f = 1.2841276e-08f;
static constexpr float kInvPi3f = 1.4685477e-16f;
static constexpr float kInvPi4f = 3.0370133e-24f;
static constexpr float kInvPi5f = -6.8864879e-32f;

// Double-precision floats that sum to 1.0 / pi.
static constexpr double kInvPid = 0.318309886183791;
static constexpr double kInvPi2d = -1.967867667518249e-17;
static constexpr double kInvPi3d = -1.072143628289300e-33;
static constexpr double kInvPi4d = 8.053563926594112e-50;
static constexpr double kInvPi5d = -1.560276590681971e-66;

// Single-precision coefficients for asin and acos. Derived from a Chebyshev interpolant over the interval (0.5, 1].
static constexpr U32 kNumAcosf = 5;
//alignas(16) static constexpr float kAcosf[kNumAcosf]{
//    4.16636914e-02f,
//    4.71566804e-03f,
//    6.20599603e-04f,
//    1.97349393e-04f,
//};
alignas(16) static constexpr float kAcosf[kNumAcosf]{
    4.16669399e-02f, 4.68410645e-03f, 7.11543311e-04f, 9.33984556e-05f, 4.15559152e-05f,
};
//alignas(16) static constexpr float kAcosf[kNumAcosf]{
//    4.166666816516390e-02f, 4.687465870707369e-03f, 6.978217644855178e-04f, 1.176050615763202e-04f,
//    2.405478090850011e-05f, 1.816758164352450e-06f, 2.118784209415026e-06f
//};
alignas(16) static constexpr float kAcosfDelta[6]{
    1.839767403479577e-03f,  -1.210919600204655e-02f, 3.120439100645461e-02f,
    -3.942244392988953e-02f, 2.444758532448345e-02f,  -5.960027609529248e-03f,
};
static constexpr U32 kNumAsinf = 5;
alignas(16) static constexpr float kAsinf[kNumAsinf]{
    1.666698531397390e-01f, 7.490092298904716e-02f, 4.594513272353140e-02f,
    2.228487155463199e-02f, 4.491649660576513e-02f,
};

// Double-precision coefficients for sin and cos. Derived from Chebyshev interpolants over the interval (-pi/4, pi/4).
static constexpr U32 kNumSind = 8;
static constexpr double kSind[kNumSind]{};
static constexpr double kCosd[kNumSind]{};
