// Milestone 1 kernel: int4 weight-only matrix-vector product, y = W x.
//
// Shape: one MLP up-projection of Qwen2.5-0.5B (896 inputs, 4864 outputs).
//
// Weight format (host packs it, see fpga/sim/test_matvec_int4.cpp):
//   - w_packed: OUT rows, each IN/2 bytes. Element k of a row sits in byte k/2:
//     even k in the low nibble, odd k in the high nibble. Stored value is q + 8,
//     where q is the signed int4 value in [-8, 7].
//   - scales: OUT rows, each IN/GROUP floats. Scale g covers inputs
//     [g*GROUP, (g+1)*GROUP).
//   - x: IN activations (float for this first version; see fpga/DESIGN-qwen05b.md).
//   - y: OUT outputs.
//
// This version is a plain reference loop with HLS pragmas. It is correct C++,
// so it builds and tests with g++. Vitis HLS reads the pragmas during synthesis
// and ignores them in software builds.

#ifndef MATVEC_INT4_H
#define MATVEC_INT4_H

#include <cstdint>

constexpr int MV_IN = 896;
constexpr int MV_OUT = 4864;
constexpr int MV_GROUP = 128;
constexpr int MV_GROUPS = MV_IN / MV_GROUP;

inline void matvec_int4(const uint8_t *w_packed,
                        const float *scales,
                        const float *x,
                        float *y) {
  for (int o = 0; o < MV_OUT; ++o) {
    float acc = 0.0f;
    for (int g = 0; g < MV_GROUPS; ++g) {
      float gacc = 0.0f;
      for (int i = 0; i < MV_GROUP; ++i) {
#ifdef __SYNTHESIS__
#pragma HLS PIPELINE II = 1
#endif
        const int k = g * MV_GROUP + i;
        const uint8_t b = w_packed[o * (MV_IN / 2) + k / 2];
        const int nib = (k & 1) ? (b >> 4) : (b & 0x0F);
        gacc += static_cast<float>(nib - 8) * x[k];
      }
      acc += gacc * scales[o * MV_GROUPS + g];
    }
    y[o] = acc;
  }
}

#endif  // MATVEC_INT4_H
