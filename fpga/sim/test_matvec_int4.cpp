// Host-side test for the milestone 1 int4 matvec kernel.
//
// Build and run (no FPGA tools needed):
//   g++ -std=c++17 -O2 -Wall -I../hls test_matvec_int4.cpp -o test_matvec_int4 && ./test_matvec_int4
//
// What it checks:
//   1. The kernel matches a reference that uses the same quantized values
//      (catches indexing and packing bugs). Tolerance: 1e-3 relative.
//   2. The kernel's output is close to the unquantized float product
//      (shows the cost of 4-bit weights on random data).

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "matvec_int4.h"

int main() {
  std::mt19937 rng(12345);
  std::normal_distribution<float> nd(0.0f, 0.05f);  // weight-like spread
  std::normal_distribution<float> xd(0.0f, 1.0f);

  std::vector<float> W(MV_OUT * MV_IN);
  std::vector<float> x(MV_IN);
  for (auto &v : W) v = nd(rng);
  for (auto &v : x) v = xd(rng);

  // Quantize per group: scale = max|w| / 7, q = round(w / scale) clamped to [-8, 7].
  std::vector<uint8_t> packed(MV_OUT * (MV_IN / 2), 0);
  std::vector<float> scales(MV_OUT * MV_GROUPS);
  std::vector<float> Wq(MV_OUT * MV_IN);  // dequantized weights, for the reference

  for (int o = 0; o < MV_OUT; ++o) {
    for (int g = 0; g < MV_GROUPS; ++g) {
      float amax = 0.0f;
      for (int i = 0; i < MV_GROUP; ++i) {
        amax = std::fmax(amax, std::fabs(W[o * MV_IN + g * MV_GROUP + i]));
      }
      const float s = amax > 0.0f ? amax / 7.0f : 1.0f;
      scales[o * MV_GROUPS + g] = s;
      for (int i = 0; i < MV_GROUP; ++i) {
        const int k = g * MV_GROUP + i;
        const float w = W[o * MV_IN + k];
        int q = static_cast<int>(std::lround(w / s));
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        Wq[o * MV_IN + k] = static_cast<float>(q) * s;
        const uint8_t nib = static_cast<uint8_t>(q + 8);
        uint8_t &b = packed[o * (MV_IN / 2) + k / 2];
        if (k & 1) {
          b = static_cast<uint8_t>(b | (nib << 4));
        } else {
          b = nib;
        }
      }
    }
  }

  // References.
  std::vector<float> y_q(MV_OUT, 0.0f);     // quantized weights, float math
  std::vector<float> y_f(MV_OUT, 0.0f);     // unquantized weights
  for (int o = 0; o < MV_OUT; ++o) {
    double aq = 0.0, af = 0.0;
    for (int k = 0; k < MV_IN; ++k) {
      aq += static_cast<double>(Wq[o * MV_IN + k]) * x[k];
      af += static_cast<double>(W[o * MV_IN + k]) * x[k];
    }
    y_q[o] = static_cast<float>(aq);
    y_f[o] = static_cast<float>(af);
  }

  // Kernel under test.
  std::vector<float> y(MV_OUT, 0.0f);
  const auto t0 = std::chrono::steady_clock::now();
  matvec_int4(packed.data(), scales.data(), x.data(), y.data());
  const auto t1 = std::chrono::steady_clock::now();

  // Errors are scaled by the largest output magnitude, so near-zero outputs
  // do not dominate the result.
  double max_abs_q = 0.0, max_err_f = 0.0, max_abs_f = 0.0, max_mag_q = 0.0;
  for (int o = 0; o < MV_OUT; ++o) {
    max_abs_q = std::fmax(max_abs_q, std::fabs(y[o] - y_q[o]));
    max_mag_q = std::fmax(max_mag_q, std::fabs(y_q[o]));
    max_err_f = std::fmax(max_err_f, std::fabs(y[o] - y_f[o]));
    max_abs_f = std::fmax(max_abs_f, std::fabs(y_f[o]));
  }
  const double scaled_q = max_abs_q / max_mag_q;

  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double weight_bytes = static_cast<double>(packed.size() + scales.size() * sizeof(float));

  std::printf("max error vs quantized reference: %.3e absolute, %.3e of max |y| (%.3f)\n",
              max_abs_q, scaled_q, max_mag_q);
  std::printf("max error vs float reference:     %.3e absolute, %.3e of max |y| (%.3f)\n",
              max_err_f, max_err_f / max_abs_f, max_abs_f);
  std::printf("host time for one matvec: %.3f ms (%.2f MB of weights read)\n",
              ms, weight_bytes / 1e6);

  const bool pass = scaled_q < 1e-4;
  std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
