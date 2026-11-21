// Top-level wrapper for Vitis HLS. The kernel itself is in matvec_int4.h.
#include "matvec_int4.h"

void matvec_int4_top(const uint8_t *w_packed,
                     const float *scales,
                     const float *x,
                     float *y) {
  matvec_int4(w_packed, scales, x, y);
}
