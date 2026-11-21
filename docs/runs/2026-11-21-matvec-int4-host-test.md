# Run log: int4 matrix-vector kernel, host test

Date: 21 Nov 2026
Milestone: 2 (one matrix-vector kernel), host-side check only. No FPGA build or board run.
Code: `fpga/hls/matvec_int4.h`, `fpga/sim/test_matvec_int4.cpp`

## Setup

- Host: Linux (sandbox), g++ with `-std=c++17 -O2 -Wall -Wextra`
- Shape: 896 inputs, 4864 outputs (MLP up-projection size of Qwen2.5-0.5B)
- Weights: random Gaussian (std 0.05), packed int4, 4-bit round-to-nearest, group size 128, float scales
- Activations: random Gaussian (std 1), float
- Seed: 12345

## Command

```
g++ -std=c++17 -O2 -Wall -Wextra -I../hls test_matvec_int4.cpp -o test_matvec_int4 && ./test_matvec_int4
```

## Results

| Check | Result |
|---|---|
| Max error vs quantized reference | 1.61e-6 absolute, 2.8e-7 of max output (5.778). Pass threshold 1e-4 |
| Max error vs unquantized float reference | 0.664 absolute, 11.2% of max output (5.909) |
| Host time, one matvec (unoptimised loop) | 4.67 ms, 2.32 MB of weights read |
| Result | PASS |

## Notes

- The first version of the test used a per-element relative error, which failed on near-zero outputs. The metric was changed to error relative to the largest output. This is a test-metric fix, not a kernel change.
- The 11% error against the unquantized product is the expected cost of plain 4-bit round-to-nearest on random data. It is not a model-quality result. The next check is the same comparison on real Qwen2.5-0.5B weights.
- The 4.67 ms figure is on the laptop-class sandbox, not the KV260. It is recorded only to confirm the code runs. Board timing comes after synthesis.

## Next

- Vitis HLS: C simulation and synthesis report for `matvec_int4.h` (utilisation, latency, initiation interval). Run on the Windows laptop with the installed Vitis tools.
- Compare the error on real Qwen2.5-0.5B layer weights, with and without AWQ-style scaling.
