# Run log: KV260 CPU thread scaling, Qwen2.5-0.5B-Instruct (Q4_K_M)

Date: 21 Nov 2026
Follows: `docs/runs/2026-11-21-kv260-cpu-baseline.md` (same board, build, model and command, except `-t`)
Status: measured. Power not yet measured.

## Setup

- Board: AMD Kria KV260, 4 × Cortex-A53, 3.8 GiB RAM
- llama.cpp build `19bc41d`, Release
- Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (462.96 MiB as reported by llama-bench)

## Command

```
for t in 1 2 3; do ./build/bin/llama-bench -m ~/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf -t $t -p 128 -n 64 -r 3; done
```

The `-t 4` figures come from the baseline run.

## Results (tokens/s, mean ± std over 3 repetitions)

| Threads | pp128 (prompt) | Speed-up vs 1 thread | tg64 (generation) | Speed-up vs 1 thread |
|---|---|---|---|---|
| 1 | 2.22 ± 0.00 | 1.00 | 1.59 ± 0.00 | 1.00 |
| 2 | 4.40 ± 0.00 | 1.98 | 3.05 ± 0.00 | 1.92 |
| 3 | 6.57 ± 0.00 | 2.96 | 4.42 ± 0.00 | 2.78 |
| 4 | 8.69 ± 0.01 | 3.91 | 5.87 ± 0.02 | 3.69 |

The 4-thread row comes from `docs/runs/2026-11-21-kv260-cpu-baseline.md`.

## Notes

- Scaling is close to linear for both tests: about 92% efficiency at 4 threads for generation and about 98% for prompt processing (speed-up divided by thread count).
- This is consistent with the estimate in `fpga/DESIGN-qwen05b.md`: the CPU is limited by compute, not by DDR bandwidth. Generation at 4 threads reads about 1.8 GB/s (estimate), well below the 19.2 GB/s rated bandwidth.
- Because scaling is near-linear, the four A53 cores are the bottleneck. Adding PL compute is the only route to a large gain.

## Not yet measured

- Power draw during each run (needed for tokens per joule)
