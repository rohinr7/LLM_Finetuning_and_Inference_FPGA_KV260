# Run log: KV260 CPU baseline, Qwen2.5-0.5B-Instruct (Q4_K_M)

Date: 21 Nov 2026
Owner: Hodgkin-Huxley Labs team
Status: measured on the board, single run set (3 repetitions per test)

## Hardware and software

- Board: AMD Kria KV260 (K26 SOM), hostname `kria`
- CPU: 4 cores (Arm Cortex-A53, per the paper's description)
- RAM: 3.8 GiB total, about 3.1 GiB available at start, no swap
- Storage: 29 GB root filesystem, 13 GB free before the run
- OS: Ubuntu 22.04.5 LTS (jammy), kernel 5.15.0-107
- Python 3.10.12, PYNQ 3.0.1
- XRT 2.13.479, VART 2.5.0 (the stock Vitis AI DPU runtime is installed; this run did not use it)
- Inference engine: llama.cpp, build `19bc41d`, built on the board with `cmake -DCMAKE_BUILD_TYPE=Release`

## Model

- File: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (469 MB on disk, 462.96 MiB as reported by llama-bench)
- Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- Quantization: Q4_K_M (llama.cpp k-quant, 4-bit)
- Parameters: 630.17 M, as llama-bench reports it (includes the embedding table)

## Command

```
./build/bin/llama-bench -m ~/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf -t 4 -p 128 -n 64 -r 3
```

## Results

| Test | Meaning | Throughput (tokens/s) |
|---|---|---|
| pp128 | Prompt processing, 128 tokens | 8.69 ± 0.01 |
| tg64 | Token generation, 64 tokens | 5.87 ± 0.02 |

Raw output saved on the board at `~/llm/baseline_qwen05b_cpu.txt`.

## Notes

- llama-bench labels the model "qwen2 1B". That is llama.cpp's naming for this architecture. The model is Qwen2.5-0.5B.
- Generation speed is above the 2.8 tokens/s baseline reported in arXiv 2504.17376. That paper's quantization, thread count, and baseline definition differ from this run, so the numbers are not directly comparable.
- Prompt processing is slow, about 8.7 tokens/s. A 500-token prompt takes about a minute before the first output token. Faster prefill is a target for the FPGA design.
- The first build was interrupted when the SSH session dropped. It was resumed inside tmux and completed without changes.

## Not yet measured

- Thread scaling (`-t 1`, `-t 2`, `-t 3`)
- Power draw during the run
- Memory bandwidth utilization
- Output quality (perplexity or task accuracy) for the quantized model

## Next

- Log thread scaling and power in a follow-up run file.
- Use this run as the target for the first FPGA design.
