# FPGA design plan: Qwen2.5-0.5B-Instruct on the KV260

Status: planning with milestone 1 started, 21 Nov 2026. The matrix-vector kernel is written and passes a host test. No hardware build or board test has run yet. Figures marked **measured** come from the board or our runs. Figures marked **estimate** are our own arithmetic from the model config and must be checked.

## 1. Target to beat

From the CPU baseline (`docs/runs/2026-11-21-kv260-cpu-baseline.md`, measured):

| Metric | CPU baseline (measured) |
|---|---|
| Generation, tg64 | 5.87 ± 0.02 tokens/s |
| Prompt processing, pp128 | 8.69 ± 0.01 tokens/s |

Thread scaling (1 to 4 threads) is measured in `docs/runs/2026-11-21-kv260-thread-scaling.md`: generation rises from 1.59 to 5.87 tokens/s, close to linear. Power is not measured yet. The FPGA design should report both.

## 2. Model facts

Qwen2.5-0.5B-Instruct config (from the Hugging Face config files, checked 21 Nov 2026): hidden size 896, 24 layers, 14 query heads, 2 key/value heads, head dimension 64, MLP intermediate size 4864, vocabulary 151,936, tied input and output embeddings.

Parameter counts (**estimate**, from the config):

- Attention per layer: about 1.84 M (Q, K, V, O projections)
- MLP per layer: about 13.07 M (gate, up, down projections)
- All 24 layers: about 358 M
- Embedding and LM head (tied): about 136 M
- Total: about 494 M

Memory traffic per decoded token (**estimate**, 4.5 bits per weight for the layers and 6.5 bits for the LM head, which is roughly the Q4_K_M mix):

- Layers: about 201 MB
- LM head: about 111 MB
- Total: about 312 MB per token

At the measured 5.87 tokens/s, the CPU reads about 1.8 GB/s. The KV260 DDR4 interface is rated at 19.2 GB/s, so the CPU baseline is not limited by memory bandwidth. It is limited by compute and latency on the A53 cores. This changes the design goal: the FPGA must add compute throughput, and it must stream about 1.8 GB/s or more of weights to beat the baseline.

KV cache (**estimate**): 12 KB per token in fp16 (24 layers × 2 heads × 64 × K and V × 2 bytes). A 4096-token context needs about 50 MB, which fits easily in DDR.

## 3. Proposed hardware and software partition

Board and part (from the existing Vivado project `kv260_edge_detector_stream.xpr`): `xck26-sfvc784-2LV-c`, Vivado 2025.2 project. The board runs XRT 2.13 and VART 2.5.0, with PYNQ 3.0.1.

| Block | Where | Reason |
|---|---|---|
| Q, K, V, O projections (all layers) | PL | Most of the MACs at short context |
| MLP gate, up, down (all layers) | PL | Largest matrices (896 × 4864) |
| LM head (tied embedding) | PL, stage 2 | Largest single read per token |
| Token embedding lookup | PS | Single row per token |
| RMSNorm, RoPE, residual adds | PS | Small, element-wise |
| Attention scores and softmax over the KV cache | PS first, PL later | Cost grows with context length |
| SiLU and gating multiply | PS first, PL later | Element-wise |
| Sampling, tokenisation | PS | Trivial |

Attention cost grows with context: at 4096 tokens it is about half the per-layer linear MACs (**estimate**), so we will profile on the board before deciding whether it moves to the PL.

## 4. Datapath

- Weights: 4-bit, grouped scales (group size 128 is typical for AWQ, which the paper uses). The exact format is an open question (see §7). Dequantisation happens inside the PL.
- Activations: 8-bit or 16-bit fixed point, or fp16 if the tools allow it. Decide after accuracy tests.
- Decode (one token): a streaming matrix-vector engine. Weights are read once per token, so the engine must keep the DDR interface busy.
- Prefill (many tokens): a tiled matrix-matrix engine that reuses each weight tile across the prompt tokens.
- Interface: AXI HP ports for weight streams, AXI-Lite for control from PYNQ.

## 5. Milestones

1. **Measure on the CPU.** Thread scaling (`-t 1`, `-t 2`, `-t 3`), power at the board input during each run, and a profile of where time goes in decode. *Owner: team, on the board.*
2. **One matrix-vector kernel.** Build a single int4 matrix-vector core for one 896 × 4864 matrix (the MLP up projection). Test it standalone from PYNQ against a NumPy reference. Record throughput in GB/s and GOPS.
3. **One transformer block.** Route all seven linear layers of one block through the PL. Run the rest on the CPU. Check the output against the CPU reference, token by token.
4. **Full decode.** All 24 layers plus the LM head. Compare tokens/s and tokens per joule against the CPU baseline.
5. **Prefill.** Add the tiled matrix-matrix engine. Compare pp128 with the baseline.

Each milestone gets a run log under `docs/runs/`.

### Progress

| Milestone | Status |
|---|---|
| 1. CPU measurements (thread scaling, profile) | Thread scaling done. Power and profile not done |
| 2. One matrix-vector kernel | Kernel written (`fpga/hls/matvec_int4.h`). Host test passes (`fpga/sim/`, run log `docs/runs/2026-11-21-matvec-int4-host-test.md`). Vitis HLS synthesis and board test not started |
| 3. One transformer block on the PL | Not started |
| 4. Full decode | Not started |
| 5. Prefill | Not started |

Milestone 2 host test: the kernel matches the quantized reference to float rounding. Against the unquantized product on random Gaussian data, the error is about 11% of the largest output. That is the cost of round-to-nearest 4-bit weights with no calibration. The next check is the same comparison on real Qwen2.5-0.5B weights, with and without AWQ-style scaling.

## 6. Success criteria (first pass)

- Decode faster than 5.87 tokens/s with the same model and quantisation, or the same speed at lower power.
- Output matches the CPU reference within an agreed tolerance on a fixed prompt set.
- Power and thread results recorded for both the CPU and FPGA runs.

## 7. Open questions

- Quantisation format: use llama.cpp's Q4_K_M layout so the CPU and FPGA runs match exactly, or our own AWQ-style int4 with group size 128 to follow the paper?
- Tooling: Vitis HLS for the kernels, or hand-written RTL? Confirm the Vivado and Vitis versions on the laptop match the 2025.2 project and the XRT 2.13 runtime on the board.
- Accuracy target: which perplexity or task score counts as "matches the CPU reference"?
- Power measurement: which meter, and at which point (12 V input or the board's own sensors)?

## 8. Sources

- Qwen2.5-0.5B-Instruct config values: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct (mirrors checked: the config values above appear in several public copies of the model's `config.json`)
- Paper: On-Device Qwen2.5, arXiv 2504.17376 (Qwen2.5-0.5B on the KV260 with AWQ and a CPU/FPGA split)
- TeLLMe, arXiv 2504.16266 (ternary LLM accelerator on the KV260, prefill and decode)
- KV260 memory bandwidth figure (19.2 GB/s), quoted in arXiv 2504.17376
