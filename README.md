# LLM Fine-tuning and Inference on FPGA (KV260)

Part of the Hodgkin-Huxley Labs NPU initiative. This repo tracks the first three steps of our work:

1. **Fine-tune** a small quantized language model (QLoRA) on modest GPUs.
2. **Run** the quantized model for local inference.
3. **Map** the inference workload onto an FPGA (AMD Kria KV260) as a design target.

## Status

| Step | Status |
|---|---|
| Literature review | Done. See the project doc `claude/phase1b-sota-llm-quantization-and-fpga.md` |
| GPU check and QLoRA fine-tuning script | Initial scaffold (this commit) |
| First fine-tune on a 0.5B model (GTX 1650, 4 GB) | Not started |
| Scale-up to 1.5B–7B on Kaggle or Colab (T4, 16 GB) | Not started |
| Quantized inference on KV260, CPU baseline (llama.cpp, Qwen2.5-0.5B Q4_K_M) | Done, 21 Nov 2026. Board: Kria KV260, 4 × Cortex-A53, 3.8 GiB RAM, Ubuntu 22.04. llama.cpp `19bc41d`, model 463 MiB. Prompt processing (pp128): 8.69 ± 0.01 tokens/s. Generation (tg64): 5.87 ± 0.02 tokens/s. 3 repetitions. Not yet measured: thread scaling, power. Log: `docs/runs/2026-11-21-kv260-cpu-baseline.md` |
| Quantized inference on GPU (GGUF or bitnet.cpp) | Not started. Planned: Qwen2.5-0.5B on GTX 1650 (4 GB) |
| FPGA design for KV260 | Not started. Target to beat: 5.87 tokens/s generation and 8.69 tokens/s prompt processing (CPU baseline). Board software ready: PYNQ 3.0.1, XRT 2.13, VART 2.5.0. The stock Vitis AI DPU is for CNNs, so an LLM needs a custom design |

## Hardware used

- **Local:** NVIDIA GeForce GTX 1650, 4 GB VRAM (Turing, no tensor cores, no bf16). Use fp16.
- **Cloud (planned):** Kaggle Notebooks (T4 or P100, 16 GB) and Google Colab free tier (T4). Both have usage limits, so save checkpoints often.
- **Target FPGA:** AMD Kria KV260 (Zynq UltraScale+). Reference design point: TeLLMe reports about 9.5 tokens/s decode at 250 MHz and 6.72 W. [arXiv 2504.16266]

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/
│   └── sample.jsonl          # 5 example records to test the pipeline
├── scripts/
│   └── check_gpu.py          # reports GPU, VRAM, fp16/bf16 support, bitsandbytes
├── src/
│   └── finetune_qlora.py     # QLoRA fine-tuning (4-bit base + LoRA adapters)
├── docs/
│   └── runs/                 # one short log per run (model, settings, results)
└── fpga/                     # KV260 design work (empty for now)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Check the GPU and libraries
python scripts/check_gpu.py

# 2. Smoke test: a tiny fine-tune on the sample data
python src/finetune_qlora.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data data/sample.jsonl \
  --out outputs/smoke-test \
  --max_steps 20
```

### Data format

Each line of a `.jsonl` file is one example:

```json
{"prompt": "What does NPU stand for?", "response": "Neural processing unit."}
```

Loss is computed on the response only, so the model learns to answer and is not trained to repeat the prompt.

### Suggested model sizes

| Hardware | Model | Notes |
|---|---|---|
| GTX 1650, 4 GB | Qwen2.5-0.5B-Instruct, SmolLM2-360M | Start here. Expect `--max_len 512` and batch size 1 |
| GTX 1650, 4 GB | Qwen2.5-1.5B-Instruct | Upper limit; may need `--max_len 256` |
| Kaggle / Colab T4, 16 GB | Qwen2.5-7B-Instruct | QLoRA with the same script; estimates to be measured |

These are estimates. We will record measured VRAM and throughput here once the first runs are done.

## Working rules

- Every figure in this repo or in our docs needs a source or a measured result, with the date.
- Do not commit model weights, adapters or datasets with personal data. `outputs/` is git-ignored.
- Record each run in a short log under `docs/runs/` (model, data, settings, VRAM, loss, time).

## References

- QLoRA: https://arxiv.org/pdf/2305.14314
- LoRA/QLoRA profiling on an 8 GB consumer GPU: https://arxiv.org/abs/2509.12229
- BitNet b1.58 2B4T: https://arxiv.org/abs/2504.12285
- TeLLMe (ternary LLM on KV260): https://arxiv.org/abs/2504.16266
- TerEffic (ternary LLM on FPGA): https://arxiv.org/abs/2502.16473
