# Run log: smoke test, QLoRA on Qwen2.5-0.5B-Instruct

- **Date:** 16 Nov 2026
- **Hardware:** Kaggle notebook, NVIDIA T4 (16 GB), compute dtype fp16
- **Commit:** 063d9b3
- **Command:**
  ```bash
  python src/finetune_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data/sample.jsonl --out outputs/smoke-test --max_steps 20
  ```
- **Settings:** 4-bit NF4 base, LoRA r=16, alpha=32, dropout 0.05, lr 2e-4 (cosine), warmup 5 steps, batch 1, grad accumulation 8, max_len 512, paged AdamW 8-bit
- **Data:** `data/sample.jsonl`, 5 examples
- **Trainable parameters:** 1,081,344 of 495,114,112 (0.22%)

## Results

| Metric | Value |
|---|---|
| Steps | 20 |
| Peak GPU memory | 1.09 GB |
| Training time | 18.6 s |
| Loss (step 5 → step 20) | 8.25 → 5.21 |
| Mean training loss | 6.68 |

## Notes

- This run only checks that the pipeline works. Five examples over 20 steps with 8-example accumulation means each example is seen about 32 times, so the loss drop reflects memorisation, not learning.
- Peak memory of 1.09 GB leaves most of the 16 GB free. A 7B model in 4-bit may fit at max_len 512; to be measured.
- Warnings seen, none fatal: `torch_dtype` is deprecated (use `dtype`), and `use_reentrant` should be passed explicitly to gradient checkpointing.

## Next

- Run `scripts/check_gpu.py` on the GTX 1650 and log the result here.
- Scale to a real dataset, then to Qwen2.5-1.5B and 7B on the T4, logging peak memory for each.
