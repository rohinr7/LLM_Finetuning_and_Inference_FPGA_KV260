"""QLoRA fine-tuning for small causal language models.

The base model is loaded in 4-bit (NF4) and frozen. Only the LoRA adapters are trained.
Loss is computed on the response tokens only.

Works on a 4 GB GPU (small models) and on 16 GB cloud GPUs (larger models).
Uses fp16 when bf16 is not supported, which is the case on the GTX 1650.

Example:
    python src/finetune_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/sample.jsonl --out outputs/smoke-test --max_steps 20
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    p.add_argument("--data", required=True, help="JSONL file with 'prompt' and 'response' fields")
    p.add_argument("--out", required=True, help="Output directory for the LoRA adapter and logs")
    p.add_argument("--max_len", type=int, default=512, help="Maximum tokens per example")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1, help="Overrides epochs when > 0 (use for smoke tests)")
    p.add_argument("--batch", type=int, default=1, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=8, help="Effective batch = batch * grad_accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_4bit", action="store_true", help="Load the base model in fp16 instead of 4-bit (more VRAM)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_example(example, tok, max_len):
    """Prompt tokens are masked with -100 so loss is computed on the response only."""
    prompt_ids = tok(example["prompt"] + "\n", add_special_tokens=False)["input_ids"]
    response_ids = tok(example["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = (prompt_ids + response_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_len]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def load_base_model(model_id, use_4bit, compute_dtype):
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", torch_dtype=compute_dtype
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, device_map="auto")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    compute_dtype = pick_compute_dtype()
    use_fp16 = compute_dtype == torch.float16
    use_bf16 = compute_dtype == torch.bfloat16
    print(f"Compute dtype: {compute_dtype}")

    tok = build_tokenizer(args.model)
    raw = load_dataset("json", data_files=args.data, split="train")
    dataset = raw.map(
        lambda ex: encode_example(ex, tok, args.max_len),
        remove_columns=raw.column_names,
    )
    print(f"Training examples: {len(dataset)}")

    model = load_base_model(args.model, not args.no_4bit, compute_dtype)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        seed=args.seed,
    )

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {peak:.2f} GB")

    summary = {
        "model": args.model,
        "data": args.data,
        "examples": len(dataset),
        "max_len": args.max_len,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "four_bit": not args.no_4bit,
        "compute_dtype": str(compute_dtype),
        "max_steps": args.max_steps,
        "epochs": args.epochs,
    }
    with open(os.path.join(args.out, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
