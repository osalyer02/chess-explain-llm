#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) with QLoRA for FEN-only chess prompts, with WandB logging
and a time limit flag to stop training after a given wall-clock duration.

"""

import argparse, os, time, re
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ------------------- Time limit support ------------------- #

def parse_duration_to_seconds(s: Optional[str]) -> Optional[int]:
    """
    Parse durations like:
      "3600"  -> 3600 seconds
      "45m"   -> 2700 seconds
      "4h"    -> 14400 seconds
      "1h30m" -> 5400 seconds
      "2h15m20s" -> 8120 seconds
    Returns None if s is None/empty.
    """
    if not s:
        return None
    s = s.strip().lower()
    # plain integer seconds
    if s.isdigit():
        return int(s)
    # pattern with h/m/s parts
    pattern = r'^\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?\s*$'
    m = re.match(pattern, s)
    if not m:
        raise ValueError(f"Unrecognized training_time format: {s!r}")
    h = int(m.group(1)) if m.group(1) else 0
    mnts = int(m.group(2)) if m.group(2) else 0
    sec = int(m.group(3)) if m.group(3) else 0
    total = h * 3600 + mnts * 60 + sec
    if total <= 0:
        raise ValueError(f"training_time must be > 0, got: {s!r}")
    return total


class TimeLimitCallback(TrainerCallback):
    """
    Stops training once `limit_seconds` have elapsed (wall clock).
    Triggers a save on the step it stops.
    """
    def __init__(self, limit_seconds: int):
        self.limit_seconds = limit_seconds
        self.start_time = None
        self._stopped = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._stopped or self.start_time is None:
            return control
        elapsed = time.time() - self.start_time
        if elapsed >= self.limit_seconds:
            # Ask Trainer to save and stop after this step
            control.should_save = True
            control.should_training_stop = True
            self._stopped = True
            print(f"\nTime limit reached ({int(elapsed)}s). "
                  f"Requesting save and graceful stop...\n")
        return control

# ------------------- Helpers ------------------- #

def format_example(ex: Dict[str, str]) -> str:
    return ex["prompt"] + "\n" + ex["response"]

# ------------------- Main ------------------- #

def main():
    ap = argparse.ArgumentParser()
    # Model / data
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="HF repo or local path to base model")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default=None)
    ap.add_argument("--out_dir", default="out/qwen3-4b-qlora")

    # WandB
    ap.add_argument("--wandb_project", default=None, help="Weights & Biases project name")
    ap.add_argument("--wandb_run", default=None, help="Optional run name for WandB")

    # Tokenization / lengths
    ap.add_argument("--max_len", type=int, default=384,
                    help="Max sequence length in TOKENS (not characters)")
    ap.add_argument("--pack_sequences", action="store_true",
                    help="Off by default to preserve label masking per example.")

    # Optim / training
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_schedule", default="cosine", choices=["cosine","linear","constant"])
    ap.add_argument("--bf16", type=bool, default=False)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--save_each_epoch", action="store_true")
    ap.add_argument("--save_steps", type=int, default=200, help="Checkpoint every N steps")
    ap.add_argument("--save_total_limit", type=int, default=2, help="Keep at most N checkpoints")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="*", default=[
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ])

    # Impl toggles
    ap.add_argument("--attn_impl", default=None,
                    choices=[None, "flash_attention_2", "eager"],
                    help="If you have FlashAttention 2 installed, set this for memory/speed wins.")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Enable if model requires custom code (safe for Qwen).")

    # Logging / misc
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluation_strategy", default="no",
                    choices=["no","steps","epoch"],
                    help="When to run validation (if val_jsonl is provided).")
    ap.add_argument("--eval_steps", type=int, default=1000)

    # NEW: time-box training
    ap.add_argument("--training_time", type=str, default=3600,
                    help="Wall-clock limit. Examples: '4h', '45m', '1h30m', '3600' (seconds).")
    
    ap.add_argument("--wandb_id", default=None, help="Optional WandB run ID to resume")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------- Initialize WandB -------- #
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run, id=args.wandb_id, resume="allow", config=vars(args))
        print(f"WandB run started: {run.name}")

    # -------- Load datasets -------- #
    train_ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    val_ds = load_dataset("json", data_files=args.val_jsonl, split="train") if args.val_jsonl else None

    # -------- Tokenizer (Qwen safe setup) -------- #
    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code or True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # -------- QLoRA config -------- #
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # -------- Base model -------- #
    from_pretrained_kwargs = dict(
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=args.trust_remote_code or True,
    )
    if args.attn_impl is not None:
        from_pretrained_kwargs["attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **from_pretrained_kwargs,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    

    # -------- Tokenize with response-only loss (correct masking) -------- #
    def tokenize_with_labels(batch):
        prompts, responses = batch["prompt"], batch["response"]
        input_ids, attention_masks, labels = [], [], []

        for p, r in zip(prompts, responses):
            # 1) Get true prompt length (no padding)
            p_tok = tok(
                p,
                truncation=True,
                max_length=args.max_len,
                add_special_tokens=True,
                padding=False,
            )
            prompt_len = len(p_tok["input_ids"])

            # 2) Tokenize full sample with fixed padding
            full_tok = tok(
                p + "\n" + r,
                truncation=True,
                max_length=args.max_len,
                add_special_tokens=True,
                padding="max_length",
            )
            ids = full_tok["input_ids"]
            attn = full_tok["attention_mask"]

            # 3) Labels: mask prompt tokens (=-100), supervise only response tokens
            start = min(prompt_len, len(ids))
            lab = [-100] * len(ids)
            for i in range(start, len(ids)):
                lab[i] = ids[i]

            input_ids.append(ids)
            attention_masks.append(attn)
            labels.append(lab)

        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    train_tok = train_ds.map(tokenize_with_labels, batched=True, remove_columns=train_ds.column_names)
    eval_tok = val_ds.map(tokenize_with_labels, batched=True, remove_columns=val_ds.column_names) if val_ds else None

    # We pre-pad; use default_data_collator to stack tensors as-is
    collator = default_data_collator 

    save_strategy = "epoch" if args.save_each_epoch else "steps"

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_schedule,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",                 # ensure step-based saves so time-stop triggers a save
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=(args.evaluation_strategy if eval_tok is not None else "no"),
        eval_steps=(args.eval_steps if eval_tok is not None and args.evaluation_strategy=="steps" else None),
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_num_workers=0,            
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_32bit",
        report_to="wandb" if args.wandb_project else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tok,  
        callbacks=[],
    )

    # Resume if checkpoint exists
    last_ckpt = get_last_checkpoint(args.out_dir) if os.path.isdir(args.out_dir) else None

    # Time limit callback if requested
    tl_seconds = parse_duration_to_seconds(args.training_time)
    if tl_seconds:
        trainer.add_callback(TimeLimitCallback(limit_seconds=tl_seconds))
        print(f"Will stop training after ~{tl_seconds} seconds.")

    if last_ckpt is not None:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("Starting fresh training run")
        trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved adapter (and tokenizer) to", args.out_dir)

    if args.wandb_project:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
