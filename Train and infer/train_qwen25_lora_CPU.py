"""LoRA fine-tuning of Qwen2.5-1.5B-Instruct for narrative -> NSS JSON.

The script downloads the base model from Hugging Face into models/base_qwen25_1_5b,
trains LoRA adapters, and saves them into models/nss_qwen25_1_5b_lora.

Example:
    python train_qwen25_lora_CPU.py --train-file train_merged_normalized.jsonl --validation-file validation_merged_normalized.jsonl --cpu-threads 4 --max-length 768 --gradient-accumulation 8 --epochs 3 --dtype bfloat16
"""
from __future__ import annotations

import truststore
truststore.inject_into_ssl()

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="train.jsonl")
    p.add_argument("--validation-file", default="validation.jsonl")
    p.add_argument("--model-root", default="models")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--cpu-threads", type=int, default=4, help="PyTorch intra-op CPU threads. 4 is a conservative laptop default.")
    p.add_argument("--resume-from-checkpoint", default=None)
    return p.parse_args()


def download_base_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def tokenize_example(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, List[int]]:
    messages = example["messages"]
    if len(messages) != 3 or messages[-1]["role"] != "assistant":
        raise ValueError("Each sample must contain system, user and assistant messages.")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    encoded = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.copy()
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))

    if all(x == -100 for x in labels):
        raise ValueError("The assistant answer was truncated. Increase --max-length.")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CausalJsonCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            n = max_len - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [pad_id] * n)
            batch["attention_mask"].append(x["attention_mask"] + [0] * n)
            batch["labels"].append(x["labels"] + [-100] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    # CPU-friendly defaults: cap BLAS/OpenMP parallelism so training does not
    # monopolize every logical thread on a laptop. Override with --cpu-threads.
    threads = max(1, args.cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f"CPU threads: {torch.get_num_threads()}")

    root = Path(args.model_root)
    base_dir = root / "base_qwen25_1_5b"
    adapter_dir = root / "nss_qwen25_1_5b_lora"
    checkpoints_dir = root / "training_checkpoints"

    print(f"Downloading/caching {MODEL_ID} in: {base_dir.resolve()}")
    download_base_model(base_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    tokenized = ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=ds["train"].column_names,
        desc="Applying Qwen2.5 chat template and masking prompts",
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        use_cpu=True,        
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        #warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        #evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=50,        
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=False,
        bf16=(args.dtype == "bfloat16"),
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=CausalJsonCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    metrics = trainer.evaluate()
    (adapter_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"LoRA adapter saved to: {adapter_dir.resolve()}")


if __name__ == "__main__":
    main()
