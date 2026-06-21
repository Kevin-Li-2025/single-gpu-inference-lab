#!/usr/bin/env python3
"""Run a measured single-L20 QLoRA training job."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

from l20_stack.qlora import (
    QLoRAConfig,
    assert_disjoint,
    contamination_report,
    dataset_fingerprint,
    normalize_messages,
    read_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = QLoRAConfig.from_file(args.config)

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_name() != "NVIDIA L20" or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this training path is calibrated only for NVIDIA L20 sm_89")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_records = read_jsonl(config.train_file)
    eval_records = read_jsonl(config.eval_file)
    assert_disjoint(train_records, eval_records)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class ChatDataset(Dataset):
        def __init__(self, records, max_length, packing=False):
            tokenized = []
            for record in records:
                messages = normalize_messages(record)
                prompt = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                answer = messages[-1]["content"] + tokenizer.eos_token
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(
                    prompt + answer,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                prompt_length = min(len(prompt_ids), len(full_ids))
                labels = [-100] * prompt_length + full_ids[prompt_length:]
                if not any(label != -100 for label in labels):
                    continue
                tokenized.append({"input_ids": full_ids, "labels": labels})
            if packing:
                self.examples = []
                packed_ids, packed_labels = [], []
                for item in tokenized:
                    packed_ids.extend(item["input_ids"])
                    packed_labels.extend(item["labels"])
                    while len(packed_ids) >= max_length:
                        self.examples.append(
                            {
                                "input_ids": packed_ids[:max_length],
                                "labels": packed_labels[:max_length],
                            }
                        )
                        packed_ids = packed_ids[max_length:]
                        packed_labels = packed_labels[max_length:]
                if packed_ids and any(label != -100 for label in packed_labels):
                    self.examples.append({"input_ids": packed_ids, "labels": packed_labels})
            else:
                self.examples = tokenized
            if not self.examples:
                raise ValueError("no trainable examples remain after tokenization")

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, index):
            return self.examples[index]

    def collate(examples):
        length = max(len(item["input_ids"]) for item in examples)
        input_ids, labels, attention_mask = [], [], []
        for item in examples:
            padding = length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    train_dataset = ChatDataset(train_records, config.max_length, packing=config.packing)
    eval_dataset = ChatDataset(eval_records, config.max_length, packing=False)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
    )

    compute_dtype = torch.bfloat16 if config.use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True,
    )

    def learning_rate(step):
        if step < config.warmup_steps:
            return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for batch in eval_loader:
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            losses.append(model(**batch).loss.float().item())
        model.train()
        return sum(losses) / len(losses)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    started = time.perf_counter()
    tokens = 0
    history = []
    best_eval = float("inf")

    for step in range(1, config.max_steps + 1):
        step_loss = 0.0
        step_tokens = 0
        for _ in range(config.gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=compute_dtype):
                loss = model(**batch).loss / config.gradient_accumulation_steps
            loss.backward()
            step_loss += loss.detach().float().item()
            step_tokens += int(batch["attention_mask"].sum().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens += step_tokens

        event = {"step": step, "train_loss": step_loss, "learning_rate": lr}
        if step % config.eval_steps == 0 or step == config.max_steps:
            event["eval_loss"] = evaluate()
            best_eval = min(best_eval, event["eval_loss"])
        history.append(event)
        if step % config.logging_steps == 0:
            print(json.dumps(event, sort_keys=True), flush=True)
        if step % config.save_steps == 0 or step == config.max_steps:
            checkpoint = output_dir / f"checkpoint-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "dataset": {
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "train_fingerprint": dataset_fingerprint(train_records),
            "eval_fingerprint": dataset_fingerprint(eval_records),
            "contamination": contamination_report(train_records, eval_records),
            "packed_train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
        },
        "model": {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "result": {
            "elapsed_seconds": elapsed,
            "tokens": tokens,
            "tokens_per_second": tokens / elapsed,
            "best_eval_loss": best_eval,
            "history": history,
        },
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
