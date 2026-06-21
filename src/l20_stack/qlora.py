"""Validated configuration and telemetry helpers for L20 QLoRA runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class QLoRAConfig:
    model_name: str
    train_file: str
    eval_file: str
    output_dir: str
    max_length: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_steps: int = 100
    warmup_steps: int = 5
    eval_steps: int = 20
    save_steps: int = 20
    logging_steps: int = 1
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    seed: int = 42
    use_gradient_checkpointing: bool = True
    use_bf16: bool = True
    packing: bool = True

    @classmethod
    def from_file(cls, path: str) -> "QLoRAConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "target_modules" in payload:
            payload["target_modules"] = tuple(payload["target_modules"])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.max_length <= 0 or self.micro_batch_size <= 0:
            raise ValueError("max_length and micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0 or self.max_steps <= 0:
            raise ValueError("gradient_accumulation_steps and max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.eval_steps <= 0 or self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("eval_steps, save_steps, and logging_steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be non-negative and less than max_steps")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if Path(self.train_file).resolve() == Path(self.eval_file).resolve():
            raise ValueError("train_file and eval_file must be different")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records


def normalize_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("each record must contain a non-empty messages list")
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("messages require a valid role and string content")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant":
        raise ValueError("the final message must be an assistant response")
    return normalized


def dataset_fingerprint(records: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        messages = normalize_messages(record)
        digest.update(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def assert_disjoint(train_records: Iterable[Dict[str, Any]], eval_records: Iterable[Dict[str, Any]]) -> None:
    train_hashes = {
        hashlib.sha256(
            json.dumps(normalize_messages(record), sort_keys=True).encode("utf-8")
        ).digest()
        for record in train_records
    }
    eval_hashes = {
        hashlib.sha256(
            json.dumps(normalize_messages(record), sort_keys=True).encode("utf-8")
        ).digest()
        for record in eval_records
    }
    overlap = train_hashes.intersection(eval_hashes)
    if overlap:
        raise ValueError(f"train/eval exact overlap detected: {len(overlap)} records")


def contamination_report(
    train_records: Iterable[Dict[str, Any]], eval_records: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """Report exact and normalized prompt overlap before a training run."""

    def prompts(records):
        result = set()
        for record in records:
            messages = normalize_messages(record)
            prompt = "\n".join(
                message["content"].strip().lower()
                for message in messages
                if message["role"] != "assistant"
            )
            result.add(" ".join(prompt.split()))
        return result

    train_prompts = prompts(train_records)
    eval_prompts = prompts(eval_records)
    overlap = sorted(train_prompts.intersection(eval_prompts))
    return {
        "train_unique_prompts": len(train_prompts),
        "eval_unique_prompts": len(eval_prompts),
        "normalized_prompt_overlap": len(overlap),
        "overlap_examples": overlap[:10],
    }
