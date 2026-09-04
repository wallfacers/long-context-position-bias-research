#!/usr/bin/env python3
"""Run a resumable, completion-only 4-bit QLoRA position-bias ablation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]


def portable_artifact_ref(value: str | Path) -> str:
    path = Path(value)
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latest_checkpoint(output: Path) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []
    if output.is_dir():
        for path in output.iterdir():
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if match and path.is_dir() and (path / "trainer_state.json").is_file():
                candidates.append((int(match.group(1)), path))
    return (None, 0) if not candidates else (max(candidates)[1], max(candidates)[0])


def load_training_data(path: Path) -> tuple[Any, bool]:
    if path.is_dir():
        dataset = load_from_disk(str(path))
        required = {"input_ids", "completion_mask"}
        if not required.issubset(dataset.column_names):
            raise ValueError(
                f"Pre-tokenized dataset requires {sorted(required)}, got {dataset.column_names}"
            )
        return dataset, True

    dataset = load_dataset("json", data_files=str(path), split="train")
    if "messages" not in dataset.column_names:
        raise ValueError(f"JSONL lacks messages column: {path}")

    original_columns = list(dataset.column_names)

    def to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError("Every row must contain system, user, assistant in that order")
        return {"prompt": messages[:-1], "completion": [messages[-1]]}

    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=original_columns,
        desc="Converting to prompt-completion format",
    )
    return dataset, False


class StopAfterStepsCallback(TrainerCallback):
    def __init__(self, stop_after_steps: int) -> None:
        self.stop_after_steps = stop_after_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step >= self.stop_after_steps:
            control.should_save = True
            control.should_training_stop = True
        return control


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local base-model directory")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8320)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=3)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise SystemExit("--model must be a local directory; remote downloads are disabled")
    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if args.max_steps <= 0 or args.save_steps <= 0:
        raise SystemExit("--max-steps and --save-steps must be positive")
    if args.stop_after_steps is not None:
        if not 0 < args.stop_after_steps < args.max_steps:
            raise SystemExit("--stop-after-steps must be between 1 and max_steps - 1")
        if args.stop_after_steps % args.save_steps:
            raise SystemExit("--stop-after-steps must be divisible by --save-steps")
    if args.resume != "auto" and args.resume != "none" and not Path(args.resume).is_dir():
        raise SystemExit(f"Resume checkpoint does not exist: {args.resume}")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    if (args.output / "TRAINING_COMPLETE.json").is_file():
        print(f"Training is already complete: {args.output}")
        return 0

    checkpoint, checkpoint_step = latest_checkpoint(args.output)
    if args.stop_after_steps is not None and checkpoint_step >= args.stop_after_steps:
        print(
            f"Canary checkpoint already reached step {checkpoint_step}; "
            "not spending GPU time again."
        )
        return 0
    if args.resume == "auto":
        resume_from: str | bool | None = str(checkpoint) if checkpoint else None
    elif args.resume == "none":
        if checkpoint:
            raise SystemExit(
                f"Existing {checkpoint}; refusing to restart and overwrite paid work. Use --resume auto."
            )
        resume_from = None
    else:
        resume_from = args.resume

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset, pretokenized = load_training_data(args.data)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(args.output),
        max_length=args.max_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # Transformers 5.x accepts a fractional warmup through warmup_steps;
        # values in [0, 1) are interpreted as a ratio of total training steps.
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        completion_only_loss=True,
        packing=False,
        dataset_num_proc=args.dataset_num_proc,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "dtype": "bfloat16",
            "local_files_only": True,
            "trust_remote_code": False,
            "attn_implementation": args.attn_implementation,
        },
    )
    callbacks: list[TrainerCallback] = []
    if args.stop_after_steps is not None:
        callbacks.append(StopAfterStepsCallback(args.stop_after_steps))

    started_at = utc_now()
    run_id = started_at.replace(":", "").replace("+00:00", "Z") + f"-pid{os.getpid()}"
    run_config = {
        "schema_version": "qlora-run-v1",
        "run_id": run_id,
        "started_at": started_at,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "model": portable_artifact_ref(args.model),
        "data": portable_artifact_ref(args.data),
        "pretokenized": pretokenized,
        "resume_from": portable_artifact_ref(str(resume_from)) if resume_from else None,
        "checkpoint_step_before_run": checkpoint_step,
        "arguments": vars(args)
        | {
            "model": portable_artifact_ref(args.model),
            "data": portable_artifact_ref(args.data),
            "output": portable_artifact_ref(args.output),
        },
    }
    write_json(args.output / "run_config.json", run_config)
    invocation_dir = args.output / "invocations"
    write_json(invocation_dir / f"{run_id}.config.json", run_config)

    trainer = SFTTrainer(
        model=str(args.model),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        quantization_config=quantization,
        peft_config=lora,
        callbacks=callbacks,
    )
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.monotonic() - started
    global_step = int(trainer.state.global_step)
    summary = {
        "schema_version": "qlora-result-v1",
        "run_id": run_id,
        "finished_at": utc_now(),
        "global_step": global_step,
        "elapsed_seconds_this_invocation": elapsed,
        "seconds_per_step_this_invocation": elapsed / max(1, global_step - checkpoint_step),
        "metrics": result.metrics,
    }
    write_json(invocation_dir / f"{run_id}.result.json", summary)

    if global_step >= args.max_steps:
        final_adapter = args.output / "final_adapter"
        trainer.save_model(str(final_adapter))
        tokenizer.save_pretrained(str(final_adapter))
        write_json(args.output / "TRAINING_COMPLETE.json", summary)
        print(f"TRAINING COMPLETE: {final_adapter}")
    elif args.stop_after_steps is not None and global_step >= args.stop_after_steps:
        write_json(args.output / "CANARY_COMPLETE.json", summary)
        print(
            f"CANARY COMPLETE at step {global_step}; resume with the same --max-steps "
            "and --resume auto."
        )
    else:
        raise RuntimeError(
            f"Training stopped unexpectedly at step {global_step}, expected {args.max_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
