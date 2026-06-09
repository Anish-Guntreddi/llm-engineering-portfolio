"""QLoRA fine-tuning for WorkflowLM.

Loads Qwen2.5-1.5B-Instruct in 4-bit (NF4), attaches a LoRA adapter, and SFT-trains it to
emit the gold JSON workflow plan for each NL description. Driven entirely by
``configs/train_v1.yaml`` with a fixed seed for reproducibility.

Training uses completion-only masking (loss on the assistant JSON only, not the prompt) when
the installed TRL supports it, and renders examples through the SAME chat template / system
prompt used at eval and serve time (via ``workflowlm.prompts``) to avoid train/serve skew.

Usage:
    python -m workflowlm.train.train_qlora --config configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Lazy heavy imports so the rest of the package works without the ML stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    from workflowlm.prompts import SYSTEM_PROMPT  # noqa: F401  (kept for parity/documentation)
    from workflowlm.prompts import build_messages

    set_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    base = cfg["model"]["base"]
    qcfg = cfg["qlora"]
    tcfg = cfg["train"]

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Build datasets as rendered chat strings (prompt + gold JSON completion) ----
    def render(rows: list[dict]) -> Dataset:
        texts = []
        for r in rows:
            messages = build_messages(r["input"])
            messages.append({"role": "assistant", "content": r["output"]})
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        return Dataset.from_dict({"text": texts})

    train_ds = render(load_jsonl(Path(cfg["data"]["train"])))
    val_ds = render(load_jsonl(Path(cfg["data"]["val"])))

    # ---- 4-bit base model ----
    compute_dtype = getattr(torch, qcfg["bnb_4bit_compute_dtype"])
    bnb = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=compute_dtype
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=qcfg["lora_r"],
        lora_alpha=qcfg["lora_alpha"],
        lora_dropout=qcfg["lora_dropout"],
        target_modules=qcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"],
        save_strategy=tcfg["save_strategy"],
        eval_strategy=tcfg["eval_strategy"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        max_grad_norm=tcfg["max_grad_norm"],
        optim=tcfg["optim"],
        max_seq_length=cfg["model"]["max_seq_len"],
        dataset_text_field="text",
        seed=cfg["seed"],
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Completion-only masking: train loss only on the assistant JSON.
    collator = None
    try:
        from trl import DataCollatorForCompletionOnlyLM

        response_template = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        print("Using completion-only loss masking.")
    except Exception as exc:  # pragma: no cover - version fallback
        print(f"Completion-only collator unavailable ({exc}); training on full text.")

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(tcfg["output_dir"])
    tokenizer.save_pretrained(tcfg["output_dir"])
    print(f"Saved adapter -> {tcfg['output_dir']}")


if __name__ == "__main__":
    main()
