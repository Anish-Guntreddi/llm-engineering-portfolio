"""QLoRA fine-tuning for StratCoder (Qwen2.5-Coder-1.5B-Instruct).

Same reproducible, config-driven setup as WorkflowLM: 4-bit base, LoRA adapter, completion-only
loss masking, fixed seed. Renders examples through the shared chat template / system prompt.

    python -m stratcoder.train.train_qlora --config configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_cfg(p: Path) -> dict:
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [__import__("json").loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    from stratcoder.prompts import build_messages

    set_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    base, qcfg, tcfg = cfg["model"]["base"], cfg["qlora"], cfg["train"]

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def render(rows):
        texts = []
        for r in rows:
            msgs = build_messages(r["input"]) + [{"role": "assistant", "content": r["output"]}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        return Dataset.from_dict({"text": texts})

    train_ds = render(load_jsonl(Path(cfg["data"]["train"])))
    val_ds = render(load_jsonl(Path(cfg["data"]["val"])))

    bnb = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"], bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        torch_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]))
    model.config.use_cache = False

    lora = LoraConfig(r=qcfg["lora_r"], lora_alpha=qcfg["lora_alpha"], lora_dropout=qcfg["lora_dropout"],
                      target_modules=qcfg["target_modules"], bias="none", task_type="CAUSAL_LM")

    sft_args = SFTConfig(
        output_dir=tcfg["output_dir"], num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"], lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"], weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"], save_strategy=tcfg["save_strategy"],
        eval_strategy=tcfg["eval_strategy"], bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"], max_grad_norm=tcfg["max_grad_norm"],
        optim=tcfg["optim"], max_seq_length=cfg["model"]["max_seq_len"], dataset_text_field="text",
        seed=cfg["seed"], report_to="none", gradient_checkpointing_kwargs={"use_reentrant": False})

    collator = None
    try:
        from trl import DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tok)
        print("Using completion-only loss masking.")
    except Exception as exc:  # pragma: no cover
        print(f"Completion-only collator unavailable ({exc}); training on full text.")

    trainer = SFTTrainer(model=model, args=sft_args, train_dataset=train_ds, eval_dataset=val_ds,
                         peft_config=lora, data_collator=collator, processing_class=tok)
    trainer.train()
    trainer.save_model(tcfg["output_dir"])
    tok.save_pretrained(tcfg["output_dir"])
    print(f"Saved adapter -> {tcfg['output_dir']}")


if __name__ == "__main__":
    main()
