"""FastAPI inference endpoint for WorkflowLM.

POST /generate  {"description": "..."}  ->  {"workflow_json": {...}|null, "validation_result": {...}, "raw": "..."}

The model (base + optional fine-tuned adapter) is loaded once at startup. The response always
includes the validation result from the SAME ``parse_and_validate`` used in training-data
generation and evaluation, so the API never returns an unchecked plan.

Env vars:
    WORKFLOWLM_CONFIG   path to config yaml (default configs/train_v1.yaml)
    WORKFLOWLM_ADAPTER  path to a trained adapter (optional; base model if unset)

Run:
    uvicorn workflowlm.serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from workflowlm.prompts import build_inference_prompt
from workflowlm.schema import extract_json_block, parse_and_validate

app = FastAPI(title="WorkflowLM", version="0.1.0")

_STATE: dict = {}


class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=3, description="Natural-language process description")


class GenerateResponse(BaseModel):
    workflow_json: dict | None
    validation_result: dict
    raw: str


def _load_config() -> dict:
    path = Path(os.environ.get("WORKFLOWLM_CONFIG", "configs/train_v1.yaml"))
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@app.on_event("startup")
def _startup() -> None:
    cfg = _load_config()
    adapter = os.environ.get("WORKFLOWLM_ADAPTER") or None
    # Reuse the eval loader so serving and eval share identical model construction.
    from workflowlm.eval.run_eval import build_model

    tokenizer, model = build_model(cfg, adapter)
    _STATE["cfg"] = cfg
    _STATE["tokenizer"] = tokenizer
    _STATE["model"] = model
    _STATE["adapter"] = adapter


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "adapter": _STATE.get("adapter"), "loaded": "model" in _STATE}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    import torch

    cfg = _STATE["cfg"]
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    gen_cfg = cfg["generation"]

    prompt = build_inference_prompt(req.description, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            do_sample=gen_cfg.get("do_sample", False),
            temperature=gen_cfg.get("temperature", 1.0) or 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    plan, result = parse_and_validate(extract_json_block(raw))
    return GenerateResponse(
        workflow_json=plan.model_dump(mode="json") if plan else None,
        validation_result=result.model_dump(mode="json"),
        raw=raw,
    )
