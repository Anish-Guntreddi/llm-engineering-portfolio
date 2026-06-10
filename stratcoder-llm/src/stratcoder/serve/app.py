"""FastAPI endpoint for StratCoder.

POST /generate {"description": "..."} ->
    {"code": "...", "validation_status": "valid"|"invalid", "warnings": [...], "report": {...}}

Every returned strategy has been run through the full validation pipeline (static checks + the
sandboxed unit tests). Code that fails validation is still returned (with status "invalid" and
the reasons) so callers can see what the model produced and why it was rejected — but it has
already been executed only inside the isolated sandbox, never in this process.

Env: STRATCODER_CONFIG (default configs/train_v1.yaml), STRATCODER_ADAPTER (optional).
    uvicorn stratcoder.serve.app:app --port 8001
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

from stratcoder.eval.metrics import extract_code
from stratcoder.prompts import build_inference_prompt
from stratcoder.validation.pipeline import run_validation

app = FastAPI(title="StratCoder-LLM", version="0.1.0")
_STATE: dict = {}


class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=3)


class GenerateResponse(BaseModel):
    code: str
    validation_status: str
    warnings: list[dict]
    report: dict


@app.on_event("startup")
def _startup() -> None:
    path = Path(os.environ.get("STRATCODER_CONFIG", "configs/train_v1.yaml"))
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    adapter = os.environ.get("STRATCODER_ADAPTER") or None
    from stratcoder.eval.run_eval import build_model

    tok, model = build_model(cfg, adapter)
    _STATE.update(cfg=cfg, tok=tok, model=model, adapter=adapter)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "adapter": _STATE.get("adapter"), "loaded": "model" in _STATE}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    import torch

    cfg, tok, model = _STATE["cfg"], _STATE["tok"], _STATE["model"]
    g = cfg["generation"]
    prompt = build_inference_prompt(req.description, tok)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=g["max_new_tokens"],
                             do_sample=g.get("do_sample", False),
                             temperature=g.get("temperature", 1.0) or 1.0,
                             pad_token_id=tok.pad_token_id)
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    code = extract_code(raw)
    report = run_validation(code)  # static + sandboxed unit tests
    return GenerateResponse(
        code=code,
        validation_status="valid" if report.valid else "invalid",
        warnings=report.warnings + report.errors,
        report=report.as_dict(),
    )
