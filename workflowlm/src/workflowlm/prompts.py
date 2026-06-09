"""Prompt construction shared by training, evaluation, and serving.

Keeping this in one place guarantees the model sees an identical system prompt and chat
format at train time, eval time, and inference time — a common silent source of
train/serve skew.
"""

from __future__ import annotations

import json

from workflowlm.schema import KNOWN_SYSTEMS, WorkflowCategory

_CATEGORIES = ", ".join(c.value for c in WorkflowCategory)
_SYSTEMS = ", ".join(KNOWN_SYSTEMS)

SYSTEM_PROMPT = f"""You are WorkflowLM, an assistant that converts a natural-language description \
of an enterprise business process into a single structured JSON workflow plan.

Respond with ONLY a JSON object (no prose, no markdown fences) matching this schema:

{{
  "workflow_name": "<short title, <= 80 chars>",
  "category": "<one of: {_CATEGORIES}>",
  "trigger": {{ "event": "<what starts the workflow>", "source_system": "<system or null>" }},
  "systems": ["<systems involved>"],
  "steps": [
    {{
      "step_id": 1,
      "action": "<imperative action>",
      "system": "<system that performs it; must be listed in systems[]>",
      "required_inputs": ["<input>", "..."],
      "condition": "<when this step applies, or null>",
      "fallback": "<what to do if it fails, or null>"
    }}
  ],
  "approval_required": true,
  "risk_level": "<low|medium|high>"
}}

Rules:
- Use systems only from this set when possible: {_SYSTEMS}. Use "Other" if none fit.
- Every step.system MUST also appear in the top-level systems[] list.
- step_id values are unique and contiguous starting at 1.
- If risk_level is "high", set approval_required to true.
- Output valid JSON and nothing else."""


def build_messages(description: str) -> list[dict[str, str]]:
    """Build the chat messages for a given NL process description."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description.strip()},
    ]


def build_training_text(description: str, target_json: dict, tokenizer) -> str:
    """Render a full supervised example (prompt + gold completion) via the chat template.

    Used by the SFT trainer. The assistant turn is the compact gold JSON.
    """
    messages = build_messages(description)
    messages.append({"role": "assistant", "content": json.dumps(target_json, ensure_ascii=False)})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_inference_prompt(description: str, tokenizer) -> str:
    """Render the prompt up to the assistant turn, for generation."""
    messages = build_messages(description)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
