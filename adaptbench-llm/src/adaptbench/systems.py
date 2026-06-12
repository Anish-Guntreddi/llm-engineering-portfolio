"""The four benchmark systems behind one interface, so the comparison is fair by construction.

Every System.generate(description) takes the SAME test description. RAG systems additionally
prepend retrieved few-shot (input->output) examples drawn ONLY from the train split. All four use
the same base model and greedy decoding.
"""

from __future__ import annotations

from workflowlm.prompts import SYSTEM_PROMPT, build_messages

MAX_NEW_TOKENS = 768


class GenModel:
    """Wraps a tokenizer + causal LM (optionally with a LoRA adapter). Greedy decoding."""

    def __init__(self, base: str, adapter: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        self.model = model
        self._torch = torch

    def generate(self, messages: list[dict]) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _fewshot_messages(description: str, shots: list[dict]) -> list[dict]:
    """system + (user=example input, assistant=example output)* + user=description."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": s["input"]})
        msgs.append({"role": "assistant", "content": s["output"]})
    msgs.append({"role": "user", "content": description})
    return msgs


class System:
    name = "base-class"

    def generate(self, description: str) -> str:
        raise NotImplementedError


class BaseSystem(System):
    name = "base"

    def __init__(self, model: GenModel):
        self.model = model

    def generate(self, description: str) -> str:
        return self.model.generate(build_messages(description))


class RAGSystem(System):
    name = "rag"

    def __init__(self, model: GenModel, retriever, k: int):
        self.model, self.retriever, self.k = model, retriever, k

    def generate(self, description: str) -> str:
        shots = self.retriever.retrieve(description, self.k)
        return self.model.generate(_fewshot_messages(description, shots))


class FineTunedSystem(System):
    name = "finetuned"

    def __init__(self, model: GenModel):
        self.model = model

    def generate(self, description: str) -> str:
        return self.model.generate(build_messages(description))


class FineTunedRAGSystem(System):
    name = "finetuned_rag"

    def __init__(self, model: GenModel, retriever, k: int):
        self.model, self.retriever, self.k = model, retriever, k

    def generate(self, description: str) -> str:
        shots = self.retriever.retrieve(description, self.k)
        return self.model.generate(_fewshot_messages(description, shots))


# --- Optional Gemini systems (only built when a key is present) ---
class GeminiSystem(System):
    name = "gemini"

    def __init__(self, api_key: str, model: str, retriever=None, k: int = 0):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._m = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT)
        self.retriever, self.k = retriever, k
        self.name = "gemini_rag" if retriever else "gemini"

    def generate(self, description: str) -> str:
        prefix = ""
        if self.retriever:
            shots = self.retriever.retrieve(description, self.k)
            prefix = "".join(f"Example input: {s['input']}\nExample output: {s['output']}\n\n" for s in shots)
        resp = self._m.generate_content(prefix + f"Input: {description}")
        return (resp.text or "").strip()
