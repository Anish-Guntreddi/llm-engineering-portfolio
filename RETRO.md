# Sprint Retro — LLM Engineering Portfolio (Phases 1–4)

What was learned building four LLM projects back-to-back on one machine.

## What went well

- **A schema/validator as the single source of truth pays off repeatedly.** In WorkflowLM the
  Pydantic validator drove dataset generation, evaluation, *and* serving — so "valid" meant one
  thing everywhere and the eval numbers were trustworthy. AdaptBench then reused that exact scorer
  to keep its four-system comparison fair. Build the contract first.
- **Self-validating datasets.** Generating examples and rejecting any that fail the validator
  (StratCoder ran every gold code sample through the full sandbox before accepting it) means the
  fine-tune never learns from broken targets. 0 rejected in WorkflowLM, 0 in StratCoder.
- **The win isn't where you expect.** Base models already emit valid JSON / parseable Python; the
  fine-tune's value showed up in *consistency* — schema adherence, correct fields, no hallucinated
  systems (WorkflowLM), and matching an exact contract (StratCoder 0.00 → 0.93 unit-test pass).
  Report field-level metrics, not just surface validity.
- **Caching made a flaky run recoverable.** AdaptBench's on-disk generation cache turned a
  mid-run failure into a no-op resume, and let me prove determinism (cache-cleared rerun reproduced
  the numbers byte-identically).
- **Pluggable providers keep things runnable.** DocuQuery defaults to local embeddings + an
  extractive answerer, so the whole RAG + eval pipeline runs with zero API cost and becomes
  Gemini-powered the moment a key is present. Same idea let AdaptBench add Gemini systems optionally.

## What bit me (and the fix)

- **transformers 5.x / trl 1.x are not yet compatible** with this QLoRA setup on Windows — they
  failed to import `TrainingArguments`/`PreTrainedModel`. Pinning to transformers 4.46.3 / trl
  0.12.2 / peft 0.14.0 fixed it. This recurred in three venvs; pin it everywhere up front.
- **Windows-specific papercuts:** PowerShell wraps native stderr as errors (benign but noisy);
  `Tee-Object` writes UTF-16 (broke log greps); the HF cache symlink warning; a one-time HF cache
  migration segfault. Use plain file redirection for logs, decode UTF-16 when parsing.
- **Lazy embedding dim.** The local embedder's dimension is only known after the first `embed()`,
  which created `vector(0)` in pgvector and would have hit `/health` on a cold start. Warm up the
  provider before creating the store.
- **Pure-Python sandboxes have known escapes.** Closed the `str.format` dunder-walk vector; the
  honest boundary is the static import allow-list (dangerous code never runs) + subprocess +
  timeout, documented as such rather than oversold.

## What I'd do next

- Scale the WorkflowLM/StratCoder datasets to 1k–3k and re-measure (the harnesses already support it).
- Add an LLM-judge faithfulness axis to DocuQuery (and a reranking pass) when a Gemini key is set.
- Run AdaptBench across multiple base models/domains to test whether "RAG ≈ fine-tuning" holds
  beyond the 1.5B / structured-JSON regime.
- Containerize the StratCoder sandbox child for true isolation before any untrusted use.
