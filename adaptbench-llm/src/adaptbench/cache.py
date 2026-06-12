"""On-disk generation cache: (system, input) -> generated text.

Makes the benchmark reproducible and cheap to regenerate, and lets a run resume after a crash.
Greedy decoding means a cache hit is identical to a fresh generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class GenerationCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _key(system: str, text: str) -> str:
        return hashlib.sha256(f"{system}{text}".encode()).hexdigest()

    def get(self, system: str, text: str) -> str | None:
        return self._data.get(self._key(system, text))

    def put(self, system: str, text: str, value: str) -> None:
        self._data[self._key(system, text)] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")
