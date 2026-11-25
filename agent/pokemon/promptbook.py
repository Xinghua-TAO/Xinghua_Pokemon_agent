"""Load the LLM instructions (system prompts) from versioned files.

agent/pokemon/prompts/<version>/{common,draft,preview,battle,reflect}.md;
"{COMMON}" in a prompt is replaced by that version's common.md (the format
rules). Keeping prompts as files makes them easy to read, quote in the
course report, and compare (tools/eval_prompts.py runs versions side by side).

POKEMON_PROMPT_VERSION picks the version (default: DEFAULT_VERSION).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_VERSION = "v2"
NAMES = ("draft", "preview", "battle", "reflect")


def current_version() -> str:
    return os.environ.get("POKEMON_PROMPT_VERSION") or DEFAULT_VERSION


@lru_cache(maxsize=None)
def load(version: str | None = None) -> dict[str, str]:
    version = version or current_version()
    folder = PROMPT_DIR / version
    if not folder.is_dir():
        raise ValueError(f"unknown prompt version {version!r}; have {sorted(p.name for p in PROMPT_DIR.iterdir())}")
    common = (folder / "common.md").read_text(encoding="utf-8").strip()
    return {name: (folder / f"{name}.md").read_text(encoding="utf-8").strip().replace("{COMMON}", common)
            for name in NAMES}
