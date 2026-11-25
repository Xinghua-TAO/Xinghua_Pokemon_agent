"""Draft cards: the fixed Pokémon sets the game is played with.

Each draft card is a complete set (species, item, ability, nature, EVs, four
moves). The server shows these sets only during the draft; from Team Preview
on, the opponent's item and moves are hidden. So the agent keeps every card it
sees: in memory for the current game, and in a JSON catalog on disk so that a
card taken before our first look at the pool (when the opponent drafts first),
or a game resumed after a restart, can still be looked up.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "card_catalog.json"

_STAT_NAMES = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def species_key(name: str | None) -> str:
    """Showdown-style id: 'Landorus-Therian' -> 'landorustherian'."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, dict]:
    """card_id -> card. An unreadable or missing file is an empty catalog."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def merge_into_catalog(cards: list[dict], path: Path = CATALOG_PATH) -> int:
    """Add cards not yet in the catalog; return how many were new.

    Never raises: a catalog problem must not cost a draft pick. The write is
    atomic (temp file + rename) because several games may run at once.
    """
    try:
        catalog = load_catalog(path)
        new = {
            card["card_id"]: card
            for card in cards
            if isinstance(card, dict) and card.get("card_id") and card["card_id"] not in catalog
        }
        if not new:
            return 0
        catalog.update(new)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".catalog-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(catalog.items())), f, indent=1, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
        return len(new)
    except Exception:  # noqa: BLE001 - best effort cache
        return 0


def format_evs(evs: dict | None) -> str:
    parts = [f"{value} {_STAT_NAMES.get(stat, stat)}" for stat, value in (evs or {}).items() if value]
    return " / ".join(parts) or "no EVs"


def format_card(card: dict) -> str:
    """One line: 'Incineroar @ Safety Goggles | Intimidate | Careful 252 HP / ... | Fake Out, ...'."""
    moves = ", ".join(card.get("moves") or [])
    return (
        f"{card.get('species')} @ {card.get('item')} | {card.get('ability')} | "
        f"{card.get('nature')} {format_evs(card.get('evs'))} | {moves}"
    )


class CardBook:
    """This game's known cards, backed by the on-disk catalog."""

    def __init__(self, catalog_path: Path = CATALOG_PATH) -> None:
        self._catalog_path = catalog_path
        self.pool: dict[str, dict] = {}  # card_id -> card, seen in this game's draft
        self._catalog = load_catalog(catalog_path)

    def remember(self, cards: list[dict]) -> None:
        fresh = [c for c in cards if isinstance(c, dict) and c.get("card_id") and c["card_id"] not in self.pool]
        for card in fresh:
            self.pool[card["card_id"]] = card
        if any(c["card_id"] not in self._catalog for c in fresh):
            merge_into_catalog(fresh, self._catalog_path)
            self._catalog = load_catalog(self._catalog_path)

    def by_id(self, card_id: str) -> dict | None:
        return self.pool.get(card_id) or self._catalog.get(card_id)

    def by_species(self, species: str | None) -> dict | None:
        """This game's card for a species (pool first, then the catalog)."""
        key = species_key(species)
        if not key:
            return None
        for source in (self.pool, self._catalog):
            for card in source.values():
                if species_key(card.get("species")) == key:
                    return card
        return None
