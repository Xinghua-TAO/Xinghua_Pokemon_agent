"""Static Pokémon data and rules: Pokédex, moves, type chart, natures, stats.

Data files come from Pokémon Showdown (the platform's battle engine) via
tools/build_dex.py.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from .cards import species_key

DATA_DIR = Path(__file__).resolve().parent / "data"
LEVEL = 50
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


@lru_cache(maxsize=1)
def pokedex() -> dict[str, dict]:
    return json.loads((DATA_DIR / "pokedex.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def moves() -> dict[str, dict]:
    return json.loads((DATA_DIR / "moves.json").read_text(encoding="utf-8"))


def species(name: str | None) -> dict | None:
    return pokedex().get(species_key(name))


def move(name: str | None) -> dict | None:
    return moves().get(species_key(name))


# Attacking type -> {defending type: multiplier}; anything missing is 1x.
TYPE_CHART: dict[str, dict[str, float]] = {
    "NORMAL": {"ROCK": 0.5, "GHOST": 0, "STEEL": 0.5},
    "FIRE": {"FIRE": 0.5, "WATER": 0.5, "GRASS": 2, "ICE": 2, "BUG": 2, "ROCK": 0.5, "DRAGON": 0.5, "STEEL": 2},
    "WATER": {"FIRE": 2, "WATER": 0.5, "GRASS": 0.5, "GROUND": 2, "ROCK": 2, "DRAGON": 0.5},
    "ELECTRIC": {"WATER": 2, "ELECTRIC": 0.5, "GRASS": 0.5, "GROUND": 0, "FLYING": 2, "DRAGON": 0.5},
    "GRASS": {"FIRE": 0.5, "WATER": 2, "GRASS": 0.5, "POISON": 0.5, "GROUND": 2, "FLYING": 0.5, "BUG": 0.5,
              "ROCK": 2, "DRAGON": 0.5, "STEEL": 0.5},
    "ICE": {"FIRE": 0.5, "WATER": 0.5, "GRASS": 2, "ICE": 0.5, "GROUND": 2, "FLYING": 2, "DRAGON": 2, "STEEL": 0.5},
    "FIGHTING": {"NORMAL": 2, "ICE": 2, "POISON": 0.5, "FLYING": 0.5, "PSYCHIC": 0.5, "BUG": 0.5, "ROCK": 2,
                 "GHOST": 0, "DARK": 2, "STEEL": 2, "FAIRY": 0.5},
    "POISON": {"GRASS": 2, "POISON": 0.5, "GROUND": 0.5, "ROCK": 0.5, "GHOST": 0.5, "STEEL": 0, "FAIRY": 2},
    "GROUND": {"FIRE": 2, "ELECTRIC": 2, "GRASS": 0.5, "POISON": 2, "FLYING": 0, "BUG": 0.5, "ROCK": 2, "STEEL": 2},
    "FLYING": {"ELECTRIC": 0.5, "GRASS": 2, "FIGHTING": 2, "BUG": 2, "ROCK": 0.5, "STEEL": 0.5},
    "PSYCHIC": {"FIGHTING": 2, "POISON": 2, "PSYCHIC": 0.5, "DARK": 0, "STEEL": 0.5},
    "BUG": {"FIRE": 0.5, "GRASS": 2, "FIGHTING": 0.5, "POISON": 0.5, "FLYING": 0.5, "PSYCHIC": 2, "GHOST": 0.5,
            "DARK": 2, "STEEL": 0.5, "FAIRY": 0.5},
    "ROCK": {"FIRE": 2, "ICE": 2, "FIGHTING": 0.5, "GROUND": 0.5, "FLYING": 2, "BUG": 2, "STEEL": 0.5},
    "GHOST": {"NORMAL": 0, "PSYCHIC": 2, "GHOST": 2, "DARK": 0.5},
    "DRAGON": {"DRAGON": 2, "STEEL": 0.5, "FAIRY": 0},
    "DARK": {"FIGHTING": 0.5, "PSYCHIC": 2, "GHOST": 2, "DARK": 0.5, "FAIRY": 0.5},
    "STEEL": {"FIRE": 0.5, "WATER": 0.5, "ELECTRIC": 0.5, "ICE": 2, "ROCK": 2, "STEEL": 0.5, "FAIRY": 2},
    "FAIRY": {"FIRE": 0.5, "FIGHTING": 2, "POISON": 0.5, "DRAGON": 2, "DARK": 2, "STEEL": 0.5},
}


def effectiveness(move_type: str, defender_types: list[str], move_id: str = "") -> float:
    multiplier = 1.0
    for def_type in defender_types:
        def_type = def_type.upper()
        if move_id == "freezedry" and def_type == "WATER":
            multiplier *= 2
        else:
            multiplier *= TYPE_CHART.get(move_type.upper(), {}).get(def_type, 1.0)
    return multiplier


NATURES: dict[str, tuple[str | None, str | None]] = {
    "adamant": ("atk", "spa"), "bold": ("def", "atk"), "brave": ("atk", "spe"), "calm": ("spd", "atk"),
    "careful": ("spd", "spa"), "gentle": ("spd", "def"), "hasty": ("spe", "def"), "impish": ("def", "spa"),
    "jolly": ("spe", "spa"), "lax": ("def", "spd"), "lonely": ("atk", "def"), "mild": ("spa", "def"),
    "modest": ("spa", "atk"), "naive": ("spe", "spd"), "naughty": ("atk", "spd"), "quiet": ("spa", "spe"),
    "rash": ("spa", "spd"), "relaxed": ("def", "spe"), "sassy": ("spd", "spe"), "timid": ("spe", "atk"),
    "bashful": (None, None), "docile": (None, None), "hardy": (None, None), "quirky": (None, None),
    "serious": (None, None),
}


def calc_stats(base: dict, evs: dict | None = None, nature: str | None = None, ivs: dict | None = None,
               level: int = LEVEL) -> dict[str, int]:
    """Actual stats from base stats, EVs, IVs (default 31) and nature."""
    evs, ivs = evs or {}, ivs or {}
    plus, minus = NATURES.get((nature or "").lower(), (None, None))
    out = {}
    for stat in STATS:
        core = (2 * base.get(stat, 0) + ivs.get(stat, 31) + evs.get(stat, 0) // 4) * level // 100
        if stat == "hp":
            out[stat] = core + level + 10
        else:
            value = core + 5
            if stat == plus:
                value = math.floor(value * 1.1)
            elif stat == minus:
                value = math.floor(value * 0.9)
            out[stat] = value
    return out


def stage_multiplier(stage: int) -> float:
    stage = max(-6, min(6, stage))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)
