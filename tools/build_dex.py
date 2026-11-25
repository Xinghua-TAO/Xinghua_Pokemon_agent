"""Build agent/pokemon/data/{pokedex,moves}.json from Pokémon Showdown's data.

Showdown is the battle engine the platform runs, so its data is the ground
truth for base stats, types, weights and move details. The files are trimmed
to the fields our damage calculator needs and committed, so the agent never
downloads anything during a game.

    python -m tools.build_dex                      # download from play.pokemonshowdown.com
    python -m tools.build_dex --from-dir <dir>     # use already-downloaded pokedex.json / moves.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

SOURCE = "https://play.pokemonshowdown.com/data/{name}.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "agent" / "pokemon" / "data"
KEEP_FLAGS = ("contact", "punch", "bite", "slicing", "sound", "wind", "pulse", "bullet")
SKIP_NONSTANDARD = {"CAP", "Custom", "LGPE"}


def _load(name: str, from_dir: Path | None) -> dict:
    if from_dir:
        return json.loads((from_dir / f"{name}.json").read_text(encoding="utf-8"))
    response = httpx.get(SOURCE.format(name=name), timeout=60, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def trim_pokedex(raw: dict) -> dict:
    out = {}
    for species_id, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("num", 0) <= 0:
            continue
        if entry.get("isNonstandard") in SKIP_NONSTANDARD:
            continue
        out[species_id] = {
            "name": entry.get("name"),
            "types": [t.upper() for t in entry.get("types", [])],
            "bs": entry.get("baseStats", {}),
            "abilities": list((entry.get("abilities") or {}).values()),
            "weight": entry.get("weightkg"),
        }
    return dict(sorted(out.items()))


def trim_moves(raw: dict) -> dict:
    out = {}
    for move_id, entry in raw.items():
        if not isinstance(entry, dict) or entry.get("num", 0) <= 0:
            continue
        accuracy = entry.get("accuracy")
        move = {
            "name": entry.get("name"),
            "type": (entry.get("type") or "???").upper(),
            "cat": entry.get("category"),
            "bp": entry.get("basePower", 0),
            "acc": 1.0 if accuracy is True else round((accuracy or 0) / 100, 2),
            "pri": entry.get("priority", 0),
            "target": entry.get("target"),
        }
        flags = [f for f in KEEP_FLAGS if (entry.get("flags") or {}).get(f)]
        if flags:
            move["flags"] = flags
        for key, short in (("multihit", "multihit"), ("willCrit", "crit"), ("overrideOffensiveStat", "off_stat"),
                           ("overrideOffensivePokemon", "off_mon"), ("overrideDefensiveStat", "def_stat"),
                           ("damage", "fixed")):
            if entry.get(key) not in (None, False):
                move[short] = entry[key]
        if entry.get("secondary") or entry.get("secondaries"):
            move["secondary"] = True
        if entry.get("recoil") or entry.get("mindBlownRecoil"):
            move["recoil"] = True
        out[move_id] = move
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", type=Path)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, trim in (("pokedex", trim_pokedex), ("moves", trim_moves)):
        data = trim(_load(name, args.from_dir))
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{path.relative_to(OUT_DIR.parent.parent.parent)}: {len(data)} entries, {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
