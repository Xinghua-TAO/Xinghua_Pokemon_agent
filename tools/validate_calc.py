"""Check agent/pokemon/calc.py against real damage from recorded battles.

Walks each recorded Showdown battle log (logs/raw/*.jsonl from
tools/recorder.py), tracks HP, stat boosts, screens, terrain, weather and
Booster Energy as the battle goes, and for every direct attack predicts a
damage range with our calculator. A prediction passes when the real damage
(percent of max HP, as the log reports it) falls inside the predicted range
(+-1 point for rounding). Hits with a critical, a missing set, or effects we
do not track are skipped and counted.

    python -m tools.validate_calc [logs/raw/*.jsonl ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

from agent.pokemon import dex
from agent.pokemon.calc import RUIN, SPREAD_TARGETS, Field, Side, build_mon, damage
from agent.pokemon.cards import CardBook, species_key

HP = re.compile(r"(\d+)/(\d+)")


def _ident(token: str) -> tuple[str, str] | None:
    m = re.match(r"(p\d)[ab]?: (.+)", token)
    return (m.group(1), m.group(2)) if m else None


def validate(path: Path, book: CardBook, verbose: bool) -> tuple[int, int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    decisions = [r for r in rows if r.get("kind") == "decision" and r.get("phase") == "moving"]
    if not decisions:
        return 0, 0, 0
    log = decisions[-1]["raw_state"]["observation"].get("protocol_log") or []
    species_of: dict[tuple[str, str], str] = {}
    hp_pct: dict[tuple[str, str], float] = {}
    boosts: dict[tuple[str, str], dict[str, int]] = {}
    paradox: dict[tuple[str, str], str] = {}
    items_gone: set[tuple[str, str]] = set()
    side_screens: dict[str, set[str]] = {"p1": set(), "p2": set()}
    terrain = weather = None
    active: dict[str, set[tuple[str, str]]] = {"p1": set(), "p2": set()}
    passed = failed = skipped = 0

    def mon_for(key):
        card = book.by_species(species_of.get(key))
        if not card:
            return None
        mon = build_mon({"species": species_of[key]}, card, exact_hp=False)
        if mon is None:
            return None
        mon.hp = round(hp_pct.get(key, 100) / 100 * mon.max_hp)
        mon.boosts = dict(boosts.get(key, {}))
        mon.active_boost = paradox.get(key)
        if key in items_gone:
            mon.item = ""
        return mon

    i = 0
    while i < len(log):
        parts = log[i].split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in ("switch", "drag") and len(parts) > 4:
            key = _ident(parts[2])
            species_of[key] = parts[3].split(",")[0]
            m = HP.match(parts[4])
            hp_pct[key] = 100.0 * int(m.group(1)) / int(m.group(2)) if m else 100.0
            boosts[key] = {}
            paradox.pop(key, None)
            active[key[0]] = {k for k in active[key[0]] if k[1] != key[1]} | {key}
        elif tag in ("-boost", "-unboost") and len(parts) > 4:
            key = _ident(parts[2])
            sign = 1 if tag == "-boost" else -1
            boosts.setdefault(key, {})[parts[3]] = boosts.get(key, {}).get(parts[3], 0) + sign * int(parts[4])
        elif tag == "-clearallboost":
            boosts = {k: {} for k in boosts}
        elif tag == "-start" and len(parts) > 3 and re.match(r"(quarkdrive|protosynthesis)(\w+)", parts[3]):
            paradox[_ident(parts[2])] = re.match(r"(?:quarkdrive|protosynthesis)(\w+)", parts[3]).group(1)
        elif tag == "-enditem" and len(parts) > 2:
            items_gone.add(_ident(parts[2]))
        elif tag == "-sidestart" and len(parts) > 3:
            side_screens[parts[2][:2]].add(species_key(parts[3].replace("move: ", "")))
        elif tag == "-sideend" and len(parts) > 3:
            side_screens[parts[2][:2]].discard(species_key(parts[3].replace("move: ", "")))
        elif tag == "-fieldstart" and len(parts) > 2:
            terrain = species_key(parts[2].replace("move: ", "")).replace("terrain", "") or None
        elif tag == "-fieldend":
            terrain = None
        elif tag == "-weather" and len(parts) > 2:
            weather = {"sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "snow": "snow"}.get(
                species_key(parts[2]))
        elif tag in ("-damage", "-heal", "-sethp") and len(parts) > 3:  # incl. Life Orb, Leech Seed, terrain
            key = _ident(parts[2])
            m = HP.match(parts[3])
            if m:
                hp_pct[key] = 100.0 * int(m.group(1)) / int(m.group(2))
            elif parts[3].startswith("0"):
                hp_pct[key] = 0
        elif tag == "move" and len(parts) > 3:
            attacker_key = _ident(parts[2])
            move_id = species_key(parts[3])
            move = dex.move(move_id) or {}
            # this move's block: everything until the next action line
            j = i + 1
            while j < len(log) and not log[j].startswith(("|move|", "|switch|", "|turn|", "|upkeep", "|cant|")):
                j += 1
            block = [line.split("|") for line in log[i + 1:j]]
            crit = any(len(p) > 1 and p[1] == "-crit" for p in block)
            direct = [p for p in block if len(p) > 3 and p[1] == "-damage" and "[from]" not in "|".join(p[4:])]
            n_hit = len({_ident(p[2]) for p in direct})
            gradable = move.get("cat") in ("Physical", "Special") and not move.get("multihit")
            spread = move.get("target") in SPREAD_TARGETS and n_hit > 1
            ruin = set()
            for side_keys in active.values():
                for k in side_keys:
                    card = book.by_species(species_of.get(k))
                    if card and species_key(card.get("ability")) in RUIN and hp_pct.get(k, 0) > 0:
                        ruin.add(species_key(card["ability"]))
            graded: set = set()
            # process the block in order: the hit itself, then heals / recoil / stat changes
            for p in block:
                if len(p) < 3:
                    continue
                key = _ident(p[2]) if p[1].startswith("-") else None
                if p[1] == "-damage" and len(p) > 3 and "[from]" not in "|".join(p[4:]) and key not in graded:
                    graded.add(key)
                    m = HP.match(p[3])
                    after = 100.0 * int(m.group(1)) / int(m.group(2)) if m else 0.0
                    before = hp_pct.get(key, 100.0)
                    attacker, defender = mon_for(attacker_key), mon_for(key)
                    if not gradable or crit or attacker is None or defender is None or after == 0:
                        skipped += 1
                    else:
                        screens = side_screens.get(key[0], set())
                        side = Side(reflect="reflect" in screens, light_screen="lightscreen" in screens,
                                    aurora_veil="auroraveil" in screens)
                        result = damage(attacker, defender, move_id, Field(weather=weather, terrain=terrain, ruin=ruin),
                                        side, spread=spread)
                        dealt = before - after
                        lo, hi = 100 * result.lo / defender.max_hp, 100 * result.hi / defender.max_hp
                        ok = lo - 1.5 <= dealt <= hi + 1.5
                        passed += ok
                        failed += not ok
                        if verbose or not ok:
                            print(f"  {'OK  ' if ok else 'FAIL'} {attacker.name} {parts[3]} -> {defender.name}: "
                                  f"real {dealt:.1f}%, predicted {lo:.1f}-{hi:.1f}%{' (spread)' if spread else ''}")
                    hp_pct[key] = after
                elif p[1] in ("-damage", "-heal", "-sethp") and len(p) > 3:
                    m = HP.match(p[3])
                    hp_pct[key] = 100.0 * int(m.group(1)) / int(m.group(2)) if m else 0.0
                elif p[1] in ("-boost", "-unboost") and len(p) > 4:
                    sign = 1 if p[1] == "-boost" else -1
                    boosts.setdefault(key, {})[p[3]] = boosts.get(key, {}).get(p[3], 0) + sign * int(p[4])
                elif p[1] == "-enditem":
                    items_gone.add(key)
            i = j
            continue
        i += 1
    return passed, failed, skipped


def main() -> None:
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    paths = [Path(p) for p in (files or sorted(glob.glob("logs/raw/*.jsonl")))]
    book = CardBook()
    total = [0, 0, 0]
    for path in paths:
        print(path.name[:70])
        result = validate(path, book, verbose="-v" in sys.argv)
        total = [a + b for a, b in zip(total, result)]
    print(f"\npassed {total[0]}, failed {total[1]}, skipped {total[2]} (crits, multi-hit, KOs, unknown sets)")


if __name__ == "__main__":
    main()
