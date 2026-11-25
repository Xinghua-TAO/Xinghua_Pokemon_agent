"""Memory of opponents across games (e.g. a best-of-3 bracket series).

Facts are computed by code from what the server shows us: the opponent's
draft picks in order, which 4 they brought and which 2 led, every move they
used per turn, how often they Protect, switch, or double-target. After a game
the LLM may add a few short notes; those are kept apart and labelled as
unverified, because a model's own reflection can be confidently wrong.

One JSON file per opponent display name under memory/opponents/ (git-ignored:
it is tournament intel). Writes are atomic because games run in parallel.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .calc import PROTECT_MOVES
from .cards import species_key

MEMORY_DIR = Path(__file__).resolve().parents[2] / "memory" / "opponents"
MAX_GAMES = 30

_SWITCH = re.compile(r"^\|(?:switch|drag)\|(p\d)([ab]): ([^|]+)\|([^|,]+)")
_MOVE = re.compile(r"^\|move\|(p\d)([ab]): ([^|]+)\|([^|]+)\|?([^|]*)")
_TURN = re.compile(r"^\|turn\|(\d+)")


def file_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9_-]+", "-", (name or "").lower()).strip("-")
    return key[:80] or "unknown"


def parse_battle(protocol_log: list[str], my_side: str | None) -> dict:
    """What the opponent did, from the Showdown log (their side = not ours)."""
    their = {"p1": "p2", "p2": "p1"}.get(my_side or "")
    turn = 0
    leads: list[str] = []
    seen: list[str] = []
    our_seen: list[str] = []
    moves_by_turn: dict[str, list[list[str]]] = {}
    switches = protects = actions = double_targets = 0
    for line in protocol_log or []:
        if not isinstance(line, str):
            continue
        if m := _TURN.match(line):
            turn = int(m.group(1))
            continue
        if m := _SWITCH.match(line):
            side, species = m.group(1), m.group(4).strip()
            if side == their:
                if species not in seen:
                    seen.append(species)
                if turn == 0 and len(leads) < 2:
                    leads.append(species)
                elif turn > 0:
                    switches += 1
            elif side == my_side and species not in our_seen:
                our_seen.append(species)
            continue
        if (m := _MOVE.match(line)) and m.group(1) == their:
            move, target = m.group(4), m.group(5)
            moves_by_turn.setdefault(str(turn), []).append([m.group(3), move, target])
            actions += 1
            if species_key(move) in PROTECT_MOVES:
                protects += 1
    for turn_moves in moves_by_turn.values():
        targets = [t for _, move, t in turn_moves if t and t.startswith(my_side or "?") and species_key(move) not in PROTECT_MOVES]
        if len(targets) >= 2 and len(set(targets)) == 1:
            double_targets += 1
    return {
        "leads": leads, "brought": seen, "our_seen": our_seen, "moves_by_turn": moves_by_turn,
        "turns": turn, "switches": switches, "protects": protects, "actions": actions,
        "double_target_turns": double_targets,
    }


class OpponentMemory:
    def __init__(self, name: str, directory: Path | None = None) -> None:
        self.name = name
        self.path = (directory or MEMORY_DIR) / f"{file_key(name)}.json"

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("games"), list):
                return data
        except (OSError, ValueError):
            pass
        return {"name": self.name, "games": []}

    def save_game(self, record: dict) -> None:
        """Insert or update this game's record (by session id). Never raises."""
        try:
            profile = self.load()
            games = [g for g in profile["games"] if g.get("session_id") != record.get("session_id")]
            games.append(record)
            profile["games"] = games[-MAX_GAMES:]
            profile["name"] = self.name
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".mem-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 - memory must never cost a move
            pass

    def past_games(self, exclude_session: str | None = None) -> list[dict]:
        return [g for g in self.load()["games"] if g.get("session_id") != exclude_session]


def history_lines(name: str, games: list[dict], *, brief: bool = False) -> list[str]:
    """A compact summary of past games against this opponent for the LLM."""
    if not games:
        return [f"OPPONENT HISTORY vs {name}: no previous games."]
    results = Counter(g.get("result") or "unfinished" for g in games)
    record = ", ".join(f"{n} {r}" for r, n in results.items())
    lines = [f"OPPONENT HISTORY vs {name} ({len(games)} previous game(s): {record}; facts computed from game logs):"]

    first_picks = Counter(g["their_picks"][0] for g in games if g.get("their_picks"))
    picked = Counter(p for g in games for p in g.get("their_picks") or [])
    leads = Counter(" + ".join(sorted(g["battle"]["leads"])) for g in games if (g.get("battle") or {}).get("leads"))
    turn1 = Counter()
    for g in games:
        for _, move, _ in ((g.get("battle") or {}).get("moves_by_turn") or {}).get("1", []):
            turn1[move] += 1
    actions = sum((g.get("battle") or {}).get("actions", 0) for g in games)
    protects = sum((g.get("battle") or {}).get("protects", 0) for g in games)
    switches = sum((g.get("battle") or {}).get("switches", 0) for g in games)
    doubles = sum((g.get("battle") or {}).get("double_target_turns", 0) for g in games)
    turns = sum((g.get("battle") or {}).get("turns", 0) for g in games)

    if not brief and picked:
        lines.append(f"  - Draft: first picks {dict(first_picks.most_common(3))}; most picked {dict(picked.most_common(6))}")
    if leads:
        lines.append(f"  - Leads: {dict(leads.most_common(3))}")
    if turn1:
        lines.append(f"  - Turn-1 moves: {dict(turn1.most_common(5))}")
    if actions:
        lines.append(f"  - Protect on {round(100 * protects / actions)}% of their moves; {switches} switches and "
                     f"{doubles} double-target turns over {turns} turns")
    if not brief:
        for g in games[-3:]:
            team = ", ".join((g.get("battle") or {}).get("brought") or g.get("their_picks") or [])
            lines.append(f"  - Game {g.get('started_at', '')[:16]} ({g.get('result') or 'unfinished'}): they used {team}")
    notes = [n for g in games[-3:] for n in g.get("notes") or []]
    if notes:
        lines.append("  - Notes written after past games (LLM, unverified):")
        lines += [f"      * {n}" for n in notes[-(3 if brief else 6):]]
    return lines


def this_game_lines(battle: dict) -> list[str]:
    """What the opponent has done so far in the current battle."""
    if not battle or not battle.get("actions"):
        return []
    recent = []
    for turn in sorted(battle["moves_by_turn"], key=int)[-2:]:
        moves = ", ".join(f"{mon} {move}" for mon, move, _ in battle["moves_by_turn"][turn])
        recent.append(f"turn {turn}: {moves}")
    return [f"OPPONENT THIS GAME: brought so far {', '.join(battle['brought'])}; {battle['protects']} Protects, "
            f"{battle['switches']} switches, {battle['double_target_turns']} double-target turns. Recent: "
            + "; ".join(recent)]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
