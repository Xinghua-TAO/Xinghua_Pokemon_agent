"""Sum up the A/B test of code-ranked team building.

    python -m tools.ab_report                  # logs/ab/teambuild.jsonl
    python -m tools.ab_report path/to/file.jsonl
    python -m tools.ab_report --since 2026-11-25T12:30   # local time (or an ISO time with offset)

Each seat of a self-play game appends one row (agent/pokemon/agent.py,
POKEMON_AB_TEAMBUILD=1): its arm ("ranked" or "plain") and its result. The
two seats of a game have opposite arms, so each finished game is one trial:
did the "ranked" seat win? Draws, games without a result and games where
both seats report the same arm or inconsistent results are listed and left
out.

Prints the ranked arm's win rate with a Wilson 95% interval and an exact
two-sided binomial test against 50%, overall and split by which arm drafted
first (to check that the draft order didn't drive the result).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "logs" / "ab" / "teambuild.jsonl"
FLIP = {"win": "loss", "loss": "win", "draw": "draw"}


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def games(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """One entry per game, from the ranked seat's point of view."""
    by_session = defaultdict(list)
    for row in rows:
        by_session[row.get("session_id")].append(row)
    out, problems = [], []
    for session, seats in by_session.items():
        seats = list({r.get("seat"): r for r in seats}.values())  # one row per seat
        arms = [r.get("ab_arm") for r in seats]
        if len(seats) == 2 and arms[0] == arms[1]:
            problems.append(f"{session}: both seats were '{arms[0]}'")
            continue
        ranked = next((r for r in seats if r.get("ab_arm") == "ranked"), None)
        plain = next((r for r in seats if r.get("ab_arm") == "plain"), None)
        result = ranked.get("result") if ranked else FLIP.get(plain.get("result")) if plain else None
        if ranked and plain and FLIP.get(plain.get("result")) != ranked.get("result"):
            problems.append(f"{session}: seats disagree ({ranked.get('result')} vs {plain.get('result')})")
            continue
        if result not in ("win", "loss", "draw"):
            problems.append(f"{session}: no result ({result}, {(ranked or plain).get('termination')})")
            continue
        drafted_first = ranked.get("drafted_first") if ranked else (
            None if plain.get("drafted_first") is None else not plain.get("drafted_first"))
        fallbacks = sum((r.get("fallbacks") or 0) for r in seats)
        out.append({"session_id": session, "result": result, "ranked_drafted_first": drafted_first,
                    "fallbacks": fallbacks, "seats_reported": len(seats)})
    return out, problems


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_p(wins: int, n: int) -> float:
    """Exact two-sided test of p = 0.5: total probability of outcomes no more likely than the observed one."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, k) / 2 ** n for k in range(n + 1)]
    observed = probs[wins]
    return min(1.0, sum(q for q in probs if q <= observed * (1 + 1e-9)))


def summary(label: str, entries: list[dict]) -> str:
    wins = sum(e["result"] == "win" for e in entries)
    losses = sum(e["result"] == "loss" for e in entries)
    draws = len(entries) - wins - losses
    n = wins + losses
    if n == 0:
        return f"{label}: no decided games"
    lo, hi = wilson(wins, n)
    return (f"{label}: ranked won {wins} of {n} ({wins / n:.0%}; 95% CI {lo:.0%}-{hi:.0%}; "
            f"p = {binomial_p(wins, n):.3f} vs 50%)" + (f", {draws} draws left out" if draws else ""))


def since_filter(rows: list[dict], since: str | None) -> list[dict]:
    """Rows of games started at or after ``since`` (naive times are local time)."""
    if not since:
        return rows
    cutoff = datetime.fromisoformat(since)
    if cutoff.tzinfo is None:
        cutoff = cutoff.astimezone()
    keep = []
    for row in rows:
        try:
            if datetime.fromisoformat(row.get("started_at") or "") >= cutoff:
                keep.append(row)
        except ValueError:
            pass
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", nargs="?", default=str(DEFAULT))
    parser.add_argument("--since", help="only games started at/after this time, e.g. 2026-11-25T12:30 (local)")
    args = parser.parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        print(f"No A/B results yet ({path}). Run the agent with POKEMON_AB_TEAMBUILD=1 and play self-play matches.")
        return 1
    entries, problems = games(since_filter(load(path), args.since))
    print(f"{len(entries)} games with a result ({path})")
    print(summary("All games", entries))
    for first, label in ((True, "Ranked seat drafted first"), (False, "Plain seat drafted first")):
        part = [e for e in entries if e["ranked_drafted_first"] is first]
        if part:
            print(summary(f"  {label}", part))
    fallbacks = sum(e["fallbacks"] for e in entries)
    one_seat = sum(e["seats_reported"] == 1 for e in entries)
    print(f"Fallback decisions (both seats): {fallbacks}; games where only one seat reported: {one_seat}")
    if problems:
        print(f"Left out ({len(problems)}):")
        for p in problems:
            print("  " + p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
