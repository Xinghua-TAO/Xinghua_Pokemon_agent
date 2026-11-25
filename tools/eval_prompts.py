"""Compare prompt versions on real battle situations with clear right/wrong answers.

Each scenario is a game state recorded in one of our test matches
(tools/eval_states/*.json) plus a rule that marks a choice as a mistake
(e.g. spending both attacks on a target one attack already knocks out, or
using Protect when it has only a 33% chance to work). The agent decides
each scenario several times per prompt version with the real LLM; the score
is the share of decisions without the mistake. This turns "the new prompt
feels better" into a measurement.

    python -m tools.eval_prompts                       # v1 vs v2, 3 runs each
    python -m tools.eval_prompts --versions v2 --runs 5
    python -m tools.eval_prompts --extract             # (dev) rebuild eval_states from logs/raw
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from altruagent import DecisionContext, GameState
from agent.pokemon.agent import PokemonAgent
from agent.pokemon.cards import CardBook

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(__file__).resolve().parent / "eval_states"


def _moves(action: dict) -> list[dict]:
    return [action.get("slot_0") or {}, action.get("slot_1") or {}]


def _attacks_on(action: dict, target: int, spread_moves: tuple[str, ...] = ()) -> int:
    """How many slots attack ``target`` (single-target, or a listed spread move)."""
    count = 0
    for m in _moves(action):
        if m.get("type") != "move" or m.get("move_id") in ("protect", "fakeout"):
            continue
        if m.get("target") == target or m.get("move_id") in spread_moves:
            count += 1
    return count


def check_thundurus(action):
    """Thundurus at 8%: Chien-Pao KOs it first, Grimmsnarl does something else.
    Also correct (the model found it; our first label missed it): double-target
    Ting-Lu for a sure KO, since Thundurus's Life Orb recoil (~10%) KOs it if
    it attacks."""
    s0, s1 = _moves(action)
    if _attacks_on(action, 1) >= 2:
        return False, "both slots into an 8% Thundurus (overkill)"
    if (s0.get("move_id"), s0.get("target"), s1.get("move_id"), s1.get("target")) == ("spiritbreak", 2, "iciclecrash", 2):
        return True, "sure double-target KO on Ting-Lu; Life Orb recoil finishes Thundurus"
    if _attacks_on(action, 1) == 0:
        return False, "nobody removes the 8% Thundurus and no sure KO elsewhere"
    return True, "one KO on Thundurus, other slot elsewhere"


def check_porygon2(action):  # Porygon2 at 4%: any hit KOs it; the other slot should hit Cresselia
    hits = _attacks_on(action, 1, spread_moves=("snarl",))
    if hits >= 2:
        return False, "both slots into a 4% Porygon2 (overkill)"
    if hits == 0:
        return False, "Porygon2 left alive"
    return True, "one KO on Porygon2, other slot elsewhere"


def check_basculegion_47(action):  # Knock Off or Sucker Punch alone KOs Basculegion before it moves
    s0, s1 = _moves(action)
    single = (s0.get("move_id"), s0.get("target")) == ("knockoff", 2) or (s1.get("move_id"), s1.get("target")) == ("suckerpunch", 2)
    double = (s0.get("move_id"), s0.get("target")) == ("flareblitz", 2) and (s1.get("move_id"), s1.get("target")) == ("iciclecrash", 2)
    if not (single or double):
        return False, "no KO on Basculegion although one move does it"
    if single and _attacks_on(action, 2) >= 2:
        return False, "overkill on Basculegion"
    ideal = (s0.get("move_id"), s0.get("target")) == ("fakeout", 1) and (s1.get("move_id"), s1.get("target")) == ("suckerpunch", 2)
    return True, "ideal: Fake Out Iron Hands + Sucker Punch KO" if ideal else "KO on Basculegion"


def check_basculegion_14(action):  # Basculegion at 14%: one KO on it, the other slot hits Iron Hands
    hits = _attacks_on(action, 2)
    if hits >= 2:
        return False, "both slots into a 14% Basculegion (overkill)"
    if hits == 0:
        return False, "Basculegion left alive"
    return True, "one KO on Basculegion, other slot elsewhere"


def no_repeat_protect(slot: str):
    def check(action):
        if (action.get(slot) or {}).get("move_id") == "protect":
            return False, f"{slot} used Protect with only a 33% chance to work"
        return True, "no low-odds Protect"
    return check


def no_tailwind(slot: str):
    def check(action):  # Tailwind is already up: using it again fails and wastes the turn
        if (action.get(slot) or {}).get("move_id") == "tailwind":
            return False, f"{slot} used Tailwind while it was already active (fails)"
        return True, "no wasted Tailwind"
    return check


SCENARIOS = [
    # name, session id prefix, seat, decision index in the recorded file, check
    ("thundurus_8pct", "59be36f2", 0, 9, check_thundurus),
    ("porygon2_4pct", "426dadc1", 0, 12, check_porygon2),
    ("basculegion_47pct", "17003b77", 0, 15, check_basculegion_47),
    ("basculegion_14pct", "17003b77", 0, 16, check_basculegion_14),
    ("protect_streak_cresselia", "426dadc1", 1, 18, no_repeat_protect("slot_1")),
    ("protect_streak_cresselia_fm", "17003b77", 0, 9, no_repeat_protect("slot_0")),
    ("protect_streak_thundurus", "6e347545", 0, 8, no_repeat_protect("slot_0")),
    # Live A/B game, 2026-11-25: the view mislabeled Tailwind's start turn as "turns left",
    # so the model "refreshed" Tailwind (it fails) and Tornadus fainted without acting.
    ("tailwind_already_up", "cf324219", 1, 8, no_tailwind("slot_1")),
]


def extract() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    for name, session, seat, index, _ in SCENARIOS:
        (path,) = glob.glob(str(ROOT / "logs" / "raw" / f"*{session}*_seat{seat}.jsonl"))
        decisions = [json.loads(line) for line in open(path, encoding="utf-8")]
        decisions = [r for r in decisions if r.get("kind") == "decision"]
        (STATE_DIR / f"{name}.json").write_text(json.dumps({"seat": seat, "raw_state": decisions[index]["raw_state"]}),
                                                encoding="utf-8")
        print("wrote", name)


class _Capture:
    def __init__(self):
        self.record = None

    def write(self, context, record):
        self.record = record


def decide(name: str, version: str) -> dict:
    data = json.loads((STATE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    raw = data["raw_state"]
    capture = _Capture()
    agent = PokemonAgent(None, log=capture, book=CardBook(), verbose=False, use_clock=False,
                         memory_dir=Path(tempfile.mkdtemp()), reflect=False, prompt_version=version)
    context = DecisionContext(session_id=raw.get("session_id", "eval"), tournament_id=None,
                              game_type=raw.get("game_type"), agent_id=(raw.get("current_actor") or {}).get("agent_id", ""),
                              seat_position=data["seat"])
    started = time.monotonic()
    agent.choose_action(GameState.from_mcp_state(raw), context)
    record = capture.record or {}
    answer = next((a.get("answer") for a in reversed(record.get("attempts", [])) if a.get("answer")), {}) or {}
    return {"action": record.get("action") or {}, "fallback": record.get("fallback"),
            "seconds": round(time.monotonic() - started, 1), "analysis": answer.get("analysis", "")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versions", default="v1,v2")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", default="", help="comma-separated scenario names")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--show", action="store_true", help="print each decision's analysis")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.extract:
        extract()
        return
    versions = args.versions.split(",")
    scenarios = [s for s in SCENARIOS if not args.only or s[0] in args.only.split(",")]
    jobs = [(s, v, r) for s in scenarios for v in versions for r in range(args.runs)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: (job, decide(job[0][0], job[1])), jobs))

    score = {v: [0, 0] for v in versions}
    print(f"{'scenario':30s} " + " ".join(f"{v:>12s}" for v in versions))
    for name, *_rest, check in scenarios:
        cells = []
        for version in versions:
            outs = [res for (s, v, _), res in results if s[0] == name and v == version]
            passes = [check(o["action"])[0] and not o["fallback"] for o in outs]
            score[version][0] += sum(passes)
            score[version][1] += len(passes)
            cells.append(f"{sum(passes)}/{len(passes)}")
            if args.show:
                for o in outs:
                    ok, why = check(o["action"])
                    print(f"   [{version}] {'PASS' if ok else 'FAIL'} {why} ({o['seconds']}s): {o['analysis'][:220]}")
        print(f"{name:30s} " + " ".join(f"{c:>12s}" for c in cells))
    print(f"{'TOTAL':30s} " + " ".join(f"{score[v][0]:>5d}/{score[v][1]:<6d}" for v in versions))


if __name__ == "__main__":
    main()
