"""Replay recorded game states through our Pokémon agent, offline.

Takes a logs/raw/*.jsonl file written by tools/recorder.py and feeds every
recorded decision state, in order, to agent.pokemon.PokemonAgent. Nothing is
sent to the game server, so you can test changes on real situations in
seconds instead of playing live matches.

    python -m tools.replay logs/raw/<file>.jsonl                 # uses the LLM (OPENAI_API_KEY)
    python -m tools.replay logs/raw/<file>.jsonl --no-llm        # code fallback only, free
    python -m tools.replay logs/raw/<file>.jsonl --show-view     # also print what the LLM reads
    python -m tools.replay logs/raw/<file>.jsonl --phase moving --limit 3

Traces go to logs/replay/ (not logs/decisions/).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from altruagent import DecisionContext, GameState
from agent.pokemon.agent import DecisionLog, PokemonAgent

REPLAY_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "replay"


class _ViewPrinter(DecisionLog):
    def __init__(self, show_view: bool) -> None:
        super().__init__(REPLAY_LOG_DIR)
        self.show_view = show_view

    def write(self, context, record):
        super().write(context, record)
        if self.show_view:
            print("-" * 100)
            print(record.get("view", ""))
            for attempt in record.get("attempts", []):
                print("  attempt:", json.dumps(attempt, ensure_ascii=False)[:600])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_log", type=Path)
    parser.add_argument("--no-llm", action="store_true", help="skip the model; play the code fallback")
    parser.add_argument("--show-view", action="store_true", help="print the text the model reads")
    parser.add_argument("--phase", choices=["draft", "team_preview", "moving"], help="only print this phase")
    parser.add_argument("--limit", type=int, default=0, help="stop after N printed decisions")
    parser.add_argument("--opponent", default="", help="pretend the opponent has this name (uses logs/replay_memory/)")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    seat = re.search(r"_seat(\d+)\.jsonl$", args.raw_log.name)
    printer = _ViewPrinter(args.show_view)
    agent = PokemonAgent(provider=False if args.no_llm else None, log=printer, verbose=True, use_clock=False,
                         memory_dir=REPLAY_LOG_DIR.parent / "replay_memory", reflect=False)
    print(f"battle model: {agent.provider!r}; draft model: {agent.draft_provider!r}")

    printed = 0
    for line in args.raw_log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") != "decision":
            continue
        raw = record["raw_state"]
        state = GameState.from_mcp_state(raw)
        if not state.legal_actions:
            continue
        context = DecisionContext(
            session_id=raw.get("session_id", "replay"),
            tournament_id=None,
            game_type=raw.get("game_type"),
            agent_id=(raw.get("current_actor") or {}).get("agent_id", ""),
            seat_position=int(seat.group(1)) if seat else None,
            opponents=(args.opponent,) if args.opponent else (),
        )
        show = args.phase is None or state.phase == args.phase
        printer.show_view = args.show_view and show
        agent.verbose = show
        agent.choose_action(state, context)  # memory (draft cards, freshness) builds up in order
        if show:
            printed += 1
            if args.limit and printed >= args.limit:
                break


if __name__ == "__main__":
    main()
