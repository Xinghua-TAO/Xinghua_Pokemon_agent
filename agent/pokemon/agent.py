"""Our Pokémon VGC doubles draft agent: the LLM decides, code keeps it safe.

Every decision follows the same loop:
  1. code builds a short text view of the game and a menu of valid choices;
  2. the LLM answers in a strict JSON schema whose choice fields are enums of
     those menu ids, so it cannot invent an illegal move;
  3. code checks what the schema can't (e.g. two slots switching into the
     same Pokémon) and asks once more with the reason if needed;
  4. out of time or out of attempts, code plays a sensible default
     (logged as FALLBACK), never a random move;
  5. everything (view, menu, answers, timing) is written to logs/decisions/
     as a trace, so a lost game can be debugged step by step.

Across games, memory.py keeps what each opponent did (draft, leads, habits)
and shows it in the next game against the same opponent.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from altruagent import DecisionContext, GameState, WithReasoning
from altruagent.worker import game_outcome
from examples.llm.providers import ProviderError

from .analysis import (
    annotate_options, assume_booster, battle_facts, card_mon, double_target_lines, fresh_from_log, speed_line,
    threat_lines,
)
from .calc import build_mon
from .cards import CardBook, species_key
from .teambuild import draft_lines, lineup_lines, rank_draft, rank_lineups
from .llm import providers_from_env
from .memory import OpponentMemory, history_lines, now_iso, parse_battle, this_game_lines
from .options import Option, build_slot_options, combo_error, fallback_choice
from . import promptbook
from .views import SNAKE_ORDER, battle_view, draft_view, my_side_id, team_preview_view

# Time budgets (seconds), always inside the server's own limits:
# draft picks have 15 s; the platform plays a random move after 50 s a turn
# (85 s at Team Preview); the whole battle has a 420 s bank.
# A strong model (gpt-5.5) takes ~10 s a turn with a long tail, so the first
# attempt may take most of the budget, but always leaves FAST_RESERVE for a
# second attempt with the fast provider (the draft model, effort "none").
DRAFT_BUDGET, DRAFT_MARGIN = 10.0, 3.0
PREVIEW_BUDGET, PREVIEW_REQUEST = 70.0, 45.0
TURN_BUDGET, TURN_REQUEST = 40.0, 30.0
CLOCK_MARGIN = 8.0
FAST_RESERVE = 8.0
LOW_BANK = 120.0  # below this many bank seconds, play turns with the fast provider
LOW_BANK_BUDGET = 12.0
MIN_ATTEMPT = 2.0
MAX_ATTEMPTS = 2
REFLECT_TIMEOUT = 30.0

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "decisions"

# A/B test of code-ranked team building (docs/team_building.md). With
# POKEMON_AB_TEAMBUILD=1, the two seats of a self-play game get opposite
# arms: "ranked" sees the CODE-RANKED shortlists, "plain" does not. Battles
# are identical, so a difference in wins comes from the draft and Team
# Preview. Each seat appends its result to AB_DIR/teambuild.jsonl
# (tools/ab_report.py sums them up).
AB_ENV = "POKEMON_AB_TEAMBUILD"
# Code-ranked shortlists are OFF by default: in the A/B test (2026-11-25) the
# ranked seat won 3 of 11 games; its 1-on-1 duel score favours bulky, slow
# teams, and team Speed predicted the winner better (docs/team_building.md).
# POKEMON_TEAMBUILD=1 turns them back on.
TEAMBUILD_ENV = "POKEMON_TEAMBUILD"
AB_DIR = Path(__file__).resolve().parents[2] / "logs" / "ab"


def ab_arm(session_id: str, seat_position: int | None, agent_id: str = "") -> str:
    """Opposite arms for the two seats of one game; which seat gets which
    changes from game to game (a hash of the session id, the same in both
    seat processes, unlike Python's salted hash())."""
    digest = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16)
    if not isinstance(seat_position, int):  # unknown seat: a coin flip per seat
        digest = int(hashlib.sha256(f"{session_id}:{agent_id}".encode()).hexdigest()[:8], 16)
        seat_position = 0
    return "ranked" if (digest + seat_position) % 2 == 0 else "plain"


def _object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


_TEXT = {"type": "string"}


class DecisionLog:
    """One JSON-lines trace file per game seat."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or LOG_DIR
        self._file = None

    def write(self, context: DecisionContext, record: dict) -> None:
        try:
            if self._file is None:
                self._directory.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                name = f"{stamp}_{context.session_id}_seat{context.seat_position}.jsonl"
                self._file = open(self._directory / name, "a", encoding="utf-8")
            record = {"t": datetime.now(timezone.utc).isoformat(), **record}
            self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001 - logging must never cost a move
            pass


class PokemonAgent:
    def __init__(self, provider=None, *, log: DecisionLog | None = None, book: CardBook | None = None,
                 verbose: bool = True, use_clock: bool = True, memory_dir: Path | None = None,
                 reflect: bool | None = None, prompt_version: str | None = None, ab: bool | None = None,
                 teambuild: bool | None = None) -> None:
        self.use_clock = use_clock  # False for offline replays of recorded states
        self.prompt_version = prompt_version or promptbook.current_version()
        self.prompts = promptbook.load(self.prompt_version)  # draft / preview / battle / reflect
        self.memory_dir = memory_dir  # None: memory/opponents/ in the repo
        # After each game, ask the LLM for short notes on the opponent (POKEMON_REFLECT=0 turns it off).
        self.reflect = (os.environ.get("POKEMON_REFLECT", "1") != "0") if reflect is None else reflect
        self.memory: OpponentMemory | None = None
        self.record: dict | None = None  # this game, as it will be remembered
        self.past: list[dict] = []  # earlier games against this opponent
        # provider=None: from the environment (battle + draft models, see llm.py);
        # provider=False: no LLM (fallback only); a provider object: used for every phase.
        if provider is None:
            self.provider, self.draft_provider = providers_from_env()
        else:
            self.provider = self.draft_provider = provider or None
        self.book = book or CardBook()
        self.log = log or DecisionLog()
        self.verbose = verbose
        self.opponent_preview: list[dict] = []
        self.ab = os.environ.get(AB_ENV, "0") not in ("", "0") if ab is None else ab
        self.arm: str | None = None  # A/B arm of this game, when the A/B test is on
        # Show the CODE-RANKED shortlists? Off by default; the A/B test overrides it per seat.
        self.use_ranking = os.environ.get(TEAMBUILD_ENV, "0") not in ("", "0") if teambuild is None else teambuild
        # Fake Out works only on a Pokémon's first turn on the field.
        self._last_active: list[str | None] = [None, None]
        self._has_acted: list[bool] = [False, False]

    # ------------------------------------------------------------ entry point

    def choose_action(self, state: GameState, context: DecisionContext):
        self._begin(context)
        try:
            if state.phase == "draft":
                return self._draft(state, context)
            if state.phase == "team_preview":
                return self._team_preview(state, context)
            if state.phase == "moving":
                return self._battle(state, context)
            return state.legal_actions[0]  # other Pokémon formats are not in the UCLA event
        finally:
            self._save()

    # ------------------------------------------------------------ memory

    def _begin(self, context: DecisionContext) -> None:
        """On this game's first decision: find the opponent's memory and start
        this game's record (or pick it up again after a restart)."""
        if self.record is not None:
            return
        name = next((o for o in (context.opponents or ()) if o), "")
        self.memory = OpponentMemory(name, self.memory_dir) if name else None
        existing = None
        if self.memory:
            games = self.memory.load()["games"]
            existing = next((g for g in games if g.get("session_id") == context.session_id), None)
            self.past = [g for g in games if g.get("session_id") != context.session_id]
        self.record = existing or {
            "session_id": context.session_id, "match_id": context.match_id, "round": context.round_label,
            "tournament": bool(context.tournament_id), "opponent": name, "started_at": now_iso(),
            "result": None, "their_picks": [], "our_picks": [], "their_roster": [], "our_bring": [],
            "our_leads": [], "battle": {}, "our_moves": {}, "notes": [],
        }
        if self.ab:
            self.arm = ab_arm(context.session_id, context.seat_position, context.agent_id)
            self.use_ranking = self.arm == "ranked"
            self.record["ab_arm"] = self.arm
            if self.verbose:
                print(f"[pokemon] A/B team building: this seat is '{self.arm}'", flush=True)
        if self.verbose:
            if self.memory:
                print(f"[pokemon] memory: {len(self.past)} earlier game(s) vs {name}", flush=True)
            else:
                print("[pokemon] memory: off (no opponent name, e.g. self-play)", flush=True)

    def _save(self) -> None:
        if self.memory and self.record:
            self.memory.save_game(self.record)

    def _history(self, *, brief: bool = False) -> list[str]:
        if not self.memory:
            return []
        return history_lines(self.memory.name, self.past, brief=brief)

    def on_game_end(self, final_state: GameState, context: DecisionContext) -> None:
        """Called by the runtime once the game is over: store the result and,
        optionally, the LLM's short notes about this opponent."""
        if not self.record:
            return
        outcome = game_outcome(final_state, getattr(context, "agent_id", ""))  # the runtime's own reading of the result
        self.record["result"] = {"won": "win", "lost": "loss", "drew": "draw"}.get(outcome, outcome)
        self.record["termination"] = getattr(final_state, "termination_reason", None) or (
            (getattr(final_state, "final_result", None) or {}).get("termination_reason"))
        self.record["finished_at"] = now_iso()
        if self.arm:
            self._write_ab(context)
        if not self.memory:
            return  # no opponent name (e.g. self-play): nothing to remember, no LLM call
        if self.reflect and self.provider and self.record.get("battle"):
            self.record["notes"] = self._reflect()
        self._save()

    def _write_ab(self, context: DecisionContext) -> None:
        keys = ("session_id", "match_id", "ab_arm", "result", "termination", "drafted_first", "fallbacks",
                "our_picks", "their_picks", "our_bring", "our_leads", "started_at", "finished_at")
        row = {k: self.record.get(k) for k in keys}
        row.update(seat=context.seat_position, agent_id=context.agent_id, prompt_version=self.prompt_version,
                   model=getattr(self.provider, "model", None))
        try:
            AB_DIR.mkdir(parents=True, exist_ok=True)
            with open(AB_DIR / "teambuild.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")  # one write: both seats append safely
        except OSError as exc:
            print(f"[pokemon] could not save the A/B result: {exc}", flush=True)
        if self.verbose:
            print(f"[pokemon] A/B result: arm {self.arm}, {row['result']} ({row['termination']})", flush=True)

    def _reflect(self) -> list[str]:
        record = self.record or {}
        battle = record.get("battle") or {}
        facts = {
            "result for us": record.get("result"),
            "their draft picks in order": record.get("their_picks"),
            "our draft picks": record.get("our_picks"),
            "their leads": battle.get("leads"), "they brought": battle.get("brought"),
            "our leads": record.get("our_leads"), "we brought": record.get("our_bring"),
            "their moves by turn [pokemon, move, target]": battle.get("moves_by_turn"),
            "our moves by turn": record.get("our_moves"),
        }
        schema = _object_schema({"notes": {"type": "array", "items": _TEXT}})
        answer, _ = self._ask("game_notes", self.prompts["reflect"], json.dumps(facts, ensure_ascii=False), schema,
                              lambda a: None if isinstance(a.get("notes"), list) else "notes must be a list",
                              REFLECT_TIMEOUT, request_timeout=REFLECT_TIMEOUT)
        return [str(n)[:300] for n in (answer or {}).get("notes", [])][:4]

    # ------------------------------------------------------------ LLM helper

    def _ask(self, name: str, system: str, view: str, schema: dict, validate, budget: float,
             provider=None, retry_provider=None, request_timeout: float = 20.0) -> tuple[dict | None, list]:
        """Ask the model until ``validate(answer)`` returns None (no error).
        The second attempt uses ``retry_provider`` (a faster model setting)
        when given. Returns (answer or None, attempt records)."""
        first = provider or self.provider
        second = retry_provider or first
        attempts: list[dict] = []
        if first is None:
            attempts.append({"error": "no OPENAI_API_KEY; using fallback"})
            return None, attempts
        deadline = time.monotonic() + budget
        messages = [{"role": "system", "content": system}, {"role": "user", "content": view}]
        for attempt in range(MAX_ATTEMPTS):
            current = first if attempt == 0 else second
            left = deadline - time.monotonic()
            if left < MIN_ATTEMPT:
                attempts.append({"error": f"out of time ({left:.1f}s left)"})
                break
            limit = min(request_timeout, left)
            if attempt == 0 and second is not first:
                limit = min(limit, max(MIN_ATTEMPT, left - FAST_RESERVE))  # keep time for the fast retry
            started = time.monotonic()
            model = getattr(current, "model", None)
            try:
                answer = current.complete_structured(messages, name, schema, timeout=limit)
            except ProviderError as exc:
                attempts.append({"error": str(exc), "model": model, "seconds": round(time.monotonic() - started, 2)})
                continue
            error = validate(answer)
            attempts.append({"answer": answer, "error": error, "model": model,
                             "seconds": round(time.monotonic() - started, 2)})
            if error is None:
                return answer, attempts
            messages += [
                {"role": "assistant", "content": json.dumps(answer)},
                {"role": "user", "content": f"That answer is invalid: {error}. Answer again."},
            ]
        return None, attempts

    def _record(self, context: DecisionContext, phase: str, started: float, view: str, attempts: list,
                action, fallback: bool, extra: dict | None = None) -> None:
        seconds = round(time.monotonic() - started, 2)
        provider = self.draft_provider if phase == "draft" else self.provider
        self.log.write(context, {
            "phase": phase, "prompt_version": self.prompt_version, "model": getattr(provider, "model", None),
            "reasoning_effort": getattr(provider, "reasoning_effort", None), "seconds": seconds,
            "fallback": fallback, "view": view, "attempts": attempts, "action": action, "ab_arm": self.arm,
            **(extra or {}),
        })
        if fallback and self.record is not None:
            self.record["fallbacks"] = self.record.get("fallbacks", 0) + 1
        if self.verbose:
            reason = ""
            for attempt in reversed(attempts):
                answer = attempt.get("answer") or {}
                if answer.get("public_reason"):
                    reason = answer["public_reason"]
                    break
            tag = "FALLBACK " if fallback else ""
            print(f"[pokemon] {phase}: {tag}{json.dumps(action, ensure_ascii=False)} ({seconds}s) {reason}", flush=True)

    # ------------------------------------------------------------ budgets

    def _budget(self, state: GameState) -> float:
        if not self.use_clock:
            return {"draft": DRAFT_BUDGET, "team_preview": PREVIEW_BUDGET}.get(state.phase, TURN_BUDGET)
        obs = (state.raw or {}).get("observation") or {}
        if state.phase == "draft":
            left = DRAFT_BUDGET + DRAFT_MARGIN
            deadline = obs.get("decision_deadline_at")
            if deadline:
                try:
                    left = (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()
                except ValueError:
                    pass
            return max(0.0, min(DRAFT_BUDGET, left - DRAFT_MARGIN))
        cap = PREVIEW_BUDGET if state.phase == "team_preview" else TURN_BUDGET
        if self._low_bank(state):
            cap = min(cap, LOW_BANK_BUDGET)
        left = ((obs.get("clock") or {}).get("decision_seconds_left"))
        if not isinstance(left, (int, float)):
            return cap
        return max(0.0, min(cap, left - CLOCK_MARGIN))

    @staticmethod
    def _low_bank(state: GameState) -> bool:
        """The 420 s battle bank is running low: answer fast from now on."""
        obs = (state.raw or {}).get("observation") or {}
        bank = (obs.get("clock") or {}).get("bank_seconds_left")
        return state.phase == "moving" and isinstance(bank, (int, float)) and bank < LOW_BANK

    def _providers_for(self, state: GameState) -> tuple:
        """(first, retry) providers for a Team Preview or battle decision."""
        fast = self.draft_provider or self.provider
        if self._low_bank(state):
            return fast, fast
        return self.provider, (fast if fast is not self.provider else None)

    # ------------------------------------------------------------ draft

    def _draft(self, state: GameState, context: DecisionContext):
        started = time.monotonic()
        obs = (state.raw or {}).get("observation") or {}
        self.book.remember(obs.get("available_cards") or [])
        legal = {
            a.action_id.split(":", 1)[1]: a
            for a in state.legal_actions
            if a.action_id.startswith("draft_pick:")
        }
        if not legal:
            return state.legal_actions[0]
        legal_ids = list(legal)
        picks = [p for p in obs.get("picks") or [] if isinstance(p, dict)]
        self.record["our_picks"] = [p.get("species") for p in picks if p.get("seat_id") == context.agent_id]
        self.record["their_picks"] = [p.get("species") for p in picks if p.get("seat_id") != context.agent_id]
        if obs.get("first_drafter"):
            self.record["drafted_first"] = obs["first_drafter"] == context.agent_id
        ranked, rank_lines, rank_error = [], [], None
        try:
            if self.use_ranking:
                ranked, opponent_next = self._rank_draft(obs, context, legal_ids)
                rank_lines = draft_lines(ranked, opponent_next)
        except Exception as exc:  # noqa: BLE001 - the ranking is advice; never let it cost the pick
            rank_error = f"{type(exc).__name__}: {exc}"
        view = draft_view(obs, context.agent_id, legal_ids, self.book, rank_lines + self._history())
        schema = _object_schema({"analysis": _TEXT, "pick": {"type": "string", "enum": legal_ids}, "public_reason": _TEXT})
        answer, attempts = self._ask(
            "draft_pick", self.prompts["draft"], view, schema,
            lambda a: None if a.get("pick") in legal else f"pick must be one of {legal_ids}",
            self._budget(state), provider=self.draft_provider, request_timeout=DRAFT_BUDGET,
        )
        fallback = answer is None
        if fallback:
            pick = ranked[0]["card_id"] if ranked and ranked[0]["card_id"] in legal else legal_ids[0]
        else:
            pick = answer["pick"]
        reason = None if fallback else answer.get("public_reason")
        self._record(context, "draft", started, view, attempts, {"pick": pick}, fallback,
                     {"code_ranking": [(r["card_id"], r["score"]) for r in ranked], "ranking_error": rank_error})
        return WithReasoning(legal[pick], reason) if reason else legal[pick]

    def _rank_draft(self, obs: dict, context: DecisionContext, legal_ids: list[str]) -> tuple[list[dict], bool]:
        available = {c["card_id"]: c for c in obs.get("available_cards") or [] if isinstance(c, dict) and c.get("card_id")}
        rosters = obs.get("rosters") or {}

        def mons(entries) -> list:
            cards = [self.book.by_id(e.get("card_id", "")) or self.book.by_species(e.get("species"))
                     for e in entries or [] if isinstance(e, dict)]
            return [m for m in (card_mon(c) for c in cards if c) if m]

        mine = mons(rosters.get(context.agent_id))
        theirs = mons(next((v for k, v in rosters.items() if k != context.agent_id), []))
        pool = [m for m in (card_mon(c) for c in available.values()) if m]
        legal = []
        for card_id in legal_ids:
            card = available.get(card_id) or self.book.by_id(card_id)
            mon = card_mon(card) if card else None
            if mon:
                legal.append((card_id, mon))
        my_letter = "A" if obs.get("first_drafter") == context.agent_id else "B"
        pick_number = int(obs.get("pick_number") or 1)
        opponent_next = pick_number < len(SNAKE_ORDER) and SNAKE_ORDER[pick_number] != my_letter
        return rank_draft(legal, mine, theirs, pool, opponent_next=opponent_next), opponent_next

    # ------------------------------------------------------------ team preview

    def _team_preview(self, state: GameState, context: DecisionContext):
        started = time.monotonic()
        obs = (state.raw or {}).get("observation") or {}
        self.opponent_preview = [e for e in obs.get("opponent_roster") or [] if isinstance(e, dict)]
        template = state.legal_actions[0].input.get("action") or {}
        roster = [s for s in template.get("roster") or [] if isinstance(s, str)]
        self.record["their_roster"] = [e.get("species") for e in self.opponent_preview]
        ranked, rank_lines, rank_error = [], [], None
        try:
            if self.use_ranking:
                ranked = self._rank_lineups(obs)
                ranked = [r for r in ranked if set(r["bring"]) <= set(roster)]
                rank_lines = lineup_lines(ranked)
        except Exception as exc:  # noqa: BLE001 - advice only
            rank_error = f"{type(exc).__name__}: {exc}"
        view = team_preview_view(obs, self.book, rank_lines + self._history())
        species = {"type": "array", "items": {"type": "string", "enum": roster}}
        schema = _object_schema({"analysis": _TEXT, "bring": species, "leads": species, "public_reason": _TEXT})

        def validate(a: dict) -> str | None:
            bring, leads = a.get("bring") or [], a.get("leads") or []
            if len(set(bring)) != 4 or len(bring) != 4:
                return "bring must list exactly 4 different species from your roster"
            if len(set(leads)) != 2 or len(leads) != 2 or not set(leads) <= set(bring):
                return "leads must be exactly 2 different species that are also in bring"
            return None

        first, retry = self._providers_for(state)
        answer, attempts = self._ask("select_lineup", self.prompts["preview"], view, schema, validate, self._budget(state),
                                     provider=first, retry_provider=retry, request_timeout=PREVIEW_REQUEST)
        fallback = answer is None
        if fallback and ranked:
            bring, leads = ranked[0]["bring"], ranked[0]["leads"]
        elif fallback:
            bring, leads = roster[:4], roster[:2]
        else:
            bring, leads = answer["bring"], answer["leads"]
        action = {"type": "select_lineup", "bring": bring, "leads": leads}
        self.record["our_bring"], self.record["our_leads"] = list(bring), list(leads)
        self._record(context, "team_preview", started, view, attempts, action, fallback,
                     {"code_ranking": [(r["bring"], r["leads"], r["score"]) for r in ranked],
                      "ranking_error": rank_error})
        reason = None if fallback else answer.get("public_reason")
        return WithReasoning(action, reason) if reason else action

    def _rank_lineups(self, obs: dict) -> list[dict]:
        def mons(entries) -> list:
            out = []
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                mon = build_mon(entry, self.book.by_species(entry.get("species")), exact_hp=False)
                if mon:
                    mon.hp = mon.max_hp
                    assume_booster(mon)
                    out.append(mon)
            return out

        past_leads = Counter(species_key(s) for g in self.past for s in (g.get("battle") or {}).get("leads") or [])
        return rank_lineups(mons(obs.get("your_roster")), mons(obs.get("opponent_roster")), past_leads=past_leads)

    # ------------------------------------------------------------ battle

    def _battle(self, state: GameState, context: DecisionContext):
        started = time.monotonic()
        template = state.legal_actions[0].input.get("action") or {}
        if template.get("type") != "doubles_turn":
            from examples import smoke_agent  # not the UCLA format; play something valid
            return smoke_agent.choose_action(state, context)
        obs = (state.raw or {}).get("observation") or {}
        slots = {s.get("slot", i): s for i, s in enumerate(template.get("slots") or []) if isinstance(s, dict)}
        moves_by_slot = obs.get("available_moves") or [[], []]
        bench = {e.get("species"): e for e in (obs.get("team") or {}).values() if isinstance(e, dict)}

        current = [species_key(((slots.get(n) or {}).get("active") or {}).get("species")) or None for n in (0, 1)]
        fresh = [current[n] != self._last_active[n] or not self._has_acted[n] for n in (0, 1)]

        # Code-computed facts (damage, threats, speed). Never let a bug here cost the move.
        facts, facts_lines, analysis_error = None, [], None
        try:
            facts = battle_facts(obs, self.book, my_side_id(obs))
            # The battle log knows whether each Pokémon has moved since it came in;
            # it beats our own tracking (and survives a process restart).
            fresh = [f if f is not None else fresh[n] for n, f in enumerate(fresh_from_log(facts))]
        except Exception as exc:  # noqa: BLE001
            analysis_error = f"{type(exc).__name__}: {exc}"

        options: dict[int, list[Option]] = {}
        for n, letter in ((0, "a"), (1, "b")):
            slot = slots.get(n) or {"options": [{"type": "pass"}]}
            partner = (slots.get(1 - n) or {}).get("active") or None
            options[n] = build_slot_options(
                slot, moves_by_slot[n] if n < len(moves_by_slot) else [],
                letter=letter, fresh=fresh[n], partner=partner, bench=bench,
            )
        by_id = {o.id: o for n in (0, 1) for o in options[n]}

        if facts is not None:
            try:
                annotate_options(options, facts)
                facts_lines = [speed_line(facts),
                               "OPPONENT THREATS (their best attack on each of your active Pokémon):",
                               *threat_lines(facts),
                               *double_target_lines(options)]
            except Exception as exc:  # noqa: BLE001
                analysis_error = f"{type(exc).__name__}: {exc}"
        try:  # what this opponent did so far, and in earlier games
            self.record["battle"] = parse_battle(obs.get("protocol_log") or [], my_side_id(obs))
            facts_lines += this_game_lines(self.record["battle"]) + self._history(brief=True)
        except Exception as exc:  # noqa: BLE001
            analysis_error = analysis_error or f"memory: {type(exc).__name__}: {exc}"

        view = battle_view(obs, options, self.book, self.opponent_preview, facts_lines)
        schema = _object_schema({
            "analysis": _TEXT,
            "slot_0": {"type": "string", "enum": [o.id for o in options[0]]},
            "slot_1": {"type": "string", "enum": [o.id for o in options[1]]},
            "public_reason": _TEXT,
        })

        def validate(a: dict) -> str | None:
            first, second = by_id.get(a.get("slot_0")), by_id.get(a.get("slot_1"))
            if first is None or first not in options[0]:
                return "slot_0 must be one of the slot 0 option ids"
            if second is None or second not in options[1]:
                return "slot_1 must be one of the slot 1 option ids"
            return combo_error(first, second)

        first_provider, retry = self._providers_for(state)
        answer, attempts = self._ask("doubles_turn", self.prompts["battle"], view, schema, validate, self._budget(state),
                                     provider=first_provider, retry_provider=retry, request_timeout=TURN_REQUEST)
        fallback = answer is None
        if fallback:
            first, second = fallback_choice(options[0], options[1])
        else:
            first, second = by_id[answer["slot_0"]], by_id[answer["slot_1"]]

        for n, choice in ((0, first), (1, second)):
            if choice.kind == "pass":
                continue
            self._has_acted[n] = choice.kind == "move" and current[n] is not None
            self._last_active[n] = current[n]

        action = {"type": "doubles_turn", "slot_0": first.payload, "slot_1": second.payload}
        turn_moves = self.record.setdefault("our_moves", {}).setdefault(str(obs.get("turn")), [])
        for n, choice in ((0, first), (1, second)):
            if choice.kind == "pass":
                continue
            name = ((slots.get(n) or {}).get("active") or {}).get("name")
            what = choice.payload.get("move_id") or f"switch to {choice.payload.get('species')}"
            turn_moves.append([name, what, choice.payload.get("target")])
        self._record(context, f"turn {obs.get('turn')}", started, view, attempts, action, fallback,
                     {"options": {o.id: o.text for n in (0, 1) for o in options[n]}, "fresh": fresh,
                      "analysis_error": analysis_error})
        reason = None if fallback else answer.get("public_reason")
        return WithReasoning(action, reason) if reason else action
