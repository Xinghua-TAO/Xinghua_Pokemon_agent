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
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from altruagent import DecisionContext, GameState, WithReasoning
from examples.llm.providers import ProviderError

from .analysis import annotate_options, battle_facts, fresh_from_log, speed_line, threat_lines
from .cards import CardBook, species_key
from .llm import providers_from_env
from .options import Option, build_slot_options, combo_error, fallback_choice
from .views import (
    BATTLE_SYSTEM, DRAFT_SYSTEM, PREVIEW_SYSTEM, battle_view, draft_view, my_side_id, team_preview_view,
)

# Time budgets (seconds), always inside the server's own limits:
# draft picks have 15 s; the platform plays a random move after 50 s a turn
# (85 s at Team Preview); the whole battle has a 420 s bank.
DRAFT_BUDGET, DRAFT_MARGIN = 10.0, 3.0
PREVIEW_BUDGET = 30.0
TURN_BUDGET = 25.0
CLOCK_MARGIN = 8.0
REQUEST_TIMEOUT = 20.0
MIN_ATTEMPT = 2.0
MAX_ATTEMPTS = 2

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "decisions"


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
                 verbose: bool = True, use_clock: bool = True) -> None:
        self.use_clock = use_clock  # False for offline replays of recorded states
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
        # Fake Out works only on a Pokémon's first turn on the field.
        self._last_active: list[str | None] = [None, None]
        self._has_acted: list[bool] = [False, False]

    # ------------------------------------------------------------ entry point

    def choose_action(self, state: GameState, context: DecisionContext):
        if state.phase == "draft":
            return self._draft(state, context)
        if state.phase == "team_preview":
            return self._team_preview(state, context)
        if state.phase == "moving":
            return self._battle(state, context)
        return state.legal_actions[0]  # other Pokémon formats are not in the UCLA event

    # ------------------------------------------------------------ LLM helper

    def _ask(self, name: str, system: str, view: str, schema: dict, validate, budget: float,
             provider=None) -> tuple[dict | None, list]:
        """Ask the model until ``validate(answer)`` returns None (no error).
        Returns (answer or None, attempt records)."""
        provider = provider or self.provider
        attempts: list[dict] = []
        if provider is None:
            attempts.append({"error": "no OPENAI_API_KEY; using fallback"})
            return None, attempts
        deadline = time.monotonic() + budget
        messages = [{"role": "system", "content": system}, {"role": "user", "content": view}]
        for _ in range(MAX_ATTEMPTS):
            left = deadline - time.monotonic()
            if left < MIN_ATTEMPT:
                attempts.append({"error": f"out of time ({left:.1f}s left)"})
                break
            started = time.monotonic()
            try:
                answer = provider.complete_structured(messages, name, schema, timeout=min(REQUEST_TIMEOUT, left))
            except ProviderError as exc:
                attempts.append({"error": str(exc), "seconds": round(time.monotonic() - started, 2)})
                continue
            error = validate(answer)
            attempts.append({"answer": answer, "error": error, "seconds": round(time.monotonic() - started, 2)})
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
            "phase": phase, "model": getattr(provider, "model", None),
            "reasoning_effort": getattr(provider, "reasoning_effort", None), "seconds": seconds,
            "fallback": fallback, "view": view, "attempts": attempts, "action": action, **(extra or {}),
        })
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
        left = ((obs.get("clock") or {}).get("decision_seconds_left"))
        if not isinstance(left, (int, float)):
            return cap
        return max(0.0, min(cap, left - CLOCK_MARGIN))

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
        view = draft_view(obs, context.agent_id, legal_ids, self.book)
        schema = _object_schema({"analysis": _TEXT, "pick": {"type": "string", "enum": legal_ids}, "public_reason": _TEXT})
        answer, attempts = self._ask(
            "draft_pick", DRAFT_SYSTEM, view, schema,
            lambda a: None if a.get("pick") in legal else f"pick must be one of {legal_ids}",
            self._budget(state), provider=self.draft_provider,
        )
        fallback = answer is None
        pick = legal_ids[0] if fallback else answer["pick"]
        reason = None if fallback else answer.get("public_reason")
        self._record(context, "draft", started, view, attempts, {"pick": pick}, fallback)
        return WithReasoning(legal[pick], reason) if reason else legal[pick]

    # ------------------------------------------------------------ team preview

    def _team_preview(self, state: GameState, context: DecisionContext):
        started = time.monotonic()
        obs = (state.raw or {}).get("observation") or {}
        self.opponent_preview = [e for e in obs.get("opponent_roster") or [] if isinstance(e, dict)]
        template = state.legal_actions[0].input.get("action") or {}
        roster = [s for s in template.get("roster") or [] if isinstance(s, str)]
        view = team_preview_view(obs, self.book)
        species = {"type": "array", "items": {"type": "string", "enum": roster}}
        schema = _object_schema({"analysis": _TEXT, "bring": species, "leads": species, "public_reason": _TEXT})

        def validate(a: dict) -> str | None:
            bring, leads = a.get("bring") or [], a.get("leads") or []
            if len(set(bring)) != 4 or len(bring) != 4:
                return "bring must list exactly 4 different species from your roster"
            if len(set(leads)) != 2 or len(leads) != 2 or not set(leads) <= set(bring):
                return "leads must be exactly 2 different species that are also in bring"
            return None

        answer, attempts = self._ask("select_lineup", PREVIEW_SYSTEM, view, schema, validate, self._budget(state))
        fallback = answer is None
        bring = roster[:4] if fallback else answer["bring"]
        leads = bring[:2] if fallback else answer["leads"]
        action = {"type": "select_lineup", "bring": bring, "leads": leads}
        self._record(context, "team_preview", started, view, attempts, action, fallback)
        reason = None if fallback else answer.get("public_reason")
        return WithReasoning(action, reason) if reason else action

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
                               *threat_lines(facts)]
            except Exception as exc:  # noqa: BLE001
                analysis_error = f"{type(exc).__name__}: {exc}"

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

        answer, attempts = self._ask("doubles_turn", BATTLE_SYSTEM, view, schema, validate, self._budget(state))
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
        self._record(context, f"turn {obs.get('turn')}", started, view, attempts, action, fallback,
                     {"options": {o.id: o.text for n in (0, 1) for o in options[n]}, "fresh": fresh,
                      "analysis_error": analysis_error})
        reason = None if fallback else answer.get("public_reason")
        return WithReasoning(action, reason) if reason else action
