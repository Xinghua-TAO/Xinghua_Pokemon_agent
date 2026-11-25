"""Recorder agent: wraps another agent and logs every decision it makes.

Used to capture what the server actually sends in each phase of
pokemon_vgc_doubles_draft (draft, team preview, battle) and how long each
decision took, so the real agent is built from real payloads and timings.

    python -m agent --match --agent tools.recorder                 # wraps examples.smoke_agent
    RECORDER_INNER=examples.llm_agent python -m agent --match --agent tools.recorder

RECORDER_INNER is a dotted module path whose create_agent() builds the
wrapped agent (default: examples.smoke_agent).

Each game seat writes one JSON-lines file under logs/raw/: one line per
decision with the full raw state, the action sent, and how long the decision
took; plus one line per server answer when the runtime reports it
(on_action_result).
"""

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from altruagent import DecisionContext, GameState

LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "raw"
DEFAULT_INNER = "examples.smoke_agent"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if all(hasattr(value, attr) for attr in ("action_id", "label", "input")):  # a LegalAction
        return {"action_id": value.action_id, "label": value.label, "input": _jsonable(value.input)}
    if hasattr(value, "action") and hasattr(value, "reasoning_summary"):  # WithReasoning
        return {"action": _jsonable(value.action), "reasoning_summary": value.reasoning_summary}
    return repr(value)


class RecorderAgent:
    def __init__(self) -> None:
        self.inner_name = os.environ.get("RECORDER_INNER") or DEFAULT_INNER
        self._inner = importlib.import_module(self.inner_name).create_agent()
        self._file = None

    def _inner_choose(self, state: GameState, context: DecisionContext):
        if callable(self._inner):
            return self._inner(state, context)
        return self._inner.choose_action(state, context)

    def _write(self, context: DecisionContext, record: dict) -> None:
        if self._file is None:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            inner = self.inner_name.rsplit(".", 1)[-1]
            name = f"{stamp}_{context.game_type}_{inner}_{context.session_id}_seat{context.seat_position}.jsonl"
            self._file = open(LOG_DIR / name, "a", encoding="utf-8")
        record = {"t": datetime.now(timezone.utc).isoformat(), "agent": self.inner_name, **record}
        self._file.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._file.flush()

    def choose_action(self, state: GameState, context: DecisionContext):
        started = time.monotonic()
        action = self._inner_choose(state, context)
        self._write(context, {
            "kind": "decision",
            "phase": state.phase,
            "state_version": state.state_version,
            "decision_ms": round((time.monotonic() - started) * 1000, 1),
            "action": action,
            "raw_state": state.raw,
        })
        return action

    def on_action_result(self, result, context: DecisionContext) -> None:
        self._write(context, {"kind": "action_result", "result": result})
        forward = getattr(self._inner, "on_action_result", None)
        if callable(forward):
            forward(result, context)

    def choose_message(self, state: GameState, context: DecisionContext):
        forward = getattr(self._inner, "choose_message", None)
        if callable(forward):
            return forward(state, context)
        from altruagent import TERMINATE_MESSAGING
        return TERMINATE_MESSAGING


def create_agent():
    return RecorderAgent()
