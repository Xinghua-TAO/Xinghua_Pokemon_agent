"""Tests for per-opponent memory (agent/pokemon/memory.py) and the end-of-game hook."""

from types import SimpleNamespace

from altruagent import DecisionContext
from altruagent.worker import _notify_game_end
from agent.agent import GameRouter
from agent.pokemon.agent import PokemonAgent
from agent.pokemon.cards import CardBook
from agent.pokemon.memory import OpponentMemory, file_key, history_lines, parse_battle

LOG = [
    "|switch|p1a: Incineroar|Incineroar, L50, M|100/100",
    "|switch|p1b: Tornadus|Tornadus, L50, M|100/100",
    "|switch|p2a: Chien-Pao|Chien-Pao, L50|156/156",
    "|switch|p2b: Amoonguss|Amoonguss, L50|221/221",
    "|turn|1",
    "|move|p1a: Incineroar|Fake Out|p2a: Chien-Pao",
    "|move|p1b: Tornadus|Tailwind|p1b: Tornadus",
    "|turn|2",
    "|move|p1a: Incineroar|Protect|p1a: Incineroar",
    "|move|p1b: Tornadus|Bleakwind Storm|p2a: Chien-Pao",
    "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",
    "|turn|3",
    "|move|p1a: Rillaboom|Wood Hammer|p2b: Amoonguss",
    "|move|p1b: Tornadus|Bleakwind Storm|p2b: Amoonguss",
]


def test_parse_battle_reads_the_opponents_side_only():
    battle = parse_battle(LOG, my_side="p2")
    assert battle["leads"] == ["Incineroar", "Tornadus"]
    assert battle["brought"] == ["Incineroar", "Tornadus", "Rillaboom"]
    assert battle["moves_by_turn"]["1"] == [["Incineroar", "Fake Out", "p2a: Chien-Pao"],
                                           ["Tornadus", "Tailwind", "p1b: Tornadus"]]
    assert battle["protects"] == 1 and battle["switches"] == 1 and battle["turns"] == 3
    assert battle["double_target_turns"] == 1  # turn 3: both into Amoonguss


def test_memory_round_trip_and_history(tmp_path):
    memory = OpponentMemory("Team Rocket!", tmp_path)
    assert memory.path.name == "team-rocket.json" and file_key("") == "unknown"
    for n, result in enumerate(["win", "loss"]):
        memory.save_game({"session_id": f"s{n}", "result": result, "started_at": "2026-11-25T00:00",
                          "their_picks": ["Incineroar", "Tornadus"], "battle": parse_battle(LOG, "p2"), "notes": [f"note {n}"]})
    memory.save_game({"session_id": "s1", "result": "loss", "battle": {}, "their_picks": []})  # update, not append
    games = OpponentMemory("Team Rocket!", tmp_path).past_games()
    assert [g["session_id"] for g in games] == ["s0", "s1"]
    text = "\n".join(history_lines("Team Rocket!", games))
    assert "1 win" in text and "1 loss" in text and "Incineroar + Tornadus" in text and "Fake Out" in text
    assert "note 0" in text and "unverified" in text
    assert history_lines("Nobody", [])[0].endswith("no previous games.")


class FakeProvider:
    model = "fake"

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        assert schema_name == "game_notes"
        return {"notes": ["Leads Incineroar + Tornadus and Tailwinds turn 1."]}


def test_on_game_end_saves_result_and_notes(tmp_path):
    agent = PokemonAgent(FakeProvider(), book=CardBook(tmp_path / "c.json"), memory_dir=tmp_path,
                         verbose=False, reflect=True, log=SimpleNamespace(write=lambda *a: None))
    ctx = DecisionContext(session_id="g1", tournament_id="t", game_type="pokemon_vgc_doubles_draft",
                          agent_id="me", opponents=("Rival",), match_id="m1", round_label="Bracket QF")
    agent._begin(ctx)
    agent.record["battle"] = parse_battle(LOG, "p2")
    agent.on_game_end(SimpleNamespace(final_result={"your_return": 1.0, "termination_reason": "completed"}), ctx)
    saved = OpponentMemory("Rival", tmp_path).past_games()[0]
    assert saved["result"] == "win" and saved["round"] == "Bracket QF" and saved["tournament"] is True
    assert saved["notes"] == ["Leads Incineroar + Tornadus and Tailwinds turn 1."]

    # The next game against Rival starts with that history.
    nxt = PokemonAgent(False, book=CardBook(tmp_path / "c.json"), memory_dir=tmp_path, verbose=False)
    nxt._begin(DecisionContext(session_id="g2", tournament_id="t", game_type="pokemon_vgc_doubles_draft",
                               agent_id="me", opponents=("Rival",)))
    assert "1 win" in "\n".join(nxt._history())


def test_no_opponent_name_means_no_memory(tmp_path):
    agent = PokemonAgent(False, book=CardBook(tmp_path / "c.json"), memory_dir=tmp_path, verbose=False)
    agent._begin(DecisionContext(session_id="g", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="me"))
    assert agent.memory is None and agent._history() == []
    agent.on_game_end(SimpleNamespace(final_result={"your_return": 0.0}), None)  # must not fail
    assert list(tmp_path.glob("*.json")) == []


def test_worker_calls_on_game_end_and_survives_errors(capsys):
    calls = []
    router = GameRouter()
    router._agent = SimpleNamespace(on_game_end=lambda state, ctx: calls.append((state, ctx)))
    _notify_game_end(router, "final", "ctx", "[match]")
    assert calls == [("final", "ctx")]

    def boom(state, ctx):
        raise RuntimeError("disk full")

    _notify_game_end(SimpleNamespace(on_game_end=boom), "final", "ctx", "[match]")
    assert "on_game_end failed: RuntimeError: disk full" in capsys.readouterr().out
    _notify_game_end(lambda s, c: None, "final", "ctx", "[match]")  # plain function agents: no hook, no error
