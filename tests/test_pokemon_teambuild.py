"""Tests for code-ranked team building (agent/pokemon/teambuild.py) and how
the agent uses it at the draft and Team Preview. No network."""

from itertools import combinations
from types import SimpleNamespace

import pytest

from altruagent import DecisionContext, GameState
from agent.pokemon import dex
from agent.pokemon.agent import PokemonAgent
from agent.pokemon.analysis import card_mon
from agent.pokemon.cards import CardBook, load_catalog, species_key
from agent.pokemon.teambuild import (
    W_DENIAL_LATER, W_DENIAL_OPP_NEXT, Duels, draft_lines, hits_to_ko, lineup_lines, rank_draft, rank_lineups, roles,
)
from examples.llm.providers import ProviderError

CATALOG = load_catalog()
OURS = ["vgc-milotic", "vgc-iron-hands", "vgc-garchomp", "vgc-dragonite", "vgc-torkoal", "vgc-arcanine"]
THEIRS = ["vgc-incineroar", "vgc-rillaboom", "vgc-thundurus", "vgc-whimsicott", "vgc-baxcalibur", "vgc-gyarados"]


def mon(card_id: str):
    return card_mon(CATALOG[card_id])


def test_hits_to_ko():
    assert [hits_to_ko(p) for p in (100, 120, 50, 49.9, 34, 33.3, 0)] == [1, 1, 2, 3, 3, 4, 99]


def test_duel_clear_win():
    duels = Duels()
    # Milotic's Scald 2HKOs Torkoal; Torkoal's Eruption does ~16%.
    assert duels.duel(mon("vgc-milotic"), mon("vgc-torkoal")) == 1.0
    assert duels.duel(mon("vgc-torkoal"), mon("vgc-milotic")) == -1.0


def test_duel_is_antisymmetric_across_the_catalog():
    duels = Duels()
    mons = [mon(c) for c in sorted(CATALOG)]
    for a, b in combinations(mons, 2):
        assert duels.duel(a, b) == -duels.duel(b, a), (a.name, b.name)


def test_roles():
    assert {"Fake Out", "Intimidate"} <= roles(mon("vgc-incineroar"))
    assert {"Tailwind", "speed control"} <= roles(mon("vgc-whimsicott"))
    assert "redirection" in roles(mon("vgc-amoonguss"))


def test_rank_lineups_returns_valid_sorted_choices():
    ours, theirs = [mon(c) for c in OURS], [mon(c) for c in THEIRS]
    ranked = rank_lineups(ours, theirs)
    roster = {m.species for m in ours}
    assert len(ranked) == 3
    assert [r["score"] for r in ranked] == sorted((r["score"] for r in ranked), reverse=True)
    for r in ranked:
        assert len(set(r["bring"])) == 4 and set(r["bring"]) <= roster
        assert len(set(r["leads"])) == 2 and set(r["leads"]) <= set(r["bring"])
    lines = lineup_lines(ranked)
    assert lines[0].startswith("CODE-RANKED LINEUPS") and lines[1].startswith("  1. bring ")


def test_past_leads_change_the_lead_weights():
    ours, theirs = [mon(c) for c in OURS], [mon(c) for c in THEIRS]
    plain = rank_lineups(ours, theirs, top=15)
    weighted = rank_lineups(ours, theirs, past_leads={"Whimsicott": 5, "Gyarados": 5}, top=15)
    assert [r["score"] for r in plain] != [r["score"] for r in weighted]


def test_rank_draft_scores_only_legal_cards_and_weights_denial():
    pool = [mon(c) for c in sorted(CATALOG)][:18]
    mine, theirs = pool[:2], pool[2:4]
    legal = [(f"id-{m.species}", m) for m in pool[4:]]
    for opponent_next, w in ((True, W_DENIAL_OPP_NEXT), (False, W_DENIAL_LATER)):
        ranked = rank_draft(legal, mine, theirs, pool, opponent_next=opponent_next, top=20)
        assert {r["card_id"] for r in ranked} == {card_id for card_id, _ in legal}
        assert [r["score"] for r in ranked] == sorted((r["score"] for r in ranked), reverse=True)
        for r in ranked:
            assert r["score"] == pytest.approx(r["gain"] + w * r["denial"], abs=2e-3)
    lines = draft_lines(ranked[:5], opponent_next=True)
    assert "weighted up" in lines[0] and len(lines) == 6


# ---------------------------------------------------------------- agent integration

class FailingProvider:
    model = "fake"

    def __init__(self):
        self.views = []

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.views.append(messages[-1]["content"])
        raise ProviderError("down")


class ScriptedProvider(FailingProvider):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.views.append(messages[-1]["content"])
        return self.answer


class NullLog:
    def write(self, context, record):
        self.last = record


CTX = DecisionContext(session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                      agent_id="me", seat_position=1)


def _agent(provider):
    agent = PokemonAgent(provider, log=NullLog(), book=CardBook(), verbose=False, use_clock=False, reflect=False,
                         ab=False)
    agent.past = []
    return agent


def _draft_state():
    cards = [dict(CATALOG[c]) for c in sorted(CATALOG)[:18]]
    mine, theirs, free = cards[:2], cards[2:4], cards[4:]
    obs = {
        "phase": "draft", "pick_number": 5, "first_drafter": "me",  # pick 5 is A's (ours); 6 is B's
        "available_cards": free,
        "rosters": {"me": [{"card_id": c["card_id"], "species": c["species"]} for c in mine],
                    "them": [{"card_id": c["card_id"], "species": c["species"]} for c in theirs]},
        "picks": [],
    }
    actions = [{"action_id": f"draft_pick:{c['card_id']}", "label": c["card_id"],
                "input": {"action": {"type": "draft_pick", "card_id": c["card_id"]}}} for c in free]
    return GameState.from_mcp_state({
        "session_id": "s", "game_type": "pokemon_vgc_doubles_draft", "status": "in_progress", "state_version": 1,
        "phase": "draft", "is_terminal": False, "is_current_actor": True, "observation": obs,
        "legal_actions": {"session_id": "s", "state_version": 1, "actions": actions}})


def _entry(card_id):
    card = CATALOG[card_id]
    info = dex.species(card["species"])
    return {"species": species_key(card["species"]), "name": info["name"], "current_hp_fraction": 1.0,
            "types": [t.upper() for t in info["types"]], "base_stats": info["bs"],
            "ability": species_key(card["ability"]), "item": species_key(card["item"]),
            "moves": [species_key(m) for m in card["moves"]]}


def _preview_state():
    obs = {"phase": "team_preview", "your_roster": [_entry(c) for c in OURS],
           "opponent_roster": [_entry(c) for c in THEIRS], "clock": {"decision_seconds_left": 80}}
    roster = [e["species"] for e in obs["your_roster"]]
    return GameState.from_mcp_state({
        "session_id": "s", "game_type": "pokemon_vgc_doubles_draft", "status": "in_progress", "state_version": 0,
        "phase": "team_preview", "is_terminal": False, "is_current_actor": True, "observation": obs,
        "legal_actions": {"session_id": "s", "state_version": 0, "actions": [{
            "action_id": "select_lineup", "label": "lineup",
            "input": {"action": {"type": "select_lineup", "roster": roster, "bring_count": 4, "lead_count": 2}}}]}})


def test_draft_shows_ranking_and_falls_back_to_its_top_pick():
    provider = FailingProvider()
    agent = _agent(provider)
    agent.book.remember(list(CATALOG.values()))
    action = agent.choose_action(_draft_state(), CTX)
    record = agent.log.last
    assert record["fallback"] and record["ranking_error"] is None
    assert "CODE-RANKED PICKS" in provider.views[0]
    assert "weighted up" in provider.views[0]  # the opponent picks next
    top = record["code_ranking"][0][0]
    assert action.action_id == f"draft_pick:{top}"


def test_preview_shows_ranking_and_model_still_decides():
    roster = [species_key(CATALOG[c]["species"]) for c in OURS]
    answer = {"analysis": "", "bring": roster[2:], "leads": roster[4:], "public_reason": "plan"}
    provider = ScriptedProvider(answer)
    agent = _agent(provider)
    agent.book.remember(list(CATALOG.values()))
    action = agent.choose_action(_preview_state(), CTX)
    record = agent.log.last
    assert "CODE-RANKED LINEUPS" in provider.views[0] and record["ranking_error"] is None
    assert len(record["code_ranking"]) == 3
    assert action.action["bring"] == roster[2:]  # the model's choice wins over the ranking


def test_preview_falls_back_to_the_top_ranked_lineup():
    agent = _agent(FailingProvider())
    agent.book.remember(list(CATALOG.values()))
    action = agent.choose_action(_preview_state(), CTX)
    bring, leads, _ = agent.log.last["code_ranking"][0]
    assert action["bring"] == bring and action["leads"] == leads


# ---------------------------------------------------------------- A/B test

def _ab_agent(provider, monkeypatch, tmp_path):
    monkeypatch.setattr("agent.pokemon.agent.AB_DIR", tmp_path)
    agent = PokemonAgent(provider, log=NullLog(), book=CardBook(), verbose=False, use_clock=False, reflect=False,
                         ab=True)
    agent.book.remember(list(CATALOG.values()))
    return agent


def _ctx(session_id, seat):
    return DecisionContext(session_id=session_id, tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                           agent_id="me", seat_position=seat)


def test_ab_arms_are_opposite_within_a_game_and_vary_across_games():
    from agent.pokemon.agent import ab_arm

    sessions = [f"session-{n}" for n in range(40)]
    for s in sessions:
        assert {ab_arm(s, 0), ab_arm(s, 1)} == {"ranked", "plain"}
    seat0 = [ab_arm(s, 0) for s in sessions]
    assert 10 < seat0.count("ranked") < 30  # each seat gets both arms
    assert ab_arm("x", None, "a") in ("ranked", "plain")


def test_ab_plain_arm_hides_the_ranking_and_ranked_arm_shows_it(monkeypatch, tmp_path):
    from agent.pokemon.agent import ab_arm

    session = "s-ab"
    for seat in (0, 1):
        provider = FailingProvider()
        agent = _ab_agent(provider, monkeypatch, tmp_path)
        action = agent.choose_action(_draft_state(), _ctx(session, seat))
        arm = ab_arm(session, seat)
        assert agent.arm == arm and agent.log.last["ab_arm"] == arm
        if arm == "ranked":
            assert "CODE-RANKED PICKS" in provider.views[0] and agent.log.last["code_ranking"]
        else:
            assert "CODE-RANKED" not in provider.views[0] and agent.log.last["code_ranking"] == []
            assert action.action_id == _draft_state().legal_actions[0].action_id  # the old fallback


def test_ab_result_row_is_written_at_game_end(monkeypatch, tmp_path):
    agent = _ab_agent(FailingProvider(), monkeypatch, tmp_path)
    ctx = _ctx("s-end", 0)
    agent.choose_action(_draft_state(), ctx)
    agent.on_game_end(SimpleNamespace(final_result={"your_return": 1.0}, termination_reason="completed"), ctx)
    row = __import__("json").loads((tmp_path / "teambuild.jsonl").read_text())
    assert row["session_id"] == "s-end" and row["seat"] == 0 and row["ab_arm"] == agent.arm
    assert row["result"] == "win" and row["termination"] == "completed"
    assert row["drafted_first"] is True and row["fallbacks"] == 1


def test_recorder_forwards_game_end(monkeypatch, tmp_path):
    from tools import recorder

    seen = []
    inner = SimpleNamespace(on_game_end=lambda state, ctx: seen.append(ctx.session_id))
    monkeypatch.setattr(recorder, "LOG_DIR", tmp_path)
    rec = recorder.RecorderAgent.__new__(recorder.RecorderAgent)
    rec.inner_name, rec._inner, rec._file = "x.inner", inner, None
    rec.on_game_end(SimpleNamespace(final_result={"your_return": 0.0}), _ctx("s-rec", 1))
    assert seen == ["s-rec"]
    assert '"game_end"' in next(tmp_path.iterdir()).read_text()


def test_ab_report_pairs_seats_and_tests_against_half():
    from tools.ab_report import binomial_p, games, summary, wilson

    rows = [
        {"session_id": "g1", "seat": 0, "ab_arm": "ranked", "result": "win", "drafted_first": True},
        {"session_id": "g1", "seat": 1, "ab_arm": "plain", "result": "loss", "drafted_first": False},
        {"session_id": "g2", "seat": 0, "ab_arm": "plain", "result": "win", "drafted_first": True},
        {"session_id": "g2", "seat": 1, "ab_arm": "ranked", "result": "loss", "drafted_first": False},
        {"session_id": "g3", "seat": 1, "ab_arm": "plain", "result": "loss", "drafted_first": True},  # one seat only
        {"session_id": "g4", "seat": 0, "ab_arm": "ranked", "result": "win"},
        {"session_id": "g4", "seat": 1, "ab_arm": "plain", "result": "win"},  # inconsistent
        {"session_id": "g5", "seat": 0, "ab_arm": "ranked", "result": "timed out", "termination": "timeout"},
    ]
    entries, problems = games(rows)
    assert [(e["session_id"], e["result"], e["ranked_drafted_first"]) for e in entries] == [
        ("g1", "win", True), ("g2", "loss", False), ("g3", "win", False)]
    assert len(problems) == 2 and "disagree" in problems[0] and "no result" in problems[1]
    assert "ranked won 2 of 3" in summary("All", entries)
    assert binomial_p(5, 10) == 1.0 and binomial_p(10, 10) == pytest.approx(2 / 1024)
    lo, hi = wilson(30, 50)
    assert lo == pytest.approx(0.461, abs=1e-3) and hi == pytest.approx(0.724, abs=1e-3)
