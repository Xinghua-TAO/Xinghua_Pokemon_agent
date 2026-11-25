"""Tests for code-ranked team building (agent/pokemon/teambuild.py) and how
the agent uses it at the draft and Team Preview. No network."""

from itertools import combinations

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
    agent = PokemonAgent(provider, log=NullLog(), book=CardBook(), verbose=False, use_clock=False, reflect=False)
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
