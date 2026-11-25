"""Tests for our Pokémon agent: the code guardrails and the decision loop.

No network: the LLM is a fake that returns scripted answers.
"""

import json

from altruagent import DecisionContext, GameState, WithReasoning
from agent.agent import GameRouter, create_agent
from agent.pokemon.agent import PokemonAgent
from agent.pokemon.cards import CardBook, format_card, species_key
from agent.pokemon.options import build_slot_options, combo_error, fallback_choice, partner_immune
from examples.llm.providers import ProviderError

LANDO = {"species": "landorustherian", "name": "Landorus", "types": ["GROUND", "FLYING"],
         "ability": "intimidate", "item": "choicescarf", "current_hp_fraction": 1.0}
TINGLU = {"species": "tinglu", "name": "Ting-Lu", "types": ["DARK", "GROUND"],
          "ability": "vesselofruin", "item": "leftovers", "current_hp_fraction": 0.75}


def _move(move_id, targets, *, bp=80, category="PHYSICAL", move_type="NORMAL", who=None):
    who = who or {-2: ("ally", "tinglu"), -1: ("ally", "self"), 1: ("opponent", "amoonguss"), 2: ("opponent", "chienpao")}
    return {
        "type": "move", "move_id": move_id, "base_power": bp, "category": category, "move_type": move_type,
        "accuracy": 1.0, "targets": targets,
        "target_options": [{"target": t, "side": who[t][0], "species": who[t][1]} for t in targets if t in who],
    }


def _slot(options, *, slot=0, active=None):
    return {"slot": slot, "board_position": -1 - slot, "active": active, "force_switch": False, "options": options}


def _options(slot, info=None, *, fresh=True, partner=None, bench=None):
    return build_slot_options(slot, info or [], letter="a", fresh=fresh, partner=partner, bench=bench or {})


# ---------------------------------------------------------------- guardrails

def test_attacks_never_target_own_partner():
    opts = _options(_slot([_move("flareblitz", [-2, 1, 2])]))
    assert [o.payload["target"] for o in opts] == [1, 2]


def test_ally_support_moves_keep_their_ally_target():
    opts = _options(_slot([_move("helpinghand", [-2], bp=0, category="STATUS"),
                           _move("pollenpuff", [-2, 1, 2], bp=90, category="SPECIAL")]))
    targets = [(o.payload["move_id"], o.payload["target"]) for o in opts]
    assert ("helpinghand", -2) in targets
    assert ("pollenpuff", -2) in targets and ("pollenpuff", 1) in targets


def test_fake_out_only_on_first_turn():
    slot = _slot([_move("fakeout", [-2, 1, 2], bp=40), _move("knockoff", [-2, 1, 2], bp=65)])
    assert {o.payload["move_id"] for o in _options(slot, fresh=True)} == {"fakeout", "knockoff"}
    assert {o.payload["move_id"] for o in _options(slot, fresh=False)} == {"knockoff"}


def test_no_target_moves_send_target_zero():
    opts = _options(_slot([_move("spikes", [0], bp=0, category="STATUS")]),
                    [{"id": "spikes", "target": "FOE_SIDE", "priority": 0}])
    assert opts[0].payload == {"type": "move", "move_id": "spikes", "target": 0}
    assert "opponent's side" in opts[0].text


def test_spread_move_hitting_partner_is_flagged_unless_partner_is_immune():
    eq = _slot([_move("earthquake", [0], bp=100, move_type="GROUND")])
    info = [{"id": "earthquake", "target": "ALL_ADJACENT", "priority": 0}]
    hits = _options(eq, info, partner=TINGLU)[0]
    assert hits.hits_ally and "ALSO HITS YOUR PARTNER" in hits.text
    safe = _options(eq, info, partner=LANDO)[0]
    assert not safe.hits_ally and "immune" in safe.text


def test_partner_immunity_by_type_ability_and_item():
    assert partner_immune("GROUND", {"types": ["FIRE", "FLYING"]})
    assert partner_immune("GROUND", {"types": ["PSYCHIC"], "ability": "levitate"})
    assert partner_immune("GROUND", {"types": ["STEEL"], "item": "airballoon"})
    assert partner_immune("WATER", {"types": ["GRASS"], "ability": "stormdrain"})
    assert not partner_immune("GROUND", {"types": ["DARK", "GROUND"]})


def test_combo_rules():
    a = _options(_slot([{"type": "switch", "species": "milotic"}, {"type": "pass"}]))
    switch, pass_ = a[0], a[1]
    assert combo_error(switch, switch) and "same" in combo_error(switch, switch)
    assert combo_error(pass_, pass_)
    assert combo_error(switch, pass_) is None


def test_fallback_prefers_strong_attack_and_valid_combo():
    s0 = _options(_slot([_move("tackle", [1, 2], bp=40), _move("closecombat", [1, 2], bp=120),
                         _move("protect", [0], bp=0, category="STATUS")]))
    s1 = build_slot_options(_slot([{"type": "switch", "species": "milotic"}], slot=1), [], letter="b",
                            fresh=True, partner=None, bench={})
    first, second = fallback_choice(s0, s1)
    assert first.payload["move_id"] == "closecombat" and second.kind == "switch"


# ---------------------------------------------------------------- cards

def test_card_book_lookup_and_format(tmp_path):
    book = CardBook(tmp_path / "catalog.json")
    card = {"card_id": "vgc-landorus-therian", "species": "Landorus-Therian", "item": "Choice Scarf",
            "ability": "Intimidate", "nature": "Jolly", "evs": {"atk": 252, "spe": 252},
            "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]}
    book.remember([card])
    assert species_key("Landorus-Therian") == "landorustherian"
    assert book.by_species("landorustherian") == card
    assert CardBook(tmp_path / "catalog.json").by_id("vgc-landorus-therian") == card  # persisted
    assert "252 Atk / 252 Spe" in format_card(card)


# ---------------------------------------------------------------- decision loop

class FakeProvider:
    model = "fake"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.calls.append((schema_name, schema))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class NullLog:
    def write(self, context, record):
        self.last = record


def _doubles_state():
    slot0 = _slot([_move("flareblitz", [-2, 1, 2], bp=120, move_type="FIRE"),
                   {"type": "switch", "species": "milotic"}], slot=0, active=TINGLU)
    slot1 = _slot([_move("iciclecrash", [-1, 1, 2], bp=85, move_type="ICE"),
                   {"type": "switch", "species": "milotic"}], slot=1, active=LANDO)
    raw = {
        "session_id": "s", "game_type": "pokemon_vgc_doubles_draft", "status": "in_progress",
        "state_version": 3, "phase": "moving", "is_terminal": False, "is_current_actor": True,
        "observation": {"turn": 2, "active_pokemon": [TINGLU, LANDO], "opponent_active_pokemon": [None, None],
                        "team": {"p1: Milotic": {"species": "milotic", "name": "Milotic", "current_hp_fraction": 1.0}},
                        "available_moves": [[], []], "protocol_log": [], "clock": {"decision_seconds_left": 45}},
        "legal_actions": {"session_id": "s", "state_version": 3, "actions": [{
            "action_id": "doubles_turn", "label": "doubles", "input": {"action": {
                "type": "doubles_turn", "slots": [slot0, slot1], "instructions": "..."}}}]},
    }
    return GameState.from_mcp_state(raw)


CTX = DecisionContext(session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                      agent_id="me", seat_position=0)


def _agent(provider):
    return PokemonAgent(provider, log=NullLog(), book=CardBook(), verbose=False)


def test_battle_uses_model_choice_and_attaches_public_reason():
    provider = FakeProvider([{"analysis": "x", "slot_0": "a1", "slot_1": "b2", "public_reason": "focus"}])
    move = _agent(provider).choose_action(_doubles_state(), CTX)
    assert isinstance(move, WithReasoning) and move.reasoning_summary == "focus"
    assert move.action == {"type": "doubles_turn",
                           "slot_0": {"type": "move", "move_id": "flareblitz", "target": 1},
                           "slot_1": {"type": "move", "move_id": "iciclecrash", "target": 2}}
    schema = provider.calls[0][1]["properties"]
    assert schema["slot_0"]["enum"] == ["a1", "a2", "a3"]  # 2 foe targets + switch, no ally target


def test_battle_retries_invalid_combo_then_accepts_fix():
    provider = FakeProvider([
        {"analysis": "", "slot_0": "a3", "slot_1": "b3", "public_reason": ""},  # both switch to milotic
        {"analysis": "", "slot_0": "a3", "slot_1": "b1", "public_reason": "ok"},
    ])
    move = _agent(provider).choose_action(_doubles_state(), CTX)
    assert len(provider.calls) == 2
    assert move.action["slot_0"] == {"type": "switch", "species": "milotic"}
    assert move.action["slot_1"]["move_id"] == "iciclecrash"


def test_battle_falls_back_when_model_fails():
    provider = FakeProvider([ProviderError("boom"), ProviderError("boom again")])
    move = _agent(provider).choose_action(_doubles_state(), CTX)
    assert move["slot_0"]["move_id"] == "flareblitz" and move["slot_0"]["target"] > 0
    assert move["slot_1"]["move_id"] == "iciclecrash" and move["slot_1"]["target"] > 0


def test_slow_model_retries_with_fast_model_inside_the_budget():
    slow = FakeProvider([ProviderError("OpenAI request failed (ReadTimeout)")])
    slow.model = "slow"
    fast = FakeProvider([{"analysis": "", "slot_0": "a2", "slot_1": "b1", "public_reason": "fast"}])
    fast.model = "fast"
    agent = PokemonAgent(slow, log=NullLog(), book=CardBook(), verbose=False)
    agent.draft_provider = fast
    move = agent.choose_action(_doubles_state(), CTX)
    assert move.reasoning_summary == "fast"
    assert agent.log.last["attempts"][0]["model"] == "slow" and agent.log.last["attempts"][1]["model"] == "fast"


def test_low_time_bank_plays_with_the_fast_model():
    slow = FakeProvider([])  # would fail if called
    fast = FakeProvider([{"analysis": "", "slot_0": "a1", "slot_1": "b1", "public_reason": "quick"}])
    agent = PokemonAgent(slow, log=NullLog(), book=CardBook(), verbose=False)
    agent.draft_provider = fast
    state = _doubles_state()
    state.raw["observation"]["clock"] = {"decision_seconds_left": 45, "bank_seconds_left": 60}
    assert agent.choose_action(state, CTX).reasoning_summary == "quick"
    assert slow.calls == []


# ---------------------------------------------------------------- model config

def test_per_phase_models_from_env(monkeypatch):
    from agent.pokemon.llm import DEFAULT_MODEL, providers_from_env

    for name in ("POKEMON_MODEL", "POKEMON_DRAFT_MODEL", "POKEMON_REASONING_EFFORT", "POKEMON_DRAFT_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert providers_from_env() == (None, None)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    battle, draft = providers_from_env()
    assert battle is draft and battle.model == DEFAULT_MODEL and battle.reasoning_effort is None
    assert "sk-test" not in repr(battle)

    monkeypatch.setenv("POKEMON_MODEL", "gpt-5.5")
    monkeypatch.setenv("POKEMON_DRAFT_REASONING_EFFORT", "none")
    battle, draft = providers_from_env()
    assert (battle.model, battle.reasoning_effort) == ("gpt-5.5", None)
    assert (draft.model, draft.reasoning_effort) == ("gpt-5.5", "none")


def test_reasoning_effort_is_sent_only_when_set():
    import httpx

    from agent.pokemon.llm import ChatProvider

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"pick": "x"}'}}]})

    for effort in (None, "low"):
        provider = ChatProvider("sk-test", "gpt-5.5", reasoning_effort=effort, transport=httpx.MockTransport(handler))
        assert provider.complete_structured([], "s", {}) == {"pick": "x"}
    assert "reasoning_effort" not in seen[0] and seen[1]["reasoning_effort"] == "low"


# ---------------------------------------------------------------- router

def test_router_sends_pokemon_to_our_agent_and_others_to_placeholder(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("agent.pokemon.agent.LOG_DIR", tmp_path)  # keep test traces out of logs/
    router = create_agent()
    assert isinstance(router, GameRouter)
    werewolf_ctx = DecisionContext(session_id="w", tournament_id=None, game_type="werewolf", agent_id="me")
    werewolf_state = GameState.from_mcp_state({
        "session_id": "w", "phase": "moving", "state_version": 1, "is_current_actor": True,
        "legal_actions": {"actions": [{"action_id": "3", "label": "vote 3", "input": {}}]}})
    assert router.choose_action(werewolf_state, werewolf_ctx).action_id == "3"

    pokemon_router = create_agent()
    move = pokemon_router.choose_action(_doubles_state(), CTX)  # no key: code fallback, still valid
    assert move["type"] == "doubles_turn"
    assert isinstance(pokemon_router._agent, PokemonAgent)
