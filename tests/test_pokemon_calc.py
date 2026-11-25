"""Tests for the damage / speed calculator (agent/pokemon/calc.py)."""

from agent.pokemon import dex
from agent.pokemon.calc import Field, Mon, Side, build_mon, damage, parse_log, speed

SETS = {
    "sableye": {"species": "Sableye", "item": "Mental Herb", "ability": "Prankster", "nature": "Careful",
                "evs": {"hp": 252, "atk": 4, "spd": 252}, "moves": ["Foul Play", "Fake Out", "Will-O-Wisp", "Taunt"]},
    "thundurus": {"species": "Thundurus", "item": "Life Orb", "ability": "Defiant", "nature": "Timid",
                  "evs": {"spa": 252, "spd": 4, "spe": 252}, "moves": ["Thunderbolt", "Sludge Bomb", "Taunt", "Protect"]},
    "landorustherian": {"species": "Landorus-Therian", "item": "Choice Scarf", "ability": "Intimidate",
                        "nature": "Jolly", "evs": {"atk": 252, "def": 4, "spe": 252},
                        "moves": ["Earthquake", "Rock Slide", "U-turn", "Protect"]},
    "incineroar": {"species": "Incineroar", "item": "Safety Goggles", "ability": "Intimidate", "nature": "Careful",
                   "evs": {"hp": 252, "atk": 4, "spd": 252}, "moves": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"]},
    "indeedeef": {"species": "Indeedee-F", "item": "Rocky Helmet", "ability": "Psychic Surge", "nature": "Bold",
                  "evs": {"hp": 252, "def": 252, "spd": 4}, "moves": ["Follow Me", "Psychic", "Helping Hand", "Protect"]},
    "chienpao": {"species": "Chien-Pao", "item": "Focus Sash", "ability": "Sword of Ruin", "nature": "Jolly",
                 "evs": {"hp": 4, "atk": 252, "spe": 252}, "moves": ["Icicle Crash", "Sacred Sword", "Sucker Punch", "Protect"]},
}


def mon(key: str, **changes) -> Mon:
    m = build_mon({"species": key}, SETS[key], exact_hp=False)
    m.hp = m.max_hp
    for name, value in changes.items():
        setattr(m, name, value)
    return m


def pct_range(result):
    return 100 * result.lo / result.defender_max, 100 * result.hi / result.defender_max


def test_stats_match_the_servers_max_hp():
    # Max HP the server reported for our own Pokémon in recorded games.
    assert mon("sableye").max_hp == 157
    assert mon("chienpao").max_hp == 156


def test_natures_and_evs():
    stats = dex.calc_stats({"hp": 80, "atk": 120, "def": 80, "spa": 90, "spd": 65, "spe": 135},
                           {"hp": 4, "atk": 252, "spe": 252}, "Jolly")
    assert stats["spe"] == 205 and stats["atk"] == 172 and stats["spa"] == 99


def test_real_hits_from_recorded_battles_fall_in_range():
    # (attacker, defender, move, real % damage, spread, attacker boosts) - read from Showdown battle logs
    cases = [
        ("thundurus", "sableye", "thunderbolt", 69.0, False, {}),
        ("landorustherian", "incineroar", "earthquake", 56.0, True, {"atk": -1}),  # after Incineroar's Intimidate
        ("thundurus", "indeedeef", "thunderbolt", 61.0, False, {}),
    ]
    for attacker, defender, move_id, real, spread, boosts in cases:
        lo, hi = pct_range(damage(mon(attacker, boosts=boosts), mon(defender), move_id, spread=spread))
        assert lo - 1.5 <= real <= hi + 1.5, (attacker, move_id, defender, real, lo, hi)


def test_eviolite_real_hit():
    # Recorded: Iron Hands (boosts reset by a switch) Close Combat -> Eviolite Porygon2: 96% of max HP.
    hands = build_mon({"species": "ironhands"}, {"species": "Iron Hands", "item": "Assault Vest",
                      "ability": "Quark Drive", "nature": "Adamant", "evs": {"hp": 236, "atk": 252, "spe": 20},
                      "moves": ["Fake Out", "Close Combat", "Wild Charge", "Drain Punch"]}, exact_hp=False)
    p2 = build_mon({"species": "porygon2"}, {"species": "Porygon2", "item": "Eviolite", "ability": "Download",
                   "nature": "Quiet", "evs": {"hp": 252, "spa": 252, "spd": 4},
                   "moves": ["Trick Room", "Tri Attack", "Ice Beam", "Recover"]}, exact_hp=False)
    p2.hp = p2.max_hp
    lo, hi = pct_range(damage(hands, p2, "closecombat"))
    assert lo - 1.5 <= 96.0 <= hi + 1.5, (lo, hi)


def test_immunity_and_type_overrides():
    flyer = mon("thundurus")
    assert damage(mon("landorustherian"), flyer, "earthquake").effectiveness == 0
    assert damage(mon("landorustherian"), mon("sableye", ability="levitate"), "earthquake").effectiveness == 0
    milotic_like = mon("sableye", types=["WATER"])
    assert damage(mon("thundurus"), milotic_like, "freezedry").effectiveness == 2


def test_ruination_is_half_current_hp():
    target = mon("incineroar")
    target.hp = 100
    result = damage(mon("chienpao"), target, "ruination")
    assert result.lo == result.hi == 50


def test_focus_sash_survives_at_full_hp_only():
    pao = mon("chienpao")
    boosted = mon("landorustherian", boosts={"atk": 4})  # strong enough to KO from full HP
    hit = damage(boosted, pao, "earthquake")
    assert hit.hi == pao.max_hp - 1 and hit.ko_chance == 0
    pao.hp = pao.max_hp - 1
    assert damage(boosted, pao, "earthquake").ko_chance == 1


def test_spread_screens_and_multiscale_reduce_damage():
    base = damage(mon("landorustherian"), mon("incineroar"), "earthquake")
    spread = damage(mon("landorustherian"), mon("incineroar"), "earthquake", spread=True)
    screened = damage(mon("landorustherian"), mon("incineroar"), "earthquake", None, Side(reflect=True))
    multiscale = damage(mon("landorustherian"), mon("incineroar", ability="multiscale"), "earthquake")
    assert spread.hi < base.hi and screened.hi < base.hi and multiscale.hi < base.hi


def test_sword_of_ruin_lowers_defense_of_others():
    plain = damage(mon("landorustherian"), mon("incineroar"), "earthquake")
    ruined = damage(mon("landorustherian"), mon("incineroar"), "earthquake", Field(ruin={"swordofruin"}))
    assert ruined.lo > plain.lo


def test_body_press_and_foul_play_use_other_stats():
    sableye = mon("sableye")
    # Foul Play uses the TARGET's Attack: raising the target's Attack raises the damage it takes.
    calm = damage(sableye, mon("chienpao"), "foulplay")
    angry = damage(sableye, mon("chienpao", boosts={"atk": 2}), "foulplay")
    assert angry.lo > calm.lo
    # Body Press uses the user's Defense.
    bulky = mon("indeedeef")  # 252 Def Bold
    frail = mon("indeedeef", stats={**bulky.stats, "def": 60})
    assert damage(bulky, mon("chienpao"), "bodypress").hi > damage(frail, mon("chienpao"), "bodypress").hi


def test_speed_modifiers_and_log_parsing():
    lando = mon("landorustherian")
    assert speed(lando, Side(), Field()) == lando.stats["spe"] * 3 // 2  # Choice Scarf
    assert speed(lando, Side(tailwind=True), Field()) > speed(lando, Side(), Field())
    log = ["|switch|p1a: Iron Bundle|Iron Bundle, L50|132/132",
           "|-start|p1a: Iron Bundle|quarkdrivespe|[fromitem]",
           "|switch|p2a: Incineroar|Incineroar, L50|100/100",
           "|move|p2a: Incineroar|Fake Out|p1a: Iron Bundle"]
    boosts, moved = parse_log(log)
    assert boosts[("p1", "Iron Bundle")] == "spe"
    assert moved[("p2", "Incineroar")] is True and moved[("p1", "Iron Bundle")] is False


def test_protect_streak_from_log():
    from agent.pokemon.calc import read_log

    log = ["|switch|p2b: Cresselia|Cresselia, L50|227/227",
           "|move|p2b: Cresselia|Protect|p2b: Cresselia", "|-singleturn|p2b: Cresselia|Protect",
           "|turn|2",
           "|move|p2b: Cresselia|Protect|p2b: Cresselia", "|-singleturn|p2b: Cresselia|Protect",
           "|turn|3"]
    assert read_log(log).protect_streak[("p2", "Cresselia")] == 2
    failed = log + ["|move|p2b: Cresselia|Protect||[still]", "|-fail|p2b: Cresselia"]
    assert read_log(failed).protect_streak[("p2", "Cresselia")] == 0
    attacked = log + ["|move|p2b: Cresselia|Moonblast|p1a: Arcanine"]
    assert read_log(attacked).protect_streak[("p2", "Cresselia")] == 0
