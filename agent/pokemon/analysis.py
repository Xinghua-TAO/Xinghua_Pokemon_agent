"""Code-computed facts for each decision, written as short text for the LLM.

Battle: damage % and KO chance on every attack option, the opponent's best
attack against each of our active Pokémon, and this turn's Speed order.
Team Preview: who out-damages whom across both full teams, and Speed tiers.
Draft: each card's roles and how it matches up with the opponent's picks.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import dex
from .calc import PROTECT_MOVES, SPREAD_TARGETS, Damage, Field, Mon, Side, best_attack, build_mon, damage, \
    field_from_obs, priority, read_log, speed
from .cards import CardBook, species_key
from .options import Option

SPEED_CONTROL = {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "bleakwindstorm", "rocktomb",
                 "scaryface", "glare", "nuzzle", "bulldoze"}
REDIRECTION = {"followme", "ragepowder"}
SUPPORT = {"helpinghand", "reflect", "lightscreen", "auroraveil", "spore", "willowisp", "taunt", "encore",
           "partingshot", "pollenpuff", "wideguard", "quickguard", "haze", "clearsmog", "yawn", "sleeppowder"}
SETUP = {"nastyplot", "swordsdance", "dragondance", "calmmind", "bulkup", "quiverdance", "shellsmash", "coil"}


def _other(side: str | None) -> str | None:
    return {"p1": "p2", "p2": "p1"}.get(side or "")


@dataclass
class BattleFacts:
    mine: list[Mon | None]
    theirs: list[Mon | None]
    field: Field
    my_side: Side
    their_side: Side
    my_moved: list[bool | None]  # per slot: has it moved since coming in (None = unknown)
    their_moved: list[bool | None]
    my_protect_streak: list[int]  # per slot: successful Protects in a row
    their_protect_streak: list[int]


def battle_facts(obs: dict, book: CardBook, my_side_id: str | None) -> BattleFacts:
    log_facts = read_log(obs.get("protocol_log") or [])
    boosts, moved = log_facts.boosts, log_facts.moved
    their_id = _other(my_side_id)

    def build(entries, exact: bool, side_id):
        mons, flags = [], []
        for entry in (entries or [None, None])[:2]:
            mon = build_mon(entry, book.by_species((entry or {}).get("species")), exact_hp=exact) if entry else None
            if mon:
                mon.active_boost = boosts.get((side_id, mon.name))
            mons.append(mon)
            flags.append(moved.get((side_id, mon.name)) if mon else None)
        while len(mons) < 2:
            mons.append(None)
            flags.append(None)
        return mons, flags

    mine, my_moved = build(obs.get("active_pokemon"), True, my_side_id)
    theirs, their_moved = build(obs.get("opponent_active_pokemon"), False, their_id)
    abilities = [m.ability for m in mine + theirs if m]
    fld, my_side, their_side = field_from_obs(obs, abilities)
    streaks = [log_facts.protect_streak.get((my_side_id, m.name), 0) if m else 0 for m in mine]
    their_streaks = [log_facts.protect_streak.get((their_id, m.name), 0) if m else 0 for m in theirs]
    return BattleFacts(mine, theirs, fld, my_side, their_side, my_moved, their_moved, streaks, their_streaks)


def _short(result: Damage) -> str:
    return result.text()


def order_note(attacker: Mon, attacker_side: Side, move_id: str, defender: Mon, defender_side: Side,
               fld: Field) -> tuple[str, bool | None]:
    """Does ``attacker`` using ``move_id`` act before ``defender`` (assuming the
    defender uses a normal-priority move)? Returns (text, first?)."""
    pri = priority(attacker, move_id, fld)
    if pri > 0:
        return f"priority {pri:+d}: hits before {defender.name} unless it uses priority too", True
    if pri < 0:
        return f"priority {pri:+d}: hits after {defender.name}", False
    mine, theirs = speed(attacker, attacker_side, fld), speed(defender, defender_side, fld)
    if mine == theirs:
        return f"speed tie with {defender.name}", None
    first = (mine > theirs) != fld.trick_room
    return (f"moves before {defender.name}" if first else f"{defender.name} moves first"), first


# ------------------------------------------------------------------ field effects
# The server reports active effects as {KEY: turn it was set} (layers for
# Spikes). Durations measured in our recorded battles: Tailwind set on turn T
# is active through T+3; screens and Trick Room through T+4; weather/terrain
# from an ability at the start (turn 0) through turn 5.

EFFECT_NAMES = {
    "TAILWIND": "Tailwind", "REFLECT": "Reflect", "LIGHT_SCREEN": "Light Screen", "AURORA_VEIL": "Aurora Veil",
    "SAFEGUARD": "Safeguard", "MIST": "Mist", "SPIKES": "Spikes", "TOXIC_SPIKES": "Toxic Spikes",
    "STEALTH_ROCK": "Stealth Rock", "STICKY_WEB": "Sticky Web", "TRICK_ROOM": "Trick Room", "GRAVITY": "Gravity",
    "MAGIC_ROOM": "Magic Room", "WONDER_ROOM": "Wonder Room",
    "GRASSY_TERRAIN": "Grassy Terrain", "ELECTRIC_TERRAIN": "Electric Terrain", "PSYCHIC_TERRAIN": "Psychic Terrain",
    "MISTY_TERRAIN": "Misty Terrain", "SUNNYDAY": "Sun", "RAINDANCE": "Rain", "SANDSTORM": "Sandstorm",
    "SNOW": "Snow", "SNOWSCAPE": "Snow", "HAIL": "Snow", "DESOLATELAND": "Harsh sun", "PRIMORDIALSEA": "Heavy rain",
    "DELTASTREAM": "Strong winds",
}
_EXACT_DURATION = {"TAILWIND": 4, "REFLECT": 5, "LIGHT_SCREEN": 5, "AURORA_VEIL": 5, "SAFEGUARD": 5, "MIST": 5,
                   "TRICK_ROOM": 5, "GRAVITY": 5, "MAGIC_ROOM": 5, "WONDER_ROOM": 5}  # only ever set by a move
_WEATHER_TERRAIN = {"GRASSY_TERRAIN", "ELECTRIC_TERRAIN", "PSYCHIC_TERRAIN", "MISTY_TERRAIN", "SUNNYDAY",
                    "RAINDANCE", "SANDSTORM", "SNOW", "SNOWSCAPE", "HAIL"}  # 5 turns; set by a move or an ability
_EXTENDER = {"REFLECT": "lightclay", "LIGHT_SCREEN": "lightclay", "AURORA_VEIL": "lightclay",
             "GRASSY_TERRAIN": "terrainextender", "ELECTRIC_TERRAIN": "terrainextender",
             "PSYCHIC_TERRAIN": "terrainextender", "MISTY_TERRAIN": "terrainextender", "SUNNYDAY": "heatrock",
             "RAINDANCE": "damprock", "SANDSTORM": "smoothrock", "SNOW": "icyrock", "SNOWSCAPE": "icyrock",
             "HAIL": "icyrock"}
_ITEM_NAMES = {"lightclay": "Light Clay", "terrainextender": "Terrain Extender", "heatrock": "Heat Rock",
               "damprock": "Damp Rock", "smoothrock": "Smooth Rock", "icyrock": "Icy Rock"}
_LAYERS = {"SPIKES", "TOXIC_SPIKES"}


def effect_text(key: str, value, turn: int, extender_items: set[str] = frozenset()) -> str:
    """One active effect, e.g. "Tailwind (set turn 1; last turn 4: 3 turns left incl. this one)"."""
    key = str(key).upper()
    name = EFFECT_NAMES.get(key, key.replace("_", " ").title())
    if key in _LAYERS:
        return f"{name} x{value}" if isinstance(value, int) else name
    if not isinstance(value, int) or (key not in _EXACT_DURATION and key not in _WEATHER_TERRAIN):
        return name
    if key in _EXACT_DURATION:
        last, approx = value + _EXACT_DURATION[key] - 1, ""
    else:  # an ability at the start lasts through turn 5; set later, through T+4 or T+5
        last, approx = (5, "") if value == 0 else (value + 4, "about ")
    left = last - turn + 1
    when = f"{left} turn(s) left incl. this one" if left > 0 else "ends this turn"
    text = f"{name} (set turn {value}; last turn {approx}{last}: {when})"
    item = _EXTENDER.get(key)
    if item and item in extender_items:
        text += f" (3 turns longer if set by the {_ITEM_NAMES[item]} holder)"
    return text


def field_lines(obs: dict, my_items: set[str] = frozenset(), their_items: set[str] = frozenset()) -> list[str]:
    turn = obs.get("turn") if isinstance(obs.get("turn"), int) else 0
    both = set(my_items) | set(their_items)
    shared = [effect_text(k, v, turn, both) for k, v in {**(obs.get("weather") or {}), **(obs.get("fields") or {})}.items()]
    mine = [effect_text(k, v, turn, set(my_items)) for k, v in (obs.get("side_conditions") or {}).items()]
    theirs = [effect_text(k, v, turn, set(their_items)) for k, v in (obs.get("opponent_side_conditions") or {}).items()]
    lines = []
    if shared:
        lines.append("FIELD: " + "; ".join(shared))
    if mine:
        lines.append("YOUR SIDE: " + "; ".join(mine))
    if theirs:
        lines.append("THEIR SIDE: " + "; ".join(theirs))
    return lines


_SCREENS = {"reflect": ("reflect", "Reflect"), "lightscreen": ("light_screen", "Light Screen"),
            "auroraveil": ("aurora_veil", "Aurora Veil")}
_WEATHER_MOVES = {"sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "snowscape": "snow", "hail": "snow"}
_TERRAIN_MOVES = {"grassyterrain": "grassy", "electricterrain": "electric", "psychicterrain": "psychic",
                  "mistyterrain": "misty"}


def failing_reason(move_id: str, facts: BattleFacts) -> str | None:
    """Why this status move would do nothing right now, or None."""
    if move_id == "tailwind" and facts.my_side.tailwind:
        return "Tailwind is already active on your side (it cannot be refreshed)"
    if move_id in _SCREENS and getattr(facts.my_side, _SCREENS[move_id][0]):
        return f"{_SCREENS[move_id][1]} is already up on your side"
    if move_id == "auroraveil" and facts.field.weather != "snow":
        return "Aurora Veil only works in snow"
    if _WEATHER_MOVES.get(move_id) and facts.field.weather == _WEATHER_MOVES[move_id]:
        return "that weather is already active"
    if _TERRAIN_MOVES.get(move_id) and facts.field.terrain == _TERRAIN_MOVES[move_id]:
        return "that terrain is already active"
    return None


def annotate_options(options: dict[int, list[Option]], facts: BattleFacts) -> None:
    """Add damage / KO numbers and move order to each attack option, and a
    score for the fallback."""
    opponents = [m for m in facts.theirs if m and m.hp > 0]
    for slot, slot_options in options.items():
        attacker = facts.mine[slot] if slot < len(facts.mine) else None
        partner = facts.mine[1 - slot] if 1 - slot < len(facts.mine) else None
        if attacker is None:
            continue
        streak = facts.my_protect_streak[slot] if slot < len(facts.my_protect_streak) else 0
        for option in slot_options:
            if option.kind == "move" and (reason := failing_reason(option.payload["move_id"], facts)):
                option.text += f" => WARNING: fails, {reason}"
                option.extra["score"] = -10.0
                continue
            if option.kind == "move" and option.payload["move_id"] == "trickroom" and facts.field.trick_room:
                option.text += (" => NOTE: Trick Room is already active; using it again ENDS it "
                                "(good only if Trick Room helps the opponent)")
            if option.kind == "move" and option.payload["move_id"] in PROTECT_MOVES and streak > 0:
                chance = round(100 / 3 ** streak)
                option.text += (f" => WARNING: used successfully {streak} turn(s) in a row, "
                                f"only {chance}% chance to work now")
                option.extra["score"] = -10.0
                continue
            if option.kind != "move" or not option.damaging:
                continue
            move_id = option.payload["move_id"]
            move = dex.move(move_id) or {}
            target = option.payload.get("target", 0)
            parts, score = [], 0.0
            if target and target > 0:
                defender = facts.theirs[target - 1] if target - 1 < len(facts.theirs) else None
                if defender:
                    result = damage(attacker, defender, move_id, facts.field, facts.their_side)
                    if result:
                        order, _ = order_note(attacker, facts.my_side, move_id, defender, facts.their_side, facts.field)
                        parts.append(f"{_short(result)}, {order}")
                        score += min(result.avg_pct, 100 * defender.hp / defender.max_hp) + 60 * result.ko_chance
                        option.extra["ko"] = {target: result.ko_chance}
                        option.extra["dmg"] = {target: (result.lo, result.hi, defender.hp, defender.name)}
            elif target == 0 and move.get("target") in SPREAD_TARGETS:
                victims = list(opponents)
                hits_partner = move.get("target") == "allAdjacent" and partner is not None and partner.hp > 0
                n_targets = len(victims) + (1 if hits_partner else 0)
                for defender in victims:
                    result = damage(attacker, defender, move_id, facts.field, facts.their_side, spread=n_targets > 1)
                    if result:
                        parts.append(f"{defender.name} {_short(result)}")
                        score += min(result.avg_pct, 100 * defender.hp / defender.max_hp) + 60 * result.ko_chance
                        index = next((i + 1 for i, m in enumerate(facts.theirs) if m is defender), None)
                        if index:
                            option.extra.setdefault("dmg", {})[index] = (result.lo, result.hi, defender.hp, defender.name)
                if hits_partner:
                    result = damage(attacker, partner, move_id, facts.field, facts.my_side, spread=n_targets > 1)
                    if result and result.effectiveness > 0:
                        parts.append(f"YOUR {partner.name} takes {_short(result)}")
                        score -= result.avg_pct + 100 * result.ko_chance
            if parts:
                option.text += " => " + "; ".join(parts)
                option.extra["score"] = score


def double_target_lines(options: dict[int, list[Option]], limit: int = 4) -> list[str]:
    """Pairs of attacks (one per slot) that KO a target only together."""
    found = []
    single_ko = {target for slot_options in options.values() for o in slot_options
                 for target, (lo, _hi, hp, _name) in (o.extra.get("dmg") or {}).items() if hp > 0 and lo >= hp}
    for first in options.get(0, []):
        for second in options.get(1, []):
            for target, (lo1, hi1, hp, name) in (first.extra.get("dmg") or {}).items():
                other = (second.extra.get("dmg") or {}).get(target)
                if not other or hp <= 0 or target in single_ko:
                    continue  # some single attack already KOs it
                lo2, hi2 = other[0], other[1]
                if lo1 >= hp or lo2 >= hp:
                    continue  # one alone already KOs: no need to double up
                if lo1 + lo2 >= hp:
                    found.append((0, f"{first.id} + {second.id} on {name} (sure KO together)"))
                elif hi1 + hi2 >= hp:
                    found.append((1, f"{first.id} + {second.id} on {name} (KO only with high rolls)"))
    found.sort()
    if not found:
        return []
    return ["DOUBLE-TARGET KOs (both slots on one target; neither KOs it alone): "
            + "; ".join(text for _, text in found[:limit])]


def threat_lines(facts: BattleFacts) -> list[str]:
    """The opponent's best attack against each of our active Pokémon, and
    whether it lands before our Pokémon can act."""
    lines = []
    for index, opp in enumerate(facts.theirs):
        if not opp or opp.hp <= 0:
            continue
        pieces = []
        for slot, mine in enumerate(facts.mine):
            if not mine or mine.hp <= 0:
                continue
            best = best_attack(opp, mine, facts.field, facts.my_side)
            if best:
                move_id, result = best
                name = (dex.move(move_id) or {}).get("name", move_id)
                _, opp_first = order_note(opp, facts.their_side, move_id, mine, facts.my_side, facts.field)
                when = "hits before YOUR " + mine.name + " moves" if opp_first else \
                    ("speed tie" if opp_first is None else "after YOUR " + mine.name + " moves")
                danger = "DANGER: " if opp_first and result.ko_chance >= 0.5 else ""
                pieces.append(f"{danger}{name} -> YOUR {mine.name} {result.text()}, {when}")
        extras = []
        if "fakeout" in opp.moves and facts.their_moved[index] is False:
            extras.append("can still Fake Out")
        fast = [(dex.move(m) or {}).get("name", m) for m in opp.moves
                if (dex.move(m) or {}).get("cat") in ("Physical", "Special") and priority(opp, m, facts.field) > 0]
        if fast:
            extras.append("priority: " + ", ".join(fast))
        streak = facts.their_protect_streak[index] if index < len(facts.their_protect_streak) else 0
        if streak:
            extras.append(f"Protected last turn: its next Protect works only {round(100 / 3 ** streak)}%")
        set_note = "" if opp.known_set else " (set unknown; assumed)"
        lines.append(f"  {opp.name}{set_note}: " + ("; ".join(pieces) or "no damaging moves")
                     + (f" [{'; '.join(extras)}]" if extras else ""))
    return lines


def speed_line(facts: BattleFacts) -> str:
    entries = []
    for mine in facts.mine:
        if mine and mine.hp > 0:
            entries.append((speed(mine, facts.my_side, facts.field), f"YOUR {mine.name}"))
    for opp in facts.theirs:
        if opp and opp.hp > 0:
            entries.append((speed(opp, facts.their_side, facts.field), f"OPP {opp.name}"))
    entries.sort(key=lambda e: e[0], reverse=not facts.field.trick_room)
    order = " > ".join(f"{name} ({value})" for value, name in entries)
    room = " TRICK ROOM is up: slowest moves first." if facts.field.trick_room else ""
    return f"Speed order this turn (same priority):{room} {order}"


def fresh_from_log(facts: BattleFacts) -> list[bool | None]:
    """Per our slot: True if it has not moved since coming in (Fake Out works)."""
    return [None if moved is None else not moved for moved in facts.my_moved]


def assume_booster(mon: Mon) -> None:
    """Before battle: Booster Energy will boost the holder's highest stat."""
    if mon.item == "boosterenergy" and mon.ability in ("quarkdrive", "protosynthesis"):
        mon.active_boost = max(("atk", "def", "spa", "spd", "spe"), key=lambda s: mon.stats[s])


def speed_label(mon: Mon) -> str:
    """'157' or '235 (Choice Scarf)' - the Speed it will actually have in battle."""
    value = speed(mon, Side(), Field())
    if mon.item == "choicescarf":
        return f"{value} (Choice Scarf)"
    if mon.active_boost == "spe":
        return f"{value} (Booster Energy)"
    return str(value)


# ------------------------------------------------------------------ team preview

def preview_lines(obs: dict, book: CardBook) -> list[str]:
    def roster(entries):
        mons = []
        for entry in entries or []:
            mon = build_mon(entry, book.by_species(entry.get("species")), exact_hp=False)
            if mon:
                mon.hp = mon.max_hp
                assume_booster(mon)
                mons.append(mon)
        return mons

    mine, theirs = roster(obs.get("your_roster")), roster(obs.get("opponent_roster"))
    if not mine or not theirs:
        return []

    def matrix(attackers: list[Mon], defenders: list[Mon], prefix: str) -> list[str]:
        out = []
        for attacker in attackers:
            fld = Field(ruin={attacker.ability} if attacker.ability in ("swordofruin", "beadsofruin",
                                                                       "tabletsofruin", "vesselofruin") else set())
            cells = []
            for defender in defenders:
                best = best_attack(attacker, defender, fld)
                if best:
                    move_id, result = best
                    name = (dex.move(move_id) or {}).get("name", move_id)
                    cells.append(f"{defender.name} {round(result.avg_pct)}% ({name})")
            out.append(f"  {prefix}{attacker.name}: " + (", ".join(cells) or "no damaging moves"))
        return out

    tiers = sorted(mine + theirs, key=lambda m: speed(m, Side(), Field()), reverse=True)
    lines = ["YOUR BEST HIT ON EACH OF THEIRS (avg % of their max HP, full HP, no boosts):"]
    lines += matrix(mine, theirs, "")
    lines.append("THEIR BEST HIT ON EACH OF YOURS:")
    lines += matrix(theirs, mine, "")
    lines.append("SPEED TIERS (battle Speed incl. Choice Scarf / Booster Energy, before Tailwind or Trick Room): "
                 + " > ".join(f"{'YOUR' if m in mine else 'OPP'} {m.name} {speed_label(m)}" for m in tiers))
    return lines


# ------------------------------------------------------------------ draft

def card_roles(card: dict) -> list[str]:
    move_ids = {species_key(m) for m in card.get("moves") or []}
    roles = []
    if move_ids & SPEED_CONTROL:
        roles.append("speed control (" + ", ".join(sorted(move_ids & SPEED_CONTROL)) + ")")
    if "fakeout" in move_ids:
        roles.append("Fake Out")
    if move_ids & REDIRECTION:
        roles.append("redirection")
    if species_key(card.get("ability")) == "intimidate":
        roles.append("Intimidate")
    spreads = [m for m in move_ids if (dex.move(m) or {}).get("target") in SPREAD_TARGETS
               and (dex.move(m) or {}).get("cat") != "Status"]
    if spreads:
        roles.append("spread attack (" + ", ".join(sorted(spreads)) + ")")
    if move_ids & SUPPORT:
        roles.append("support (" + ", ".join(sorted(move_ids & SUPPORT)) + ")")
    if move_ids & SETUP:
        roles.append("setup")
    if "protect" in move_ids:
        roles.append("Protect")
    return roles


def card_mon(card: dict) -> Mon | None:
    mon = build_mon({"species": card.get("species")}, card, exact_hp=False)
    if mon:
        mon.hp = mon.max_hp
        assume_booster(mon)
    return mon


def draft_card_summary(card: dict, opponent_cards: list[dict]) -> str:
    mon = card_mon(card)
    if mon is None:
        return ""
    types = "/".join(t.title() for t in mon.types)
    roles = ", ".join(card_roles(card)) or "pure attacker"
    text = f"[{types}; Speed {speed_label(mon)}; roles: {roles}]"
    if opponent_cards:
        threatens, threatened = [], []
        for opp_card in opponent_cards:
            opp = card_mon(opp_card)
            if not opp:
                continue
            mine_best = best_attack(mon, opp)
            if mine_best and mine_best[1].avg_pct >= 50:
                threatens.append(f"{opp.name} {round(mine_best[1].avg_pct)}%")
            their_best = best_attack(opp, mon)
            if their_best and their_best[1].avg_pct >= 50:
                threatened.append(f"{opp.name} {round(their_best[1].avg_pct)}%")
        if threatens:
            text += f" hits hard: {', '.join(threatens)}."
        if threatened:
            text += f" hit hard by: {', '.join(threatened)}."
    return text
