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


def annotate_options(options: dict[int, list[Option]], facts: BattleFacts) -> None:
    """Add damage / KO numbers to each attack option and a score for the fallback."""
    opponents = [m for m in facts.theirs if m and m.hp > 0]
    for slot, slot_options in options.items():
        attacker = facts.mine[slot] if slot < len(facts.mine) else None
        partner = facts.mine[1 - slot] if 1 - slot < len(facts.mine) else None
        if attacker is None:
            continue
        streak = facts.my_protect_streak[slot] if slot < len(facts.my_protect_streak) else 0
        for option in slot_options:
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
                        parts.append(_short(result))
                        score += min(result.avg_pct, 100 * defender.hp / defender.max_hp) + 60 * result.ko_chance
                        option.extra["ko"] = {target: result.ko_chance}
            elif target == 0 and move.get("target") in SPREAD_TARGETS:
                victims = list(opponents)
                hits_partner = move.get("target") == "allAdjacent" and partner is not None and partner.hp > 0
                n_targets = len(victims) + (1 if hits_partner else 0)
                for defender in victims:
                    result = damage(attacker, defender, move_id, facts.field, facts.their_side, spread=n_targets > 1)
                    if result:
                        parts.append(f"{defender.name} {_short(result)}")
                        score += min(result.avg_pct, 100 * defender.hp / defender.max_hp) + 60 * result.ko_chance
                if hits_partner:
                    result = damage(attacker, partner, move_id, facts.field, facts.my_side, spread=n_targets > 1)
                    if result and result.effectiveness > 0:
                        parts.append(f"YOUR {partner.name} takes {_short(result)}")
                        score -= result.avg_pct + 100 * result.ko_chance
            if parts:
                option.text += " => " + "; ".join(parts)
                option.extra["score"] = score


def threat_lines(facts: BattleFacts) -> list[str]:
    """The opponent's best attack against each of our active Pokémon."""
    lines = []
    for index, opp in enumerate(facts.theirs):
        if not opp or opp.hp <= 0:
            continue
        pieces = []
        for mine in facts.mine:
            if not mine or mine.hp <= 0:
                continue
            best = best_attack(opp, mine, facts.field, facts.my_side)
            if best:
                move_id, result = best
                name = (dex.move(move_id) or {}).get("name", move_id)
                pieces.append(f"{name} -> YOUR {mine.name} {result.text()}")
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
