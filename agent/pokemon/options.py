"""Turn the server's doubles_turn template into a numbered menu per slot.

Code, not the LLM, owns the mechanics. Every option on the menu is a complete
payload the server accepts; the LLM only chooses option ids. While building
the menu, code removes choices that are legal but almost always wrong:

- attacking your own partner (unless the move is meant for allies);
- first-turn-only moves (Fake Out, First Impression, Mat Block) after the
  Pokémon's first turn on the field, when they always fail;

and it fills in what the baseline model kept getting wrong, such as target 0
for moves that take no target. Combinations the server rejects (both slots
switching into the same Pokémon, both passing) are checked separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FIRST_TURN_ONLY = frozenset({"fakeout", "firstimpression", "matblock"})

# Moves that are meant to be aimed at your own partner.
ALLY_TARGET_OK = frozenset({
    "pollenpuff", "healpulse", "floralhealing", "decorate", "instruct",
    "acupressure", "helpinghand", "aromaticmist", "coaching",
})

# Who is unaffected by a move of a given type (for spread moves that also
# hit the partner): by type, by ability, by item.
_TYPE_IMMUNE = {
    "GROUND": {"FLYING"}, "ELECTRIC": {"GROUND"}, "NORMAL": {"GHOST"}, "FIGHTING": {"GHOST"},
    "POISON": {"STEEL"}, "PSYCHIC": {"DARK"}, "GHOST": {"NORMAL"}, "DRAGON": {"FAIRY"},
}
_ABILITY_IMMUNE = {
    "GROUND": {"levitate", "eartheater"},
    "WATER": {"waterabsorb", "stormdrain", "dryskin"},
    "ELECTRIC": {"voltabsorb", "lightningrod", "motordrive"},
    "FIRE": {"flashfire", "wellbakedbody"},
    "GRASS": {"sapsipper"},
}
_ITEM_IMMUNE = {"GROUND": {"airballoon"}}


def partner_immune(move_type: str | None, partner: dict | None) -> bool:
    """True if the partner takes no damage from a move of this type."""
    if not partner or not move_type:
        return False
    move_type = move_type.upper()
    types = {t.upper() for t in partner.get("types") or []}
    ability = (partner.get("ability") or "").lower().replace(" ", "")
    item = (partner.get("item") or "").lower().replace(" ", "")
    return bool(
        types & _TYPE_IMMUNE.get(move_type, set())
        or ability in _ABILITY_IMMUNE.get(move_type, set())
        or item in _ITEM_IMMUNE.get(move_type, set())
    )


_NO_TARGET_TEXT = {
    "SELF": "self",
    "ALL_ADJACENT_FOES": "both opponents (spread)",
    "ALL_ADJACENT": "everyone adjacent (spread)",
    "FOE_SIDE": "the opponent's side",
    "ALLY_SIDE": "your side",
    "ALL": "the whole field",
    "ALLIES": "you and your partner",
    "ADJACENT_ALLY": "your partner",
    "RANDOM_NORMAL": "a random opponent",
}


@dataclass
class Option:
    id: str
    kind: str  # "move" | "switch" | "pass"
    payload: dict
    text: str
    base_power: int = 0
    damaging: bool = False
    target: int | None = None
    hits_ally: bool = False
    spread: bool = False
    extra: dict = field(default_factory=dict)


def _pct(entry: dict | None) -> str:
    if not entry:
        return "?"
    frac = entry.get("current_hp_fraction")
    if isinstance(frac, (int, float)):
        return f"{round(frac * 100)}%"
    return "?"


def build_slot_options(
    slot: dict,
    move_info: list[dict] | None,
    *,
    letter: str,
    fresh: bool,
    partner: dict | None,
    bench: dict[str, dict],
) -> list[Option]:
    """Options for one slot. ``move_info`` is observation.available_moves[slot]
    (it carries each move's target kind and priority); ``partner`` is the other
    slot's active Pokémon entry (None if empty); ``bench`` maps species id ->
    team entry for switch descriptions."""
    ally_species = (partner or {}).get("name") or (partner or {}).get("species")
    own_position = slot.get("board_position")
    info = {m.get("id"): m for m in (move_info or []) if isinstance(m, dict)}
    raw = [o for o in (slot.get("options") or []) if isinstance(o, dict)]

    moves = [o for o in raw if o.get("type") == "move" and o.get("move_id")]
    usable = [m for m in moves if fresh or m["move_id"] not in FIRST_TURN_ONLY] or moves

    options: list[Option] = []

    def add(kind: str, payload: dict, text: str, **kw) -> None:
        options.append(Option(id=f"{letter}{len(options) + 1}", kind=kind, payload=payload, text=text, **kw))

    for move in usable:
        move_id = move["move_id"]
        meta = info.get(move_id, {})
        target_kind = meta.get("target") or ""
        targets = [t for t in (move.get("targets") or []) if isinstance(t, int) and not isinstance(t, bool)] or [0]
        has_foe_target = any(t > 0 for t in targets)
        who = {
            t.get("target"): (t.get("side"), t.get("species"))
            for t in (move.get("target_options") or [])
            if isinstance(t, dict)
        }
        base_power = int(move.get("base_power") or 0)
        category = (move.get("category") or "").upper()
        damaging = category in ("PHYSICAL", "SPECIAL") and base_power > 0
        details = [move.get("move_type") or "?", category.lower() or "?"]
        if base_power:
            details.append(f"{base_power} BP")
        priority = meta.get("priority")
        if isinstance(priority, int) and priority:
            details.append(f"priority {priority:+d}")
        accuracy = move.get("accuracy")
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and accuracy < 1:
            details.append(f"{round(accuracy * 100)}% acc")

        for target in targets:
            if target < 0 and has_foe_target and move_id not in ALLY_TARGET_OK:
                continue  # never aim an attack or a debuff at our own side
            hits_ally = False
            spread = False
            if target == 0:
                where = _NO_TARGET_TEXT.get(target_kind, "no target needed")
                spread = target_kind in ("ALL_ADJACENT_FOES", "ALL_ADJACENT")
                if target_kind == "ALL_ADJACENT" and ally_species:
                    if partner_immune(move.get("move_type"), partner):
                        where += f" - your partner {ally_species} is immune"
                    else:
                        hits_ally = True
                        where += f" - ALSO HITS YOUR PARTNER {ally_species}"
            elif target > 0:
                side, species = who.get(target, ("opponent", None))
                where = f"opponent {species or '?'} (target {target})"
            else:
                side, species = who.get(target, ("ally", None))
                where = "itself" if target == own_position else f"your partner {species or ally_species or '?'}"
            add(
                "move",
                {"type": "move", "move_id": move_id, "target": target},
                f"{move_id} -> {where} [{', '.join(details)}]",
                base_power=base_power,
                damaging=damaging,
                target=target,
                hits_ally=hits_ally,
                spread=spread,
            )

    for option in raw:
        if option.get("type") == "switch" and option.get("species"):
            species = option["species"]
            entry = bench.get(species)
            status = (entry or {}).get("status")
            status_text = f", {status}" if status else ""
            add("switch", {"type": "switch", "species": species},
                f"switch to {species} ({_pct(entry)} HP{status_text})")

    if any(o.get("type") == "pass" for o in raw):
        add("pass", {"type": "pass"}, "pass (nothing to do in this slot)")

    return options


def combo_error(first: Option, second: Option) -> str | None:
    """Why the server would reject this pair of slot choices, or None."""
    if first.kind == "switch" and second.kind == "switch" and first.payload["species"] == second.payload["species"]:
        return f"both slots switch into the same Pokémon ({first.payload['species']}); pick different ones"
    if first.kind == "pass" and second.kind == "pass":
        return "both slots cannot pass"
    return None


def _fallback_score(option: Option) -> float:
    if "score" in option.extra:  # computed damage, see analysis.annotate_options
        return option.extra["score"]
    if option.kind == "move":
        if option.damaging and (option.target is None or option.target > 0 or option.spread):
            return option.base_power * (0.75 if option.spread else 1.0) - (60 if option.hits_ally else 0)
        return 5.0  # status / protect
    if option.kind == "switch":
        return 1.0
    return 0.0


def _pair_score(first: Option, second: Option) -> float:
    score = _fallback_score(first) + _fallback_score(second)
    # Don't spend both attacks on one target when either alone already KOs it.
    for a, b in ((first, second), (second, first)):
        target = a.payload.get("target")
        if a.kind == b.kind == "move" and target and target > 0 and target == b.payload.get("target"):
            if (a.extra.get("ko") or {}).get(target, 0) >= 0.9:
                score -= 0.8 * _fallback_score(b)
                break
    return score


def fallback_choice(slot0: list[Option], slot1: list[Option]) -> tuple[Option, Option]:
    """A legal, non-random pair when the LLM can't answer in time: the best
    combined damage (computed when available), avoiding invalid combinations
    and overkill on a target one attack already knocks out."""
    pairs = [(a, b) for a in slot0 for b in slot1 if combo_error(a, b) is None]
    if not pairs:
        return slot0[0], slot1[0]
    return max(pairs, key=lambda pair: _pair_score(*pair))
