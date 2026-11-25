"""What the LLM reads: system instructions and compact text views of the game.

The baseline agent sent the raw observation JSON (~4,500 tokens a turn). These
views give the same decision-relevant facts in plain text, plus what the
server no longer shows after the draft: every opponent's full set.
"""

from __future__ import annotations

import re

from .cards import CardBook, format_card, species_key
from .options import Option

SNAKE_ORDER = ["A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B", "A"]

_COMMON = (
    "Format: Pokémon VGC doubles (Gen 9), every Pokémon level 50, no Terastallization. "
    "Each team drafts 6 fixed sets, brings 4 to the battle and leads with 2; the battle is 4v4 doubles. "
    "The winner knocks out all 4 opposing Pokémon; if the match timer runs out, more total remaining HP wins."
)

DRAFT_SYSTEM = f"""You are an expert Pokémon VGC player drafting a team.
{_COMMON}
Two players snake-draft 6 cards each from a shared 18-card pool. Each card is a fixed set (species, item, ability, nature, EVs, moves). Item Clause: your six cards must hold six different items, so some cards may be blocked for you.

Draft a TEAM, not six strong individuals:
- Speed control (Tailwind, Trick Room, Icy Wind, Electroweb, Thunder Wave) and disruption (Fake Out, Intimidate, Follow Me / Rage Powder, Spore, Taunt, Encore).
- Strong attackers, ideally with spread moves (Rock Slide, Heat Wave, Dazzling Gleam, Make It Rain, Earthquake with a Flying or Levitate partner).
- Defensive type synergy: do not stack the same weaknesses; mix physical and special attackers. Protect on most members is valuable.
- Denial: if a card would badly hurt your team or perfectly complete the opponent's, taking it first can be worth more than a small upgrade.
Use the snake order: when the opponent picks twice in a row next, take the card you would hate to lose.

Answer with: analysis (at most 60 words), pick (one card_id from the legal list), public_reason (at most 20 words; it is shown to spectators, so do not reveal your plan)."""

PREVIEW_SYSTEM = f"""You are an expert Pokémon VGC player at Team Preview.
{_COMMON}
You see both full teams with exact sets. Choose the 4 Pokémon to bring and which 2 of them lead.
- Bring the 4 that best handle the opponent's likely four, covering their biggest threats.
- Lead with a pair that applies immediate pressure or support (Fake Out, Intimidate, speed control, strong spread damage) and works well together.
- Keep answers to the opponent's likely leads in the back.

Answer with: analysis (at most 60 words), bring (exactly 4 species ids from your roster), leads (exactly 2 of those 4), public_reason (at most 20 words, shown to spectators)."""

BATTLE_SYSTEM = f"""You are an expert Pokémon VGC doubles player controlling both of your active Pokémon this turn.
{_COMMON}
Both players choose simultaneously; then moves resolve by priority, then Speed (Tailwind doubles Speed; Trick Room reverses the order).
You get a numbered menu for each of your two slots. Every option is already legal; pick exactly one option id per slot.
Doubles principles:
- Knock out threats: focus both attacks on one target when that secures a KO or removes the biggest threat; avoid wasting damage on a target that will already faint.
- Type matchups and Speed decide most turns; spread moves (hitting two targets) deal 75% damage to each.
- Protect when your Pokémon is threatened and its partner can act freely, or to scout; do not Protect repeatedly (it can fail).
- Use Fake Out (first turn on the field only) to stop the most dangerous opponent; use speed control and redirection early.
- Switch out of a losing matchup when a benched Pokémon takes the hit well.
- A move marked "ALSO HITS YOUR PARTNER" damages your own Pokémon unless the partner is immune (e.g. Earthquake vs a Flying type or Levitate).
Opponent HP is a percentage. Opponent sets are exact: they come from the draft.

Answer with: analysis (at most 80 words), slot_0 and slot_1 (one option id each), public_reason (at most 20 words, shown to spectators)."""


# ---------------------------------------------------------------- helpers

def _types(entry: dict | None) -> str:
    types = [t.title() for t in (entry or {}).get("types") or []]
    return "/".join(types) or "?"


def _base_stats(entry: dict | None) -> str:
    stats = (entry or {}).get("base_stats") or {}
    order = [("hp", "HP"), ("atk", "Atk"), ("def", "Def"), ("spa", "SpA"), ("spd", "SpD"), ("spe", "Spe")]
    return " / ".join(f"{label} {stats[key]}" for key, label in order if key in stats) or "?"


def _boosts(entry: dict | None) -> str:
    boosts = (entry or {}).get("boosts") or {}
    parts = [f"{value:+d} {stat}" for stat, value in boosts.items() if value]
    return f" [boosts {', '.join(parts)}]" if parts else ""


def _status(entry: dict | None) -> str:
    status = (entry or {}).get("status")
    return f" [{status}]" if status else ""


def _pct(entry: dict | None) -> str:
    frac = (entry or {}).get("current_hp_fraction")
    return f"{round(frac * 100)}%" if isinstance(frac, (int, float)) else "?"


def _card_line(book: CardBook, species: str | None, entry: dict | None) -> str:
    card = book.by_species(species)
    if card:
        return format_card(card)
    entry = entry or {}
    moves = ", ".join(entry.get("moves") or []) or "unknown moves"
    return f"{entry.get('name') or species} @ {entry.get('item') or '?'} | {entry.get('ability') or '?'} | {moves}"


# ---------------------------------------------------------------- draft

def draft_view(obs: dict, me: str, legal_ids: list[str], book: CardBook) -> str:
    rosters = obs.get("rosters") or {}
    mine = rosters.get(me) or []
    opponent = next((cards for seat, cards in rosters.items() if seat != me), [])
    my_letter = "A" if obs.get("first_drafter") == me else "B"
    pick_number = int(obs.get("pick_number") or (len(mine) + len(opponent) + 1))
    upcoming = SNAKE_ORDER[pick_number:pick_number + 3]
    upcoming_text = ", ".join("you" if letter == my_letter else "opponent" for letter in upcoming) or "none"

    def roster_lines(cards: list[dict]) -> list[str]:
        lines = []
        for entry in cards:
            card = book.by_id(entry.get("card_id", "")) or book.by_species(entry.get("species"))
            lines.append(f"  - {format_card(card) if card else entry.get('species')}")
        return lines or ["  (none yet)"]

    available = {c.get("card_id"): c for c in obs.get("available_cards") or [] if isinstance(c, dict)}
    blocked = [cid for cid in available if cid not in legal_ids]
    lines = [
        f"Draft pick {pick_number} of 12. You are drafter {my_letter}. Snake order: {' '.join(SNAKE_ORDER)}.",
        f"After this pick, the next picks go to: {upcoming_text}.",
        f"YOUR ROSTER ({len(mine)}/6):", *roster_lines(mine),
        f"OPPONENT ROSTER ({len(opponent)}/6):", *roster_lines(opponent),
        "LEGAL PICKS (card_id: set):",
    ]
    for card_id in legal_ids:
        card = available.get(card_id) or book.by_id(card_id)
        lines.append(f"  - {card_id}: {format_card(card) if card else '?'}")
    if blocked:
        lines.append("BLOCKED FOR YOU by Item Clause (the opponent may still take them):")
        lines += [f"  - {format_card(available[cid])}" for cid in blocked]
    return "\n".join(lines)


# ---------------------------------------------------------------- team preview

def team_preview_view(obs: dict, book: CardBook) -> str:
    def block(title: str, roster: list[dict]) -> list[str]:
        out = [title]
        for entry in roster:
            out.append(
                f"  - {entry.get('species')} ({_types(entry)}; base {_base_stats(entry)}): "
                f"{_card_line(book, entry.get('species'), entry)}"
            )
        return out

    lines = block("YOUR ROSTER (species id, types, base stats, set):", obs.get("your_roster") or [])
    lines += block("OPPONENT ROSTER:", obs.get("opponent_roster") or [])
    return "\n".join(lines)


# ---------------------------------------------------------------- battle

_LOG_KEEP = {
    "move", "-damage", "-heal", "faint", "switch", "drag", "-status", "-curestatus", "-boost",
    "-unboost", "-weather", "-fieldstart", "-fieldend", "-sidestart", "-sideend", "-crit",
    "-supereffective", "-resisted", "-immune", "-fail", "-miss", "cant", "-ability", "-item",
    "-enditem", "-start", "-end", "-activate", "-clearallboost", "turn",
}


def my_side_id(obs: dict) -> str | None:
    for key in (obs.get("team") or {}):
        match = re.match(r"(p\d):", key)
        if match:
            return match.group(1)
    return None


def last_turn_lines(protocol_log: list[str], my_side: str | None, limit: int = 40) -> list[str]:
    """The last completed turn's events (and anything since), as short lines."""
    log = [line for line in protocol_log or [] if isinstance(line, str)]
    turn_marks = [i for i, line in enumerate(log) if line.startswith("|turn|")]
    start = turn_marks[-2] + 1 if len(turn_marks) >= 2 else 0
    out = []
    for line in log[start:]:
        parts = line.split("|")
        if len(parts) < 2 or parts[1] not in _LOG_KEEP:
            continue
        text = " ".join(p for p in parts[1:] if p)
        if my_side:
            other = "p2" if my_side == "p1" else "p1"
            text = re.sub(rf"\b{my_side}[ab]?: ", "YOUR ", text)
            text = re.sub(rf"\b{other}[ab]?: ", "OPP ", text)
        out.append(text)
    return out[-limit:]


def battle_view(
    obs: dict,
    options: dict[int, list[Option]],
    book: CardBook,
    opponent_preview: list[dict],
) -> str:
    my_side = my_side_id(obs)
    switchable = {o.payload["species"] for slot_options in options.values() for o in slot_options if o.kind == "switch"}
    lines = [f"Turn {obs.get('turn')}."]
    weather = obs.get("weather") or {}
    fields = obs.get("fields") or {}
    if weather:
        lines.append(f"Weather: {weather}")
    if fields:
        lines.append(f"Field: {fields}")
    if obs.get("side_conditions"):
        lines.append(f"Your side conditions (turns left): {obs['side_conditions']}")
    if obs.get("opponent_side_conditions"):
        lines.append(f"Opponent side conditions (turns left): {obs['opponent_side_conditions']}")

    actives = obs.get("active_pokemon") or [None, None]
    lines.append("YOUR ACTIVE POKÉMON:")
    for slot, entry in enumerate(actives):
        if not entry:
            lines.append(f"  slot {slot}: (empty)")
            continue
        lines.append(
            f"  slot {slot}: {entry.get('name')} {entry.get('current_hp')}/{entry.get('max_hp')} HP ({_pct(entry)})"
            f"{_status(entry)}{_boosts(entry)} | {_types(entry)} | base {_base_stats(entry)} | "
            f"item {entry.get('item') or 'none'} | ability {entry.get('ability')}"
        )

    # obs["team"] also lists the 2 Pokémon we did not bring. In play = could
    # switch in this turn, or already fainted (both only happen to brought ones).
    active_species = {e.get("species") for e in actives if e}
    bench = [
        e for e in (obs.get("team") or {}).values()
        if isinstance(e, dict) and e.get("species") not in active_species
        and (e.get("species") in switchable or e.get("fainted"))
    ]
    lines.append("YOUR BENCH: " + ("; ".join(
        f"{e.get('name')} {'FAINTED' if e.get('fainted') else _pct(e)}{_status(e) if not e.get('fainted') else ''}"
        for e in bench) or "none"))

    opp_actives = obs.get("opponent_active_pokemon") or [None, None]
    lines.append("OPPONENT ACTIVE POKÉMON (target number in brackets):")
    for index, entry in enumerate(opp_actives):
        if not entry:
            lines.append(f"  [target {index + 1}] (empty)")
            continue
        lines.append(
            f"  [target {index + 1}] {entry.get('name') or entry.get('species')} {_pct(entry)} HP"
            f"{_status(entry)}{_boosts(entry)} | {_types(entry)} | base {_base_stats(entry)} | "
            f"set: {_card_line(book, entry.get('species'), entry)}"
        )

    opp_active_species = {e.get("species") for e in opp_actives if e}
    seen = {e.get("species"): e for e in (obs.get("opponent_team") or {}).values() if isinstance(e, dict)}
    others = []
    for entry in opponent_preview:
        species = entry.get("species")
        if species in opp_active_species:
            continue
        state = seen.get(species)
        if state is None:
            tag = "not seen yet (maybe not brought)"
        elif state.get("status") == "FNT" or state.get("fainted"):
            tag = "FAINTED"
        else:
            tag = f"seen, {_pct(state)} HP{_status(state)}"
        others.append(f"  - {species} ({_types(entry)}): {tag}")
    lines.append("OPPONENT'S OTHER POKÉMON (they brought 4 of these 6):")
    lines += others or ["  (none)"]

    log = last_turn_lines(obs.get("protocol_log") or [], my_side)
    if log:
        lines.append("LAST TURN (Showdown log; OPP HP is out of 100):")
        lines += [f"  {line}" for line in log]

    lines.append("YOUR OPTIONS:")
    slot_names = {s: (actives[s] or {}).get("name") if s < len(actives) else None for s in (0, 1)}
    for slot in (0, 1):
        lines.append(f"  slot {slot} ({slot_names.get(slot) or 'empty'}):")
        lines += [f"    {o.id}: {o.text}" for o in options.get(slot, [])]
    return "\n".join(lines)


def species_ids(entries: list[dict]) -> list[str]:
    return [species_key(e.get("species")) for e in entries if isinstance(e, dict)]
