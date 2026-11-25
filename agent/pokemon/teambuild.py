"""Code-ranked team building: lineups at Team Preview and picks in the draft.

The LLM still decides; code gives it a short, explained shortlist computed
from the verified damage calculator, instead of leaving the combinatorics
(15 ways to bring 4 of 6, 6 lead pairs each; 17 draft cards) to its head.

Building block, duel(a, b): would ``a`` beat ``b`` one-on-one? From the
hits each needs to knock the other out with its best attack (full HP) and
who moves first (priority, then Speed). +1 is a clear win, -1 a clear loss.

A team *answers* an opposing Pokémon when one of its members wins that duel.
Opposing Pokémon that beat many of ours count as bigger threats. Roles
(speed control, Fake Out, Intimidate, redirection, spread damage, a
physical/special mix) earn bonuses; a Tailwind + Trick Room mix and stacked
weaknesses (one attacking type hitting 3+ members super-effectively) cost.
These are heuristics, so their weights are plain numbers below.
"""

from __future__ import annotations

import math
from itertools import combinations

from . import dex
from .calc import ABSORB_ABILITIES, RUIN, Field, Mon, Side, best_attack, priority, speed
from .cards import species_key

SPEED_CONTROL = {"tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "bleakwindstorm", "rocktomb",
                 "scaryface", "glare", "nuzzle"}
REDIRECTION = {"followme", "ragepowder"}
SPREAD = {"allAdjacentFoes", "allAdjacent"}

# Weights of the role terms, relative to the answer score (which runs -1..1).
W_SPEED_CONTROL, W_SECOND_SPEED_CONTROL = 0.15, 0.05
W_FAKE_OUT, W_INTIMIDATE, W_REDIRECTION = 0.10, 0.05, 0.05
W_SPREAD, W_MIXED = 0.05, 0.05
W_TAILWIND_AND_TRICK_ROOM = -0.10
W_STACKED_WEAKNESS = -0.08
W_LEADS = 0.5
W_DENIAL_OPP_NEXT, W_DENIAL_LATER = 0.6, 0.3


def hits_to_ko(pct: float) -> int:
    return 99 if pct <= 0 else max(1, math.ceil(100 / pct - 1e-9))


class Duels:
    """duel() with a cache: the same pairs come up many times per decision."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], float] = {}
        self._best: dict[tuple[str, str], tuple[str | None, float, int]] = {}

    def best(self, a: Mon, b: Mon) -> tuple[str | None, float, int]:
        key = (a.species, b.species)
        if key not in self._best:
            fld = Field(ruin={a.ability} if a.ability in RUIN else set())
            found = best_attack(a, b, fld)
            if found:
                move_id, result = found
                self._best[key] = (move_id, result.avg_pct, priority(a, move_id, fld))
            else:
                self._best[key] = (None, 0.0, 0)
        return self._best[key]

    def duel(self, a: Mon, b: Mon) -> float:
        key = (a.species, b.species)
        if key in self._cache:
            return self._cache[key]
        _, off, pri_a = self.best(a, b)
        _, dfn, pri_b = self.best(b, a)
        na, nb = hits_to_ko(off), hits_to_ko(dfn)
        if na < nb:
            value = min(1.0, 0.5 + 0.25 * (nb - na))
        elif na > nb:
            value = -min(1.0, 0.5 + 0.25 * (na - nb))
        elif na >= 99:
            value = 0.0
        elif pri_a != pri_b:
            value = 0.5 if pri_a > pri_b else -0.5
        else:
            spe_a, spe_b = speed(a, Side(), Field()), speed(b, Side(), Field())
            value = 0.0 if spe_a == spe_b else (0.5 if spe_a > spe_b else -0.5)  # a Speed tie is a coin flip
        self._cache[key] = value
        return value


def roles(mon: Mon) -> set[str]:
    moves = set(mon.moves)
    out = set()
    if moves & SPEED_CONTROL:
        out.add("speed control")
    if "tailwind" in moves:
        out.add("Tailwind")
    if "trickroom" in moves:
        out.add("Trick Room")
    if "fakeout" in moves:
        out.add("Fake Out")
    if mon.ability == "intimidate":
        out.add("Intimidate")
    if moves & REDIRECTION:
        out.add("redirection")
    for move_id in moves:
        move = dex.move(move_id) or {}
        if move.get("cat") in ("Physical", "Special"):
            out.add(move["cat"].lower())
            if move.get("target") in SPREAD:
                out.add("spread")
    return out


def _weak_to(mon: Mon, move_type: str) -> bool:
    if ABSORB_ABILITIES.get(mon.ability) == move_type or (move_type == "GROUND" and mon.item == "airballoon"):
        return False
    return dex.effectiveness(move_type, mon.types) >= 2


def attacking_types(mons: list[Mon]) -> set[str]:
    types = set()
    for mon in mons:
        for move_id in mon.moves:
            move = dex.move(move_id) or {}
            if move.get("cat") in ("Physical", "Special") and move.get("bp", 0) >= 60:
                types.add(move["type"])
    return types


def team_value(team: list[Mon], foes: list[Mon], weights: dict[str, float], duels: Duels,
               *, size_for_weakness: int = 4) -> tuple[float, dict]:
    """How well ``team`` handles ``foes`` (weighted), plus role terms. Returns (score, details)."""
    if not team or not foes:
        return 0.0, {}
    total_w = sum(weights.get(f.species, 1.0) for f in foes)
    answer = 0.0
    answered, unanswered = {}, []
    for foe in foes:
        w = weights.get(foe.species, 1.0)
        best_mon, best_value = max(((m, duels.duel(m, foe)) for m in team), key=lambda x: x[1])
        answer += w * best_value
        if best_value > 0:
            answered[foe.name] = best_mon.name
        else:
            unanswered.append(foe.name)
    score = answer / total_w

    team_roles = [roles(m) for m in team]
    all_roles = set().union(*team_roles)
    n_speed = sum("speed control" in r for r in team_roles)
    bonus = 0.0
    if n_speed:
        bonus += W_SPEED_CONTROL + (W_SECOND_SPEED_CONTROL if n_speed > 1 else 0)
    bonus += W_FAKE_OUT * ("Fake Out" in all_roles) + W_INTIMIDATE * ("Intimidate" in all_roles)
    bonus += W_REDIRECTION * ("redirection" in all_roles) + W_SPREAD * ("spread" in all_roles)
    bonus += W_MIXED * ({"physical", "special"} <= all_roles)
    if {"Tailwind", "Trick Room"} <= all_roles:
        bonus += W_TAILWIND_AND_TRICK_ROOM
    stacked = []
    limit = 3 if size_for_weakness <= 4 else 4
    for move_type in sorted(attacking_types(foes)):
        weak = [m.name for m in team if _weak_to(m, move_type)]
        if len(weak) >= limit:
            stacked.append(f"{move_type.title()} hits {', '.join(weak)}")
            bonus += W_STACKED_WEAKNESS
    return score + bonus, {"answered": answered, "unanswered": unanswered, "roles": all_roles, "stacked": stacked}


def threat_weights(foes: list[Mon], ours: list[Mon], duels: Duels, extra: dict[str, float] | None = None) -> dict[str, float]:
    """Foes that beat many of ours matter more (1.0 .. 2.0), times any extra weight (e.g. past leads)."""
    weights = {}
    for foe in foes:
        beats = sum(duels.duel(foe, m) > 0 for m in ours) / max(1, len(ours))
        weights[foe.species] = (1.0 + beats) * (extra or {}).get(foe.species, 1.0)
    return weights


def _role_text(found: set[str]) -> str:
    keep = [r for r in ("Tailwind", "Trick Room", "speed control", "Fake Out", "Intimidate", "redirection", "spread")
            if r in found]
    if "Tailwind" in found or "Trick Room" in found:
        keep = [r for r in keep if r != "speed control"]
    return ", ".join(keep) or "none"


# ------------------------------------------------------------------ Team Preview

def rank_lineups(ours: list[Mon], theirs: list[Mon], *, past_leads: dict[str, int] | None = None,
                 top: int = 3) -> list[dict]:
    """Score every (bring 4, lead 2) choice against their six."""
    duels = Duels()
    lead_weight = {species_key(s): 1.0 + n for s, n in (past_leads or {}).items()}
    weights = threat_weights(theirs, ours, duels)
    results = []
    for bring in combinations(ours, 4):
        base, details = team_value(list(bring), theirs, weights, duels)
        best_lead = None
        for leads in combinations(bring, 2):
            lead_score = 0.0
            total = 0.0
            for foe in theirs:
                w = weights[foe.species] * lead_weight.get(foe.species, 1.0)
                lead_score += w * max(duels.duel(m, foe) for m in leads)
                total += w
            lead_score /= total
            lead_roles = set().union(*(roles(m) for m in leads))
            lead_score += 0.1 * ("Fake Out" in lead_roles) + 0.1 * ("speed control" in lead_roles)
            if best_lead is None or lead_score > best_lead[0]:
                best_lead = (lead_score, leads)
        results.append({
            "bring": [m.species for m in bring], "leads": [m.species for m in best_lead[1]],
            "names": [m.name for m in bring], "lead_names": [m.name for m in best_lead[1]],
            "score": round(base + W_LEADS * best_lead[0], 3), **details,
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top]


def lineup_lines(ranked: list[dict]) -> list[str]:
    if not ranked:
        return []
    lines = ["CODE-RANKED LINEUPS (a heuristic from one-on-one damage/Speed matchups and team roles; "
             "a starting point, you decide):"]
    for n, r in enumerate(ranked, 1):
        answers = "; ".join(f"{foe} <- {mine}" for foe, mine in r["answered"].items())
        lines.append(f"  {n}. bring {', '.join(r['names'])}; lead {' + '.join(r['lead_names'])} (score {r['score']})")
        lines.append(f"     answers: {answers or 'none'}; unanswered: {', '.join(r['unanswered']) or 'none'}; "
                     f"roles: {_role_text(r['roles'])}" + (f"; weakness: {'; '.join(r['stacked'])}" if r["stacked"] else ""))
    return lines


# ------------------------------------------------------------------ draft

def rank_draft(legal: list[tuple[str, Mon]], mine: list[Mon], theirs: list[Mon], pool: list[Mon], *,
               opponent_next: bool, top: int = 5) -> list[dict]:
    """Score each legal card: how much it improves our roster's answers and
    roles, plus how much it would improve the opponent's (denial)."""
    duels = Duels()
    mine_ids = {m.species for m in mine}
    their_ids = {m.species for m in theirs}
    free = [m for m in pool if m.species not in mine_ids | their_ids]  # still undrafted

    w_denial = W_DENIAL_OPP_NEXT if opponent_next else W_DENIAL_LATER
    results = []
    for card_id, card in legal:
        rest = [m for m in free if m.species != card.species]
        # Who we must beat: their picks (double weight) and, as a proxy for
        # their future picks, the rest of the undrafted pool.
        foes = theirs + rest
        weights = {m.species: (2.0 if m.species in their_ids else 1.0) for m in foes}
        before, _ = team_value(mine, foes, weights, duels, size_for_weakness=6) if mine else (0.0, {})
        after, details = team_value(mine + [card], foes, weights, duels, size_for_weakness=6)
        gain = after - before
        # Denial: how much the card would help them against our roster (+ the pool).
        our_side = mine + rest
        their_w = {m.species: (2.0 if m.species in mine_ids else 1.0) for m in our_side}
        opp_before, _ = team_value(theirs, our_side, their_w, duels, size_for_weakness=6) if theirs else (0.0, {})
        opp_after, _ = team_value(theirs + [card], our_side, their_w, duels, size_for_weakness=6)
        denial = opp_after - opp_before
        beats = [f.name for f in theirs if duels.duel(card, f) > 0]
        beaten_by = [f.name for f in theirs if duels.duel(f, card) > 0]
        results.append({
            "card_id": card_id, "name": card.name, "score": round(gain + w_denial * denial, 3),
            "gain": round(gain, 3), "denial": round(denial, 3), "roles": roles(card),
            "beats": beats, "beaten_by": beaten_by, "stacked": details.get("stacked", []),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top]


def draft_lines(ranked: list[dict], opponent_next: bool) -> list[str]:
    if not ranked:
        return []
    lines = ["CODE-RANKED PICKS (a heuristic: how much each card improves your one-on-one answers and team roles, "
             f"plus denial value{' (weighted up: the opponent picks next)' if opponent_next else ''}; "
             "a starting point, you decide):"]
    for n, r in enumerate(ranked, 1):
        extra = []
        if r["beats"]:
            extra.append(f"beats {', '.join(r['beats'])}")
        if r["beaten_by"]:
            extra.append(f"loses to {', '.join(r['beaten_by'])}")
        if r["stacked"]:
            extra.append("team weakness: " + "; ".join(r["stacked"]))
        lines.append(f"  {n}. {r['card_id']} (score {r['score']}: team {r['gain']:+}, denial {r['denial']:+}) "
                     f"roles: {_role_text(r['roles'])}" + (f"; {'; '.join(extra)}" if extra else ""))
    return lines
