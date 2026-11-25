"""Damage, KO-chance and speed calculations (Gen 9 doubles, level 50).

Follows Pokémon Showdown's damage formula: stats, boosts, the 16 random rolls,
4096-based modifiers with Showdown's rounding, STAB, type effectiveness,
items, abilities, Ruin abilities, screens, weather and terrain, plus the
special moves seen in this format (Body Press, Foul Play, Ruination,
multi-hit with Loaded Dice, Knock Off, always-crit moves, ...). The LLM is
bad at arithmetic, so code does it and the LLM reads the results.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from . import dex
from .cards import species_key

TYPE_BOOST_ITEMS = {
    "mysticwater": "WATER", "charcoal": "FIRE", "miracleseed": "GRASS", "magnet": "ELECTRIC",
    "nevermeltice": "ICE", "blackbelt": "FIGHTING", "poisonbarb": "POISON", "softsand": "GROUND",
    "sharpbeak": "FLYING", "twistedspoon": "PSYCHIC", "silverpowder": "BUG", "hardstone": "ROCK",
    "spelltag": "GHOST", "dragonfang": "DRAGON", "blackglasses": "DARK", "metalcoat": "STEEL",
    "silkscarf": "NORMAL", "fairyfeather": "FAIRY",
}
RESIST_BERRIES = {
    "occaberry": "FIRE", "passhoberry": "WATER", "wacanberry": "ELECTRIC", "rindoberry": "GRASS",
    "yacheberry": "ICE", "chopleberry": "FIGHTING", "kebiaberry": "POISON", "shucaberry": "GROUND",
    "cobaberry": "FLYING", "payapaberry": "PSYCHIC", "tangaberry": "BUG", "chartiberry": "ROCK",
    "kasibberry": "GHOST", "habanberry": "DRAGON", "colburberry": "DARK", "babiriberry": "STEEL",
    "roseliberry": "FAIRY", "chilanberry": "NORMAL",
}
ABSORB_ABILITIES = {
    "levitate": "GROUND", "eartheater": "GROUND", "waterabsorb": "WATER", "stormdrain": "WATER",
    "dryskin": "WATER", "voltabsorb": "ELECTRIC", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC",
    "flashfire": "FIRE", "wellbakedbody": "FIRE", "sapsipper": "GRASS",
}
RUIN = {"swordofruin": "def", "beadsofruin": "spd", "tabletsofruin": "atk", "vesselofruin": "spa"}
SPREAD_TARGETS = {"allAdjacentFoes", "allAdjacent"}


def _id(value) -> str:
    return species_key(value) if value else ""


def chain(multipliers: list[float]) -> int:
    """Showdown's chainModify: combine modifiers into one 4096-based value."""
    mod = 4096
    for m in multipliers:
        mod = (mod * round(m * 4096) + 2048) >> 12
    return mod


def apply_mod(value: int, mod4096: int) -> int:
    """Showdown's modify(): multiply by mod/4096, rounding half down."""
    return (value * mod4096 + 2047) // 4096


@dataclass
class Mon:
    species: str
    name: str
    types: list[str]
    stats: dict[str, int]
    hp: int
    ability: str = ""
    item: str = ""
    boosts: dict[str, int] = field(default_factory=dict)
    status: str = ""
    moves: list[str] = field(default_factory=list)
    active_boost: str | None = None  # Protosynthesis / Quark Drive stat, when active
    weight: float = 50.0
    known_set: bool = True

    @property
    def max_hp(self) -> int:
        return self.stats["hp"]

    @property
    def grounded(self) -> bool:
        return "FLYING" not in self.types and self.ability != "levitate" and self.item != "airballoon"


@dataclass
class Field:
    weather: str | None = None  # sun / rain / sand / snow
    terrain: str | None = None  # grassy / electric / psychic / misty
    trick_room: bool = False
    ruin: set[str] = field(default_factory=set)  # Ruin abilities among the active Pokémon


@dataclass
class Side:
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False
    tailwind: bool = False


@dataclass
class Damage:
    rolls: list[int]  # 16 total-damage values (all hits), lowest to highest
    defender_max: int
    defender_hp: int
    effectiveness: float
    hits: int = 1
    note: str = ""

    @property
    def lo(self) -> int:
        return self.rolls[0]

    @property
    def hi(self) -> int:
        return self.rolls[-1]

    def pct(self, value: int) -> int:
        return round(100 * value / max(1, self.defender_max))

    @property
    def ko_chance(self) -> float:
        if self.defender_hp <= 0:
            return 0.0
        return sum(r >= self.defender_hp for r in self.rolls) / len(self.rolls)

    @property
    def avg_pct(self) -> float:
        return 100 * sum(self.rolls) / len(self.rolls) / max(1, self.defender_max)

    def text(self) -> str:
        if self.effectiveness == 0:
            return "no effect (immune)"
        ko = self.ko_chance
        ko_text = "guaranteed KO" if ko >= 1 else (f"{round(ko * 100)}% KO chance" if ko > 0 else "no KO")
        hits = f" over {self.hits} hits" if self.hits > 1 else ""
        note = f"; {self.note}" if self.note else ""
        lo, hi = self.pct(self.lo), self.pct(self.hi)
        amount = f"{lo}%" if lo == hi else f"{lo}-{hi}%"
        return f"{amount}{hits} ({ko_text}{note})"


# ------------------------------------------------------------------ builders

def _set_stats(base: dict, card: dict | None) -> tuple[dict[str, int], bool]:
    if card:
        return dex.calc_stats(base, card.get("evs"), card.get("nature"), card.get("ivs")), True
    # Unknown set: assume full HP and best attacking stat investment, neutral nature.
    best = "atk" if base.get("atk", 0) >= base.get("spa", 0) else "spa"
    return dex.calc_stats(base, {"hp": 252, best: 252, "def": 4}, None), False


def build_mon(entry: dict | None, card: dict | None, *, exact_hp: bool) -> Mon | None:
    """A battle participant from the server's entry (live state) and its draft
    card (exact set). ``exact_hp``: our own Pokémon report HP in points; the
    opponent's only as a fraction."""
    entry = entry or {}
    species_name = entry.get("species") or (card or {}).get("species")
    dex_entry = dex.species(species_name)
    base = entry.get("base_stats") or (dex_entry or {}).get("bs")
    if not base:
        return None
    stats, known = _set_stats(base, card)
    if exact_hp and isinstance(entry.get("max_hp"), int) and entry.get("max_hp"):
        stats["hp"] = entry["max_hp"]
    if exact_hp and isinstance(entry.get("current_hp"), int):
        hp = entry["current_hp"]
    elif isinstance(entry.get("current_hp_fraction"), (int, float)):
        hp = round(entry["current_hp_fraction"] * stats["hp"])
    else:
        hp = stats["hp"]
    item = entry.get("item")
    if item in (None, "unknown_item") and "item" not in entry or item == "unknown_item":
        item = _id((card or {}).get("item"))
    ability = _id(entry.get("ability")) or _id((card or {}).get("ability"))
    status = (entry.get("status") or "").lower()
    types = [t.upper() for t in (entry.get("types") or (dex_entry or {}).get("types") or [])]
    move_ids = [_id(m) for m in ((card or {}).get("moves") or entry.get("moves") or [])]
    return Mon(
        species=_id(species_name), name=entry.get("name") or (dex_entry or {}).get("name") or species_name,
        types=types, stats=stats, hp=hp, ability=ability, item=_id(item), boosts=dict(entry.get("boosts") or {}),
        status=status, moves=move_ids, weight=(dex_entry or {}).get("weight") or 50.0, known_set=known,
    )


def field_from_obs(obs: dict, active_abilities: list[str]) -> tuple[Field, Side, Side]:
    weather_keys = " ".join(str(k).upper() for k in (obs.get("weather") or {}))
    weather = None
    for key, name in (("SUN", "sun"), ("DESOLATE", "sun"), ("RAIN", "rain"), ("PRIMORDIAL", "rain"),
                      ("SAND", "sand"), ("SNOW", "snow"), ("HAIL", "snow")):
        if key in weather_keys:
            weather = name
            break
    field_keys = " ".join(str(k).upper() for k in (obs.get("fields") or {}))
    terrain = next((name for key, name in (("GRASSY", "grassy"), ("ELECTRIC", "electric"),
                                           ("PSYCHIC", "psychic"), ("MISTY", "misty")) if key in field_keys), None)

    def side(conditions: dict | None) -> Side:
        keys = " ".join(str(k).upper() for k in (conditions or {}))
        return Side(reflect="REFLECT" in keys, light_screen="LIGHT_SCREEN" in keys,
                    aurora_veil="AURORA_VEIL" in keys, tailwind="TAILWIND" in keys)

    ruin = {a for a in active_abilities if a in RUIN}
    return (Field(weather=weather, terrain=terrain, trick_room="TRICK_ROOM" in field_keys, ruin=ruin),
            side(obs.get("side_conditions")), side(obs.get("opponent_side_conditions")))


_BOOST_START = re.compile(r"^\|-start\|(p\d)[ab]: ([^|]+)\|(?:quarkdrive|protosynthesis)(atk|def|spa|spd|spe)")
_SWITCH = re.compile(r"^\|(?:switch|drag)\|(p\d)[ab]: ([^|]+)\|")
_MOVE = re.compile(r"^\|move\|(p\d)[ab]: ([^|]+)\|([^|]*)")

# Moves that share Protect's "fails more often when used in a row" counter.
PROTECT_MOVES = frozenset({"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
                           "burningbulwark", "obstruct", "maxguard", "endure"})


@dataclass
class LogFacts:
    boosts: dict[tuple[str, str], str]     # active Booster Energy / Paradox stat per (side, name)
    moved: dict[tuple[str, str], bool]     # has it moved since it last came in (False: Fake Out works)
    protect_streak: dict[tuple[str, str], int]  # successful Protects in a row (next one works 1/3**n)


def read_log(protocol_log: list[str]) -> LogFacts:
    """Facts the observation doesn't give directly, read from the battle log."""
    boosts: dict[tuple[str, str], str] = {}
    moved: dict[tuple[str, str], bool] = {}
    streak: dict[tuple[str, str], int] = {}
    log = [line for line in protocol_log or [] if isinstance(line, str)]
    for index, line in enumerate(log):
        if m := _SWITCH.match(line):
            key = (m.group(1), m.group(2))
            boosts.pop(key, None)
            moved[key] = False
            streak[key] = 0
        elif m := _BOOST_START.match(line):
            boosts[(m.group(1), m.group(2))] = m.group(3)
        elif m := _MOVE.match(line):
            key = (m.group(1), m.group(2))
            moved[key] = True
            if species_key(m.group(3)) in PROTECT_MOVES:
                following = log[index + 1] if index + 1 < len(log) else ""
                failed = "[still]" in line or following.startswith("|-fail|")
                streak[key] = 0 if failed else streak.get(key, 0) + 1
            else:
                streak[key] = 0
        elif line.startswith("|-end|") and ("quarkdrive" in line.lower() or "protosynthesis" in line.lower()):
            parts = line.split("|")
            if len(parts) > 2 and (m2 := re.match(r"(p\d)[ab]: (.+)", parts[2])):
                boosts.pop((m2.group(1), m2.group(2)), None)
    return LogFacts(boosts, moved, streak)


def parse_log(protocol_log: list[str]) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], bool]]:
    """(boosts, moved) - see read_log."""
    facts = read_log(protocol_log)
    return facts.boosts, facts.moved


# ------------------------------------------------------------------ speed

def speed(mon: Mon, side: Side, fld: Field) -> int:
    value = math.floor(mon.stats["spe"] * dex.stage_multiplier(mon.boosts.get("spe", 0)))
    mods = []
    if mon.item == "choicescarf":
        mods.append(1.5)
    if mon.active_boost == "spe":
        mods.append(1.5)
    if (fld.weather == "rain" and mon.ability == "swiftswim") or (fld.weather == "sun" and mon.ability == "chlorophyll") \
            or (fld.weather == "sand" and mon.ability == "sandrush") or (fld.weather == "snow" and mon.ability == "slushrush"):
        mods.append(2.0)
    if side.tailwind:
        mods.append(2.0)
    value = apply_mod(value, chain(mods)) if mods else value
    if mon.status == "par" and mon.ability != "quickfeet":
        value = value // 2
    return value


def priority(mon: Mon, move_id: str, fld: Field) -> int:
    move = dex.move(move_id) or {}
    pri = move.get("pri", 0)
    if move_id == "grassyglide" and fld.terrain == "grassy" and mon.grounded:
        pri += 1
    if mon.ability == "galewings" and move.get("type") == "FLYING" and mon.hp >= mon.max_hp:
        pri += 1
    if mon.ability == "prankster" and move.get("cat") == "Status":
        pri += 1
    return pri


# ------------------------------------------------------------------ damage

def _type_immune(move_type: str, defender: Mon) -> bool:
    if ABSORB_ABILITIES.get(defender.ability) == move_type:
        return True
    return move_type == "GROUND" and defender.item == "airballoon"


def _hits(move: dict, attacker: Mon) -> int:
    multihit = move.get("multihit")
    if isinstance(multihit, int):
        return multihit
    if isinstance(multihit, list) and len(multihit) == 2:
        if attacker.ability == "skilllink":
            return multihit[1]
        if attacker.item == "loadeddice" and multihit[1] == 5:
            return 4
        return 3 if multihit[1] == 5 else multihit[0]
    return 1


def damage(attacker: Mon, defender: Mon, move_id: str, fld: Field | None = None, defender_side: Side | None = None,
           *, spread: bool = False) -> Damage | None:
    """Damage of one use of ``move_id`` against ``defender``; None for status moves."""
    fld = fld or Field()
    defender_side = defender_side or Side()
    move = dex.move(move_id)
    if not move or move.get("cat") not in ("Physical", "Special"):
        return None
    move_id = species_key(move_id)
    physical = move["cat"] == "Physical"
    move_type = move["type"]
    bp = move.get("bp", 0)

    # Type changes from weather / terrain moves.
    if move_id == "weatherball" and fld.weather:
        move_type = {"sun": "FIRE", "rain": "WATER", "sand": "ROCK", "snow": "ICE"}[fld.weather]
        bp = 100
    if move_id == "terrainpulse" and fld.terrain and attacker.grounded:
        move_type = {"grassy": "GRASS", "electric": "ELECTRIC", "psychic": "PSYCHIC", "misty": "FAIRY"}[fld.terrain]
        bp = 100

    eff = dex.effectiveness(move_type, defender.types, move_id)
    if _type_immune(move_type, defender) or (defender.ability == "wonderguard" and eff <= 1):
        eff = 0
    if eff == 0:
        return Damage([0] * 16, defender.max_hp, defender.hp, 0)

    # Fixed damage.
    fixed = move.get("fixed")
    if fixed == "level":
        return Damage([dex.LEVEL] * 16, defender.max_hp, defender.hp, eff)
    if move_id in ("ruination", "superfang", "naturesmadness"):
        value = max(1, defender.hp // 2)
        return Damage([value] * 16, defender.max_hp, defender.hp, eff, note="half of current HP")

    # Base power.
    if move_id == "knockoff" and defender.item:
        bp = bp * 3 // 2
    elif move_id == "acrobatics" and not attacker.item:
        bp *= 2
    elif move_id == "facade" and attacker.status in ("brn", "par", "psn", "tox"):
        bp *= 2
    elif move_id in ("hex", "infernalparade") and defender.status:
        bp *= 2
    elif move_id in ("eruption", "waterspout", "dragonenergy"):
        bp = max(1, 150 * attacker.hp // max(1, attacker.max_hp))
    elif move_id in ("lowkick", "grassknot"):
        w = defender.weight
        bp = 20 if w < 10 else 40 if w < 25 else 60 if w < 50 else 80 if w < 100 else 100 if w < 200 else 120
    elif move_id in ("heavyslam", "heatcrash"):
        ratio = attacker.weight / max(0.1, defender.weight)
        bp = 120 if ratio >= 5 else 100 if ratio >= 4 else 80 if ratio >= 3 else 60 if ratio >= 2 else 40
    elif move_id in ("storedpower", "powertrip"):
        bp = 20 + 20 * sum(v for v in attacker.boosts.values() if v > 0)
    elif move_id == "risingvoltage" and fld.terrain == "electric" and defender.grounded:
        bp *= 2
    elif move_id == "expandingforce" and fld.terrain == "psychic" and attacker.grounded:
        bp = bp * 3 // 2
    if move_id == "tripleaxel":
        bp = 40  # 20 + 40 + 60 spread over 3 hits

    flags = set(move.get("flags") or [])
    bp_mods: list[float] = []
    a = attacker.ability
    if a == "technician" and bp <= 60:
        bp_mods.append(1.5)
    if TYPE_BOOST_ITEMS.get(attacker.item) == move_type:
        bp_mods.append(1.2)
    if attacker.item == "punchingglove" and "punch" in flags:
        bp_mods.append(1.1)
    if attacker.item == "muscleband" and physical or attacker.item == "wiseglasses" and not physical:
        bp_mods.append(1.1)
    ability_bp = {
        "toughclaws": ("contact", 1.3), "ironfist": ("punch", 1.2), "strongjaw": ("bite", 1.5),
        "sharpness": ("slicing", 1.5), "megalauncher": ("pulse", 1.5), "punkrock": ("sound", 1.3),
    }
    if a in ability_bp and ability_bp[a][0] in flags:
        bp_mods.append(ability_bp[a][1])
    if a == "sheerforce" and move.get("secondary"):
        bp_mods.append(1.3)
    if a == "reckless" and move.get("recoil"):
        bp_mods.append(1.2)
    type_abilities = {"transistor": ("ELECTRIC", 1.3), "dragonsmaw": ("DRAGON", 1.5), "rockypayload": ("ROCK", 1.5),
                      "steelworker": ("STEEL", 1.5)}
    if a in type_abilities and type_abilities[a][0] == move_type:
        bp_mods.append(type_abilities[a][1])
    if a == "sandforce" and fld.weather == "sand" and move_type in ("ROCK", "GROUND", "STEEL"):
        bp_mods.append(1.3)
    if fld.terrain and attacker.grounded and (fld.terrain, move_type) in (
            ("grassy", "GRASS"), ("electric", "ELECTRIC"), ("psychic", "PSYCHIC")):
        bp_mods.append(1.3)
    if fld.terrain == "misty" and move_type == "DRAGON" and defender.grounded:
        bp_mods.append(0.5)
    if fld.terrain == "grassy" and move_id in ("earthquake", "bulldoze") and defender.grounded:
        bp_mods.append(0.5)
    if bp_mods:
        bp = max(1, apply_mod(bp, chain(bp_mods)))

    crit = bool(move.get("crit"))

    # Attack stat.
    off_mon = defender if move.get("off_mon") == "target" else attacker
    atk_key = "def" if move.get("off_stat") == "def" else ("atk" if physical else "spa")
    atk_stage = 0 if defender.ability == "unaware" else off_mon.boosts.get(atk_key, 0)
    if crit and atk_stage < 0:
        atk_stage = 0
    attack = math.floor(off_mon.stats[atk_key] * dex.stage_multiplier(atk_stage))
    atk_mods: list[float] = []
    if off_mon is attacker:
        if attacker.item == "choiceband" and atk_key == "atk" or attacker.item == "choicespecs" and atk_key == "spa":
            atk_mods.append(1.5)
        if a in ("hugepower", "purepower") and atk_key == "atk":
            atk_mods.append(2.0)
        if a in ("gorillatactics", "hustle") and atk_key == "atk":
            atk_mods.append(1.5)
        if a == "guts" and attacker.status and atk_key == "atk":
            atk_mods.append(1.5)
        if a == "solarpower" and fld.weather == "sun" and atk_key == "spa":
            atk_mods.append(1.5)
        if attacker.active_boost == atk_key:
            atk_mods.append(5325 / 4096)
        if a == "orichalcumpulse" and fld.weather == "sun" and atk_key == "atk" or \
                a == "hadronengine" and fld.terrain == "electric" and atk_key == "spa":
            atk_mods.append(5461 / 4096)
        if a == "waterbubble" and move_type == "WATER":
            atk_mods.append(2.0)
    if defender.ability == "thickfat" and move_type in ("FIRE", "ICE") or \
            defender.ability in ("heatproof", "waterbubble") and move_type == "FIRE" or \
            defender.ability == "purifyingsalt" and move_type == "GHOST":
        atk_mods.append(0.5)
    ruin_atk = "tabletsofruin" if atk_key == "atk" else "vesselofruin" if atk_key == "spa" else None
    if ruin_atk and ruin_atk in fld.ruin and attacker.ability != ruin_atk:
        atk_mods.append(0.75)
    if atk_mods:
        attack = max(1, apply_mod(attack, chain(atk_mods)))

    # Defense stat.
    def_key = "def" if physical or move.get("def_stat") == "def" else "spd"
    def_stage = 0 if a == "unaware" else defender.boosts.get(def_key, 0)
    if crit and def_stage > 0:
        def_stage = 0
    defense = math.floor(defender.stats[def_key] * dex.stage_multiplier(def_stage))
    def_mods: list[float] = []
    if defender.item == "assaultvest" and def_key == "spd":
        def_mods.append(1.5)
    if defender.item == "eviolite":  # only not-fully-evolved Pokémon carry it (e.g. Porygon2)
        def_mods.append(1.5)
    if defender.ability == "furcoat" and def_key == "def":
        def_mods.append(2.0)
    if defender.ability == "marvelscale" and defender.status and def_key == "def":
        def_mods.append(1.5)
    if defender.active_boost == def_key:
        def_mods.append(5325 / 4096)
    if fld.weather == "snow" and "ICE" in defender.types and def_key == "def":
        def_mods.append(1.5)
    if fld.weather == "sand" and "ROCK" in defender.types and def_key == "spd":
        def_mods.append(1.5)
    ruin_def = "swordofruin" if def_key == "def" else "beadsofruin"
    if ruin_def in fld.ruin and defender.ability != ruin_def:
        def_mods.append(0.75)
    if def_mods:
        defense = max(1, apply_mod(defense, chain(def_mods)))

    # Base damage, then Showdown's modifier order.
    base = (2 * dex.LEVEL // 5 + 2) * bp * attack // defense // 50 + 2
    if spread:
        base = apply_mod(base, 3072)
    if fld.weather == "sun" and move_type in ("FIRE", "WATER") or fld.weather == "rain" and move_type in ("FIRE", "WATER"):
        boosted = (fld.weather == "sun") == (move_type == "FIRE")
        base = apply_mod(base, 6144 if boosted else 2048)
    if crit:
        base = math.floor(base * 1.5)

    stab = 1.0
    if move_type in attacker.types or a in ("protean", "libero"):
        stab = 2.0 if a == "adaptability" else 1.5

    final_mods: list[float] = []
    if not crit:
        screen = defender_side.aurora_veil or (defender_side.reflect if physical else defender_side.light_screen)
        if screen and a != "infiltrator":
            final_mods.append(2732 / 4096)
    if defender.ability in ("multiscale", "shadowshield") and defender.hp >= defender.max_hp:
        final_mods.append(0.5)
    if eff > 1:
        if defender.ability in ("filter", "solidrock", "prismarmor"):
            final_mods.append(0.75)
        if attacker.item == "expertbelt":
            final_mods.append(4915 / 4096)
        if RESIST_BERRIES.get(defender.item) == move_type:
            final_mods.append(0.5)
        if a == "neuroforce":
            final_mods.append(1.25)
    if eff < 1 and a == "tintedlens":
        final_mods.append(2.0)
    if defender.ability == "fluffy":
        if "contact" in flags:
            final_mods.append(0.5)
        if move_type == "FIRE":
            final_mods.append(2.0)
    if defender.ability == "icescales" and not physical:
        final_mods.append(0.5)
    if defender.ability == "punkrock" and "sound" in flags:
        final_mods.append(0.5)
    if attacker.item == "lifeorb":
        final_mods.append(5324 / 4096)
    final_mod = chain(final_mods) if final_mods else 4096

    per_hit = []
    for r in range(85, 101):
        value = base * r // 100
        if stab != 1.0:
            value = apply_mod(value, round(stab * 4096))
        e = eff
        while e >= 2:
            value *= 2
            e /= 2
        while e <= 0.5 and e > 0:
            value //= 2
            e *= 2
        if attacker.status == "brn" and physical and a != "guts" and move_id != "facade":
            value = apply_mod(value, 2048)
        value = apply_mod(value, final_mod)
        per_hit.append(max(1, value))

    hits = _hits(move, attacker)
    if move_id == "tripleaxel":
        hits = 3
    rolls = [v * hits for v in per_hit]
    note = ""
    if hits == 1 and defender.hp >= defender.max_hp and (defender.item == "focussash" or defender.ability == "sturdy"):
        rolls = [min(v, defender.hp - 1) for v in rolls]
        note = "Focus Sash/Sturdy: survives at 1 HP"
    return Damage(sorted(rolls), defender.max_hp, defender.hp, eff, hits=hits, note=note)


def best_attack(attacker: Mon, defender: Mon, fld: Field | None = None, side: Side | None = None,
                *, move_ids: list[str] | None = None) -> tuple[str, Damage] | None:
    """The attacker's highest average-damage move against the defender."""
    best = None
    for move_id in move_ids or attacker.moves:
        move = dex.move(move_id) or {}
        result = damage(attacker, defender, move_id, fld, side, spread=move.get("target") in SPREAD_TARGETS)
        if result and (best is None or result.avg_pct > best[1].avg_pct):
            best = (move_id, result)
    return best
