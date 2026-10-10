"""LOCKED layout for the Stage-2 STRUCTURED-SCHEMA card-effect vector (mirrors funcnet/spec.py's style).

This module is the SINGLE SOURCE OF TRUTH for the effect-semantics ontology. It is shared by BOTH ends
of the pipeline so they can never drift:
  * the offline LLM extraction (tools/extract_card_schema -> out/card_schema.jsonl) validates its rows
    against LLM_SCHEMA / the enum tuples here, and
  * the deterministic encoder schema_row(json) packs a validated row into a fixed-width float32 vector.

The ontology was fixed DATA-DRIVEN after dumping + frequency-probing the real card text (see
funcnet/text/corpus.py and out/card_schema_report.md), NOT pre-committed blind.

LOCKED: SCHEMA_FEAT_DIM and every offset below are frozen+asserted so a committed extraction stays
consistent with any model that later projects this vector (mirror of funcnet/spec.py's frozen dims).
schema_row() is DEPLOY-SAFE: unknown / missing enum values fall to the catch-all bucket and NEVER raise.
"""
import numpy as np

# ---------------------------------------------------------------- effect-type ontology (multi-hot) ---
# ~22 data-driven buckets + an explicit catch-all ('other'). A card may carry SEVERAL effect types.
# Order is FROZEN — appending only at the end (before re-freezing the dim) is the only safe edit.
EFFECT_TYPES = (
    "draw",                 # draw cards into hand
    "search_deck",          # search your deck for cards (to hand/bench/attach)
    "search_discard",       # retrieve / look at / play from YOUR discard pile
    "heal",                 # heal HP / remove damage counters from your Pokemon
    "damage_boost",         # conditional EXTRA damage ("does N more damage if ...")
    "damage_reduce",        # reduce/partly prevent damage done to your Pokemon ("takes N less")
    "prevent_protect",      # full immunity: prevent ALL damage/effects, "can't be affected"
    "energy_accel",         # ATTACH energy (from deck/discard/hand/zone) — acceleration
    "energy_discard",       # DISCARD energy (own cost or opponent disruption)
    "switch_own",           # switch/retreat YOUR Pokemon, move your own energy around
    "gust_opponent",        # force opponent to switch / drag up a benched Pokemon ("gust")
    "status_condition",     # inflict asleep/paralyzed/confused/poisoned/burned
    "snipe_bench",          # deal damage / place damage counters on opponent's BENCH
    "scaling_damage",       # damage scales "for each ..." / "times the number of ..."
    "disrupt_hand",         # opponent discards/reveals/shuffles hand; hand disruption
    "lock",                 # disable: item-lock, ability-lock, can't attack/retreat/evolve
    "prize_manipulation",   # take extra / fewer prizes; prize-card effects
    "recoil",               # this Pokemon does damage to ITSELF
    "coin_flip",            # outcome gated on coin flip(s) (randomness vehicle)
    "energy_provide",       # (energy cards) provides energy of some type(s)
    "tool_stadium",         # Pokemon Tool / Stadium attach/effect mechanics
    "deck_manipulation",    # look at / rearrange / mill top of a deck (yours or opponent's)
    # --- v2 audit additions: recurring mechanics that the catch-all was masking (see card_schema_audit.md)
    "evolution_effect",     # evolve early/flexibly, devolve, Rare Candy (skip stage), use prev-stage attacks
    "ignore_defense",       # damage/effect ignores Weakness/Resistance/effects on the target ("isn't affected by")
    "stat_buff",            # modify a Pokemon's max HP / type / weakness / retreat cost (+HP tools/stadiums, etc.)
    "copy_attack",          # use an opponent's or another Pokemon's attack as this attack
    "direct_ko",            # Knock Out effect (outright or HP/counter-threshold), not via normal attack damage
    "place_counters",       # place / move damage counters outside the attack's main number (active or anywhere)
    "other",                # CATCH-ALL: effect text present but not captured above
)
N_EFFECT = len(EFFECT_TYPES)
OTHER_IDX = EFFECT_TYPES.index("other")
EFFECT_IDX = {name: i for i, name in enumerate(EFFECT_TYPES)}

# ---------------------------------------------------------------- single-valued categoricals (one-hot)
# trigger: WHEN the (most salient) effect fires
TRIGGERS = (
    "none",                 # text-free / not applicable
    "attack",               # it is an attack's effect
    "activated_ability",    # "Once during your turn, you may ..." (player-activated)
    "passive_ability",      # "As long as ..." continuous/static ability
    "on_play",              # "When you play this Pokemon ... onto your Bench"
    "on_ko_or_damaged",     # triggers when this Pokemon is damaged / Knocked Out
    "opponent_turn",        # effect applies during the opponent's next turn
)
N_TRIGGER = len(TRIGGERS)
TRIGGER_IDX = {name: i for i, name in enumerate(TRIGGERS)}

# frequency: HOW OFTEN / for how long
FREQUENCIES = (
    "none",
    "once_per_turn",        # once during your turn
    "once_per_game",        # once during the game / once while in play
    "each_turn",            # repeats every turn
    "while_in_play",        # continuous static effect
    "this_turn",            # one-shot, this turn only
    "next_turn",            # applies during the next (your/opponent's) turn
)
N_FREQ = len(FREQUENCIES)
FREQ_IDX = {name: i for i, name in enumerate(FREQUENCIES)}

# target: the PRIMARY object the effect acts on
TARGETS = (
    "none",
    "self",                 # this Pokemon itself
    "own_active",
    "own_bench",
    "own_all",              # all of your Pokemon
    "opponent_active",
    "opponent_bench",
    "opponent_all",         # all of opponent's Pokemon
    "both_players",
    "deck",                 # a deck (search/mill/look)
    "discard",              # a discard pile
    "hand",                 # a hand
)
N_TARGET = len(TARGETS)
TARGET_IDX = {name: i for i, name in enumerate(TARGETS)}

# ---------------------------------------------------------------- layout (frozen offsets) ------------
SCH_EFFECT   = 0                       # N_EFFECT   multi-hot effect types
SCH_TRIGGER  = SCH_EFFECT + N_EFFECT   # N_TRIGGER  one-hot trigger
SCH_FREQ     = SCH_TRIGGER + N_TRIGGER # N_FREQ     one-hot frequency
SCH_TARGET   = SCH_FREQ + N_FREQ       # N_TARGET   one-hot target
SCH_SCALARS  = SCH_TARGET + N_TARGET   # 8 scalar features (see below)

# the 8 scalar slots
SC_AMOUNT      = SCH_SCALARS + 0       # primary magnitude / 100, clipped [0,3]
SC_SCALING     = SCH_SCALARS + 1       # effect scales "for each ..." (bool)
SC_RANDOMNESS  = SCH_SCALARS + 2       # outcome depends on a coin flip / random (bool)
SC_NCOND       = SCH_SCALARS + 3       # #conditions / 4, clipped [0,1]
SC_NEFFECT     = SCH_SCALARS + 4       # #effect types / 4, clipped [0,1]
SC_CONFIDENCE  = SCH_SCALARS + 5       # extractor confidence [0,1]
SC_NEEDS_REVIEW = SCH_SCALARS + 6      # flagged for review (bool)
SC_HAS_EFFECT  = SCH_SCALARS + 7       # card carries ANY ability/attack effect text (bool)
N_SCALAR = 8

SCHEMA_FEAT_DIM = SCH_SCALARS + N_SCALAR
assert SCHEMA_FEAT_DIM == N_EFFECT + N_TRIGGER + N_FREQ + N_TARGET + N_SCALAR
assert SC_HAS_EFFECT + 1 == SCHEMA_FEAT_DIM
# Frozen value — bump this DELIBERATELY (and re-extract) if the ontology ever changes.
# v1=57 (22 effect types + other). v2=63 after the audit added 6 buckets (29 effect types + other).
assert SCHEMA_FEAT_DIM == 63, f"SCHEMA_FEAT_DIM changed to {SCHEMA_FEAT_DIM}; re-freeze deliberately"

_ZERO = np.zeros(SCHEMA_FEAT_DIM, dtype=np.float32)


def _f(x, default=0.0) -> float:
    """Coerce to float, defensively (None / bad str -> default)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def schema_row(row) -> np.ndarray:
    """Deterministically pack a validated/raw schema dict into a (SCHEMA_FEAT_DIM,) float32 vector.

    DEPLOY-SAFE: tolerant of missing keys and unknown enum strings — unknown effect types map to the
    'other' bucket, unknown trigger/frequency/target map to index 0 ('none'). NEVER raises."""
    out = np.zeros(SCHEMA_FEAT_DIM, dtype=np.float32)
    if not isinstance(row, dict):
        return out

    ets = row.get("effect_types") or []
    if isinstance(ets, str):
        ets = [ets]
    n_et = 0
    for et in ets:
        idx = EFFECT_IDX.get(str(et).strip().lower(), OTHER_IDX)
        if out[SCH_EFFECT + idx] == 0.0:
            n_et += 1
        out[SCH_EFFECT + idx] = 1.0

    out[SCH_TRIGGER + TRIGGER_IDX.get(str(row.get("trigger", "none")).strip().lower(), 0)] = 1.0
    out[SCH_FREQ + FREQ_IDX.get(str(row.get("frequency", "none")).strip().lower(), 0)] = 1.0
    out[SCH_TARGET + TARGET_IDX.get(str(row.get("target", "none")).strip().lower(), 0)] = 1.0

    out[SC_AMOUNT] = float(np.clip(_f(row.get("amount")) / 100.0, 0.0, 3.0))
    out[SC_SCALING] = 1.0 if row.get("scaling") else 0.0
    out[SC_RANDOMNESS] = 1.0 if row.get("randomness") else 0.0
    conds = row.get("conditions") or []
    n_cond = len(conds) if isinstance(conds, (list, tuple)) else 0
    out[SC_NCOND] = float(np.clip(n_cond / 4.0, 0.0, 1.0))
    out[SC_NEFFECT] = float(np.clip(n_et / 4.0, 0.0, 1.0))
    out[SC_CONFIDENCE] = float(np.clip(_f(row.get("confidence"), 0.0), 0.0, 1.0))
    out[SC_NEEDS_REVIEW] = 1.0 if row.get("needs_review") else 0.0
    # has_effect: explicit field if present, else inferred from any effect type / non-'none' trigger
    if "has_effect" in row:
        out[SC_HAS_EFFECT] = 1.0 if row.get("has_effect") else 0.0
    else:
        out[SC_HAS_EFFECT] = 1.0 if (n_et > 0 or str(row.get("trigger", "none")) != "none") else 0.0
    return out


# ---------------------------------------------------------------- JSON Schema for LLM extraction ------
# Shared with the workflow so the StructuredOutput tool forces valid rows against THIS exact vocab.
SCHEMA_ROW_JSON = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cardId", "effect_types", "trigger", "frequency", "target",
                 "amount", "scaling", "randomness", "conditions", "confidence", "needs_review"],
    "properties": {
        "cardId": {"type": "integer"},
        "effect_types": {"type": "array", "items": {"type": "string", "enum": list(EFFECT_TYPES)}},
        "trigger": {"type": "string", "enum": list(TRIGGERS)},
        "frequency": {"type": "string", "enum": list(FREQUENCIES)},
        "target": {"type": "string", "enum": list(TARGETS)},
        "amount": {"type": "number", "description": "primary numeric magnitude (damage/draw/counters/"
                                                    "energy/heal); 0 if none or variable"},
        "scaling": {"type": "boolean", "description": "does an effect scale 'for each ...'?"},
        "randomness": {"type": "boolean", "description": "outcome gated on a coin flip / random?"},
        "conditions": {"type": "array", "items": {"type": "string"},
                       "description": "short conditional clauses gating the effect (free text)"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_review": {"type": "boolean",
                         "description": "true if the text is ambiguous / didn't fit the ontology well"},
    },
}

LLM_BATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rows"],
    "properties": {"rows": {"type": "array", "items": SCHEMA_ROW_JSON}},
}
