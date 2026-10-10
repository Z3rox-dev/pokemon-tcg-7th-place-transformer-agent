"""LOCKED layout for the EFFECT-ATOM per-card encoder view (replaces the old 63-d card-schema view).

numpy-only (no torch) so it is importable by the offline table builder AND any deploy-safe lookup. Turns a
card's list of effect-atoms (out/card_atoms.jsonl, the gold dataset built from the engine skeleton + verified
LLM semantics) into a fixed-width FUNCTIONAL FINGERPRINT by pooling its atoms:

  effect_type histogram | target histogram | unit histogram | modifier histogram | trigger histogram |
  per-unit (sum,max) value  | a few scalars

The per-unit value split is the key fix the old single ambiguous `amount` lacked: 120 damage and 120 heal
land in DIFFERENT numeric slots, so the net never shares numeric space across units.

The canonical vocab ORDER is frozen from out/atom_vocab.json at import (the user-approved 72/32/15/40/72 set).
"""
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_PATH = os.path.join(_HERE, "..", "..", "out", "atom_vocab.json")

with open(_VOCAB_PATH, encoding="utf-8") as _f:
    _V = json.load(_f)

EFFECT_TYPES = [e["canonical"] for e in _V["effect_types"]]   # 72
TARGETS      = list(_V["targets"])                            # 32
UNITS        = list(_V["units"])                              # 15
MODIFIERS    = list(_V["modifiers"])                          # 72
TRIGGERS     = list(_V["triggers"])                           # 40

EFF_IDX  = {n: i for i, n in enumerate(EFFECT_TYPES)}
TGT_IDX  = {n: i for i, n in enumerate(TARGETS)}
UNIT_IDX = {n: i for i, n in enumerate(UNITS)}
MOD_IDX  = {n: i for i, n in enumerate(MODIFIERS)}
TRIG_IDX = {n: i for i, n in enumerate(TRIGGERS)}

N_EFF, N_TGT, N_UNIT, N_MOD, N_TRIG = len(EFFECT_TYPES), len(TARGETS), len(UNITS), len(MODIFIERS), len(TRIGGERS)

# ---------------------------------------------------------------- layout (offsets) ----
A_EFF   = 0                       # effect_type histogram (count/4 clip[0,1])
A_TGT   = A_EFF + N_EFF           # target histogram
A_UNIT  = A_TGT + N_TGT           # unit histogram
A_MOD   = A_UNIT + N_UNIT         # modifier histogram
A_TRIG  = A_MOD + N_MOD           # trigger histogram
A_VSUM  = A_TRIG + N_TRIG         # per-unit summed value (normalised per unit)
A_VMAX  = A_VSUM + N_UNIT         # per-unit max value (normalised per unit)
A_SCALAR = A_VMAX + N_UNIT        # scalars block
# scalars: n_atoms/4, n_effects/8, has_attack, has_ability, max_base_damage/300, any_scaling
SC_N_ATOMS, SC_N_EFF, SC_HAS_ATK, SC_HAS_ABIL, SC_MAX_DMG, SC_ANY_SCALING = range(A_SCALAR, A_SCALAR + 6)
ATOM_FEAT_DIM = A_SCALAR + 6

# per-unit value normalisers (so each unit's magnitude lands in ~[0,3])
_UNIT_NORM = {"damage": 100.0, "hp": 100.0, "damage_counter": 4.0, "energy": 4.0, "cards": 4.0,
              "card": 4.0, "pokemon": 3.0, "prize": 3.0, "coin_flip": 4.0, "special_condition": 1.0,
              "retreat_cost": 4.0, "tool": 2.0, "turn": 2.0, "hand_size": 8.0, "none": 1.0}


def card_atom_vector(card_obj, base_damages=None):
    """Pool one card's atoms (a dict {cardId, atoms:[...]}) into the ATOM_FEAT_DIM fingerprint.
    base_damages: optional list of engine attack damages (by attack index) to feed SC_MAX_DMG exactly."""
    v = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    if not card_obj:
        return v
    atoms = card_obj.get("atoms", []) or []
    n_eff = 0
    has_atk = has_abil = any_scaling = 0.0
    for atom in atoms:
        src = atom.get("source", "")
        if src.startswith("attack"):
            has_atk = 1.0
        elif src.startswith("ability"):
            has_abil = 1.0
        ti = TRIG_IDX.get(atom.get("trigger"))
        if ti is not None:
            v[A_TRIG + ti] = min(1.0, v[A_TRIG + ti] + 0.5)
        for e in atom.get("effects", []):
            n_eff += 1
            ei = EFF_IDX.get(e.get("type"))
            if ei is not None:
                v[A_EFF + ei] = min(1.0, v[A_EFF + ei] + 0.34)
            gi = TGT_IDX.get(e.get("target"))
            if gi is not None:
                v[A_TGT + gi] = min(1.0, v[A_TGT + gi] + 0.34)
            ui = UNIT_IDX.get(e.get("unit"))
            val = float(e.get("value") or 0.0)
            if ui is not None:
                v[A_UNIT + ui] = min(1.0, v[A_UNIT + ui] + 0.34)
                norm = _UNIT_NORM.get(e.get("unit"), 1.0)
                v[A_VSUM + ui] = min(3.0, v[A_VSUM + ui] + val / norm)
                v[A_VMAX + ui] = min(3.0, max(v[A_VMAX + ui], val / norm))
            for m in e.get("modifiers", []):
                mi = MOD_IDX.get(m)
                if mi is not None:
                    v[A_MOD + mi] = min(1.0, v[A_MOD + mi] + 0.34)
                if m == "scaling":
                    any_scaling = 1.0
            if e.get("scaling_ref"):
                any_scaling = 1.0
    v[SC_N_ATOMS] = min(1.0, len(atoms) / 4.0)
    v[SC_N_EFF] = min(1.0, n_eff / 8.0)
    v[SC_HAS_ATK] = has_atk
    v[SC_HAS_ABIL] = has_abil
    if base_damages:
        v[SC_MAX_DMG] = min(1.0, (max(base_damages) if base_damages else 0) / 300.0)
    v[SC_ANY_SCALING] = any_scaling
    return v


assert A_SCALAR + 6 == ATOM_FEAT_DIM
