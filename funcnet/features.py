"""Offline cardId -> STATIC functional-feature vector (FUNC_TABLE) + bounds-checked lookups.

Static = everything derivable from CardData/Attack alone (type, hp, weakness, costs, ...). Per-instance
LIVE features (current hp, attached energies, tools, appearThisTurn) are added at encode time in
funcnet/encoding.py, NOT here.

Deploy safety: card_features(cid) NEVER raises on an unknown / out-of-range id — it returns a zero
vector and logs once (so an unseen-set id during a real match degrades gracefully instead of crashing).
"""
import numpy as np

from cg.api import CardType, EnergyType, all_attack
from .spec import (
    CARD_TABLE, card_count, FUNC_FEAT_DIM,
    S_CARDTYPE, S_IS_POKEMON, S_MAXHP, S_RETREAT, S_STAGE, S_ENERGYTYPE,
    S_WEAKNESS, S_RESISTANCE, S_FLAGS, S_HAS_ABILITY, S_NUM_ATTACKS,
    S_ATK0_DMG, S_ATK0_COSTLEN, S_ATK0_COST, S_ATK1_DMG, S_ATK1_COSTLEN, S_ATK1_COLORED,
)

ATTACKS = {a.attackId: a for a in all_attack()}

_N_CARDTYPE = 7
_N_ENERGY_STATIC = 11

_warned_ids = set()


def _energy_idx(e) -> int:
    """EnergyType -> clamped int in [0,10] (TEAM_ROCKET=11 -> COLORLESS=0; defensive on unseen)."""
    t = int(e)
    if t < 0 or t >= _N_ENERGY_STATIC:
        return int(EnergyType.COLORLESS)
    return t


def _attack_cost_counts(att, out, base):
    """Fill per-type energy-cost counts (types 0..8, 9 dims) for attack `att` into out[base:base+9]."""
    if att is None:
        return
    for e in att.energies:
        t = int(e)
        if 0 <= t <= 8:
            out[base + t] += 1.0 / 3.0


def _build_row(card) -> np.ndarray:
    out = np.zeros(FUNC_FEAT_DIM, dtype=np.float32)
    ct = int(card.cardType)
    if 0 <= ct < _N_CARDTYPE:
        out[S_CARDTYPE + ct] = 1.0
    is_poke = (card.cardType == CardType.POKEMON)
    out[S_IS_POKEMON] = 1.0 if is_poke else 0.0

    if is_poke:
        out[S_MAXHP] = (card.hp or 0) / 300.0
        out[S_RETREAT] = (card.retreatCost or 0) / 4.0
        out[S_STAGE + 0] = 1.0 if card.basic else 0.0
        out[S_STAGE + 1] = 1.0 if card.stage1 else 0.0
        out[S_STAGE + 2] = 1.0 if card.stage2 else 0.0
        if card.weakness is not None:
            w = int(card.weakness)
            if 1 <= w <= 8:
                out[S_WEAKNESS + (w - 1)] = 1.0
        if card.resistance is not None:
            r = int(card.resistance)
            if 1 <= r <= 8:
                out[S_RESISTANCE + (r - 1)] = 1.0

    out[S_ENERGYTYPE + _energy_idx(card.energyType)] = 1.0

    out[S_FLAGS + 0] = 1.0 if card.ex else 0.0
    out[S_FLAGS + 1] = 1.0 if card.megaEx else 0.0
    out[S_FLAGS + 2] = 1.0 if card.tera else 0.0
    out[S_FLAGS + 3] = 1.0 if card.aceSpec else 0.0
    out[S_HAS_ABILITY] = 1.0 if (card.skills and len(card.skills) > 0) else 0.0

    atk_ids = card.attacks or []
    out[S_NUM_ATTACKS] = len(atk_ids) / 2.0

    if len(atk_ids) >= 1:
        a0 = ATTACKS.get(atk_ids[0])
        if a0 is not None:
            out[S_ATK0_DMG] = (a0.damage or 0) / 300.0
            out[S_ATK0_COSTLEN] = len(a0.energies) / 5.0
            _attack_cost_counts(a0, out, S_ATK0_COST)
    if len(atk_ids) >= 2:
        a1 = ATTACKS.get(atk_ids[1])
        if a1 is not None:
            out[S_ATK1_DMG] = (a1.damage or 0) / 300.0
            out[S_ATK1_COSTLEN] = len(a1.energies) / 5.0
            out[S_ATK1_COLORED] = 1.0 if any(int(e) != int(EnergyType.COLORLESS)
                                             for e in a1.energies) else 0.0
    return out


FUNC_TABLE = np.zeros((card_count, FUNC_FEAT_DIM), dtype=np.float32)
for _cid, _card in CARD_TABLE.items():
    if 0 <= _cid < card_count:
        FUNC_TABLE[_cid] = _build_row(_card)

_ZERO = np.zeros(FUNC_FEAT_DIM, dtype=np.float32)


def card_features(cid: int) -> np.ndarray:
    """Bounds-checked static feature lookup. Returns a (60,) float32 row; zeros + log-once if the id is
    unknown / out of range (deploy safety: an unseen-set id must NEVER crash mid-match)."""
    if cid is None or cid < 0 or cid >= card_count:
        if cid not in _warned_ids:
            _warned_ids.add(cid)
            print(f"[funcnet.features] card id {cid} out of range [0,{card_count}); using zero features",
                  flush=True)
        return _ZERO
    return FUNC_TABLE[cid]
