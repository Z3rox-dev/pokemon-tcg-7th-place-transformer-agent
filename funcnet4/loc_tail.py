"""LOC tail: where my cards are, by category and by zone, as explicit scalars.

This flag decides the encoder row width (1109 without, 1145 with), so a mismatch does not degrade
quietly - the checkpoint migration refuses the crop and the load fails.
"""
import numpy as np

from cg.api import CardType

from funcnet.spec import CARD_TABLE, card_count
from funcnet4.spec import (ENC_LOC_DIM, LC_N_TYPES, LC_DECK_T0, LC_DISC_T0, LC_PRZ_T0,
                           LC_DECK_CLS0, LC_CLS_POKEMON, LC_CLS_BASIC, LC_CLS_SUPPORTER,
                           LC_CLS_ITEM, LC_CLS_TOOL, LC_CLS_STADIUM, LC_DECK_NRG_SP,
                           LC_DISC_POKE, LC_DISC_NRG)

_NRG_TYPE      = np.full(card_count, -1, np.int8)
_IS_SP_NRG     = np.zeros(card_count, bool)
_CLS           = np.full(card_count, -1, np.int8)
_IS_BASIC_POKE = np.zeros(card_count, bool)
_IS_POKE       = np.zeros(card_count, bool)
_CT2CLS = {int(CardType.POKEMON): LC_CLS_POKEMON, int(CardType.SUPPORTER): LC_CLS_SUPPORTER,
           int(CardType.ITEM): LC_CLS_ITEM, int(CardType.TOOL): LC_CLS_TOOL,
           int(CardType.STADIUM): LC_CLS_STADIUM}
for _cid, _cd in CARD_TABLE.items():
    _cid = int(_cid)
    if not (0 <= _cid < card_count):
        continue
    _ct = int(_cd.cardType)
    if _ct == int(CardType.BASIC_ENERGY):
        _t = int(_cd.energyType or 0)
        if 0 <= _t < LC_N_TYPES:
            _NRG_TYPE[_cid] = _t
        else:
            _IS_SP_NRG[_cid] = True
    elif _ct == int(CardType.SPECIAL_ENERGY):
        _IS_SP_NRG[_cid] = True
    else:
        _CLS[_cid] = _CT2CLS.get(_ct, -1)
        if _ct == int(CardType.POKEMON):
            _IS_POKE[_cid] = True
            _IS_BASIC_POKE[_cid] = bool(getattr(_cd, "basic", False))


def fill_loc_dense(row, off, rem, prize_tracker, discard_cards, my_index, stats=None):
    """Write the LOC columns in place, starting at the given offset."""
    prized = (prize_tracker.prized
              if (prize_tracker is not None and getattr(prize_tracker, "resolved", False)) else None)
    deck = (rem - prized) if prized else rem
    for cid, k in deck.items():
        cid = int(cid)
        if not (0 <= cid < card_count):
            continue
        t = int(_NRG_TYPE[cid])
        if t >= 0:
            row[off + LC_DECK_T0 + t] += k
        elif _IS_SP_NRG[cid]:
            row[off + LC_DECK_NRG_SP] += k
        else:
            c = int(_CLS[cid])
            if c >= 0:
                row[off + LC_DECK_CLS0 + c] += k
            if _IS_BASIC_POKE[cid]:
                row[off + LC_DECK_CLS0 + LC_CLS_BASIC] += k
    if prized:
        for cid, k in prized.items():
            cid = int(cid)
            t = int(_NRG_TYPE[cid]) if 0 <= cid < card_count else -1
            if t >= 0:
                row[off + LC_PRZ_T0 + t] += k
    for card in (discard_cards or []):
        cid = getattr(card, "id", None)
        if cid is None or not (0 <= int(cid) < card_count):
            continue
        if getattr(card, "playerIndex", None) != my_index:
            if stats is not None:
                stats["disc_foreign"] = stats.get("disc_foreign", 0) + 1
            continue
        if stats is not None and (getattr(card, "energyCards", None) or getattr(card, "tools", None)
                                  or getattr(card, "preEvolution", None)):
            stats["disc_nested"] = stats.get("disc_nested", 0) + 1
        cid = int(cid)
        t = int(_NRG_TYPE[cid])
        if t >= 0:
            row[off + LC_DISC_T0 + t] += 1
            row[off + LC_DISC_NRG] += 1
        elif _IS_SP_NRG[cid]:
            row[off + LC_DISC_NRG] += 1
        elif _IS_POKE[cid]:
            row[off + LC_DISC_POKE] += 1
    row[off:off + ENC_LOC_DIM] /= 60.0
