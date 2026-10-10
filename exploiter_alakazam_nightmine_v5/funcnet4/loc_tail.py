"""LOC-TAIL (2026-08-07) — DOVE sono le mie carte, per categoria e per zona, come scalari espliciti.

Perche' esiste (misurato): una sonda lineare sulle 29 carte del mazzo legge UNA sola carta con
R2>0,5 (mediana 0,148), contro R2 0,88 di uno scalare ESPLICITO come deckCount. Il residuo del
mazzo la rete ce l'ha gia' — la DKR-tail lo calcola ESATTO in Python — ma glielo consegniamo con
_fill_set_dense, cioe' una SOMMA DeepSets su tutte le carte rimaste: per sapere "e' rimasta una
Buio?" dovrebbe accorgersi che manca UN addendo dentro una somma di 14. Il conto e' esatto,
l'ACCESSO no. Qui lo stesso conto diventa una colonna dedicata per (categoria, zona).

Tabelle precalcolate all'import e indicizzate per cardId (stesso pattern di funcnet/features.py
FUNC_TABLE): riga 0 = padding, gli id validi sono 1..1267 (card_count=1268).

DEPLOY-SAFE: nessuna lettura di env, nessun import di torch — il modulo entra nel bundle Kaggle.
"""
import numpy as np

from cg.api import CardType

from funcnet.spec import CARD_TABLE, card_count
from funcnet4.spec import (ENC_LOC_DIM, LC_N_TYPES, LC_DECK_T0, LC_DISC_T0, LC_PRZ_T0,
                           LC_DECK_CLS0, LC_CLS_POKEMON, LC_CLS_BASIC, LC_CLS_SUPPORTER,
                           LC_CLS_ITEM, LC_CLS_TOOL, LC_CLS_STADIUM, LC_DECK_NRG_SP,
                           LC_DISC_POKE, LC_DISC_NRG)

_NRG_TYPE      = np.full(card_count, -1, np.int8)    # energyType 0..8 SOLO per cardType==BASIC_ENERGY
_IS_SP_NRG     = np.zeros(card_count, bool)          # cardType==SPECIAL_ENERGY (+ base con tipo fuori 0..8)
_CLS           = np.full(card_count, -1, np.int8)    # indice LC_CLS_* per Pokemon/Supporter/Item/Tool/Stadium
_IS_BASIC_POKE = np.zeros(card_count, bool)
_IS_POKE       = np.zeros(card_count, bool)
_CT2CLS = {int(CardType.POKEMON): LC_CLS_POKEMON, int(CardType.SUPPORTER): LC_CLS_SUPPORTER,
           int(CardType.ITEM): LC_CLS_ITEM, int(CardType.TOOL): LC_CLS_TOOL,
           int(CardType.STADIUM): LC_CLS_STADIUM}
for _cid, _cd in CARD_TABLE.items():
    _cid = int(_cid)
    if not (0 <= _cid < card_count):
        continue
    _ct = int(_cd.cardType)                          # ⚠ int PURO, non un IntEnum (cg/utils.to_dataclass
    if _ct == int(CardType.BASIC_ENERGY):            #    non coercizza): isinstance(_cd.cardType, CardType)
        _t = int(_cd.energyType or 0)                #    e' False. Confrontare sempre int con int.
        if 0 <= _t < LC_N_TYPES:
            _NRG_TYPE[_cid] = _t
        else:                                        # non esiste nel pool odierno; se comparisse NON
            _IS_SP_NRG[_cid] = True                  # va persa (romperebbe la partizione del mazzo)
    elif _ct == int(CardType.SPECIAL_ENERGY):
        _IS_SP_NRG[_cid] = True
    else:
        _CLS[_cid] = _CT2CLS.get(_ct, -1)
        if _ct == int(CardType.POKEMON):
            _IS_POKE[_cid] = True
            _IS_BASIC_POKE[_cid] = bool(getattr(_cd, "basic", False))
# ⚠ `basic` da solo NON basta: 5 ITEM (i fossili Antique 1099/1136/1138/1150/1151) hanno basic=True e
#   hp=60. Il gate su cardType==POKEMON li tiene fuori (restano ITEM, come li tratta il motore).
# ⚠ NON classificare per NOME ne' per energyType: 4 Special Energy hanno energyType tipato
#   (Grow Grass=1, Telepath=5, Rock Fighting=6, Legacy=10) e Crispin NON le pesca; e "Energy" nel nome
#   ha 7 falsi positivi ITEM (Energy Switch, Energy Retrieval, ...). Vedi tools/energy_choice_probe.py
#   per l'euristica DA NON COPIARE.


def fill_loc_dense(row, off, rem, prize_tracker, discard_cards, my_index, stats=None):
    """Scrive le ENC_LOC_DIM colonne a partire da `off` (in-place; la riga parte da zero).

      row            : riga dense della parola own-player (np.float32, larga ENC_DENSE_DIM_TF)
      off            : ENC_LOC_OFF
      rem            : Counter cid->n di deck_tracker.unseen_counts = (mazzo UNION premi)
      prize_tracker  : DeckPrizeTracker o None; i premi contano SOLO se .resolved
      discard_cards  : ps.discard del MIO seat (lista PIATTA di Card — stessa convenzione di
                       P_DISCARD e della DeepSets scarti nella stessa parola)
      my_index       : seat proprietario (filtro ownership identico a unseen_counts._sub)
      stats          : dict opzionale di telemetria (gate H): carte scartate dal filtro ownership,
                       Pokemon negli scarti che portano ancora carte annidate.

    Scala: conteggi di CARTE / 60.0 su TUTTE le colonne (convenzione _fill_set_dense)."""
    prized = (prize_tracker.prized
              if (prize_tracker is not None and getattr(prize_tracker, "resolved", False)) else None)
    # Counter - Counter droppa i <= 0: "nel mazzo" non puo' mai andare negativo. Con la PRZ non
    # risolta `prized` e' None -> mazzo = unseen = stima ALTA (prudente e dichiarata).
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
                row[off + LC_DECK_CLS0 + LC_CLS_BASIC] += k     # SOTTOINSIEME di POKEMON, non partizione
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
        if getattr(card, "playerIndex", None) != my_index:      # stesso filtro di unseen_counts._sub
            if stats is not None:
                stats["disc_foreign"] = stats.get("disc_foreign", 0) + 1
            continue
        if stats is not None and (getattr(card, "energyCards", None) or getattr(card, "tools", None)
                                  or getattr(card, "preEvolution", None)):
            # PUNTO APERTO dichiarato: se il motore lasciasse carte ANNIDATE su un Pokemon negli
            # scarti, le colonne-scarti le perderebbero (qui si conta piatto, come P_DISCARD).
            stats["disc_nested"] = stats.get("disc_nested", 0) + 1
        cid = int(cid)
        t = int(_NRG_TYPE[cid])
        if t >= 0:
            row[off + LC_DISC_T0 + t] += 1
            row[off + LC_DISC_NRG] += 1
        elif _IS_SP_NRG[cid]:
            row[off + LC_DISC_NRG] += 1                          # speciali: SOLO nel totale energie
        elif _IS_POKE[cid]:
            row[off + LC_DISC_POKE] += 1
    row[off:off + ENC_LOC_DIM] /= 60.0                # la /60 vive in UN posto solo
