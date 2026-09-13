"""funcnet4 (text-first) dimensions. Re-exports funcnet.spec's base dims UNCHANGED (shared vocab/feature
layout) and ADDS the text-first-only layout: per-word ROLE tags, per-action PLANE tags, the SET-group
scalar slots, and the wider decoder dense (id-bag deleted, card/attack TEXT appended).

The base func/atom/text feature layout is identical to funcnet.spec — we reuse the SAME precomputed tables
and the SAME _fill_* extractors; text-first only changes how those features are READ (shared MLP, no id-bag)
and adds the role/plane tags that the deleted id-bags used to carry implicitly."""
from funcnet.spec import *                       # noqa: F401,F403  base dims (card_count, ENC_DENSE_DIM, ...)
from funcnet.spec import (
    card_count, decoder_main_feature, decoder_card_offset, decoder_size, EMB_DIM, DEC_DENSE_DIM,
)

ROLE_PAD        = 0
ROLE_OWN_ACTIVE = 1
ROLE_OWN_BENCH  = 2
ROLE_OPP_ACTIVE = 3
ROLE_OPP_BENCH  = 4
ROLE_OWN_PLAYER = 5
ROLE_OPP_PLAYER = 6
ROLE_HAND       = 7
ROLE_DECK       = 8
ROLE_DISCARD    = 9
ROLE_STADIUM    = 10
ROLE_TURN       = 11
ROLE_HAND_CARD  = 12
N_ROLE          = 13
import os as _os
N_HAND_TOKENS   = int(_os.environ.get("FUNCNET_N_HAND_TOKENS", "12"))
NUM_WORDS       = 24 + N_HAND_TOKENS

SET_COUNT_OFF = 0
SET_MULT_OFF  = 1

PLANE_PAD       = 0
PLANE_ATTACK    = 14
PLANE_MAIN_OFF  = 15
PLANE_CTX_OFF   = PLANE_MAIN_OFF + decoder_main_feature
_N_CTX = (decoder_size - decoder_card_offset) // card_count
N_PLANE = PLANE_CTX_OFF + _N_CTX + 1

DEC_TEXT_OFF     = DEC_DENSE_DIM

DEC_NRG_OFF  = DEC_TEXT_OFF + EMB_DIM
DG_MISS_MIN  = 0
DG_MISS_MAX  = 1
DG_USEFUL    = 2
DEC_NRG_DIM  = 3
assert DG_USEFUL + 1 == DEC_NRG_DIM
NRG_FEAT = _os.environ.get("FUNCNET4_NRG_FEAT", "0") not in ("", "0")

DEC_CNT_OFF   = DEC_NRG_OFF + DEC_NRG_DIM
DH_CNT_TO_KO  = 0
DH_KILL_PICK  = 1
DH_KILL_BUDGET = 2
DH_PRIZE      = 3
DEC_CNT_DIM   = 4
assert DH_PRIZE + 1 == DEC_CNT_DIM
CNT_FEAT = _os.environ.get("FUNCNET4_CNT_FEAT", "0") not in ("", "0")

DEC_UNLOCK_OFF   = DEC_CNT_OFF + DEC_CNT_DIM
DJ_UNLOCK_DMG    = 0
DJ_UNLOCK_GAIN   = 1
DJ_UNLOCK_KILLS  = 2
DEC_UNLOCK_DIM   = 3
assert DJ_UNLOCK_KILLS + 1 == DEC_UNLOCK_DIM
UNLOCK_FEAT = _os.environ.get("FUNCNET4_UNLOCK_FEAT", "0") not in ("", "0")

DEC_TGT_OFF   = DEC_UNLOCK_OFF + DEC_UNLOCK_DIM
TG_NRG_N      = 0
TG_NRG_T0     = 1
TG_N_TYPES    = 9
TG_IS_ACTIVE  = 10
TG_N_TOOLS    = 11
TG_APPEARED   = 12
TG_HP         = 13
TG_MAXHP      = 14
DEC_TGT_DIM   = 15
assert TG_NRG_T0 + TG_N_TYPES == TG_IS_ACTIVE and TG_MAXHP + 1 == DEC_TGT_DIM
TGT_FEAT = _os.environ.get("FUNCNET4_TGT_FEAT", "0") not in ("", "0")

DEC_BENCH_OFF = DEC_TGT_OFF + DEC_TGT_DIM
BN_DMG        = 0
BN_KOS        = 1
BN_MIN_HP     = 2
DEC_BENCH_DIM = 3
assert BN_MIN_HP + 1 == DEC_BENCH_DIM
BENCH_FEAT = _os.environ.get("FUNCNET4_BENCH_FEAT", "0") not in ("", "0")

DEC_DENSE_DIM_TF = DEC_BENCH_OFF + DEC_BENCH_DIM


ROLE_DECK_REMAINING = 13
ROLE_PRIZE_KNOWN    = 14
PZ_RESOLVED_OFF = 2
DKR_FEAT = _os.environ.get("FUNCNET4_DKR_FEAT", "0") not in ("", "0")
PRZ_FEAT = _os.environ.get("FUNCNET4_PRZ_FEAT", "0") not in ("", "0")
assert not (PRZ_FEAT and not DKR_FEAT), \
    "FUNCNET4_PRZ_FEAT=1 requires FUNCNET4_DKR_FEAT=1"
if DKR_FEAT:
    N_ROLE = 15
    NUM_WORDS = NUM_WORDS + 2

ZONE_WORDS = (
    tuple(range(0, 8)),
    tuple(range(8, 16)),
    (16,),
    (17,),
    (18,),
    (19,),
    (20,),
    (21,),
    (22,),
    (23,),
    tuple(range(24, 24 + N_HAND_TOKENS)),
) + ((tuple(range(24 + N_HAND_TOKENS, NUM_WORDS)),) if DKR_FEAT else ())
N_ZONES = len(ZONE_WORDS)
assert sum(len(_z) for _z in ZONE_WORDS) == NUM_WORDS, "ZONE_WORDS must partition the words"

LOC_WORD      = 18

LC_N_TYPES    = 9
LC_DECK_T0    = 0
LC_DISC_T0    = 9
LC_PRZ_T0     = 18
LC_DECK_CLS0  = 27
LC_CLS_POKEMON   = 0
LC_CLS_BASIC     = 1
LC_CLS_SUPPORTER = 2
LC_CLS_ITEM      = 3
LC_CLS_TOOL      = 4
LC_CLS_STADIUM   = 5
LC_N_CLS      = 6
LC_DECK_NRG_SP = 33
LC_DISC_POKE  = 34
LC_DISC_NRG   = 35
ENC_LOC_DIM   = 36
assert LC_DECK_T0 + LC_N_TYPES == LC_DISC_T0 and LC_DISC_T0 + LC_N_TYPES == LC_PRZ_T0
assert LC_PRZ_T0 + LC_N_TYPES == LC_DECK_CLS0 and LC_DECK_CLS0 + LC_N_CLS == LC_DECK_NRG_SP
assert LC_DISC_NRG + 1 == ENC_LOC_DIM
LOC_FEAT = _os.environ.get("FUNCNET4_LOC_FEAT", "0") not in ("", "0")
ENC_LOC_OFF      = ENC_DENSE_DIM
ENC_DENSE_DIM_TF = ENC_DENSE_DIM + (ENC_LOC_DIM if LOC_FEAT else 0)
