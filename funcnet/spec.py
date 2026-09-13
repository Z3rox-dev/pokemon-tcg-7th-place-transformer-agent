"""Feature layout - the single source of truth for the functional-feature dimensions.
"""
from bytrl.spec import (                      # noqa: F401  (re-export id-vocab dims UNCHANGED)
    ALL_CARDS, CARD_TABLE,
    card_count, attack_count,
    num_words_encoder, encoder_size,
    decoder_main_feature, decoder_attack_offset, decoder_card_offset, decoder_size,
)
from funcnet.text.atom_spec import ATOM_FEAT_DIM

FUNC_FEAT_DIM = 60

S_CARDTYPE      = 0
S_IS_POKEMON    = 7
S_MAXHP         = 8
S_RETREAT       = 9
S_STAGE         = 10
S_ENERGYTYPE    = 13
S_WEAKNESS      = 24
S_RESISTANCE    = 32
S_FLAGS         = 40
S_HAS_ABILITY   = 44
S_NUM_ATTACKS   = 45
S_ATK0_DMG      = 46
S_ATK0_COSTLEN  = 47
S_ATK0_COST     = 48
S_ATK1_DMG      = 57
S_ATK1_COSTLEN  = 58
S_ATK1_COLORED  = 59
assert S_ATK1_COLORED + 1 == FUNC_FEAT_DIM

DYN_DIM = 14
D_HP            = FUNC_FEAT_DIM + 0
D_MAXHP         = FUNC_FEAT_DIM + 1
D_ENERGY_COUNT  = FUNC_FEAT_DIM + 2
D_APPEARED      = FUNC_FEAT_DIM + 3
D_ENERGY_TYPES  = FUNC_FEAT_DIM + 4
D_HAS_TOOL      = FUNC_FEAT_DIM + 13
assert D_HAS_TOOL + 1 == FUNC_FEAT_DIM + DYN_DIM

DENSE_DIM = FUNC_FEAT_DIM + DYN_DIM

EMB_DIM = 768
VIEW_DIM = ATOM_FEAT_DIM + EMB_DIM
ENC_DENSE_DIM = DENSE_DIM + VIEW_DIM
V_ATOM_OFF = DENSE_DIM
V_TEXT_OFF = DENSE_DIM + ATOM_FEAT_DIM
assert ENC_DENSE_DIM == DENSE_DIM + ATOM_FEAT_DIM + EMB_DIM

DEC_FEAT_DIM = 10
DEC_ATK_DMG       = 0
DEC_ATK_COSTLEN   = 1
DEC_ATK_COLORED   = 2
DEC_TGT_IS_OPP    = 3
DEC_TGT_HP        = 4
DEC_TGT_MAXHP     = 5
DEC_TGT_WEAK      = 6
DEC_TGT_EX        = 7
DEC_TGT_IS_POKE   = 8
DEC_TGT_RETREAT   = 9
assert DEC_TGT_RETREAT + 1 == DEC_FEAT_DIM

DEC_EXTRA_DIM = 6
DX_EFFDMG     = 0
DX_WEAK_MATCH = 1
DX_RES_MATCH  = 2
DX_CAN_KO     = 3
DX_TGT_HP     = 4
DX_TGT_EX     = 5
assert DX_TGT_EX + 1 == DEC_EXTRA_DIM

DEC_ATOM_DIM = ATOM_FEAT_DIM
DEC_ATOM_OFF = DEC_FEAT_DIM + DEC_EXTRA_DIM

DEC_DENSE_DIM = DEC_FEAT_DIM + DEC_EXTRA_DIM + DEC_ATOM_DIM
