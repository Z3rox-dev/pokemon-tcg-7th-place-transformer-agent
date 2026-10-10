"""funcnet dimensions — single source of truth for the functional-feature layout.

Re-exports bytrl.spec's id-vocab sizes UNCHANGED (so encoder_bag/decoder_bag warm-start cleanly) and
adds the dense functional-feature layout (static per-card + dynamic per-instance for the encoder, and
attack + opponent-target function for the decoder).

LOCKED layout (frozen so a warm-started + later-trained ckpt stays consistent):
  STATIC card features (FUNC_FEAT_DIM=60) | DYNAMIC per-instance (DYN_DIM=14) -> DENSE_DIM=74
  DECODER per-action dense (DEC_FEAT_DIM) carries attack damage/cost AND opponent-target function
  (so the decoder NEVER needs the opponent card id — see encoding.get_decoder_input).
"""
from bytrl.spec import (                      # noqa: F401  (re-export id-vocab dims UNCHANGED)
    ALL_CARDS, CARD_TABLE,
    card_count, attack_count,
    num_words_encoder, encoder_size,
    decoder_main_feature, decoder_attack_offset, decoder_card_offset, decoder_size,
)
from funcnet.text.schema_spec import SCHEMA_FEAT_DIM   # legacy Stage-2 schema width (kept for back-compat)
from funcnet.text.atom_spec import ATOM_FEAT_DIM        # Stage-5 effect-atom fingerprint width (numpy-only)

# ---------------------------------------------------------------- STATIC per-card vector (FUNC_TABLE)
FUNC_FEAT_DIM = 60

# named offsets into the static 60-d card-feature vector
S_CARDTYPE      = 0    # 7  one-hot CardType 0..6
S_IS_POKEMON    = 7    # 1  gate
S_MAXHP         = 8    # 1  hp/300
S_RETREAT       = 9    # 1  retreatCost/4
S_STAGE         = 10   # 3  basic, stage1, stage2
S_ENERGYTYPE    = 13   # 11 one-hot EnergyType 0..10
S_WEAKNESS      = 24   # 8  one-hot weakness types 1..8
S_RESISTANCE    = 32   # 8  one-hot resistance types 1..8
S_FLAGS         = 40   # 4  ex, megaEx, tera, aceSpec
S_HAS_ABILITY   = 44   # 1
S_NUM_ATTACKS   = 45   # 1  len(attacks)/2
S_ATK0_DMG      = 46   # 1  dmg/300
S_ATK0_COSTLEN  = 47   # 1  len(cost)/5
S_ATK0_COST     = 48   # 9  per-type cost counts 0..8 (count/3)
S_ATK1_DMG      = 57   # 1  dmg/300
S_ATK1_COSTLEN  = 58   # 1  len(cost)/5
S_ATK1_COLORED  = 59   # 1  any non-colorless in attack1 cost
assert S_ATK1_COLORED + 1 == FUNC_FEAT_DIM

# ---------------------------------------------------------------- DYNAMIC per-instance slice
DYN_DIM = 14
# offsets relative to the START of the dense row (i.e. after the 60 static dims)
D_HP            = FUNC_FEAT_DIM + 0    # current hp/400
D_MAXHP         = FUNC_FEAT_DIM + 1    # current maxHp/400
D_ENERGY_COUNT  = FUNC_FEAT_DIM + 2    # len(energies)/8
D_APPEARED      = FUNC_FEAT_DIM + 3    # appearThisTurn
D_ENERGY_TYPES  = FUNC_FEAT_DIM + 4    # 9 per-type attached-energy counts 0..8 (count/4)
D_HAS_TOOL      = FUNC_FEAT_DIM + 13   # has-tool flag
assert D_HAS_TOOL + 1 == FUNC_FEAT_DIM + DYN_DIM

DENSE_DIM = FUNC_FEAT_DIM + DYN_DIM    # 74  (encoder per-word FUNCTIONAL dense width — Stage 1)

# ---------------------------------------------------------------- STAGE 2: semantic card-effect views
# Two STATIC-per-card views appended to each encoder POKEMON word's dense row (looked up by card id in
# funcnet/encoding.py; opponent cards get them too — by id LOOKUP, never an id embedding). Projected by a
# SEPARATE zero-init Linear (model.enc_view_proj) so warm-starting a Stage-1 checkpoint is bit-identical
# (the view contributes 0 at iter 0) AND the Stage-1 functional projection (enc_feat_proj) is preserved.
EMB_DIM = 768                                  # all-mpnet-base-v2 text-embedding width (asserted in encoding)
# STAGE 5: the rebuilt EFFECT-ATOM card fingerprint (ATOM_FEAT_DIM) REPLACES the old 63-d card-schema view.
# Encoder per-POKEMON view row = [atom fingerprint | text embedding]; projected by enc_view_proj (re-inits on a
# Stage-2 warm-start because the width changed 831->1035 -> the osfp warm-start drops shape-mismatched keys, so
# this projection re-learns from the better atom signal while the trunk/functional/id-bag warm-start intact).
VIEW_DIM = ATOM_FEAT_DIM + EMB_DIM             # 267 + 768 = 1035  (enc_view_proj input width)
ENC_DENSE_DIM = DENSE_DIM + VIEW_DIM           # 74 + 1035 = 1109  (FULL encoder dense-row width)
V_ATOM_OFF = DENSE_DIM                          # atom slots: [V_ATOM_OFF : V_ATOM_OFF+ATOM_FEAT_DIM]
V_TEXT_OFF = DENSE_DIM + ATOM_FEAT_DIM          # text slots: [V_TEXT_OFF : ENC_DENSE_DIM]
assert ENC_DENSE_DIM == DENSE_DIM + ATOM_FEAT_DIM + EMB_DIM

# ---------------------------------------------------------------- DECODER per-action dense
# Carries attack (damage,cost) AND opponent-target Pokemon function so the net can pick which opponent
# to snipe/gust WITHOUT the opponent id (the leak fix). Layout:
#   [0] attack damage/300
#   [1] attack cost-len/5
#   [2] attack colored-cost flag
#   [3] is-opponent-target flag (1 if the resolved card is opponent-owned)
#   [4] target current hp/400
#   [5] target current maxHp/400
#   [6] target weakness present flag (any weakness)
#   [7] target is-ex flag (prize value)
#   [8] target is-pokemon flag
#   [9] target retreatCost/4
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

# ---------------------------------------------------------------- DECODER resolved-combat EXTRA slice
# Stage-4 (RESOLVED DAMAGE): a SEPARATE dense slice carrying the engine's EXACT weakness/resistance damage
# transform for an ATTACK option (the implicit target is the opponent's active Pokemon). The transform was
# MEASURED from the native engine's own HP_CHANGE outputs, NOT assumed: weakness => damage*2, resistance =>
# damage-30 (floor 0), verified bit-exact on fixed-damage attacks (77/77 weakness, 14/15 resistance; the one
# miss is an attack whose text ignores resistance). See tools/measure_damage_rule.py. This is the one piece
# the net could NOT derive from the decoder token (the attacker's element vs the defender's weakness TYPE join
# is absent there). Scaling/conditional attacks (Phantom Dive etc.) keep their dynamic component in the text
# embedding + the encoder's live energy/board features; here we feed the deterministic resolved core only.
#
# Projected by a SEPARATE zero-init Linear (model.dec_extra_proj) so a Stage-2/3 warm-start is BIT-IDENTICAL
# at iter 0 (this slice contributes 0) AND the existing dec_feat_proj (the 10-d attack/target block, which
# carries trained weights) is preserved untouched. Filled ONLY for OptionType.ATTACK; zero for every other
# option (and for padding rows).
DEC_EXTRA_DIM = 6
DX_EFFDMG     = 0   # resolved effective dmg to opp active / 300  (base*2 if weak-match; -30 if resist-match; floor 0)
DX_WEAK_MATCH = 1   # 1 if attacker.energyType == defender.weakness
DX_RES_MATCH  = 2   # 1 if attacker.energyType == defender.resistance
DX_CAN_KO     = 3   # 1 if resolved effdmg >= defender CURRENT hp (exact for vanilla attacks; conservative for scaling)
DX_TGT_HP     = 4   # defender current hp / 400   (target context for attacks — was 0 before this slice)
DX_TGT_EX     = 5   # 1 if defender is ex / megaEx (the prize value of KOing it)
assert DX_TGT_EX + 1 == DEC_EXTRA_DIM

# ---------------------------------------------------------------- DECODER effect-atom token slice
# Stage-5: the per-ACTION effect-atom fingerprint (what the SPECIFIC chosen action does — for ATTACK the
# attack's atoms by attackId; for PLAY/ABILITY the card's atoms by id). Projected by a SEPARATE zero-init
# Linear (model.dec_atom_proj) -> warm-start bit-identical; closes the decoder gap from the feature audit
# (PLAY/EVOLVE/ability tokens previously carried no functional signal). Width = ATOM_FEAT_DIM.
DEC_ATOM_DIM = ATOM_FEAT_DIM
DEC_ATOM_OFF = DEC_FEAT_DIM + DEC_EXTRA_DIM    # atom slots: [DEC_ATOM_OFF : DEC_ATOM_OFF+DEC_ATOM_DIM]

DEC_DENSE_DIM = DEC_FEAT_DIM + DEC_EXTRA_DIM + DEC_ATOM_DIM   # 10 + 6 + 267 = 283 (FULL decoder dense-row width)
