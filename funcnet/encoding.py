"""funcnet encoding — asymmetric functional encoder + leak-fixed functional decoder.

Keeps bytrl's 24-word structure and the id-bag CONTRACT byte-identical for OWN-side words (so
encoder_bag/decoder_bag warm-start), and adds a PARALLEL per-word dense matrix carrying functional
features. Two asymmetries vs bytrl:

  ENCODER: opponent Pokemon emit NO card-id index (functional features only via the dense row); cursor
           advance is preserved identically so every other word's id indices stay aligned.
  DECODER: when an action's resolved card is OPPONENT-owned, NO card-id index is emitted; instead the
           opponent target's function (hp/maxHp/weakness/ex/...) rides in the dense row. This fixes the
           verified opponent-id leak on snipe/gust target options (DAMAGE_COUNTER_ANY / SWITCH on the
           opponent bench) while still giving the net signal to distinguish targets.

The dense rows are stashed ON the SparseVector instances (`sv.dense`).
Decision(sv_enc, sv_dec, ...) is constructed unchanged (bytrl.sample.Decision does no type check).
"""
import os
import random

import numpy as np

from cg.api import AreaType, OptionType

from .spec import (
    card_count, decoder_main_feature, decoder_attack_offset, decoder_card_offset,
    DENSE_DIM, DEC_FEAT_DIM, DEC_DENSE_DIM, DEC_ATOM_OFF, DEC_ATOM_DIM, ENC_DENSE_DIM, EMB_DIM,
    ATOM_FEAT_DIM, V_ATOM_OFF, V_TEXT_OFF, FUNC_FEAT_DIM,
    D_HP, D_MAXHP, D_ENERGY_COUNT, D_APPEARED, D_ENERGY_TYPES, D_HAS_TOOL,
    DEC_ATK_DMG, DEC_ATK_COSTLEN, DEC_ATK_COLORED, DEC_TGT_IS_OPP, DEC_TGT_HP,
    DEC_TGT_MAXHP, DEC_TGT_WEAK, DEC_TGT_EX, DEC_TGT_IS_POKE, DEC_TGT_RETREAT,
    DX_EFFDMG, DX_WEAK_MATCH, DX_RES_MATCH, DX_CAN_KO, DX_TGT_HP, DX_TGT_EX,
)
from .features import card_features, ATTACKS, CARD_TABLE
from funcnet.text.tables import atom_features, attack_atom_features, text_embedding, EMB_DIM as _TABLE_EMB_DIM
from cg.api import EnergyType, CardType

assert _TABLE_EMB_DIM == EMB_DIM, (
    f"text-embedding width {_TABLE_EMB_DIM} != spec.EMB_DIM {EMB_DIM}; re-freeze spec.EMB_DIM + re-embed")

_COLORLESS = int(EnergyType.COLORLESS)

DISABLE_RESOLVED = os.environ.get("FUNCNET_DISABLE_RESOLVED") == "1"


class SparseVector:
    """bytrl.SparseVector + a parallel per-word dense matrix (width `dense_dim`)."""
    def __init__(self, dense_dim: int):
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []
        self.dense: list[list[float]] = []
        self.dense_dim = dense_dim
        self.pos: int = 0

    def add(self, index: int, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))
        self.dense.append(np.zeros(self.dense_dim, dtype=np.float32))


def add_card(sv: SparseVector, card):
    """OWN-side: emit the id index (then advance the cursor). Bounds-safe: an id >= card_count emits no
    index (out of EmbeddingBag vocab) but keeps the cursor aligned."""
    if card is not None and 0 <= card.id < card_count:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def add_cards(sv: SparseVector, cards, value: float):
    if cards is not None:
        for card in cards:
            if 0 <= card.id < card_count:
                sv.add(card.id, value)
    sv.add_pos(card_count)


def _fill_pokemon_dense(row, poke):
    """Fill the CURRENT word's dense row with `poke`'s static + dynamic functional features."""
    feat = card_features(poke.id)
    row[:FUNC_FEAT_DIM] = feat
    row[D_HP] = (poke.hp or 0) / 400.0
    row[D_MAXHP] = (poke.maxHp or 0) / 400.0
    row[D_ENERGY_COUNT] = len(poke.energies) / 8.0
    row[D_APPEARED] = 1.0 if poke.appearThisTurn else 0.0
    for e in poke.energies:
        t = int(e)
        if 0 <= t <= 8:
            row[D_ENERGY_TYPES + t] += 1.0 / 4.0
    row[D_HAS_TOOL] = 1.0 if poke.tools else 0.0
    row[V_ATOM_OFF:V_ATOM_OFF + ATOM_FEAT_DIM] = atom_features(poke.id)
    row[V_TEXT_OFF:V_TEXT_OFF + EMB_DIM] = text_embedding(poke.id)


def add_pokemon(sv: SparseVector, poke, is_own: bool, view_dropout: float = 0.0,
                id_ablation: float = 0.0, ids_out=None):
    """Encode one Pokemon slot. Cursor advance is IDENTICAL across all three branches (None / own /
    opponent) = 1 + 3*card_count, so id-bag alignment for every later word is preserved.

    STAGE-3 id-ablation (id_ablation>0, TRAINING-ONLY): on OWN Pokemon words, with that probability ZERO
    the own id-bag (id + tools + energy indices) — keeping the cursor aligned with three add_pos(card_count)
    (the EXACT opponent-branch advance) — so the net cannot answer the aux 'exam' from a MEMORISED own id
    and is FORCED to read function/effect/views. The dense func+dyn+VIEWS row (_fill_pokemon_dense, below) is
    UNTOUCHED. This is our analogue of Cardsformer stat-randomisation; NEVER applied to opponent words
    (they carry no id anyway), the hand/deck/discard id-bags (zeroing those would blind the net to its own
    hand), or at eval/deploy (callers pass id_ablation=0).

    TRAINING-ONLY modality dropout (view_dropout>0): with that probability the OPPONENT's static stat
    slice (its card 'fingerprint': hp/type/cost/weakness/...) is zeroed, so the net cannot identify or
    react to that opponent from stats alone and is FORCED to read identity+effect from the semantic VIEWS
    (atom+text). Targets the lazy-student failure (every in-distribution opponent is uniquely stat-
    identifiable, so without this the views can stay unused). NEVER applied to own cards, and never at
    eval/deploy (callers pass view_dropout=0 there). Keeps the dynamic slice [60:74] (live hp/energy) and
    the views [74:] intact — only the static identity fingerprint is dropped.

    Optional id capture (ids_out, default None): when a list is passed, the per-word card id used for the
    atom/text lookup is appended in word order, once per Pokemon word; an empty slot or an out-of-range id
    appends -1 (the lookups write zeros for those). The agent never passes it."""
    if ids_out is not None:
        ids_out.append(poke.id if (poke is not None and 0 <= poke.id < card_count) else -1)
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
        return
    sv.add_single(0)
    sv.add_single((poke.hp or 0) / 400)
    if is_own:
        if id_ablation > 0.0 and random.random() < id_ablation:
            sv.add_pos(card_count)
            sv.add_pos(card_count)
            sv.add_pos(card_count)
        else:
            add_card(sv, poke)
            add_cards(sv, poke.tools, 1.0)
            add_cards(sv, poke.energyCards, 0.5)
    else:
        sv.add_pos(card_count)
        sv.add_pos(card_count)
        sv.add_pos(card_count)
    _fill_pokemon_dense(sv.dense[-1], poke)
    if (not is_own) and view_dropout > 0.0 and random.random() < view_dropout:
        sv.dense[-1][:FUNC_FEAT_DIM] = 0.0


def add_player(sv: SparseVector, ps):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs, your_deck: list[int], opp_belief=None, belief_word: bool = False,
                      view_dropout: float = 0.0, id_ablation: float = 0.0, ids_out=None) -> SparseVector:
    """Asymmetric encoder. Structurally identical to bytrl.get_encoder_input EXCEPT `is_own` is threaded
    into add_pokemon (own = the seat equal to yourIndex). belief_word kept for signature parity (Phase 1
    uses num_words=24; opp_belief unused since opponent ids are deliberately not embedded).

    view_dropout (TRAINING ONLY) is forwarded to add_pokemon -> opponent stat-fingerprint dropout. Eval
    and deploy MUST pass 0.0 (the default) so the live representation is never degraded.

    id_ablation (TRAINING ONLY, Stage-3) is forwarded to add_pokemon -> OWN-Pokemon id-bag ablation, the
    unblocker that forces the net to read function/effect/views instead of memorised own ids. Eval/deploy
    pass 0.0 (default). Applied INSIDE get_encoder_input at generation time so the recorded sv_enc already
    carries the ablation -> recorded==served (identical to how view_dropout works).

    Optional id capture (ids_out, default None): when a list is passed, it is filled with the 24 per-word
    card ids in word order (18 Pokemon words via add_pokemon, then -1 for the 6 non-Pokemon words: 2 players,
    hand, deck, stadium, turn info). The agent never passes it."""
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector(ENC_DENSE_DIM)
    for i in range(2):
        ps = state.players[i ^ your_index]
        is_own = ((i ^ your_index) == your_index)
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j], is_own, view_dropout=view_dropout,
                            id_ablation=id_ablation, ids_out=ids_out)
            else:
                add_pokemon(sv, None, is_own, view_dropout=view_dropout,
                            id_ablation=id_ablation, ids_out=ids_out)
            if j != 7:
                sv.pos = pos

    for i in range(2):
        ps = state.players[i ^ your_index]
        is_own = ((i ^ your_index) == your_index)
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0], is_own, view_dropout=view_dropout,
                        id_ablation=id_ablation, ids_out=ids_out)
        else:
            add_pokemon(sv, None, is_own, view_dropout=view_dropout,
                        id_ablation=id_ablation, ids_out=ids_out)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)
        if ids_out is not None:
            ids_out.append(-1)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)
    if ids_out is not None:
        ids_out.append(-1)

    sv.word_start()
    for cid in your_deck:
        if 0 <= cid < card_count:
            sv.add(cid, 0.25)
    sv.add_pos(card_count)
    if ids_out is not None:
        ids_out.append(-1)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)
    if ids_out is not None:
        ids_out.append(-1)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    sv.add_single(state.supporterPlayed)
    sv.add_single(state.stadiumPlayed)
    sv.add_single(state.energyAttached)
    sv.add_single(state.retreated)
    if ids_out is not None:
        ids_out.append(-1)
    return sv


def get_card(obs, area, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def decoder_main(sv: SparseVector, feature_index: int, card):
    if card is not None and 0 <= card.id < card_count:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


def decoder_card_id(sv: SparseVector, context, card_id: int):
    if 0 <= card_id < card_count:
        sv.add(decoder_card_offset + (decoder_main_feature + int(context)) * card_count + card_id, 1)


def _fill_attack_dense(row, attack_id):
    att = ATTACKS.get(attack_id)
    if att is None:
        return
    row[DEC_ATK_DMG] = (att.damage or 0) / 300.0
    row[DEC_ATK_COSTLEN] = len(att.energies) / 5.0
    row[DEC_ATK_COLORED] = 1.0 if any(int(e) != _COLORLESS for e in att.energies) else 0.0


def _fill_attack_resolved(row, attack_id, my_active, opp_active):
    """Engine-EXACT resolved-combat features for an ATTACK option, written into the DEC_EXTRA slice
    [DEC_FEAT_DIM:DEC_DENSE_DIM] (projected by the zero-init dec_extra_proj -> warm-start-safe). The
    implicit target of an attack is the OPPONENT'S ACTIVE Pokemon. The transform is the engine's MEASURED
    rule (tools/measure_damage_rule.py): weakness (attacker.energyType == defender.weakness) doubles the
    PRINTED damage; resistance (== defender.resistance) subtracts 30 (floor 0). Weakness/resistance key off
    the ATTACKING POKEMON's TYPE (energyType), not the attack's energy cost (verified 100% on fixed-damage
    attacks). Scaling/conditional attacks keep their dynamic damage in the text view + encoder energy/board
    features; here we feed the deterministic resolved core. No-op (slice stays 0) if attacker/defender/attack
    is unknown — identical to every non-attack option. `my_active`/`opp_active` are Pokemon obs instances."""
    if DISABLE_RESOLVED:
        return
    att = ATTACKS.get(attack_id)
    if att is None or my_active is None or opp_active is None:
        return
    cd_att = CARD_TABLE.get(getattr(my_active, "id", None))
    cd_def = CARD_TABLE.get(getattr(opp_active, "id", None))
    if cd_att is None or cd_def is None:
        return
    atype = int(cd_att.energyType) if cd_att.energyType is not None else None
    wmatch = atype is not None and cd_def.weakness is not None and int(cd_def.weakness) == atype
    rmatch = atype is not None and cd_def.resistance is not None and int(cd_def.resistance) == atype
    base = att.damage or 0
    eff = base * 2 if wmatch else base
    if rmatch:
        eff = max(0, eff - 30)
    def_hp = getattr(opp_active, "hp", None) or 0
    o = DEC_FEAT_DIM
    row[o + DX_EFFDMG] = eff / 300.0
    row[o + DX_WEAK_MATCH] = 1.0 if wmatch else 0.0
    row[o + DX_RES_MATCH] = 1.0 if rmatch else 0.0
    row[o + DX_CAN_KO] = 1.0 if (base > 0 and def_hp > 0 and eff >= def_hp) else 0.0
    row[o + DX_TGT_HP] = def_hp / 400.0
    row[o + DX_TGT_EX] = 1.0 if (cd_def.ex or getattr(cd_def, "megaEx", False)) else 0.0


def _fill_action_atom(row, vec):
    """Write a per-action effect-atom fingerprint into the decoder DEC_ATOM slice (projected by the zero-init
    dec_atom_proj -> warm-start-safe). vec is (ATOM_FEAT_DIM,) from tables.attack_atom_features / atom_features."""
    row[DEC_ATOM_OFF:DEC_ATOM_OFF + DEC_ATOM_DIM] = vec


def _fill_target_dense(row, card, is_opp):
    """Functional features of an action's TARGET card (used when the target is opponent-owned so we can
    suppress its id but still describe it). `card` is a Pokemon/Card observation instance."""
    row[DEC_TGT_IS_OPP] = 1.0 if is_opp else 0.0
    hp = getattr(card, "hp", None)
    maxhp = getattr(card, "maxHp", None)
    if hp is not None:
        row[DEC_TGT_HP] = hp / 400.0
    if maxhp is not None:
        row[DEC_TGT_MAXHP] = maxhp / 400.0
    cd = CARD_TABLE.get(getattr(card, "id", None))
    if cd is not None:
        row[DEC_TGT_IS_POKE] = 1.0 if cd.cardType == CardType.POKEMON else 0.0
        row[DEC_TGT_WEAK] = 1.0 if cd.weakness is not None else 0.0
        row[DEC_TGT_EX] = 1.0 if cd.ex else 0.0
        row[DEC_TGT_RETREAT] = (cd.retreatCost or 0) / 4.0


def _card_option_target(sv, obs, o, your_index, context):
    """Handle a CARD/TOOL_CARD/ENERGY(_CARD) option. OWN-owned -> emit id (warm-start path). OPPONENT-
    owned -> NO id; functional target features into the dense row instead (leak fix)."""
    is_opp = (o.playerIndex is not None and o.playerIndex != your_index)
    holder = get_card(obs, o.area, o.index, o.playerIndex)
    if o.type == OptionType.CARD:
        target = holder
    elif o.type == OptionType.TOOL_CARD:
        target = holder.tools[o.toolIndex] if holder is not None else None
    else:
        target = holder.energyCards[o.energyIndex] if holder is not None else None
    if target is None:
        return
    if is_opp:
        descr = holder if getattr(holder, "id", None) is not None else target
        _fill_target_dense(sv.dense[-1], descr, is_opp=True)
    else:
        decoder_card(sv, context, target)


def decoder_card(sv: SparseVector, context, card):
    if card is not None and getattr(card, "id", None) is not None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs, actions: list[list[int]]) -> SparseVector:
    """bytrl-identical id-bag layout for OWN-side action words (decoder_bag warm-start), PLUS a parallel
    dense row per action word carrying attack (damage,cost) and OPPONENT-target function (no opp id)."""
    sv = SparseVector(DEC_DENSE_DIM)
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    opp = obs.current.players[1 - your_index]
    my_active = ps.active[0] if ps.active else None
    opp_active = opp.active[0] if opp.active else None
    context = obs.select.context
    for action in actions:
        sv.word_start()
        row = sv.dense[-1]

        if len(action) == 0:
            sv.add(0, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                    _fill_attack_dense(row, o.attackId)
                    _fill_attack_resolved(row, o.attackId, my_active, opp_active)
                    _fill_action_atom(row, attack_atom_features(o.attackId))
                case OptionType.PLAY:
                    pc = ps.hand[o.index]
                    decoder_main(sv, 0, pc)
                    if pc is not None:
                        _fill_action_atom(row, atom_features(pc.id))
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    ec = get_card(obs, o.area, o.index, your_index)
                    decoder_main(sv, 3, ec)
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                    if ec is not None:
                        _fill_action_atom(row, atom_features(ec.id))
                case OptionType.ABILITY:
                    ac = get_card(obs, o.area, o.index, your_index)
                    decoder_main(sv, 5, ac)
                    if ac is not None:
                        _fill_action_atom(row, atom_features(ac.id))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD | OptionType.TOOL_CARD | OptionType.ENERGY_CARD | OptionType.ENERGY:
                    _card_option_target(sv, obs, o, your_index, context)
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)

    return sv
