"""Text-first encoding: the encoder words and the per-action decoder rows.

A card is read from its precomputed text and effect features, never from a learned per-card
identity, so cards the network has never seen still encode.
"""
import random
from collections import Counter

import numpy as np

from cg.api import AreaType, OptionType, SelectContext

from funcnet4.spec import (
    card_count, ENC_DENSE_DIM, ENC_DENSE_DIM_TF, ENC_LOC_OFF, ENC_LOC_DIM, LOC_WORD, LOC_FEAT,
    DEC_DENSE_DIM_TF, DEC_TEXT_OFF, EMB_DIM,
    ATOM_FEAT_DIM, V_ATOM_OFF, V_TEXT_OFF, FUNC_FEAT_DIM,
    DEC_ATOM_OFF, DEC_ATOM_DIM,
    DEC_NRG_OFF, DG_MISS_MIN, DG_MISS_MAX, DG_USEFUL, NRG_FEAT,
    DEC_CNT_OFF, DH_CNT_TO_KO, DH_KILL_PICK, DH_KILL_BUDGET, DH_PRIZE, CNT_FEAT,
    DEC_UNLOCK_OFF, DJ_UNLOCK_DMG, DJ_UNLOCK_GAIN, DJ_UNLOCK_KILLS, UNLOCK_FEAT,
    DEC_TGT_OFF, TG_NRG_N, TG_NRG_T0, TG_N_TYPES, TG_IS_ACTIVE, TG_N_TOOLS, TG_APPEARED,
    TG_HP, TG_MAXHP, TGT_FEAT,
    DEC_BENCH_OFF, BN_DMG, BN_KOS, BN_MIN_HP, BENCH_FEAT,
    SET_COUNT_OFF, SET_MULT_OFF,
    ROLE_OWN_ACTIVE, ROLE_OWN_BENCH, ROLE_OPP_ACTIVE, ROLE_OPP_BENCH,
    ROLE_OWN_PLAYER, ROLE_OPP_PLAYER, ROLE_HAND, ROLE_DECK, ROLE_STADIUM, ROLE_TURN,
    ROLE_HAND_CARD, N_HAND_TOKENS,
    ROLE_DECK_REMAINING, ROLE_PRIZE_KNOWN, PZ_RESOLVED_OFF, DKR_FEAT, PRZ_FEAT,
    PLANE_PAD, PLANE_ATTACK, PLANE_MAIN_OFF, PLANE_CTX_OFF,
)
from funcnet4.deck_tracker import unseen_counts
from funcnet4.loc_tail import fill_loc_dense
from funcnet.encoding import (
    SparseVector, get_card,
    _fill_pokemon_dense, _fill_attack_dense, _fill_attack_resolved,
    _fill_action_atom, _fill_target_dense,
)
from funcnet4.energy_mirror import (attach_nrg_scalars, cost_change_ctx,
                                    attack_missing_for_mon, provided_types as _provided_types)
from funcnet.text.tables import atom_features, attack_atom_features, text_embedding
from funcnet4.tables import attack_text_embedding
from funcnet.spec import CARD_TABLE as _CARD_TABLE_CNT
from funcnet.features import ATTACKS as _ATTACKS

_MUNKIDORI_CID = 112
_DARK_CID = 7
_ADRENA_VALUE = 30.0


def _resolved_dmg(base, cd_att, cd_def):
    """Printed damage after weakness and resistance, engine rules."""
    if not base:
        return 0.0
    et = int(cd_att.energyType) if (cd_att is not None and cd_att.energyType is not None) else None
    d = float(base)
    if cd_def is not None and et is not None:
        if cd_def.weakness is not None and int(cd_def.weakness) == et:
            d *= 2
        if cd_def.resistance is not None and int(cd_def.resistance) == et:
            d = max(0.0, d - 30)
    return d


def _mon_best_payable(mon, ctx, cd_def, extra_cid=None):
    """Highest resolved damage among the attacks this Pokemon can currently pay for."""
    cd_att = _CARD_TABLE_CNT.get(getattr(mon, "id", None))
    if cd_att is None:
        return 0.0
    best = 0.0
    for aid in (getattr(cd_att, "attacks", None) or []):
        att = _ATTACKS.get(int(aid))
        if att is None:
            continue
        miss = attack_missing_for_mon(mon, att, ctx, extra_energy_cid=extra_cid)
        if miss == 0:
            best = max(best, _resolved_dmg(int(att.damage or 0), cd_att, cd_def))
    return best


def _unlock_value(my_mons, ctx, cd_def, extra_cid=None):
    """UNLOCK: best payable damage over my Pokemon in play (plus the {D} ability value of Munkidori).
    extra_cid=None gives the current value; an energy card id gives the value if that energy were attached."""
    v = 0.0
    for mon in my_mons:
        v = max(v, _mon_best_payable(mon, ctx, cd_def, extra_cid=extra_cid))
    if extra_cid == _DARK_CID:
        for mon in my_mons:
            if int(getattr(mon, "id", -1)) == _MUNKIDORI_CID:
                has_dark = any(int(getattr(e, "id", -1)) == _DARK_CID
                               for e in (getattr(mon, "energyCards", None) or []))
                if not has_dark:
                    v = max(v, _ADRENA_VALUE)
    return v


def _prize_value_cnt(cid):
    """Prizes the opponent gives up on a knockout: mega ex 3, ex 2, otherwise 1."""
    cd = _CARD_TABLE_CNT.get(cid) if cid is not None else None
    if cd is None:
        return 1
    if getattr(cd, "megaEx", False):
        return 3
    return 2 if getattr(cd, "ex", False) else 1


P_DECK = 0; P_DISCARD = 1; P_HAND = 2; P_BENCH = 3; P_PRIZE = 4
P_POISON = 5; P_BURN = 6; P_ASLEEP = 7; P_PARAL = 8; P_CONFUSE = 9
P_DISC_CNT = 10; P_DISC_MULT = 11
T_TURN = 0; T_FIRST = 1; T_SUPPORTER = 2; T_STADIUM = 3; T_ENERGY = 4; T_RETREAT = 5


def _fill_set_dense(row, card_ids, count_off=SET_COUNT_OFF, mult_off=SET_MULT_OFF):
    """DeepSets group encoding: SUM of [atom|text] over card ids (VIEW slots) + count + multiplicity. count/
    mult are written to count_off/mult_off (default the dedicated SET slots 0/1; the PLAYER word passes
    P_DISC_CNT/P_DISC_MULT so they do NOT clobber its P_DECK/P_DISCARD scalars at slots 0/1). card_ids:
    iterable of int card ids."""
    cnt = Counter(c for c in card_ids if c is not None and 0 <= c < card_count)
    n = int(sum(cnt.values()))
    row[count_off] = n / 60.0
    row[mult_off] = (sum(1 for _c, k in cnt.items() if k >= 2) / max(1, len(cnt)))
    if n:
        atom = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
        txt = np.zeros(EMB_DIM, dtype=np.float32)
        for cid, k in cnt.items():
            atom += k * atom_features(cid)
            txt += k * text_embedding(cid)
        row[V_ATOM_OFF:V_ATOM_OFF + ATOM_FEAT_DIM] = atom
        row[V_TEXT_OFF:V_TEXT_OFF + EMB_DIM] = txt


def _card_ids(cards):
    return [getattr(c, "id", None) for c in (cards or [])]


def _valid_ids(card_ids):
    """The card ids that actually contribute to a DeepSets atom/text SUM (matches _fill_set_dense's filter:
    drop None / out-of-range). The FARM compact codec captures these so the learner can rebuild the atom+text
    view by table-lookup SUM, byte-identically to the direct dense."""
    return [int(c) for c in card_ids if c is not None and 0 <= c < card_count]


def get_encoder_input(obs, your_deck, view_dropout: float = 0.0, ids_out=None,
                      dkr: bool = False, prize_tracker=None, loc=None) -> SparseVector:
    """Text-first encoder. Same 24-word order as funcnet.encoding.get_encoder_input (8 own bench, 8 opp"""
    if loc is None:
        if LOC_FEAT:
            raise RuntimeError(
                "get_encoder_input: FUNCNET4_LOC_FEAT is set but loc= was not passed, so the LOC "
                "tail would be silently zero. Pass loc=LOC_FEAT (and prize_tracker=, or the 9 prize "
                "columns stay zero), or loc=False to disable the tail on purpose.")
        loc = False
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector(ENC_DENSE_DIM_TF)
    sv.role = []
    _rem = None
    if (loc and LOC_FEAT) or (dkr and DKR_FEAT):
        _rem = unseen_counts(obs, list(your_deck), your_index)

    def _poke_word(poke, is_own, role):
        sv.word_start()
        wids = []
        if poke is not None:
            _fill_pokemon_dense(sv.dense[-1], poke)
            wids = [poke.id]
            if (not is_own) and view_dropout > 0.0 and random.random() < view_dropout:
                sv.dense[-1][:FUNC_FEAT_DIM] = 0.0
        sv.role.append(role)
        if ids_out is not None:
            ids_out.append(_valid_ids(wids) if poke is not None else [])

    for i in range(2):
        ps = state.players[i ^ your_index]
        is_own = (i == 0)
        role = ROLE_OWN_BENCH if is_own else ROLE_OPP_BENCH
        for j in range(8):
            _poke_word(ps.bench[j] if j < len(ps.bench) else None, is_own, role)

    for i in range(2):
        ps = state.players[i ^ your_index]
        is_own = (i == 0)
        _poke_word(ps.active[0] if ps.active else None, is_own,
                   ROLE_OWN_ACTIVE if is_own else ROLE_OPP_ACTIVE)

    for i in range(2):
        ps = state.players[i ^ your_index]
        is_own = (i == 0)
        sv.word_start(); row = sv.dense[-1]
        row[P_DECK] = ps.deckCount / 60.0
        row[P_DISCARD] = len(ps.discard) / 60.0
        row[P_HAND] = ps.handCount / 8.0
        row[P_BENCH] = len(ps.bench) / 5.0
        row[P_PRIZE] = len(ps.prize) / 6.0
        row[P_POISON] = float(ps.poisoned); row[P_BURN] = float(ps.burned)
        row[P_ASLEEP] = float(ps.asleep); row[P_PARAL] = float(ps.paralyzed); row[P_CONFUSE] = float(ps.confused)
        _fill_set_dense(row, _card_ids(ps.discard), count_off=P_DISC_CNT, mult_off=P_DISC_MULT)
        if is_own and loc and LOC_FEAT:
            fill_loc_dense(row, ENC_LOC_OFF, _rem, prize_tracker, ps.discard, your_index)
        sv.role.append(ROLE_OWN_PLAYER if is_own else ROLE_OPP_PLAYER)
        if ids_out is not None:
            ids_out.append(_valid_ids(_card_ids(ps.discard)))

    sv.word_start()
    _fill_set_dense(sv.dense[-1], _card_ids(state.players[your_index].hand))
    sv.role.append(ROLE_HAND)
    if ids_out is not None:
        ids_out.append(_valid_ids(_card_ids(state.players[your_index].hand)))

    sv.word_start()
    _fill_set_dense(sv.dense[-1], list(your_deck))
    sv.role.append(ROLE_DECK)
    if ids_out is not None:
        ids_out.append(_valid_ids(list(your_deck)))

    sv.word_start()
    _fill_set_dense(sv.dense[-1], _card_ids(state.stadium))
    sv.role.append(ROLE_STADIUM)
    if ids_out is not None:
        ids_out.append(_valid_ids(_card_ids(state.stadium)))

    sv.word_start(); row = sv.dense[-1]
    row[T_TURN] = state.turn / 10.0
    row[T_FIRST] = float(state.firstPlayer == your_index)
    row[T_SUPPORTER] = float(state.supporterPlayed)
    row[T_STADIUM] = float(state.stadiumPlayed)
    row[T_ENERGY] = float(state.energyAttached)
    row[T_RETREAT] = float(state.retreated)
    sv.role.append(ROLE_TURN)
    if ids_out is not None:
        ids_out.append([])

    hand_cids = sorted(_valid_ids(_card_ids(state.players[your_index].hand)))
    for j in range(N_HAND_TOKENS):
        sv.word_start()
        if j < len(hand_cids):
            cid = hand_cids[j]
            row = sv.dense[-1]
            row[V_ATOM_OFF:V_ATOM_OFF + ATOM_FEAT_DIM] = atom_features(cid)
            row[V_TEXT_OFF:V_TEXT_OFF + EMB_DIM] = text_embedding(cid)
            if ids_out is not None:
                ids_out.append([cid])
        else:
            if ids_out is not None:
                ids_out.append([])
        sv.role.append(ROLE_HAND_CARD)

    if dkr and DKR_FEAT:
        rem_ids = list(_rem.elements())
        sv.word_start()
        _fill_set_dense(sv.dense[-1], rem_ids)
        sv.role.append(ROLE_DECK_REMAINING)
        if ids_out is not None:
            ids_out.append(_valid_ids(rem_ids))
        sv.word_start(); row = sv.dense[-1]
        prz_ids = []
        if PRZ_FEAT and prize_tracker is not None and prize_tracker.resolved:
            prz_ids = prize_tracker.prized_ids()
            _fill_set_dense(row, prz_ids)
            row[PZ_RESOLVED_OFF] = 1.0
        sv.role.append(ROLE_PRIZE_KNOWN)
        if ids_out is not None:
            ids_out.append(_valid_ids(prz_ids))
    return sv


_IN_PLAY_AREAS = (AreaType.ACTIVE, AreaType.BENCH)


def _in_play_mon(obs, area, index, player_index):
    """The in-play Pokemon at (area, index), or None if the area is neither active nor bench."""
    if area not in _IN_PLAY_AREAS or index is None or player_index is None:
        return None
    try:
        mon = get_card(obs, area, int(index), int(player_index))
    except (IndexError, KeyError, TypeError):
        return None
    return mon if getattr(mon, "id", None) is not None else None


def _fill_tgt_tail(row, mon, area):
    """TGT tail: the live state of the Pokemon this action acts on, so two copies of a card never encode identically."""
    if mon is None:
        return
    energies = getattr(mon, "energies", None) or []
    row[DEC_TGT_OFF + TG_NRG_N] += len(energies) / 8.0
    for e in energies:
        t = int(e)
        if 0 <= t < TG_N_TYPES:
            row[DEC_TGT_OFF + TG_NRG_T0 + t] += 1.0 / 4.0
    row[DEC_TGT_OFF + TG_IS_ACTIVE] += 1.0 if area == AreaType.ACTIVE else 0.0
    row[DEC_TGT_OFF + TG_N_TOOLS] += len(getattr(mon, "tools", None) or []) / 2.0
    row[DEC_TGT_OFF + TG_APPEARED] += 1.0 if getattr(mon, "appearThisTurn", False) else 0.0
    row[DEC_TGT_OFF + TG_HP] += (getattr(mon, "hp", None) or 0) / 400.0
    row[DEC_TGT_OFF + TG_MAXHP] += (getattr(mon, "maxHp", None) or 0) / 400.0


def _fill_action_target(row, obs, area, index, player_index, is_opp=False):
    """Describe the in-play Pokemon an action targets."""
    if not TGT_FEAT:
        return None
    mon = _in_play_mon(obs, area, index, player_index)
    if mon is None:
        return None
    _fill_target_dense(row, mon, is_opp=is_opp)
    _fill_tgt_tail(row, mon, area)
    return mon


def _fill_card_views(row, cid, cap=None):
    """ACCUMULATE the action's atom (DEC_ATOM slot) + text (DEC_TEXT slot) from a card id. Uses += (not =) so
    a MULTI-card selection (e.g. discard A+B) SUMS all chosen cards' views (DeepSets), instead of keeping only
    the LAST (the deleted id-bag used to accumulate; text-first must too). The row starts at zeros, so a
    single-card action is bit-identical to assignment. If cap is a list, the contributing cid is appended (FARM
    capture: the learner rebuilds this action's atom+text as the table SUM over cap)."""
    if cid is None or not (0 <= cid < card_count):
        return
    row[DEC_ATOM_OFF:DEC_ATOM_OFF + DEC_ATOM_DIM] += atom_features(cid)
    row[DEC_TEXT_OFF:DEC_TEXT_OFF + EMB_DIM] += text_embedding(cid)
    if cap is not None:
        cap.append(int(cid))


def get_decoder_input(obs, actions, ids_out=None, bench_vals=None) -> SparseVector:
    """Text-first decoder. Per action: a dense row (attack/target/resolved/atom + the acting card/attack"""
    sv = SparseVector(DEC_DENSE_DIM_TF)
    sv.plane = []
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    opp = obs.current.players[1 - your_index]
    my_active = ps.active[0] if ps.active else None
    opp_active = opp.active[0] if opp.active else None
    context = obs.select.context
    nrg_ctx = cost_change_ctx(obs.current, your_index) if NRG_FEAT else None
    cnt_place = CNT_FEAT and int(context) in (int(SelectContext.DAMAGE_COUNTER),
                                              int(SelectContext.DAMAGE_COUNTER_ANY))
    cnt_remain = getattr(obs.select, "remainDamageCounter", None) if cnt_place else None
    unlock_ctx = unlock_base = None
    unlock_cache = {}
    _unlock_mons = None
    if UNLOCK_FEAT:
        unlock_ctx = nrg_ctx if nrg_ctx is not None else cost_change_ctx(obs.current, your_index)
        _unlock_mons = [m for z in (ps.active, ps.bench) for m in (z or []) if m is not None]
        _cd_def_u = _CARD_TABLE_CNT.get(getattr(opp_active, "id", None)) if opp_active is not None else None
        unlock_base = _unlock_value(_unlock_mons, unlock_ctx, _cd_def_u)
        _uohp = int(getattr(opp_active, "hp", 0) or 0)

    for _ai, action in enumerate(actions):
        sv.word_start(); row = sv.dense[-1]
        if len(action) == 0:
            sv.plane.append(PLANE_PAD)
            if ids_out is not None:
                ids_out.append((-1, []))
            continue
        plane = PLANE_PAD
        cap_attack = -1
        cap_cids = []
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    plane = 1
                case OptionType.YES:
                    plane = 2
                case OptionType.NO:
                    plane = 3
                case OptionType.SPECIAL_CONDITION:
                    plane = 4 + int(o.specialConditionType)
                case OptionType.NUMBER:
                    plane = 9 + min(int(o.number), 4)
                case OptionType.ATTACK:
                    plane = PLANE_ATTACK
                    _fill_attack_dense(row, o.attackId)
                    _fill_attack_resolved(row, o.attackId, my_active, opp_active)
                    _fill_action_atom(row, attack_atom_features(o.attackId))
                    row[DEC_TEXT_OFF:DEC_TEXT_OFF + EMB_DIM] = attack_text_embedding(o.attackId)
                    cap_attack = int(o.attackId)
                    if BENCH_FEAT and bench_vals:
                        _bv = bench_vals.get(int(o.attackId))
                        if _bv is not None:
                            row[DEC_BENCH_OFF + BN_DMG] = float(_bv[0]) / 400.0
                            row[DEC_BENCH_OFF + BN_KOS] = float(_bv[1]) / 3.0
                            row[DEC_BENCH_OFF + BN_MIN_HP] = float(_bv[2]) / 400.0
                case OptionType.PLAY:
                    plane = PLANE_MAIN_OFF + 0
                    _fill_card_views(row, getattr(ps.hand[o.index], "id", None), cap=cap_cids)
                case OptionType.ATTACH:
                    plane = PLANE_MAIN_OFF + 1
                    ecard = get_card(obs, o.area, o.index, your_index)
                    tmon = get_card(obs, o.inPlayArea, o.inPlayIndex, your_index)
                    _fill_card_views(row, getattr(ecard, "id", None), cap=cap_cids)
                    _fill_target_dense(row, tmon, is_opp=False)
                    _fill_action_target(row, obs, o.inPlayArea, o.inPlayIndex, your_index)
                    if NRG_FEAT and tmon is not None:
                        nrg = attach_nrg_scalars(getattr(ecard, "id", None), tmon, nrg_ctx)
                        if nrg is not None:
                            row[DEC_NRG_OFF + DG_MISS_MIN] = nrg[0]
                            row[DEC_NRG_OFF + DG_MISS_MAX] = nrg[1]
                            row[DEC_NRG_OFF + DG_USEFUL] = nrg[2]
                case OptionType.EVOLVE:
                    plane = PLANE_MAIN_OFF + 3
                    _fill_card_views(row, getattr(get_card(obs, o.area, o.index, your_index), "id", None), cap=cap_cids)
                    _fill_target_dense(row, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index), is_opp=False)
                    _fill_action_target(row, obs, o.inPlayArea, o.inPlayIndex, your_index)
                case OptionType.ABILITY:
                    plane = PLANE_MAIN_OFF + 5
                    _fill_card_views(row, getattr(get_card(obs, o.area, o.index, your_index), "id", None), cap=cap_cids)
                    _fill_action_target(row, obs, o.area, o.index, your_index)
                case OptionType.DISCARD:
                    plane = PLANE_MAIN_OFF + 6
                    _fill_card_views(row, getattr(get_card(obs, o.area, o.index, your_index), "id", None), cap=cap_cids)
                    _fill_action_target(row, obs, o.area, o.index, your_index)
                case OptionType.RETREAT:
                    plane = PLANE_MAIN_OFF + 7
                    _fill_target_dense(row, my_active, is_opp=False)
                    if TGT_FEAT:
                        _fill_tgt_tail(row, my_active, AreaType.ACTIVE)
                case OptionType.CARD | OptionType.TOOL_CARD | OptionType.ENERGY_CARD | OptionType.ENERGY:
                    plane = PLANE_CTX_OFF + int(context)
                    is_opp = (o.playerIndex is not None and o.playerIndex != your_index)
                    holder = get_card(obs, o.area, o.index, o.playerIndex)
                    if o.type == OptionType.CARD:
                        target = holder
                    elif o.type == OptionType.TOOL_CARD:
                        target = holder.tools[o.toolIndex] if holder is not None else None
                    else:
                        target = holder.energyCards[o.energyIndex] if holder is not None else None
                    if target is not None:
                        if is_opp:
                            descr = holder if getattr(holder, "id", None) is not None else target
                            _fill_target_dense(row, descr, is_opp=True)
                            _fill_card_views(row, getattr(descr, "id", None), cap=cap_cids)
                            _fill_action_target(row, obs, o.area, o.index, o.playerIndex, is_opp=True)
                            if cnt_place and o.type == OptionType.CARD:
                                _hp = getattr(holder, "hp", None)
                                if _hp is not None and _hp > 0:
                                    _need = -(-int(_hp) // 10)
                                    row[DEC_CNT_OFF + DH_CNT_TO_KO] = _need / 30.0
                                    row[DEC_CNT_OFF + DH_KILL_PICK] = 1.0 if _need == 1 else 0.0
                                    row[DEC_CNT_OFF + DH_KILL_BUDGET] = (
                                        1.0 if cnt_remain is not None and cnt_remain >= _need else 0.0)
                                    row[DEC_CNT_OFF + DH_PRIZE] = _prize_value_cnt(
                                        getattr(holder, "id", None)) / 3.0
                        else:
                            _fill_card_views(row, getattr(target, "id", None), cap=cap_cids)
                            _fill_action_target(row, obs, o.area, o.index,
                                                o.playerIndex if o.playerIndex is not None else your_index)
                            if UNLOCK_FEAT:
                                _ecid = getattr(target, "id", None)
                                if _ecid is not None and _ecid in unlock_cache:
                                    _u = unlock_cache[_ecid]
                                elif _ecid is not None:
                                    _u = (_unlock_value(_unlock_mons, unlock_ctx, _cd_def_u,
                                                        extra_cid=int(_ecid))
                                          if _provided_types(int(_ecid)) is not None else None)
                                    unlock_cache[_ecid] = _u
                                else:
                                    _u = None
                                if _u is not None:
                                    row[DEC_UNLOCK_OFF + DJ_UNLOCK_DMG] = _u / 400.0
                                    row[DEC_UNLOCK_OFF + DJ_UNLOCK_GAIN] = max(0.0, _u - unlock_base) / 400.0
                                    row[DEC_UNLOCK_OFF + DJ_UNLOCK_KILLS] = (
                                        1.0 if _uohp > 0 and _u >= _uohp else 0.0)
                case OptionType.SKILL:
                    plane = PLANE_CTX_OFF + int(context)
                    _fill_card_views(row, o.cardId, cap=cap_cids)
        sv.plane.append(plane)
        if ids_out is not None:
            ids_out.append((cap_attack, cap_cids))
    return sv
