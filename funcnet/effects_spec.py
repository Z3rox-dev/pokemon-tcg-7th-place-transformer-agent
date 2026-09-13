"""Target layout of the auxiliary effect-prediction head, a training-only task.

The agent imports only EFF_VOCAB_DIM and EFF_DELTA_DIM, which size the head so the checkpoint loads; the head
is never evaluated during play. The label has two groups, computed from the engine logs of the chosen action
(see scope_chosen_action_logs):
  * eff multi-hot (EFF_VOCAB_DIM=13, uint8): which semantic effects the action produced.
  * eff signed deltas (EFF_DELTA_DIM=8, float16): normalised magnitudes (damage / heal / counters / energy /
    draw + two obs-diff scalars: prizes_taken, ko_count).
"""
from dataclasses import dataclass

import numpy as np

from cg.api import LogType

_TURN_START        = int(LogType.TURN_START)
_TURN_END          = int(LogType.TURN_END)
_DRAW              = int(LogType.DRAW)
_DRAW_REVERSE      = int(LogType.DRAW_REVERSE)
_MOVE_CARD         = int(LogType.MOVE_CARD)
_MOVE_CARD_REVERSE = int(LogType.MOVE_CARD_REVERSE)
_SWITCH            = int(LogType.SWITCH)
_CHANGE            = int(LogType.CHANGE)
_PLAY              = int(LogType.PLAY)
_ATTACH            = int(LogType.ATTACH)
_EVOLVE            = int(LogType.EVOLVE)
_DEVOLVE           = int(LogType.DEVOLVE)
_ATTACK            = int(LogType.ATTACK)
_HP_CHANGE         = int(LogType.HP_CHANGE)
_POISONED          = int(LogType.POISONED)
_BURNED            = int(LogType.BURNED)
_ASLEEP            = int(LogType.ASLEEP)
_PARALYZED         = int(LogType.PARALYZED)
_CONFUSED          = int(LogType.CONFUSED)
_COIN              = int(LogType.COIN)
_RESULT            = int(LogType.RESULT)
_SHUFFLE           = int(LogType.SHUFFLE)
_HAS_BASIC         = int(LogType.HAS_BASIC_POKEMON)

_STATUS_TYPES = frozenset((_POISONED, _BURNED, _ASLEEP, _PARALYZED, _CONFUSED))
_NOISE = frozenset((_MOVE_CARD, _MOVE_CARD_REVERSE, _DRAW_REVERSE, _SHUFFLE, _HAS_BASIC))

EFF_VOCAB = (
    "deal_damage",
    "place_counter",
    "take_damage",
    "heal",
    "attach_energy",
    "evolve",
    "devolve",
    "switch_or_gust",
    "attack",
    "draw",
    "play_card",
    "status_inflict",
    "ko_or_result",
)
EFF_VOCAB_DIM = len(EFF_VOCAB)
EFF_IDX = {name: i for i, name in enumerate(EFF_VOCAB)}

EI_DEAL_DAMAGE    = EFF_IDX["deal_damage"]
EI_PLACE_COUNTER  = EFF_IDX["place_counter"]
EI_TAKE_DAMAGE    = EFF_IDX["take_damage"]
EI_HEAL           = EFF_IDX["heal"]
EI_ATTACH         = EFF_IDX["attach_energy"]
EI_EVOLVE         = EFF_IDX["evolve"]
EI_DEVOLVE        = EFF_IDX["devolve"]
EI_SWITCH         = EFF_IDX["switch_or_gust"]
EI_ATTACK         = EFF_IDX["attack"]
EI_DRAW           = EFF_IDX["draw"]
EI_PLAY           = EFF_IDX["play_card"]
EI_STATUS         = EFF_IDX["status_inflict"]
EI_KO             = EFF_IDX["ko_or_result"]

EFF_DELTA_DIM = 8
ED_DMG_TO_OPP      = 0
ED_DMG_TO_ME       = 1
ED_HEAL_TOTAL      = 2
ED_COUNTERS_PLACED = 3
ED_ENERGY_ATTACHED = 4
ED_CARDS_DRAWN     = 5
ED_PRIZES_TAKEN    = 6
ED_KO_COUNT        = 7

_N_DEC = 12429
_SCOUT_INCIDENCE = {
    EI_DEAL_DAMAGE:  1314,
    EI_PLACE_COUNTER: 1781,
    EI_TAKE_DAMAGE:   305,
    EI_HEAL:          330,
    EI_ATTACH:       1942,
    EI_EVOLVE:        934,
    EI_DEVOLVE:         0,
    EI_SWITCH:        505,
    EI_ATTACK:       1330,
    EI_DRAW:          991,
    EI_PLAY:         3738,
    EI_STATUS:          6,
    EI_KO:            422,
}


def _pos_weight():
    w = np.ones(EFF_VOCAB_DIM, dtype=np.float32)
    for k in range(EFF_VOCAB_DIM):
        n_k = _SCOUT_INCIDENCE.get(k, 1)
        w[k] = float(np.clip((_N_DEC - n_k) / max(n_k, 1), 1.0, 50.0))
    return w


EFF_POS_WEIGHT = _pos_weight()

assert EFF_VOCAB_DIM == 13, f"EFF_VOCAB_DIM changed to {EFF_VOCAB_DIM}; re-freeze deliberately"
assert EFF_VOCAB_DIM == len(EFF_VOCAB)
assert EFF_DELTA_DIM == 8
assert EFF_POS_WEIGHT.shape == (EFF_VOCAB_DIM,)
assert ED_KO_COUNT + 1 == EFF_DELTA_DIM


@dataclass(frozen=True)
class EffectLabel:
    """Compact, engine-exact label for the CHOSEN action's scoped effect. ~29 bytes (13 + 16)."""
    mh: np.ndarray
    dl: np.ndarray


def _is_opp(log, my_index):
    """Owner of the hp-changed/affected card != me. (HP_CHANGE.playerIndex = OWNER of the changed mon.)"""
    pi = log.get("playerIndex")
    return pi is not None and pi != my_index


def _is_mine(log, my_index):
    return log.get("playerIndex") == my_index


def scope_chosen_action_logs(logs, my_index):
    """CAVIL-1 cut: obs['logs'] are 'events since the last selection' and can SPAN TURN BOUNDARIES. Scope
    them to the CHOSEN ACTION's effect only. Pure, dict-based (obs['logs'] is a list of DICTS).

      1) DROP LEADING cascade: my turn's events begin AFTER the LAST TURN_START whose playerIndex==me.
         (no TURN_START(me) -> i0=0: the clean mid-my-turn case, e.g. ['ATTACK','HP_CHANGE',...]).
      2) TRUNCATE TRAILING cascade: my action's effects end at the FIRST TURN_END(me) at/after i0.
         (an ATTACK that ENDS the turn emits ATTACK+HP_CHANGE(+RESULT) BEFORE that TURN_END, so they are
         captured; the opponent's whole ensuing turn lives strictly after it and is excluded.)
      3) WITHIN the window, drop pure bookkeeping noise; KEEP my-turn effect logs (incl. opp-owned
         HP_CHANGE/SWITCH/status I CAUSED — they sit inside my window before my TURN_END, so they are mine).
    """
    ev = list(logs or [])
    i0 = 0
    for j, l in enumerate(ev):
        if l.get("type") == _TURN_START and l.get("playerIndex") == my_index:
            i0 = j + 1
    i1 = len(ev)
    for j in range(i0, len(ev)):
        if ev[j].get("type") == _TURN_END and ev[j].get("playerIndex") == my_index:
            i1 = j
            break
    window = ev[i0:i1]
    return [l for l in window if l.get("type") not in _NOISE]


def effect_label_from_scoped(scoped, my_index, prize_before, prize_after,
                             opp_pokes_before, opp_pokes_after):
    """Pack the scoped log set + the two obs-diff scalars into an EffectLabel. Pure, dict-based.

    `scoped` = scope_chosen_action_logs(...) output. prize_before/after = len(my prize) BEFORE/AFTER the
    action; opp_pokes_before/after = (len(active)+len(bench)) for the OPP seat BEFORE/AFTER. All engine-
    exact from the scoped logs except dims 6,7 (prizes_taken, ko_count) which are obs-diff.
    """
    mh = np.zeros(EFF_VOCAB_DIM, dtype=np.uint8)
    dl = np.zeros(EFF_DELTA_DIM, dtype=np.float32)

    for l in scoped:
        t = l.get("type")
        if t == _HP_CHANGE:
            v = l.get("value") or 0
            put = bool(l.get("putDamageCounter"))
            rec = bool(l.get("isRecover"))
            opp = _is_opp(l, my_index)
            mine = _is_mine(l, my_index)
            if v < 0 and opp and not put:
                mh[EI_DEAL_DAMAGE] = 1
                dl[ED_DMG_TO_OPP] += (-v)
            if put and opp:
                mh[EI_PLACE_COUNTER] = 1
                dl[ED_COUNTERS_PLACED] += 1.0
            if v < 0 and mine:
                mh[EI_TAKE_DAMAGE] = 1
                dl[ED_DMG_TO_ME] += (-v)
            if v > 0 or rec:
                mh[EI_HEAL] = 1
                if v > 0:
                    dl[ED_HEAL_TOTAL] += v
        elif t == _ATTACH:
            if _is_mine(l, my_index):
                mh[EI_ATTACH] = 1
                dl[ED_ENERGY_ATTACHED] += 1.0
        elif t == _EVOLVE:
            if _is_mine(l, my_index):
                mh[EI_EVOLVE] = 1
        elif t == _DEVOLVE:
            mh[EI_DEVOLVE] = 1
        elif t == _SWITCH or t == _CHANGE:
            mh[EI_SWITCH] = 1
        elif t == _ATTACK:
            if _is_mine(l, my_index):
                mh[EI_ATTACK] = 1
        elif t == _DRAW:
            if _is_mine(l, my_index):
                mh[EI_DRAW] = 1
                dl[ED_CARDS_DRAWN] += 1.0
        elif t == _PLAY:
            if _is_mine(l, my_index):
                mh[EI_PLAY] = 1
        elif t in _STATUS_TYPES:
            if not bool(l.get("isRecover")):
                mh[EI_STATUS] = 1
        elif t == _RESULT:
            if l.get("result") == my_index:
                mh[EI_KO] = 1

    prizes_taken = max(0, (prize_before or 0) - (prize_after or 0))
    ko_count = max(0, (opp_pokes_before or 0) - (opp_pokes_after or 0))
    dl[ED_PRIZES_TAKEN] = prizes_taken
    dl[ED_KO_COUNT] = ko_count
    if ko_count > 0:
        mh[EI_KO] = 1

    dl[ED_DMG_TO_OPP]      /= 100.0
    dl[ED_DMG_TO_ME]       /= 100.0
    dl[ED_HEAL_TOTAL]      /= 100.0
    dl[ED_COUNTERS_PLACED] /= 4.0
    dl[ED_ENERGY_ATTACHED] /= 4.0
    dl[ED_CARDS_DRAWN]     /= 8.0
    dl[ED_PRIZES_TAKEN]    /= 6.0
    dl[ED_KO_COUNT]        /= 2.0

    return EffectLabel(mh=mh, dl=dl.astype(np.float16))


def opp_pokes(obs_dict, my_index):
    """(len(active)+len(bench)) for the OPP seat, read from an obs DICT. Always visible (count only)."""
    p = obs_dict["current"]["players"][1 - my_index]
    return len(p.get("active") or []) + len(p.get("bench") or [])


def my_prize_count(obs_dict, my_index):
    """len(my prize), read from an obs DICT. Always visible from my perspective."""
    return len(obs_dict["current"]["players"][my_index].get("prize") or [])
