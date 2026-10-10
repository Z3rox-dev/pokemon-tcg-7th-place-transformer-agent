"""LOCKED layout for the Stage-3 EFFECT-PREDICTION label (mirrors funcnet/text/schema_spec.py's style).

SINGLE SOURCE OF TRUTH for the train-time-only auxiliary effect head's TARGET. numpy-only (no torch) so
it is importable by both the encoding-free label code (selfplay/vecselfplay record paths, scout) and the
trainer (funcnet/losses.py). NEVER imported by agents/ — the head is dropped at deploy.

WHY this exists: the Stage-2 schema+text views are heavily USED but did not improve generalization (the
observed null). We gave the net the textbook (views as inputs) but never the exam (a task that forces it
to understand what abilities/attacks DO). This module defines that exam's answer key, derived ENGINE-EXACT
from obs.logs (94% of genuine decisions emit a real effect) scoped to the chosen action (CAVIL 1).

The label is two groups, computed from the SCOPED log set of the chosen action (see scope_chosen_action_logs):
  * eff multi-hot (EFF_VOCAB_DIM=13, uint8): which semantic effects the action produced.
  * eff signed deltas (EFF_DELTA_DIM=8, float16): normalised magnitudes (damage / heal / counters / energy /
    draw + two obs-diff scalars: prizes_taken, ko_count).

DEPLOY/EVAL paths never construct a label (Decision.eff defaults None); the trainer adds the aux BCE+Huber
term ONLY on recorded learner decisions that carry a label (lambda_aux-gated).
"""
from dataclasses import dataclass

import numpy as np

from cg.api import LogType

# ---------------------------------------------------------------- LogType int constants (frozen) -----
# Read once from the engine enum so the label code consumes plain int comparisons against dict logs.
_TURN_START        = int(LogType.TURN_START)         # 2
_TURN_END          = int(LogType.TURN_END)           # 3
_DRAW              = int(LogType.DRAW)                # 4
_DRAW_REVERSE      = int(LogType.DRAW_REVERSE)       # 5
_MOVE_CARD         = int(LogType.MOVE_CARD)          # 6
_MOVE_CARD_REVERSE = int(LogType.MOVE_CARD_REVERSE)  # 7
_SWITCH            = int(LogType.SWITCH)             # 8
_CHANGE            = int(LogType.CHANGE)             # 9
_PLAY              = int(LogType.PLAY)               # 10
_ATTACH            = int(LogType.ATTACH)             # 11
_EVOLVE            = int(LogType.EVOLVE)             # 12
_DEVOLVE           = int(LogType.DEVOLVE)            # 13
_ATTACK            = int(LogType.ATTACK)             # 15
_HP_CHANGE         = int(LogType.HP_CHANGE)          # 16
_POISONED          = int(LogType.POISONED)           # 17
_BURNED            = int(LogType.BURNED)             # 18
_ASLEEP            = int(LogType.ASLEEP)             # 19
_PARALYZED         = int(LogType.PARALYZED)          # 20
_CONFUSED          = int(LogType.CONFUSED)           # 21
_COIN              = int(LogType.COIN)               # 22
_RESULT            = int(LogType.RESULT)             # 23
_SHUFFLE           = int(LogType.SHUFFLE)            # 0
_HAS_BASIC         = int(LogType.HAS_BASIC_POKEMON)  # 1

_STATUS_TYPES = frozenset((_POISONED, _BURNED, _ASLEEP, _PARALYZED, _CONFUSED))
# pure bookkeeping noise dropped INSIDE the scoped window (cursor/shuffle/face-down/turn-tail)
_NOISE = frozenset((_MOVE_CARD, _MOVE_CARD_REVERSE, _DRAW_REVERSE, _SHUFFLE, _HAS_BASIC))

# ---------------------------------------------------------------- EFFECT MULTI-HOT VOCAB (index LOCKED)
EFF_VOCAB = (
    "deal_damage",      # 0  >=1 HP_CHANGE value<0 on an OPP-owned card, putDamageCounter falsy
    "place_counter",    # 1  >=1 HP_CHANGE putDamageCounter True on an OPP card (snipe / counter mechanic)
    "take_damage",      # 2  >=1 HP_CHANGE value<0 on a MY-owned card (recoil / opp-caused dmg on my mon)
    "heal",             # 3  >=1 HP_CHANGE value>0 (any owner) OR isRecover True on an HP_CHANGE
    "attach_energy",    # 4  >=1 ATTACH with playerIndex==me
    "evolve",           # 5  >=1 EVOLVE with playerIndex==me
    "devolve",          # 6  >=1 DEVOLVE (any owner) — rare but kept for unseen-card coverage
    "switch_or_gust",   # 7  >=1 SWITCH or CHANGE (any owner; owner split lives in deltas)
    "attack",           # 8  >=1 ATTACK with playerIndex==me
    "draw",             # 9  >=1 DRAW with playerIndex==me
    "play_card",        # 10 >=1 PLAY with playerIndex==me
    "status_inflict",   # 11 >=1 of POISONED/BURNED/ASLEEP/PARALYZED/CONFUSED, isRecover falsy (any owner)
    "ko_or_result",     # 12 a RESULT log with result==me, OR an HP_CHANGE that drops an OPP target to hp<=0
)
EFF_VOCAB_DIM = len(EFF_VOCAB)
EFF_IDX = {name: i for i, name in enumerate(EFF_VOCAB)}

# named indices (used by builders + tests)
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

# ---------------------------------------------------------------- SIGNED DELTAS (index LOCKED) -------
EFF_DELTA_DIM = 8
ED_DMG_TO_OPP      = 0   # (-sum value for HP_CHANGE value<0 on OPP)        / 100.0   (>=0)
ED_DMG_TO_ME       = 1   # (-sum value for HP_CHANGE value<0 on ME)         / 100.0   (>=0)
ED_HEAL_TOTAL      = 2   # ( sum value for HP_CHANGE value>0 any owner)     / 100.0   (>=0)
ED_COUNTERS_PLACED = 3   # ( count HP_CHANGE putDamageCounter True on OPP)  / 4.0
ED_ENERGY_ATTACHED = 4   # ( count ATTACH playerIndex==me)                  / 4.0
ED_CARDS_DRAWN     = 5   # ( count DRAW   playerIndex==me)                  / 8.0
ED_PRIZES_TAKEN    = 6   # clamp(my_prize_before - my_prize_after, 0)       / 6.0   (obs-diff)
ED_KO_COUNT        = 7   # clamp(opp_pokes_before - opp_pokes_after, 0)     / 2.0   (obs-diff)

# ---------------------------------------------------------------- POS_WEIGHT (inverse-frequency) -----
# clamp((N_dec - n_k)/max(n_k,1), 1, 50). RE-FROZEN 2026-06-22 from a REAL self-play iteration: 80 mirror
# games of funcnet_s2 / 11602 labeled decisions via tools/refreeze_pos_weight.py (the EXACT
# selfplay.generate label path, scoped CAVIL-1) — supersedes the earlier random-play scout (24g/3932dec).
# Self-play shifted the meaty effects a LOT vs random play (place_counter 35->1841 pw 50->5.3; deal_damage
# 136->1145 pw 27.9->9.1; attack 144->1151 pw 26.3->9.1; ko 25->753 pw 50->14.4) — random under-counted real
# combat, so this calibration matters. Rare bits (status n=6, devolve n=0 with these 2 pilot decks) still hit
# the 50 cap. Re-run tools/refreeze_pos_weight.py if the pilot set changes.
# Re-frozen 2026-06-22 from 80 mirror self-play games over the LABEL-TIMING-FIXED selfplay (Method A: finalize
# on the action's own post-vselect obs). Prior freeze (below, commented) was on the deferred/buggy timing; the
# aggregate shift is modest because attacks are ~10% of decisions, but per-attack labels were ~80% wrong before.
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


EFF_POS_WEIGHT = _pos_weight()   # np.float32 (13,)

# ---------------------------------------------------------------- module asserts (frozen) ------------
assert EFF_VOCAB_DIM == 13, f"EFF_VOCAB_DIM changed to {EFF_VOCAB_DIM}; re-freeze deliberately"
assert EFF_VOCAB_DIM == len(EFF_VOCAB)
assert EFF_DELTA_DIM == 8
assert EFF_POS_WEIGHT.shape == (EFF_VOCAB_DIM,)
assert ED_KO_COUNT + 1 == EFF_DELTA_DIM


# ---------------------------------------------------------------- EffectLabel + builders -------------
@dataclass(frozen=True)
class EffectLabel:
    """Compact, engine-exact label for the CHOSEN action's scoped effect. ~29 bytes (13 + 16)."""
    mh: np.ndarray   # uint8 (EFF_VOCAB_DIM,)
    dl: np.ndarray   # float16 (EFF_DELTA_DIM,)


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

    # obs-diff scalars (logs don't carry these cleanly)
    prizes_taken = max(0, (prize_before or 0) - (prize_after or 0))
    ko_count = max(0, (opp_pokes_before or 0) - (opp_pokes_after or 0))
    dl[ED_PRIZES_TAKEN] = prizes_taken
    dl[ED_KO_COUNT] = ko_count
    if ko_count > 0:                       # obs-diff KO cross-checks / sets the engine-exact KO bit
        mh[EI_KO] = 1

    # normalise the deltas to ~[-3,3] for Huber(beta=0.5)
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
