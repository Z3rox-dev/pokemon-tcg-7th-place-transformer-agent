"""Determinization of hidden information over the engine Search API.

Only determinize() and its helpers remain. The Gumbel search that used to live here - 18
definitions, 375 lines - was removed from this release: with --recurrent gru the learner refused
--search through an explicit assert, and the submitted model is recurrent, so it could never run.
determinize() itself is still used by the lethal finder, by bench_counters and by the rollout
search to open a determinized world.
"""
import math
import random
from collections import Counter

import numpy as np
import torch

from cg.api import search_begin, search_step, search_end
from bytrl.actions import enumerate_actions
from bytrl.config import DEFAULT

from .encoding import get_encoder_input, get_decoder_input
from funcnet.spec import CARD_TABLE

POLICY_TEMP = DEFAULT.model.policy_temp

SNORLAX_ID = 1072
BASIC_ENERGY_ID = 1

MAX_AUTO_STEPS = 16
STEP_BUDGET_MULT = 4


def _is_basic_pokemon(cid):
    c = CARD_TABLE.get(cid)
    return bool(c and getattr(c, "basic", False))


def _mon_cards(counter, mons):
    for p in mons:
        if p is None:
            continue
        counter[p.id] += 1
        for c in (getattr(p, "energyCards", None) or []):
            counter[c.id] += 1
        for c in (getattr(p, "tools", None) or []):
            counter[c.id] += 1
        for c in (getattr(p, "preEvolution", None) or []):
            counter[c.id] += 1


def _visible_opp_cards(opp):
    """Multiset of opponent cards already visible (active+bench+attached+discard)."""
    seen = Counter()
    _mon_cards(seen, ([opp.active[0]] if (opp.active and opp.active[0]) else []) + [b for b in opp.bench if b])
    for c in opp.discard:
        seen[c.id] += 1
    return seen


def _visible_own_cards(me):
    """Multiset of OUR cards whose location we know exactly (hand+board+discard) — everything NOT in
    our own hidden pool (deck+prize)."""
    seen = Counter()
    for c in (getattr(me, "hand", None) or []):
        seen[c.id] += 1
    _mon_cards(seen, ([me.active[0]] if (me.active and me.active[0]) else []) + [b for b in me.bench if b])
    for c in me.discard:
        seen[c.id] += 1
    return seen


def _hidden_pool(deck_list, seen, need, rng=None):
    remaining = Counter(deck_list) - seen
    hidden = [cid for cid, n in remaining.items() for _ in range(max(0, n))]
    if rng is None:
        random.shuffle(hidden)
    else:
        hidden = [hidden[i] for i in rng.permutation(len(hidden))]
    while len(hidden) < need:
        hidden.append(BASIC_ENERGY_ID)
    return hidden


def _fill_prizes(prize_slots, hidden_fill):
    """Keep face-up (revealed) prizes pinned in their slots; fill only the facedown ones from the pool.
    AUDIT FIX: the old code overwrote every slot with pool cards, discarding known information and
    duplicating any revealed prize (its copy stayed in the pool too)."""
    it = iter(hidden_fill)
    return [(c.id if c is not None else next(it)) for c in prize_slots]


def determinize(obs, your_index, your_deck, opp_deck, rng=None, opp_hand=None):
    """Open ONE search world: our own hidden zones (prize+deck) are partitioned WITHOUT REPLACEMENT from
    one shuffled own-hidden pool (fix vs ml/search.py:81-82's independent double-sampling), and the
    opponent's hidden zones are sampled from their TRUE deck minus visible cards (ORACLE — legitimate in
    training only: the generator knows the opponent's deck; the deployed net never searches).
    If ``opp_hand`` is supplied, those exact ids are pinned in the opponent hand and removed from
    the remaining hidden pool before sampling deck/prizes. This stronger oracle is eval-only.
    opp_deck=None falls back to the blind Snorlax fill (diagnostic parity with the old baseline).
    Visible-card accounting (AUDIT FIX 2026-07-02): pre-evolution chains (_mon_cards), the in-play
    stadium (by Card.playerIndex), and face-up prizes are all subtracted from the hidden pools."""
    state = obs.current
    me = state.players[your_index]
    opp = state.players[1 - your_index]
    active_facedown = len(opp.active) > 0 and opp.active[0] is None

    own_seen = _visible_own_cards(me)
    opp_seen = _visible_opp_cards(opp)
    for c in (getattr(state, "stadium", None) or []):
        if c is not None:
            (own_seen if c.playerIndex == your_index else opp_seen)[c.id] += 1
    for c in (getattr(state, "looking", None) or []):
        if c is not None:
            (own_seen if c.playerIndex == your_index else opp_seen)[c.id] += 1
    for c in me.prize:
        if c is not None:
            own_seen[c.id] += 1
    for c in opp.prize:
        if c is not None:
            opp_seen[c.id] += 1

    n_my_facedown_prize = sum(1 for c in me.prize if c is None)
    own_hidden = _hidden_pool(your_deck, own_seen, n_my_facedown_prize + me.deckCount, rng)
    your_prize = _fill_prizes(me.prize, own_hidden[:n_my_facedown_prize])
    your_deck_fill = own_hidden[n_my_facedown_prize:n_my_facedown_prize + me.deckCount]

    if opp_deck is None:
        if opp_hand is not None:
            raise ValueError("opp_hand requires opp_deck")
        opp_deck_fill = [SNORLAX_ID] * opp.deckCount
        opp_prize_fill = [BASIC_ENERGY_ID] * len(opp.prize)
        opp_hand_fill = [BASIC_ENERGY_ID] * opp.handCount
        opp_active = [SNORLAX_ID] if active_facedown else []
    else:
        exact_hand = None if opp_hand is None else [int(cid) for cid in opp_hand]
        if exact_hand is not None and len(exact_hand) != opp.handCount:
            raise ValueError(f"opp_hand has {len(exact_hand)} cards, expected {opp.handCount}")
        if exact_hand is not None:
            opp_seen.update(exact_hand)
        n_opp_facedown_prize = sum(1 for c in opp.prize if c is None)
        hidden_hand_n = 0 if exact_hand is not None else opp.handCount
        need = hidden_hand_n + n_opp_facedown_prize + opp.deckCount + (1 if active_facedown else 0)
        hidden = _hidden_pool(opp_deck, opp_seen, need, rng)
        opp_active = []
        if active_facedown:
            bi = next((i for i, cid in enumerate(hidden) if _is_basic_pokemon(cid)), None)
            opp_active = [hidden.pop(bi)] if bi is not None else [SNORLAX_ID]
        opp_hand_fill = exact_hand if exact_hand is not None else hidden[:opp.handCount]
        off = hidden_hand_n
        opp_prize_fill = _fill_prizes(opp.prize, hidden[off:off + n_opp_facedown_prize])
        opp_deck_fill = hidden[off + n_opp_facedown_prize:
                               off + n_opp_facedown_prize + opp.deckCount]

    return search_begin(
        obs,
        your_deck=your_deck_fill,
        your_prize=your_prize,
        opponent_deck=opp_deck_fill,
        opponent_prize=opp_prize_fill,
        opponent_hand=opp_hand_fill,
        opponent_active=opp_active,
    )
