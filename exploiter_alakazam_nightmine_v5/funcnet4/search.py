"""Gumbel-ISMCTS over the engine Search API — funcnet4's search-in-training (expert iteration).

WHAT: at a searched decision the generator runs a small Gumbel MCTS (K sims, sequential halving,
completed-Q — Danihelka et al. 2022) over D oracle determinizations of the hidden zones, producing an
improved policy pi'. The action PLAYED is sampled from pi' and Decision.mu = pi'[action] (V-trace's
behavior prob must be the TRUE sampling prob — recording the net's prob while playing the search's move
poisons the importance weights silently). The learner distills pi' via losses._pi_prime_kl_loss.
At deploy there is NO search (measured negative); the net stays greedy.

MATH SOURCE: the pure functions are a numpy port of mctx (ref_repos/mctx/mctx/_src/{seq_halving,
qtransforms,policies}.py — the paper authors' canonical implementation), cross-checked against
LightZero's C++ (ref_repos/LightZero/.../mcts_gumbel_alphazero.cpp). Two LightZero quirks are
deliberately NOT ported: its Gumbel noise is drawn once per PROCESS with a fixed seed (we draw fresh
per decision) and its interior selection uses pre-softmax probs (we use the mctx formula).

ISMCTS: each determinization d = its own engine tree (search_begin fixes the hidden zones), but the
ROOT statistics are SHARED across the D trees (OpenSpiel info-set pattern): a wave's visits round-robin
over trees, so every root action's Q averages over worlds by construction and pi' is computed once.

vs ml/search.py (the PUCT ancestor), three correctness fixes:
  1. at OPPONENT nodes the encoder gets the OPPONENT's deck (the "own deck" word belongs to the player
     to move; ml/search.py:152 always passed the searcher's);
  2. determinize partitions ONE shuffled own-hidden pool into prize+deck (ml/search.py:81-82 sampled
     both independently from the full 60 -> could double-allocate copies);
  3. every search is wrapped so `search_end()` ALWAYS runs (ml/search.py:227 leaks the native
     agent_ptr on any mid-tree exception -> "agent_ptr broken").

Cost model: forwards per searched decision ~= K (leaf evals, wave-BATCHED into single forwards) and
D*search_begin + ~K*(1+auto_steps) search_step calls. Bench with tools/bench_mcts_tf.py before farm use.
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

SNORLAX_ID = 1072       # placeholder Basic used only by the (non-training) blind fill
BASIC_ENERGY_ID = 1

# hygiene caps: exceeding any -> SearchAbort -> the caller falls back to the plain net policy
MAX_AUTO_STEPS = 16     # forced/empty-select chain length inside the tree
STEP_BUDGET_MULT = 4    # hard cap on engine search_step calls: MULT * K * D


class SearchAbort(Exception):
    pass


# ===================================================================== pure Gumbel math (numpy, no engine)
def considered_visit_sequence(m_top, n_sims):
    """Sequential-halving visit schedule (mctx seq_halving.get_sequence_of_considered_visits).
    Element t = the visit count a root action must have to be considered by simulation t."""
    if m_top <= 1:
        return tuple(range(n_sims))
    log2max = int(math.ceil(math.log2(m_top)))
    sequence, visits, num_considered = [], [0] * m_top, m_top
    while len(sequence) < n_sims:
        num_extra_visits = max(1, int(n_sims / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:n_sims])


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def mixed_value(v_root, q, visits, prior):
    """mctx qtransforms._compute_mixed_value: interpolate the root raw value with the prior-weighted
    mean Q of the VISITED actions. q entries with visits==0 are undefined (ignored)."""
    visits = np.asarray(visits)
    prior = np.maximum(np.asarray(prior, dtype=np.float64), np.finfo(np.float64).tiny)
    n_total = visits.sum()
    vis = visits > 0
    sum_probs = prior[vis].sum()
    weighted_q = float((prior[vis] * np.asarray(q, dtype=np.float64)[vis]).sum() / sum_probs) if vis.any() else 0.0
    return (float(v_root) + n_total * weighted_q) / (n_total + 1.0)


def completed_q(v_root, q, visits, prior):
    """Visited actions keep their tree Q; unvisited get the mixed value (mctx _complete_qvalues)."""
    visits = np.asarray(visits)
    out = np.full(len(visits), mixed_value(v_root, q, visits, prior), dtype=np.float64)
    vis = visits > 0
    out[vis] = np.asarray(q, dtype=np.float64)[vis]
    return out


def sigma_q(cq, max_visit, c_visit=50.0, c_scale=0.1, minmax=False):
    """Monotone Q transform sigma: (c_visit + max_visit) * c_scale * rescale(cq).
    rescale: FIXED [-1,1]->[0,1] by default — our V is tanh-bounded (bytrl value_from) and per-node
    min-max at K<=16 visits amplifies noise through the eps-gap stretch. minmax=True keeps the
    mctx/LightZero per-node min-max for the --search-minmax-q A/B."""
    cq = np.asarray(cq, dtype=np.float64)
    if minmax:
        lo, hi = cq.min(), cq.max()
        r = (cq - lo) / max(hi - lo, 1e-8)
    else:
        r = (np.clip(cq, -1.0, 1.0) + 1.0) / 2.0
    return (c_visit + float(max_visit)) * c_scale * r


def improved_policy(logits, cq, max_visit, minmax=False):
    """pi' = softmax(norm_logits + sigma(completed_q)) over the FULL action set (mctx
    gumbel_muzero_policy). Full support: unvisited actions enter through the mixed-value completion."""
    logits = np.asarray(logits, dtype=np.float64)
    return _softmax((logits - logits.max()) + sigma_q(cq, max_visit, minmax=minmax))


def root_wave_pick(logits, gumbel, cq, max_visit, visits, considered_visit, width, minmax=False):
    """Indices of the top-`width` root actions among {a: visits[a]==considered_visit}, scored by
    gumbel + norm_logits + sigma(completed_q) (wave-batched mctx seq_halving.score_considered)."""
    logits = np.asarray(logits, dtype=np.float64)
    score = gumbel + (logits - logits.max()) + sigma_q(cq, max_visit, minmax=minmax)
    score = np.where(np.asarray(visits) == considered_visit, score, -np.inf)
    order = np.argsort(-score)
    picks = [int(a) for a in order[:width] if np.isfinite(score[a])]
    return picks


def interior_pick(logits, cq, visits, max_visit, minmax=False):
    """Deterministic interior selection (mctx gumbel_muzero_interior_action_selection):
    argmax_a [ pi'(a) - visits[a] / (1 + sum(visits)) ]. NOT LightZero's pre-softmax variant."""
    pi = improved_policy(logits, cq, max_visit, minmax=minmax)
    visits = np.asarray(visits, dtype=np.float64)
    return int(np.argmax(pi - visits / (1.0 + visits.sum())))


# ===================================================================== determinization (oracle ISMCTS)
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
        # AUDIT FIX 2026-07-02: cards UNDER an evolved Pokemon (the whole pre-evolution chain, a flat
        # list) are in play, not hidden. Omitting them re-entered evolution-line cards into the hidden
        # pool: determinized worlds held phantom duplicates (measured: 6 copies of a 4-of; 76% of real
        # decisions affected on evolution archetypes), silently biasing the Q estimates behind pi'.
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
    else:                          # AUDIT FIX: search rng, not global random -> reproducible worlds
        hidden = [hidden[i] for i in rng.permutation(len(hidden))]
    while len(hidden) < need:      # safety pad (miscounted visibles); shouldn't trigger on clean data
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
    for c in (getattr(state, "stadium", None) or []):          # stadium belongs to its owner's deck
        if c is not None:
            (own_seen if c.playerIndex == your_index else opp_seen)[c.id] += 1
    # Cards temporarily exposed in LOOKING are already outside their owner's hidden deck.  Leaving
    # them in the pool can create phantom duplicate copies when a search starts mid-effect.
    for c in (getattr(state, "looking", None) or []):
        if c is not None:
            (own_seen if c.playerIndex == your_index else opp_seen)[c.id] += 1
    for c in me.prize:                                         # face-up prizes are visible
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
            # AUDIT FIX: the facedown active is one of the opponent's hidden cards — RESERVE it from the
            # pool (the old code took the decklist's first basic without removal: deterministic across
            # all D worlds AND duplicated). Shuffled pool -> first basic = a random legal basic.
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


# ===================================================================== net leaf evaluation (batched)
class _EvalItem:
    """Minimal shim with the attrs losses._build reads (sv_enc/sv_dec/n_actions)."""
    __slots__ = ("sv_enc", "sv_dec", "n_actions")

    def __init__(self, sv_enc, sv_dec, n_actions):
        self.sv_enc, self.sv_dec, self.n_actions = sv_enc, sv_dec, n_actions


class NetEval:
    """Batched FuncNetTF leaf evaluator. decks_by_seat maps seat index -> that player's 60-card list;
    the encoder's "own deck" word is filled with the deck of the PLAYER TO MOVE (correctness fix #1)."""

    def __init__(self, model, device, decks_by_seat):
        self.model = model
        self.device = device
        self.decks = decks_by_seat
        self.n_forwards = 0

    def batch(self, items):
        """items: list of (obs, actions). Returns [(value_for_mover: float, logits: np(A,)), ...] where
        logits = raw policy * POLICY_TEMP (exactly what NetActor softmaxes)."""
        if not items:
            return []
        from . import losses as _losses
        shims = []
        for obs, actions in items:
            mover = obs.current.yourIndex
            sv_e = get_encoder_input(obs, self.decks[mover])
            sv_d = get_decoder_input(obs, actions)
            shims.append(_EvalItem(sv_e, sv_d, len(actions)))
        enc, dec, _mask = _losses._build(shims, self.device)
        with torch.inference_mode():
            value, policy, _ = self.model(enc[0], enc[1], dec[0], dec[1])
        self.n_forwards += 1
        out = []
        for b, sh in enumerate(shims):
            out.append((float(value[b, 0]),
                        (policy[b, :sh.n_actions].float().cpu().numpy() * POLICY_TEMP)))
        return out


# ===================================================================== engine tree
class SearchNode:
    """One expanded interior state in ONE determinization's tree. Child stats are stored in the
    SEARCHER's perspective; `sign` = +1 if the player to move here is the searcher else -1."""
    __slots__ = ("sid", "actions", "logits", "v_search", "sign", "turn", "terminal", "children",
                 "child_visits", "child_qsum")

    def __init__(self, sid, actions, logits, v_search, sign, turn, terminal=False):
        self.sid = sid
        self.actions = actions
        self.logits = logits
        self.v_search = v_search
        self.sign = sign
        self.turn = int(turn)
        self.terminal = terminal
        self.children = {}
        n = len(actions) if actions else 0
        self.child_visits = np.zeros(n, dtype=np.int64)
        self.child_qsum = np.zeros(n, dtype=np.float64)


class SearchResult:
    __slots__ = ("pi_prime", "root_q", "root_q_raw", "root_visits", "n_forwards", "n_steps",
                 "terminal_backups", "value_backups", "turn1_backups", "turn2_backups",
                 "max_turn_offset", "mean_leaf_depth", "max_leaf_depth", "ok")

    def __init__(self, pi_prime, root_q, root_q_raw, root_visits, n_forwards, n_steps,
                 terminal_backups=0, value_backups=0, turn1_backups=0, turn2_backups=0,
                 max_turn_offset=0, mean_leaf_depth=0.0, max_leaf_depth=0, ok=True):
        self.pi_prime = pi_prime
        self.root_q = root_q
        self.root_q_raw = root_q_raw
        self.root_visits = root_visits
        self.n_forwards = n_forwards
        self.n_steps = n_steps
        self.terminal_backups = terminal_backups
        self.value_backups = value_backups
        self.turn1_backups = turn1_backups
        self.turn2_backups = turn2_backups
        self.max_turn_offset = max_turn_offset
        self.mean_leaf_depth = mean_leaf_depth
        self.max_leaf_depth = max_leaf_depth
        self.ok = ok


class _Budget:
    __slots__ = ("steps", "cap")

    def __init__(self, cap):
        self.steps, self.cap = 0, cap

    def step(self, sid, select):
        self.steps += 1
        if self.steps > self.cap:
            raise SearchAbort(f"step budget exceeded ({self.cap})")
        return search_step(sid, select)


def _terminal_value(state, searcher):
    return 0.0 if state.result == 2 else (1.0 if state.result == searcher else -1.0)


def _auto_advance(ss, budget):
    """Advance through forced/empty selects (no choice => no eval, no visit). Returns (ss, actions):
    actions is None when the reached state is terminal."""
    for _ in range(MAX_AUTO_STEPS):
        state = ss.observation.current
        if state.result >= 0:
            return ss, None
        sel = ss.observation.select
        if sel is None or getattr(sel, "maxCount", 0) == 0 or len(sel.option) == 0:
            ss = budget.step(ss.searchId, [])
            continue
        actions = enumerate_actions(sel)
        if len(actions) == 1:
            ss = budget.step(ss.searchId, actions[0])
            continue
        return ss, actions
    raise SearchAbort("auto-step chain exceeded MAX_AUTO_STEPS")


class _GumbelSearch:
    """One decision's search: D determinization trees with SHARED root statistics."""

    def __init__(self, obs, actions, root_logits, root_value, net_eval, your_deck, opp_deck,
                 k_sims, n_det, m_top, max_depth, minmax_q, rng, opp_hand=None):
        self.obs = obs
        self.actions = actions
        self.A = len(actions)
        self.searcher = obs.current.yourIndex
        self.net_eval = net_eval
        self.start_forwards = int(net_eval.n_forwards)
        self.your_deck = your_deck
        self.opp_deck = opp_deck
        self.opp_hand = opp_hand
        self.k = k_sims
        self.d = max(1, n_det)
        self.m = max(2, min(m_top, self.A, k_sims))
        self.max_depth = max_depth
        self.minmax = minmax_q
        self.rng = rng
        self.budget = _Budget(STEP_BUDGET_MULT * k_sims * self.d)
        # shared root stats (searcher perspective) + per-tree root handles/children
        self.root_logits = np.asarray(root_logits, dtype=np.float64)
        self.root_value = float(root_value)
        self.root_visits = np.zeros(self.A, dtype=np.int64)
        self.root_qsum = np.zeros(self.A, dtype=np.float64)
        self.root_prior = _softmax(self.root_logits)
        self.root_turn = int(obs.current.turn)
        self.gumbel = rng.gumbel(size=self.A)
        self.roots = []                        # per-tree root SearchState
        self.root_children = [dict() for _ in range(self.d)]   # tree d: action -> SearchNode | "terminal"
        # Eval-only telemetry; none of these fields participates in selection or backup.
        self.terminal_backups = 0
        self.value_backups = 0
        self.turn1_backups = 0
        self.turn2_backups = 0
        self.max_turn_offset = 0
        self.leaf_depth_sum = 0
        self.leaf_count = 0
        self.max_leaf_depth = 0

    # ---- stats helpers -------------------------------------------------------------------------
    def _root_q(self):
        q = np.zeros(self.A, dtype=np.float64)
        vis = self.root_visits > 0
        q[vis] = self.root_qsum[vis] / self.root_visits[vis]
        return q

    def _root_cq(self):
        return completed_q(self.root_value, self._root_q(), self.root_visits, self.root_prior)

    def _backup(self, path, v_search):
        """path = [(node_or_'root', tree_d, action), ...] root-first; v_search in searcher persp."""
        for node, _d, a in path:
            if node == "root":
                self.root_qsum[a] += v_search
                self.root_visits[a] += 1
            else:
                node.child_qsum[a] += v_search
                node.child_visits[a] += 1

    def _leaf_telemetry(self, path, turn, terminal):
        off = max(0, int(turn) - self.root_turn)
        self.terminal_backups += int(bool(terminal))
        self.value_backups += int(not terminal)
        self.turn1_backups += int(off >= 1)
        self.turn2_backups += int(off >= 2)
        self.max_turn_offset = max(self.max_turn_offset, off)
        depth = len(path)
        self.leaf_depth_sum += depth
        self.leaf_count += 1
        self.max_leaf_depth = max(self.max_leaf_depth, depth)

    # ---- node expansion ------------------------------------------------------------------------
    def _make_node(self, ss, actions, evald):
        state = ss.observation.current
        if actions is None:                    # terminal
            return SearchNode(ss.searchId, None, None, _terminal_value(state, self.searcher),
                              +1, state.turn, terminal=True)
        v_mover, logits = evald
        sign = 1.0 if state.yourIndex == self.searcher else -1.0
        return SearchNode(ss.searchId, actions, np.asarray(logits, dtype=np.float64),
                          sign * float(v_mover), sign, state.turn)

    # ---- one simulation's engine descent (NO net eval; leaves are collected for batching) -------
    def _descend(self, d, a):
        """Descend tree d starting with root action a. Returns (path, leaf) where leaf is either
        ('terminal', v_search) or ('pending', parent_node_or_'root', tree, action, ss, actions)."""
        path = [("root", d, a)]
        child = self.root_children[d].get(a)
        if child is None:
            ss = self.budget.step(self.roots[d].searchId, self.actions[a])
            ss, actions = _auto_advance(ss, self.budget)
            if actions is None:
                node = self._make_node(ss, None, None)
                self.root_children[d][a] = node
                self._leaf_telemetry(path, node.turn, True)
                return path, ("terminal", node.v_search)
            return path, ("pending", "root", d, a, ss, actions)
        node = child
        for _ in range(self.max_depth):
            if node.terminal:
                self._leaf_telemetry(path, node.turn, True)
                return path, ("terminal", node.v_search)
            # interior pick in the MOVER's perspective (sign-flip of searcher-persp stats)
            vis = node.child_visits > 0
            q_search = np.zeros(len(node.actions), dtype=np.float64)
            q_search[vis] = node.child_qsum[vis] / node.child_visits[vis]
            q_mover = node.sign * q_search
            prior = _softmax(node.logits)
            cq = completed_q(node.sign * node.v_search, q_mover, node.child_visits, prior)
            ai = interior_pick(node.logits, cq, node.child_visits,
                               int(node.child_visits.max()) if len(node.child_visits) else 0,
                               minmax=self.minmax)
            path.append((node, d, ai))
            nxt = node.children.get(ai)
            if nxt is None:
                ss = self.budget.step(node.sid, node.actions[ai])
                ss, actions = _auto_advance(ss, self.budget)
                if actions is None:
                    leaf = self._make_node(ss, None, None)
                    node.children[ai] = leaf
                    self._leaf_telemetry(path, leaf.turn, True)
                    return path, ("terminal", leaf.v_search)
                return path, ("pending", node, d, ai, ss, actions)
            node = nxt
        # depth cap: score the frontier node by its own stored value (no expansion past the cap)
        self._leaf_telemetry(path, node.turn, False)
        return path, ("terminal", node.v_search)

    # ---- main ----------------------------------------------------------------------------------
    def run(self):
        for _ in range(self.d):
            self.roots.append(determinize(self.obs, self.searcher, self.your_deck, self.opp_deck,
                                          rng=self.rng, opp_hand=self.opp_hand))

        seq = considered_visit_sequence(self.m, self.k)
        i = 0
        wave_idx = 0
        while i < len(seq):
            cv = seq[i]
            j = i
            while j < len(seq) and seq[j] == cv:
                j += 1
            width = j - i
            picks = root_wave_pick(self.root_logits, self.gumbel, self._root_cq(),
                                   int(self.root_visits.max()), self.root_visits, cv, width,
                                   minmax=self.minmax)
            if not picks:
                break
            descents, pending = [], []
            for w, a in enumerate(picks):
                d = (wave_idx + w) % self.d
                path, leaf = self._descend(d, a)
                if leaf[0] == "terminal":
                    self._backup(path, leaf[1])
                else:
                    descents.append((path, leaf))
                    pending.append((leaf[4].observation, leaf[5]))
            if pending:
                evals = self.net_eval.batch(pending)
                for (path, leaf), ev in zip(descents, evals):
                    _tag, parent, d, a, ss, actions = leaf
                    node = self._make_node(ss, actions, ev)
                    if parent == "root":
                        self.root_children[d][a] = node
                    else:
                        parent.children[a] = node
                    self._leaf_telemetry(path, node.turn, False)
                    self._backup(path, node.v_search)
            i = j
            wave_idx += 1

        cq = self._root_cq()
        pi = improved_policy(self.root_logits, cq, int(self.root_visits.max()), minmax=self.minmax)
        if (not np.all(np.isfinite(pi))) or abs(pi.sum() - 1.0) > 1e-3 or (pi < 0).any():
            raise SearchAbort("pi_prime failed the sanity check")
        raw_q = self._root_q()
        return SearchResult(
            pi.astype(np.float32), cq.astype(np.float32), raw_q.astype(np.float32),
            self.root_visits.astype(np.int32), self.net_eval.n_forwards - self.start_forwards,
            self.budget.steps,
            terminal_backups=self.terminal_backups, value_backups=self.value_backups,
            turn1_backups=self.turn1_backups, turn2_backups=self.turn2_backups,
            max_turn_offset=self.max_turn_offset,
            mean_leaf_depth=(self.leaf_depth_sum / max(self.leaf_count, 1)),
            max_leaf_depth=self.max_leaf_depth)


def run_gumbel_search(obs, actions, root_logits, root_value, net_eval, your_deck, opp_deck,
                      k_sims=8, n_det=2, m_top=4, max_depth=6, minmax_q=False, rng=None,
                      opp_hand=None):
    """The ONLY entry point generation code should call. Returns SearchResult or None (any failure ->
    caller falls back to the plain net policy and records NO pi_prime). search_end() ALWAYS runs — it
    frees every SearchState of every tree in one call (fix #3: the native agent_ptr must never be left
    mid-tree; ml/search.py:227 had no such guard)."""
    if rng is None:
        rng = np.random.default_rng()
    try:
        return _GumbelSearch(obs, actions, root_logits, root_value, net_eval, your_deck, opp_deck,
                             k_sims, n_det, m_top, max_depth, minmax_q, rng,
                             opp_hand=opp_hand).run()
    except Exception:
        return None
    finally:
        try:
            search_end()
        except Exception:
            pass
